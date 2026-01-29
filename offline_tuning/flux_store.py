# python sample_store.py --store_data
import argparse
import os
import time
from pathlib import Path
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union
import torch
from diffusers import DiffusionPipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from transformer_store import FluxTransformer2DModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="/home/ma-user/work/w50048495/flux.1_dev_test/ckpts", help="Path to FLUX.1-dev checkpoint directory")
    p.add_argument(
        "--prompt",
        type=str,
        default="A cute cat wearing a pink beret and a light pink scarf, holding a bouquet of sparkling light pink roses.",
        help="Single prompt string OR path to a .txt file (one prompt per line)"
    )
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--output_root", type=str, default=None)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument(
        "--store_data",
        action="store_true",
        help="Enable storing strategy data for X-Slim if set"
    )
    return p.parse_args()


def load_prompts(p: str) -> list[str]:
    path = Path(p)
    if path.is_file() and path.suffix == ".txt":
        return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [p]


def forward_storedata(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        inp = hidden_states.clone()
        for index_block, block in enumerate(self.transformer_blocks):
            double_inp = hidden_states.clone()
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
            # ==================================== X-Slim store Block-Level strategy data =========================
            if self.is_store:
                self.double_l1loss.append((hidden_states - double_inp).abs().mean().cpu().item())
                if index_block == len(self.transformer_blocks) - 1:
                    block_level_dir = os.path.join(self.strategy_data_dir, f"img{self.fig}", "block_level", "double_block")
                    os.makedirs(block_level_dir, exist_ok=True)
                    save_path = os.path.join(block_level_dir, f"step{self.cnt}_double_l1loss.pth")
                    torch.save(self.double_l1loss, save_path)
                    self.double_l1loss = []
            # =====================================================================================================
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            single_inp = hidden_states[:, encoder_hidden_states.shape[1] :, ...].clone()
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    temb,
                    image_rotary_emb,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )
            # ==================================== X-Slim store Block-Level strategy data =========================
            if self.is_store:
                self.single_l1loss.append((hidden_states[:, encoder_hidden_states.shape[1] :, ...] - single_inp).abs().mean().cpu().item())
                if index_block == len(self.single_transformer_blocks) - 1:
                    block_level_dir = os.path.join(self.strategy_data_dir, f"img{self.fig}", "block_level", "single_block")
                    os.makedirs(block_level_dir, exist_ok=True)
                    save_path = os.path.join(block_level_dir, f"step{self.cnt}_single_l1loss.pth")
                    torch.save(self.single_l1loss, save_path)
                    self.single_l1loss = []
            # =====================================================================================================
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        # ==================================== X-Slim store Step-Level strategy data ==============================
        if self.is_store:
            step_data = hidden_states - inp
            if self.prev_step_data is not None:
                self.step_l1loss.append((step_data - self.prev_step_data).abs().mean().cpu().item())
                if self.cnt == self.num_steps - 1:
                    step_level_dir = os.path.join(self.strategy_data_dir, f"img{self.fig}", "step_level")
                    os.makedirs(step_level_dir, exist_ok=True)
                    save_path = os.path.join(step_level_dir, "step_l1loss.pth")
                    torch.save(self.step_l1loss, save_path)
            self.prev_step_data = step_data

            self.cnt = (self.cnt + 1) % self.num_steps
            if self.cnt == 0:
                self.fig += 1
                self.prev_step_data = None
                self.step_l1loss = []
        # =========================================================================================================

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    num_steps = args.num_steps
    base_seed = args.base_seed

    current_dir = Path(__file__).parent
    out_root = Path(args.output_root) if args.output_root else current_dir / "ori_outputs"
    img_dir = out_root / "image"
    txt_dir = out_root / "text"
    img_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompt)
    print(f"loaded {len(prompts)} prompt(s)")

    transformer = FluxTransformer2DModel.from_pretrained(
        model_path / "transformer", torch_dtype=torch.float16
    )
    pipe = DiffusionPipeline.from_pretrained(
        model_path, transformer=transformer, torch_dtype=torch.float16
    )

    # ================ Configs for storing strategy data in X-Slim =============
    pipe.transformer.__class__.is_store = args.store_data
    if args.store_data:
        FluxTransformer2DModel.forward = forward_storedata
        pipe.transformer.__class__.strategy_data_dir = out_root / "strategy_data"
        pipe.transformer.__class__.num_steps = args.num_steps
        pipe.transformer.__class__.cnt = 0
        pipe.transformer.__class__.fig = 0
        # step-level
        pipe.transformer.__class__.prev_step_data = None
        pipe.transformer.__class__.step_l1loss = []
        # block-level
        pipe.transformer.__class__.double_l1loss = []
        pipe.transformer.__class__.single_l1loss = []
    # ==========================================================================

    # NPU device
    torch.npu.set_device(0)
    pipe.to("npu")

    for i, prompt in enumerate(prompts):
        seed = base_seed + i
        gen = torch.Generator("cpu").manual_seed(seed)

        t0 = time.perf_counter()
        image = pipe(prompt, num_inference_steps=num_steps, generator=gen).images[0]
        torch.npu.synchronize()
        t = time.perf_counter() - t0

        name = f"{i:04d}.png"
        image.save(img_dir / name)
        (txt_dir / name.replace(".png", ".txt")).write_text(prompt, encoding="utf-8")
        print(f"[{i+1}/{len(prompts)}] {img_dir / name}, {t:.2f}s")

    print("done.")


if __name__ == "__main__":
    main()
