from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FluxTransformer2DLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers, is_torch_version
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    # FluxAttnProcessor2_0,
    FluxAttnProcessor2_0_NPU,
    FusedFluxAttnProcessor2_0,
)
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, FluxPosEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
from cache_utils.cache_manager import MANAGER as cache_manager_obj
from .transformer_blocks import FluxSingleTransformerBlock, FluxTransformerBlock
import os

def process_hidden_states(hidden_states, cache_data, dtype=torch.float16):
    return torch.clamp((hidden_states.to(torch.float32) + cache_data), -65504.0, 65504.0).to(dtype)

class FluxTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, FluxTransformer2DLoadersMixin, CacheMixin
):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        patch_size (`int`, defaults to `1`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `19`):
            The number of layers of dual stream DiT blocks to use.
        num_single_layers (`int`, defaults to `38`):
            The number of layers of single stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `4096`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        pooled_projection_dim (`int`, defaults to `768`):
            The number of dimensions to use for the pooled projection.
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
        # ------------------- X-Slim Cache ------------------------
        self.cnt = 0
        self.previous_residual = {}
        self.cachestep = []

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedFluxAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedFluxAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)
    
    
    
    def forward(
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
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
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
        else:
            guidance = None

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

        cache_manager_obj.enable_cache = self.enable_xslimcache
        if self.enable_xslimcache:
            if not hasattr(cache_manager_obj, 'curr_step') or cache_manager_obj.curr_step >= self.num_steps or cache_manager_obj.curr_step == 0:
                cache_manager_obj._init_cache_config(hidden_states)
                cache_manager_obj.seq_len = hidden_states.shape[1]
                cache_manager_obj.reset_cache()
            cache_manager_obj.curr_step = self.cnt
            cache_manager_obj.check_cache_step()
            ori_hidden_states = hidden_states.clone()
            if cache_manager_obj.is_cache_step:
                should_calc = False
            else:
                should_calc = True
                if cache_manager_obj.is_fully_infer_step:
                    cache_manager_obj.reset_cache()
                if cache_manager_obj.is_token_refresh_step:
                    cache_manager_obj.img_seq_len = img_ids.shape[0]
                    cache_manager_obj.txt_seq_len = txt_ids.shape[0]
                    diff = torch.abs(self.previous_residual[-1] - self.previous_residual[-2])
                    cache_manager_obj.cached_patchified_index, cache_manager_obj.other_patchified_index = cache_manager_obj.tokencache_selection(diff)
                    hidden_states = hidden_states[:, cache_manager_obj.other_patchified_index]

            self.cnt += 1 
            if self.cnt == self.num_steps:
                # print(f"Cache Ratio:{len(self.cachestep)}/{self.num_steps}; Speedup:{self.num_steps/(self.num_steps-len(self.cachestep)):.3f}")
                self.cachestep = []
                self.cnt = 0           
        
        if self.enable_xslimcache:
            if not should_calc:
                hidden_states += self.previous_residual[-1]
                self.cachestep.append(self.cnt-1)
                if cache_manager_obj.next_token_refresh_step:
                    cache_manager_obj.last_img = hidden_states     
            else:
                for index_block, block in enumerate(self.transformer_blocks):
                    if cache_manager_obj.is_block_refresh_step and index_block in cache_manager_obj.skip_block_index["double"]:
                        # --------------------------------------------- Blocks reuse ---------------------------------------
                        double_img_cachedata = (cache_manager_obj.block_cache_data["double"]["hidden_states"][index_block]).to(torch.float32)
                        double_txt_cachedata = (cache_manager_obj.block_cache_data["double"]["encoder_hidden_states"][index_block]).to(torch.float32)
                        hidden_states = process_hidden_states(hidden_states, double_img_cachedata)
                        encoder_hidden_states = process_hidden_states(encoder_hidden_states, double_txt_cachedata)   
                    else:
                        hidden_states_inp = hidden_states.clone()
                        encoder_hidden_states_inp = encoder_hidden_states.clone()
                        if torch.is_grad_enabled() and self.gradient_checkpointing:
                            def create_custom_forward(module, return_dict=None):
                                def custom_forward(*inputs):
                                    if return_dict is not None:
                                        return module(*inputs, return_dict=return_dict)
                                    else:
                                        return module(*inputs)

                                return custom_forward

                            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                            encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block),
                                hidden_states,
                                encoder_hidden_states,
                                temb,
                                image_rotary_emb,
                                **ckpt_kwargs,
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
                        
                        # --------------------------- Double Block Resore -----------------------------
                        if cache_manager_obj.is_fully_infer_step and cache_manager_obj.next_block_refresh_step:
                            doubleblock_should_record = cache_manager_obj.blockcache_selection(index_block, hidden_states, hidden_states_inp, mod="double")
                            if doubleblock_should_record:
                                cache_manager_obj.block_cache_data["double"]["precalc_step"] = cache_manager_obj.curr_step
                                cache_manager_obj.block_cache_data["double"]["hidden_states"][index_block] = torch.sub(hidden_states, hidden_states_inp)
                                cache_manager_obj.block_cache_data["double"]["encoder_hidden_states"][index_block] = torch.sub(encoder_hidden_states, encoder_hidden_states_inp)

                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

                for index_block, block in enumerate(self.single_transformer_blocks):
                    if cache_manager_obj.is_block_refresh_step and index_block in cache_manager_obj.skip_block_index["single"]:
                        cachedata = (cache_manager_obj.block_cache_data["single"][index_block]).to(torch.float32)
                        hidden_states = process_hidden_states(hidden_states, cachedata)
                    else:
                        hidden_states_singleinp = hidden_states.clone()
                        if torch.is_grad_enabled() and self.gradient_checkpointing:
                            def create_custom_forward(module, return_dict=None):
                                def custom_forward(*inputs):
                                    if return_dict is not None:
                                        return module(*inputs, return_dict=return_dict)
                                    else:
                                        return module(*inputs)

                                return custom_forward

                            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                            hidden_states = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block),
                                hidden_states,
                                temb,
                                image_rotary_emb,
                                **ckpt_kwargs,
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
                        # --------------------------- Single Block Resore -----------------------------
                        if cache_manager_obj.is_fully_infer_step and cache_manager_obj.next_block_refresh_step:
                            hidden_states_img_inp = hidden_states_singleinp[:, encoder_hidden_states.shape[1] :, ...]
                            hidden_states_img = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                            singleblock_should_record = cache_manager_obj.blockcache_selection(index_block, hidden_states_img, hidden_states_img_inp, mod="single")
                            if singleblock_should_record:
                                cache_manager_obj.block_cache_data["single"]["precalc_step"] = cache_manager_obj.curr_step
                                cache_manager_obj.block_cache_data["single"][index_block] = torch.sub(hidden_states, hidden_states_singleinp) 
                hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

                if cache_manager_obj.next_token_refresh_step and (not cache_manager_obj.is_token_refresh_step):
                    cache_manager_obj.last_img = hidden_states.clone()
                if cache_manager_obj.is_token_refresh_step:
                    cache_manager_obj.last_img[:, cache_manager_obj.other_patchified_index] = hidden_states
                    hidden_states = cache_manager_obj.last_img    

                if self.previous_residual:
                    previous_residual = self.previous_residual[-1].clone()
                    self.previous_residual[-2] = previous_residual
                self.previous_residual[-1] = hidden_states - ori_hidden_states
        else:
            for index_block, block in enumerate(self.transformer_blocks):
                double_inp_img = hidden_states.clone()
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
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

            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            for index_block, block in enumerate(self.single_transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
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
            hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        
        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
  