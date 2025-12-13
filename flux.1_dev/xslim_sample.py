# cd flux.1_dev
# CUDA_VISIBLE_DEVICES=1 python xslim_sample.py   --model_path your/flux/ckpt   --mode original
import argparse
from pathlib import Path
import torch
from diffusers import DiffusionPipeline
from cache_utils.cache_manager import MANAGER as cache_manager_obj
from cache_utils.transformer import FluxTransformer2DModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="Path to FLUX.1-dev checkpoint directory")
    p.add_argument(
        "--prompt", 
        type=str, 
        default="A cute cat wearing a pink beret and a light pink scarf, holding a bouquet of sparkling light pink roses.", 
        help="Single prompt string OR path to a .txt file (one prompt per line)"
    )
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument(
        "--mode",
        type=str,
        default="slow",    # choose "slow" or "S-slow" for a good balance of speed and quality
        choices=["original", "slow", "fast", "S-slow", "S-fast"],
        help="cache mode: original(no cache), slow/fast(ours X-Slim C2F), S-slow/S-fast(ours step-only variant)"
    )
    p.add_argument("--output_root", type=str, default=None)
    p.add_argument("--num_steps", type=int, default=50)
    return p.parse_args()


def load_prompts(p: str) -> list[str]:
    path = Path(p)
    if path.is_file() and path.suffix == ".txt":
        return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [p]


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    num_steps = args.num_steps
    base_seed = args.base_seed
    mode = args.mode

    out_root = Path(args.output_root) if args.output_root else Path(f"outputs/X-Slim-{mode}/")
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

    if mode == "original":
        pipe.transformer.__class__.enable_xslimcache = False
    else:
        pipe.transformer.__class__.enable_xslimcache = True
        pipe.transformer.__class__.num_steps = num_steps
        cache_manager_obj.set_mode(mode)

    pipe.to("cuda")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i, prompt in enumerate(prompts):
        seed = base_seed + i
        gen = torch.Generator("cpu").manual_seed(seed)

        start.record()
        image = pipe(prompt, num_inference_steps=num_steps, generator=gen).images[0]
        end.record()
        torch.cuda.synchronize()
        t = start.elapsed_time(end) * 1e-3

        name = f"{i:04d}.png"
        image.save(img_dir / name)
        (txt_dir / name.replace(".png", ".txt")).write_text(prompt, encoding="utf-8")
        print(f"[{i+1}/{len(prompts)}] {img_dir / name}, {t:.2f}s")

    print("done.")


if __name__ == "__main__":
    main()
