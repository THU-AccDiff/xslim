# CUDA_VISIBLE_DEVICES=1 python3 xslim_sample.py --video-size 540 960 --video-length 81 --infer-steps 50 --prompt prompt.txt --flow-reverse --use-cpu-offload --save-path ./results --seed 42 --mode fast
import os
import time
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from loguru import logger

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.modules.cache_manager import MANAGER as cache_manager_obj


def read_prompt_list(p: str):
    path = Path(p)
    if path.is_file():
        ext = path.suffix.lower()
        with path.open("r", encoding="utf-8") as f:
            if ext == ".json":
                data = json.load(f)
                return [item["prompt_en"] for item in data]
            elif ext == ".txt":
                return [line.strip() for line in f if line.strip()]
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
    return [p]


def parse_xslim_mode():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--mode",
        type=str,
        default="original",   # choose "slow" or "S-slow" for a good balance of speed and quality
        choices=["original", "S-slow", "S-fast", "slow", "fast"],
    )
    xargs, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest 
    return xargs.mode


def main():
    mode = parse_xslim_mode()
    args = parse_args()

    args.mode = mode
    prompt_arg = getattr(args, "prompt", "prompt.txt")   # also support .txt file

    args.save_path = f"{args.save_path}/xslim-{args.mode}/"
    print(args)

    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    save_path = args.save_path if args.save_path_suffix == "" else f"{args.save_path}_{args.save_path_suffix}"
    os.makedirs(save_path, exist_ok=True)


    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = hunyuan_video_sampler.args  


    if args.mode == "original":
        hunyuan_video_sampler.pipeline.transformer.__class__.enable_slimcache = False
    else:
        hunyuan_video_sampler.pipeline.transformer.__class__.enable_slimcache = True
        hunyuan_video_sampler.pipeline.transformer.__class__.cnt = 0
        hunyuan_video_sampler.pipeline.transformer.__class__.num_steps = args.infer_steps
        hunyuan_video_sampler.pipeline.transformer.__class__.previous_residual = {}
        hunyuan_video_sampler.pipeline.transformer.__class__.cachestep = []
        cache_manager_obj.set_mode(args.mode)


    prompt_list = read_prompt_list(prompt_arg)
    args.num_videos = 1

    for index, prompt in enumerate(prompt_list):
        start = time.time()
        outputs = hunyuan_video_sampler.predict(
            prompt=prompt,
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale,
        )
        time_consume = time.time() - start
        samples = outputs["samples"]

        if "LOCAL_RANK" not in os.environ or int(os.environ["LOCAL_RANK"]) == 0:
            for i, sample in enumerate(samples):
                sample = samples[i].unsqueeze(0)
                time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
                save_path_single = f"{save_path}/video{index}-{time_flag}_{args.seed}.mp4"
                save_videos_grid(sample, save_path_single, fps=24)
                logger.info(f"Sample save to: {save_path_single}, time consume:{time_consume:.2f}sec")


if __name__ == "__main__":
    main()