# X-Slim for HunyuanVideo

This repository adds an **X-Slim (Extreme-Slimming Caching)** inference-time acceleration strategy on top of the original **HunyuanVideo** implementation.

X-Slim provides several **static caching schedules** over diffusion timesteps  
(`S-slow`, `S-fast`, `slow`, `fast`), allowing you to trade off **speed vs. quality** without retraining.

---

## 1. Environment / Setup

Follow the **official HunyuanVideo repository** and complete the original setup:
   - Create and activate the conda environment  
   - Install all dependencies  
   - Download and place model weights (e.g., `model_base`)


## 2. Integrate X-Slim

Copy the X-Slim files from `cache_utils/` into `HunyuanVideo/` so that it looks like:

```text
HunyuanVideo/
  ├─ hyvideo/
  │   ├─ modules/
  │   │   ├─ __init__.py
  │   │   ├─ ...
  │   │   ├─ models.py          # REPLACE original models.py with the X-Slim version
  │   │   ├─ cache_manager.py   # ADD
  │   │   └─ x_slim_config.pth  # ADD
  ├─ xslim_sample.py            # X-Slim sampling script
  └─ ...
```

## 3. Running X-Slim Sampling

From the project root (`HunyuanVideo/`), run for example:

```bash
python3 xslim_sample.py \
    --video-size 540 960 \
    --video-length 81 \
    --infer-steps 50 \
    --prompt prompt.txt \
    --flow-reverse \
    --use-cpu-offload \
    --save-path ./results \
    --seed 42 \
    --mode fast
```

### `--prompt`

Supports:
- a single prompt string
- a `.txt` file (one prompt per line)
- a `.json` file (with `prompt_en` field for vbench.json)


```bash
--prompt "A fluffy teddy bear sits on soft pillows among toys, waving as the camera glides."
```

### `--mode`

- `original` – no caching (original HunyuanVideo)  
- `S-slow`, `S-fast` – our step-level variant X-Slim(S)
- `slow`, `fast` – coarse-to-fine X-Slim (step + block&token level)
