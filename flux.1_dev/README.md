# X-Slim for FLUX.1-dev (Diffusers)

This script adds **X-Slim (Extreme-Slimming Caching)** inference-time acceleration on top of **FLUX.1-dev** using the 🤗 `diffusers(0.34.0)` library.

X-Slim modes:
- `original` – no cache (baseline)
- `S-slow`, `S-fast` – our step-only variant
- `slow`, `fast` – X-Slim C2F (step + block/token)

---

## 1. Setup

1. Install `diffusers` and dependencies as required by your FLUX.1-dev setup.

2. Folder structure (relevant parts):

   ```text
   flux.1_dev/
     ├─ xslim_sample.py
     ├─ cache_utils/
     │   ├─ __init__.py
     │   ├─ cache_manager.py     # X-Slim manager (uses x_slim_config.pth)
     │   ├─ transformer.py       # FluxTransformer2DModel wrapper
     │   └─ x_slim_config.pth    # X-Slim strategies for modes
     └─ ...
   ```

   - `cache_manager.py` should expose `MANAGER` with `set_mode(mode)`.
   - `transformer.py` should define `FluxTransformer2DModel` (compatible with FLUX.1-dev).

3. Make sure you have a **FLUX.1-dev checkpoint** folder, e.g.:

   ```text
   flux/ckpt/
     ├─ transformer/
     ├─ scheduler/
     ├─ vae/
     └─ ...
   ```

## 2. Usage

From inside `flux.1_dev/`:

```bash
CUDA_VISIBLE_DEVICES=0 python xslim_sample.py \
  --model_path /flux/ckpt \
  --mode original
```

### Common arguments

- `--model_path`  
  Path to the FLUX.1-dev checkpoint directory.

- `--prompt`  
  - Single prompt string (default is a cat prompt), e.g.  
    `--prompt "A cute cat wearing a pink beret and a light pink scarf, holding a bouquet of sparkling light pink roses."`
  - Or a `.txt` file path (one prompt per line).

- `--mode`  
  - `original` – no X-Slim caching  
  - `slow`, `fast` – X-Slim C2F  
  - `S-slow`, `S-fast` – X-Slim step-only

- `--output_root`  
  Output root directory (default: `outputs/X-Slim-<mode>/`).

---

## 3. Outputs

Images and prompts are saved to:

```text
outputs/X-Slim-<mode>/
  ├─ image/
  │   ├─ 0000.png
  │   ├─ 0001.png
  │   └─ ...
  └─ text/
      ├─ 0000.txt
      ├─ 0001.txt
      └─ ...
```

Each `.png` has a matching `.txt` file containing the prompt. 
