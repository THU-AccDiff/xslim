# X-Slim Offline Preprocessing

## 1) Store strategy data (offline)
The full threshold set can be easily obtained with one prompt and a single offline profiling run.
```bash
# conda activate xslim
cd flux.1_dev_xslim_plain/offline_preprocessing
ASCEND_RT_VISIBLE_DEVICES=0 python sample_store.py --store_data --prompt "A cute cat wearing a pink beret and a light pink scarf, holding a bouquet of sparkling light pink roses."
```

This writes:
- `ori_outputs/strategy_data/img0/step_level/step_l1loss.pth`
- `ori_outputs/strategy_data/img0/block_level/{double_block,single_block}/step*_*.pth`

## 2) Build the default offline schedule

```bash
python /path/to/cache_schedule.py flux.1_dev_xslim_plain/offline_preprocessing/ori_outputs/strategy_data --details
```

`cache_schedule.py` will automatically create `block_level/block_avg/` if missing, then print:
- default thresholds (`step=mean`, `double/single=median`)
- STEP schedule (Stage 1 + Stage 2) when `--details` is enabled

## 3) Tune speed–quality trade-off (optional)

```bash
python /path/to/cache_schedule.py flux.1_dev_xslim_plain/offline_preprocessing/ori_outputs/strategy_data --details --step-thresh 12.0
```

Tune `step_thresh` to reach your desired speed–quality trade-off.
