# python /path/to/cache_schedule.py <PATH_TO_STRATEGY_DATA_DIR> [--details] [--step-thresh {mean|median|FLOAT}]

import argparse, statistics, torch
from pathlib import Path


def load_list(p: Path):
    x = torch.load(p)
    if isinstance(x, torch.Tensor): x = x.tolist()
    return [float(v) for v in x]


def save_block_avg(img: Path):
    bl = img / "block_level"
    out = bl / "block_avg"
    if out.exists(): return
    for kind, d in (("double", bl / "double_block"), ("single", bl / "single_block")):
        if not d.exists(): continue
        fs = sorted(d.glob("step*_*l1loss.pth"))
        if not fs: continue
        per = [load_list(f) for f in fs]
        m = max(len(v) for v in per)
        avg = [sum(v[i] for v in per if i < len(v)) / sum(1 for v in per if i < len(v)) for i in range(m)]
        out.mkdir(parents=True, exist_ok=True)
        torch.save(avg, out / f"{kind}_l1loss_avg.pth")


def thresh(vals, kind, arg):
    if arg is None:
        return (float(statistics.fmean(vals)), "mean") if kind == "step" else (float(statistics.median(vals)), "median")
    k = arg.strip().lower()
    if k in ("mean", "average"): return float(statistics.fmean(vals)), "mean"
    if k == "median": return float(statistics.median(vals)), "median"
    return float(arg), "user-float"


def schedule(vals, s, e, th):
    n = len(vals)
    s, e = max(0, s), min(e, n - 1)
    cache, calc, acc = [], [], 0.0
    for i, v in enumerate(vals):
        if i < s or i > e:
            calc.append(i); acc = 0.0; continue
        acc += v
        (cache if acc < th else calc).append(i)
        if acc >= th: acc = 0.0
    return cache, calc


def rows(n, cache):
    w = len(str(max(n - 1, 0)))
    idx = " ".join(str(i).rjust(w) for i in range(n))
    c = set(cache)
    pat = " ".join(("x" if i in c else "o").rjust(w) for i in range(n))
    return idx, pat, w


def stage2(vals, cache_idx, calc_idx, sv, ev, num_steps, w):
    med = statistics.median(vals)
    ss_v = next((i for i in range(sv, ev + 1) if vals[i] < med), None)
    se_v = next((i for i in range(ev, sv - 1, -1) if vals[i] > med), None)
    ok = ss_v is not None and se_v is not None and ss_v < se_v

    block, token, stable = [], [], []
    ss_step = se_step = None
    if ok:
        ss_step, se_step = ss_v + 1, se_v
        cand = [s for s in range(ss_step, se_step) if s >= 1 and vals[s - 1] <= med]
        cset = set(calc_idx)
        stable = sorted(s for s in cand if s in cset)
        for k, s in enumerate(stable):
            (block if k % 3 == 1 else token if k % 3 == 2 else []).append(s)

    cache_s, block_s, token_s = set(cache_idx), set(block), set(token)
    full = [i for i in range(num_steps) if i not in cache_s and i not in block_s and i not in token_s]
    pat = " ".join(("x" if i in cache_s else "b" if i in block_s else "t" if i in token_s else "o").rjust(w)
                   for i in range(num_steps))
    return med, ok, ss_step, se_step, stable, full, block, token, pat


def process_step(step_pth: Path, start: int, end: int, step_thr_arg, details: bool):
    vals = load_list(step_pth)
    n = len(vals)
    th, src = thresh(vals, "step", step_thr_arg)
    num_steps = n + 1

    start_step = max(1, start)
    end_step = min(end if end is not None else num_steps - 2, num_steps - 1)
    sv, ev = max(0, start_step - 1), min(n - 1, end_step - 1)

    cache_raw, calc_raw = schedule(vals, sv, ev, th)
    cache_idx = [i + 1 for i in cache_raw]
    calc_idx = [0] + [i + 1 for i in calc_raw]

    cache_cnt, calc_cnt = len(cache_idx), len(calc_idx)
    cache_pct = cache_cnt / num_steps * 100.0
    speedup = num_steps / calc_cnt

    idx_row, step_row, w = rows(num_steps, cache_idx)

    print("############ Cache schedule Summary ############")
    print(f"File       : {step_pth}")
    print("Kind       : step")
    print(f"Threshold  : {th:.3f} (source: {src})")
    print(f"Reduced    : ↑{cache_pct:.2f}%  (cache_cnt / total_cnt = {cache_cnt} / {num_steps})")
    print(f"Speedup    : ↑{speedup:.2f}×  (total_cnt / calc_cnt = {num_steps} / {calc_cnt})")
    print()
    if not details: return

    print("------------ Cache schedule Details  -----------")
    print(f"Value len  : {n}  (len(step_l1loss))")
    print(f"Total cnt  : {num_steps}  (num steps)")
    print("[X-Slim Stage 1: Coarse Slimming]")
    print(f" -Start idx  : {start_step}")
    print(f" -End idx    : {end_step}")
    print(f" -Cache idx  ({cache_cnt}): {cache_idx}")
    print(f" -Calc idx   ({calc_cnt}): {calc_idx}")

    med, ok, ss, se, stable, full, block, token, all_pat = stage2(vals, cache_idx, calc_idx, sv, ev, num_steps, w)

    print("[X-Slim Stage 2: Fine Slimming]")
    print(f" -Stable span : [{ss}, {se})  (median(values) = {med:.3f})" if ok else f" -Stable span : <none>  (median(values) = {med:.3f})")
    print(f" -Flat calc idx ({len(stable)}): {stable}")
    print(f"   |Full infer idx      ({len(full)}): {full}")
    print(f"   |Block refresh idx    ({len(block)}): {block}")
    print(f"   |Token refresh idx    ({len(token)}): {token}")
    print("------------ Cache Pattern Visualization -----------")
    print(f"Index                : {idx_row}")
    print(f"Step-only Pattern    : {step_row}")
    print(f"All-level Pattern    : {all_pat}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy_root")
    ap.add_argument("--start", type=int, default=4)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--step-thresh", type=str, default=None)
    ap.add_argument("--details", action="store_true")
    args = ap.parse_args()

    root = Path(args.strategy_root)
    for img in sorted(root.glob("img*")):
        step_pth = img / "step_level/step_l1loss.pth"
        if not step_pth.exists():  # minimal: only handle complete img dirs
            continue

        save_block_avg(img)
        dbl_pth = img / "block_level/block_avg/double_l1loss_avg.pth"
        sgl_pth = img / "block_level/block_avg/single_l1loss_avg.pth"

        step_vals = load_list(step_pth)
        step_th, step_src = thresh(step_vals, "step", args.step_thresh)
        dbl_th = statistics.median(load_list(dbl_pth)) if dbl_pth.exists() else None
        sgl_th = statistics.median(load_list(sgl_pth)) if sgl_pth.exists() else None

        print(f"==================== {img.name} ====================")
        print(f"[DEFAULT] step_thresh = {step_th:.6g} ({step_src})")
        print(f"[DEFAULT] double_thresh = {dbl_th:.6g} (median)" if dbl_th is not None else "[DEFAULT] double_thresh = <missing>")
        print(f"[DEFAULT] single_thresh = {sgl_th:.6g} (median)" if sgl_th is not None else "[DEFAULT] single_thresh = <missing>")
        print()

        if args.details:
            process_step(step_pth, args.start, args.end, args.step_thresh, True)
            print("Note: This is the acceleration schedule under the default settings. You can tune step_thresh to reach your desired speed–quality trade-off.")
            print()


if __name__ == "__main__":
    main()
