<div align="center">

<p align="center">
  <img src="https://thu-accdiff.github.io/xslim-page/xslim_files/figs/rocket.png" height="100" alt="X-Slim rocket">
</p>

<h1>X-Slim: No Cache Left Idle</h1>
<h3>Accelerating Diffusion Models via <i>Extreme-Slimming Caching</i></h3>

<div>
  <b>Tingyan Wen<sup>1*</sup></b> · <b>Haoyu Li<sup>1*</sup></b> · <b>Yihuang Chen<sup>2†</sup></b> · <b>Xing Zhou<sup>2</sup></b> · <b>Lifei Zhu<sup>2</sup></b> · <b>XueQian Wang<sup>1†</sup></b>
</div>

<div>
  <sup>1</sup> Tsinghua University &nbsp;&nbsp; <sup>2</sup> Central Media Technology Institute, Huawei
</div>

<div>
  (<sup>*</sup> Equal Contribution &nbsp;&nbsp; <sup>†</sup> Corresponding Author)
</div>

<!-- <br> -->

<a href="https://thu-accdiff.github.io/xslim-page/">
  <img src="https://img.shields.io/badge/Project-Website-green" alt="Project Page">
</a>
<a href="https://github.com/THU-AccDiff/xslim/">
  <img src="https://img.shields.io/badge/Code-GitHub-blue" alt="Code">
</a>
<a href="https://thu-accdiff.github.io/xslim-page/">
  <img src="https://img.shields.io/badge/Paper-TODO-lightgrey" alt="Paper">
</a>

<p align="center">
  <img src="https://thu-accdiff.github.io/xslim-page/xslim_files/figs/fig_display.png" width="100%" alt="X-Slim overview">
</p>

<strong>🚀 X-Slim is a <u>training-free</u> cache-based accelerator that jointly exploits redundancy across <u>temporal</u> (timesteps), <u>structural</u> (blocks), and <u>spatial</u> (tokens) dimensions.</strong>

<details>
<summary><b>📖 Abstract</b> (click to expand)</summary>

Diffusion models deliver strong generative quality, but inference cost scales with timestep count, model depth, and token length. Feature caching reuses nearby computations, yet aggressive timestep skipping often hurts fidelity while conservative block or token refresh yields limited speedup. We present <b>X-Slim</b> (e<b>X</b>treme-<b>Slim</b>ming Caching), a training-free, cache-based accelerator that jointly exploits redundancy across temporal, structural, and spatial dimensions.

X-Slim introduces a dual-threshold <b>push-then-polish</b> controller: it first pushes timestep-level reuse up to an early-warning line, then polishes residual error with lightweight block- and token-level refresh; a critical line triggers full inference to reset error. Level-specific, context-aware indicators guide when and where to cache, shrinking search overhead.

On FLUX.1-dev and HunyuanVideo, X-Slim reduces latency by up to 4.97× and 3.52× with minimal perceptual loss, and on DiT-XL/2 it reaches 3.13× acceleration with a FID improvement of 2.42 over prior methods.

</details>

</div>

---

## ✨ Highlights

- 🔥 **Push-then-Polish** caching with a **dual-threshold controller** (early-warning + critical reset).
- ⚡ **Up to 4.97×** latency reduction on **FLUX.1-dev** and **3.52×** on **HunyuanVideo** (minimal perceptual loss).
- 🏆 **3.13×** acceleration on **DiT-XL/2**, with **FID improved by 2.42** over prior methods.
- 🧩 **Level-specific, context-aware indicators** guide *when* and *where* to reuse vs refresh.

---

## 🎞️ Video Demo

<div align="center">
  <video muted loop playsinline controls width="100%">
    <source src="https://raw.githubusercontent.com/THU-AccDiff/xslim/c851c5741a28398373f40763086dfa579d8802a3/assets/xslim_demo.mp4" type="video/mp4">
  </video>
</div>

---

## 🧭 Todo


- [ ] 🧩 Release a **plug-and-play `xslim_manager` interface** (one-line integration across backbones)
- [ ] 📼 Release a **plug-and-play offline strategy pipeline**: `xslim_recorder` (online recording) + offline strategy export (YAML/PTH)
- [x] 🔗 Release the project page
- [x] 🖼️ Release X-Slim for **FLUX.1-dev**
- [x] 🎞️ Release X-Slim for **HunyuanVideo**
- [x] 📝 Release paper

Our goal is to make X-Slim **truly plug-and-play**: readers can drop in the manager interface, record their own statistics, and design custom schedules—no intrusive model edits required.

---

## 🧠 Method at a Glance

### 1) Push-then-Polish Caching
X-Slim **pushes** step-level reuse until an **early-warning line**, then **polishes** residual error by selectively refreshing **blocks/tokens**.  
When a **critical line** is triggered, X-Slim performs a **full inference step** to reset accumulated error.

### 2) Level-specific Strategy
Different reuse levels follow different dynamics:
- **Step-level**: adjacent timesteps show a *U-shaped* change pattern (weakly prompt-dependent).
- **Block-level**: sensitivity varies with depth, but exhibits *consistent depth-wise patterns*.
- **Token-level**: largely *content-dependent*; refresh focuses on high-change regions.

<p align="center">
  <img src="https://thu-accdiff.github.io/xslim-page/xslim_files/figs/fig_framework.png" width="100%" alt="X-Slim framework">
</p>



---

## 🔧 Installation

> **Note:** X-Slim is training-free and plug-and-play, so it does **not** require any extra environment setup on its own.   
> Keep your existing runnable **FLUX.1-dev / HunyuanVideo** setup (dependencies + checkpoints) and apply X-Slim directly.  



```bash
git clone https://github.com/THU-AccDiff/xslim.git
cd xslim
```

### 1) X-Slim + FLUX.1-dev (🤗 Diffusers)

X-Slim for FLUX.1-dev lives in **`flux.1_dev/`** and relies on your existing FLUX+Diffusers setup. **You need:**
- a working `diffusers` environment for FLUX.1-dev
- a FLUX.1-dev checkpoint folder containing `transformer/`, `vae/`, `scheduler/`, etc.

👉 See: `flux.1_dev/README.md` for folder structure & usage details.

### 2) X-Slim + HunyuanVideo

X-Slim for HunyuanVideo is implemented as a lightweight patch on top of the official repo. **You need:**
- the official HunyuanVideo environment and checkpoints (e.g., `model_base`)
- to integrate X-Slim files (replace/add a few modules)

👉 See: `HunyuanVideo/README.md` for step-by-step integration.



---
## 🔖 Citation

If you find X-Slim helpful, please consider giving a star ⭐ and citing 📝

```bibtex
@article{xslim2026,
  title={No Cache Left Idle: Accelerating Diffusion Model via Extreme-Slimming Caching},
  author={Anonymous},
  journal={CVPR Submission},
  year={2026}
}
```

---

## 🛡️ Disclaimer

This is the official code release of **X-Slim**.  
Demo images/videos are from community users; please contact us if you would like them removed.

---

## 💞 Acknowledgements

We thank the open-source community and upstream projects that made this work possible, including (but not limited to):
- FLUX
- HunyuanVideo
- DiT backbones
- Caching-acceleration works (TeaCache, TaylorSeer,etc.)


