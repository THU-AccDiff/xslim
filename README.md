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

<a href="https://thu-accdiff.github.io/xslim-page/">
  <img src="https://img.shields.io/badge/Project-Website-green" alt="Project Page">
</a>
<a href="https://github.com/THU-AccDiff/xslim/">
  <img src="https://img.shields.io/badge/Code-GitHub-blue" alt="Code">
</a>
<a href="https://arxiv.org/abs/2512.12604">
  <img src="https://img.shields.io/badge/arXiv-2512.12604-b31b1b.svg" alt="arXiv">
</a>

<br>
---
</div>

> ### 🙏 Apology on Code Release
> Part of this work was completed during an internship at **Huawei**, so releasing the full code is currently subject to internal compliance review. We’re sorry that the repository can’t yet include the complete implementation.
>
> - If the review is approved, we will publish the full code here promptly.
> - If it is not approved, we will release a clean, self-written **plug-and-play interface** plus a **minimal demo** and **backbone adaptation notes**.
>
> Thanks for your patience and understanding. We sincerely appreciate your continued interest and support.

<p align="center">
  <img src="https://thu-accdiff.github.io/xslim-page/xslim_files/figs/fig_display.png" width="100%" alt="X-Slim overview">
</p>

<div align="center">
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

https://github.com/user-attachments/assets/743e743b-16db-4076-85dc-5a7065ad26ab

---


## 🧭 Todo

- [ ] 📼 **Pending review:** Publish the full code in this repo once the internal review is cleared. If it cannot be released, we will ship a **clean, self-written demo + backbone adaptation guide** so you can try X-Slim on your own model with minimal effort.
- [x] 🔗 Release the project page
- [x] 📝 Release the arXiv paper


Our goal is to make X-Slim **truly plug-and-play**: readers can drop in the manager interface, record their own statistics, and design custom schedules.

---

## 🧠 Method at a Glance

### 1) Push-then-Polish Caching
X-Slim **pushes** step-level reuse until an **early-warning line**, then **polishes** residual error by selectively refreshing **blocks/tokens**. When a **critical line** is triggered, X-Slim performs a **full inference step** to reset accumulated error.

### 2) Level-specific Strategy
Different reuse levels follow different dynamics:
- **Step-level**: adjacent timesteps show a *U-shaped* change pattern (weakly prompt-dependent).
- **Block-level**: sensitivity varies with depth, but exhibits *consistent depth-wise patterns*.
- **Token-level**: largely *content-dependent*; refresh focuses on high-change regions.

<p align="center">
  <img src="https://thu-accdiff.github.io/xslim-page/xslim_files/figs/fig_framework.png" width="100%" alt="X-Slim framework">
</p>


---
## 🔖 Citation

If you find X-Slim helpful, please consider giving a star ⭐ and citing 📝

```bibtex
@article{xslimcache2025,
  title={No Cache Left Idle: Accelerating Diffusion Model via Extreme-Slimming Caching},
  author={Wen, Tingyan and Li, Haoyu and Chen, Yihuang and Zhou, Xing and Zhu, Lifei and Wang, XueQian},
  journal={arXiv preprint arXiv:2512.12604},
  year={2025}
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


