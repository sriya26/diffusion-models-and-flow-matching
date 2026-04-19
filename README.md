# HW5: Diffusion Models & Flow Matching
**COMS 4732W — Computer Vision 2 — Spring 2026**
**Sriya Rallabandi · sr4326**

---

## Overview

This project has two parts:

- **Part A** — Exploring the pre-trained [DeepFloyd IF-I-L-v1.0](https://huggingface.co/DeepFloyd/IF-I-L-v1.0) diffusion model by implementing sampling loops, inpainting, image-to-image translation (SDEdit), visual anagrams, and hybrid images from scratch. This code is implemented in `coms4732_hw5a_completed.ipynb`, which is ready to run end-to-end with the provided assets.(replace the HuggingFace token in the notebook with your own to run).
- **Part B** — Training a flow matching model from scratch on MNIST, with a custom UNet, time-conditioning, class-conditioning, and classifier-free guidance. This is implemented in `coms4732_hw5b_fixed.ipynb`, which can be run end-to-end to reproduce all results. 
Both notebooks are well-commented and structured for readability.

The project website (`index.html`) contains all deliverable visualizations with explanations.

---

## File Structure

```
hw5/
├── index.html                        # Project website (open in browser)
├── README.md                         # This file
│
├── assets/
│   ├── part_a/                       # All Part A images
│   │   ├── campanile.jpg             # Source test image
│   │   ├── magnolia-royalty-free-image-1750196468.avif  # Web image source
│   │   ├── myImage.png               # Hand-drawn sketch 1 (flower)
│   │   ├── myImage2.png              # Hand-drawn sketch 2 (star/figure)
│   │   │
│   │   ├── upsampled_stage1_[1-3]_256x256.png   # Part 0: text-to-image (stage 2)
│   │   │
│   │   ├── noisy_campanile_256_t{250,500,750}.png    # 1.1 forward process
│   │   ├── gaussian_blurred_256_t{250,500,750}.png   # 1.2 classical denoising
│   │   ├── one_step_denoised_256_t{250,500,750}.png  # 1.3 one-step denoising
│   │   │
│   │   ├── iterative_denoised_256_t690.png    # 1.4 iterative denoising (final)
│   │   ├── one_step_denoised_256_t690.png     # 1.4 one-step at t=690
│   │   ├── gaussian_blurred_256_t690.png      # 1.4 Gaussian at t=690
│   │   │
│   │   ├── a_1_5_sample_[1-5].png             # 1.5 unconditional samples
│   │   ├── a_1_6_cfg_[1-5].png                # 1.6 CFG samples (γ=7)
│   │   │
│   │   ├── a_1_7_campanile_[1,3,5,7,10,20].png       # 1.7 SDEdit on Campanile
│   │   ├── a_1_7_1_web_[1,3,5,7,10,20].png           # 1.7.1 web image SDEdit
│   │   ├── a_1_7_1_drawn1_[original,1,3,5,7,10,20].png  # 1.7.1 sketch 1 SDEdit
│   │   ├── a_1_7_1_drawn2_[original,1,3,5,7,10,20].png  # 1.7.1 sketch 2 SDEdit
│   │   │
│   │   ├── a_1_7_2_original.png               # 1.7.2 inpainting — Campanile orig
│   │   ├── a_1_7_2_mask.png                   # 1.7.2 mask 1
│   │   ├── a_1_7_2_inpainted.png              # 1.7.2 result 1
│   │   ├── a_1_7_2_mask2.png                  # 1.7.2 mask 2 (web image)
│   │   ├── a_1_7_2_inpainted2.png             # 1.7.2 result 2
│   │   ├── a_1_7_2_mask3.png                  # 1.7.2 mask 3 (drawn image)
│   │   ├── a_1_7_2_inpainted3.png             # 1.7.2 result 3
│   │   │
│   │   ├── a_1_7_3_[1,3,5,7,10,20].png        # 1.7.3 text-conditional SDEdit
│   │   │
│   │   ├── a_1_8_illusion1_normal.png          # 1.8 visual anagram 1 (upright)
│   │   ├── a_1_8_illusion1_flipped.png         # 1.8 visual anagram 1 (flipped)
│   │   ├── a_1_8_illusion2_normal.png          # 1.8 visual anagram 2 (upright)
│   │   ├── a_1_8_illusion2_flipped.png         # 1.8 visual anagram 2 (flipped)
│   │   │
│   │   ├── a_1_9_hybrid1.png                   # 1.9 hybrid image 1
│   │   └── a_1_9_hybrid2.png                   # 1.9 hybrid image 2
│   │
│   └── part_b/                                 # All Part B images
│       ├── part1_noising_visualization.png     # B1.2 noising at σ=[0,0.2,...,1]
│       ├── part1_training_curve.png            # B1.2 training loss (σ=0.5)
│       ├── part1_denoising_epoch[1,5].png      # B1.2 denoising results
│       ├── part1_ood_testing.png               # B1.2.2 OOD testing
│       ├── part1_pure_noise_curve.png          # B1.2.3 pure noise training loss
│       ├── part1_pure_noise_epoch[1,5].png     # B1.2.3 pure noise outputs
│       ├── part2_tc_training_curve.png         # B2.2 time-conditioned training loss
│       ├── part2_cc_training_curve.png         # B2.5 class-conditioned training loss
│       ├── part2_cc_no_scheduler.png           # B&W no-scheduler results
│       ├── fm_samples/
│       │   └── part2_tc_epoch[1,5,10].png      # B2.3 time-conditioned samples
│       └── cc_samples/
│           └── part2_cc_epoch[1,5,10].png      # B2.6 class-conditioned samples
│
└── coms4732_hw5a_completed.ipynb     # Part A notebook
└── coms4732_hw5b_fixed.ipynb         # Part B notebook
```

---

## Part A Summary

### Setup (Part 0)
Three custom prompts sampled at `num_inference_steps=20` using DeepFloyd stage 1 (64×64) + stage 2 upsampling (256×256). Seed: 100.

### 1.1 Forward Process
Implemented `forward(im, t)`: applies the DDPM noising formula `x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε` at timesteps t ∈ {250, 500, 750}.

### 1.2 Classical Denoising
Gaussian blur (kernel=5, σ=2) as a baseline. Works partially at low noise, fails at high noise — motivates learned diffusion.

### 1.3 One-Step Denoising
Single UNet forward pass estimates noise; clean image recovered as `x̂₀ = (x_t − √(1−ᾱ_t)·ε̂) / √ᾱ_t`. Quality degrades with t.

### 1.4 Iterative Denoising
Implemented `iterative_denoise()` using strided timesteps (990→0, stride 30). DDPM update with learned variance. Dramatically outperforms one-step at the same noise level.

### 1.5 Unconditional Sampling
Generate images from pure noise using `iterative_denoise` with `i_start=0`. Quality limited without CFG.

### 1.6 Classifier-Free Guidance
Implemented `iterative_denoise_cfg()`. CFG formula: `ε = εᵤ + γ(εc − εᵤ)`, γ=7. Dramatically improves image quality and coherence.

### 1.7 Image-to-Image Translation (SDEdit)
Add noise to a real image at varying levels, then denoise with CFG. Lower `i_start` → more creative edit; higher → preserves original.
- **1.7.1** Applied to web image (magnolia) and two hand-drawn sketches.
- **1.7.2** Inpainting via RePaint: `x_t ← m·x_t + (1−m)·forward(x_orig, t)` at each step.
- **1.7.3** Text-conditioned: same SDEdit but guided by a descriptive prompt.

### 1.8 Visual Anagrams
At each step: `ε = (CFG(x_t, p₁) + flip(CFG(flip(x_t), p₂))) / 2`. Produces images that read differently upright vs. upside-down.

### 1.9 Hybrid Images
Factorized Diffusion: `ε = lowpass(ε₁) + highpass(ε₂)` using Gaussian blur (kernel=33, σ=2). Subject appears to change with viewing distance.

---

## Part B Summary

### B1.1 UNet Architecture
Custom UNet with Conv/DownConv/UpConv/Flatten/Unflatten blocks and skip connections. D=128 for Part 1, D=64 for Part 2.

### B1.2 Single-Step Denoiser
Trained on MNIST with `z = x + σ·ε` (σ=0.5), MSE loss, 5 epochs. Also trained a separate model on pure Gaussian noise — outputs converge to the mean/centroid of the training distribution due to MSE averaging.

### B2.1–2.3 Time-Conditioned Flow Matching
Flow matching interpolation `x_t = (1−t)x₀ + tx₁`, target `u = x₁−x₀`. Time t injected via FCBlocks: `unflatten = unflatten × t₁`, `up1 = up1 × t₂`. Euler sampling over 50 steps.
- D=64, lr=1e-2, ExponentialLR (γ=0.1^(1/10)), 10 epochs.

### B2.4–2.6 Class-Conditioned Flow Matching with CFG
Added class conditioning via one-hot c ∈ ℝ¹⁰: `unflatten = c₁·unflatten + t₁`, `up1 = c₂·up1 + t₂`. Dropout p_uncond=0.1. CFG at sampling: `u_cfg = u_uncond + 5·(u_cond − u_uncond)`.

### Bells & Whistles
Removed exponential LR scheduler; compensated with constant lr=1e-3. Adam's adaptive rates provide implicit stabilization — digit quality is comparable to the scheduled run.

---

## Key Hyperparameters

| Setting | Part A | Part B Part 1 | Part B Part 2 |
|---|---|---|---|
| Model | DeepFloyd IF-I-L-v1.0 | Custom UNet D=128 | Custom UNet D=64 |
| Dataset | — | MNIST | MNIST |
| Optimizer | — | Adam lr=1e-4 | Adam lr=1e-2 |
| Batch size | — | 256 | 64 |
| Epochs | — | 5 | 10 |
| LR schedule | — | None | ExponentialLR γ=0.1^(1/10) |
| CFG scale | γ=7 | — | γ=5 |
| Seed | 100 | — | — |

---

## How to View

Open `index.html` in any modern browser. No server required, all assets are local.