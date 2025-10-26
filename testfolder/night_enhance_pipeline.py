# -*- coding: utf-8 -*-
"""
Night/Low-light Enhancement Pipeline (PLUS)
- Gray-World AWB
- (옵션) Denoise (fastNlMeansDenoisingColored)
- CLAHE on L* with highlight-protection blending
- Gamma-based shadow lift (weighted)
- Mild saturation
- (옵션 a) Detail boost (bilateral/gaussian/DoG)
- (옵션 b) Single-image Exposure Fusion (multi-scale, Laplacian pyramid)
- (옵션) Unsharp mask

Usage (기본):
    python night_enhance_plus.py input.jpg

극적 프리셋 예:
    python night_enhance_plus.py input.jpg --clip 3.4 --gamma 0.80 --sat 1.20 \
        --w-alpha 0.88 --w-start 0.72 --w-end 0.90 \
        --detail --detail-gain 1.35 --detail-method bilateral \
        --xfuse --xfuse-pos before --xfuse-gammas 0.6,0.8,1.0,1.25 --xfuse-levels 5 \
        --sharpen 0.7 1.3 4
"""

import cv2
import numpy as np
import argparse
import os
from datetime import datetime

# ---------------------------
# Utilities
# ---------------------------

def imread_color(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def clamp_u8(arr):
    return np.clip(arr, 0, 255).astype(np.uint8)

def to_float01(img_bgr_u8):
    return (img_bgr_u8.astype(np.float32) / 255.0).clip(0,1)

def from_float01(img_bgr_f):
    return (np.clip(img_bgr_f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

def basename_with_suffix(path, suffix):
    base, ext = os.path.splitext(os.path.basename(path))
    ext = ext if ext else ".png"
    return f"{base}_{suffix}{ext}"

def parse_floats_csv(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]

# ---------------------------
# Color/WB & Denoise
# ---------------------------

def gray_world_awb(bgr, strength=1.0):
    if strength <= 0: return bgr
    eps = 1e-6
    bgrf = bgr.astype(np.float32)
    means = bgrf.reshape(-1, 3).mean(axis=0)  # B,G,R
    gray = means.mean()
    gains = gray / (means + eps)
    gains = (1.0 - strength) * np.array([1.0,1.0,1.0], np.float32) + strength * gains
    out = bgrf * gains
    return clamp_u8(out)

def denoise_colored(bgr, h=7, hColor=5, template=7, search=21):
    return cv2.fastNlMeansDenoisingColored(
        bgr, None, h=float(h), hColor=float(hColor),
        templateWindowSize=int(template), searchWindowSize=int(search)
    )

# ---------------------------
# CLAHE on L* + highlight protection
# ---------------------------

def clahe_on_L(bgr, clip_limit=2.8, tile_size=8):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=(int(tile_size), int(tile_size)))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR), L, L2

def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def shadow_weight_map_from_L(L, high_start=0.78, high_end=0.94, alpha=0.95):
    Lf = L.astype(np.float32) / 255.0
    hi = smoothstep(high_start, high_end, Lf)
    w = 1.0 - alpha * hi  # dark~1, highlight~1-alpha
    return w[..., None].astype(np.float32)

def gamma_lift_weighted(bgr, gamma=0.85, weight=None):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32) / 255.0
    Lg = np.power(np.clip(L, 0.0, 1.0), float(gamma))
    if weight is not None:
        w = weight.squeeze()
        Lout = w * Lg + (1.0 - w) * L
    else:
        Lout = Lg
    lab[:, :, 0] = np.clip(Lout * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def adjust_saturation(bgr, sat=1.12):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= float(sat)
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# ---------------------------
# (a) Detail boost
# ---------------------------

def detail_boost(bgr, method="bilateral", gain=1.3, sigma=2.0, sigma_color=12.0):
    """
    method: 'bilateral' | 'gaussian' | 'dog'
    gain:   detail exaggeration factor (>=1)
    sigma:  space sigma or Gaussian sigma
    sigma_color: bilateral sigmaColor
    """
    f = bgr.astype(np.float32)
    if method == "bilateral":
        base = cv2.bilateralFilter(bgr, d=0, sigmaColor=float(sigma_color), sigmaSpace=float(sigma))
        base = base.astype(np.float32)
        detail = f - base
        out = base + float(gain) * detail
    elif method == "gaussian":
        base = cv2.GaussianBlur(f, (0,0), sigmaX=float(sigma), sigmaY=float(sigma))
        detail = f - base
        out = base + float(gain) * detail
    elif method == "dog":  # Difference of Gaussians
        blur1 = cv2.GaussianBlur(f, (0,0), sigmaX=float(sigma), sigmaY=float(sigma))
        blur2 = cv2.GaussianBlur(f, (0,0), sigmaX=float(sigma)*1.6, sigmaY=float(sigma)*1.6)
        detail = blur1 - blur2
        out = f + float(gain-1.0) * detail  # gain around 1.2~1.6 추천
    else:
        out = f
    return clamp_u8(out)

# ---------------------------
# (b) Exposure Fusion (Mertens-like; Laplacian pyramids)
# ---------------------------

def _gaussian_pyramid(img, levels):
    pyr = [img]
    for _ in range(levels-1):
        img = cv2.pyrDown(img)
        pyr.append(img)
    return pyr

def _laplacian_pyramid(img, levels):
    gp = _gaussian_pyramid(img, levels)
    lp = []
    for i in range(levels-1):
        up = cv2.pyrUp(gp[i+1])
        up = cv2.resize(up, (gp[i].shape[1], gp[i].shape[0]), interpolation=cv2.INTER_LINEAR)
        lp.append(gp[i] - up)
    lp.append(gp[-1])
    return lp

def _pyramid_reconstruct(lp):
    img = lp[-1]
    for i in range(len(lp)-2, -1, -1):
        up = cv2.pyrUp(img)
        up = cv2.resize(up, (lp[i].shape[1], lp[i].shape[0]), interpolation=cv2.INTER_LINEAR)
        img = up + lp[i]
    return img

def _exposure_fusion_core(imgs_f01, w_contrast=1.0, w_saturation=1.0, w_well=1.0, levels=5, well_sigma=0.2):
    """
    imgs_f01: list of float32 BGR images in [0,1]
    returns float32 BGR [0,1]
    """
    eps = 1e-12
    H, W = imgs_f01[0].shape[:2]

    # 1) Weight maps
    weights = []
    for img in imgs_f01:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Contrast weight: |Laplacian|
        c = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        # Saturation weight: per-pixel std across channels
        s = np.std(img, axis=2)
        # Well-exposedness: Gaussian around 0.5
        we = np.exp(-0.5 * ((img - 0.5) / well_sigma) ** 2).prod(axis=2)
        w = (c + eps) ** w_contrast * (s + eps) ** w_saturation * (we + eps) ** w_well
        weights.append(w)

    # Normalize weights
    Wsum = np.sum(weights, axis=0) + eps
    weights = [ (w / Wsum).astype(np.float32) for w in weights ]

    # 2) Pyramid blend
    # Convert weights to 3-ch
    weights3 = [cv2.merge([w, w, w]) for w in weights]
    # Build pyramids
    img_laps = [ _laplacian_pyramid(img, levels) for img in imgs_f01 ]
    w_gps   = [ _gaussian_pyramid(w3, levels)   for w3 in weights3   ]

    out_lp = []
    for lvl in range(levels):
        acc = np.zeros_like(img_laps[0][lvl], dtype=np.float32)
        for i in range(len(imgs_f01)):
            acc += w_gps[i][lvl] * img_laps[i][lvl]
        out_lp.append(acc)
    fused = _pyramid_reconstruct(out_lp)
    return np.clip(fused, 0.0, 1.0)

def exposure_fusion_from_single(bgr, gammas=(0.6, 0.8, 1.0, 1.25), levels=5,
                                w_contrast=1.0, w_saturation=1.0, w_well=1.0, well_sigma=0.2):
    """
    Create virtual exposures by gamma, then fuse via multi-scale exposure fusion.
    """
    base = to_float01(bgr)
    imgs = []
    for g in gammas:
        out = np.power(np.clip(base, 0, 1), float(g))
        imgs.append(out.astype(np.float32))
    fused = _exposure_fusion_core(imgs, w_contrast, w_saturation, w_well, levels, well_sigma)
    return from_float01(fused)

# ---------------------------
# Unsharp mask
# ---------------------------

def unsharp_mask(bgr, amount=0.5, radius=1.2, threshold=3):
    blurred = cv2.GaussianBlur(bgr, (0,0), sigmaX=float(radius), sigmaY=float(radius))
    diff = bgr.astype(np.int16) - blurred.astype(np.int16)
    if threshold > 0:
        mask = (np.max(np.abs(diff), axis=2) > threshold).astype(np.float32)[..., None]
    else:
        mask = 1.0
    sharp = bgr.astype(np.float32) + float(amount) * diff.astype(np.float32)
    out = mask * sharp + (1.0 - mask) * bgr.astype(np.float32)
    return clamp_u8(out)

# ---------------------------
# Main pipeline
# ---------------------------

def enhance_lowlight_plus(
    bgr,
    # Stage 0: AWB + Denoise
    awb_strength=1.0,
    denoise_h=7, denoise_hc=5, denoise_template=7, denoise_search=21,
    # Stage XFUSE
    use_xfuse=False, xfuse_pos="before", xfuse_gammas=(0.6,0.8,1.0,1.25),
    xfuse_levels=5, xfuse_contrast=1.0, xfuse_saturation=1.0, xfuse_well=1.0, xfuse_sigma=0.2,
    # Stage 1: CLAHE
    clahe_clip=2.8, clahe_tile=8,
    w_hi_start=0.78, w_hi_end=0.94, w_alpha=0.95,
    # Stage 2: Gamma & saturation
    gamma=0.85, sat=1.12,
    # Stage A: Detail boost
    use_detail=False, detail_method="bilateral", detail_gain=1.3, detail_sigma=2.0, detail_sigma_color=12.0,
    # Stage 3: Sharpen
    sharpen_amount=0.5, sharpen_radius=1.2, sharpen_threshold=3
):
    # 0) AWB
    bgr0 = gray_world_awb(bgr, strength=awb_strength)

    # 0.5) Denoise
    bgr1 = denoise_colored(bgr0, h=denoise_h, hColor=denoise_hc,
                           template=denoise_template, search=denoise_search) \
           if (denoise_h>0 or denoise_hc>0) else bgr0

    # (b) Exposure fusion (선택) - before CLAHE
    if use_xfuse and xfuse_pos == "before":
        bgr1 = exposure_fusion_from_single(bgr1, gammas=xfuse_gammas, levels=xfuse_levels,
                                           w_contrast=xfuse_contrast, w_saturation=xfuse_saturation,
                                           w_well=xfuse_well, well_sigma=xfuse_sigma)

    # 1) CLAHE + highlight protection
    bgr_clahe, L_orig, _ = clahe_on_L(bgr1, clip_limit=clahe_clip, tile_size=clahe_tile)
    w = shadow_weight_map_from_L(L_orig, high_start=w_hi_start, high_end=w_hi_end, alpha=w_alpha)
    bgr2 = clamp_u8(w * bgr_clahe.astype(np.float32) + (1.0 - w) * bgr1.astype(np.float32))

    # 2) Gamma lift (weighted) + mild saturation
    bgr3 = gamma_lift_weighted(bgr2, gamma=gamma, weight=w)

    # (b) Exposure fusion (선택) - after CLAHE
    if use_xfuse and xfuse_pos == "after":
        bgr3 = exposure_fusion_from_single(bgr3, gammas=xfuse_gammas, levels=xfuse_levels,
                                           w_contrast=xfuse_contrast, w_saturation=xfuse_saturation,
                                           w_well=xfuse_well, well_sigma=xfuse_sigma)

    bgr4 = adjust_saturation(bgr3, sat=sat)

    # (a) Detail boost (선택)
    if use_detail and detail_gain > 1.0:
        bgr4 = detail_boost(bgr4, method=detail_method, gain=detail_gain,
                            sigma=detail_sigma, sigma_color=detail_sigma_color)

    # 3) Unsharp (선택)
    bgr5 = unsharp_mask(bgr4, amount=sharpen_amount, radius=sharpen_radius,
                        threshold=int(sharpen_threshold)) if sharpen_amount>0 else bgr4

    return bgr5

# ---------------------------
# CLI
# ---------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Night / Low-light Enhancement (PLUS: detail boost & exposure fusion)")
    p.add_argument("--image", help="Input image path")
    p.add_argument("--out", help="Output path (default: <name>_enhanced_YYYYmmdd_HHMMSS.png)", default="./imglog/enhanced/")

    # AWB & Denoise
    p.add_argument("--awb", type=float, default=1.0, help="Gray-world AWB strength [0..1], default=1.0")
    p.add_argument("--denoise", nargs=2, type=float, metavar=("h","hColor"), default=[7,5],
                   help="fastNlMeans Denoise strengths: h hColor (0 0 to disable)")
    p.add_argument("--dn-template", type=int, default=7, help="Denoise templateWindowSize (default 7)")
    p.add_argument("--dn-search", type=int, default=21, help="Denoise searchWindowSize (default 21)")

    # Exposure Fusion
    p.add_argument("--xfuse", action="store_true", help="Enable exposure fusion from single image")
    p.add_argument("--xfuse-pos", choices=["before","after"], default="before",
                   help="Apply exposure fusion before or after CLAHE (default: before)")
    p.add_argument("--xfuse-gammas", type=str, default="0.6,0.8,1.0,1.25",
                   help="Comma list of gammas for virtual exposures (default '0.6,0.8,1.0,1.25')")
    p.add_argument("--xfuse-levels", type=int, default=5, help="Pyramid levels (default 5)")
    p.add_argument("--xfuse-contrast", type=float, default=1.0, help="Weight exponent for contrast (default 1.0)")
    p.add_argument("--xfuse-saturation", type=float, default=1.0, help="Weight exponent for saturation (default 1.0)")
    p.add_argument("--xfuse-well", type=float, default=1.0, help="Weight exponent for well-exposedness (default 1.0)")
    p.add_argument("--xfuse-sigma", type=float, default=0.2, help="Well-exposedness sigma (default 0.2)")

    # CLAHE
    p.add_argument("--clip", type=float, default=2.8, help="CLAHE clipLimit (default 2.8)")
    p.add_argument("--tile", type=int, default=8, help="CLAHE tileGridSize (square, default 8)")

    # Highlight protection weights
    p.add_argument("--w-start", type=float, default=0.78, help="Weight high_start in [0..1] (default 0.78)")
    p.add_argument("--w-end", type=float, default=0.94, help="Weight high_end in [0..1] (default 0.94)")
    p.add_argument("--w-alpha", type=float, default=0.95, help="Highlight protection alpha [0..1] (default 0.95)")

    # Gamma & Saturation
    p.add_argument("--gamma", type=float, default=0.85, help="Gamma for shadow lift (<1 brightens)")
    p.add_argument("--sat", type=float, default=1.12, help="Saturation scale (default 1.12)")

    # Detail boost
    p.add_argument("--detail", action="store_true", help="Enable detail boost")
    p.add_argument("--detail-method", choices=["bilateral","gaussian","dog"], default="bilateral",
                   help="Detail base filter (default bilateral)")
    p.add_argument("--detail-gain", type=float, default=1.3, help="Detail exaggeration factor (>=1, default 1.3)")
    p.add_argument("--detail-sigma", type=float, default=2.0, help="Detail base sigma/space (default 2.0)")
    p.add_argument("--detail-sigma-color", type=float, default=12.0, help="Bilateral sigmaColor (default 12.0)")

    # Sharpen
    p.add_argument("--sharpen", nargs=3, type=float, metavar=("amount","radius","threshold"),
                   default=[0.5, 1.2, 3.0], help="Unsharp mask params; set amount<=0 to disable")

    # Save steps (optional)
    p.add_argument("--save-steps", action="store_true", help="Save final image (and optionally inspect by editing code)")
    return p

def main():
    args = build_parser().parse_args()

    img = imread_color(args.image)

    out_img = enhance_lowlight_plus(
        img,
        awb_strength=float(args.awb),
        denoise_h=float(args.denoise[0]),
        denoise_hc=float(args.denoise[1]),
        denoise_template=int(args.dn_template),
        denoise_search=int(args.dn_search),

        use_xfuse=bool(args.xfuse),
        xfuse_pos=str(args.xfuse_pos),
        xfuse_gammas=tuple(parse_floats_csv(args.xfuse_gammas)),
        xfuse_levels=int(args.xfuse_levels),
        xfuse_contrast=float(args.xfuse_contrast),
        xfuse_saturation=float(args.xfuse_saturation),
        xfuse_well=float(args.xfuse_well),
        xfuse_sigma=float(args.xfuse_sigma),

        clahe_clip=float(args.clip),
        clahe_tile=int(args.tile),
        w_hi_start=float(args.w_start),
        w_hi_end=float(args.w_end),
        w_alpha=float(args.w_alpha),

        gamma=float(args.gamma),
        sat=float(args.sat),

        use_detail=bool(args.detail),
        detail_method=str(args.detail_method),
        detail_gain=float(args.detail_gain),
        detail_sigma=float(args.detail_sigma),
        detail_sigma_color=float(args.detail_sigma_color),

        sharpen_amount=float(args.sharpen[0]),
        sharpen_radius=float(args.sharpen[1]),
        sharpen_threshold=float(args.sharpen[2]),
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out + basename_with_suffix(args.image, f"enhanced_{ts}")
    cv2.imwrite(out_path, out_img)
    print(f"[+] Saved: {out_path}")

    if args.save_steps:
        base = os.path.splitext(out_path)[0]
        cv2.imwrite(f"{base}_final.png", out_img)

if __name__ == "__main__":
    main()
