# -*- coding: utf-8 -*-
import os
import glob
import math
import argparse
from typing import Optional, Tuple, List, Dict

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ==== 프로젝트 유틸 (학습/추론 코드와 동일 모듈 재사용) ====
from tools import (
    StereoFolderDataset,
    denorm_imagenet,
    warp_right_to_left_image,
    PhotometricLoss,
    load_ms2_gt_depth_batch,
    compute_ms2_disparity_metrics,
)

# ============================================================
# 공용 저장/시각화 유틸
# ============================================================
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _basename_wo_ext(path):
    return os.path.splitext(os.path.basename(path))[0]

def save_npy(path, np_array):
    _ensure_dir(os.path.dirname(path))
    np.save(path, np_array.astype(np.float32), allow_pickle=False)

def save_gray_png(path, np_array_uint8):
    from PIL import Image
    _ensure_dir(os.path.dirname(path))
    Image.fromarray(np_array_uint8.astype(np.uint8), mode="L").save(path)

def save_colormap_png_with_colorbar_auto_range(
    path,
    np_array,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_name: str = "magma",
    label: str = "",
    bg_color: str = "#1e1e1e",
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, to_rgba

    _ensure_dir(os.path.dirname(path))
    arr = np.array(np_array, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        vmin_eff = 0.0 if vmin is None else float(vmin)
        vmax_eff = 1.0 if vmax is None else float(vmax)
    else:
        vmin_eff = float(np.nanmin(arr[finite])) if vmin is None else float(vmin)
        vmax_eff = float(np.nanmax(arr[finite])) if vmax is None else float(vmax)
        vmax_eff = max(vmax_eff, vmin_eff + 1e-6)

    import numpy as _np
    base = plt.get_cmap(cmap_name)
    cmap = ListedColormap(base(_np.linspace(0, 1, 256)))
    cmap.set_bad(to_rgba(bg_color))

    H, W = arr.shape
    dpi = 200.0
    figsize = (W / dpi, H / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin_eff, vmax=vmax_eff)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if label:
        cbar.set_label(label, rotation=270, labelpad=12)

    plt.tight_layout(pad=0.1)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1, facecolor=fig.get_facecolor())
    plt.close(fig)

def annotate_png_top_left(path: str, text: str, margin: int = 5):
    from PIL import Image, ImageDraw, ImageFont
    try:
        img = Image.open(path).convert("RGBA")
    except Exception:
        return
    W, H = img.size
    fs = max(10, int(min(W, H) * 0.025))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", fs)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", fs)
        except Exception:
            font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)

    try:
        bbox = draw.multiline_textbbox((0, 0), text, font=font, stroke_width=2, spacing=2)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        try:
            bbox = draw.textbbox((0, 0), text, font=font, stroke_width=2)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = draw.textsize(text, font=font)

    bg = Image.new("RGBA", (tw + 2 * margin, th + 2 * margin), (0, 0, 0, 100))
    img.paste(bg, (margin, margin), bg)
    draw.multiline_text(
        (margin * 2, margin * 2),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 255),
        spacing=2,
    )
    img = img.convert("RGB")
    img.save(path)

def save_disp_overlay_on_image(
    path: str,
    base_bgr_u8: np.ndarray,
    disp_np: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_name: str = "magma",
    alpha: float = 0.8,
):
    """
    disparity(colormap)를 입력 이미지 위에 얹어서 저장.

    base_bgr_u8 : HxWx3, uint8 (BGR)
    disp_np     : HxW, float32 (임의 스케일)
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import cm

    _ensure_dir(os.path.dirname(path))

    base = base_bgr_u8
    H, W = base.shape[:2]

    arr = np.array(disp_np, dtype=np.float32)
    if arr.shape != (H, W):
        arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)

    # valid mask: finite & >0
    valid = np.isfinite(arr) & (arr > 0)
    finite = np.isfinite(arr) & valid

    if not finite.any():
        vmin_eff = 0.0 if vmin is None else float(vmin)
        vmax_eff = 1.0 if vmax is None else float(vmax)
    else:
        vmin_eff = float(np.nanmin(arr[finite])) if vmin is None else float(vmin)
        vmax_eff = float(np.nanmax(arr[finite])) if vmax is None else float(vmax)
        vmax_eff = max(vmax_eff, vmin_eff + 1e-6)

    normed = (arr - vmin_eff) / (vmax_eff - vmin_eff)
    normed = np.clip(normed, 0.0, 1.0)

    cmap = cm.get_cmap(cmap_name)
    disp_rgba = cmap(normed)  # HxWx4, 0~1
    disp_rgb = (disp_rgba[..., :3] * 255.0).astype(np.uint8)
    disp_bgr = disp_rgb[..., ::-1]
    disp_bgr[~valid] = 0

    base_f = base.astype(np.float32)
    disp_f = disp_bgr.astype(np.float32)

    out = (1.0 - float(alpha)) * base_f + float(alpha) * disp_f
    out = np.clip(out, 0, 255).astype(np.uint8)

    cv2.imwrite(path, out)

# ============================================================
# Photometric error 계산 (L1+SSIM)
# ============================================================
@torch.no_grad()
def compute_pth_error_map(
    imgL_01: torch.Tensor,
    imgR_01: torch.Tensor,
    disp_px: torch.Tensor,
    w_l1=0.15,
    w_ssim=0.85,
) -> Tuple[torch.Tensor, torch.Tensor]:
    imgR_warp, valid = warp_right_to_left_image(imgR_01, disp_px)
    pth = PhotometricLoss([w_l1, w_ssim]).simple_photometric_loss(
        imgL_01, imgR_warp, weights=[w_l1, w_ssim]
    )
    pth = torch.where(valid > 0.5, pth, torch.full_like(pth, float("nan")))
    return pth, valid

# ============================================================
# 파일 매칭 유틸
# ============================================================
def find_file_by_stem(folder: str, stem: str, exts: List[str]) -> Optional[str]:
    for ext in exts:
        cand = os.path.join(folder, f"{stem}{ext}")
        if os.path.isfile(cand):
            return cand
    for ext in exts:
        ms = glob.glob(os.path.join(folder, f"{stem}*{ext}"))
        if len(ms) > 0:
            return ms[0]
    return None

# ============================================================
# 워핑 유틸
# ============================================================
def _warp_right_to_left_gray_cv2(right_gray_u8: np.ndarray, disp_px: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H, W = right_gray_u8.shape
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mapx = xs - disp_px.astype(np.float32)
    mapy = ys
    right_warp = cv2.remap(
        right_gray_u8,
        mapx,
        mapy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = (mapx >= 0) & (mapx <= (W - 1)) & (mapy >= 0) & (mapy <= (H - 1))
    return right_warp, valid

def _warp_right_to_left_bgr_cv2(right_bgr_u8: np.ndarray, disp_px: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H, W, _ = right_bgr_u8.shape
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mapx = xs - disp_px.astype(np.float32)
    mapy = ys
    right_warp = cv2.remap(
        right_bgr_u8,
        mapx,
        mapy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = (mapx >= 0) & (mapx <= (W - 1)) & (mapy >= 0) & (mapy <= (H - 1))
    return right_warp, valid

# ============================================================
# Residual-SGM matcher 구성
# ============================================================
def _build_sgbm_for_residual(
    r_px: float,
    block_size: int,
    uniqueness: int,
    speckle_win: int,
    speckle_range: int,
    mode: str,
    use_color: bool = False,
):
    r = max(1, int(round(float(r_px))))
    min_disp = -r
    num_disp = int(np.ceil(2 * r / 16.0) * 16)
    if num_disp < 16:
        num_disp = 16

    block_size = int(block_size) if int(block_size) % 2 == 1 else int(block_size) + 1
    block_size = max(3, min(block_size, 11))

    cn = 3 if use_color else 1
    P1 = 8 * cn * (block_size**2)
    P2 = 32 * cn * (block_size**2)

    if mode.upper() == "HH":
        sgbm_mode = cv2.STEREO_SGBM_MODE_HH
    elif mode.upper() == "SGBM":
        sgbm_mode = cv2.STEREO_SGBM_MODE_SGBM
    else:
        sgbm_mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY

    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        preFilterCap=31,
        uniquenessRatio=int(uniqueness),
        speckleWindowSize=int(speckle_win),
        speckleRange=int(speckle_range),
        mode=sgbm_mode,
    )
    return matcher, min_disp, num_disp

# ============================================================
# 후처리 유틸(WLS, WMF, LR-consistency, plane refine)
# ============================================================
def _apply_wls_generic_if_available(
    disp_px: np.ndarray, guide_bgr_u8: np.ndarray, lam: float, sigma_col: float
) -> np.ndarray:
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "createDisparityWLSFilterGeneric"):
        return disp_px
    wls = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
    wls.setLambda(float(lam))
    wls.setSigmaColor(float(sigma_col))
    d16 = np.clip(disp_px, 0, None).astype(np.float32) * 16.0
    d16 = np.clip(d16, -32768, 32767).astype(np.int16)
    out16 = wls.filter(d16, guide_bgr_u8)
    out = (out16.astype(np.float32) / 16.0)
    out = np.clip(out, 0, None)
    return out

def _weighted_median_if_available(
    disp_px: np.ndarray, guide_bgr_u8: np.ndarray, radius: int, sigma_color: float
) -> np.ndarray:
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "weightedMedianFilter"):
        try:
            disp32 = disp_px.astype(np.float32)
            disp_filtered = cv2.ximgproc.weightedMedianFilter(
                guide_bgr_u8, disp32, radius, sigma_color
            )
            return disp_filtered.astype(np.float32)
        except Exception:
            pass
    disp32 = disp_px.astype(np.float32)
    scale = 16.0
    d16 = np.clip(disp32 * scale, 0, 65535).astype(np.uint16)
    ksize = max(3, radius | 1)
    d16f = cv2.medianBlur(d16, ksize)
    return (d16f.astype(np.float32) / scale)

def _build_right_from_left(dispL: np.ndarray) -> np.ndarray:
    H, W = dispL.shape
    xs = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
    d = dispL
    xr = np.round(xs - d).astype(np.int32)
    valid = (d > 0) & (xr >= 0) & (xr < W)
    dispR = np.zeros_like(d, dtype=np.float32)
    flat_index = (np.arange(H)[:, None] * W + xr)[valid]
    np.maximum.at(dispR.ravel(), flat_index, d[valid])
    return dispR

def _lr_consistency_mask(dispL: np.ndarray, dispR: np.ndarray, tau: float) -> np.ndarray:
    H, W = dispL.shape
    xs = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
    dL = dispL
    xR = np.round(xs - dL).astype(np.int32)
    valid = (dL > 0) & (xR >= 0) & (xR < W)
    if not valid.any():
        return np.zeros_like(dL, dtype=bool)
    y_idx = np.arange(H)[:, None].repeat(W, axis=1)[valid]
    xR_idx = xR[valid]
    dR = dispR[y_idx, xR_idx]
    valid2 = dR > 0
    good = np.zeros_like(dL, dtype=bool)
    good_idx = valid.copy()
    good_idx[valid] = valid2
    if good_idx.any():
        diff_vals = np.abs(dL[good_idx] - dR[valid2])
        good[good_idx] = diff_vals <= tau
    return good

def _edge_aware_hole_fill(disp: np.ndarray, guide_bgr: np.ndarray, max_iter: int = 3) -> np.ndarray:
    H, W = disp.shape
    d = disp.copy().astype(np.float32)
    for _ in range(max_iter):
        valid = (d > 0).astype(np.float32)
        if valid.sum() == H * W:
            break
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        sum_nb = cv2.filter2D(d, -1, kernel)
        cnt_nb = cv2.filter2D(valid, -1, kernel)
        to_fill = (d <= 0) & (cnt_nb > 0)
        new_vals = sum_nb / (cnt_nb + 1e-6)
        d[to_fill] = new_vals[to_fill]
    return d

def _plane_refine_superpixel_if_available(
    disp: np.ndarray, guide_bgr: np.ndarray, min_samples: int = 30
) -> np.ndarray:
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "createSuperpixelSLIC"):
        return disp
    H, W = disp.shape
    img = guide_bgr
    slic = cv2.ximgproc.createSuperpixelSLIC(
        img, algorithm=cv2.ximgproc.SLICO, region_size=20, ruler=10.0
    )
    slic.iterate(10)
    labels = slic.getLabels()
    n_sp = slic.getNumberOfSuperpixels()
    out = disp.copy().astype(np.float32)
    ys_idx, xs_idx = np.indices((H, W))
    for k in range(n_sp):
        mask_all = labels == k
        mask_val = mask_all & (disp > 0)
        ys = ys_idx[mask_val]
        xs = xs_idx[mask_val]
        if xs.size < min_samples:
            continue
        dvals = disp[mask_val]
        A = np.stack(
            [xs.astype(np.float32), ys.astype(np.float32), np.ones_like(xs, dtype=np.float32)],
            axis=1,
        )
        try:
            coeff, *_ = np.linalg.lstsq(A, dvals.astype(np.float32), rcond=None)
        except Exception:
            continue
        a, b, c = coeff
        mask_fill = mask_all & (out <= 0)
        if not mask_fill.any():
            continue
        ys_f = ys_idx[mask_fill].astype(np.float32)
        xs_f = xs_idx[mask_fill].astype(np.float32)
        plane_vals = a * xs_f + b * ys_f + c
        out[mask_fill] = plane_vals
    return out

# ============================================================
# Residual-SGM 한 패스
# ============================================================
def residual_sgm_pass(
    L_bgr: np.ndarray,
    R_bgr: np.ndarray,
    L_gray: np.ndarray,
    R_gray: np.ndarray,
    D0_np: np.ndarray,
    matcher,
    r_px: float,
    use_color: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    r = int(round(float(r_px)))
    if use_color:
        Rw, _ = _warp_right_to_left_bgr_cv2(R_bgr, D0_np)
        delta16 = matcher.compute(L_bgr, Rw)
    else:
        Rw, _ = _warp_right_to_left_gray_cv2(R_gray, D0_np)
        delta16 = matcher.compute(L_gray, Rw)
    delta = (delta16.astype(np.float32) / 16.0)
    delta = np.clip(delta, -r, r).astype(np.float32)
    D_ref = (D0_np + delta).astype(np.float32)
    return D_ref, delta

# ============================================================
# Metrics 유틸 (MIN/MAX depth 적용)
# ============================================================
MIN_DEPTH_M = 1e-3  # evaluate.py의 MIN_DEPTH와 동일하게 사용

@torch.no_grad()
def compute_epe_d1_from_gt_full(
    pred_disp_full_px: torch.Tensor,
    gt_depth_full_m: torch.Tensor,
    focal_px: float,
    baseline_m: float,
    max_depth_m: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    depth_valid = gt_depth_full_m > MIN_DEPTH_M
    if max_depth_m is not None and max_depth_m > 0:
        depth_valid = depth_valid & (gt_depth_full_m < max_depth_m)

    valid = depth_valid.float()
    if valid.sum() <= 0:
        return None, None

    gt_disp_full_px = (float(focal_px) * float(baseline_m)) / gt_depth_full_m.clamp_min(1e-6)
    m = compute_ms2_disparity_metrics(pred_disp_full_px, gt_disp_full_px, valid)
    epe = float(m.get("EPE", float("nan")))
    d1 = float(m.get("D1_all", float("nan")))
    if not math.isfinite(epe):
        epe = None
    if not math.isfinite(d1):
        d1 = None
    return epe, d1

@torch.no_grad()
def compute_epe_d1_from_gt_scaled(
    pred_disp_scaled_px: torch.Tensor,
    gt_depth_scaled_m: torch.Tensor,
    focal_px: float,
    baseline_m: float,
    scale: float,
    max_depth_m: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    depth_valid = gt_depth_scaled_m > MIN_DEPTH_M
    if max_depth_m is not None and max_depth_m > 0:
        depth_valid = depth_valid & (gt_depth_scaled_m < max_depth_m)

    valid = depth_valid.float()
    if valid.sum() <= 0:
        return None, None

    fb = float(focal_px) * float(baseline_m)
    gt_disp_scaled_px = (fb / gt_depth_scaled_m.clamp_min(1e-6)) * float(scale)
    m = compute_ms2_disparity_metrics(pred_disp_scaled_px, gt_disp_scaled_px, valid)
    epe = float(m.get("EPE", float("nan")))
    d1 = float(m.get("D1_all", float("nan")))
    if not math.isfinite(epe):
        epe = None
    if not math.isfinite(d1):
        d1 = None
    return epe, d1

@torch.no_grad()
def compute_depth_metrics_from_disp(
    pred_disp_full_px: torch.Tensor,
    gt_depth_full_m: torch.Tensor,
    focal_px: float,
    baseline_m: float,
    max_depth_m: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    if gt_depth_full_m is None or pred_disp_full_px is None:
        return None

    fb = float(focal_px) * float(baseline_m)
    # disparity->depth 변환
    pred_depth = fb / pred_disp_full_px.clamp_min(1e-6)

    # pred_depth 클램핑 (MIN_DEPTH_M ~ max_depth_m)
    pred_depth = pred_depth.clamp_min(MIN_DEPTH_M)
    if max_depth_m is not None and max_depth_m > 0:
        pred_depth = pred_depth.clamp_max(max_depth_m)

    # GT도 동일 범위에서만 사용
    valid = (
        (gt_depth_full_m > MIN_DEPTH_M)
        & torch.isfinite(gt_depth_full_m)
        & torch.isfinite(pred_depth)
    )
    if max_depth_m is not None and max_depth_m > 0:
        valid = valid & (gt_depth_full_m < max_depth_m)

    if valid.sum() == 0:
        return None

    pd = pred_depth[valid].reshape(-1)
    gd = gt_depth_full_m[valid].reshape(-1)

    diff = pd - gd
    absrel = torch.mean(torch.abs(diff) / gd).item()
    sqrel = torch.mean((diff**2) / gd).item()
    rmse = torch.sqrt(torch.mean(diff**2)).item()

    eps = 1e-6
    rmselog = torch.sqrt(
        torch.mean((torch.log(pd + eps) - torch.log(gd + eps)) ** 2)
    ).item()

    ratio = torch.maximum(pd / gd, gd / pd)
    d1 = torch.mean((ratio < 1.25).float()).item()
    d2 = torch.mean((ratio < (1.25**2)).float()).item()
    d3 = torch.mean((ratio < (1.25**3)).float()).item()

    return {
        "AbsRel": absrel,
        "SqRel": sqrel,
        "RMSE": rmse,
        "LogRMSE": rmselog,
        "delta1": d1,
        "delta2": d2,
        "delta3": d3,
    }

@torch.no_grad()
def compute_depth_metrics_from_disp_scaled(
    pred_disp_scaled_px: torch.Tensor,
    gt_depth_scaled_m: torch.Tensor,
    focal_px: float,
    baseline_m: float,
    scale: float,
    max_depth_m: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    if gt_depth_scaled_m is None or pred_disp_scaled_px is None:
        return None

    fb = float(focal_px) * float(baseline_m)
    # disparity->depth 변환 (스케일 반영)
    pred_depth = (fb * float(scale)) / pred_disp_scaled_px.clamp_min(1e-6)

    # pred_depth 클램핑
    pred_depth = pred_depth.clamp_min(MIN_DEPTH_M)
    if max_depth_m is not None and max_depth_m > 0:
        pred_depth = pred_depth.clamp_max(max_depth_m)

    # GT도 동일 범위에서만 사용
    valid = (
        (gt_depth_scaled_m > MIN_DEPTH_M)
        & torch.isfinite(gt_depth_scaled_m)
        & torch.isfinite(pred_depth)
    )
    if max_depth_m is not None and max_depth_m > 0:
        valid = valid & (gt_depth_scaled_m < max_depth_m)

    if valid.sum() == 0:
        return None

    pd = pred_depth[valid].reshape(-1)
    gd = gt_depth_scaled_m[valid].reshape(-1)

    diff = pd - gd
    absrel = torch.mean(torch.abs(diff) / gd).item()
    sqrel = torch.mean((diff**2) / gd).item()
    rmse = torch.sqrt(torch.mean(diff**2)).item()

    eps = 1e-6
    rmselog = torch.sqrt(
        torch.mean((torch.log(pd + eps) - torch.log(gd + eps)) ** 2)
    ).item()

    ratio = torch.maximum(pd / gd, gd / pd)
    d1 = torch.mean((ratio < 1.25).float()).item()
    d2 = torch.mean((ratio < (1.25**2)).float()).item()
    d3 = torch.mean((ratio < (1.25**3)).float()).item()

    return {
        "AbsRel": absrel,
        "SqRel": sqrel,
        "RMSE": rmse,
        "LogRMSE": rmselog,
        "delta1": d1,
        "delta2": d2,
        "delta3": d3,
    }

# ============================================================
# 요약 파일 유틸
# ============================================================
def _fmt(v: Optional[float], spec: str) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "N/A"
    try:
        return format(float(v), spec)
    except Exception:
        return str(v)

def _delta_str(bef: Optional[float], aft: Optional[float], spec: str) -> str:
    if (
        (bef is None)
        or (aft is None)
        or any(
            [
                isinstance(x, float) and (math.isnan(x) or math.isinf(x))
                for x in [bef, aft]
            ]
        )
    ):
        return "N/A"
    return f"{_fmt(bef, spec)} → {_fmt(aft, spec)} (Δ{_fmt(aft - bef, spec)})"

def save_valid_mask_png(path: str, arr: np.ndarray):
    """
    arr: depth(m) 또는 disparity(px) 같은 2D float.
    >0 인 곳을 흰색(255), 나머지는 검정(0)으로 저장.
    """
    from PIL import Image
    _ensure_dir(os.path.dirname(path))
    mask = (np.isfinite(arr) & (arr > 0)).astype(np.uint8) * 255
    Image.fromarray(mask, mode="L").save(path)

def write_metrics_summary(
    out_root: str, records: List[Dict[str, Dict[str, float]]], note_quarter: bool = False
):
    if len(records) == 0:
        return
    path = os.path.join(out_root, "metrics_refine_summary.txt")
    keys = [
        "EPE_px",
        "D1_pct",
        "AbsRel",
        "SqRel",
        "RMSE_m",
        "LogRMSE",
        "delta1",
        "delta2",
        "delta3",
    ]

    def collect_mean(k: str, which: str) -> Optional[float]:
        vals = []
        for r in records:
            v = r[which].get(k)
            if v is None or (
                isinstance(v, float) and (math.isnan(v) or math.isinf(v))
            ):
                continue
            vals.append(float(v))
        if len(vals) == 0:
            return None
        return float(np.mean(vals))

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Residual-SGM refine metrics summary\n")
        f.write(
            "# Columns: EPE(px), D1(%), AbsRel, SqRel, RMSE(m), LogRMSE, δ<1.25, δ<1.25^2, δ<1.25^3\n"
        )
        if note_quarter:
            f.write(
                "# NOTE: --half enabled → disparity metrics are in 'px @ 1/2-res' (full-res px × 0.5).\n\n"
            )
        else:
            f.write("\n")

        for r in records:
            name = r.get("name", "unknown")
            bef, aft = r["before"], r["after"]
            line = (
                f"{name} | "
                f"EPE(px) {_delta_str(bef.get('EPE_px'), aft.get('EPE_px'), '.3f')} | "
                f"D1(%) {_delta_str(bef.get('D1_pct'), aft.get('D1_pct'), '.2f')} | "
                f"AbsRel {_delta_str(bef.get('AbsRel'), aft.get('AbsRel'), '.4f')} | "
                f"SqRel {_delta_str(bef.get('SqRel'), aft.get('SqRel'), '.4f')} | "
                f"RMSE(m) {_delta_str(bef.get('RMSE_m'), aft.get('RMSE_m'), '.3f')} | "
                f"LogRMSE {_delta_str(bef.get('LogRMSE'), aft.get('LogRMSE'), '.4f')} | "
                f"δ1 {_delta_str(bef.get('delta1'), aft.get('delta1'), '.4f')} | "
                f"δ2 {_delta_str(bef.get('delta2'), aft.get('delta2'), '.4f')} | "
                f"δ3 {_delta_str(bef.get('delta3'), aft.get('delta3'), '.4f')}\n"
            )
            f.write(line)

        f.write("\n# Averages over images (before → after, Δafter-before)\n")
        avg_bef = {k: collect_mean(k, "before") for k in keys}
        avg_aft = {k: collect_mean(k, "after") for k in keys}

        def avg_line(label, b, a, spec):
            return f"{label} {_delta_str(b, a, spec)}\n"

        f.write(avg_line("EPE(px)", avg_bef["EPE_px"], avg_aft["EPE_px"], ".3f"))
        f.write(avg_line("D1(%)", avg_bef["D1_pct"], avg_aft["D1_pct"], ".2f"))
        f.write(avg_line("AbsRel", avg_bef["AbsRel"], avg_aft["AbsRel"], ".4f"))
        f.write(avg_line("SqRel", avg_bef["SqRel"], avg_aft["SqRel"], ".4f"))
        f.write(avg_line("RMSE(m)", avg_bef["RMSE_m"], avg_aft["RMSE_m"], ".3f"))
        f.write(avg_line("LogRMSE", avg_bef["LogRMSE"], avg_aft["LogRMSE"], ".4f"))
        f.write(avg_line("δ1", avg_bef["delta1"], avg_aft["delta1"], ".4f"))
        f.write(avg_line("δ2", avg_bef["delta2"], avg_aft["delta2"], ".4f"))
        f.write(avg_line("δ3", avg_bef["delta3"], avg_aft["delta3"], ".4f"))

    print(f"[Summary] metrics written → {path}")

# ============================================================
# 메인 루프
# ============================================================
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # 출력 폴더
    out_root     = _ensure_dir(args.output_dir)
    out_delta    = _ensure_dir(os.path.join(out_root, "delta"))
    out_disp_ref = _ensure_dir(os.path.join(out_root, "disp_refined"))
    out_pth_bef  = _ensure_dir(os.path.join(out_root, "pth_before"))
    out_pth_aft  = _ensure_dir(os.path.join(out_root, "pth_after"))
    out_disp_err = _ensure_dir(os.path.join(out_root, "disp_error"))
    out_gt_disp  = _ensure_dir(os.path.join(out_root, "gt_disp"))   # GT disparity 저장 폴더

    # ★ 새로 추가: disparity overlay 저장 폴더
    out_disp0_overlay   = _ensure_dir(os.path.join(out_root, "disp_init_overlay"))
    out_disp_ref_overlay = _ensure_dir(os.path.join(out_root, "disp_refined_overlay"))

    dataset = StereoFolderDataset(args.left_dir, args.right_dir,
                                  height=args.height, width=args.width)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=2, pin_memory=True, drop_last=False)

    matcher, min_disp, num_disp = _build_sgbm_for_residual(
        r_px=args.residual_px,
        block_size=args.block_size,
        uniqueness=args.uniqueness,
        speckle_win=args.speckle_win,
        speckle_range=args.speckle_range,
        mode=args.mode,
        use_color=bool(args.color),
    )

    all_records: List[Dict[str, Dict[str, float]]] = []

    scale = 0.5 if args.half else 1.0
    unit_px = "px@1/2" if args.half else "px"
    neg_scale_note = args.half

    for it, (imgL, imgR, names) in enumerate(loader, start=1):
        name = names[0] if isinstance(names, (list, tuple)) else names
        stem = _basename_wo_ext(name)

        # 초기 disparity 로드
        disp_path = find_file_by_stem(args.disp_dir, stem, [".npy"])
        if disp_path is None:
            print(f"[Skip] init disp npy not found for stem={stem}")
            continue
        D0_full = np.load(disp_path).astype(np.float32)
        H_t, W_t = int(imgL.shape[-2]), int(imgL.shape[-1])
        H_run = int(round(H_t * scale))
        W_run = int(round(W_t * scale))

        imgL01_full = denorm_imagenet(imgL.to(device))
        imgR01_full = denorm_imagenet(imgR.to(device))
        if args.half:
            imgL01 = F.interpolate(imgL01_full, size=(H_run, W_run),
                                   mode="bilinear", align_corners=False)
            imgR01 = F.interpolate(imgR01_full, size=(H_run, W_run),
                                   mode="bilinear", align_corners=False)
        else:
            imgL01, imgR01 = imgL01_full, imgR01_full

        L_rgb = (imgL01[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        R_rgb = (imgR01[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        L_bgr = cv2.cvtColor(L_rgb, cv2.COLOR_RGB2BGR)
        R_bgr = cv2.cvtColor(R_rgb, cv2.COLOR_RGB2BGR)
        L_gray = cv2.cvtColor(L_bgr, cv2.COLOR_BGR2GRAY)
        R_gray = cv2.cvtColor(R_bgr, cv2.COLOR_BGR2GRAY)

        # disparity 해상도 스케일
        if D0_full.shape != (H_run, W_run):
            scale_x = W_run / D0_full.shape[1]
            D0_np = cv2.resize(D0_full, (W_run, H_run), interpolation=cv2.INTER_LINEAR) * scale_x
        else:
            scale_x = W_run / D0_full.shape[1]
            D0_np = D0_full * scale_x

        D0_t = torch.from_numpy(D0_np).to(device=device,
                                          dtype=torch.float32).view(1, 1, H_run, W_run)

        # [DEBUG] 예측 disparity coverage (첫 이미지만)
        if it == 1:
            valid_D0 = np.isfinite(D0_np) & (D0_np > 0)
            print(f"[DEBUG] D0>0    : {valid_D0.sum()} / {D0_np.size}")

        # ★ 초기 disparity overlay 저장
        save_disp_overlay_on_image(
            os.path.join(out_disp0_overlay, f"{stem}_disp_init_overlay.png"),
            L_bgr,
            D0_np,
            vmin=0.0,
            vmax=args.vmax_disp,
            cmap_name=args.disp_cmap,
        )

        # photometric before
        pth_before, _ = compute_pth_error_map(
            imgL01, imgR01, D0_t, w_l1=args.pth_l1_w, w_ssim=args.pth_ssim_w
        )
        pth_bef_np = pth_before[0, 0].detach().cpu().numpy()
        save_colormap_png_with_colorbar_auto_range(
            os.path.join(out_pth_bef, f"{stem}_pth_before.png"),
            pth_bef_np,
            vmin=0.0,
            vmax=None,
            cmap_name=args.err_cmap,
            label="Photometric error",
            bg_color=args.bg_color,
        )

        # multi_stage (coarse→fine, 옵션)
        if args.multi_stage:
            s_c = 0.5
            Hc = int(max(8, round(H_run * s_c)))
            Wc = int(max(8, round(W_run * s_c)))
            Lc_bgr = cv2.resize(L_bgr, (Wc, Hc), interpolation=cv2.INTER_AREA)
            Rc_bgr = cv2.resize(R_bgr, (Wc, Hc), interpolation=cv2.INTER_AREA)
            Lc_gray = cv2.cvtColor(Lc_bgr, cv2.COLOR_BGR2GRAY)
            Rc_gray = cv2.cvtColor(Rc_bgr, cv2.COLOR_BGR2GRAY)
            D0_c = cv2.resize(D0_np, (Wc, Hc), interpolation=cv2.INTER_LINEAR) * s_c
            D0_c_ref, _ = residual_sgm_pass(
                Lc_bgr,
                Rc_bgr,
                Lc_gray,
                Rc_gray,
                D0_c,
                matcher,
                r_px=args.residual_px,
                use_color=bool(args.color),
            )
            D0_np = cv2.resize(D0_c_ref, (W_run, H_run), interpolation=cv2.INTER_LINEAR) / s_c

        # 메인 residual-SGM 패스
        D_ref_raw, delta_raw = residual_sgm_pass(
            L_bgr,
            R_bgr,
            L_gray,
            R_gray,
            D0_np,
            matcher,
            r_px=args.residual_px,
            use_color=bool(args.color),
        )

        # WLS
        if args.wls:
            D_ref_smooth = _apply_wls_generic_if_available(
                D_ref_raw, L_bgr, args.wls_lambda, args.wls_sigma
            )
        else:
            D_ref_smooth = D_ref_raw.copy()

        if args.clip_nonneg:
            D_ref_smooth = np.clip(D_ref_smooth, 0, None)

        D_for_post = D_ref_smooth.copy()

        # LR-consistency
        if args.lr_consistency:
            D_right = _build_right_from_left(D_for_post)
            lr_mask = _lr_consistency_mask(D_for_post, D_right, args.lr_tau)
            D_for_post[~lr_mask] = 0.0
            D_for_post = _edge_aware_hole_fill(D_for_post, L_bgr, max_iter=3)

        # weighted median
        if args.wmf:
            D_for_post = _weighted_median_if_available(
                D_for_post, L_bgr, radius=args.wmf_radius, sigma_color=args.wmf_sigma_color
            )

        # plane refine
        if args.plane_refine:
            D_for_post = _plane_refine_superpixel_if_available(D_for_post, L_bgr)

        D_ref = D_for_post.astype(np.float32)
        delta_final = (D_ref - D0_np).astype(np.float32)

        # [DEBUG] refine 후 coverage
        if it == 1:
            valid_Dref = np.isfinite(D_ref) & (D_ref > 0)
            print(f"[DEBUG] D_ref>0 : {valid_Dref.sum()} / {D_ref.size}")

        # ★ refined disparity overlay 저장
        save_disp_overlay_on_image(
            os.path.join(out_disp_ref_overlay, f"{stem}_disp_refined_overlay.png"),
            L_bgr,
            D_ref,
            vmin=0.0,
            vmax=args.vmax_disp,
            cmap_name=args.disp_cmap,
        )

        # Δ / D_ref 저장
        r = int(round(float(args.residual_px)))
        save_npy(os.path.join(out_delta, f"{stem}_delta.npy"), delta_final)
        save_colormap_png_with_colorbar_auto_range(
            os.path.join(out_delta, f"{stem}_delta_cb.png"),
            delta_final,
            vmin=-r,
            vmax=+r,
            cmap_name="RdBu_r",
            label=f"Δ disparity ({unit_px})",
            bg_color=args.bg_color,
        )

        save_npy(os.path.join(out_disp_ref, f"{stem}.npy"), D_ref)
        disp_png = os.path.join(out_disp_ref, f"{stem}_disp_refined_cb.png")
        save_colormap_png_with_colorbar_auto_range(
            disp_png,
            D_ref,
            vmin=0.0,
            vmax=args.vmax_disp,
            cmap_name=args.disp_cmap,
            label=f"Disparity ({unit_px})",
            bg_color=args.bg_color,
        )

        # photometric after
        D_ref_t = torch.from_numpy(D_ref).to(device=device,
                                             dtype=torch.float32).view(1, 1, H_run, W_run)
        pth_after, _ = compute_pth_error_map(
            imgL01, imgR01, D_ref_t, w_l1=args.pth_l1_w, w_ssim=args.pth_ssim_w
        )
        pth_after_np = pth_after[0, 0].detach().cpu().numpy()
        pth_png_after = os.path.join(out_pth_aft, f"{stem}_pth_after.png")
        save_colormap_png_with_colorbar_auto_range(
            pth_png_after,
            pth_after_np,
            vmin=0.0,
            vmax=None,
            cmap_name=args.err_cmap,
            label="Photometric error",
            bg_color=args.bg_color,
        )

        # GT 메트릭 + disparity error + GT disparity 저장
        record = {"name": stem, "before": {}, "after": {}}
        if args.gt_depth_dir and args.focal_px > 0 and args.baseline_m > 0:
            has_gt = False
            gt_depth = None
            try:
                gt_depth = load_ms2_gt_depth_batch(
                    names=[name],
                    gt_depth_dir=args.gt_depth_dir,
                    scale=args.gt_depth_scale,
                    target_hw=(H_run, W_run),
                    device=device,
                )
                if gt_depth is not None:
                    valid_gt = torch.isfinite(gt_depth) & (gt_depth > 0)
                    has_gt = bool(valid_gt.sum().item() > 0)
            except Exception:
                has_gt = False

            if has_gt:
                # max_depth (m) 설정
                max_depth_m = args.max_depth if getattr(args, "max_depth", -1.0) > 0 else None

                # DEBUG: gt coverage
                if it == 1:
                    depth_np = gt_depth[0, 0].detach().cpu().numpy()
                    valid_depth = np.isfinite(depth_np) & (depth_np > 0)
                    print(f"[DEBUG] gt_depth>0 (after load+resize): {valid_depth.sum()} / {depth_np.size}")

                    raw_gt_path = find_file_by_stem(args.gt_depth_dir, stem, [".npy", ".png"])
                    if raw_gt_path is not None:
                        try:
                            if raw_gt_path.lower().endswith(".npy"):
                                d_full = np.load(raw_gt_path).astype(np.float32)
                            else:
                                img_raw = cv2.imread(raw_gt_path, -1)
                                if img_raw is None:
                                    raise RuntimeError("cv2.imread failed")
                                d_full = img_raw.astype(np.float32)
                                if args.gt_depth_scale > 0:
                                    d_full /= float(args.gt_depth_scale)
                            v_full = np.isfinite(d_full) & (d_full > 0)
                            print(f"[DEBUG] gt_depth>0 (full-res raw png/npy): {v_full.sum()} / {v_full.size}")
                        except Exception as e:
                            print(f"[DEBUG] full-res GT load failed: {e}")

                metrics_use_scaled = bool(args.reshape or args.half)
                scale_for_metrics = float(scale) if metrics_use_scaled else 1.0

                if metrics_use_scaled:
                    epe_bef, d1_bef = compute_epe_d1_from_gt_scaled(
                        D0_t, gt_depth, args.focal_px, args.baseline_m,
                        scale_for_metrics, max_depth_m=max_depth_m
                    )
                    epe_aft, d1_aft = compute_epe_d1_from_gt_scaled(
                        D_ref_t, gt_depth, args.focal_px, args.baseline_m,
                        scale_for_metrics, max_depth_m=max_depth_m
                    )
                    depth_bef = compute_depth_metrics_from_disp_scaled(
                        D0_t, gt_depth, args.focal_px, args.baseline_m,
                        scale_for_metrics, max_depth_m=max_depth_m
                    )
                    depth_aft = compute_depth_metrics_from_disp_scaled(
                        D_ref_t, gt_depth, args.focal_px, args.baseline_m,
                        scale_for_metrics, max_depth_m=max_depth_m
                    )
                else:
                    epe_bef, d1_bef = compute_epe_d1_from_gt_full(
                        D0_t, gt_depth, args.focal_px, args.baseline_m,
                        max_depth_m=max_depth_m
                    )
                    epe_aft, d1_aft = compute_epe_d1_from_gt_full(
                        D_ref_t, gt_depth, args.focal_px, args.baseline_m,
                        max_depth_m=max_depth_m
                    )
                    depth_bef = compute_depth_metrics_from_disp(
                        D0_t, gt_depth, args.focal_px, args.baseline_m,
                        max_depth_m=max_depth_m
                    )
                    depth_aft = compute_depth_metrics_from_disp(
                        D_ref_t, gt_depth, args.focal_px, args.baseline_m,
                        max_depth_m=max_depth_m
                    )

                record["before"].update(
                    {
                        "EPE_px": epe_bef,
                        "D1_pct": d1_bef,
                        "AbsRel": None if depth_bef is None else depth_bef["AbsRel"],
                        "SqRel": None if depth_bef is None else depth_bef["SqRel"],
                        "RMSE_m": None if depth_bef is None else depth_bef["RMSE"],
                        "LogRMSE": None if depth_bef is None else depth_bef["LogRMSE"],
                        "delta1": None if depth_bef is None else depth_bef["delta1"],
                        "delta2": None if depth_bef is None else depth_bef["delta2"],
                        "delta3": None if depth_bef is None else depth_bef["delta3"],
                    }
                )
                record["after"].update(
                    {
                        "EPE_px": epe_aft,
                        "D1_pct": d1_aft,
                        "AbsRel": None if depth_aft is None else depth_aft["AbsRel"],
                        "SqRel": None if depth_aft is None else depth_aft["SqRel"],
                        "RMSE_m": None if depth_aft is None else depth_aft["RMSE"],
                        "LogRMSE": None if depth_aft is None else depth_aft["LogRMSE"],
                        "delta1": None if depth_aft is None else depth_aft["delta1"],
                        "delta2": None if depth_aft is None else depth_aft["delta2"],
                        "delta3": None if depth_aft is None else depth_aft["delta3"],
                    }
                )
                all_records.append(record)

                # ---- GT disparity (px) 계산 & 저장 ----
                fb = float(args.focal_px) * float(args.baseline_m)
                if metrics_use_scaled:
                    gt_disp = (fb / gt_depth.clamp_min(1e-6)) * scale_for_metrics
                else:
                    gt_disp = fb / gt_depth.clamp_min(1e-6)

                gt_disp_np  = gt_disp[0,0].detach().cpu().numpy()
                depth_np    = gt_depth[0,0].detach().cpu().numpy()

                valid_mask = np.isfinite(depth_np) & (depth_np > 0)

                gt_disp_np_save = np.zeros_like(gt_disp_np, dtype=np.float32)
                gt_disp_np_save[valid_mask] = gt_disp_np[valid_mask]
                save_npy(os.path.join(out_gt_disp, f"{stem}_gt_disp.npy"), gt_disp_np_save)
                save_valid_mask_png(
                    os.path.join(out_gt_disp, f"{stem}_gt_valid_mask.png"),
                    gt_disp_np_save
                )

                gt_disp_np_vis = np.full_like(gt_disp_np, np.nan, dtype=np.float32)
                gt_disp_np_vis[valid_mask] = gt_disp_np[valid_mask]
                save_colormap_png_with_colorbar_auto_range(
                    os.path.join(out_gt_disp, f"{stem}_gt_disp.png"),
                    gt_disp_np_vis, vmin=0.0, vmax=args.vmax_disp,
                    cmap_name=args.disp_cmap,
                    label=f"GT disparity ({unit_px})",
                    bg_color=args.bg_color,
                )

                if it == 1:
                    print(f"[DEBUG] gt_disp>0 (saved npy): {np.count_nonzero(gt_disp_np_save > 0)} / {gt_disp_np_save.size}")

                # ---- disparity error (before/after) 저장 (depth 범위 적용) ----
                pred0 = D0_t
                pred1 = D_ref_t

                err0 = torch.full_like(gt_disp, float("nan"))
                err1 = torch.full_like(gt_disp, float("nan"))

                valid0 = (
                    (gt_depth > MIN_DEPTH_M)
                    & torch.isfinite(gt_depth)
                    & torch.isfinite(pred0)
                    & (pred0 > 0)
                )
                valid1 = (
                    (gt_depth > MIN_DEPTH_M)
                    & torch.isfinite(gt_depth)
                    & torch.isfinite(pred1)
                    & (pred1 > 0)
                )
                if max_depth_m is not None and max_depth_m > 0:
                    valid0 = valid0 & (gt_depth < max_depth_m)
                    valid1 = valid1 & (gt_depth < max_depth_m)

                if it == 1:
                    print(f"[DEBUG] disp_err valid0 (before): {valid0.sum().item()} / {valid0.numel()}")
                    print(f"[DEBUG] disp_err valid1 (after) : {valid1.sum().item()} / {valid1.numel()}")

                err0[valid0] = torch.abs(pred0[valid0] - gt_disp[valid0])
                err1[valid1] = torch.abs(pred1[valid1] - gt_disp[valid1])

                err0_np = err0[0, 0].detach().cpu().numpy()
                err1_np = err1[0, 0].detach().cpu().numpy()

                save_npy(os.path.join(out_disp_err, f"{stem}_disp_err_before.npy"), err0_np)
                save_npy(os.path.join(out_disp_err, f"{stem}_disp_err_after.npy"), err1_np)

                save_colormap_png_with_colorbar_auto_range(
                    os.path.join(out_disp_err, f"{stem}_disp_err_before.png"),
                    err0_np,
                    vmin=0.0,
                    vmax=3.0,
                    cmap_name=args.err_cmap,
                    label="Disp error (px)",
                    bg_color=args.bg_color,
                )
                save_colormap_png_with_colorbar_auto_range(
                    os.path.join(out_disp_err, f"{stem}_disp_err_after.png"),
                    err1_np,
                    vmin=0.0,
                    vmax=3.0,
                    cmap_name=args.err_cmap,
                    label="Disp error (px)",
                    bg_color=args.bg_color,
                )
            else:
                print(
                    f"[Info] GT depth not found or empty for '{stem}', skipping metrics for this image."
                )
        else:
            annotate_png_top_left(
                disp_png, f"Metrics: N/A (no GT/focal/baseline)  |  unit={unit_px}"
            )
            annotate_png_top_left(
                pth_png_after, f"Metrics: N/A (no GT/focal/baseline)  |  unit={unit_px}"
            )

        print(
            f"[{it:04d}] residual-SGM refined & saved: {stem} "
            f"(Δ in [-{r}, +{r}])  |  mode={'COLOR' if args.color else 'GRAY'}  "
            f"|  scale={'1/2' if args.half else '1/1'}"
        )

    write_metrics_summary(out_root, all_records, note_quarter=bool(neg_scale_note))
    print(f"[Done] outputs → {out_root}")

# ============================================================
# argparse
# ============================================================
def get_args():
    p = argparse.ArgumentParser(
        "Residual-SGM refine: warp right by D0 → SGBM on ±r → D = D0+Δ (WLS/WMF/LR-check optional)"
    )

    p.add_argument("--left_dir", type=str, required=True)
    p.add_argument("--right_dir", type=str, required=True)
    p.add_argument(
        "--disp_dir",
        type=str,
        required=True,
        help="*.npy (full-res px) initial disparity per image stem",
    )

    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=640)

    p.add_argument("--output_dir", type=str, default="./log/rsgm_out_robotcar")

    p.add_argument(
        "--residual_px",
        type=float,
        default=1.0,
        help="잔차 탐색 반경 r (px) → minDisp=-r, numDisp≈2r(16의 배수)",
    )
    p.add_argument("--block_size", type=int, default=5, help="SGBM block size (odd 3..11)")
    p.add_argument(
        "--uniqueness", type=int, default=10, help="uniqueness ratio (5~15)"
    )
    p.add_argument(
        "--speckle_win", type=int, default=200, help="speckle window size"
    )
    p.add_argument(
        "--speckle_range", type=int, default=2, help="speckle range"
    )
    p.add_argument(
        "--mode", type=str, default="HH", choices=["HH", "3WAY", "SGBM"], help="SGBM mode"
    )
    p.add_argument(
        "--color", action="store_true", help="컬러(BGR, 3채널)로 SGBM 수행"
    )

    p.add_argument(
        "--wls",
        action="store_true",
        help="WLS(Generic) 정제 사용 (opencv-contrib 필요)",
    )
    p.add_argument("--wls_lambda", type=float, default=8000.0)
    p.add_argument("--wls_sigma", type=float, default=1.5)
    p.add_argument(
        "--clip_nonneg", action="store_true", help="최종 disparity를 [0,∞)로 클립"
    )

    p.add_argument(
        "--lr_consistency",
        action="store_true",
        help="Left-Right consistency check + hole fill 사용",
    )
    p.add_argument(
        "--lr_tau",
        type=float,
        default=1.0,
        help="LR-consistency 허용 범위 |dL-dR|<=tau",
    )

    p.add_argument(
        "--wmf", action="store_true", help="ximgproc weightedMedianFilter 후처리 사용"
    )
    p.add_argument(
        "--wmf_radius", type=int, default=5, help="weighted median radius (픽셀)"
    )
    p.add_argument(
        "--wmf_sigma_color",
        type=float,
        default=25.0,
        help="weighted median sigmaColor",
    )

    p.add_argument(
        "--plane_refine",
        action="store_true",
        help="슈퍼픽셀 평면 기반 hole fill 사용 (ximgproc SLIC 필요)",
    )

    p.add_argument(
        "--multi_stage",
        action="store_true",
        help="coarse→fine 잔차 정련(1/2 scale 선행 후 full refine)",
    )

    p.add_argument("--pth_l1_w", type=float, default=0.15)
    p.add_argument("--pth_ssim_w", type=float, default=0.85)

    p.add_argument(
        "--gt_depth_dir",
        type=str,
        default=None,
        help="GT depth root (파일명 기준 매칭)",
    )
    p.add_argument("--gt_depth_scale", type=float, default=1000.0)

    # ROBOTCAR_SCALE_FACTOR ≈ baseline(0.239) * focal(100) 을 맞추기 위해 기본값 100.0
    p.add_argument("--focal_px", type=float, default=983.0446)
    p.add_argument("--baseline_m", type=float, default=0.239)

    # ★ max_depth: 이 값(m) 이내만 metric 계산
    p.add_argument(
        "--max_depth",
        type=float,
        default=50.0,
        help="metric 계산에 사용할 최대 깊이 (meter). "
             "≤0 이면 depth 제한 없이 전체 픽셀 사용.",
    )

    p.add_argument(
        "--vmax_disp",
        type=float,
        default=26.0,
        help="disparity 시각화 상한(px)",
    )
    p.add_argument("--disp_cmap", type=str, default="magma")
    p.add_argument("--err_cmap", type=str, default="magma")
    p.add_argument("--bg_color", type=str, default="#1e1e1e")

    p.add_argument(
        "--half",
        action="store_true",
        help="전체 과정을 1/2 해상도/단위(px@1/2)로 수행",
    )
    p.add_argument(
        "--reshape",
        action="store_true",
        help=(
            "정량 지표 계산 시 실행 스케일을 반영해 disparity→depth 변환을 수행하고 "
            "GT depth도 실행 해상도로 리쉐이프하여 평가"
        ),
    )

    p.add_argument("--cpu", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    run(args)
