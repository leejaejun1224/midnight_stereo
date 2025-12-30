# -*- coding: utf-8 -*-
import os
import glob
import math
import argparse
from typing import Optional, Tuple, List

import numpy as np
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

# -------------------------------
# 공용 저장/시각화 유틸
# -------------------------------
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

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
    path, np_array, vmin: Optional[float] = None, vmax: Optional[float] = None,
    cmap_name: str = "magma", label: str = "", bg_color: str = "#1e1e1e"
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
    cmap = ListedColormap(base(_np.linspace(0,1,256)))
    cmap.set_bad(to_rgba(bg_color))

    H, W = arr.shape; dpi = 200.0; figsize = (W/dpi, H/dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(bg_color); ax.set_facecolor(bg_color)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin_eff, vmax=vmax_eff); ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if label: cbar.set_label(label, rotation=270, labelpad=12)
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
        try: font = ImageFont.truetype("arial.ttf", fs)
        except Exception: font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    try:
        bbox = draw.textbbox((0,0), text, font=font, stroke_width=2)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)
    bg = Image.new("RGBA", (tw+2*margin, th+2*margin), (0,0,0,100))
    img.paste(bg, (margin, margin), bg)
    draw.text((margin*2, margin*2), text, font=font,
              fill=(255,255,255,255), stroke_width=2, stroke_fill=(0,0,0,255))
    img = img.convert("RGB"); img.save(path)

# -------------------------------
# 그래디언트/로컬 집계 유틸
# -------------------------------
def sobel_xy(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ img: [B,1 or 3,H,W] in [0,1] """
    kx = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=img.dtype, device=img.device).view(1,1,3,3)/8.0
    ky = torch.tensor([[1, 2, 1],
                       [0, 0, 0],
                       [-1,-2,-1]], dtype=img.dtype, device=img.device).view(1,1,3,3)/8.0
    if img.shape[1] == 3:
        gray = 0.2989*img[:,0:1] + 0.5870*img[:,1:2] + 0.1140*img[:,2:3]
    else:
        gray = img
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return gx, gy

def avg_pool_rect(x: torch.Tensor, kh: int, kw: int) -> torch.Tensor:
    if kh <= 1 and kw <= 1: return x
    pad = (kw//2, kw//2, kh//2, kh//2)  # left,right,top,bottom
    return F.avg_pool2d(F.pad(x, pad, mode="replicate"), kernel_size=(kh,kw), stride=1)

# ====== 새로 추가: 임의 모양(가중) 윈도우 집계 ======
def _normalize_kernel(k: torch.Tensor) -> torch.Tensor:
    s = k.sum()
    if float(s) <= 1e-12:
        return k
    return k / s

def _gaussian_kernel_2d(kh: int, kw: int, sy: float, sx: float, device, dtype) -> torch.Tensor:
    cy, cx = (kh-1)/2.0, (kw-1)/2.0
    ys = torch.arange(kh, device=device, dtype=dtype).unsqueeze(1).repeat(1, kw)
    xs = torch.arange(kw, device=device, dtype=dtype).unsqueeze(0).repeat(kh, 1)
    ky = torch.exp(-0.5*((ys-cy)/max(sy,1e-6))**2)
    kx = torch.exp(-0.5*((xs-cx)/max(sx,1e-6))**2)
    k = ky * kx
    return _normalize_kernel(k)

def _cross_kernel(kh: int, kw: int, thick: int, device, dtype) -> torch.Tensor:
    thick = max(1, int(thick))
    if thick % 2 == 0: thick += 1  # 홀수 권장
    k = torch.zeros((kh, kw), device=device, dtype=dtype)
    cy, cx = kh//2, kw//2
    k[cy - thick//2: cy + thick//2 + 1, :] = 1.0
    k[:, cx - thick//2: cx + thick//2 + 1] = 1.0
    return _normalize_kernel(k)

def _parse_custom_kernel(spec: str, device, dtype) -> torch.Tensor:
    # 파일 경로(.npy)면 로드
    if os.path.isfile(spec):
        arr = np.load(spec).astype(np.float32)
        k = torch.from_numpy(arr).to(device=device, dtype=dtype)
        return _normalize_kernel(k)

    # 문자열 패턴: "001;111;001" 혹은 "0 0 1; 1 1 1; 0 0 1"
    rows = [r.strip() for r in spec.strip().split(';') if len(r.strip()) > 0]
    parsed = []
    for r in rows:
        if ',' in r or ' ' in r:
            toks = [t for t in r.replace(',', ' ').split(' ') if t != '']
            parsed.append([float(t) for t in toks])
        else:
            parsed.append([float(c) for c in list(r)])
    k_np = np.array(parsed, dtype=np.float32)
    k = torch.from_numpy(k_np).to(device=device, dtype=dtype)
    return _normalize_kernel(k)

def build_agg_kernel(
    shape: str, kh: int, kw: int,
    device, dtype,
    sigma_y: float = 3.0, sigma_x: float = 3.0,
    custom: Optional[str] = None,
    cross_thick: int = 1
) -> torch.Tensor:
    """ shape: rect|vert|hori|cross|gauss|custom """
    if shape == "vert":
        kw = max(1, kw) if kw is not None else 1
        kh = max(1, kh)
        k = torch.ones((kh, kw), device=device, dtype=dtype)
        return _normalize_kernel(k)
    if shape == "hori":
        kh = max(1, kh) if kh is not None else 1
        kw = max(1, kw)
        k = torch.ones((kh, kw), device=device, dtype=dtype)
        return _normalize_kernel(k)
    if shape == "rect":
        kh = max(1, kh); kw = max(1, kw)
        k = torch.ones((kh, kw), device=device, dtype=dtype)
        return _normalize_kernel(k)
    if shape == "gauss":
        kh = max(1, kh); kw = max(1, kw)
        return _gaussian_kernel_2d(kh, kw, sigma_y, sigma_x, device, dtype)
    if shape == "cross":
        kh = max(1, kh); kw = max(1, kw)
        return _cross_kernel(kh, kw, cross_thick, device, dtype)
    if shape == "custom":
        if custom is None or len(custom.strip()) == 0:
            raise ValueError("agg_shape=custom 인데 --agg_custom 이 비었습니다.")
        return _parse_custom_kernel(custom, device, dtype)
    raise ValueError(f"알 수 없는 agg_shape: {shape}")

def masked_avg2d_with_kernel(x: torch.Tensor, k2d: torch.Tensor) -> torch.Tensor:
    """
    x: [B,C,H,W], k2d: [kh,kw] (sum=1 권장)
    depthwise conv로 가중 평균. 가장자리 보정(denominator) 포함.
    """
    B, C, H, W = x.shape
    kh, kw = int(k2d.shape[0]), int(k2d.shape[1])
    pad = (kw//2, kw//2, kh//2, kh//2)
    k = k2d.view(1,1,kh,kw).repeat(C,1,1,1).to(device=x.device, dtype=x.dtype)
    num = F.conv2d(F.pad(x, pad, mode="replicate"), k, groups=C)
    den = F.conv2d(F.pad(torch.ones_like(x), pad, mode="replicate"), k, groups=C)
    return num / (den + 1e-8)

# -------------------------------
# Photometric cost (기존 간단 버전)
# -------------------------------
@torch.no_grad()
def robust_photo_cost(
    imgL_01: torch.Tensor, imgR_warp_01: torch.Tensor, valid: torch.Tensor,
    w_l1: float=0.15, w_ssim: float=0.85, w_grad: float=0.5,
    # 새로: 임의 모양 윈도우
    agg_kernel: Optional[torch.Tensor] = None,
    fallback_k: int = 5
) -> torch.Tensor:
    """
    반환: [B,1,H,W] cost (invalid은 +inf)
    """
    photo = PhotometricLoss([w_l1, w_ssim]).simple_photometric_loss(
        imgL_01, imgR_warp_01, weights=[w_l1, w_ssim]
    )  # [B,1,H,W]

    gxL, gyL = sobel_xy(imgL_01); gxR, gyR = sobel_xy(imgR_warp_01)
    grad = (gxL - gxR).abs() + (gyL - gyR).abs()         # [B,1,H,W]

    if agg_kernel is not None:
        cost = masked_avg2d_with_kernel(photo, agg_kernel) \
             + w_grad * masked_avg2d_with_kernel(grad, agg_kernel)
    else:
        cost = avg_pool_rect(photo, fallback_k, fallback_k) \
             + w_grad * avg_pool_rect(grad, fallback_k, fallback_k)

    cost = torch.where(valid > 0.5, cost, torch.full_like(cost, float("inf")))
    return cost

# -------------------------------
# 보조 유틸
# -------------------------------
def _to_gray01(img):  # [B,1 or 3,H,W] -> [B,1,H,W] in [0,1]
    if img.shape[1] == 3:
        gray = 0.2989*img[:,0:1] + 0.5870*img[:,1:2] + 0.1140*img[:,2:3]
    else:
        gray = img
    return gray.clamp(0,1)

def _soft_census(gray, k=7, T=0.03):
    """
    Soft Center-Symmetric Census (CS-Census) 근사. (사각 패치 전제)
    gray: [B,1,H,W] in [0,1]
    반환: bits [B, M, H, W], M=(k*k-1)//2 (센터-대칭 비교쌍)
    """
    B, _, H, W = gray.shape
    unfold = F.unfold(gray, kernel_size=k, padding=k//2)  # [B, k*k, HW]
    kk = k*k
    pairs = []
    for u in range(k):
        for v in range(k):
            i = u*k + v
            if i == kk//2:  # center skip
                continue
            u2, v2 = (k-1-u), (k-1-v)
            j = u2*k + v2
            if i < j and i != kk//2 and j != kk//2:
                pairs.append((i, j))
    if len(pairs) == 0:
        raise ValueError("census pairs empty")
    xi = torch.stack([unfold[:, i, :] for (i, j) in pairs], dim=1)  # [B, M, HW]
    xj = torch.stack([unfold[:, j, :] for (i, j) in pairs], dim=1)  # [B, M, HW]
    bits = torch.sigmoid((xi - xj) / T)  # [B, M, HW]
    bits = bits.view(B, len(pairs), H, W)
    return bits  # 0..1 (0/1 근사)

# -------------------------------
# 강건 매칭 코스트 (윈도우 모양 지정 지원)
# -------------------------------
@torch.no_grad()
def strong_match_cost(
    imgL_01: torch.Tensor, imgR_warp_01: torch.Tensor, valid: torch.Tensor,
    *,
    # Photometric (L1+SSIM) 가중치
    w_photo: float = 0.2, w_ssim: float = 0.8,
    # Gradient/Orientation
    w_grad: float = 0.5, w_gori: float = 0.2,
    # ZNCC
    w_zncc: float = 0.5, zncc_h: int = 7, zncc_w: int = 7, zncc_eps: float = 1e-3,
    # Census (사각 패치 유지)
    w_census: float = 0.5, census_k: int = 7, census_T: float = 0.03,
    # SSIM 풀링 창
    ssim_h: int = 11, ssim_w: int = 11,
    # 최종 집계용 임의 모양 윈도우
    agg_kernel: Optional[torch.Tensor] = None,
    fallback_k: int = 9
) -> torch.Tensor:
    """
    반환: cost [B,1,H,W], invalid → +inf
    """
    B, C, H, W = imgL_01.shape
    L = imgL_01.clamp(0,1); Rw = imgR_warp_01.clamp(0,1)
    grayL = _to_gray01(L); grayRw = _to_gray01(Rw)

    # 1) Photometric (L1 + SSIM)  -> 0..1
    l1 = (L - Rw).abs().mean(1, keepdim=True)  # [B,1,H,W]

    muL = avg_pool_rect(L, ssim_h, ssim_w)
    muR = avg_pool_rect(Rw, ssim_h, ssim_w)
    sigmaL = avg_pool_rect(L*L, ssim_h, ssim_w) - muL*muL
    sigmaR = avg_pool_rect(Rw*Rw, ssim_h, ssim_w) - muR*muR
    sigmaLR= avg_pool_rect(L*Rw, ssim_h, ssim_w) - muL*muR
    C1, C2 = 0.01**2, 0.03**2
    ssim = ((2*muL*muR + C1)*(2*sigmaLR + C2)) / ((muL*muL + muR*muR + C1)*(sigmaL + sigmaR + C2) + 1e-6)
    ssim_cost = (1 - ssim.mean(1, keepdim=True)) * 0.5  # 0(best)~1

    photo = (w_photo * l1 + w_ssim * ssim_cost)

    # 2) Gradient constancy + orientation
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=L.dtype, device=L.device).view(1,1,3,3)/8.0
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=L.dtype, device=L.device).view(1,1,3,3)/8.0
    gLx = F.conv2d(grayL, kx, padding=1); gLy = F.conv2d(grayL, ky, padding=1)
    gRx = F.conv2d(grayRw, kx, padding=1); gRy = F.conv2d(grayRw, ky, padding=1)
    grad_cost = (gLx-gRx).abs() + (gLy-gRy).abs()
    nL = torch.sqrt(gLx*gLx + gLy*gLy + 1e-6)
    nR = torch.sqrt(gRx*gRx + gRy*gRy + 1e-6)
    cos = (gLx*gRx + gLy*gRy) / (nL*nR + 1e-6)
    gori_cost = (1 - cos).clamp(0,2) * 0.5  # 0..1

    # 3) ZNCC  -> 0(best)..1  (직사각 지원)
    muLz = avg_pool_rect(L, zncc_h, zncc_w)
    muRz = avg_pool_rect(Rw,zncc_h, zncc_w)
    stdL = torch.sqrt((avg_pool_rect(L*L, zncc_h, zncc_w) - muLz*muLz).clamp_min(zncc_eps))
    stdR = torch.sqrt((avg_pool_rect(Rw*Rw,zncc_h, zncc_w) - muRz*muRz).clamp_min(zncc_eps))
    corr = (avg_pool_rect(L*Rw, zncc_h, zncc_w) - muLz*muRz) / (stdL*stdR + zncc_eps)
    zncc_cost = (1 - corr.mean(1, keepdim=True)) * 0.5  # 0..1

    # 4) Soft CS‑Census Hamming  -> 0..1 (사각 패치 유지)
    bitsL  = _soft_census(grayL, k=census_k, T=census_T)       # [B,M,H,W]
    bitsRw = _soft_census(grayRw, k=census_k, T=census_T)
    census_cost = (bitsL - bitsRw).abs().mean(1, keepdim=True) # [B,1,H,W]

    # 5) 최종 집계(임의 모양)
    mix = photo + w_grad*grad_cost + w_gori*gori_cost + w_zncc*zncc_cost + w_census*census_cost
    if agg_kernel is not None:
        agg = masked_avg2d_with_kernel(mix, agg_kernel)
    else:
        agg = avg_pool_rect(mix, fallback_k, fallback_k)

    # invalid → +inf
    cost = torch.where(valid>0.5, agg, torch.full_like(agg, float("inf")))
    return cost  # [B,1,H,W]

# -------------------------------
# Photometric error 계산 (시각화/마스킹용: L1+SSIM)
# -------------------------------
@torch.no_grad()
def compute_pth_error_map(imgL_01: torch.Tensor, imgR_01: torch.Tensor, disp_full_px: torch.Tensor,
                          w_l1=0.15, w_ssim=0.85) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    반환:
      pth_map: [1,1,H,W]  (invalid은 NaN)
      valid:   [1,1,H,W]  (0/1)
    """
    imgR_warp, valid = warp_right_to_left_image(imgR_01, disp_full_px)  # [1,3,H,W], [1,1,H,W]
    pth = PhotometricLoss([w_l1, w_ssim]).simple_photometric_loss(imgL_01, imgR_warp, weights=[w_l1, w_ssim])
    pth = torch.where(valid>0.5, pth, torch.full_like(pth, float("nan")))
    return pth, valid

# -------------------------------
# 국소 재탐색 (마스크 영역만) — Δdisp(잔차) 반환
# -------------------------------
@torch.no_grad()
def local_rescan_refine_delta(
    imgL_01: torch.Tensor, imgR_01: torch.Tensor, disp_full_px: torch.Tensor, mask: torch.Tensor,
    delta_px: float=2.0, step_px: float=0.5,
    # 코스트 구성 파라미터
    w_l1: float=0.15, w_ssim: float=0.85, w_grad: float=0.5,
    w_photo: float = 0.2, w_photo_ssim: float = 0.8,
    w_gori: float = 0.2, w_zncc: float = 0.5, zncc_h: int = 7, zncc_w: int = 7,
    w_census: float = 0.5, census_k: int = 7, census_T: float = 0.03,
    ssim_h: int = 11, ssim_w: int = 11,
    # 집계 윈도우 (임의 모양)
    agg_kernel: Optional[torch.Tensor] = None,
    fallback_k: int = 9,
    # 수락 규칙
    accept_lam: float=0.02, accept_eps: float=1e-4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    출력:
      delta_best:   [1,1,H,W]  ← 픽셀별 최적 Δdisp (잔차)
      best_cost:    [1,1,H,W]  ← 강건 photometric 코스트
      accepted_any: [1,1,H,W]  ← 한 번이라도 갱신된 픽셀
    """
    device = disp_full_px.device
    B, _, H, W = disp_full_px.shape
    assert B == 1, "batch=1 기준 구현"

    # 초기 코스트(off=0)
    Rw0, valid0 = warp_right_to_left_image(imgR_01, disp_full_px)
    best_cost = strong_match_cost(
        imgL_01, Rw0, valid0,
        w_photo=w_photo, w_ssim=w_photo_ssim,
        w_grad=w_grad, w_gori=w_gori,
        w_zncc=w_zncc, zncc_h=zncc_h, zncc_w=zncc_w,
        w_census=w_census, census_k=census_k, census_T=census_T,
        ssim_h=ssim_h, ssim_w=ssim_w,
        agg_kernel=agg_kernel, fallback_k=fallback_k
    )
    delta_best = torch.zeros_like(disp_full_px)  # 시작 Δ=0
    accepted_any = torch.zeros_like(disp_full_px, dtype=torch.bool)

    offsets = torch.arange(-delta_px, delta_px + 1e-9, step_px, device=device, dtype=disp_full_px.dtype)
    for off in offsets:
        if abs(float(off)) < 1e-12:  # 0은 이미 평가
            continue
        cand = disp_full_px + off
        Rw, valid = warp_right_to_left_image(imgR_01, cand)
        cand_cost = strong_match_cost(
            imgL_01, Rw, valid,
            w_photo=w_photo, w_ssim=w_photo_ssim,
            w_grad=w_grad, w_gori=w_gori,
            w_zncc=w_zncc, zncc_h=zncc_h, zncc_w=zncc_w,
            w_census=w_census, census_k=census_k, census_T=census_T,
            ssim_h=ssim_h, ssim_w=ssim_w,
            agg_kernel=agg_kernel, fallback_k=fallback_k
        )
        # 수락 규칙: (cost + λΔd^2) < best_cost - ε
        accept = (cand_cost + accept_lam * (off**2)) < (best_cost - accept_eps)
        accept = accept & (mask.bool())

        delta_best = torch.where(accept, torch.as_tensor(off, device=device, dtype=delta_best.dtype), delta_best)
        best_cost  = torch.where(accept, cand_cost, best_cost)
        accepted_any = accepted_any | accept

    return delta_best, best_cost, accepted_any

# -------------------------------
# EPE/D1 계산 (full-res px 기준)
# -------------------------------
@torch.no_grad()
def compute_epe_d1_from_gt_full(
    pred_disp_full_px: torch.Tensor,  # [1,1,H,W]
    gt_depth_full_m: torch.Tensor,    # [1,1,H,W]
    focal_px: float, baseline_m: float
) -> Tuple[Optional[float], Optional[float]]:
    valid = (gt_depth_full_m > 0).float()
    if valid.sum() <= 0: return None, None
    gt_disp_full_px = (float(focal_px) * float(baseline_m)) / gt_depth_full_m.clamp_min(1e-6)
    m = compute_ms2_disparity_metrics(pred_disp_full_px, gt_disp_full_px, valid)
    epe = float(m.get("EPE", float("nan"))); d1 = float(m.get("D1_all", float("nan")))
    if not math.isfinite(epe): epe = None
    if not math.isfinite(d1): d1 = None
    return epe, d1

# -------------------------------
# 파일 매칭 유틸
# -------------------------------
def find_file_by_stem(folder: str, stem: str, exts: List[str]) -> Optional[str]:
    # 정확 이름 우선
    for ext in exts:
        cand = os.path.join(folder, f"{stem}{ext}")
        if os.path.isfile(cand): return cand
    # fallback: stem*ext
    for ext in exts:
        ms = glob.glob(os.path.join(folder, f"{stem}*{ext}"))
        if len(ms) > 0: return ms[0]
    return None

# -------------------------------
# 메인 루프
# -------------------------------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # 출력 폴더
    out_root     = _ensure_dir(args.output_dir)
    out_mask     = _ensure_dir(os.path.join(out_root, "mask"))
    out_ref_disp = _ensure_dir(os.path.join(out_root, "disp_refined"))
    out_delta    = _ensure_dir(os.path.join(out_root, "delta"))
    out_acc      = _ensure_dir(os.path.join(out_root, "accepted"))
    out_pth_bef  = _ensure_dir(os.path.join(out_root, "photometric_error_before"))
    out_pth_aft  = _ensure_dir(os.path.join(out_root, "photometric_error_after"))

    # 데이터 로더(이미지 전처리/리사이즈를 인퍼런스와 동일하게)
    dataset = StereoFolderDataset(args.left_dir, args.right_dir, height=args.height, width=args.width)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    # === 집계 커널 생성 (한 번 만들어 재사용) ===
    agg_kernel = build_agg_kernel(
        shape=args.agg_shape,
        kh=args.agg_h, kw=args.agg_w,
        device=device, dtype=torch.float32,
        sigma_y=args.agg_sigma_y, sigma_x=args.agg_sigma_x,
        custom=args.agg_custom,
        cross_thick=args.agg_cross_thick
    )

    thr = float(args.pth_thr)

    for it, (imgL, imgR, names) in enumerate(loader, start=1):
        name = names[0] if isinstance(names, (list, tuple)) else names
        stem = _basename_wo_ext(name)

        # --- 0) 대응 파일 로드 ---
        disp_path = find_file_by_stem(args.disp_dir, stem, [".npy"])
        if disp_path is None:
            print(f"[Skip] disp npy not found for stem={stem}")
            continue

        pth_path = None
        if args.pth_dir:
            # 보통 {stem}_pth_error.npy
            pp = find_file_by_stem(args.pth_dir, f"{stem}_pth_error", [".npy"])
            if pp is None:
                pp = find_file_by_stem(args.pth_dir, stem, [".npy"])
            pth_path = pp

        # --- 1) 이미지 [0,1] 복원 (+옵션: 야간 보정) ---
        imgL = imgL.to(device, non_blocking=True)
        imgR = imgR.to(device, non_blocking=True)
        imgL01 = denorm_imagenet(imgL)
        imgR01 = denorm_imagenet(imgR)

        if args.lowlight:
            gamma = float(args.gamma)
            imgL01 = imgL01.clamp(0,1).pow(gamma)
            imgR01 = imgR01.clamp(0,1).pow(gamma)

            # LCN (depthwise, per_channel=True)
            def lcn(x: torch.Tensor, k: int = 9, sigma: float = 3.0, per_channel: bool = True) -> torch.Tensor:
                B, C, H, W = x.shape
                rad = int(k // 2)
                xs = torch.arange(-rad, rad + 1, device=x.device, dtype=x.dtype)
                g1d = torch.exp(-0.5 * (xs / sigma) ** 2)
                g1d = (g1d / (g1d.sum() + 1e-8)).view(1, 1, 1, k)  # [1,1,1,k] (horizontal)
                if per_channel:
                    g_h = g1d.repeat(C, 1, 1, 1)                    # [C,1,1,k]
                    g_v = g1d.transpose(2, 3).repeat(C, 1, 1, 1)    # [C,1,k,1]
                    x_blur = F.conv2d(F.conv2d(x, g_h, padding=(0, rad), groups=C),
                                      g_v, padding=(rad, 0), groups=C)
                else:
                    if C == 3:
                        gray = (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]).clamp(0, 1)
                    else:
                        gray = x
                    blur_gray = F.conv2d(F.conv2d(gray, g1d, padding=(0, rad)),
                                         g1d.transpose(2, 3), padding=(rad, 0))
                    x_blur = blur_gray.repeat(1, C, 1, 1)
                return x - x_blur

            imgL01 = (imgL01 + lcn(imgL01, k=9, sigma=3.0, per_channel=True)).clamp(0, 1)
            imgR01 = (imgR01 + lcn(imgR01, k=9, sigma=3.0, per_channel=True)).clamp(0, 1)

        # --- 2) disparity/photometric error 로드 ---
        disp_np = np.load(disp_path).astype(np.float32)
        p = np.nanpercentile(disp_np, [0, 50, 95, 99])
        print(f"[CHK] {stem} disparity percentiles (px assumed): "
              f"min={p[0]:.3f}, med={p[1]:.3f}, p95={p[2]:.3f}, p99={p[3]:.3f}")
        disp_full = torch.from_numpy(disp_np).float().to(device).view(1,1,*imgL01.shape[-2:])
        if pth_path and os.path.isfile(pth_path):
            pth_before = torch.from_numpy(np.load(pth_path)).float().to(device).view(1,1,*imgL01.shape[-2:])
        else:
            pth_before, _ = compute_pth_error_map(imgL01, imgR01, disp_full, w_l1=args.w_photo_l1, w_ssim=args.w_photo_ssim)

        # photometric before 저장(선택)
        pth_bef_np = pth_before[0,0].detach().cpu().numpy()
        save_colormap_png_with_colorbar_auto_range(
            os.path.join(out_pth_bef, f"{stem}_pth_error_before.png"),
            pth_bef_np, vmin=0.0, vmax=None, cmap_name=args.err_cmap,
            label="Photometric error", bg_color=args.bg_color
        )
        if args.save_pth_npy:
            save_npy(os.path.join(out_pth_bef, f"{stem}_pth_error_before.npy"), pth_bef_np)

        # --- 3) photometric mask 생성/저장 ---
        pth_np = pth_before.detach().cpu().numpy()[0,0]
        mask_np = np.zeros_like(pth_np, dtype=np.uint8)
        finite = np.isfinite(pth_np)
        mask_np[(finite) & (pth_np >= args.pth_thr)] = 255
        save_gray_png(os.path.join(out_mask, f"{stem}_pth{args.pth_thr:.2f}_mask.png"), mask_np)
        mask = torch.from_numpy((mask_np > 0).astype(np.uint8)).to(device=device, dtype=torch.bool).view(1,1,*mask_np.shape)

        # --- 4) 국소 재탐색(Δ) & 최종 합성(절대) ---
        delta_map, best_cost, accepted_mask = local_rescan_refine_delta(
            imgL01, imgR01, disp_full, mask,
            delta_px=args.delta_px, step_px=args.step_px,
            w_l1=args.w_photo_l1, w_ssim=args.w_photo_ssim, w_grad=args.w_photo_grad,
            w_photo=args.w_photo_l1, w_photo_ssim=args.w_photo_ssim,
            w_gori=args.w_gori, w_zncc=args.w_zncc, zncc_h=args.zncc_h, zncc_w=args.zncc_w,
            w_census=args.w_census, census_k=args.census_k, census_T=args.census_T,
            ssim_h=args.ssim_h, ssim_w=args.ssim_w,
            agg_kernel=agg_kernel, fallback_k=args.agg_fallback_k,
            accept_lam=args.accept_lam, accept_eps=args.accept_eps
        )

        disp_final = disp_full + delta_map
        if args.max_disp_px is not None:
            disp_final = disp_final.clamp(0.0, float(args.max_disp_px))

        # --- 5) 저장 (Δ/최종) ---
        # Δ
        delta_np = delta_map[0,0].detach().cpu().numpy()
        save_npy(os.path.join(out_delta, f"{stem}_delta.npy"), delta_np)
        save_colormap_png_with_colorbar_auto_range(
            os.path.join(out_delta, f"{stem}_delta_cb.png"),
            delta_np, vmin=None, vmax=None, cmap_name="RdBu_r",
            label="Δ disparity (px)", bg_color=args.bg_color
        )
        # 수락 픽셀 시각화
        save_gray_png(os.path.join(out_acc, f"{stem}_accepted.png"),
                      (accepted_mask[0,0].detach().cpu().numpy().astype(np.uint8) * 255))

        # 최종 disparity (절대)
        disp_final_np = disp_final[0,0].detach().cpu().numpy()
        save_npy(os.path.join(out_ref_disp, f"{stem}.npy"), disp_final_np)
        disp_png = os.path.join(out_ref_disp, f"{stem}_disp_refined_cb.png")
        save_colormap_png_with_colorbar_auto_range(
            disp_png, disp_final_np, vmin=0.0, vmax=args.vmax_disp,
            cmap_name=args.disp_cmap, label="Disparity (px)", bg_color=args.bg_color
        )

        # --- 6) Photometric error after 저장 ---
        pth_after, valid_after = compute_pth_error_map(imgL01, imgR01, disp_final, w_l1=args.w_photo_l1, w_ssim=args.w_photo_ssim)
        pth_after_np = pth_after[0,0].detach().cpu().numpy()
        pth_png_after = os.path.join(out_pth_aft, f"{stem}_pth_error_after.png")
        save_colormap_png_with_colorbar_auto_range(
            pth_png_after, pth_after_np, vmin=0.0, vmax=None,
            cmap_name=args.err_cmap, label="Photometric error", bg_color=args.bg_color
        )
        if args.save_pth_npy:
            save_npy(os.path.join(out_pth_aft, f"{stem}_pth_error_after.npy"), pth_after_np)

        # --- 7) (선택) GT 메트릭: EPE/D1 오버레이 ---
        if args.gt_depth_dir and args.focal_px > 0 and args.baseline_m > 0:
            gt_depth = load_ms2_gt_depth_batch(
                names=[name], gt_depth_dir=args.gt_depth_dir,
                scale=args.gt_depth_scale, target_hw=disp_final.shape[-2:], device=device
            )
            epe, d1 = compute_epe_d1_from_gt_full(disp_final, gt_depth, args.focal_px, args.baseline_m)
            if (epe is not None) and (d1 is not None):
                annotate_png_top_left(disp_png,     f"EPE {epe:.3f} px | D1 {d1:.2f}%")
                annotate_png_top_left(pth_png_after,f"EPE {epe:.3f} px | D1 {d1:.2f}%")
            else:
                annotate_png_top_left(disp_png, "EPE/D1 : N/A")
                annotate_png_top_left(pth_png_after, "EPE/D1 : N/A")
        else:
            annotate_png_top_left(disp_png, "EPE/D1 : N/A (no GT)")
            annotate_png_top_left(pth_png_after, "EPE/D1 : N/A (no GT)")

        if it % max(1, args.log_every) == 0:
            finite_delta = delta_np[np.isfinite(delta_np)]
            if finite_delta.size > 0:
                dr = (finite_delta.min(), finite_delta.max())
                print(f"[{it:04d}] refined & saved: {stem}  (Δ range: {dr[0]:.3f} ~ {dr[1]:.3f})")
            else:
                print(f"[{it:04d}] refined & saved: {stem}  (Δ range: N/A)")

    print(f"[Done] outputs → {out_root}")

# argparse
# -------------------------------
def get_args():
    p = argparse.ArgumentParser("External photometric refine (mask≥thr → local rescan ±Δd → EPE/D1 overlay)")

    # 입력 경로
    p.add_argument("--left_dir",  type=str, default="/home/jaejun/dataset/MS2/sync_data/tester/rgb/img_left")
    p.add_argument("--right_dir", type=str, default="/home/jaejun/dataset/MS2/sync_data/tester/rgb/img_right")
    p.add_argument("--disp_dir",  type=str, required=True, help="*.npy (full-res px) predicted disparity")
    p.add_argument("--pth_dir",   type=str, default=None, help="*.npy photometric error (optional; recompute if absent)")

    # 데이터 크기(인퍼런스와 동일하게 맞추기)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width",  type=int, default=1224)
    p.add_argument("--max_disp_px",  type=int, default=56)

    # 출력
    p.add_argument("--output_dir", type=str, default="./log/refine_out")
    p.add_argument("--log_every",  type=int, default=1)

    # 마스킹
    p.add_argument("--pth_thr", type=float, default=0.1)

    # 재탐색 하이퍼
    p.add_argument("--delta_px", type=float, default=8.0)
    p.add_argument("--step_px",  type=float, default=0.25)

    # === 집계 윈도우 (모양 지정) ===
    p.add_argument("--agg_shape", type=str, default="gauss",
                   choices=["rect","vert","hori","cross","gauss","custom"],
                   help="집계(aggregation) 윈도우 모양")
    p.add_argument("--agg_h", type=int, default=25, help="집계 창 높이(odd 권장)")
    p.add_argument("--agg_w", type=int, default=5,  help="집계 창 너비(odd 권장); vert면 1~3 권장")
    p.add_argument("--agg_sigma_y", type=float, default=5.0, help="gauss: 세로 sigma")
    p.add_argument("--agg_sigma_x", type=float, default=1.5, help="gauss: 가로 sigma")
    p.add_argument("--agg_cross_thick", type=int, default=1, help="cross 모양 두께(홀수)")
    p.add_argument("--agg_custom", type=str, default=None,
                   help='custom일 때 마스크: ".npy" 경로 or 문자열 "001;111;001" 등')
    p.add_argument("--agg_fallback_k", type=int, default=9, help="agg_kernel 미사용시 박스 크기")

    # photometric/gradient 가중치
    p.add_argument("--w_photo_l1",   type=float, default=0.2)
    p.add_argument("--w_photo_ssim", type=float, default=0.8)
    p.add_argument("--w_photo_grad", type=float, default=1.0)
    p.add_argument("--w_gori", type=float, default=1.0)
    p.add_argument("--w_zncc", type=float, default=1.0)
    p.add_argument("--w_census", type=float, default=1.0)
    p.add_argument("--census_k", type=int, default=7)
    p.add_argument("--census_T", type=float, default=0.03)

    # ZNCC/SSIM 윈도우 (직사각형 지원)
    p.add_argument("--zncc_h", type=int, default=25)
    p.add_argument("--zncc_w", type=int, default=5)
    p.add_argument("--ssim_h", type=int, default=25)
    p.add_argument("--ssim_w", type=int, default=5)

    # 수락 규칙
    p.add_argument("--accept_lam", type=float, default=0.02)
    p.add_argument("--accept_eps", type=float, default=1e-4)

    # 야간/저조 대응(선택)
    p.add_argument("--lowlight", action="store_true")
    p.add_argument("--gamma", type=float, default=0.6, help="lowlight 시 감마 보정(작을수록 밝힘)")

    # (선택) GT 메트릭
    p.add_argument("--gt_depth_dir",  type=str, default="/home/jaejun/dataset/MS2/proj_depth/tester/rgb/depth_filtered")
    p.add_argument("--gt_depth_scale", type=float, default=256.0)
    p.add_argument("--focal_px", type=float, default=764.5138549804688)
    p.add_argument("--baseline_m", type=float, default=0.29918420530585865)

    # 시각화
    p.add_argument("--vmax_disp", type=float, default=None)
    p.add_argument("--disp_cmap", type=str, default="magma")
    p.add_argument("--err_cmap",  type=str, default="magma")
    p.add_argument("--bg_color",  type=str, default="#1e1e1e")
    p.add_argument("--save_pth_npy", action="store_true")

    # 기타
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    run(args)
