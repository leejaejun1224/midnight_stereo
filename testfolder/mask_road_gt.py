# -*- coding: utf-8 -*-
import os
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

# 백엔드 없이 저장만
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# roi_entropy_fill_and_viz.py
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import math
import os

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import cm

# ===========================================
# 0) DINO 로드 & 전처리
# ===========================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

def load_dino(device: torch.device):
    """
    facebookresearch/dino 의 ViT-B/8(dino_vitb8) 로드
    """
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    model.eval().to(device)
    return model

def pil_to_tensor(img_pil: Image.Image) -> torch.Tensor:
    """
    PIL → Tensor [1,3,H,W] + ImageNet 정규화
    """
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    x = tfm(img_pil).unsqueeze(0)  # [1,3,H,W]
    return x


# ===========================================
# 1) ViT-B/8 패치 토큰 추출 + 1/4 격자 생성(학습 없음)
#    (0/4 px 시프트-패딩 4회 → 인터리빙)
# ===========================================
@torch.no_grad()
def _extract_patch_tokens_dino(model, x: torch.Tensor) -> torch.Tensor:
    """
    입력 x: [1,3,Hpad,Wpad] (이미 패딩 완료본)
    반환: [H8, W8, C]  (CLS 토큰 제거)
    """
    out = model.get_intermediate_layers(x, n=1)[0]  # [1, N(+cls), C]
    B, N, C = out.shape
    Hpad, Wpad = x.shape[-2:]
    h8 = Hpad // 8
    w8 = Wpad // 8

    if N == h8 * w8 + 1:
        tokens = out[:, 1:, :]
    elif N == h8 * w8:
        tokens = out
    else:
        raise RuntimeError(f"Unexpected token count: N={N}, expected {h8*w8} or {h8*w8+1}")

    tokens_hw = tokens.reshape(B, h8, w8, C).squeeze(0).contiguous()  # [H8,W8,C]
    return tokens_hw

@torch.no_grad()
def _shift_once(model, img_tensor: torch.Tensor, dx: int, dy: int, pad_mode='replicate') -> torch.Tensor:
    """
    (dx,dy) ∈ {0,4} 시프트-패딩 입력에서 패치 토큰을 추출.
    반환: [H//8, W//8, C] (원본 영역 기준으로 잘라냄)
    """
    _, _, H, W = img_tensor.shape
    right  = (8 - (W + dx) % 8) % 8
    bottom = (8 - (H + dy) % 8) % 8

    # F.pad: (left, right, top, bottom)
    x_pad = F.pad(img_tensor, (dx, right, dy, bottom), mode=pad_mode)
    tokens = _extract_patch_tokens_dino(model, x_pad)  # [H8',W8',C]

    H8 = H // 8
    W8 = W // 8
    tokens = tokens[:H8, :W8, :]
    return tokens

@torch.no_grad()
def build_quarter_features(model, img_tensor: torch.Tensor) -> torch.Tensor:
    """
    최종 1/4 해상도 [H//4, W//4, C] 피처맵 생성 (L2 정규화 포함).
    """
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device, non_blocking=True)

    _, _, H, W = img_tensor.shape
    assert H % 8 == 0 and W % 8 == 0, "입력 H,W는 8의 배수여야 합니다. (--pad_to_8 권장)"

    H4, W4 = H // 4, W // 4

    f00 = _shift_once(model, img_tensor, dx=0, dy=0)  # [H8,W8,C]
    f40 = _shift_once(model, img_tensor, dx=4, dy=0)
    f04 = _shift_once(model, img_tensor, dx=0, dy=4)
    f44 = _shift_once(model, img_tensor, dx=4, dy=4)

    C = f00.shape[-1]
    Fq = torch.empty((H4, W4, C), device=device, dtype=f00.dtype)

    # 인터리빙
    Fq[1::2, 1::2, :] = f00
    Fq[1::2, 0::2, :] = f40
    Fq[0::2, 1::2, :] = f04
    Fq[0::2, 0::2, :] = f44

    Fq = F.normalize(Fq, dim=-1)  # 코사인 유사도용 L2 정규화
    return Fq  # [H4,W4,C]


# ===========================================
# 2) Cost Volume 구축 (좌→우, 수평 시차만)
# ===========================================
@torch.no_grad()
def build_cost_volume(featL: torch.Tensor, featR: torch.Tensor, max_disp: int) -> torch.Tensor:
    """
    featL, featR: [H4, W4, C] (L2 정규화됨)
    반환: cost_vol [D+1, H4, W4]
    """
    assert featL.shape == featR.shape
    H4, W4, C = featL.shape
    device = featL.device
    D = int(max_disp)

    cost_vol = torch.full((D + 1, H4, W4), float('-inf'), device=device, dtype=featL.dtype)

    for d in range(D + 1):
        if d == 0:
            sim = (featL * featR).sum(dim=-1)  # [H4,W4]
            cost_vol[0] = sim
        else:
            left_slice  = featL[:, d:, :]      # [H4, W4-d, C]
            right_slice = featR[:, :-d, :]     # [H4, W4-d, C]
            sim = (left_slice * right_slice).sum(dim=-1)  # [H4, W4-d]
            cost_vol[d, :, d:] = sim  # invalid(0..d-1)은 -inf 유지

    return cost_vol  # [D+1,H4,W4]

@torch.no_grad()
def argmax_disparity(cost_vol: torch.Tensor):
    """
    반환:
      - disp_map: [H4,W4] (long)
      - peak_sim: [H4,W4] (float)
    """
    peak_sim, disp_map = cost_vol.max(dim=0)
    return disp_map, peak_sim


# ===========================================
# 2-1) Entropy / Gap
# ===========================================
@torch.no_grad()
def build_entropy_map(cost_vol: torch.Tensor,
                      T: float = 0.1,
                      eps: float = 1e-8,
                      normalize: bool = True) -> torch.Tensor:
    m = torch.amax(cost_vol, dim=0, keepdim=True)
    logits = (cost_vol - m) / max(T, eps)

    prob = torch.softmax(logits, dim=0)
    p = prob.clamp_min(eps)
    ent = -(p * p.log()).sum(dim=0)  # [H4,W4]

    if normalize:
        valid = torch.isfinite(cost_vol)
        Deff  = valid.sum(dim=0).clamp_min(1).to(p.dtype)
        ent = torch.where(Deff > 1, ent / (Deff.log() + eps), torch.zeros_like(ent))
        ent = ent.clamp_(0.0, 1.0)
    return ent

@torch.no_grad()
def build_top2_gap_map(cost_vol: torch.Tensor) -> torch.Tensor:
    Dp1 = cost_vol.shape[0]
    if Dp1 < 2:
        H4, W4 = cost_vol.shape[1:]
        return torch.full((H4, W4), float('nan'), device=cost_vol.device, dtype=cost_vol.dtype)

    valid = torch.isfinite(cost_vol)
    Deff  = valid.sum(dim=0)

    _, idxs = torch.topk(cost_vol, k=2, dim=0)   # [2,H4,W4]
    d1 = idxs[0].to(torch.float32)
    d2 = idxs[1].to(torch.float32)
    gap = (d1 - d2).abs()
    gap = torch.where(Deff >= 2, gap, torch.full_like(gap, float('nan')))
    return gap


# ===========================================
# NEW) ROI ∩ (entropy > thr)만 adaptive window 재매칭
# ===========================================
@torch.no_grad()
def _invalid_run_extents(entropy: torch.Tensor, thr: float):
    """
    entropy: [H4,W4]
    반환: a,b, invalid
      - invalid = (entropy > thr)
      - (y,x)에서 좌/우로 '연속 invalid'만 지나 '첫 valid(<=thr)' 직전까지 확장
      - 경계에 valid 없으면 영상 경계까지
    """
    H4, W4 = entropy.shape
    device = entropy.device
    valid = (entropy <= float(thr))
    invalid = ~valid
    x = torch.arange(W4, device=device).view(1, W4).expand(H4, -1)

    # 왼쪽 마지막 valid (없으면 -1)
    left_valid_idx = torch.where(valid, x, torch.full_like(x, -1))
    prev_valid = torch.cummax(left_valid_idx, dim=1)[0]

    # 오른쪽 첫 valid (없으면 W4)
    valid_rev = torch.flip(valid, dims=[1])
    idx_rev = torch.where(valid_rev, x, torch.full_like(x, -1))
    prev_rev = torch.cummax(idx_rev, dim=1)[0]
    prev_rev = torch.flip(prev_rev, dims=[1])
    next_valid = (W4 - 1) - prev_rev
    next_valid = torch.where(prev_rev >= 0, next_valid, torch.full_like(next_valid, W4))

    # 연속 invalid만 포함
    L = (x - prev_valid - 1).clamp_min(0)
    R = (next_valid - x - 1).clamp_min(0)
    a = (x - L).clamp_min(0).to(torch.long)
    b = (x + R).clamp_max(W4 - 1).to(torch.long)
    return a, b, invalid

@torch.no_grad()
def build_roi_mask(H4: int, W4: int,
                   mode: str,
                   u0: float, u1: float,
                   v0: float, v1: float,
                   device: torch.device) -> torch.Tensor:
    """
    ROI 마스크 생성 (1/4 격자 기준)
    - mode='frac': u0,u1,v0,v1 ∈ [0,1], 비율(좌상 원점)
    - mode='abs4': u0,u1,v0,v1 는 1/4 격자 인덱스(정수 권장, inclusive)
    반환: [H4,W4] bool
    """
    mode = str(mode).lower()
    if mode not in ("frac", "abs4"):
        raise ValueError("roi_mode must be 'frac' or 'abs4'.")

    def clamp(v, lo, hi): return max(lo, min(hi, v))

    if mode == "frac":
        u0f = clamp(float(u0), 0.0, 1.0)
        u1f = clamp(float(u1), 0.0, 1.0)
        v0f = clamp(float(v0), 0.0, 1.0)
        v1f = clamp(float(v1), 0.0, 1.0)
        if u1f < u0f: u0f, u1f = u1f, u0f
        if v1f < v0f: v0f, v1f = v1f, v0f

        u0i = int(math.floor(u0f * W4))
        u1i = int(math.ceil (u1f * W4) - 1)
        v0i = int(math.floor(v0f * H4))
        v1i = int(math.ceil (v1f * H4) - 1)
    else:
        u0i = int(round(u0)); u1i = int(round(u1))
        v0i = int(round(v0)); v1i = int(round(v1))
        if u1i < u0i: u0i, u1i = u1i, u0i
        if v1i < v0i: v0i, v1i = v1i, v0i

    u0i = clamp(u0i, 0, W4 - 1); u1i = clamp(u1i, 0, W4 - 1)
    v0i = clamp(v0i, 0, H4 - 1); v1i = clamp(v1i, 0, H4 - 1)

    mask = torch.zeros(H4, W4, dtype=torch.bool, device=device)
    if (u1i >= u0i) and (v1i >= v0i):
        mask[v0i:v1i+1, u0i:u1i+1] = True
    return mask

@torch.no_grad()
def refine_cost_for_uncertain_roi(cost_vol: torch.Tensor,
                                  entropy_before: torch.Tensor,
                                  ent_thr: float,
                                  roi_mask: torch.Tensor,
                                  max_half: int = None,
                                  ent_T: float = 0.1):
    """
    cost_vol: [D+1,H4,W4]
    entropy_before: [H4,W4]
    roi_mask: [H4,W4] bool (1/4 격자)
    ent_thr: 시각화/유효 판단 임계
    return:
      cost_vol_ref:  [D+1,H4,W4]  (ROI∩invalid만 갱신)
      entropy_after: [H4,W4]      (보정 후 전체 엔트로피)
      refine_mask:   [H4,W4]      (ROI∩invalid)
    """
    Dp1, H4, W4 = cost_vol.shape
    device = cost_vol.device

    # invalid-run 윈도우
    a, b, invalid = _invalid_run_extents(entropy_before, ent_thr)   # [H4,W4]
    refine_mask = (roi_mask & invalid)                              # ROI ∩ invalid

    if max_half is not None:
        x = torch.arange(W4, device=device).view(1, W4).expand(H4, -1)
        a = torch.max(a, (x - max_half).clamp_min(0))
        b = torch.min(b, (x + max_half).clamp_max(W4 - 1))

    # 가로 prefix-sum (d-평면별)
    finite = torch.isfinite(cost_vol)
    cv = torch.where(finite, cost_vol, torch.zeros_like(cost_vol))   # -inf → 0
    pref = torch.zeros(Dp1, H4, W4 + 1, device=device, dtype=cv.dtype)
    pref[:, :, 1:] = torch.cumsum(cv, dim=2)
    cnt  = torch.zeros(Dp1, H4, W4 + 1, device=device, dtype=cv.dtype)
    cnt[:, :, 1:] = torch.cumsum(finite.to(cv.dtype), dim=2)

    a_idx = a.unsqueeze(0).expand(Dp1, -1, -1)
    b_idx = (b + 1).unsqueeze(0).expand(Dp1, -1, -1)
    num = pref.gather(2, b_idx) - pref.gather(2, a_idx)
    den = cnt.gather(2, b_idx) - cnt.gather(2, a_idx)
    agg = num / den.clamp_min(1.0)
    agg = torch.where(den > 0, agg, torch.full_like(agg, float('-inf')))

    # ROI∩invalid 위치만 치환
    cost_vol_ref = torch.where(refine_mask.unsqueeze(0), agg, cost_vol)

    # 보정 후 엔트로피(전체) — 시각화 마스크 합집합 계산에 사용
    entropy_after = build_entropy_map(cost_vol_ref, T=ent_T, normalize=True)
    return cost_vol_ref, entropy_after, refine_mask

@torch.no_grad()
def build_union_viz_mask(ent_before: torch.Tensor,
                         ent_after: torch.Tensor,
                         thr: float,
                         roi_mask: torch.Tensor):
    """
    ent_before/ent_after: [H4,W4] (torch)
    roi_mask: [H4,W4] (bool)  — ROI 위치에서만 after 반영
    return: mask_union [H4,W4] (bool)
    """
    m0 = (ent_before <= float(thr))
    m1 = (ent_after  <= float(thr)) & roi_mask
    return (m0 | m1)


# ===========================================
# 3) 시각화 (개별 패널 저장용 + 기존 4-up)
# ===========================================
def upsample_nearest_4x(map_2d: np.ndarray) -> np.ndarray:
    """
    [H4,W4] → [H4*4, W4*4] 최근접 업샘플(시각화용)
    """
    return np.kron(map_2d, np.ones((4, 4), dtype=map_2d.dtype))

def _get_transparent_disp_cmap():
    # turbo가 없거나 with_extremes 미지원 대비
    try:
        cmap = cm.get_cmap('turbo')
    except Exception:
        cmap = cm.get_cmap('plasma')
    if hasattr(cmap, 'with_extremes'):
        cmap = cmap.with_extremes(bad=(0, 0, 0, 0))
    else:
        try:
            cmap = cmap.copy()
        except Exception:
            pass
        try:
            cmap.set_bad((0, 0, 0, 0))
        except Exception:
            pass
    return cmap

def visualize_results(left_img_pil: Image.Image,
                      disp_map: np.ndarray,
                      peak_sim: np.ndarray,
                      max_disp: int,
                      save_path: Path = None,
                      entropy_map: np.ndarray = None,
                      ent_vis_thr: float = None,
                      top2_gap_map: np.ndarray = None,
                      gap_min: float = 2.0,
                      mask_override: np.ndarray = None):
    """
    4~5패널 시각화.
    mask_override가 주어지면 그 마스크로 disparity를 가림(엔트로피 대신).
    """
    img_np = np.asarray(left_img_pil)

    # 마스크 적용 (disparity에만)
    disp_for_vis = disp_map.copy()
    if mask_override is not None:
        mask_q = mask_override.astype(bool)
        disp_for_vis = np.where(mask_q, disp_for_vis, np.nan)
    elif entropy_map is not None and ent_vis_thr is not None:
        mask_q = (entropy_map <= float(ent_vis_thr))
        disp_for_vis = np.where(mask_q, disp_for_vis, np.nan)

    # 업샘플
    disp_up = upsample_nearest_4x(disp_for_vis).astype(np.float32)
    sim_up  = upsample_nearest_4x(peak_sim).astype(np.float32)
    ent_up  = upsample_nearest_4x(entropy_map).astype(np.float32) if entropy_map is not None else None
    gap_up  = upsample_nearest_4x(top2_gap_map).astype(np.float32) if top2_gap_map is not None else None

    has_ent = ent_up is not None
    has_gap = gap_up is not None
    num_cols = 3 + int(has_ent) + int(has_gap)

    fig, axes = plt.subplots(1, num_cols, figsize=(6 * num_cols, 6))
    if num_cols == 1:
        axes = [axes]
    else:
        axes = np.ravel(axes).tolist()

    fig.suptitle(f"Stereo Cost Volume (D=0..{max_disp}) from DINO ViT-B/8 (1/4-grid)")
    cmap_disp = _get_transparent_disp_cmap()

    # (1) 오버레이
    ax0 = axes[0]
    ax0.imshow(img_np)
    im0 = ax0.imshow(disp_up, vmin=0, vmax=max_disp, alpha=0.55, cmap=cmap_disp)
    ax0.set_title("Left image + disparity overlay")
    ax0.axis("off")
    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    cbar0.set_label("disparity (grid units; ~pixels = disp*4)")

    # (2) disparity
    ax1 = axes[1]
    im1 = ax1.imshow(disp_up, vmin=0, vmax=max_disp, cmap=cmap_disp)
    ax1.set_title("Disparity map (nearest x4)")
    ax1.axis("off")
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("disparity")

    # (3) peak similarity
    ax2 = axes[2]
    im2 = ax2.imshow((sim_up + 1.0) / 2.0, vmin=0.0, vmax=1.0, cmap='viridis')
    ax2.set_title("Peak cosine similarity (0..1)")
    ax2.axis("off")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("cosine similarity")

    col_idx = 3

    # (4) entropy (옵션)
    if has_ent:
        ax = axes[col_idx]; col_idx += 1
        im = ax.imshow(ent_up, vmin=0.0, vmax=1.0, cmap='magma')
        ax.set_title("Entropy (0..1)")
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("normalized entropy")

    # (5) top-2 gap (옵션)
    if has_gap:
        ax = axes[col_idx]
        gap_vis = np.where(gap_up >= float(gap_min), gap_up, np.nan)
        cmap_gap = _get_transparent_disp_cmap()
        im = ax.imshow(gap_vis, vmin=float(gap_min), vmax=max_disp, cmap=cmap_gap)
        ax.set_title(f"Top-2 disparity index gap (>= {gap_min})")
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("|d1 - d2|")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[Saved figure] {save_path}")

    plt.show()

def save_viz_panels(left_img_pil: Image.Image,
                    disp_map: np.ndarray,
                    peak_sim: np.ndarray,
                    max_disp: int,
                    entropy_map: np.ndarray,
                    ent_vis_thr: float,
                    out_dir: Path,
                    stem: str,
                    top2_gap_map: np.ndarray = None,
                    gap_min: float = 2.0,
                    mask_override: np.ndarray = None):
    """
    개별 PNG 저장:
      - overlay/<stem>.png
      - disp/<stem>.png
      - peak_sim/<stem>.png
      - entropy/<stem>.png
      - top2_gap/<stem>.png (옵션)
    """
    out_dir = Path(out_dir)
    (out_dir / "overlay").mkdir(parents=True, exist_ok=True)
    (out_dir / "disp").mkdir(parents=True, exist_ok=True)
    (out_dir / "peak_sim").mkdir(parents=True, exist_ok=True)
    (out_dir / "entropy").mkdir(parents=True, exist_ok=True)
    if top2_gap_map is not None:
        (out_dir / "top2_gap").mkdir(parents=True, exist_ok=True)

    img_np = np.asarray(left_img_pil)

    # ---- 마스크 적용(1/4 그리드) ----
    if mask_override is not None:
        mask_q = mask_override.astype(bool)
    else:
        mask_q = (entropy_map <= float(ent_vis_thr))

    disp_for_vis = np.where(mask_q, disp_map, np.nan)

    # 1/4 → 원본 크기
    disp_up = upsample_nearest_4x(disp_for_vis).astype(np.float32)  # [H,W]
    sim_up  = upsample_nearest_4x(peak_sim).astype(np.float32)      # [H,W]
    ent_up  = upsample_nearest_4x(entropy_map).astype(np.float32)   # [H,W]

    cmap_disp = _get_transparent_disp_cmap()

    # (1) overlay
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.imshow(img_np)
    im = ax.imshow(disp_up, vmin=0, vmax=max_disp, alpha=0.55, cmap=cmap_disp)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("disparity (grid units; ~pixels = disp*4)")
    overlay_path = out_dir / "overlay" / f"{stem}.png"
    fig.savefig(overlay_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    # (2) disparity
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(disp_up, vmin=0, vmax=max_disp, cmap=cmap_disp)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("disparity")
    disp_path = out_dir / "disp" / f"{stem}.png"
    fig.savefig(disp_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    # (3) peak similarity (0..1)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow((sim_up + 1.0) / 2.0, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cosine similarity")
    sim_path = out_dir / "peak_sim" / f"{stem}.png"
    fig.savefig(sim_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    # (4) entropy (0..1)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(ent_up, vmin=0.0, vmax=1.0, cmap="magma")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("normalized entropy")
    ent_path = out_dir / "entropy" / f"{stem}.png"
    fig.savefig(ent_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    # (5) top-2 disparity gap (>= gap_min만 보이도록)
    if top2_gap_map is not None:
        gap_up = upsample_nearest_4x(top2_gap_map).astype(np.float32)  # [H,W]
        gap_vis = np.where(gap_up >= float(gap_min), gap_up, np.nan)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        cmap_gap = _get_transparent_disp_cmap()
        im = ax.imshow(gap_vis, vmin=float(gap_min), vmax=max_disp, cmap=cmap_gap)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("top-2 disparity gap (grid units)")
        gap_path = out_dir / "top2_gap" / f"{stem}.png"
        fig.savefig(gap_path, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)

    print(f"[Saved panels] {overlay_path}, {disp_path}, {sim_path}, {ent_path}")


# =========================================================
# 4) ROIEntropySmoothL1Loss (그대로 사용 가능; 본 스크립트에선 옵션)
# =========================================================
class ROIEntropySmoothL1Loss(nn.Module):
    """
    교사(teacher): DINO 코스트볼륨 E_base → ROI∩invalid만 window 평균 보정 → E_ref
                   → argmax(E_ref) (default) 또는 soft-argmax 기대값
    감독 픽셀: ROI ∩ (after-entropy ≤ ent_thr)
    손실: SmoothL1(student_disp_cell, teacher_disp_cell)  — 1/4 cell 단위 기준

    Args
      max_disp: inclusive (bins = max_disp + 1)
      roi_mode: 'frac' | 'abs4'
      ent_T: entropy softmax 온도
      ent_thr: after-entropy 임계 (수용 마스크)
      max_half: 가로 윈도우 반폭 상한 (1/4 grid)
      teacher_type: 'arg' | 'soft'
      T_teacher: teacher softmax 온도(soft일 때)
      beta: SmoothL1 beta (0이면 L1)
      apply_area: 'roi' | 'all'
      student_unit: 'cell' | 'px' (px면 내부에서 /4)
    """
    def __init__(self,
                 max_disp: int,
                 roi_mode: str = "frac",
                 roi_u0: float = 0.3, roi_u1: float = 0.7,
                 roi_v0: float = 2/3, roi_v1: float = 1.0,
                 ent_T: float = 0.1,
                 ent_thr: float = 0.65,
                 max_half: int = None,
                 teacher_type: str = "arg",
                 T_teacher: float = 0.1,
                 beta: float = 1.0,
                 apply_area: str = "roi",
                 student_unit: str = "cell"):
        super().__init__()
        self.max_disp = int(max_disp)
        self.roi_mode = roi_mode
        self.u0, self.u1 = roi_u0, roi_u1
        self.v0, self.v1 = roi_v0, roi_v1
        self.ent_T = ent_T
        self.ent_thr = ent_thr
        self.max_half = max_half
        self.teacher_type = teacher_type
        self.T_teacher = T_teacher
        self.beta = beta
        self.apply_area = apply_area
        self.student_unit = student_unit

    def forward(self,
                FqL: torch.Tensor,            # [B,H4,W4,C] (L2 norm)
                FqR: torch.Tensor,            # [B,H4,W4,C]
                student_disp: torch.Tensor):  # [B,1,H4,W4] or [B,H4,W4] (cell 또는 px)
        assert FqL.shape == FqR.shape and FqL.dim() == 4
        B, H4, W4, C = FqL.shape
        device = FqL.device
        D = self.max_disp

        # 학생 단위 맞춤
        if student_disp.dim() == 3:
            student_disp = student_disp.unsqueeze(1)
        if self.student_unit == "px":
            student_disp = student_disp / 4.0  # 1/4-cell 단위로 변환

        losses = []
        for b in range(B):
            featL = FqL[b]; featR = FqR[b]                                 # [H4,W4,C]
            E_base = build_cost_volume(featL, featR, D)                    # [D+1,H4,W4]
            ent_before = build_entropy_map(E_base, T=self.ent_T, normalize=True)

            roi_mask = build_roi_mask(H4, W4, self.roi_mode,
                                      self.u0, self.u1, self.v0, self.v1, device)

            E_ref, ent_after, refine_mask = refine_cost_for_uncertain_roi(
                E_base, ent_before, self.ent_thr, roi_mask,
                max_half=self.max_half, ent_T=self.ent_T
            )

            if self.teacher_type == "soft":
                m = torch.amax(E_ref, dim=0, keepdim=True)
                prob = torch.softmax((E_ref - m) / max(self.T_teacher, 1e-6), dim=0)  # [D+1,H4,W4]
                disp_idx = torch.arange(D+1, device=device, dtype=prob.dtype).view(D+1,1,1)
                teacher = (prob * disp_idx).sum(dim=0)                                 # [H4,W4]
            else:
                teacher, _ = argmax_disparity(E_ref)                                    # long
                teacher = teacher.to(torch.float32)

            invalid_before = (ent_before > float(self.ent_thr))                         # [H4,W4]
            if self.apply_area == "roi":
                accept = (invalid_before & roi_mask)                                    # [H4,W4]
            else:
                accept = invalid_before

            tgt = teacher.unsqueeze(0)       # [1,H4,W4]
            pred = student_disp[b]           # [1,H4,W4]
            # SmoothL1(reduction='none') 후 마스크 평균
            diff = F.smooth_l1_loss(pred, tgt, beta=self.beta, reduction='none')       # [1,H4,W4]
            m = accept.unsqueeze(0).to(diff.dtype)
            denom = m.sum().clamp_min(1.0)
            loss_b = (diff * m).sum() / denom
            losses.append(loss_b)

        return torch.stack(losses).mean()


# ===========================================
# 5) 메인
# ===========================================
def main():
    parser = argparse.ArgumentParser(description="DINO 1/4 stereo + ROI entropy fill + viz (+ optional ROIEntropySmoothL1Loss)")
    # 단일 파일 모드
    parser.add_argument("--left",  type=str, default=None, help="Left image path")
    parser.add_argument("--right", type=str, default=None, help="Right image path")
    # 디렉터리 모드
    parser.add_argument("--left_dir",  type=str, default="/home/jaejun/dataset/MS2/sync_data/_2021-08-13-22-36-41/rgb/img_left", help="Directory of left images")
    parser.add_argument("--right_dir", type=str, default="/home/jaejun/dataset/MS2/sync_data/_2021-08-13-22-36-41/rgb/img_right", help="Directory of right images")
    parser.add_argument("--glob", type=str, default="*.png", help="Glob pattern for left images in left_dir")

    parser.add_argument("--max_disp", type=int, default=22, help="Max disparity on 1/4 grid (inclusive)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pad_to_8", action="store_true",
                        help="Pad right/bottom so H,W become multiples of 8 (both left & right).")

    # 엔트로피/시각화
    parser.add_argument("--ent_T", type=float, default=0.1,
                        help="Temperature for entropy; smaller → sharper distribution (try 0.05~0.2)")
    parser.add_argument("--ent_vis_thr", type=float, default=0.65,
                        help="Threshold: supervise/fill where after-entropy <= thr (on 1/4 grid)")
    parser.add_argument("--ent_vis_thr_after", type=float, default=1.0,
                        help="Threshold: supervise/fill where after-entropy <= thr (on 1/4 grid)")

    # 윈도우 반폭 상한(1/4 격자 픽셀 수)
    parser.add_argument("--win_half_max", type=int, default=None,
                        help="Cap for half window size on 1/4 grid (None: unlimited)")

    # ROI: u,v 각각 상/하한 지정
    parser.add_argument("--roi_mode", type=str, default="frac", choices=["frac", "abs4"],
                        help="ROI bounds: 'frac' in [0,1] or 'abs4' in 1/4-grid indices (inclusive).")
    parser.add_argument("--roi_u0", type=float, default=0.3)
    parser.add_argument("--roi_u1", type=float, default=0.7)
    parser.add_argument("--roi_v0", type=float, default=2/3)
    parser.add_argument("--roi_v1", type=float, default=1.0)

    # 출력
    parser.add_argument("--save_fig", type=str, default=None,
                        help="(Single-file mode) path to save a multi-panel visualization (png)")
    parser.add_argument("--out_dir", type=str, default="./out_dir_filled",
                        help="(Directory mode) Root output directory for panels")
    parser.add_argument("--save_numpy", action="store_true",
                        help="(Directory mode) Save npz (disp_base/disp_teacher/disp_filled/entropy_after etc.)")

    # (옵션) 학생 디스패리티로 SmoothL1 loss 확인
    parser.add_argument("--compute_loss", action="store_true",
                        help="If set, compute ROIEntropySmoothL1Loss using student_disp=disp_filled (cell units)")
    parser.add_argument("--loss_teacher_type", type=str, default="arg", choices=["arg","soft"])
    parser.add_argument("--loss_beta", type=float, default=1.0)

    args = parser.parse_args()
    device = torch.device(args.device)

    # 모드 판별
    dir_mode = (args.left_dir is not None and args.right_dir is not None and args.left is None and args.right is None)
    file_mode = (args.left is not None and args.right is not None)

    if not (dir_mode or file_mode):
        raise AssertionError("하나를 선택: (1) --left/--right (단일 파일) 또는 (2) --left_dir/--right_dir (디렉터리).")

    # DINO 모델 1회 로드
    model = load_dino(device)

    def process_pair(L_pil: Image.Image, R_pil: Image.Image, stem: str, save_root: Path = None):
        assert L_pil.size == R_pil.size, f"size mismatch: {L_pil.size} vs {R_pil.size}"

        if args.pad_to_8:
            Lp = pad_right_bottom_to_multiple(L_pil, mult=8)
            Rp = pad_right_bottom_to_multiple(R_pil, mult=8)
        else:
            Lp, Rp = L_pil, R_pil

        W, H = Lp.size
        assert (H % 8 == 0) and (W % 8 == 0), "H,W must be multiples of 8. Use --pad_to_8."

        xL = pil_to_tensor(Lp); xR = pil_to_tensor(Rp)  # [1,3,H,W]

        # ---- 특징/코스트/보정/추정 ----
        with torch.no_grad():
            featL = build_quarter_features(model, xL)[...]  # [H4,W4,C]
            featR = build_quarter_features(model, xR)[...]

            # Base/entropy-before
            cost_vol = build_cost_volume(featL, featR, args.max_disp)                    # [D+1,H4,W4]
            entropy_before = build_entropy_map(cost_vol, T=args.ent_T, normalize=True)   # [H4,W4]

            # ROI
            _, H4, W4 = cost_vol.shape
            roi_mask = build_roi_mask(H4, W4, args.roi_mode,
                                      args.roi_u0, args.roi_u1, args.roi_v0, args.roi_v1,
                                      device=entropy_before.device)

            # Adaptive window refine (ROI∩invalid만 보정)
            cost_vol_ref, entropy_after, refine_mask = refine_cost_for_uncertain_roi(
                cost_vol, entropy_before, ent_thr=args.ent_vis_thr,
                roi_mask=roi_mask, max_half=args.win_half_max, ent_T=args.ent_T
            )

            # base/teacher disparity (cell units)
            disp_base_cell, _   = argmax_disparity(cost_vol)       # [H4,W4], long
            disp_teacher_cell, peak_sim_t = argmax_disparity(cost_vol_ref)
            top2_gap_t = build_top2_gap_map(cost_vol_ref)

        # ---- FILL: ROI ∩ (after-entropy ≤ thr)만 teacher로 덮어쓰기 ----
        # fill_mask = ((entropy_after <= float(args.ent_vis_thr)) & roi_mask)  # [H4,W4], bool
        fill_mask = refine_mask  # [H4,W4], bool
        disp_filled_cell = disp_base_cell.clone().to(torch.float32)
        disp_filled_cell[fill_mask] = disp_teacher_cell[fill_mask].to(torch.float32)

        # ---- 시각화(전체 프레임; ROI만 자르지 않음) ----
        disp_filled_np = disp_filled_cell.detach().cpu().numpy().astype(np.float32)
        peak_sim_np    = peak_sim_t.detach().cpu().numpy().astype(np.float32)
        ent_after_np   = entropy_after.detach().cpu().numpy().astype(np.float32)
        gap_np         = top2_gap_t.detach().cpu().numpy().astype(np.float32)

        # 멀티패널: disp_filled를 전체 영역에 표시 + entropy(after) 패널 제공
        if save_root is None and args.save_fig:
            visualize_results(
                left_img_pil=Lp,
                disp_map=disp_filled_np,
                peak_sim=peak_sim_np,
                max_disp=args.max_disp,
                save_path=Path(args.save_fig),
                entropy_map=ent_after_np,
                ent_vis_thr=args.ent_vis_thr_after,
                top2_gap_map=gap_np,
                gap_min=1.0,
                mask_override=np.ones_like(ent_after_np, dtype=bool)   # 전체 표시
            )
        elif save_root is not None:
            out_dir = Path(save_root)
            save_viz_panels(
                left_img_pil=Lp,
                disp_map=disp_filled_np,
                peak_sim=peak_sim_np,
                max_disp=args.max_disp,
                entropy_map=ent_after_np,
                ent_vis_thr=args.ent_vis_thr_after,
                out_dir=out_dir,
                stem=stem,
                top2_gap_map=gap_np,
                gap_min=1.0,
                # mask_override=np.ones_like(ent_after_np, dtype=bool)  # 전체 표시
                mask_override=None  # 전체 표시
            )

        # (옵션) ROIEntropySmoothL1Loss 계산(학생=disp_filled_cell, cell 단위)
        if args.compute_loss:
            # 1/4 특징을 배치 차원 붙여 전달
            FqL = featL.unsqueeze(0)  # [1,H4,W4,C]
            FqR = featR.unsqueeze(0)  # [1,H4,W4,C]
            student = disp_filled_cell.unsqueeze(0).unsqueeze(0)  # [1,1,H4,W4] (cell)

            criterion = ROIEntropySmoothL1Loss(
                max_disp=args.max_disp,
                roi_mode=args.roi_mode, roi_u0=args.roi_u0, roi_u1=args.roi_u1,
                roi_v0=args.roi_v0, roi_v1=args.roi_v1,
                ent_T=args.ent_T, ent_thr=args.ent_vis_thr, max_half=args.win_half_max,
                teacher_type=args.loss_teacher_type, T_teacher=0.1,
                beta=args.loss_beta, apply_area="roi", student_unit="cell"
            ).to(device)

            with torch.no_grad():
                loss_val = criterion(FqL, FqR, student)
            print(f"[Loss] {stem}: ROIEntropySmoothL1Loss = {float(loss_val.item()):.6f}")

        # (옵션) npz 저장
        return {
            "disp_filled": disp_filled_np,
            "entropy_after": ent_after_np,
            "fill_ratio": float(fill_mask.float().mean().item())
        }

    if file_mode:
        left_path  = Path(args.left);  right_path = Path(args.right)
        assert left_path.exists(),  f"Left image not found: {left_path}"
        assert right_path.exists(), f"Right image not found: {right_path}"

        L_pil = Image.open(str(left_path)).convert("RGB")
        R_pil = Image.open(str(right_path)).convert("RGB")
        _ = process_pair(L_pil, R_pil, stem=left_path.stem, save_root=None)
        return

    # ===== 디렉터리 모드 =====
    left_dir  = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    out_dir   = Path(args.out_dir)

    assert left_dir.is_dir(),  f"--left_dir not found: {left_dir}"
    assert right_dir.is_dir(), f"--right_dir not found: {right_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # left_dir에서 glob으로 왼쪽 이미지 목록
    left_files = sorted(left_dir.glob(args.glob))
    assert len(left_files) > 0, f"No files matching {args.glob} in {left_dir}"

    # 오른쪽 이미지 매칭 함수(동일 stem 우선)
    def find_right_for_left(left_path: Path) -> Path:
        stem = left_path.stem
        for ext in IMG_EXTS:
            cand = right_dir / f"{stem}{ext}"
            if cand.exists():
                return cand
        return None

    processed, skipped = 0, 0
    for lp in left_files:
        rp = find_right_for_left(lp)
        if rp is None:
            print(f"[Skip] Right not found for left={lp.name}")
            skipped += 1
            continue

        try:
            L_pil = Image.open(str(lp)).convert("RGB")
            R_pil = Image.open(str(rp)).convert("RGB")

            if L_pil.size != R_pil.size:
                print(f"[Skip] size mismatch: {lp.name} vs {rp.name} => {L_pil.size} != {R_pil.size}")
                skipped += 1
                continue

            ret = process_pair(L_pil, R_pil, stem=lp.stem, save_root=out_dir)

            if args.save_numpy:
                npz_path = out_dir / f"{lp.stem}.npz"
                np.savez_compressed(
                    npz_path,
                    disp_filled=ret["disp_filled"],
                    entropy_after=ret["entropy_after"],
                    fill_ratio=np.array([ret["fill_ratio"]], dtype=np.float32),
                )
            processed += 1

        except Exception as e:
            print(f"[Error] {lp.name}: {e}")
            skipped += 1

    print(f"[Done] processed={processed}, skipped={skipped}, out_dir={out_dir.resolve()}")


# ===========================================
# 6) 보조: 오른쪽/아래 패딩으로 8의 배수 맞추기
# ===========================================
def pad_right_bottom_to_multiple(img_pil: Image.Image, mult: int = 8) -> Image.Image:
    """
    입력 이미지를 오른쪽/아래 방향으로만 패딩해서 (H,W)가 mult의 배수로.
    """
    W, H = img_pil.size
    pad_w = (mult - (W % mult)) % mult
    pad_h = (mult - (H % mult)) % mult
    if pad_w == 0 and pad_h == 0:
        return img_pil
    new_img = Image.new("RGB", (W + pad_w, H + pad_h))
    new_img.paste(img_pil, (0, 0))
    return new_img


# if __name__ == "__main__":
#     main()

def _parse_invalid_values(s: str):
    return [int(v) for v in s.split(",") if v.strip()!=""]

# -----------------------------
# 유틸: GT depth 로딩/리사이즈/변환
# -----------------------------
def _load_depth_any(depth_path: Path, scale: float = 256.0) -> np.ndarray:
    """
    depth 파일 로딩:
      - .png / .tif(f) : raw / scale → m
      - .npy           : 이미 m 단위라면 그대로, raw라면 scale로 나눠주세요
      - .npz           : 'depth' 키 우선, 없으면 첫 배열
    반환: float32 [H,W] (m)
    """
    ext = depth_path.suffix.lower()
    if ext in [".png", ".tif", ".tiff"]:
        arr = np.array(Image.open(str(depth_path)))
        arr = arr.astype(np.float32) / float(scale)
        return arr
    elif ext == ".npy":
        arr = np.load(str(depth_path))
        arr = arr.astype(np.float32)
        return arr
    elif ext == ".npz":
        data = np.load(str(depth_path))
        if "depth" in data:
            arr = data["depth"].astype(np.float32)
        else:
            # 첫 키
            k0 = list(data.keys())[0]
            arr = data[k0].astype(np.float32)
        return arr
    else:
        raise ValueError(f"Unsupported depth file: {depth_path}")

def _resize_depth_nearest(depth_m: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    Ht, Wt = target_hw
    if depth_m.shape[0] == Ht and depth_m.shape[1] == Wt:
        return depth_m
    # 깊이의 스케일 보존을 위해 nearest 권장
    return np.array(Image.fromarray(depth_m).resize((Wt, Ht), resample=Image.NEAREST))

def depth_to_disp_px(depth_m: np.ndarray, focal_px: float, baseline_m: float) -> np.ndarray:
    disp = (focal_px * baseline_m) / np.clip(depth_m, 1e-6, None)
    # depth==0 → disp=nan 으로 처리
    disp[~np.isfinite(disp)] = np.nan
    disp[depth_m <= 0] = np.nan
    return disp.astype(np.float32)

# -----------------------------
# 정량지표
# -----------------------------
def compute_disp_metrics(pred_px: np.ndarray, gt_px: np.ndarray, mask: np.ndarray) -> Optional[Dict[str, float]]:
    """
    pred_px, gt_px: [H,W] (px), NaN 허용
    mask: bool [H,W] (유효영역 제한; 예: fill-mask 업샘플)
    """
    valid = np.isfinite(pred_px) & np.isfinite(gt_px) & (gt_px > 0) & mask
    if valid.sum() == 0:
        return None
    err = np.abs(pred_px - gt_px)
    e   = err[valid]
    g   = gt_px[valid]
    EPE = float(e.mean())
    gt1 = float((e > 1.0).mean() * 100.0)
    gt2 = float((e > 2.0).mean() * 100.0)
    D1  = float((e > np.maximum(3.0, 0.05 * g)).mean() * 100.0)  # KITTI-style
    return {"EPE": EPE, "D1_all": D1, ">1px": gt1, ">2px": gt2, "Npix": int(valid.sum())}

def fmt_metrics(m: Optional[Dict[str, float]]) -> str:
    if not m: return "N/A"
    return (f"EPE={m['EPE']:.3f} | D1={m['D1_all']:.2f}% | "
            f">1px={m['>1px']:.2f}% | >2px={m['>2px']:.2f}% | N={m['Npix']}")
def load_gt_disp_px(gt_path, mode, depth_scale, disp_scale, focal_px, baseline_m,
                    target_hw, invalid_values=(0,65535), max_depth_m=200.0):
    raw = np.array(Image.open(str(gt_path))).astype(np.float32)  # raw 그대로
    raw = _resize_depth_nearest(raw, target_hw)

    # 1) 무효 마스크 (센티널)
    inv = np.zeros_like(raw, dtype=bool)
    if invalid_values:
        for v in invalid_values:
            inv |= (raw == float(v))

    if mode == "depth":
        depth_m = raw / float(depth_scale)
        # 2) 비현실 큰 깊이도 무효(센티널 필터 못잡은 경우 보호장치)
        if max_depth_m > 0:
            inv |= (depth_m > float(max_depth_m))
        inv |= ~np.isfinite(depth_m) | (depth_m <= 0)
        depth_m[inv] = np.nan
        disp_px = (focal_px * baseline_m) / np.clip(depth_m, 1e-6, None)
        disp_px[~np.isfinite(disp_px)] = np.nan
    else:  # "disp"
        disp_px = raw / float(disp_scale)
        inv |= ~np.isfinite(disp_px) | (disp_px <= 0)
        disp_px[inv] = np.nan

    return disp_px.astype(np.float32)

# -----------------------------
# 시각화 묶음 (Pred vs GT)
# -----------------------------
def visualize_pred_vs_gt(
    left_img_pil: Image.Image,
    disp_pred_cell: np.ndarray,   # [H4,W4] in 1/4-cell units
    fill_mask_q: np.ndarray,      # [H4,W4] bool (ROI∩invalid)
    disp_gt_px_full: np.ndarray,  # [H,W] in px
    max_disp_cell: int,
    save_path: Path,
    title: str = ""
):
    """
    다섯 패널:
      (1) Left + Pred(채운 부분만) overlay
      (2) Left + GT overlay
      (3) Pred disparity (full, px)
      (4) GT disparity (px)
      (5) |Pred−GT| (px) — 채운 부분만
    """
    img_np = np.asarray(left_img_pil)

    # Pred @ full-res(px) + Fill-mask 업샘플
    disp_pred_px_full = upsample_nearest_4x(disp_pred_cell).astype(np.float32) * 4.0  # [H,W]
    mask_full = upsample_nearest_4x(fill_mask_q.astype(np.uint8)).astype(bool)        # [H,W]

    # 컬러맵/스케일
    vmax_px = float(max_disp_cell * 4)
    cmap_disp = _get_transparent_disp_cmap()

    # (A) overlay용: 채운 부분만 가시화
    pred_vis = np.where(mask_full, disp_pred_px_full, np.nan)
    gt_vis   = np.where(np.isfinite(disp_gt_px_full), disp_gt_px_full, np.nan)

    # 에러 (채운 부분만)
    err = np.abs(disp_pred_px_full - disp_gt_px_full)
    err_vis = np.where(mask_full & np.isfinite(err), err, np.nan)

    # 정량 — 채운 부분 마스크 기준
    metrics_filled = compute_disp_metrics(disp_pred_px_full, disp_gt_px_full, mask_full)
    # 전체 유효 GT (참고)
    metrics_all    = compute_disp_metrics(disp_pred_px_full, disp_gt_px_full, np.isfinite(disp_gt_px_full))

    # ---- 그리기 ----
    fig, axes = plt.subplots(1, 5, figsize=(5*5.6, 5.6))
    axes = np.ravel(axes).tolist()
    if title:
        fig.suptitle(title, fontsize=12)

    # (1) Left + Pred(채운 부분) overlay
    ax = axes[0]
    ax.imshow(img_np)
    im = ax.imshow(pred_vis, vmin=0, vmax=vmax_px, alpha=0.55, cmap=cmap_disp)
    ax.set_title("Pred (filled region only)")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("disparity [px]")

    # (2) Left + GT overlay
    ax = axes[1]
    ax.imshow(img_np)
    im = ax.imshow(gt_vis, vmin=0, vmax=vmax_px, alpha=0.55, cmap=cmap_disp)
    ax.set_title("GT disparity")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("disparity [px]")

    # (3) Pred disparity (px, full)
    ax = axes[2]
    im = ax.imshow(disp_pred_px_full, vmin=0, vmax=vmax_px, cmap=cmap_disp)
    ax.set_title("Pred (px, full)")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("disparity [px]")

    # (4) GT disparity (px)
    ax = axes[3]
    im = ax.imshow(disp_gt_px_full, vmin=0, vmax=vmax_px, cmap=cmap_disp)
    ax.set_title("GT (px)")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("disparity [px]")

    # (5) |Pred-GT| (px) — filled만
    ax = axes[4]
    vmax_err = max(5.0, min(20.0, vmax_px/6.0))  # 보기 좋은 범위
    im = ax.imshow(err_vis, vmin=0, vmax=vmax_err, cmap="magma")
    ax.set_title("|Pred−GT| on filled region")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("absolute error [px]")

    # 하단 텍스트: metrics
    txt = (f"[Filled] {fmt_metrics(metrics_filled)}\n"
           f"[All-Valid] {fmt_metrics(metrics_all)}")
    fig.text(0.5, 0.02, txt, ha="center", va="bottom", fontsize=10)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"[Saved] {save_path}")

# -----------------------------
# DINO 기반 채움 + GT 비교 파이프라인
# -----------------------------
def process_pair_and_viz(
    model,
    left_pil: Image.Image, right_pil: Image.Image, stem: str,
    args
):
    # --- 패딩(옵션) ---
    if args.pad_to_8:
        Lp = pad_right_bottom_to_multiple(left_pil,  mult=8)
        Rp = pad_right_bottom_to_multiple(right_pil, mult=8)
    else:
        Lp, Rp = left_pil, right_pil

    W, H = Lp.size
    assert (H % 8 == 0) and (W % 8 == 0), "H,W must be multiples of 8. Use --pad_to_8."

    # --- 텐서 변환 ---
    xL = pil_to_tensor(Lp)  # [1,3,H,W] (ImageNet 정규화)
    xR = pil_to_tensor(Rp)

    with torch.no_grad():
        # 1/4 특징
        featL = build_quarter_features(model, xL)  # [H4,W4,C]
        featR = build_quarter_features(model, xR)

        # 코스트볼륨 & 엔트로피(before)
        cost_vol = build_cost_volume(featL, featR, args.max_disp)
        ent_before = build_entropy_map(cost_vol, T=args.ent_T, normalize=True)

        # ROI
        _, H4, W4 = cost_vol.shape
        roi_mask = build_roi_mask(
            H4, W4, args.roi_mode,
            args.roi_u0, args.roi_u1, args.roi_v0, args.roi_v1,
            device=ent_before.device
        )

        # Adaptive window refine (ROI∩invalid만 보정)
        cost_vol_ref, ent_after, refine_mask = refine_cost_for_uncertain_roi(
            cost_vol, ent_before, ent_thr=args.ent_vis_thr,
            roi_mask=roi_mask, max_half=args.win_half_max, ent_T=args.ent_T
        )

        # base/teacher disparity (cell units)
        disp_base_cell, _   = argmax_disparity(cost_vol)        # [H4,W4] long
        disp_teacher_cell, _ = argmax_disparity(cost_vol_ref)   # [H4,W4] long

    # ---- FILL: ROI∩invalid만 teacher로 덮어쓰기 ----
    fill_mask = refine_mask  # ROI ∩ invalid
    disp_base_cell = disp_base_cell.to(torch.float32)
    disp_teacher_cell = disp_teacher_cell.to(torch.float32)
    disp_filled_cell = disp_base_cell.clone()
    disp_filled_cell[fill_mask] = disp_teacher_cell[fill_mask]

    # numpy로 변환
    disp_filled_np = disp_filled_cell.cpu().numpy().astype(np.float32)
    fill_mask_np   = fill_mask.cpu().numpy().astype(bool)

    # ---- GT depth → disparity(px) ----
    # depth 파일 찾기
    depth_path = find_depth_for_left(Path(args.gt_depth_dir), stem)
    if depth_path is None:
        print(f"[Skip] GT depth not found for {stem}")
        return

    invalid_vals = _parse_invalid_values(args.gt_invalid_values)
    disp_gt_px = load_gt_disp_px(
        depth_path,
        mode=args.gt_mode,
        depth_scale=args.gt_depth_scale,
        disp_scale=args.gt_disp_scale,
        focal_px=args.focal_px,
        baseline_m=args.baseline_m,
        target_hw=(H, W),
        invalid_values=invalid_vals,
        max_depth_m=args.gt_max_depth_m,
    )

    # ---- 시각화 저장 ----
    out_png = Path(args.out_dir) / "cmp" / f"{stem}.png"
    visualize_pred_vs_gt(
        left_img_pil=Lp,
        disp_pred_cell=disp_filled_np,
        fill_mask_q=fill_mask_np,
        disp_gt_px_full=disp_gt_px,
        max_disp_cell=args.max_disp,
        save_path=out_png,
        title=f"{stem} | ROI-fill vs GT"
    )

def find_right_for_left(right_dir: Path, left_stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        cand = right_dir / f"{left_stem}{ext}"
        if cand.exists(): return cand
    return None

def find_depth_for_left(gt_depth_dir: Path, left_stem: str) -> Optional[Path]:
    exts = [".png", ".tif", ".tiff", ".npy", ".npz"]
    for ext in exts:
        cand = gt_depth_dir / f"{left_stem}{ext}"
        if cand.exists(): return cand
    return None

# -----------------------------
# CLI
# -----------------------------
def get_args():
    p = argparse.ArgumentParser("ROI entropy fill result vs GT disparity — visualization")
    # 입력(단일/디렉터리)
    p.add_argument("--left",  type=str, default=None)
    p.add_argument("--right", type=str, default=None)
    p.add_argument("--left_dir",  type=str, default="/home/jaejun/dataset/MS2/sync_data/tester/rgb/img_left")
    p.add_argument("--right_dir", type=str, default="/home/jaejun/dataset/MS2/sync_data/tester/rgb/img_right")
    p.add_argument("--glob", type=str, default="*.png")

    # DINO/코스트볼륨
    p.add_argument("--max_disp", type=int, default=14, help="1/4-grid max disparity (inclusive)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pad_to_8", action="store_true")

    # 엔트로피/ROI 파라미터 (기본값: 학습 코드와 동일 계열)
    p.add_argument("--ent_T", type=float, default=0.1)
    p.add_argument("--ent_vis_thr", type=float, default=0.8)
    p.add_argument("--roi_mode", type=str, default="frac", choices=["frac","abs4"])
    p.add_argument("--roi_u0", type=float, default=0.3)
    p.add_argument("--roi_u1", type=float, default=0.7)
    p.add_argument("--roi_v0", type=float, default=2/3)
    p.add_argument("--roi_v1", type=float, default=1.0)
    p.add_argument("--win_half_max", type=int, default=12)

    # GT/캘리브 (학습 코드 기본값 참고)
    p.add_argument("--gt_depth_dir",  type=str, default="/home/jaejun/dataset/MS2/proj_depth/tester/rgb/depth_filtered")
    p.add_argument("--gt_depth_scale", type=float, default=256.0)
    p.add_argument("--focal_px", type=float, default=764.5138549804688)
    p.add_argument("--baseline_m", type=float, default=0.29918420530585865)

    # 출력
    p.add_argument("--out_dir", type=str, default="./log/out_fill_vs_gt")
    p.add_argument("--gt_mode", type=str, default="depth", choices=["depth","disp"])
    # p.add_argument("--gt_depth_scale", type=float, default=256.0)
    p.add_argument("--gt_disp_scale",  type=float, default=1.0)
    p.add_argument("--gt_invalid_values", type=str, default="0,65535",
                help="raw GT에서 무효로 볼 값들(쉼표 구분). 예: 0,65535")
    p.add_argument("--gt_max_depth_m", type=float, default=200.0,
                help="이 값보다 큰 깊이는 무효 처리(센티널/오류 방지). 0이면 비활성")

    return p.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device)

    # 모드 판별
    dir_mode = (args.left is None and args.right is None and args.left_dir and args.right_dir)
    file_mode = (args.left is not None and args.right is not None)
    assert dir_mode or file_mode, "하나를 선택: (1) --left/--right 또는 (2) --left_dir/--right_dir"

    # DINO 로드
    model = load_dino(device)

    if file_mode:
        lp = Path(args.left); rp = Path(args.right)
        assert lp.exists(), f"Not found: {lp}"
        assert rp.exists(), f"Not found: {rp}"
        L = Image.open(str(lp)).convert("RGB")
        R = Image.open(str(rp)).convert("RGB")
        assert L.size == R.size, f"size mismatch: {L.size} vs {R.size}"
        process_pair_and_viz(model, L, R, lp.stem, args)
        return

    # 디렉터리 모드
    left_dir  = Path(args.left_dir);  right_dir = Path(args.right_dir)
    out_dir   = Path(args.out_dir)
    assert left_dir.is_dir() and right_dir.is_dir(), "입력 디렉터리 확인"
    Path(args.gt_depth_dir).mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    left_files = sorted(left_dir.glob(args.glob))
    assert len(left_files) > 0, f"No files matching {args.glob} in {left_dir}"

    processed, skipped = 0, 0
    for lp in left_files:
        rp = find_right_for_left(right_dir, lp.stem)
        if rp is None:
            print(f"[Skip] right not found for {lp.name}"); skipped += 1; continue
        depth_path = find_depth_for_left(Path(args.gt_depth_dir), lp.stem)
        if depth_path is None:
            print(f"[Skip] gt depth not found for {lp.name}"); skipped += 1; continue
        try:
            L = Image.open(str(lp)).convert("RGB")
            R = Image.open(str(rp)).convert("RGB")
            if L.size != R.size:
                print(f"[Skip] size mismatch: {lp.name} vs {rp.name}"); skipped += 1; continue
            process_pair_and_viz(model, L, R, lp.stem, args)
            processed += 1
        except Exception as e:
            print(f"[Error] {lp.name}: {e}")
            skipped += 1

    print(f"[Done] processed={processed}, skipped={skipped}, out_dir={out_dir.resolve()}")

if __name__ == "__main__":
    main()
