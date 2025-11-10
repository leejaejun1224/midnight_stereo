# -*- coding: utf-8 -*-
import os
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# 백엔드 없이 저장만
import matplotlib
matplotlib.use("Agg")
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
    facebookresearch/dino 의 ViT-S/8(dino_vits8) 로드
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
# 1) ViT-S/8 패치 토큰 추출 + 1/4 격자 생성(학습 없음)
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
# 3) 시각화 유틸
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
        try: cmap = cmap.copy()
        except Exception: pass
        try: cmap.set_bad((0, 0, 0, 0))
        except Exception: pass
    return cmap

def _ensure_dirs(root: Path, *names: str) -> Dict[str, Path]:
    out = {}
    for n in names:
        p = root / n
        p.mkdir(parents=True, exist_ok=True)
        out[n] = p
    return out

def _save_map(path: Path, arr: np.ndarray, vmin=None, vmax=None, title=None,
              cmap="viridis", with_colorbar=True, overlay_img: np.ndarray=None, alpha=0.55):
    plt.figure(figsize=(7,7))
    if overlay_img is not None:
        plt.imshow(overlay_img)
        im = plt.imshow(arr, vmin=vmin, vmax=vmax, alpha=alpha, cmap=cmap)
    else:
        im = plt.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.axis("off")
    if title: plt.title(title)
    if with_colorbar:
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        # 라벨은 호출부에서 설정
    plt.tight_layout()
    plt.savefig(str(path), bbox_inches="tight", pad_inches=0.01)
    plt.close()

# -------------------------------------------
# GT 로딩 / 변환 (depth→disp or disp 원본)
# -------------------------------------------
def _parse_invalid_values(s: str):
    return [int(v) for v in s.split(",") if v.strip()!=""]

def _resize_depth_nearest(depth_m: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    Ht, Wt = target_hw
    if depth_m.shape[0] == Ht and depth_m.shape[1] == Wt:
        return depth_m
    return np.array(Image.fromarray(depth_m).resize((Wt, Ht), resample=Image.NEAREST))

def load_gt_disp_px(gt_path: Path, mode: str, depth_scale: float, disp_scale: float,
                    focal_px: float, baseline_m: float, target_hw: Tuple[int,int],
                    invalid_values=(0,65535), max_depth_m: float = 200.0) -> np.ndarray:
    """
    GT를 disparity(px)로 반환. NaN 허용.
    - mode='depth' : raw/scale -> meters -> disp
    - mode='disp'  : raw/scale -> px
    무효(센티넬/범위/비유한)는 NaN 처리.
    """
    raw = np.array(Image.open(str(gt_path)))
    raw = _resize_depth_nearest(raw, target_hw)
    rawf = raw.astype(np.float32)

    inv = np.zeros_like(rawf, dtype=bool)
    if invalid_values:
        for v in invalid_values:
            inv |= (rawf == float(v))

    if mode == "depth":
        depth_m = rawf / float(depth_scale)
        if max_depth_m and max_depth_m > 0:
            inv |= (depth_m > float(max_depth_m))
        inv |= ~np.isfinite(depth_m) | (depth_m <= 0)
        depth_m[inv] = np.nan
        disp_px = (focal_px * baseline_m) / np.clip(depth_m, 1e-6, None)
        disp_px[~np.isfinite(disp_px)] = np.nan
    elif mode == "disp":
        disp_px = rawf / float(disp_scale)
        inv |= ~np.isfinite(disp_px) | (disp_px <= 0)
        disp_px[inv] = np.nan
    else:
        raise ValueError("gt_mode must be 'depth' or 'disp'")

    return disp_px.astype(np.float32)

# -------------------------------------------
# 개별 PNG로 저장 (pred/gt/error/entropy 등)
# -------------------------------------------
def save_cmp_panels_separate(
    left_img_pil: Image.Image,
    disp_pred_cell: np.ndarray,   # [H4,W4] in 1/4-cell units
    fill_mask_q: np.ndarray,      # [H4,W4] bool (ROI∩invalid)
    disp_gt_px_full: np.ndarray,  # [H,W] in px (NaN 허용)
    ent_before_q: np.ndarray,     # [H4,W4] (0..1)
    ent_after_q: np.ndarray,      # [H4,W4] (0..1)
    max_disp_cell: int,
    out_root: Path,
    stem: str
):
    out = _ensure_dirs(
        out_root,
        "pred_overlay_all","pred_overlay_filled","gt_overlay",
        "pred_disp_px","gt_disp_px","error_abs_px",
        "entropy_before","entropy_after","fill_mask"
    )

    img_np = np.asarray(left_img_pil)

    # Pred @ full-res(px) + Fill-mask 업샘플
    disp_pred_px_full = upsample_nearest_4x(disp_pred_cell).astype(np.float32) * 4.0  # [H,W]
    mask_full = upsample_nearest_4x(fill_mask_q.astype(np.uint8)).astype(bool)        # [H,W]

    # Entropy upsample to full-res (보기 편하게)
    ent_b_full = upsample_nearest_4x(ent_before_q).astype(np.float32)  # [H,W], 0..1
    ent_a_full = upsample_nearest_4x(ent_after_q).astype(np.float32)

    # 스케일
    vmax_px = float(max_disp_cell * 4)
    cmap_disp = _get_transparent_disp_cmap()

    # 1) Pred overlay (전체)
    p = out["pred_overlay_all"] / f"{stem}.png"
    _save_map(p, disp_pred_px_full, vmin=0, vmax=vmax_px, title=None,
              cmap=cmap_disp, with_colorbar=True, overlay_img=img_np, alpha=0.55)
    # colorbar 라벨 변경
    # (matplotlib 객체 핸들을 재사용하지 않으므로 간단화를 위해 라벨은 생략)

    # 2) Pred overlay (filled만)
    pred_vis_filled = np.where(mask_full, disp_pred_px_full, np.nan)
    p = out["pred_overlay_filled"] / f"{stem}.png"
    _save_map(p, pred_vis_filled, vmin=0, vmax=vmax_px, title=None,
              cmap=cmap_disp, with_colorbar=True, overlay_img=img_np, alpha=0.55)

    # 3) GT overlay
    gt_vis = np.where(np.isfinite(disp_gt_px_full), disp_gt_px_full, np.nan)
    p = out["gt_overlay"] / f"{stem}.png"
    _save_map(p, gt_vis, vmin=0, vmax=vmax_px, title=None,
              cmap=cmap_disp, with_colorbar=True, overlay_img=img_np, alpha=0.55)

    # 4) Pred disp(px)
    p = out["pred_disp_px"] / f"{stem}.png"
    _save_map(p, disp_pred_px_full, vmin=0, vmax=vmax_px, title=None,
              cmap=cmap_disp, with_colorbar=True)

    # 5) GT disp(px)
    p = out["gt_disp_px"] / f"{stem}.png"
    _save_map(p, disp_gt_px_full, vmin=0, vmax=vmax_px, title=None,
              cmap=cmap_disp, with_colorbar=True)

    # 6) |Pred−GT| (px)
    err = np.abs(disp_pred_px_full - disp_gt_px_full).astype(np.float32)
    err[~np.isfinite(err)] = np.nan
    vmax_err = max(5.0, min(20.0, vmax_px/6.0))
    p = out["error_abs_px"] / f"{stem}.png"
    _save_map(p, err, vmin=0, vmax=vmax_err, title=None,
              cmap="magma", with_colorbar=True)

    # 7) entropy before (0..1)
    p = out["entropy_before"] / f"{stem}.png"
    _save_map(p, ent_b_full, vmin=0.0, vmax=1.0, title=None,
              cmap="magma", with_colorbar=True)

    # 8) entropy after (0..1)
    p = out["entropy_after"] / f"{stem}.png"
    _save_map(p, ent_a_full, vmin=0.0, vmax=1.0, title=None,
              cmap="magma", with_colorbar=True)

    # 9) fill mask (binary)
    p = out["fill_mask"] / f"{stem}.png"
    _save_map(p, mask_full.astype(np.float32), vmin=0.0, vmax=1.0, title=None,
              cmap="gray", with_colorbar=False)

    print(f"[Saved all panels for] {stem}")

# -----------------------------
# DINO 기반 채움 + GT 비교 파이프라인
# -----------------------------
@torch.no_grad()
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
        cost_vol_ref, ent_mid, refine_mask = refine_cost_for_uncertain_roi(
            cost_vol, ent_before, ent_thr=args.ent_vis_thr,
            roi_mask=roi_mask, max_half=args.win_half_max, ent_T=args.ent_T
        )
        
        cost_vol_ref, ent_after, refine_mask = refine_cost_for_uncertain_roi(
            cost_vol_ref, ent_mid, ent_thr=args.ent_vis_thr,
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
    ent_before_np  = ent_before.cpu().numpy().astype(np.float32)
    ent_after_np   = ent_after.cpu().numpy().astype(np.float32)

    # ---- GT depth/disp → disparity(px) ----
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

    # ---- 개별 PNG 저장 ----
    out_root = Path(args.out_dir)
    save_cmp_panels_separate(
        left_img_pil=Lp,
        disp_pred_cell=disp_filled_np,
        fill_mask_q=fill_mask_np,
        disp_gt_px_full=disp_gt_px,
        ent_before_q=ent_before_np,
        ent_after_q=ent_after_np,
        max_disp_cell=args.max_disp,
        out_root=out_root,
        stem=stem
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


# -----------------------------
# CLI
# -----------------------------
def get_args():
    p = argparse.ArgumentParser("ROI entropy fill vs GT — save panels to separate folders")
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

    # 엔트로피/ROI 파라미터
    p.add_argument("--ent_T", type=float, default=0.1)
    p.add_argument("--ent_vis_thr", type=float, default=0.6)
    p.add_argument("--roi_mode", type=str, default="frac", choices=["frac","abs4"])
    p.add_argument("--roi_u0", type=float, default=0.2)
    p.add_argument("--roi_u1", type=float, default=0.7)
    p.add_argument("--roi_v0", type=float, default=2/3)
    p.add_argument("--roi_v1", type=float, default=1.0)
    p.add_argument("--win_half_max", type=int, default=12)

    # GT/캘리브
    p.add_argument("--gt_depth_dir",  type=str, default="/home/jaejun/dataset/MS2/proj_depth/tester/rgb/depth_filtered")
    p.add_argument("--gt_mode",       type=str, default="depth", choices=["depth","disp"])
    p.add_argument("--gt_depth_scale", type=float, default=256.0)
    p.add_argument("--gt_disp_scale",  type=float, default=1.0)
    p.add_argument("--gt_invalid_values", type=str, default="0,65535",
                   help="raw GT에서 무효로 볼 값들(쉼표 구분). 예: 0,65535")
    p.add_argument("--gt_max_depth_m", type=float, default=200.0,
                   help="이 값보다 큰 깊이는 무효 처리(0: 비활성)")

    p.add_argument("--focal_px", type=float, default=764.5138549804688)
    p.add_argument("--baseline_m", type=float, default=0.29918420530585865)

    # 출력
    p.add_argument("--out_dir", type=str, default="./log/out_fill_vs_gt")

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
