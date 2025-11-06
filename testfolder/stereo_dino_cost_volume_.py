# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import math

import numpy as np
from PIL import Image

import torch
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
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
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
#      재매칭 후 entropy로 시각화에서 새로 포함될 수 있게 함
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
    - mode='frac': u0,u1,v0,v1 ∈ [0,1], 비율(좌상 원점), u∈[0..1], v∈[0..1]
    - mode='abs4': u0,u1,v0,v1 는 1/4 격자 인덱스(정수 권장, inclusive)
    반환: [H4,W4] bool
    """
    mode = str(mode).lower()
    if mode not in ("frac", "abs4"):
        raise ValueError("roi_mode must be 'frac' or 'abs4'.")

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

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
        # abs4 (인덱스 inclusive)
        u0i = int(round(u0))
        u1i = int(round(u1))
        v0i = int(round(v0))
        v1i = int(round(v1))
        if u1i < u0i: u0i, u1i = u1i, u0i
        if v1i < v0i: v0i, v1i = v1i, v0i

    u0i = clamp(u0i, 0, W4 - 1)
    u1i = clamp(u1i, 0, W4 - 1)
    v0i = clamp(v0i, 0, H4 - 1)
    v1i = clamp(v1i, 0, H4 - 1)

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
    H, W = img_np.shape[:2]

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

    # (4) entropy (0..1) — before를 그대로 저장(일관성)
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
    gap_path = None
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

    print(f"[Saved panels] {overlay_path}, {disp_path}, {sim_path}, {ent_path}" + \
          (f", {gap_path}" if gap_path is not None else ""))


# ===========================================
# 4) 보조: 오른쪽/아래 패딩으로 8의 배수 맞추기
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


# ===========================================
# 5) 메인
# ===========================================
def main():
    parser = argparse.ArgumentParser(description="Stereo cost-volume with DINO ViT-B/8 (1/4 grid) + ROI adaptive window (invalid-only, ROI by u/v bounds)")
    # 단일 파일 모드
    parser.add_argument("--left",  type=str, default=None, help="Left image path")
    parser.add_argument("--right", type=str, default=None, help="Right image path")
    # 디렉터리 모드
    parser.add_argument("--left_dir",  type=str, default=None, help="Directory of left images")
    parser.add_argument("--right_dir", type=str, default=None, help="Directory of right images")
    parser.add_argument("--glob", type=str, default="*.png", help="Glob pattern for left images in left_dir")

    parser.add_argument("--max_disp", type=int, default=22, help="Max disparity on 1/4 grid (inclusive)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pad_to_8", action="store_true",
                        help="Pad right/bottom so H,W become multiples of 8 (both left & right).")
    parser.add_argument("--save_fig", type=str, default=None,
                        help="(Optional) path to save multi-panel visualization (png) for single-file mode")
    parser.add_argument("--out_dir", type=str, default="./out_dir",
                        help="(Directory mode) Root output directory for panels")
    parser.add_argument("--save_numpy", action="store_true",
                        help="(Directory mode) Save disp/peak_sim/entropy_before/entropy_after/top2_gap/viz_mask as npz")

    # 엔트로피/시각화
    parser.add_argument("--ent_T", type=float, default=0.1,
                        help="Temperature for entropy; smaller → sharper distribution (try 0.05~0.2)")
    parser.add_argument("--ent_vis_thr", type=float, default=0.6,
                        help="Visualization threshold: show disparity where entropy <= thr")

    # (옵션) 윈도우 반폭 상한(1/4 격자 픽셀 수)
    parser.add_argument("--win_half_max", type=int, default=None,
                        help="Cap for half window size on 1/4 grid (None: unlimited)")

    # -------- ROI: u,v 각각 상/하한 지정 --------
    parser.add_argument("--roi_mode", type=str, default="frac", choices=["frac", "abs4"],
                        help="ROI bounds mode: 'frac' in [0,1], or 'abs4' in 1/4-grid indices (inclusive).")
    # 기본: u 전체, v는 아래 1/3
    parser.add_argument("--roi_u0", type=float, default=0.3, help="Left bound of ROI in u (frac or abs4).")
    parser.add_argument("--roi_u1", type=float, default=0.7, help="Right bound of ROI in u (frac or abs4).")
    parser.add_argument("--roi_v0", type=float, default=2/3, help="Top bound of ROI in v (frac or abs4).")
    parser.add_argument("--roi_v1", type=float, default=1.0, help="Bottom bound of ROI in v (frac or abs4).")

    args = parser.parse_args()
    device = torch.device(args.device)

    # 모드 판별
    dir_mode = (args.left_dir is not None and args.right_dir is not None)
    file_mode = (args.left is not None and args.right is not None)

    assert dir_mode or file_mode, \
        "하나를 선택하세요: (1) --left/--right (단일 파일) 또는 (2) --left_dir/--right_dir (디렉터리)."
    if dir_mode and file_mode:
        raise AssertionError("단일 파일 모드와 디렉터리 모드를 동시에 사용할 수 없습니다.")

    # DINO 모델 1회 로드
    model = load_dino(device)

    if file_mode:
        # ===== 단일 파일 모드 =====
        left_path  = Path(args.left);  right_path = Path(args.right)
        assert left_path.exists(),  f"Left image not found: {left_path}"
        assert right_path.exists(), f"Right image not found: {right_path}"

        L_pil = Image.open(str(left_path)).convert("RGB")
        R_pil = Image.open(str(right_path)).convert("RGB")
        assert L_pil.size == R_pil.size, \
            f"Left/Right size mismatch: left={L_pil.size}, right={R_pil.size}"

        if args.pad_to_8:
            L_pil = pad_right_bottom_to_multiple(L_pil, mult=8)
            R_pil = pad_right_bottom_to_multiple(R_pil, mult=8)

        W, H = L_pil.size
        assert (H % 8 == 0) and (W % 8 == 0), \
            "H,W must be multiples of 8. Use --pad_to_8 to pad automatically."

        xL = pil_to_tensor(L_pil)  # [1,3,H,W]
        xR = pil_to_tensor(R_pil)

        with torch.no_grad():
            featL = build_quarter_features(model, xL)  # [H//4,W//4,C]
            featR = build_quarter_features(model, xR)

            cost_vol = build_cost_volume(featL, featR, args.max_disp)          # [D+1,H4,W4]
            entropy_before = build_entropy_map(cost_vol, T=args.ent_T, normalize=True)  # [H4,W4]

            # ROI 마스크 (u,v 상/하한)
            _, H4, W4 = cost_vol.shape
            roi_mask = build_roi_mask(H4, W4,
                                      args.roi_mode,
                                      args.roi_u0, args.roi_u1,
                                      args.roi_v0, args.roi_v1,
                                      device=entropy_before.device)

            # ---- ROI∩invalid만 adaptive window 재추정 ----
            cost_vol_ref, entropy_after, refine_mask = refine_cost_for_uncertain_roi(
                cost_vol, entropy_before, ent_thr=args.ent_vis_thr,
                roi_mask=roi_mask, max_half=args.win_half_max, ent_T=args.ent_T
            )

            # 최종 추정은 보정된 볼륨로
            disp_map_t, peak_sim_t = argmax_disparity(cost_vol_ref)
            top2_gap_t = build_top2_gap_map(cost_vol_ref)

            # 시각화 마스크 = (before≤thr) ∪ (after≤thr in ROI)
            viz_mask = build_union_viz_mask(entropy_before, entropy_after, args.ent_vis_thr, roi_mask)

        disp_map = disp_map_t.detach().cpu().numpy().astype(np.int32)
        peak_sim = peak_sim_t.detach().cpu().numpy().astype(np.float32)
        ent_before_np = entropy_before.detach().cpu().numpy().astype(np.float32)  # (패널: before 유지)
        ent_after_np  = entropy_after.detach().cpu().numpy().astype(np.float32)
        top2_gap = top2_gap_t.detach().cpu().numpy().astype(np.float32)
        viz_mask_np = viz_mask.detach().cpu().numpy().astype(bool)

        visualize_results(
            L_pil, disp_map, peak_sim, args.max_disp,
            save_path=Path(args.save_fig) if args.save_fig else None,
            entropy_map=ent_before_np,          # entropy 패널은 'before'를 그대로
            ent_vis_thr=args.ent_vis_thr,
            top2_gap_map=top2_gap,
            gap_min=1.0,
            mask_override=viz_mask_np           # 합집합 마스크로 표시
        )
        return

    # ===== 디렉터리 모드 =====
    left_dir  = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    out_dir   = Path(args.out_dir)

    assert left_dir.is_dir(),  f"--left_dir not found: {left_dir}"
    assert right_dir.is_dir(), f"--right_dir not found: {right_dir}"

    # left_dir에서 glob으로 왼쪽 이미지 목록
    left_files = sorted(left_dir.glob(args.glob))
    assert len(left_files) > 0, f"No files matching {args.glob} in {left_dir}"

    # npz 저장 폴더(옵션)
    if args.save_numpy:
        (out_dir / "npy").mkdir(parents=True, exist_ok=True)

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
            # 1) 로드
            L_pil = Image.open(str(lp)).convert("RGB")
            R_pil = Image.open(str(rp)).convert("RGB")

            # 2) 크기 체크
            if L_pil.size != R_pil.size:
                print(f"[Skip] size mismatch: {lp.name} vs {rp.name} => {L_pil.size} != {R_pil.size}")
                skipped += 1
                continue

            # 3) 패딩
            if args.pad_to_8:
                L_pil = pad_right_bottom_to_multiple(L_pil, mult=8)
                R_pil = pad_right_bottom_to_multiple(R_pil, mult=8)

            W, H = L_pil.size
            if (H % 8 != 0) or (W % 8 != 0):
                print(f"[Skip] not multiple of 8 even after padding: {lp.name}")
                skipped += 1
                continue

            # 4) 텐서 변환
            xL = pil_to_tensor(L_pil)
            xR = pil_to_tensor(R_pil)

            # 5) 특징/코스트/보정/추정
            with torch.no_grad():
                featL = build_quarter_features(model, xL)
                featR = build_quarter_features(model, xR)

                cost_vol = build_cost_volume(featL, featR, args.max_disp)            # [D+1,H4,W4]
                entropy_before = build_entropy_map(cost_vol, T=args.ent_T, normalize=True)

                # ROI 마스크
                _, H4, W4 = cost_vol.shape
                roi_mask = build_roi_mask(H4, W4,
                                          args.roi_mode,
                                          args.roi_u0, args.roi_u1,
                                          args.roi_v0, args.roi_v1,
                                          device=entropy_before.device)

                cost_vol_ref, entropy_after, refine_mask = refine_cost_for_uncertain_roi(
                    cost_vol, entropy_before, ent_thr=args.ent_vis_thr,
                    roi_mask=roi_mask, max_half=args.win_half_max, ent_T=args.ent_T
                )

                disp_map_t, peak_sim_t = argmax_disparity(cost_vol_ref)
                top2_gap_t = build_top2_gap_map(cost_vol_ref)

                # 시각화 마스크 = (before≤thr) ∪ (after≤thr in ROI)
                viz_mask = build_union_viz_mask(entropy_before, entropy_after, 1.0, roi_mask)

            # 6) NumPy 변환
            disp_map = disp_map_t.detach().cpu().numpy().astype(np.int32)
            peak_sim = peak_sim_t.detach().cpu().numpy().astype(np.float32)
            ent_before_np = entropy_before.detach().cpu().numpy().astype(np.float32)
            ent_after_np  = entropy_after.detach().cpu().numpy().astype(np.float32)
            top2_gap = top2_gap_t.detach().cpu().numpy().astype(np.float32)
            viz_mask_np = viz_mask.detach().cpu().numpy().astype(bool)

            # 7) 패널 저장
            stem = lp.stem
            save_viz_panels(
                L_pil, disp_map, peak_sim, args.max_disp,
                entropy_map=ent_before_np,         # entropy 패널은 'before'를 그대로
                ent_vis_thr=args.ent_vis_thr,
                out_dir=out_dir,
                stem=stem,
                top2_gap_map=top2_gap,
                gap_min=1.0,
                mask_override=viz_mask_np          # 합집합 마스크로 표시
            )

            # 8) (옵션) npz 저장
            if args.save_numpy:
                npz_path = out_dir / "npy" / f"{stem}.npz"
                np.savez_compressed(
                    npz_path,
                    disp=disp_map,                # int32 (refined for viz)
                    peak_sim=peak_sim,            # float32 (refined for viz)
                    entropy_before=ent_before_np, # float32 (0..1)
                    entropy_after=ent_after_np,   # float32 (0..1)
                    top2_gap=top2_gap,            # float32
                    viz_mask=viz_mask_np.astype(np.uint8)
                )
            processed += 1

        except Exception as e:
            print(f"[Error] {lp.name}: {e}")
            skipped += 1

    print(f"[Done] processed={processed}, skipped={skipped}, out_dir={out_dir.resolve()}")

if __name__ == "__main__":
    main()
