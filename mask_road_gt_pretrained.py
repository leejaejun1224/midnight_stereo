# -*- coding: utf-8 -*-
"""
Stereo pseudo-label generation with
 - generic ViT-based stereo backbone (vit_cn.StereoModel)
 - 1/4-resolution cost volume + entropy-based ROI refinement

기존 DINO ViT-S/8 + 인터리빙 부분을 제거하고,
사용자가 학습한 vit_cn.StereoModel 백본만 가져와서
1/4 해상도 feature → cost volume → entropy ROI fill → GT 비교/시각화 + 정량평가
까지 하는 전체 파이프라인입니다.

※ 중요한 포인트
- vit_cn.StereoModel 의 forward 출력 dict 안에서
  1/4 해상도 피처맵을 어떻게 꺼내는지만 실제 구현에 맞게 수정하면 됩니다.
  (기본 가정: bb_out["left"]["cossim_feat_1_4"], bb_out["right"]["cossim_feat_1_4"])
"""

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

# 백엔드 없이 저장만
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

# =========================
# (사용자 프로젝트의 ViT 기반 스테레오 백본)
# =========================
from vit_cn import StereoModel  # 사용자의 기존 코드


# ===========================================
# 0) 공통 전처리 (ImageNet 정규화)
# ===========================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def pil_to_tensor(img_pil: Image.Image) -> torch.Tensor:
    """
    PIL → Tensor [1,3,H,W] + ImageNet 정규화
    """
    from torchvision import transforms

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    x = tfm(img_pil).unsqueeze(0)  # [1,3,H,W]
    return x


# =========================================================
# padding 유틸 (학습 코드와 동일 개념)
# =========================================================
def pad_to_multiple(x: torch.Tensor, mult: int = 16, mode: str = "replicate"):
    """
    x: [B,C,H,W]
    H,W 를 mult 의 배수가 되도록 오른쪽/아래 패딩.
    반환: (패딩된 텐서, (pad_bottom, pad_right))
    """
    H, W = x.shape[-2], x.shape[-1]
    pad_r = (-W) % mult
    pad_b = (-H) % mult
    if pad_r or pad_b:
        x = F.pad(x, (0, pad_r, 0, pad_b), mode=mode)
    return x, (pad_b, pad_r)


def unpad_last2(x: torch.Tensor, pad: Tuple[int, int]) -> torch.Tensor:
    """
    pad_to_multiple 로 패딩한 텐서를 다시 잘라냄.
    pad: (pad_bottom, pad_right)
    """
    pad_b, pad_r = pad
    if pad_b == 0 and pad_r == 0:
        return x
    H, W = x.shape[-2], x.shape[-1]
    return x[..., : H - pad_b, : W - pad_r].contiguous()


# =========================================================
# 1) StereoModel 기반 1/4 해상도 백본 래퍼
# =========================================================
def _strip_prefix_once(name: str, prefix: str) -> str:
    return name[len(prefix):] if name.startswith(prefix) else name


def load_backbone_weights(model: nn.Module, ckpt_path: str, device, verbose: bool = True):
    """
    다양한 형태의 .pth 에서 StereoModel(=backbone) 부분만 로드하기 위한 helper.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Backbone ckpt not found: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location=device)

    def _strip_common_prefixes(sd: dict) -> dict:
        prefixes = ["module.", "model.", "ckpt_model.", "ema.", "net.", "nets."]
        out = {}
        for k, v in sd.items():
            nk = k
            for p in prefixes:
                nk = _strip_prefix_once(nk, p)
            out[nk] = v
        return out

    def _try_load(sd_raw: dict) -> bool:
        """
        stereo.xxx / decoder.xxx 같이 섞여 있을 수 있는 state_dict 에서
        stereo.xxx 만 떼어내서 모델에 넣는다.
        """
        if not isinstance(sd_raw, dict) or len(sd_raw) == 0:
            return False

        sd = _strip_common_prefixes(sd_raw)

        # stereo.* 키만 따로 있는 경우 → 그것만 사용
        if any(k.startswith("stereo.") for k in sd.keys()):
            stereo_sd = {_strip_prefix_once(k, "stereo."): v
                         for k, v in sd.items() if k.startswith("stereo.")}
        else:
            # stereo.* 가 없으면, 그냥 전체를 backbone 으로 간주
            stereo_sd = sd

        try:
            missing, unexpected = model.load_state_dict(stereo_sd, strict=False)
            if verbose:
                print(f"[Backbone] load_state_dict(strict=False)  "
                      f"missing={len(missing)}, unexpected={len(unexpected)}")
            return True
        except Exception as e:
            if verbose:
                print(f"[Backbone] load_state_dict 실패: {e}")
            return False

    # --- 후보 state_dict 추출 ---
    candidates: List[Dict[str, torch.Tensor]] = []

    if isinstance(obj, nn.Module):
        candidates.append(obj.state_dict())

    if isinstance(obj, dict):
        # raw state_dict 형태
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            candidates.append(obj)

        # 중첩 구조
        for key in ["state_dict", "model_state_dict", "model",
                    "ckpt_model", "ema", "net", "nets", "stereo"]:
            if key in obj:
                v = obj[key]
                if isinstance(v, nn.Module):
                    candidates.append(v.state_dict())
                elif isinstance(v, dict):
                    candidates.append(v)

    if not candidates:
        raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path} (type={type(obj)})")

    last_err: Optional[Exception] = None
    for sd in candidates:
        try:
            if _try_load(sd):
                if verbose:
                    print(f"[Backbone] Loaded weights from {ckpt_path}")
                return
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to load backbone weights from {ckpt_path}: {last_err}")


class StereoFeatureBackbone(nn.Module):
    """
    vit_cn.StereoModel 을 이용해서 1/4 해상도 좌/우 피처맵을 뽑아주는 래퍼.

    StereoModel forward(imgL, imgR) → dict:
        {
          "left": {
            "cossim_feat_1_4": [B,H/4,W/4,C],  # ★ 우리가 쓸 것
            ...
          },
          "right": { ... },
          "meta": { ... }
        }
    """

    def __init__(
        self,
        device: torch.device,
        backbone_ckpt: str,
        pad_mult: int = 16,
        verbose: bool = True,
    ):
        super().__init__()
        self.device = device
        self.verbose = verbose

        self.model = StereoModel(
            device=device,
            freeze_vit=True,
            amp=True,
            autopad_to_8=True,
        ).to(device).eval()

        if backbone_ckpt is not None and backbone_ckpt != "":
            load_backbone_weights(self.model, backbone_ckpt, device, verbose=verbose)
        else:
            if verbose:
                print("[Backbone] WARNING: no --backbone_ckpt provided → random init!")

        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def extract_quarter_features(
        self,
        imgL: torch.Tensor,  # [1,3,H,W]
        imgR: torch.Tensor,  # [1,3,H,W]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        반환:
          featL_hw: [H4, W4, C], L2 정규화
          featR_hw: [H4, W4, C], L2 정규화
        """
        imgL = imgL.to(self.device, non_blocking=True)
        imgR = imgR.to(self.device, non_blocking=True)

        bb_out = self.model(imgL, imgR)   # dict

        if "left" not in bb_out or "right" not in bb_out:
            raise KeyError(f"StereoModel output must have 'left' and 'right' keys, got {bb_out.keys()}")

        outL = bb_out["left"]
        outR = bb_out["right"]

        if "cossim_feat_1_4" not in outL or "cossim_feat_1_4" not in outR:
            raise KeyError(
                "Expected 'cossim_feat_1_4' in StereoModel backbone output.\n"
                "StereoModel._backbone_single 의 반환 dict 구조를 다시 확인해 주세요."
            )

        # [B,H4,W4,C]
        featL_q = outL["cossim_feat_1_4"]
        featR_q = outR["cossim_feat_1_4"]

        if featL_q.dim() != 4 or featR_q.dim() != 4:
            raise ValueError(
                f"'cossim_feat_1_4' must be [B,H4,W4,C], got {featL_q.shape}, {featR_q.shape}"
            )

        featL_hw = featL_q[0]   # [H4,W4,C]
        featR_hw = featR_q[0]

        featL_hw = F.normalize(featL_hw, dim=-1)
        featR_hw = F.normalize(featR_hw, dim=-1)

        if self.verbose:
            print(f"[Backbone] cossim_feat_1_4 → featL shape={featL_hw.shape}, featR shape={featR_hw.shape}")

        return featL_hw, featR_hw


# ===========================================
# 2) Cost Volume 구축 (좌→우, 수평 시차만)
# ===========================================
@torch.no_grad()
def build_cost_volume(featL: torch.Tensor, featR: torch.Tensor, max_disp: int) -> torch.Tensor:
    """
    featL, featR: [H4, W4, C]
    반환: cost_vol [D+1, H4, W4]
    """
    assert featL.shape == featR.shape
    H4, W4, C = featL.shape
    device = featL.device
    D = int(max_disp)

    cost_vol = torch.full((D + 1, H4, W4), float("-inf"), device=device, dtype=featL.dtype)

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
def argmax_disparity(cost_vol: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    반환:
      - disp_map: [H4,W4] (long, index units)
      - peak_sim: [H4,W4] (float)
    """
    peak_sim, disp_map = cost_vol.max(dim=0)
    return disp_map, peak_sim


# ===========================================
# 2-1) Entropy / Gap
# ===========================================
@torch.no_grad()
def build_entropy_map(
    cost_vol: torch.Tensor,
    T: float = 0.1,
    eps: float = 1e-8,
    normalize: bool = True,
) -> torch.Tensor:
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
        return torch.full((H4, W4), float("nan"), device=cost_vol.device, dtype=cost_vol.dtype)

    valid = torch.isfinite(cost_vol)
    Deff  = valid.sum(dim=0)

    _, idxs = torch.topk(cost_vol, k=2, dim=0)   # [2,H4,W4]
    d1 = idxs[0].to(torch.float32)
    d2 = idxs[1].to(torch.float32)
    gap = (d1 - d2).abs()
    gap = torch.where(Deff >= 2, gap, torch.full_like(gap, float("nan")))
    return gap


# ===========================================
# 3) ROI ∩ (entropy > thr)만 adaptive window 재매칭
# ===========================================
@torch.no_grad()
def _invalid_run_extents(entropy: torch.Tensor, thr: float):
    """
    entropy: [H4,W4]
    반환: a,b, invalid
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
def build_roi_mask(
    H4: int,
    W4: int,
    mode: str,
    u0: float,
    u1: float,
    v0: float,
    v1: float,
    device: torch.device,
) -> torch.Tensor:
    """
    ROI 마스크 생성 (1/4 격자 기준)
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
        if u1f < u0f:
            u0f, u1f = u1f, u0f
        if v1f < v0f:
            v0f, v1f = v1f, v0f

        u0i = int(math.floor(u0f * W4))
        u1i = int(math.ceil(u1f * W4) - 1)
        v0i = int(math.floor(v0f * H4))
        v1i = int(math.ceil(v1f * H4) - 1)
    else:
        u0i = int(round(u0))
        u1i = int(round(u1))
        v0i = int(round(v0))
        v1i = int(round(v1))
        if u1i < u0i:
            u0i, u1i = u1i, u0i
        if v1i < v0i:
            v0i, v1i = v1i, v0i

    u0i = clamp(u0i, 0, W4 - 1)
    u1i = clamp(u1i, 0, W4 - 1)
    v0i = clamp(v0i, 0, H4 - 1)
    v1i = clamp(v1i, 0, H4 - 1)

    mask = torch.zeros(H4, W4, dtype=torch.bool, device=device)
    if (u1i >= u0i) and (v1i >= v0i):
        mask[v0i:v1i + 1, u0i:u1i + 1] = True
    return mask


@torch.no_grad()
def refine_cost_for_uncertain_roi(
    cost_vol: torch.Tensor,
    entropy_before: torch.Tensor,
    ent_thr: float,
    roi_mask: torch.Tensor,
    max_half: int = None,
    ent_T: float = 0.1,
):
    """
    cost_vol: [D+1,H4,W4]
    entropy_before: [H4,W4]
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
    agg = torch.where(den > 0, agg, torch.full_like(agg, float("-inf")))

    # ROI∩invalid 위치만 치환
    cost_vol_ref = torch.where(refine_mask.unsqueeze(0), agg, cost_vol)

    # 보정 후 엔트로피(전체)
    entropy_after = build_entropy_map(cost_vol_ref, T=ent_T, normalize=True)
    return cost_vol_ref, entropy_after, refine_mask


@torch.no_grad()
def build_union_viz_mask(
    ent_before: torch.Tensor,
    ent_after: torch.Tensor,
    thr: float,
    roi_mask: torch.Tensor,
):
    """
    시각화용 union mask
    """
    m0 = (ent_before <= float(thr))
    m1 = (ent_after  <= float(thr)) & roi_mask
    return (m0 | m1)


# ===========================================
# 4) 시각화 유틸
# ===========================================
def upsample_nearest_4x(map_2d: np.ndarray) -> np.ndarray:
    """
    [H4,W4] → [H4*4, W4*4] 최근접 업샘플(시각화용)
    """
    return np.kron(map_2d, np.ones((4, 4), dtype=map_2d.dtype))


def _get_transparent_disp_cmap():
    # turbo가 없거나 with_extremes 미지원 대비
    try:
        cmap = cm.get_cmap("turbo")
    except Exception:
        cmap = cm.get_cmap("plasma")
    if hasattr(cmap, "with_extremes"):
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


def _ensure_dirs(root: Path, *names: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for n in names:
        p = root / n
        p.mkdir(parents=True, exist_ok=True)
        out[n] = p
    return out


def _save_map(
    path: Path,
    arr: np.ndarray,
    vmin=None,
    vmax=None,
    title=None,
    cmap="viridis",
    with_colorbar=True,
    overlay_img: np.ndarray = None,
    alpha=0.55,
):
    """
    (overlay가 필요한 경우에만 사용하는 기본 시각화 함수)
    """
    plt.figure(figsize=(7, 7))
    if overlay_img is not None:
        plt.imshow(overlay_img)
        im = plt.imshow(arr, vmin=vmin, vmax=vmax, alpha=alpha, cmap=cmap)
    else:
        im = plt.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.axis("off")
    if title:
        plt.title(title)
    if with_colorbar:
        _ = plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(str(path), bbox_inches="tight", pad_inches=0.01)
    plt.close()


def save_colormap_png_with_colorbar_auto_range(
    path,
    np_array,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_name: str = "magma",
    label: str = "",
    bg_color: str = "#1e1e1e",
):
    """
    일반 수치맵(예: disparity, entropy)용 컬러 PNG 저장 + colorbar.
    """
    from matplotlib.colors import ListedColormap, to_rgba

    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    arr = np.array(np_array, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        vmin_eff = 0.0 if vmin is None else float(vmin)
        vmax_eff = 1.0 if vmax is None else float(vmax)
    else:
        vmin_eff = float(np.nanmin(arr[finite])) if vmin is None else float(vmin)
        vmax_eff = float(np.nanmax(arr[finite])) if vmax is None else float(vmax)
        vmax_eff = max(vmax_eff, vmin_eff + 1e-6)

    base = plt.get_cmap(cmap_name)
    cmap = ListedColormap(base(np.linspace(0, 1, 256)))
    cmap.set_bad(to_rgba(bg_color))

    H, W = arr.shape
    dpi = 200.0
    figsize = (W / dpi, H / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin_eff, vmax=10)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if label:
        cbar.set_label(label, rotation=270, labelpad=12)

    plt.tight_layout(pad=0.1)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1, facecolor=fig.get_facecolor())
    plt.close(fig)


# -------------------------------------------
# GT 로딩 / 변환 (depth→disp or disp 원본)
# -------------------------------------------
def _parse_invalid_values(s: str):
    return [int(v) for v in s.split(",") if v.strip() != ""]


def _resize_depth_nearest(depth_m: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    Ht, Wt = target_hw
    if depth_m.shape[0] == Ht and depth_m.shape[1] == Wt:
        return depth_m
    return np.array(Image.fromarray(depth_m).resize((Wt, Ht), resample=Image.NEAREST))


def load_gt_disp_px(
    gt_path: Path,
    mode: str,
    depth_scale: float,
    disp_scale: float,
    focal_px: float,
    baseline_m: float,
    target_hw: Tuple[int, int],
    invalid_values=(0, 65535),
    max_depth_m: float = 200.0,
) -> np.ndarray:
    """
    GT를 disparity(px)로 반환. NaN 허용.
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
# GT 다운샘플링 (full-res disparity → 1/4-grid index units)
# -------------------------------------------
def downsample_disp_to_quarter_index(disp_px: np.ndarray, H4: int, W4: int) -> np.ndarray:
    """
    full-res disparity(px) [H,W] -> [H4,W4] in index units (1 index ≈ 4px).
    H4, W4 는 backbone feature 크기와 동일해야 함.
    """
    H, W = disp_px.shape
    assert H == H4 * 4 and W == W4 * 4, \
        f"GT size {H}x{W} not compatible with 1/4 grid size {H4}x{W4}."
    disp = disp_px.reshape(H4, 4, W4, 4)  # [H4,4,W4,4]
    disp_block_px = np.nanmean(np.nanmean(disp, axis=3), axis=1)  # [H4,W4]
    disp_block_idx = disp_block_px / 4.0  # px → index
    return disp_block_idx.astype(np.float32)


# -------------------------------------------
# 1/4-grid 에서 EPE / D1 계산 (index units)
# -------------------------------------------
def compute_epe_d1_quarter(
    pred_idx: np.ndarray,
    gt_idx: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
):
    """
    1/4-grid (index units)에서 EPE와 D1 계산.
    - pred_idx, gt_idx: [H4,W4], disparity in 'index units' (1 index ≈ 4px)
    - valid_mask: 평가에 포함할 픽셀 (예: pseudo-label 마스크). None이면 GT valid만 사용.
    - EPE: index 단위 절대 오차 평균
    - D1: KITTI 기준 3px, 5% 를 index 단위로 환산 (3px ≈ 0.75 index)
    """
    assert pred_idx.shape == gt_idx.shape, "pred/gt must share shape"

    gt_valid = np.isfinite(gt_idx) & (gt_idx > 0.0)
    if valid_mask is not None:
        valid = gt_valid & valid_mask
    else:
        valid = gt_valid

    n_valid = int(valid.sum())
    if n_valid == 0:
        return float("nan"), float("nan"), 0, 0.0, 0

    pred = pred_idx.astype(np.float32)
    gt   = gt_idx.astype(np.float32)

    diff = (pred - gt)[valid]  # index units
    abs_diff = np.abs(diff)    # index units

    # EPE (index units)
    sum_abs = float(abs_diff.sum())
    epe = sum_abs / n_valid

    # D1: |err_px| > 3px & |err_px| / gt_px > 0.05
    # index→px 변환에서 scale은 서로 약분되므로,
    # 3px ≈ 3/4 index threshold 만 신경 쓰면 됨.
    tau_abs_idx = 3.0 / 4.0
    gt_vals = gt[valid]  # index units
    bad = (abs_diff > tau_abs_idx) & (abs_diff / gt_vals > 0.05)
    bad_count = int(bad.sum())
    d1 = bad_count / n_valid

    return epe, d1, n_valid, sum_abs, bad_count


# -------------------------------------------
# 개별 PNG로 저장 (pred/gt/error/entropy 등)
# -------------------------------------------
def save_cmp_panels_separate(
    left_img_pil: Image.Image,
    disp_pred_cell: np.ndarray,   # [H4,W4] in 1/4-index units
    fill_mask_q: np.ndarray,      # [H4,W4] bool
    disp_gt_px_full: np.ndarray,  # [H,W] in px (NaN 허용)
    ent_before_q: np.ndarray,     # [H4,W4] (0..1)
    ent_after_q: np.ndarray,      # [H4,W4] (0..1)
    max_disp_cell: int,
    out_root: Path,
    stem: str,
):
    out = _ensure_dirs(
        out_root,
        "pred_overlay_all", "pred_overlay_filled", "gt_overlay",
        "pred_disp_px", "gt_disp_px", "error_abs_px",
        "entropy_before", "entropy_after", "fill_mask",
    )

    img_np = np.asarray(left_img_pil)

    # Pred @ full-res(index) + Fill-mask 업샘플
    disp_pred_idx_full = upsample_nearest_4x(disp_pred_cell).astype(np.float32)  # [H,W]
    mask_full = upsample_nearest_4x(fill_mask_q.astype(np.uint8)).astype(bool)   # [H,W]

    # Entropy upsample to full-res (보기 편하게)
    ent_b_full = upsample_nearest_4x(ent_before_q).astype(np.float32)  # [H,W]
    ent_a_full = upsample_nearest_4x(ent_after_q).astype(np.float32)

    # 스케일
    vmax_idx = float(max_disp_cell)
    cmap_disp = _get_transparent_disp_cmap()

    # 1) Pred overlay (전체) — index space
    p = out["pred_overlay_all"] / f"{stem}.png"
    _save_map(
        p, disp_pred_idx_full, vmin=0, vmax=vmax_idx, title=None,
        cmap=cmap_disp, with_colorbar=True, overlay_img=img_np, alpha=0.55,
    )

    # 4) Pred disp(index) — colormap
    p = out["pred_disp_px"] / f"{stem}.png"
    save_colormap_png_with_colorbar_auto_range(
        p,
        disp_pred_idx_full,
        vmin=0.0,
        vmax=vmax_idx,
        cmap_name="magma",
        label="Pred disparity (index)",
        bg_color="#1e1e1e",
    )

    # 5) GT disp(px) — 참고용
    p = out["gt_disp_px"] / f"{stem}.png"
    save_colormap_png_with_colorbar_auto_range(
        p,
        disp_gt_px_full,
        vmin=0.0,
        vmax=vmax_idx,
        cmap_name="magma",
        label="GT disparity (px)",
        bg_color="#1e1e1e",
    )

    # 7) entropy before (0..1)
    p = out["entropy_before"] / f"{stem}.png"
    save_colormap_png_with_colorbar_auto_range(
        p,
        ent_b_full,
        vmin=0.0,
        vmax=1.0,
        cmap_name="magma",
        label="Entropy (before)",
        bg_color="#1e1e1e",
    )

    # 8) entropy after (0..1)
    p = out["entropy_after"] / f"{stem}.png"
    save_colormap_png_with_colorbar_auto_range(
        p,
        ent_a_full,
        vmin=0.0,
        vmax=1.0,
        cmap_name="magma",
        label="Entropy (after)",
        bg_color="#1e1e1e",
    )

    print(f"[Saved all panels for] {stem}")


# -----------------------------
# StereoModel 기반 채움 + GT 비교 + 정량평가 파이프라인
# -----------------------------
@torch.no_grad()
def process_pair_and_viz(
    backbone: StereoFeatureBackbone,
    left_pil: Image.Image,
    right_pil: Image.Image,
    stem: str,
    args,
):
    # 원본 크기
    W, H = left_pil.size

    # 텐서 변환
    xL = pil_to_tensor(left_pil)  # [1,3,H,W]
    xR = pil_to_tensor(right_pil)

    # 1/4 특징
    featL, featR = backbone.extract_quarter_features(xL, xR)  # [H4,W4,C]
    H4, W4, C = featL.shape

    if H4 * 4 != H or W4 * 4 != W:
        print(f"[WARN] feature size *4 != image size for {stem}: feat={H4}x{W4}, img={H}x{W}")

    # 코스트볼륨 & 엔트로피(before)
    cost_vol = build_cost_volume(featL, featR, args.max_disp)
    ent_before = build_entropy_map(cost_vol, T=args.ent_T, normalize=True)

    # ROI
    _, H4c, W4c = cost_vol.shape
    assert H4c == H4 and W4c == W4
    roi_mask = build_roi_mask(
        H4, W4, args.roi_mode,
        args.roi_u0, args.roi_u1, args.roi_v0, args.roi_v1,
        device=ent_before.device,
    )

    # Adaptive window refine (ROI∩invalid만 보정)
    cost_vol_ref, ent_after, refine_mask = refine_cost_for_uncertain_roi(
        cost_vol, ent_before, ent_thr=args.ent_vis_thr,
        roi_mask=roi_mask, max_half=args.win_half_max, ent_T=args.ent_T,
    )

    # base/teacher disparity (index units)
    disp_base_cell, _    = argmax_disparity(cost_vol)      # refine 전
    disp_teacher_cell, _ = argmax_disparity(cost_vol_ref)  # refine 후

    # ---- FILL: ROI∩invalid만 teacher로 덮어쓰기 ----
    fill_mask = refine_mask  # ROI ∩ invalid

    # (1) argmax(before) / argmax(after) 비교 mask
    same_argmax_mask = (disp_base_cell == disp_teacher_cell)  # [H4,W4]

    # (2) float 변환
    disp_base_cell_f    = disp_base_cell.to(torch.float32)
    disp_teacher_cell_f = disp_teacher_cell.to(torch.float32)

    # ---- BEFORE: refine/채움 없는 순수 argmax ----
    disp_before_cell = disp_base_cell_f.clone()

    # (3) 기본 방식: ROI∩invalid 전체에 teacher 채움
    disp_filled_cell = disp_base_cell_f.clone()
    disp_filled_cell[fill_mask] = disp_teacher_cell_f[fill_mask]

    # (4) argmax(before/after)가 같은 위치만 teacher로 채우는 버전
    fill_mask_same = fill_mask & same_argmax_mask
    disp_filled_same_cell = disp_base_cell_f.clone()
    disp_filled_same_cell[fill_mask_same] = disp_teacher_cell_f[fill_mask_same]

    # (5) 두 AFTER disp가 같은 픽셀만 값 유지, 나머지는 0
    equal_mask_cell = (disp_filled_cell == disp_filled_same_cell)  # [H4,W4]
    disp_equal_cell = torch.where(
        equal_mask_cell,
        disp_filled_cell,
        torch.zeros_like(disp_filled_cell),
    )  # 나머지는 0

    # numpy로 변환
    disp_before_np       = disp_before_cell.cpu().numpy().astype(np.float32)
    disp_filled_np       = disp_filled_cell.cpu().numpy().astype(np.float32)
    disp_filled_same_np  = disp_filled_same_cell.cpu().numpy().astype(np.float32)
    disp_equal_np        = disp_equal_cell.cpu().numpy().astype(np.float32)

    fill_mask_np         = fill_mask.cpu().numpy().astype(bool)
    fill_mask_same_np    = fill_mask_same.cpu().numpy().astype(bool)
    equal_mask_np        = equal_mask_cell.cpu().numpy().astype(bool)

    ent_before_np        = ent_before.cpu().numpy().astype(np.float32)
    ent_after_np         = ent_after.cpu().numpy().astype(np.float32)

    # ---- GT depth/disp → disparity(px) ----
    depth_path = find_depth_for_left(Path(args.gt_depth_dir), stem)
    if depth_path is None:
        print(f"[Skip] GT depth not found for {stem}")
        return None
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

    # ---- GT를 1/4-grid index units로 다운샘플 ----
    disp_gt_idx = downsample_disp_to_quarter_index(disp_gt_px, H4, W4)  # [H4,W4]

    # ---- (0) BEFORE: cost volume에서 바로 argmax ----
    out_root_before = Path(args.out_dir) / "before"
    save_cmp_panels_separate(
        left_img_pil=left_pil,
        disp_pred_cell=disp_before_np,
        fill_mask_q=fill_mask_np,
        disp_gt_px_full=disp_gt_px,
        ent_before_q=ent_before_np,
        ent_after_q=ent_before_np,
        max_disp_cell=args.max_disp,
        out_root=out_root_before,
        stem=stem,
    )

    # ---- (1) AFTER 기본 FILL 결과 ----
    out_root = Path(args.out_dir)
    save_cmp_panels_separate(
        left_img_pil=left_pil,
        disp_pred_cell=disp_filled_np,
        fill_mask_q=fill_mask_np,
        disp_gt_px_full=disp_gt_px,
        ent_before_q=ent_before_np,
        ent_after_q=ent_after_np,
        max_disp_cell=args.max_disp,
        out_root=out_root,
        stem=stem,
    )

    # ---- (2) AFTER + argmax same 결과 ----
    out_root_same = Path(args.out_dir) / "argmax_same"
    save_cmp_panels_separate(
        left_img_pil=left_pil,
        disp_pred_cell=disp_filled_same_np,
        fill_mask_q=fill_mask_same_np,
        disp_gt_px_full=disp_gt_px,
        ent_before_q=ent_before_np,
        ent_after_q=ent_after_np,
        max_disp_cell=args.max_disp,
        out_root=out_root_same,
        stem=stem,
    )

    # ---- (3) 두 AFTER disp가 같은 픽셀만 유지, 나머지는 0인 결과 (= results) ----
    out_root_res = Path(args.out_dir) / "results"
    save_cmp_panels_separate(
        left_img_pil=left_pil,
        disp_pred_cell=disp_equal_np,
        fill_mask_q=equal_mask_np,
        disp_gt_px_full=disp_gt_px,
        ent_before_q=ent_before_np,
        ent_after_q=ent_after_np,
        max_disp_cell=args.max_disp,
        out_root=out_root_res,
        stem=stem,
    )

    # ---- results 에 대한 1/4-grid 정량 평가 (index 단위) ----
    epe_res, d1_res, n_valid_res, sum_abs_res, bad_cnt_res = compute_epe_d1_quarter(
        pred_idx=disp_equal_np,
        gt_idx=disp_gt_idx,
        valid_mask=equal_mask_np,
    )

    metrics = {
        "stem": stem,
        "epe": epe_res,          # index units
        "d1": d1_res,            # ratio (0~1)
        "valid": n_valid_res,
        "sum_abs": sum_abs_res,  # sum of |err| in index units
        "bad_count": bad_cnt_res,
    }
    return metrics


def find_right_for_left(right_dir: Path, left_stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        cand = right_dir / f"{left_stem}{ext}"
        if cand.exists():
            return cand
    return None


def find_depth_for_left(gt_depth_dir: Path, left_stem: str) -> Optional[Path]:
    exts = [".png", ".tif", ".tiff", ".npy", ".npz"]
    for ext in exts:
        cand = gt_depth_dir / f"{left_stem}{ext}"
        if cand.exists():
            return cand
    return None


# =========================================================
# metrics txt 저장
# =========================================================
def write_metrics_txt(metrics_list: List[Dict], out_dir: Path, filename: str = "results_eval_quarter.txt"):
    """
    metrics_list: 각 원소는
      {
        "stem": str,
        "epe": float,        # 1/4-grid index units
        "d1": float,         # ratio
        "valid": int,
        "sum_abs": float,    # sum of |err| in index units
        "bad_count": int,
      }
    파일 포맷:
      <stem> <epe> <d1>
      ...
      mean <mean_epe> <mean_d1>
    """
    if not metrics_list:
        print("[Eval] No metrics to write.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / filename

    total_valid = sum(m.get("valid", 0) for m in metrics_list)
    total_sum_abs = sum(m.get("sum_abs", 0.0) for m in metrics_list)
    total_bad = sum(m.get("bad_count", 0) for m in metrics_list)

    with open(eval_path, "w") as f:
        # per-image
        for m in metrics_list:
            stem = m.get("stem", "unknown")
            valid = m.get("valid", 0)
            epe = m.get("epe", float("nan"))
            d1  = m.get("d1", float("nan"))

            if valid <= 0 or not np.isfinite(epe) or not np.isfinite(d1):
                epe_str = "nan"
                d1_str  = "nan"
            else:
                epe_str = f"{epe:.6f}"
                d1_str  = f"{d1:.6f}"

            f.write(f"{stem} {epe_str} {d1_str}\n")

        # global mean (pixel-weighted)
        if total_valid > 0:
            mean_epe = total_sum_abs / total_valid
            mean_d1  = total_bad / total_valid
            f.write(f"mean {mean_epe:.6f} {mean_d1:.6f}\n")
        else:
            f.write("mean nan nan\n")

    print(f"[Eval] Wrote metrics to {eval_path}")


# =========================================================
# argparse
# =========================================================
def get_args():
    p = argparse.ArgumentParser(
        "Stereo pseudo-label (StereoModel backbone) — 1/4 cost-volume + entropy ROI fill vs GT — save panels + metrics"
    )
    # 입력(단일/디렉터리)
    p.add_argument("--left",  type=str, default=None)
    p.add_argument("--right", type=str, default=None)
    p.add_argument("--left_dir",  type=str, default="/home/jaejun/dataset/MS2/sync_data/tester/rgb/img_left")
    p.add_argument("--right_dir", type=str, default="/home/jaejun/dataset/MS2/sync_data/tester/rgb/img_right")
    p.add_argument("--glob", type=str, default="*.png")

    # Cost-volume 설정
    p.add_argument("--max_disp", type=int, default=14, help="1/4-grid max disparity (inclusive)")

    # Backbone 설정
    p.add_argument("--backbone_ckpt", type=str, required=True, help="vit_cn.StereoModel 학습 체크포인트(.pth)")
    p.add_argument("--pad_mult", type=int, default=16, help="StereoModel 입력 패딩 배수 (학습과 동일 값 사용 권장, 예: 16)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # 엔트로피/ROI 파라미터
    p.add_argument("--ent_T", type=float, default=0.1)
    p.add_argument("--ent_vis_thr", type=float, default=0.6)
    p.add_argument("--roi_mode", type=str, default="frac", choices=["frac", "abs4"])
    p.add_argument("--roi_u0", type=float, default=0.0)
    p.add_argument("--roi_u1", type=float, default=1.0)
    p.add_argument("--roi_v0", type=float, default=0.0)
    p.add_argument("--roi_v1", type=float, default=1.0)
    p.add_argument("--win_half_max", type=int, default=48)

    # GT/캘리브
    p.add_argument("--gt_depth_dir",  type=str, default="/home/jaejun/dataset/MS2/proj_depth/tester/rgb/depth_filtered")
    p.add_argument("--gt_mode",       type=str, default="depth", choices=["depth", "disp"])
    p.add_argument("--gt_depth_scale", type=float, default=256.0)
    p.add_argument("--gt_disp_scale",  type=float, default=1.0)
    p.add_argument(
        "--gt_invalid_values", type=str, default="0,65535",
        help="raw GT에서 무효로 볼 값들(쉼표 구분). 예: 0,65535",
    )
    p.add_argument(
        "--gt_max_depth_m", type=float, default=200.0,
        help="이 값보다 큰 깊이는 무효 처리(0: 비활성)",
    )

    p.add_argument("--focal_px", type=float, default=764.5138549804688)
    p.add_argument("--baseline_m", type=float, default=0.29918420530585865)

    # 출력
    p.add_argument("--out_dir", type=str, default="./log/MS2_tester_rgb_image_pseudo_label_stereo_robotcar_backbone")

    return p.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device)

    # 모드 판별
    dir_mode = (args.left is None and args.right is None and args.left_dir and args.right_dir)
    file_mode = (args.left is not None and args.right is not None)
    assert dir_mode or file_mode, "하나를 선택: (1) --left/--right 또는 (2) --left_dir/--right_dir"

    # StereoModel 백본 로드
    backbone = StereoFeatureBackbone(
        device=device,
        backbone_ckpt=args.backbone_ckpt,
        pad_mult=args.pad_mult,
        verbose=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if file_mode:
        lp = Path(args.left)
        rp = Path(args.right)
        assert lp.exists(), f"Not found: {lp}"
        assert rp.exists(), f"Not found: {rp}"
        L = Image.open(str(lp)).convert("RGB")
        R = Image.open(str(rp)).convert("RGB")
        assert L.size == R.size, f"size mismatch: {L.size} vs {R.size}"
        metrics = process_pair_and_viz(backbone, L, R, lp.stem, args)
        if metrics is not None:
            write_metrics_txt([metrics], out_dir)
        return

    # 디렉터리 모드
    left_dir  = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    assert left_dir.is_dir() and right_dir.is_dir(), "입력 디렉터리 확인"
    Path(args.gt_depth_dir).mkdir(parents=True, exist_ok=True)

    left_files = sorted(left_dir.glob(args.glob))
    assert len(left_files) > 0, f"No files matching {args.glob} in {left_dir}"

    processed, skipped = 0, 0
    metrics_list: List[Dict] = []

    for lp in left_files:
        rp = find_right_for_left(right_dir, lp.stem)
        if rp is None:
            print(f"[Skip] right not found for {lp.name}")
            skipped += 1
            continue

        depth_path = find_depth_for_left(Path(args.gt_depth_dir), lp.stem)
        if depth_path is None:
            print(f"[Skip] gt depth not found for {lp.name}")
            skipped += 1
            continue

        L = Image.open(str(lp)).convert("RGB")
        R = Image.open(str(rp)).convert("RGB")
        if L.size != R.size:
            print(f"[Skip] size mismatch: {lp.name} vs {rp.name}")
            skipped += 1
            continue

        metrics = process_pair_and_viz(backbone, L, R, lp.stem, args)
        if metrics is None:
            skipped += 1
            continue

        metrics_list.append(metrics)
        processed += 1

    # metrics 파일 작성
    write_metrics_txt(metrics_list, out_dir)
    print(f"[Done] processed={processed}, skipped={skipped}, out_dir={out_dir.resolve()}")


if __name__ == "__main__":
    main()
