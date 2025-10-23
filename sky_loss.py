# -*- coding: utf-8 -*-
# sky_loss.py (revised)
"""
1) 1/2 해상도에서 'near-max disparity & 상단 제한'으로 sky 후보 마스크 생성(토치 morphology).
2) 1/2 마스크를 k=patch_size//2 블록 평균으로 1/8로 내릴 때 all-ones 대신 >=(1 - tol) 기준 허용.
3) 1/8 마스크 위치에 seed disparity=0(인덱스 단위)로 Huber anchor.
4) 디버그 PNG 저장(half/1of8).
"""

from typing import Optional, Tuple, Dict, List
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False


# ---------------- util: save ----------------
def _ensure_dir(d: str):
    if d:
        os.makedirs(d, exist_ok=True)

def _to_u8(mask: torch.Tensor) -> "np.ndarray":
    import numpy as np
    m = mask.detach().float().clamp(0, 1).cpu().numpy()
    return (m * 255.0 + 0.5).astype("uint8")

def _save_png(mask: torch.Tensor, path: str):
    _ensure_dir(os.path.dirname(path))
    u8 = _to_u8(mask)
    if HAS_CV2:
        cv2.imwrite(path, u8)
    elif HAS_PIL:
        Image.fromarray(u8).save(path)
    else:
        raise RuntimeError("No OpenCV/PIL to save images.")

def save_batch_mask_png(mask: torch.Tensor, debug_dir: str,
                        names: Optional[List[str]], prefix: str, step: Optional[int]):
    if debug_dir is None:
        return
    _ensure_dir(debug_dir)
    B = mask.shape[0]
    for b in range(B):
        stem = names[b] if (names is not None and b < len(names)) else f"b{b:02d}"
        fn = f"{prefix}_{(step if step is not None else 0):06d}_{stem}.png"
        _save_png(mask[b, :1], os.path.join(debug_dir, fn))


# ---------------- torch morphology -----------
@torch.no_grad()
def _binary_dilate(x: torch.Tensor, k: int) -> torch.Tensor:
    pad = k // 2
    return (F.max_pool2d(x, kernel_size=k, stride=1, padding=pad) > 0).to(x.dtype)

@torch.no_grad()
def _binary_erode(x: torch.Tensor, k: int) -> torch.Tensor:
    pad = k // 2
    avg = F.avg_pool2d(x, kernel_size=k, stride=1, padding=pad)
    return (avg >= 1.0).to(x.dtype)

@torch.no_grad()
def _binary_close(x: torch.Tensor, k: int) -> torch.Tensor:
    return _binary_erode(_binary_dilate(x, k), k)


# ---------------- half mask ------------------
@torch.no_grad()
def build_sky_mask_half(
    disp_half_px: torch.Tensor,           # [B,1,Hh,Wh] 픽셀 단위(@1/2)
    max_disp_px: int,
    thr_px: float = 3.0,
    y_max_ratio: float = 0.6,
    morph_kernel_half: int = 7,
    apply_roi: bool = False,
    roi_half: Optional[torch.Tensor] = None  # [B,1,Hh,Wh] (0/1)
) -> torch.Tensor:
    """
    반환: [B,1,Hh,Wh] 0/1 float
    """
    B, C, Hh, Wh = disp_half_px.shape
    near_max = (disp_half_px >= (float(max_disp_px) - float(thr_px))).to(disp_half_px.dtype)

    yy = torch.arange(Hh, device=disp_half_px.device, dtype=disp_half_px.dtype).view(1,1,Hh,1)
    ymask = (yy < float(y_max_ratio) * Hh).to(disp_half_px.dtype)

    m = near_max * ymask
    if apply_roi and (roi_half is not None):
        # 주의: ROI가 하늘을 0으로 만드는 데이터셋이라면 여기서 0으로 사라집니다.
        m = m * (roi_half > 0).to(m.dtype)

    k = max(3, int(morph_kernel_half))
    if k % 2 == 0:
        k += 1
    m = _binary_close(m, k)
    return (m > 0.5).to(disp_half_px.dtype)


# -------- half -> 1/8 (avg >= 1 - tol) -------
@torch.no_grad()
def downsample_mask_to_1of8_all1_tol(
    mask_half: torch.Tensor,   # [B,1,Hh,Wh] 0/1
    patch_size: int,           # ex) 8
    all1_tol: float = 0.0      # 0이면 '모두 1', 0.05면 '평균>=0.95'
) -> torch.Tensor:
    """
    반환: [B,1,H8,W8] 0/1 float
    """
    k = int(patch_size // 2)
    assert k >= 1, "patch_size//2 must be >= 1"
    B, C, Hh, Wh = mask_half.shape
    assert (Hh % k == 0) and (Wh % k == 0), f"Half resolution must be divisible by {k}."

    avg = F.avg_pool2d(mask_half, kernel_size=k, stride=k)
    thr = max(0.0, 1.0 - float(all1_tol))
    return (avg >= thr).to(mask_half.dtype)


# ------------- sharp expectation --------------
def _sharp_expectation_fp32(refined_logits_masked: torch.Tensor, tau: float) -> torch.Tensor:
    logits32 = (refined_logits_masked.float() / float(tau))
    p = torch.softmax(logits32, dim=2)              # [B,1,D+1,H,W]
    Dp1 = refined_logits_masked.shape[2]
    disp_vals = torch.arange(0, Dp1, dtype=torch.float32, device=logits32.device).view(1,1,Dp1,1,1)
    disp = (p * disp_vals).sum(dim=2)               # [B,1,H,W] (index 단위)
    return disp.to(refined_logits_masked.dtype)


# ---------------- main loss -------------------
class SkyGridZeroLoss(nn.Module):
    def __init__(self,
                 max_disp_px: int,
                 patch_size: int = 8,
                 thr_px: float = 3.0,
                 y_max_ratio: float = 0.6,
                 morph_kernel_half: int = 7,
                 # 1/2->1/8 다운샘플 허용 오차
                 all1_tol: float = 0.0,
                 # 빈 마스크일 때 완화 옵션
                 empty_fallback: bool = True,
                 fallback_add_thr_px: float = 3.0,
                 fallback_increase_kernel: int = 4,
                 fallback_all1_tol: float = 0.05,
                 # ROI 적용 여부(기본 False: 하늘이 ROI로 지워지지 않게)
                 use_roi_in_half: bool = False,
                 use_roi_in_1of8: bool = False,
                 # 샤프닝/허버
                 tau_sharp: float = 0.30,
                 huber_delta_idx: float = 0.50,
                 detach_mask: bool = True,
                 # 디버그 저장
                 debug_dir: Optional[str] = None,
                 save_every: int = 0):
        super().__init__()
        assert max_disp_px % patch_size == 0, "max_disp_px는 patch_size의 배수여야 합니다."
        self.max_disp_px = int(max_disp_px) // 2
        self.patch_size  = int(patch_size)
        self.D           = self.max_disp_px // self.patch_size
        self.thr_px      = float(thr_px)
        self.y_max_ratio = float(y_max_ratio)
        self.morph_kernel_half = int(morph_kernel_half)

        self.all1_tol = float(all1_tol)
        self.empty_fb = bool(empty_fallback)
        self.fb_add_thr = float(fallback_add_thr_px)
        self.fb_increase_k = int(fallback_increase_kernel)
        self.fb_all1_tol = float(fallback_all1_tol)

        self.use_roi_half  = bool(use_roi_in_half)
        self.use_roi_1of8  = bool(use_roi_in_1of8)

        self.tau       = float(tau_sharp)
        self.delta_idx = float(huber_delta_idx)
        self.detach_mask = bool(detach_mask)

        self.debug_dir   = debug_dir
        self.save_every  = int(save_every)

    def _maybe_save(self, mask: torch.Tensor, prefix: str,
                    names: Optional[List[str]], step: Optional[int]):
        if self.debug_dir is None:
            return
        if (self.save_every > 0) and (step is not None) and (step % self.save_every != 0):
            return
        save_batch_mask_png(mask, self.debug_dir, names, prefix, step)

    def _build_masks(self,
                     disp_half_px: torch.Tensor,
                     roi_half: Optional[torch.Tensor],
                     roi_patch: Optional[torch.Tensor],
                     names: Optional[List[str]],
                     step: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # 1) half
        mask_half = build_sky_mask_half(
            disp_half_px=disp_half_px,
            max_disp_px=self.max_disp_px,
            thr_px=self.thr_px,
            y_max_ratio=self.y_max_ratio,
            morph_kernel_half=self.morph_kernel_half,
            apply_roi=self.use_roi_half,
            roi_half=roi_half
        )
        if self.detach_mask:
            mask_half = mask_half.detach()

        # 2) 1/8
        mask_1of8 = downsample_mask_to_1of8_all1_tol(mask_half, self.patch_size, self.all1_tol)
        if self.use_roi_1of8 and (roi_patch is not None):
            mask_1of8 = mask_1of8 * (roi_patch > 0).to(mask_1of8.dtype)

        if self.detach_mask:
            mask_1of8 = mask_1of8.detach()

        # 비면 완화 시도
        if self.empty_fb and (mask_1of8.sum() < 1):
            mask_half_fb = build_sky_mask_half(
                disp_half_px=disp_half_px,
                max_disp_px=self.max_disp_px,
                thr_px=self.thr_px + self.fb_add_thr,
                y_max_ratio=self.y_max_ratio,
                morph_kernel_half=self.morph_kernel_half + self.fb_increase_k,
                apply_roi=self.use_roi_half,
                roi_half=roi_half
            )
            mask_1of8_fb = downsample_mask_to_1of8_all1_tol(mask_half_fb, self.patch_size, self.fb_all1_tol)
            if self.use_roi_1of8 and (roi_patch is not None):
                mask_1of8_fb = mask_1of8_fb * (roi_patch > 0).to(mask_1of8_fb.dtype)
            if mask_1of8_fb.sum() > mask_1of8.sum():
                mask_half, mask_1of8 = mask_half_fb, mask_1of8_fb

        # save
        self._maybe_save(mask_half,  "sky_half", names, step)
        self._maybe_save(mask_1of8,  "sky_1of8", names, step)

        aux = {
            "mask_half": mask_half,
            "mask_1of8": mask_1of8,
            "count_half": mask_half.sum().detach(),
            "ratio_half": mask_half.mean().detach(),
            "count_1of8": mask_1of8.sum().detach(),
            "ratio_1of8": mask_1of8.mean().detach(),
        }
        return mask_half, mask_1of8, aux

    def forward(self,
                refined_logits_masked: torch.Tensor,   # [B,1,D+1,H8,W8]
                disp_half_px: torch.Tensor,            # [B,1,H/2,W/2]
                roi_half: Optional[torch.Tensor] = None,  # [B,1,H/2,W/2]
                roi_patch: Optional[torch.Tensor] = None, # [B,1,H8,W8]
                names: Optional[List[str]] = None,
                step: Optional[int] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # mask 만들기
        _, mask_1of8, aux = self._build_masks(disp_half_px, roi_half, roi_patch, names, step)

        denom = mask_1of8.sum() + 1e-6

        # 샤프닝 기대값(인덱스) vs seed(0)
        disp_sharp_idx = _sharp_expectation_fp32(refined_logits_masked, self.tau)  # [B,1,H8,W8]
        seed_zero = torch.zeros_like(disp_sharp_idx)
        diff = torch.where(mask_1of8 > 0,
                           (disp_sharp_idx - seed_zero).abs(),
                           torch.zeros_like(disp_sharp_idx))

        small = (diff < self.delta_idx).to(diff.dtype)
        huber = 0.5 * (diff**2) / (self.delta_idx + 1e-6) * small + (diff - 0.5*self.delta_idx) * (1.0 - small)
        loss_huber = (huber * mask_1of8).sum() / denom

        aux["huber"] = loss_huber.detach()
        return loss_huber, aux
