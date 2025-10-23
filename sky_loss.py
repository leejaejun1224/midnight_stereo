# sky_loss.py
# -*- coding: utf-8 -*-
"""
Sky prior를 학습/추론에서 함께 쓰기 위한 유틸:
  1) compute_sky_mask_from_disp: (NumPy/OpenCV) 추론 결과 disp에서 sky 마스크 생성 (+옵션 저장)
  2) 학습용 1/2 해상도(업샘플) 마스크 생성 (토치)
  3) SkyZeroLoss: sky 후보를 0으로 유도하는 손실 (업샘플 해상도에서 계산)
     - 옵션: 학습 중 마스크 PNG 자동 저장 (debug_dir, names, step, save_every)
"""

from typing import Optional, Tuple, Dict, List
import os
import numpy as np

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


# -----------------------------------------------------------
# 공통: PNG 저장 유틸
# -----------------------------------------------------------
def _ensure_dir(d: str):
    if d is not None and len(d) > 0:
        os.makedirs(d, exist_ok=True)

def _to_uint8_mask(x: np.ndarray) -> np.ndarray:
    """float/bool/0~1 → 0/255 uint8"""
    if x.dtype != np.uint8:
        x = (x > 0).astype(np.uint8) * 255
    return x

def save_mask_png_np(mask: np.ndarray, path: str) -> None:
    """(H,W) binary/float → PNG 저장"""
    mask_u8 = _to_uint8_mask(mask)
    _ensure_dir(os.path.dirname(path))
    if HAS_CV2:
        cv2.imwrite(path, mask_u8)
    elif HAS_PIL:
        Image.fromarray(mask_u8).save(path)
    else:
        raise RuntimeError("Neither OpenCV nor PIL is available to save PNG.")

def save_mask_png_torch(mask: torch.Tensor, path: str) -> None:
    """torch [H,W] or [1,H,W] or [B,1,H,W] → PNG 저장(배치면 첫 샘플만)"""
    if mask.dim() == 4:
        mask = mask[0, 0]
    elif mask.dim() == 3 and mask.size(0) == 1:
        mask = mask[0]
    mask_np = mask.detach().float().cpu().numpy()
    save_mask_png_np(mask_np, path)

def save_batch_masks(mask: torch.Tensor,
                     debug_dir: str,
                     names: Optional[List[str]] = None,
                     prefix: str = "sky_half",
                     step: Optional[int] = None) -> None:
    """
    mask: [B,1,H,W] (0/1 float)
    names: 파일명 stem 리스트(옵션). 없으면 인덱스 사용.
    step: 전역스텝(옵션). 파일명에 포함.
    """
    _ensure_dir(debug_dir)
    B = mask.size(0)
    for b in range(B):
        stem = names[b] if (names is not None and b < len(names)) else f"b{b:02d}"
        if step is None:
            fname = f"{prefix}_{stem}.png"
        else:
            fname = f"{prefix}_{step:06d}_{stem}.png"
        path = os.path.join(debug_dir, fname)
        save_mask_png_torch(mask[b:b+1], path)


# -----------------------------------------------------------
# (A) Inference용: disp(픽셀 단위) → sky mask (OpenCV, 컨투어 기반)
#     + save_path 옵션으로 바로 PNG 저장 지원
# -----------------------------------------------------------
def compute_sky_mask_from_disp(disp_px: np.ndarray,
                               max_disp_px: float,
                               thr_px: float = 3.0,
                               vmax_ratio: float = 0.6,
                               morph_kernel: int = 11,
                               min_area: int = 1000,
                               save_path: Optional[str] = None) -> Optional[np.ndarray]:
    """
    disp_px:     (H,W) float32, 'disp_px_out' (선택 스케일/업샘플 반영본)
    max_disp_px: 체크포인트의 최대 시차(px)
    thr_px:      near-max 임계. disp >= max_disp_px - thr_px → sky 후보
    vmax_ratio:  컨투어 y_max < H * vmax_ratio여야 sky 인정(기본 0.6)
    morph_kernel:모폴로지 클로징 커널 크기(홀수 권장)
    min_area:    너무 작은 잡영역 제거(픽셀 수)
    save_path:   지정하면 PNG 저장
    return: (H,W) uint8 (sky=255, else=0) or OpenCV 미설치 시 None
    """
    if not HAS_CV2 and not HAS_PIL:
        return None

    H, W = disp_px.shape[:2]

    # 1) near-max 이진화
    thr_val = float(max_disp_px) - float(thr_px)
    bin0 = (disp_px >= thr_val).astype(np.uint8) * 255

    # 2) 모폴로지 클로징
    k = max(3, int(morph_kernel))
    if k % 2 == 0:
        k += 1
    if HAS_CV2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bin1 = cv2.morphologyEx(bin0, cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        # PIL만 있을 때는 간단히 사용: dilation -> erosion 근사 (3x3 2회)
        bin1 = bin0

    # 3) 컨투어 (OpenCV 필요)
    if HAS_CV2:
        _cont = cv2.findContours(bin1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = _cont[0] if len(_cont) == 2 else _cont[1]
    else:
        contours = []

    # 4) 조건 필터 + 채우기
    sky_mask = np.zeros((H, W), dtype=np.uint8)
    if HAS_CV2:
        y_max_lim = int(H * float(vmax_ratio))
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < float(min_area):
                continue
            ys = cnt[:, :, 1].reshape(-1)
            y_min = int(ys.min()) if ys.size else H
            y_max = int(ys.max()) if ys.size else -1
            if (y_min <= 1) and (y_max < y_max_lim):
                cv2.drawContours(sky_mask, [cnt], contourIdx=-1, color=255, thickness=-1)
    else:
        # OpenCV 없으면 near-max 결과(bin1) 자체를 반환
        sky_mask = (bin1 > 0).astype(np.uint8) * 255

    # 저장 옵션
    if save_path is not None:
        save_mask_png_np(sky_mask, save_path)

    return sky_mask


# -----------------------------------------------------------
# (B) 학습용 토치 Morphology (0/1 텐서용)
# -----------------------------------------------------------
@torch.no_grad()
def _binary_dilate(x: torch.Tensor, k: int) -> torch.Tensor:
    pad = k // 2
    return (F.max_pool2d(x, kernel_size=k, stride=1, padding=pad) > 0.0).to(x.dtype)

@torch.no_grad()
def _binary_erode(x: torch.Tensor, k: int) -> torch.Tensor:
    pad = k // 2
    avg = F.avg_pool2d(x, kernel_size=k, stride=1, padding=pad)
    return (avg >= 1.0).to(x.dtype)

@torch.no_grad()
def _binary_close(x: torch.Tensor, k: int) -> torch.Tensor:
    return _binary_erode(_binary_dilate(x, k), k)


# -----------------------------------------------------------
# (C) 학습용: 1/2 해상도(픽셀 단위)에서 sky 후보 마스크 만들기
#       - disp_half_px: [B,1,Hh,Wh], 픽셀 단위(half-resolution)
#       - 조건: near-max(px) & 상단 비율 & (선택) ROI_half
#       - morphology: 닫힘(팽창→침식)
# -----------------------------------------------------------
@torch.no_grad()
def build_sky_mask_half_torch(disp_half_px: torch.Tensor,
                              max_disp_px: int,
                              thr_px: float = 3.0,
                              y_max_ratio: float = 0.6,
                              morph_kernel: int = 7,
                              roi_half: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    반환: [B,1,Hh,Wh] float {0,1}
    """
    B, C, Hh, Wh = disp_half_px.shape
    near_max = (disp_half_px >= (float(max_disp_px) - float(thr_px))).to(disp_half_px.dtype)

    yy = torch.arange(Hh, device=disp_half_px.device, dtype=disp_half_px.dtype).view(1, 1, Hh, 1)
    ymask = (yy < (float(y_max_ratio) * Hh)).to(disp_half_px.dtype)

    m = near_max * ymask
    if roi_half is not None:
        m = m * (roi_half > 0).to(m.dtype)

    k = max(3, int(morph_kernel))
    if (k % 2) == 0:
        k += 1
    m = _binary_close(m, k)  # 0/1 float 유지
    return (m > 0.5).to(disp_half_px.dtype)


# -----------------------------------------------------------
# (D) SkyZeroLoss: 업샘플(기본 1/2) 해상도에서 계산 + PNG 저장 옵션
#     - Huber(|disp_half_px|)  : 픽셀 단위 회귀
#     - CE(-log P(d=0))        : prob을 공간 업샘플해 앵커(옵션)
# -----------------------------------------------------------
def _huber_abs(x: torch.Tensor, delta: float) -> torch.Tensor:
    a = x.abs()
    small = (a < delta).to(x.dtype)
    return 0.5 * (a ** 2) / (delta + 1e-6) * small + (a - 0.5 * delta) * (1.0 - small)

class SkyZeroLoss(nn.Module):
    """
    Sky 영역(상단 near-max)에서 disparity를 0으로 유도하는 손실.
    기본은 1/8 → 1/2 업샘플 해상도에서 계산합니다.

    Args:
      max_disp_px : 최대 시차(px)
      patch_size  : ViT/네트워크 패치 크기 (업샘플 변환에 사용)
      compute_at  : 'half' | 'lo' (기본 half)
      thr_px      : near-max 임계(px). disp >= max_disp_px - thr_px → sky 후보
      y_max_ratio : 상단 영역 비율(0~1). 0.6 → 상단 60%만 sky 후보
      morph_kernel_half : half 해상도 morphology 커널(홀수 권장)
      morph_kernel_lo   : lo(1/8) 해상도 morphology 커널(홀수 권장, 'lo' 모드에서만)
      w_huber     : Huber 회귀 가중치
      huber_delta_px : Huber delta (픽셀 단위, half 모드에서 사용)
      huber_delta_idx: Huber delta (인덱스 단위, lo 모드에서 사용)
      w_ce        : -log P(d=0) 가중치(0이면 비활성) — 공간 업샘플 후 half에서 계산
      detach_mask : 마스크는 gradient 끊음(권장)

      # 디버그 저장
      debug_dir   : 마스크 PNG 저장 디렉터리(옵션)
      save_every  : step % save_every == 0일 때만 저장(0이면 매번 저장)
    """
    def __init__(self,
                 max_disp_px: int,
                 patch_size: int,
                 compute_at: str = 'half',
                 thr_px: float = 3.0,
                 y_max_ratio: float = 0.6,
                 morph_kernel_half: int = 7,
                 morph_kernel_lo: int = 5,
                 w_huber: float = 1.0,
                 huber_delta_px: float = 0.5,
                 huber_delta_idx: float = 0.5,
                 w_ce: float = 0.0,
                 detach_mask: bool = True,
                 debug_dir: Optional[str] = None,
                 save_every: int = 0):
        super().__init__()
        assert max_disp_px % patch_size == 0, "max_disp_px는 patch_size의 배수여야 함"
        self.max_disp_px = int(max_disp_px) // 2
        self.patch_size = int(patch_size)
        self.D = max_disp_px // patch_size
        self.compute_at = compute_at
        self.thr_px = float(thr_px)
        self.y_max_ratio = float(y_max_ratio)
        self.morph_kernel_half = int(morph_kernel_half)
        self.morph_kernel_lo = int(morph_kernel_lo)
        self.w_huber = float(w_huber)
        self.delta_px = float(huber_delta_px)
        self.delta_idx = float(huber_delta_idx)
        self.w_ce = float(w_ce)
        self.detach_mask = bool(detach_mask)

        # debug 저장
        self.debug_dir = debug_dir
        self.save_every = int(save_every)

    def _ensure_half_disp(self, disp_soft: torch.Tensor) -> torch.Tensor:
        """disp_soft(인덱스 단위, 1/8)을 1/2 픽셀 단위로 변환"""
        # 1/8 → 1/2 : ×4 bilinear
        disp_half = F.interpolate(disp_soft, scale_factor=4.0, mode='bilinear', align_corners=False)
        # 인덱스 → half 픽셀 단위: patch_size/2
        return disp_half * float(self.patch_size / 2.0)

    def _maybe_save_mask(self,
                         mask: torch.Tensor,       # [B,1,H,W]
                         prefix: str,
                         names: Optional[List[str]],
                         step: Optional[int]) -> None:
        if self.debug_dir is None:
            return
        if (self.save_every > 0) and (step is not None) and (step % self.save_every != 0):
            return
        save_batch_masks(mask, debug_dir=self.debug_dir, names=names, prefix=prefix, step=step)

    def forward(self,
                prob_5d: torch.Tensor,        # [B,1,D+1,H8,W8]
                disp_soft: torch.Tensor,      # [B,1,H8,W8] (index)
                roi_patch: Optional[torch.Tensor] = None,   # [B,1,H8,W8]
                disp_half_px: Optional[torch.Tensor] = None,# [B,1,H/2,W/2]
                roi_half: Optional[torch.Tensor] = None,    # [B,1,H/2,W/2]
                mask_half_override: Optional[torch.Tensor] = None,
                mask_lo_override: Optional[torch.Tensor] = None,
                # 디버그 저장용
                names: Optional[List[str]] = None,
                step: Optional[int] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert prob_5d.dim() == 5 and prob_5d.size(2) == self.D + 1, "prob_5d shape 확인"

        # -------------------------
        # 모드 A: half (업샘플 해상도) — 권장
        # -------------------------
        if self.compute_at.lower() == 'half':
            # 1) half 해상도 disparity (픽셀 단위)
            if disp_half_px is None:
                disp_half_px = self._ensure_half_disp(disp_soft)  # [B,1,Hh,Wh]

            # 2) ROI half가 없으면 roi_patch를 nearest 업샘플
            if roi_half is None:
                if roi_patch is not None:
                    roi_half = F.interpolate(roi_patch, scale_factor=4.0, mode='nearest')
                else:
                    roi_half = torch.ones_like(disp_half_px)

            # 3) sky 후보 마스크 (half)
            if mask_half_override is None:
                with torch.no_grad():
                    mask_half = build_sky_mask_half_torch(
                        disp_half_px=disp_half_px,
                        max_disp_px=self.max_disp_px,
                        thr_px=self.thr_px,
                        y_max_ratio=self.y_max_ratio,
                        morph_kernel=int(self.morph_kernel_half),
                        roi_half=roi_half
                    )
            else:
                mask_half = mask_half_override
                if roi_half is not None:
                    mask_half = mask_half * (roi_half > 0).to(mask_half.dtype)

            if self.detach_mask:
                mask_half = mask_half.detach()

            # ★ 디버그 PNG 저장
            self._maybe_save_mask(mask_half, prefix="sky_half", names=names, step=step)

            denom = mask_half.sum() + 1e-6

            # 4) Huber(|d(px)|) @ half
            loss_h = torch.tensor(0.0, device=disp_half_px.device, dtype=disp_half_px.dtype)
            if self.w_huber > 0:
                hub = _huber_abs(disp_half_px, self.delta_px)
                loss_h = (hub * mask_half).sum() / denom

            # 5) -log P(d=0) @ half  (prob을 공간 업샘플)
            loss_ce = torch.tensor(0.0, device=disp_half_px.device, dtype=disp_half_px.dtype)
            if self.w_ce > 0:
                p0_lo = prob_5d.squeeze(1)[:, 0:1, :, :]    # [B,1,H8,W8]
                p0_hi = F.interpolate(p0_lo, size=disp_half_px.shape[-2:], mode='bilinear', align_corners=False).clamp_min(1e-8)
                ce_map = -p0_hi.log()
                loss_ce = (ce_map * mask_half).sum() / denom

            loss = self.w_huber * loss_h + self.w_ce * loss_ce
            aux = {
                "mask_half": mask_half,
                "count": mask_half.sum(),
                "ratio": mask_half.mean(),
                "huber": loss_h.detach(),
                "ce": loss_ce.detach()
            }
            return loss, aux

        # -------------------------
        # 모드 B: lo (1/8) — 필요시 사용
        # -------------------------
        else:
            B, _, H8, W8 = disp_soft.shape
            # 1) sky 후보(1/8, 인덱스 단위 near-max)
            thr_idx = float(self.thr_px) / float(self.patch_size)
            if mask_lo_override is None:
                with torch.no_grad():
                    near_max = (disp_soft >= (float(self.D) - thr_idx)).to(disp_soft.dtype)
                    yy = torch.arange(H8, device=disp_soft.device, dtype=disp_soft.dtype).view(1, 1, H8, 1)
                    ymask = (yy < (self.y_max_ratio * H8)).to(disp_soft.dtype)
                    mask_lo = near_max * ymask
                    if roi_patch is not None:
                        mask_lo = mask_lo * (roi_patch > 0).to(mask_lo.dtype)
                    k = max(3, int(self.morph_kernel_lo))
                    if (k % 2) == 0:
                        k += 1
                    mask_lo = _binary_close(mask_lo, k)
                    mask_lo = (mask_lo > 0.5).to(disp_soft.dtype)
            else:
                mask_lo = mask_lo_override
                if roi_patch is not None:
                    mask_lo = mask_lo * (roi_patch > 0).to(mask_lo.dtype)

            if self.detach_mask:
                mask_lo = mask_lo.detach()

            # ★ 디버그 PNG 저장
            self._maybe_save_mask(mask_lo, prefix="sky_lo", names=names, step=step)

            denom = mask_lo.sum() + 1e-6

            # 2) Huber(|disp_idx|) @ 1/8
            loss_h = torch.tensor(0.0, device=disp_soft.device, dtype=disp_soft.dtype)
            if self.w_huber > 0:
                hub = _huber_abs(disp_soft, self.delta_idx)
                loss_h = (hub * mask_lo).sum() / denom

            # 3) -log P(d=0) @ 1/8
            loss_ce = torch.tensor(0.0, device=disp_soft.device, dtype=disp_soft.dtype)
            if self.w_ce > 0:
                p0 = prob_5d.squeeze(1)[:, 0:1, :, :].clamp_min(1e-8)
                ce_map = -p0.log()
                loss_ce = (ce_map * mask_lo).sum() / denom

            loss = self.w_huber * loss_h + self.w_ce * loss_ce
            aux = {
                "mask_lo": mask_lo,
                "count": mask_lo.sum(),
                "ratio": mask_lo.mean(),
                "huber": loss_h.detach(),
                "ce": loss_ce.detach()
            }
            return loss, aux
