# seed_1of8_loss_safe.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

@torch.no_grad()
def top1_top2_margin_patch(prob_5d: torch.Tensor) -> torch.Tensor:
    pp = prob_5d.squeeze(1).clamp_min(1e-8)     # [B,D+1,H,W]
    top2 = torch.topk(pp, k=2, dim=1).values    # [B,2,H,W]
    return (top2[:, 0] - top2[:, 1]).unsqueeze(1)

def make_norm_rect_mask_like(base: torch.Tensor,
                             y_min: float = 0.0, y_max: float = 1.0,
                             x_min: float = 0.0, x_max: float = 1.0) -> torch.Tensor:
    """
    base: [B,1,H,W] 같은 크기의 텐서를 기준으로 정규화(0~1) 사각형 마스크 생성
    좌상단 원점(0,0), y는 아래로 증가. H,W는 보통 1/8 패치 해상도.
    return: [B,1,H,W] (0/1 float)
    """
    B, C, H, W = base.shape
    # 범위 클램프 및 정규화
    y_min = float(max(0.0, min(1.0, y_min)))
    y_max = float(max(0.0, min(1.0, y_max)))
    x_min = float(max(0.0, min(1.0, x_min)))
    x_max = float(max(0.0, min(1.0, x_max)))
    if y_max < y_min: y_min, y_max = y_max, y_min
    if x_max < x_min: x_min, x_max = x_max, x_min

    # 정규화→인덱스 (start=floor, end=ceil; end는 슬라이스 상한)
    y0 = int(np.floor(y_min * H))
    y1 = int(np.ceil (y_max * H))
    x0 = int(np.floor(x_min * W))
    x1 = int(np.ceil (x_max * W))
    y0 = max(0, min(H, y0)); y1 = max(0, min(H, y1))
    x0 = max(0, min(W, x0)); x1 = max(0, min(W, x1))

    # 최소 1픽셀 보장
    if y0 == y1: y1 = min(H, y0 + 1)
    if x0 == x1: x1 = min(W, x0 + 1)

    mask = base.new_zeros((B, 1, H, W))
    mask[:, :, y0:y1, x0:x1] = 1.0
    return mask

@torch.no_grad()
def build_bad_seed_mask_1of8(disp_soft: torch.Tensor,
                             prob_5d: torch.Tensor,
                             roi_patch: torch.Tensor,
                             low_idx_thr: float = 1.0,
                             high_idx_thr: float = 1.0,
                             conf_thr: float = 0.05,
                             road_ymin: float = None,
                             use_extremes: bool = True,
                             use_conf: bool = True) -> torch.Tensor:
    B, _, H, W = disp_soft.shape
    D = prob_5d.shape[2] - 1
    bad = torch.zeros_like(disp_soft, dtype=torch.bool)
    if use_extremes:
        near_low  = (disp_soft <= float(low_idx_thr))
        near_high = (disp_soft >= float(D) - float(high_idx_thr))
        bad = bad | near_low | near_high
    if use_conf:
        conf = top1_top2_margin_patch(prob_5d)
        bad  = bad | (conf < float(conf_thr))
    if road_ymin is not None and 0.0 < road_ymin < 1.0:
        yy = torch.arange(H, device=disp_soft.device).view(1,1,H,1).float() / float(H)
        bad = bad & (yy >= float(road_ymin))
    return bad & (roi_patch > 0)

@torch.no_grad()
def rowwise_mode_idx_1of8(disp_soft: torch.Tensor,
                          good_mask: torch.Tensor,
                          D: int,
                          bin_size: float = 1.0,
                          min_count: int = 8):
    B, _, H, W = disp_soft.shape
    nbins = int(torch.ceil(torch.tensor(float(D) / max(bin_size, 1e-6))).item()) + 1
    row_mode_idx = torch.full((B,1,H,1), float('nan'), device=disp_soft.device, dtype=disp_soft.dtype)
    row_valid    = torch.zeros((B,1,H,1), device=disp_soft.device, dtype=torch.bool)
    for b in range(B):
        for y in range(H):
            m = good_mask[b,0,y] > 0
            vals = disp_soft[b,0,y, m]
            if vals.numel() < int(min_count):
                continue
            idx  = torch.clamp(torch.round(vals / float(bin_size)).long(), 0, nbins-1)
            hist = torch.bincount(idx, minlength=nbins)
            k = int(torch.argmax(hist))
            row_mode_idx[b,0,y,0] = k * float(bin_size)
            row_valid[b,0,y,0]    = True
    return row_mode_idx, row_valid

def _sharp_expectation_fp32(refined_logits_masked: torch.Tensor, tau: float):
    # FP32에서 softmax → 기대값 계산 (AMP 안전)
    logits32 = (refined_logits_masked.float() / float(tau))
    p32 = torch.softmax(logits32, dim=2)
    Dp1 = refined_logits_masked.shape[2]
    disp_vals32 = torch.arange(0, Dp1, dtype=torch.float32, device=logits32.device).view(1,1,Dp1,1,1)
    disp32 = (p32 * disp_vals32).sum(dim=2)  # [B,1,H,W]
    return disp32.to(refined_logits_masked.dtype)

class SeedAnchorHuberLoss(nn.Module):
    """
    - 마스크 밖은 연산하지 않음(완전 마스킹) → NaN * 0 방지
    - softmax/기대값은 FP32로 계산 → AMP 안정
    """
    def __init__(self, tau: float = 0.3, huber_delta: float = 0.5, detach_seed: bool = True):
        super().__init__()
        self.tau = float(tau)
        self.delta = float(huber_delta)
        self.detach_seed = bool(detach_seed)

    def forward(self,
                refined_logits_masked: torch.Tensor,  # [B,1,D+1,H,W]
                seed_idx_map: torch.Tensor,           # [B,1,H,W]
                anchor_mask: torch.Tensor) -> torch.Tensor:  # [B,1,H,W] bool/float
        disp_sharp = _sharp_expectation_fp32(refined_logits_masked, self.tau)  # [B,1,H,W]
        seed = seed_idx_map.detach() if self.detach_seed else seed_idx_map
        mask = anchor_mask.bool()

        # ★ 마스크된 차이만 계산 (나머지는 0으로 채움)
        masked_diff = torch.where(mask, (disp_sharp - seed).abs(), torch.zeros_like(disp_sharp))

        small = (masked_diff < self.delta).float()
        huber = 0.5 * (masked_diff**2) / (self.delta + 1e-6) * small + (masked_diff - 0.5*self.delta) * (1 - small)

        m = mask.float()
        return (huber * m).sum() / (m.sum() + 1e-6)
