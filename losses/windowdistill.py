# -*- coding: utf-8 -*-
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn


# ============================================================
# 1) Batched Cost Volume (from interleaved 1/4 features)
#    FqL, FqR: [B,H4,W4,C]  (L2-normalized; cosine dot)
#    return:   [B, D+1, H4, W4]  (invalid = -inf)
# ============================================================
@torch.no_grad()
def build_cost_volume_from_feats(FqL: torch.Tensor,
                                 FqR: torch.Tensor,
                                 max_disp: int) -> torch.Tensor:
    assert FqL.shape == FqR.shape and FqL.dim() == 4, "Fq must be [B,H4,W4,C]"
    B, H4, W4, C = FqL.shape
    D = int(max_disp)
    device, dtype = FqL.device, FqL.dtype

    cv = torch.full((B, D+1, H4, W4), float('-inf'), device=device, dtype=dtype)

    # d = 0
    sim0 = (FqL * FqR).sum(dim=-1)               # [B,H4,W4]
    cv[:, 0] = sim0

    # d > 0
    for d in range(1, D+1):
        left  = FqL[:, :, d:, :]                 # [B,H4,W4-d,C]
        right = FqR[:, :, :-d, :]                # [B,H4,W4-d,C]
        sim = (left * right).sum(dim=-1)         # [B,H4,W4-d]
        cv[:, d, :, d:] = sim
    return cv


# ============================================================
# 2) Argmax disparity / Entropy / Top-2 gap (batched)
# ============================================================
@torch.no_grad()
def argmax_disparity_batched(cost_vol: torch.Tensor):
    """
    cost_vol: [B, D+1, H4, W4]
    return:
      disp_map: [B,H4,W4] (long),
      peak_sim: [B,H4,W4] (float)
    """
    peak_sim, disp_map = cost_vol.max(dim=1)     # over disparity axis
    return disp_map, peak_sim

@torch.no_grad()
def build_entropy_map_batched(cost_vol: torch.Tensor,
                              T: float = 0.1,
                              eps: float = 1e-8,
                              normalize: bool = True) -> torch.Tensor:
    """
    cost_vol: [B,D+1,H4,W4], invalid = -inf
    return: ent [B,H4,W4] in [0,1] if normalize=True
    """
    m = torch.amax(cost_vol, dim=1, keepdim=True)                      # [B,1,H4,W4]
    logits = (cost_vol - m) / max(T, eps)
    prob = torch.softmax(logits, dim=1)                                # [B,D+1,H4,W4]
    p = prob.clamp_min(eps)
    ent = -(p * p.log()).sum(dim=1)                                    # [B,H4,W4]

    if normalize:
        valid = torch.isfinite(cost_vol).to(p.dtype)                   # [B,D+1,H4,W4]
        Deff = valid.sum(dim=1).clamp_min(1)                           # [B,H4,W4]
        ent = torch.where(Deff > 1, ent / (Deff.log() + eps), torch.zeros_like(ent))
        ent = ent.clamp_(0.0, 1.0)
    return ent

@torch.no_grad()
def build_top2_gap_map_batched(cost_vol: torch.Tensor) -> torch.Tensor:
    """
    cost_vol: [B,D+1,H4,W4]
    return: gap [B,H4,W4] (float); if <2 valid candidates → NaN
    """
    B, Dp1, H4, W4 = cost_vol.shape
    if Dp1 < 2:
        return torch.full((B,H4,W4), float('nan'), device=cost_vol.device, dtype=cost_vol.dtype)

    valid = torch.isfinite(cost_vol)             # [B,D+1,H4,W4]
    Deff  = valid.sum(dim=1)                     # [B,H4,W4]

    vals, idxs = torch.topk(cost_vol, k=2, dim=1)    # [B,2,H4,W4], [B,2,H4,W4]
    d1 = idxs[:,0].to(torch.float32)
    d2 = idxs[:,1].to(torch.float32)
    gap = (d1 - d2).abs()                           # [B,H4,W4]
    gap = torch.where(Deff >= 2, gap, torch.full_like(gap, float('nan')))
    return gap


# ============================================================
# 3) ROI mask & invalid-run extents (single-sample utilities)
# ============================================================
@torch.no_grad()
def build_roi_mask_single(H4: int, W4: int,
                          mode: str,
                          u0: float, u1: float,
                          v0: float, v1: float,
                          device: torch.device) -> torch.Tensor:
    """
    return: [H4,W4] bool
    mode='frac'  : u0,u1,v0,v1 ∈ [0,1] (비율)
    mode='abs4'  : 1/4-grid 인덱스(inclusive)
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
        u0i, u1i = int(round(u0)), int(round(u1))
        v0i, v1i = int(round(v0)), int(round(v1))
        if u1i < u0i: u0i, u1i = u1i, u0i
        if v1i < v0i: v0i, v1i = v1i, v0i

    u0i = clamp(u0i, 0, W4-1); u1i = clamp(u1i, 0, W4-1)
    v0i = clamp(v0i, 0, H4-1); v1i = clamp(v1i, 0, H4-1)

    m = torch.zeros(H4, W4, dtype=torch.bool, device=device)
    if (u1i >= u0i) and (v1i >= v0i):
        m[v0i:v1i+1, u0i:u1i+1] = True
    return m

@torch.no_grad()
def _invalid_run_extents_single(entropy: torch.Tensor, thr: float):
    """
    entropy: [H4,W4]  → returns a,b,invalid (all [H4,W4], long/bool)
    각 (y,x)에서 좌/우로 연속 invalid만 지나 첫 valid 직전까지 확장
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

    L = (x - prev_valid - 1).clamp_min(0)
    R = (next_valid - x - 1).clamp_min(0)
    a = (x - L).clamp_min(0).to(torch.long)
    b = (x + R).clamp_max(W4 - 1).to(torch.long)
    return a, b, invalid


# ============================================================
# 4) Adaptive window refine (single-sample) + batched wrapper
# ============================================================
@torch.no_grad()
def refine_cost_single(cost_vol: torch.Tensor,          # [D+1,H4,W4]
                       entropy_before: torch.Tensor,     # [H4,W4]
                       ent_thr: float,
                       roi_mask: torch.Tensor,           # [H4,W4] bool
                       max_half: Optional[int] = None,
                       ent_T: float = 0.1):
    """
    ROI ∩ invalid(>thr) 위치만 window 평균으로 보정
    return:
      cost_vol_ref [D+1,H4,W4],
      entropy_after [H4,W4],
      refine_mask [H4,W4] (ROI∩invalid),
      win_len [H4,W4] (가중치용)
    """
    Dp1, H4, W4 = cost_vol.shape
    device = cost_vol.device

    a, b, invalid = _invalid_run_extents_single(entropy_before, ent_thr)
    refine_mask = (roi_mask & invalid)  # ROI ∩ invalid

    if max_half is not None:
        x = torch.arange(W4, device=device).view(1, W4).expand(H4, -1)
        a = torch.max(a, (x - max_half).clamp_min(0))
        b = torch.min(b, (x + max_half).clamp_max(W4 - 1))
    win_len = (b - a + 1)

    finite = torch.isfinite(cost_vol)
    cv = torch.where(finite, cost_vol, torch.zeros_like(cost_vol))
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

    cost_vol_ref = torch.where(refine_mask.unsqueeze(0), agg, cost_vol)
    entropy_after = build_entropy_map_batched(cost_vol_ref.unsqueeze(0), T=ent_T, normalize=True)[0]
    return cost_vol_ref, entropy_after, refine_mask, win_len

@torch.no_grad()
def refine_cost_batched(cost_vol: torch.Tensor,          # [B,D+1,H4,W4]
                        entropy_before: torch.Tensor,     # [B,H4,W4]
                        roi_mask: torch.Tensor,           # [H4,W4] bool (공통)
                        ent_thr: float,
                        max_half: Optional[int] = None,
                        ent_T: float = 0.1):
    B, Dp1, H4, W4 = cost_vol.shape
    device = cost_vol.device
    cost_ref  = torch.empty_like(cost_vol)
    ent_after = torch.empty(B, H4, W4, device=device, dtype=cost_vol.dtype)
    refine_ms = torch.empty(B, H4, W4, device=device, dtype=torch.bool)
    win_len   = torch.empty(B, H4, W4, device=device, dtype=torch.long)

    for b in range(B):
        c_ref, e_a, m_ref, wlen = refine_cost_single(cost_vol[b], entropy_before[b],
                                                     ent_thr, roi_mask, max_half, ent_T)
        cost_ref[b]  = c_ref
        ent_after[b] = e_a
        refine_ms[b] = m_ref
        win_len[b]   = wlen
    return cost_ref, ent_after, refine_ms, win_len


# ============================================================
# 5) Loss class: Adaptive-Window Teacher Distillation (from Fq)
# ============================================================
class AdaptiveWindowDistillLoss(nn.Module):
    """
    입력:  FqL, FqR ∈ R^{B×H4×W4×C} (L2-normalized interleaved 1/4 features)
           student_out_L: logits [B,D+1,H4,W4] 또는 disp [B,1,H4,W4]/[B,H4,W4]
    내부:  Fq → base cost volume → ROI∩invalid만 adaptive-window로 보정 → teacher 분포/시차
    손실:  KL(soft CE) + Charbonnier 회귀 (픽셀별 가중치)

    ※ 모든 단위는 1/4 grid
    """
    def __init__(self,
                 max_disp: int,
                 # ROI
                 roi_mode: str = "frac",
                 roi_u0: float = 0.0, roi_u1: float = 1.0,
                 roi_v0: float = 2/3, roi_v1: float = 1.0,
                 # entropy / window
                 ent_T: float = 0.1,
                 ent_vis_thr: float = 0.9,       # invalid 판정(윈도우 경계)
                 train_ent_thr: Optional[float] = None,  # after-entropy 수용 임계(없으면 미사용)
                 max_half: Optional[int] = None, # 윈도우 반폭 상한(1/4 grid)
                 # KD temperature
                 T_teacher: float = 0.1,
                 T_student: float = 1.0,
                 # loss weights
                 lambda_kl: float = 1.0,
                 lambda_reg: float = 0.5,
                 # pixel weights
                 use_peak_weight: bool = True,
                 weight_alpha: float = 1.0,  # (1 - ent_after)^alpha
                 weight_beta: float  = 1.0,  # (gap_normed)^beta
                 weight_gamma: float = 0.5,  # peak^gamma
                 weight_delta: float = 1.0,  # exp(- (L-1)/L0 )^delta
                 gap_min: Optional[float] = None,  # None or float (e.g., 1.0)
                 gap_norm: float = 4.0,
                 L0: float = 8.0):
        super().__init__()
        self.max_disp = int(max_disp) - 1

        self.roi_mode = roi_mode
        self.roi_u0, self.roi_u1 = roi_u0, roi_u1
        self.roi_v0, self.roi_v1 = roi_v0, roi_v1

        self.ent_T = ent_T
        self.ent_vis_thr = ent_vis_thr
        self.train_ent_thr = train_ent_thr
        self.max_half = max_half

        self.T_teacher = T_teacher
        self.T_student = T_student

        self.lambda_kl = lambda_kl
        self.lambda_reg = lambda_reg

        self.use_peak_weight = use_peak_weight
        self.weight_alpha = weight_alpha
        self.weight_beta  = weight_beta
        self.weight_gamma = weight_gamma
        self.weight_delta = weight_delta
        self.gap_min = gap_min
        self.gap_norm = gap_norm
        self.L0 = L0

    @staticmethod
    def _charbonnier(x, eps: float = 1e-3):
        return torch.sqrt(x * x + eps * eps)

    @staticmethod
    def _soft_argmax_from_logits(logits: torch.Tensor):
        """
        logits: [B,D+1,H4,W4] → disp: [B,H4,W4]
        """
        q = F.softmax(logits, dim=1)
        Dp1 = logits.shape[1]
        disp_idx = torch.arange(Dp1, device=logits.device, dtype=logits.dtype).view(1, Dp1, 1, 1)
        return (q * disp_idx).sum(dim=1)

    def forward(self,
                FqL: torch.Tensor,               # [B,H4,W4,C]
                FqR: torch.Tensor,               # [B,H4,W4,C]
                student_out_L: torch.Tensor,     # [B,D+1,H4,W4] or [B,1,H4,W4]/[B,H4,W4]
                student_out_R: Optional[torch.Tensor] = None,  # (옵션) 좌↔우 일관성용
                return_debug: bool = False):
        assert FqL.shape == FqR.shape and FqL.dim() == 4, "Fq must be [B,H4,W4,C]"
        B, H4, W4, C = FqL.shape
        device, dtype = FqL.device, FqL.dtype
        EPS = 1e-8

        # ---- 1) Base cost volume & before-entropy ----
        E_base = build_cost_volume_from_feats(FqL, FqR, self.max_disp)            # [B,D+1,H4,W4]
        ent_before = build_entropy_map_batched(E_base, T=self.ent_T, normalize=True)  # [B,H4,W4]

        # ---- 2) ROI mask (공통) ----
        roi_mask_single = build_roi_mask_single(H4, W4, self.roi_mode,
                                                self.roi_u0, self.roi_u1,
                                                self.roi_v0, self.roi_v1,
                                                device=device)                    # [H4,W4] bool

        # ---- 3) Adaptive window refine (ROI ∩ invalid만) ----
        E_ref, ent_after, refine_mask, win_len = refine_cost_batched(
            cost_vol=E_base, entropy_before=ent_before, roi_mask=roi_mask_single,
            ent_thr=self.ent_vis_thr, max_half=self.max_half, ent_T=self.ent_T
        )  # [B,D+1,H4,W4], [B,H4,W4], [B,H4,W4], [B,H4,W4]

        # ---- 4) Teacher distribution & soft disparity ----
        m = torch.amax(E_ref, dim=1, keepdim=True)                                 # [B,1,H4,W4]
        y_soft = F.softmax((E_ref - m) / max(self.T_teacher, 1e-6), dim=1)         # [B,D+1,H4,W4]
        Dp1 = y_soft.shape[1]
        disp_idx = torch.arange(Dp1, device=device, dtype=dtype).view(1, Dp1, 1, 1)
        d_soft = (y_soft * disp_idx).sum(dim=1)                                     # [B,H4,W4]
        # detach teacher
        y_soft = y_soft.detach()
        d_soft = d_soft.detach()

        # ---- 5) Pixel confidence weights ----
        gap_after = build_top2_gap_map_batched(E_ref)                               # [B,H4,W4]
        peak_after, _ = E_ref.max(dim=1)                                            # [B,H4,W4] (cosine sim)

        invalid_before = (ent_before > float(self.ent_vis_thr))                     # [B,H4,W4] bool
        roi_mask = roi_mask_single.unsqueeze(0).expand(B, -1, -1)                   # [B,H4,W4] bool

        # 기본 수용: ROI ∩ invalid(before)
        accept = roi_mask & invalid_before

        # (선택) after-entropy 수용 임계
        if self.train_ent_thr is not None:
            accept = accept & (ent_after <= float(self.train_ent_thr))

        # (선택) gap 최소치
        if self.gap_min is not None:
            gap_clean = torch.nan_to_num(gap_after, nan=0.0, posinf=0.0, neginf=0.0)
            accept = accept & (gap_clean >= float(self.gap_min))
        else:
            gap_clean = torch.nan_to_num(gap_after, nan=0.0, posinf=0.0, neginf=0.0)

        # 연속 가중치(0..1)
        w_ent  = (1.0 - ent_after).clamp(0.0, 1.0) ** float(self.weight_alpha)
        w_gap  = (gap_clean / float(self.gap_norm)).clamp(0.0, 1.0) ** float(self.weight_beta)
        if self.use_peak_weight:
            w_peak = ((peak_after + 1.0) * 0.5).clamp(0.0, 1.0) ** float(self.weight_gamma)
        else:
            w_peak = torch.ones_like(w_ent)
        w_win  = torch.exp(-((win_len - 1.0).clamp_min(0.0) / float(self.L0))) ** float(self.weight_delta)

        w = (w_ent * w_gap * w_peak * w_win)
        w = torch.where(accept, w, torch.zeros_like(w))                              # [B,H4,W4]
        wsum = w.sum(dim=(1,2)) + EPS                                               # [B]

        # ---- 6) Student → Loss (KL + Regression) ----
        total_kl  = torch.zeros([], device=device, dtype=dtype)
        total_reg = torch.zeros([], device=device, dtype=dtype)
        total_cnt = 0

        is_logits = (student_out_L.dim()==4 and student_out_L.shape[1]==Dp1)

   
        if is_logits:
            # KL(soft CE)
            log_q = F.log_softmax(student_out_L / max(self.T_student, 1e-6), dim=1)     # [B,D+1,H4,W4]
            ce_map = -(y_soft * log_q).sum(dim=1)                                       # [B,H4,W4]
            L_kl_b = (ce_map * w).sum(dim=(1,2)) / wsum                                  # [B]
            total_kl = L_kl_b.mean() * (self.T_teacher**2)

            # Regression (soft-argmax)
            d_pred = self._soft_argmax_from_logits(student_out_L)                        # [B,H4,W4]
            reg_map = self._charbonnier(d_pred - d_soft)                                 # [B,H4,W4]
            L_reg_b = (reg_map * w).sum(dim=(1,2)) / wsum                                # [B]
            total_reg = L_reg_b.mean()
            total_cnt = int((w > 0).sum().item())

        else:
            # Regression-only (student disp)
            if student_out_L.dim()==4 and student_out_L.shape[1]==1:
                d_pred = student_out_L[:,0]
            elif student_out_L.dim()==3:
                d_pred = student_out_L
            else:
                raise ValueError("student_out_L must be [B,D+1,H4,W4] logits or disparity [B,1,H4,W4]/[B,H4,W4]")

            reg_map = self._charbonnier(d_pred - d_soft)
            L_reg_b = (reg_map * w).sum(dim=(1,2)) / wsum
            total_reg = L_reg_b.mean()
            total_cnt = int((w > 0).sum().item())

        loss = self.lambda_kl * total_kl + self.lambda_reg * total_reg

        if not return_debug:
            return loss

        debug = {
            "loss_total": float(loss.item()),
            "loss_kl": float(total_kl.item()),
            "loss_reg": float(total_reg.item()),
            "num_pixels": total_cnt,
            "wsum_mean": float(wsum.mean().item()),
            "accept_ratio": float((accept.float().mean()).item()),
            "roi_ratio": float((roi_mask.float().mean()).item()),
            "invalid_before_ratio": float((invalid_before.float().mean()).item()),
        }
        return loss


# # ============================================================
# # 6) Quick self-test (random tensors) — shape sanity
# # ============================================================
# if __name__ == "__main__":
#     torch.manual_seed(0)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     B, H4, W4, C = 2, 48, 160, 384
#     D = 22  # disparity max (1/4 grid)

#     # Fake interleaved features (L2-normalized)
#     FqL = torch.randn(B, H4, W4, C, device=device)
#     FqR = torch.randn(B, H4, W4, C, device=device)
#     FqL = FqL / (FqL.norm(dim=-1, keepdim=True) + 1e-8)
#     FqR = FqR / (FqR.norm(dim=-1, keepdim=True) + 1e-8)

#     # Student logits (distribution head)
#     student_logits = torch.randn(B, D+1, H4, W4, device=device)

#     criterion = AdaptiveWindowDistillLoss(
#         max_disp=D,
#         roi_mode="frac", roi_u0=0.0, roi_u1=1.0, roi_v0=2/3, roi_v1=1.0,
#         ent_T=0.1, ent_vis_thr=0.9, train_ent_thr=None,  # after-entropy gating off
#         max_half=12, T_teacher=0.1, T_student=1.0,
#         lambda_kl=1.0, lambda_reg=0.5,
#         use_peak_weight=True, weight_alpha=1.0, weight_beta=1.0,
#         weight_gamma=0.5, weight_delta=1.0,
#         gap_min=1.0, gap_norm=4.0, L0=8.0
#     ).to(device)

#     loss, info = criterion(FqL, FqR, student_logits, return_debug=True)

