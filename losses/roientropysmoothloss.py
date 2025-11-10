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
def argmax_disparity(cost_vol: torch.Tensor):
    """
    반환:
      - disp_map: [H4,W4] (long)
      - peak_sim: [H4,W4] (float)
    """
    peak_sim, disp_map = cost_vol.max(dim=0)
    return disp_map, peak_sim



class EntropySmoothnessLoss(nn.Module):
    """
    DINO 1/4-특징(FqL/FqR)로 코스트볼륨을 만들고,
    - refine#1: invalid(mask=entropy_before>thr) ∩ ROI만 가로-윈도 평균으로 치환
    - entropy 재계산
    - refine#2: (다시) invalid(=entropy_after#1>thr) ∩ ROI만 치환
    => 최종 코스트 E_final에서 teacher 시차 생성

    손실은 disp_filled_cell(= base argmax에서 마지막 refine 마스크만 교사로 덮어쓴 지도)와
    학생 예측(student_disp) 간 Smooth-L1.

    Args:
        max_disp: 1/4-그리드 최대 시차 (정수, inclusive)
        roi_mode: 'frac' | 'abs4'
        roi_u0,u1,v0,v1: ROI 범위
        ent_T: entropy softmax 온도
        ent_thr: invalid 임계(0..1)
        win_half_max: 가로 윈도 반폭 상한(1/4 grid)
        teacher_type: 'arg' | 'soft'  (soft는 기대값)
        T_teacher: soft teacher 온도
        beta: SmoothL1 beta (0이면 L1)
        supervise_area: 'filled' | 'after' | 'before' | 'union' | 'all'
                        - 'filled' : 마지막 refine 마스크 위치만 감독 (기본)
                        - 'after'  : (entropy_after <= ent_thr) ∩ ROI
                        - 'before' : (entropy_before <= ent_thr) ∩ ROI
                        - 'union'  : build_union_viz_mask(before, after, thr, ROI)
                        - 'all'    : 전체
        student_unit: 'px' | 'cell' (px면 내부에서 /4)
        two_stage: True면 refine을 2회 수행(E0→E1→E2)
        stop_teacher_grad: True면 FqL/FqR에서 교사 경로로의 gradient 차단(detach)
    """
    def __init__(self,
                 max_disp: int,
                 roi_mode: str = "frac",
                 roi_u0: float = 0.2, roi_u1: float = 0.7,
                 roi_v0: float = 2/3, roi_v1: float = 1.0,
                 ent_T: float = 0.1,
                 ent_thr: float = 0.6,
                 win_half_max: int = 12,
                 teacher_type: str = "arg",
                 T_teacher: float = 0.1,
                 beta: float = 1.0,
                 supervise_area: str = "filled",
                 student_unit: str = "px",
                 two_stage: bool = True,
                 stop_teacher_grad: bool = True):
        super().__init__()
        self.max_disp = int(max_disp)
        self.roi_mode = roi_mode
        self.u0, self.u1 = roi_u0, roi_u1
        self.v0, self.v1 = roi_v0, roi_v1
        self.ent_T = ent_T
        self.ent_thr = ent_thr
        self.win_half_max = win_half_max
        self.teacher_type = teacher_type
        self.T_teacher = T_teacher
        self.beta = beta
        self.supervise_area = supervise_area
        self.student_unit = student_unit
        self.two_stage = two_stage
        self.stop_teacher_grad = stop_teacher_grad

    def _teacher_maps(self, featL, featR):
        """
        featL, featR: [H4,W4,C] (L2-norm, BHWC에서 배치 분리된 한 샘플)
        return:
            base_cell:    [H4,W4] (E0 argmax, float)
            teacher_cell: [H4,W4] (E_final에서 arg/soft, float)
            refine_mask:  [H4,W4] (bool, 마지막 refine 마스크)
            ent_before:   [H4,W4] (float, 0..1)
            ent_after:    [H4,W4] (float, 0..1, 마지막)
            roi_mask:     [H4,W4] (bool)
        """
        device = featL.device
        D = self.max_disp

        # --- E0 & entropy_before ---
        E0 = build_cost_volume(featL, featR, D)                    # [D+1,H4,W4]
        ent0 = build_entropy_map(E0, T=self.ent_T, normalize=True) # [H4,W4]

        H4, W4 = ent0.shape
        roi = build_roi_mask(H4, W4, self.roi_mode,
                             self.u0, self.u1, self.v0, self.v1, device)

        # --- refine #1 ---
        E1, ent1, rm1 = refine_cost_for_uncertain_roi(
            E0, ent0, ent_thr=self.ent_thr, roi_mask=roi,
            max_half=self.win_half_max, ent_T=self.ent_T
        )

        # --- refine #2 (옵션) ---
        if self.two_stage:
            # 요청대로: refine 한 번 후 entropy 재산출 결과로 다시 invalid/ROI 연산
            E2, ent2, rm2 = refine_cost_for_uncertain_roi(
                E1, ent1, ent_thr=self.ent_thr, roi_mask=roi,
                max_half=self.win_half_max, ent_T=self.ent_T
            )
            E_final, ent_final, refine_mask = E2, ent2, rm2
        else:
            E_final, ent_final, refine_mask = E1, ent1, rm1

        # --- base/teacher disparity (cell units) ---
        base_cell, _ = argmax_disparity(E0)         # long → float
        base_cell = base_cell.to(torch.float32)

        if self.teacher_type == "soft":
            m = torch.amax(E_final, dim=0, keepdim=True)
            prob = torch.softmax((E_final - m) / max(self.T_teacher, 1e-6), dim=0)  # [D+1,H4,W4]
            disp_idx = torch.arange(D+1, device=device, dtype=prob.dtype).view(D+1,1,1)
            teacher_cell = (prob * disp_idx).sum(dim=0)                              # [H4,W4]
        else:
            teacher_cell, _ = argmax_disparity(E_final)
            teacher_cell = teacher_cell.to(torch.float32)

        return base_cell, teacher_cell, refine_mask, ent0, ent_final, roi

    def forward(self,
                FqL: torch.Tensor,            # [B,H4,W4,C] (BHWC, L2-norm)
                FqR: torch.Tensor,            # [B,H4,W4,C]
                student_disp: torch.Tensor,   # [B,1,H4,W4] or [B,H4,W4] (px or cell)
                return_debug: bool = False):
        assert FqL.shape == FqR.shape and FqL.dim() == 4, "FqL/FqR must be [B,H4,W4,C]"
        B, H4, W4, C = FqL.shape
        device = FqL.device

        # 학생 단위/shape 정리 → [B,1,H4,W4], cell 단위
        if student_disp.dim() == 3:
            student_disp = student_disp.unsqueeze(1)    # [B,1,H4,W4]
        pred = student_disp
        if self.student_unit == "px":
            pred = pred / 4.0                           # px → cell

        # (옵션) teacher 경로 grad 차단
        FqL_t = FqL.detach() if self.stop_teacher_grad else FqL
        FqR_t = FqR.detach() if self.stop_teacher_grad else FqR

        losses = []
        dbg = []

        for b in range(B):
            featL = FqL_t[b]  # [H4,W4,C]
            featR = FqR_t[b]

            base_cell, teacher_cell, refine_mask, ent0, entF, roi = self._teacher_maps(featL, featR)

            # disp_filled_cell = base ⊕ teacher (마지막 refine 마스크만 교사로 대체)
            filled = base_cell.clone()
            filled[refine_mask] = teacher_cell[refine_mask]
            tgt = filled.unsqueeze(0)  # [1,H4,W4]

            # 감독 영역 선택
            sa = self.supervise_area.lower()
            if sa == "filled":
                accept = refine_mask
            elif sa == "after":
                accept = (entF <= float(self.ent_thr)) & roi
            elif sa == "before":
                accept = (ent0 <= float(self.ent_thr)) & roi
            elif sa == "union":
                accept = build_union_viz_mask(ent0, entF, thr=self.ent_thr, roi_mask=roi)
            elif sa == "all":
                accept = torch.ones_like(refine_mask, dtype=torch.bool, device=refine_mask.device)
            else:
                raise ValueError(f"Unknown supervise_area: {self.supervise_area}")

            # Smooth‑L1 (mask 평균)
            diff = F.smooth_l1_loss(pred[b], tgt, beta=self.beta, reduction='none')  # [1,H4,W4]
            m = accept.unsqueeze(0).to(diff.dtype)
            denom = m.sum().clamp_min(1.0)
            loss_b = (diff * m).sum() / denom
            losses.append(loss_b)

            if return_debug:
                dbg.append({
                    "base_cell": base_cell.detach(),
                    "teacher_cell": teacher_cell.detach(),
                    "disp_filled_cell": filled.detach(),
                    "refine_mask": refine_mask.detach(),
                    "entropy_before": ent0.detach(),
                    "entropy_after": entF.detach(),
                    "roi_mask": roi.detach(),
                })

        loss = torch.stack(losses).mean()
        if return_debug:
            return loss, dbg
        return loss
