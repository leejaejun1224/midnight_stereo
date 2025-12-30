import torch
import torch.nn as nn
import torch.nn.functional as F

# === 기존 헬퍼와 동일한 의미/역할 ===
def _huber_positive(z: torch.Tensor, delta: float) -> torch.Tensor:
    z_pos = torch.clamp(z, min=0.0)
    small = (z_pos < delta).to(z.dtype)
    return 0.5 * (z_pos ** 2) / (delta + 1e-6) * small + (z_pos - 0.5 * delta) * (1.0 - small)

def _rel_gate_from_cossim(sim_raw: torch.Tensor, valid: torch.Tensor,
                          thr: float = 0.75, gamma: float = 0.0,
                          use_dynamic_thr: bool = True, dynamic_q: float = 0.7) -> torch.Tensor:
    eps = 1e-6
    sim01 = 0.5 * (sim_raw + 1.0)  # [-1,1]→[0,1]
    if use_dynamic_thr:
        v = sim01[valid > 0]
        thr_eff = torch.quantile(v, dynamic_q).item() if v.numel() > 0 else thr
    else:
        thr_eff = thr
    if gamma is None or gamma <= 0.0:
        w = (sim01 >= thr_eff).to(sim01.dtype)
    else:
        w = torch.sigmoid((sim01 - thr_eff) / (gamma + eps))
    return w * valid

def _window_violation(delta: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """
    허용구간 [a,b]의 '밖'으로 벗어난 양(>=0)을 반환.
      - 아래로 벗어남: a - delta
      - 위로 벗어남:   delta - b
    """
    below = torch.clamp(a - delta, min=0.0)
    above = torch.clamp(delta - b, min=0.0)
    return below + above  # [0, +∞)

def _sgm_two_stage(v: torch.Tensor,
                   t1: float, t2: float,  # 두 임계 (t1<=t2)
                   P1: float, P2: float,  # 작은/큰 위반의 계수 (P2>P1)
                   delta_small: float, delta_large: float) -> torch.Tensor:
    """
    SGM의 'P1(작은 변화) / P2(큰 점프)' 철학을 연속화한 2단 경사 페널티.
      v: 위반량(>=0)
      단계1: v > t1 에 대해 P1 * Huber(v - t1; delta_small)
      단계2: v > t2 에 대해 P2 * Huber(v - t2; delta_large)
    - 큰 위반에선 단계1+2가 모두 작동해 기울기가 커짐(과도한 자르기 없이 안정적).
    """
    loss_small = P1 * _huber_positive(v - t1, delta_small)
    loss_large = P2 * _huber_positive(v - t2, delta_large)
    return loss_small + loss_large

class DirectionalRelScaleDispLossSGM(nn.Module):
    """
    기존 손실의 구조(세로 비대칭 + 가로 대칭 + cos-sim 게이트)는 유지.
    단, 페널티를 '단일 Huber' → 'SGM풍 2단 경사(P1/P2)'로 변경.

    입력:
      disp             [B,1,H/4,W/4]  (px)
      cossim_feat_1_4  [B,H/4,W/4,C]  (L2 norm, ch-last)
      roi              [B,1,H/4,W/4]
    """
    def __init__(self,
                 # 게이트
                 sim_thr: float = 0.75, sim_gamma: float = 0.0,
                 use_dynamic_thr: bool = True, dynamic_q: float = 0.7,
                 # 허용 구간(기하 제약) — 기존과 동일 의미
                 vert_up_allow: float = 1.0,     # 위쪽: [-U, 0]
                 vert_down_allow: float = 1.0,   # 아래: [0, +D]
                 horiz_margin: float = 0.0,      # |Δ| <= m
                 # 가중치
                 lambda_v: float = 1.0, lambda_h: float = 1.0,
                 # === SGM풍 2단 경사 하이퍼 ===
                 t1_px: float = 0.5,             # 작은 위반 임계 (~0.3~0.7 px)
                 t2_px: float = 1.5,             # 큰 위반 임계 (~1.0~2.0 px)
                 P1: float = 1.0,                # 작은 위반 기울기
                 P2: float = 4.0,                # 큰 위반 기울기(>P1)
                 huber_small: float = 0.5,       # 단계1의 허버 델타
                 huber_large: float = 1.0):      # 단계2의 허버 델타
        super().__init__()
        self.sim_thr, self.sim_gamma = sim_thr, sim_gamma
        self.use_dynamic_thr, self.dynamic_q = use_dynamic_thr, dynamic_q
        self.vert_up_allow, self.vert_down_allow = float(vert_up_allow), float(vert_down_allow)
        self.horiz_margin = float(horiz_margin)
        self.lambda_v, self.lambda_h = lambda_v, lambda_h

        self.t1, self.t2 = float(t1_px), float(t2_px)
        assert self.t1 <= self.t2, "t1_px <= t2_px 여야 합니다."
        self.P1, self.P2 = float(P1), float(P2)
        self.huber_small, self.huber_large = float(huber_small), float(huber_large)

        # 이웃 쌍 정의(기존과 동일)
        self.vert_up_pair   = (-1, 0)
        self.vert_down_pair = (+1, 0)
        self.hori_pairs     = [(0, +1), (0, -1)]

    def _sim_gate(self, cossim_cf: torch.Tensor, dy: int, dx: int, roi: torch.Tensor):
        from tools import shift_with_mask
        f_nb, valid_b = shift_with_mask(cossim_cf, dy, dx)
        roi_nb, _     = shift_with_mask(roi,      dy, dx)
        valid = valid_b * roi * roi_nb
        sim_raw = (cossim_cf * f_nb).sum(dim=1, keepdim=True)  # [-1,1]
        w = _rel_gate_from_cossim(sim_raw, valid,
                                  thr=self.sim_thr, gamma=self.sim_gamma,
                                  use_dynamic_thr=self.use_dynamic_thr, dynamic_q=self.dynamic_q)
        return w, valid

    def _accum_vertical(self, disp: torch.Tensor, cossim_cf: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
        from tools import shift_with_mask
        loss_sum = torch.tensor(0.0, device=disp.device)
        weight_sum = torch.tensor(0.0, device=disp.device)

        # --- 위쪽 (dy=-1): 허용 [-U, 0] ---
        d_up, _   = shift_with_mask(disp, *self.vert_up_pair)
        roi_up, _ = shift_with_mask(roi,  *self.vert_up_pair)
        w_up, _   = self._sim_gate(cossim_cf, *self.vert_up_pair, roi=roi)
        w_up = w_up * roi * roi_up
        delta_up = disp - d_up
        v_up = _window_violation(delta_up, a=-self.vert_up_allow, b=0.0)  # [0,+∞)
        pen_up = _sgm_two_stage(v_up, self.t1, self.t2, self.P1, self.P2, self.huber_small, self.huber_large)
        loss_sum   += (w_up * pen_up).sum()
        weight_sum += w_up.sum()

        # --- 아래쪽 (dy=+1): 허용 [0, +D] ---
        d_dn, _   = shift_with_mask(disp, *self.vert_down_pair)
        roi_dn, _ = shift_with_mask(roi,  *self.vert_down_pair)
        w_dn, _   = self._sim_gate(cossim_cf, *self.vert_down_pair, roi=roi)
        w_dn = w_dn * roi * roi_dn
        delta_dn = disp - d_dn
        v_dn = _window_violation(delta_dn, a=0.0, b=+self.vert_down_allow)
        pen_dn = _sgm_two_stage(v_dn, self.t1, self.t2, self.P1, self.P2, self.huber_small, self.huber_large)
        loss_sum   += (w_dn * pen_dn).sum()
        weight_sum += w_dn.sum()

        return loss_sum / (weight_sum + 1e-6)

    def _accum_horizontal(self, disp: torch.Tensor, cossim_cf: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
        from tools import shift_with_mask
        loss_sum = torch.tensor(0.0, device=disp.device)
        weight_sum = torch.tensor(0.0, device=disp.device)

        for dy, dx in self.hori_pairs:
            d_nb, _   = shift_with_mask(disp, dy, dx)
            roi_nb, _ = shift_with_mask(roi,  dy, dx)
            w, _      = self._sim_gate(cossim_cf, dy, dx, roi)
            w = w * roi * roi_nb

            # |Δ| <= margin 허용 → 위반량 v = max(0, |Δ| - margin)
            delta = (disp - d_nb).abs()
            v = torch.clamp(delta - self.horiz_margin, min=0.0)
            pen = _sgm_two_stage(v, self.t1, self.t2, self.P1, self.P2, self.huber_small, self.huber_large)

            loss_sum   += (w * pen).sum()
            weight_sum += w.sum()

        return loss_sum / (weight_sum + 1e-6)

    def forward(self, disp: torch.Tensor, cossim_feat_1_4: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
        # 입력 형태 점검(기존 가정 유지)
        assert cossim_feat_1_4.dim() == 4 and cossim_feat_1_4.size(1) == disp.size(-2), \
            "cossim_feat_1_4 shape must be [B,H/4,W/4,C] (channel-last)"
        cossim_cf = cossim_feat_1_4.permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]

        loss_v = self._accum_vertical(disp, cossim_cf, roi)
        loss_h = self._accum_horizontal(disp, cossim_cf, roi)
        return self.lambda_v * loss_v + self.lambda_h * loss_h
