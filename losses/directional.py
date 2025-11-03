import torch
import torch.nn as nn
import torch.nn.functional as F

# shift_with_mask 는 기존 tools의 것을 재사용합니다 (채널-퍼스트 입력).
# cossim_feat_1_4 는 [B,H,W,C] 이므로 내부에서 [B,C,H,W] 로 변환해 사용합니다.

def _huber_positive(z: torch.Tensor, delta: float) -> torch.Tensor:
    """
    hinge + huber:
      z <= 0 → 0 (무벌점)
      z >  0 → Huber(z; delta)
    """
    z_pos = torch.clamp(z, min=0.0)
    small = (z_pos < delta).to(z.dtype)
    return 0.5 * (z_pos ** 2) / (delta + 1e-6) * small + (z_pos - 0.5 * delta) * (1.0 - small)

def _huber_window(delta: torch.Tensor, a: float, b: float, huber_delta: float) -> torch.Tensor:
    """
    허용 구간 [a,b] 외부만 벌점:
      penalty = Huber(delta - b)_+ + Huber(a - delta)_+
    """
    return _huber_positive(delta - b, huber_delta) + _huber_positive(a - delta, huber_delta)

def _rel_gate_from_cossim(sim_raw: torch.Tensor, valid: torch.Tensor,
                          thr: float = 0.75, gamma: float = 0.0,
                          use_dynamic_thr: bool = True, dynamic_q: float = 0.7) -> torch.Tensor:
    """
    sim_raw: 코사인 유사도 [-1,1], shape [B,1,H,W]
    valid  : {0,1},         shape [B,1,H,W]
    반환: 게이팅 w ∈ [0,1], valid 영역 밖은 0
    """
    eps = 1e-6
    sim01 = 0.5 * (sim_raw + 1.0)  # [-1,1] → [0,1]

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

class DirectionalRelScaleDispLoss(nn.Module):
    """
    세로 방향 비대칭 허용 + cossim_feat 기반 게이팅 버전.
      - 위쪽(dy=-1): Δ = disp - disp_up  ∈ [-up_allow, 0] 이면 0, 그 외 Huber-윈도우 벌점
      - 아래(dy=+1): Δ = disp - disp_dn  ∈ [0, +down_allow] 이면 0, 그 외 Huber-윈도우 벌점
      - 가로: |Δ| <= horiz_margin (대칭) 허용

    입력:
      disp              [B,1,H/4,W/4]  (pixel disparity)
      cossim_feat_1_4   [B,H/4,W/4,C]  (L2 normalized, channel-last)
      roi               [B,1,H/4,W/4]

    y축은 아래로 증가한다고 가정(dy=-1: 위쪽 이웃, dy=+1: 아래쪽 이웃).
    """
    def __init__(self,
                 sim_thr: float = 0.75, sim_gamma: float = 0.0,
                 sample_k: int = 512,               # 호환성 유지(미사용)
                 use_dynamic_thr: bool = True, dynamic_q: float = 0.7,
                 vert_up_allow: float = 1.0,        # 위쪽 허용폭: [-1, 0]
                 vert_down_allow: float = 1.0,      # 아래 허용폭: [0, +1]
                 horiz_margin: float = 0.0,         # |Δ| <= margin
                 lambda_v: float = 1.0, lambda_h: float = 1.0,
                 huber_delta: float = 1.0):
        super().__init__()
        self.sim_thr, self.sim_gamma = sim_thr, sim_gamma
        self.sample_k = sample_k
        self.use_dynamic_thr = use_dynamic_thr
        self.dynamic_q = dynamic_q

        self.vert_up_allow   = float(vert_up_allow)
        self.vert_down_allow = float(vert_down_allow)
        self.horiz_margin    = float(horiz_margin)

        self.lambda_v, self.lambda_h = lambda_v, lambda_h
        self.huber_delta = huber_delta

        self.vert_up_pair   = (-1, 0)
        self.vert_down_pair = (+1, 0)
        self.hori_pairs     = [(0, +1), (0, -1)]

    def _sim_gate(self, cossim_cf: torch.Tensor, dy: int, dx: int, roi: torch.Tensor):
        from tools import shift_with_mask
        f_nb, valid_b = shift_with_mask(cossim_cf, dy, dx)
        roi_nb, _     = shift_with_mask(roi,      dy, dx)
        valid = valid_b * roi * roi_nb
        sim_raw = (cossim_cf * f_nb).sum(dim=1, keepdim=True)  # [B,1,H,W], [-1,1]
        w = _rel_gate_from_cossim(sim_raw, valid,
                                  thr=self.sim_thr, gamma=self.sim_gamma,
                                  use_dynamic_thr=self.use_dynamic_thr, dynamic_q=self.dynamic_q)
        return w, valid

    def _accum_vertical(self, disp: torch.Tensor, cossim_cf: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
        from tools import shift_with_mask
        loss_sum = torch.tensor(0.0, device=disp.device)
        weight_sum = torch.tensor(0.0, device=disp.device)

        # --- 위쪽 (dy = -1): Δ ∈ [-up_allow, 0] 허용 ---
        d_up, _      = shift_with_mask(disp, *self.vert_up_pair)
        roi_up, _    = shift_with_mask(roi,  *self.vert_up_pair)
        w_up, valid_up = self._sim_gate(cossim_cf, *self.vert_up_pair, roi=roi)
        valid_up = valid_up * roi * roi_up
        delta_up = disp - d_up
        viol_up  = _huber_window(delta_up, a=-self.vert_up_allow, b=0.0, huber_delta=self.huber_delta)
        loss_sum   += (w_up * viol_up).sum()
        weight_sum += w_up.sum()

        # --- 아래쪽 (dy = +1): Δ ∈ [0, +down_allow] 허용 ---
        d_dn, _      = shift_with_mask(disp, *self.vert_down_pair)
        roi_dn, _    = shift_with_mask(roi,  *self.vert_down_pair)
        w_dn, valid_dn = self._sim_gate(cossim_cf, *self.vert_down_pair, roi=roi)
        valid_dn = valid_dn * roi * roi_dn
        delta_dn = disp - d_dn
        viol_dn  = _huber_window(delta_dn, a=0.0, b=+self.vert_down_allow, huber_delta=self.huber_delta)
        loss_sum   += (w_dn * viol_dn).sum()
        weight_sum += w_dn.sum()

        return loss_sum / (weight_sum + 1e-6)

    def _accum_horizontal(self, disp: torch.Tensor, cossim_cf: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
        from tools import shift_with_mask
        loss_sum = torch.tensor(0.0, device=disp.device)
        weight_sum = torch.tensor(0.0, device=disp.device)

        for dy, dx in self.hori_pairs:
            d_nb, _   = shift_with_mask(disp, dy, dx)
            roi_nb, _ = shift_with_mask(roi,  dy, dx)
            w, valid  = self._sim_gate(cossim_cf, dy, dx, roi)
            valid = valid * roi * roi_nb

            delta = (disp - d_nb).abs() - self.horiz_margin  # [ -m, +∞ )
            viol  = _huber_positive(delta, self.huber_delta) # >0만 벌점
            loss_sum   += (w * viol).sum()
            weight_sum += w.sum()

        return loss_sum / (weight_sum + 1e-6)

    def forward(self, disp: torch.Tensor, cossim_feat_1_4: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
        """
        disp:             [B,1,H/4,W/4]
        cossim_feat_1_4:  [B,H/4,W/4,C] (L2 norm, channel-last)
        roi:              [B,1,H/4,W/4]
        """
        assert cossim_feat_1_4.dim() == 4 and cossim_feat_1_4.size(1) == disp.size(-2), \
            "cossim_feat_1_4 shape must be [B,H/4,W/4,C] (channel-last)"

        # [B,H,W,C] → [B,C,H,W]
        cossim_cf = cossim_feat_1_4.permute(0, 3, 1, 2).contiguous()

        loss_v = self._accum_vertical(disp, cossim_cf, roi)
        loss_h = self._accum_horizontal(disp, cossim_cf, roi)
        return self.lambda_v * loss_v + self.lambda_h * loss_h
