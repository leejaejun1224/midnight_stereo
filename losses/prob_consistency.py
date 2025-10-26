import torch
import torch.nn as nn
import torch.nn.functional as F

def _shift2d(t, dy=0, dx=0):
    b, c, h, w = t.shape
    pad = (max(dx,0), max(-dx,0), max(dy,0), max(-dy,0))
    t_pad = F.pad(t, pad, mode='replicate')
    y0 = max(-dy,0); x0 = max(-dx,0)
    return t_pad[:, :, y0:y0+h, x0:x0+w]

class NeighborProbConsistencyLoss(nn.Module):
    """
    이웃 픽셀과의 확률 분포 일관성 (라벨 없음, self-sup).
    호출부: forward(prob, feat, roi)
    prob: [B,D,H,W] 또는 [B,1,D,H,W] (자동 처리)
    allow_shift_v/h: 이웃 고려 반경
    """
    def __init__(self,
                 sim_thr=0.7, sim_gamma=0.1, sample_k=1024,
                 allow_shift_v=1, allow_shift_h=0,
                 use_dynamic_thr=True, dynamic_q=0.9,
                 conf_alpha=1.0):
        super().__init__()
        self.allow_v = int(allow_shift_v)
        self.allow_h = int(allow_shift_h)

    def forward(self, prob, feat, roi):
        if prob.dim() == 5:   # [B,1,D,H,W]
            prob = prob[:, 0]
        # 정규화 보장
        prob = prob.clamp_min(1e-8)
        prob = prob / prob.sum(dim=1, keepdim=True)

        B, D, H, W = prob.shape
        loss_acc = 0.0
        count = 0

        # 수직/수평 이웃
        shifts = []
        for dy in range(-self.allow_v, self.allow_v+1):
            for dx in range(-self.allow_h, self.allow_h+1):
                if dy == 0 and dx == 0: 
                    continue
                shifts.append((dy, dx))

        prob_roi = roi
        for dy, dx in shifts:
            p_n = _shift2d(prob, dy, dx)  # [B,D,H,W]
            # 분포 L1 거리
            l = (prob - p_n).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
            # 유효한 위치만 평균
            valid = (roi * _shift2d(roi, dy, dx)).float()
            loss_acc = loss_acc + (l * valid).sum() / (valid.sum() + 1e-6)
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=prob.device, dtype=prob.dtype)
        return loss_acc / count
