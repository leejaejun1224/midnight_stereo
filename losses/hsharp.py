import torch
import torch.nn as nn
import torch.nn.functional as F

def _huber(x, delta):
    absx = x.abs()
    return torch.where(absx <= delta, 0.5*absx*absx/delta, absx - 0.5*delta)

class HorizontalSharpenedConsistency(nn.Module):
    """
    분포 예리화(샤프닝): D축 확률의 최대값을 키우는 식(에지 근처에서 sharpen).
    호출부: forward(refined_masked, FL_dino, roi)
    refined_masked: [B,1,D,H,W] or [B,D,H,W] (logit/score)
    """
    def __init__(self, D, tau_sharp=1.0, huber_delta=1.0,
                 use_fixed_denom=True,
                 sim_thr=0.7, sim_gamma=0.1, sample_k=1024,
                 use_dynamic_thr=False, dynamic_q=0.9):
        super().__init__()
        self.tau = float(tau_sharp)
        self.delta = float(huber_delta)

    def forward(self, refined_masked, feat, roi):
        if refined_masked.dim() == 5:   # [B,1,D,H,W]
            logits = refined_masked[:, 0]
        else:                           # [B,D,H,W]
            logits = refined_masked
        # 확률화(샤프닝)
        p = torch.softmax(logits / max(self.tau, 1e-3), dim=1)  # [B,D,H,W]
        conf, _ = p.max(dim=1, keepdim=True)                    # [B,1,H,W], 0~1
        # 1 - conf 를 허버로 줄임 => conf↑
        loss_map = _huber(1.0 - conf, self.delta)
        loss = (loss_map * roi).sum() / (roi.sum() + 1e-6)
        return loss
