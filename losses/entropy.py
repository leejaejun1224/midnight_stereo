import torch
import torch.nn as nn
import torch.nn.functional as F

class EntropySharpnessLoss(nn.Module):
    """
    확률 분포의 엔트로피를 낮춰(샤프하게) 주는 손실.
    호출부: forward(prob, feat, roi)
    """
    def __init__(self, conf_alpha=1.0, sim_thr=0.7, sim_gamma=0.1,
                 sample_k=1024, use_dynamic_thr=True, dynamic_q=0.9):
        super().__init__()

    def forward(self, prob, feat, roi):
        if prob.dim() == 5:   # [B,1,D,H,W]
            prob = prob[:, 0]
        prob = prob.clamp_min(1e-8)
        prob = prob / prob.sum(dim=1, keepdim=True)
        ent = -(prob * torch.log(prob)).sum(dim=1, keepdim=True)  # [B,1,H,W]
        loss = (ent * roi).sum() / (roi.sum() + 1e-6)
        return loss
