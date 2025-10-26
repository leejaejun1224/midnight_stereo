import torch
import torch.nn as nn
import torch.nn.functional as F
from .geometry import warp_right_to_left

class FeatureReprojLoss(nn.Module):
    """
    간단하고 안정한 특징 재투영 L1
    forward(FL, FR, disp_soft, roi=None)  # 호출부 동일
    FL, FR: [B,C,H,W] (예: H/4, W/4), disp_soft: [B,1,H,W]
    """
    def __init__(self, charbonnier_eps=1e-3):
        super().__init__()
        self.eps = float(charbonnier_eps)

    def charbonnier(self, x):
        return (x*x + self.eps*self.eps).sqrt()

    def forward(self, FL, FR, disp_soft, roi=None):
        FR_warp, valid = warp_right_to_left(FR, disp_soft, padding_mode='border', align_corners=True)
        diff = self.charbonnier(FL - FR_warp).mean(1, keepdim=True)  # [B,1,H,W]
        mask = valid if roi is None else (valid * roi)
        loss = (diff * mask).sum() / (mask.sum() + 1e-6)
        return loss
