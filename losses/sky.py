import torch
import torch.nn as nn
import torch.nn.functional as F

def _huber(x, delta):
    absx = x.abs()
    return torch.where(absx <= delta, 0.5*absx*absx/delta, absx - 0.5*delta)

class SkyGridZeroLoss(nn.Module):
    """
    간단한 sky 제로 시차 유도:
    - refined_logits_masked를 D축 softmax 후 작은 disparity(0~2 bin) 확률을 sky proxy로 사용.
    - disp_half_px를 해당 격자 해상도로 내려서 |disp|를 허버로 벌점.
    호출부: loss, aux = forward(refined_logits_masked, disp_half_px, roi_half, roi_patch, names, step)
    """
    def __init__(self, max_disp_px=192, patch_size=4):
        super().__init__()
        self.delta = 1.0

    def forward(self, refined_logits_masked, disp_half_px,
                roi_half=None, roi_patch=None, names=None, step=None):
        # refined_logits_masked: [B,1,D,H8,W8]
        if refined_logits_masked.dim() != 5:
            raise ValueError("refined_logits_masked must be [B,1,D,H,W]")
        logits = refined_logits_masked[:, 0]              # [B,D,H8,W8]
        p = torch.softmax(logits, dim=1)                  # [B,D,H8,W8]
        D = logits.size(1)
        k = min(3, D)                                     # 0~2 bin
        sky_w = p[:, :k].sum(dim=1, keepdim=True)         # [B,1,H8,W8], 0~1

        # disp_half_px: [B,1,H2,W2] -> H8,W8 로 다운샘플
        H8, W8 = logits.size(-2), logits.size(-1)
        disp_coarse = F.interpolate(disp_half_px, size=(H8, W8), mode='bilinear', align_corners=False)

        loss_map = _huber(disp_coarse.abs(), self.delta) * sky_w
        denom = sky_w.sum() + 1e-6
        loss = loss_map.sum() / denom

        aux = {"sky_weight_mean": sky_w.mean().item(), "valid": float(denom.item())}
        return loss, aux
