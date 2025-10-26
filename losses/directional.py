import torch
import torch.nn as nn
import torch.nn.functional as F

def _shift(t, dy=0, dx=0):
    b, c, h, w = t.shape
    pad = (max(dx,0), max(-dx,0), max(dy,0), max(-dy,0))
    t_pad = F.pad(t, pad, mode='replicate')
    y0 = max(-dy,0); x0 = max(-dx,0)
    return t_pad[:, :, y0:y0+h, x0:x0+w]

def _huber(x, delta):
    absx = x.abs()
    return torch.where(absx <= delta, 0.5*absx*absx/delta, absx - 0.5*delta)

class DirectionalRelScaleDispLoss(nn.Module):
    """
    수직/수평 방향으로 시차 기울기를 제어하는 안정형 버전.
    호출부: forward(disp_soft, FL_dino, roi)
    - vert_margin, horiz_margin 이하의 변화는 무시(허용), 초과분만 허버 페널티.
    - lambda_v, lambda_h 로 가중.
    - sim_* 인자는 API 호환을 위해 받지만, 여기서는 안정형 기본 동작에 사용 최소화.
    """
    def __init__(self,
                 sim_thr=0.7, sim_gamma=0.1, sample_k=1024,
                 use_dynamic_thr=False, dynamic_q=0.9,
                 vert_margin=1.0, horiz_margin=0.0,
                 lambda_v=1.0, lambda_h=0.0,
                 huber_delta=1.0):
        super().__init__()
        self.vert_margin = float(vert_margin)
        self.horiz_margin = float(horiz_margin)
        self.lambda_v = float(lambda_v)
        self.lambda_h = float(lambda_h)
        self.delta = float(huber_delta)

    def forward(self, disp_soft, feat, roi):
        # disp_soft: [B,1,H,W], feat: [B,C,H,W], roi: [B,1,H,W]
        dx = (disp_soft - _shift(disp_soft, 0, 1)).abs()
        dy = (disp_soft - _shift(disp_soft, 1, 0)).abs()

        # 허용 마진 초과분만 페널티
        px = torch.clamp(dx - self.horiz_margin, min=0.0)
        py = torch.clamp(dy - self.vert_margin,  min=0.0)

        loss_x = _huber(px, self.delta)
        loss_y = _huber(py, self.delta)

        loss_map = self.lambda_h * loss_x + self.lambda_v * loss_y  # [B,1,H,W]
        loss = (loss_map * roi).sum() / (roi.sum() + 1e-6)
        return loss
