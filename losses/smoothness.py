import torch
import torch.nn.functional as F

def _shift(t, dy=0, dx=0):
    b, c, h, w = t.shape
    pad = (max(dx,0), max(-dx,0), max(dy,0), max(-dy,0))  # (left,right,top,bottom)
    t_pad = F.pad(t, pad, mode='replicate')
    y0 = max(-dy,0); x0 = max(-dx,0)
    return t_pad[:, :, y0:y0+h, x0:x0+w]

def get_disparity_smooth_loss(disp, img, eps=1e-3):
    """
    Edge-aware smoothness (일반적 구현)
    disp: [B,1,H,W], img: [B,3,H,W] in [0,1]
    """
    img_dx = (img - _shift(img, 0, 1)).abs().mean(1, keepdim=True)
    img_dy = (img - _shift(img, 1, 0)).abs().mean(1, keepdim=True)

    disp_dx = (disp - _shift(disp, 0, 1)).abs()
    disp_dy = (disp - _shift(disp, 1, 0)).abs()

    wx = torch.exp(-img_dx)
    wy = torch.exp(-img_dy)

    loss = (wx * disp_dx + wy * disp_dy).mean()
    return loss
