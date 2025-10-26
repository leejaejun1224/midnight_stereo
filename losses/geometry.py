import torch
import torch.nn.functional as F

def _make_base_grid(b, h, w, device, dtype):
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing='ij'
    )
    xx = xx.unsqueeze(0).expand(b, -1, -1)  # [B,H,W]
    yy = yy.unsqueeze(0).expand(b, -1, -1)  # [B,H,W]
    return xx, yy

def warp_right_to_left(x_right, disp_left, padding_mode='border', align_corners=True):
    """
    Generic warp: right->[left] for tensor (이미지/특징 공통)
    x_right: [B,C,H,W], disp_left: [B,1,H,W] (픽셀 단위, +는 좌로 이동)
    return: warped [B,C,H,W], valid_mask [B,1,H,W]
    """
    b, c, h, w = x_right.shape
    xx, yy = _make_base_grid(b, h, w, x_right.device, x_right.dtype)
    d = disp_left[:, 0]  # [B,H,W]
    x_src = xx - d
    y_src = yy

    x_norm = 2.0 * (x_src / (w - 1)) - 1.0
    y_norm = 2.0 * (y_src / (h - 1)) - 1.0
    grid = torch.stack((x_norm, y_norm), dim=-1)  # [B,H,W,2]

    warped = F.grid_sample(
        x_right, grid, mode='bilinear',
        padding_mode=padding_mode, align_corners=align_corners
    )
    valid = (x_src >= 0) & (x_src <= w - 1)
    return warped, valid.unsqueeze(1).float()

def warp_right_to_left_image(img_right, disp_left, padding_mode='border', align_corners=True):
    """
    호출부 호환용 래퍼 (이름 유지)
    img_right: [B,3,H,W] in [0,1]
    """
    return warp_right_to_left(img_right, disp_left, padding_mode, align_corners)
