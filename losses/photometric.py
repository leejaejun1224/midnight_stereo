import torch
import torch.nn as nn
import torch.nn.functional as F

def _gaussian_window(window_size=11, sigma=1.5, channels=3, device='cpu', dtype=torch.float32):
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(1)   # [K,1]
    kernel_2d = (g @ g.t())          # [K,K]
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel_2d = kernel_2d.view(1, 1, window_size, window_size)
    kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)
    return kernel_2d

class SSIM(nn.Module):
    """
    안정형 SSIM: 가우시안 창 + eps 가드
    x,y는 [0, data_range] 범위
    """
    def __init__(self, window_size=11, sigma=1.5, data_range=1.0, channels=3, eps=1e-6):
        super().__init__()
        self.data_range = float(data_range)
        self.C1 = (0.01 * self.data_range) ** 2
        self.C2 = (0.03 * self.data_range) ** 2
        self.eps = eps
        window = _gaussian_window(window_size, sigma, channels)
        self.register_buffer('window', window)
        self.pad = window_size // 2

    def forward(self, x, y):
        C = x.size(1)
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        y_pad = F.pad(y, (self.pad, self.pad, self.pad, self.pad), mode='reflect')

        mu_x = F.conv2d(x_pad, self.window, groups=C)
        mu_y = F.conv2d(y_pad, self.window, groups=C)

        sigma_x  = F.conv2d(F.pad(x*x, (self.pad,)*4, mode='reflect'), self.window, groups=C) - mu_x*mu_x
        sigma_y  = F.conv2d(F.pad(y*y, (self.pad,)*4, mode='reflect'), self.window, groups=C) - mu_y*mu_y
        sigma_xy = F.conv2d(F.pad(x*y, (self.pad,)*4, mode='reflect'), self.window, groups=C) - mu_x*mu_y

        sigma_x = torch.clamp(sigma_x, min=0.0)
        sigma_y = torch.clamp(sigma_y, min=0.0)

        num = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        den = (mu_x*mu_x + mu_y*mu_y + self.C1) * (sigma_x + sigma_y + self.C2) + self.eps
        ssim_map = num / den
        ssim_map = torch.clamp(ssim_map, -1.0, 1.0)
        return (1.0 - ssim_map) * 0.5  # [B,C,H,W] in [0,1]

class PhotometricLoss(nn.Module):
    """
    호출부 호환: simple_photometric_loss(original_image, reconstructed_image, weights=[w_l1, w_ssim])
    - 내부적으로 L1은 Charbonnier, SSIM은 위 구현 사용.
    """
    def __init__(self, w_l1=0.15, w_ssim=0.85, data_range=1.0, charbonnier_eps=1e-3):
        super().__init__()
        self.w_l1 = float(w_l1)
        self.w_ssim = float(w_ssim)
        self.charb_eps = float(charbonnier_eps)
        self.ssim = SSIM(data_range=data_range)

    def _charbonnier_l1(self, diff):
        # [B,C,H,W] -> [B,1,H,W] (채널 평균 유지)
        c = (diff*diff + self.charb_eps*self.charb_eps).sqrt()
        return c.mean(1, keepdim=True)

    @torch.no_grad()
    def _validate_range(self, x):
        # 경고성 체크만, 실패해도 학습은 계속
        if x.min().item() < -1e-3 or x.max().item() > 1.0 + 1e-3:
            pass

    def simple_photometric_loss(self, original_image, reconstructed_image, weights=None):
        """
        original_image, reconstructed_image: [B,3,H,W] in [0,1]
        return: [B,1,H,W] (호출부 동일)
        """
        self._validate_range(original_image)
        diff = original_image - reconstructed_image
        l1 = self._charbonnier_l1(diff)                           # [B,1,H,W]
        ssim_map = self.ssim(original_image, reconstructed_image).mean(1, keepdim=True)  # [B,1,H,W]
        w = weights if (weights is not None) else [self.w_l1, self.w_ssim]
        return w[0] * l1 + w[1] * ssim_map
