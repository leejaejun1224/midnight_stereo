# loss.py
# ------------------------------------------------------------
# Stereo loss 계산 모듈
# - 포토메트릭(L1+SSIM)
# - DINO ViT-B/8 기반 flat 패치 마스크 -> edge-aware smoothness 강화
# - 방향 제약(soft disparity)
# ------------------------------------------------------------
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# 공용 유틸
# ---------------------------

def _shift_with_mask(x: torch.Tensor, dy: int, dx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    B, C, H, W = x.shape
    pt, pb = max(dy, 0), max(-dy, 0)
    pl, pr = max(dx, 0), max(-dx, 0)
    x_pad = F.pad(x, (pl, pr, pt, pb))
    x_shift = x_pad[:, :, pb:pb + H, pl:pl + W]
    valid = torch.ones((B, 1, H, W), device=x.device, dtype=x.dtype)
    if dy > 0:   valid[:, :, :dy, :] = 0
    if dy < 0:   valid[:, :, H + dy:, :] = 0
    if dx > 0:   valid[:, :, :, :dx] = 0
    if dx < 0:   valid[:, :, :, W + dx:] = 0
    return x_shift, valid


# ---------------------------
# SSIM / Photometric
# ---------------------------

class SSIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
    def forward(self, x, y):
        x = self.refl(x); y = self.refl(y)
        mu_x = self.mu_x_pool(x); mu_y = self.mu_y_pool(y)
        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        num = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        den = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - num / den) / 2, 0, 1)

class PhotometricLoss(nn.Module):
    def __init__(self, l1_w: float = 0.15, ssim_w: float = 0.85):
        super().__init__()
        self.l1_w = l1_w
        self.ssim_w = ssim_w
        self.ssim = SSIM()
    def forward(self, original_image: torch.Tensor, reconstructed_image: torch.Tensor) -> torch.Tensor:
        l1_loss = torch.abs(original_image - reconstructed_image).mean(1, True)   # [B,1,H,W]
        ssim_loss = self.ssim(original_image, reconstructed_image).mean(1, True)  # [B,1,H,W]
        return self.l1_w * l1_loss + self.ssim_w * ssim_loss


# ---------------------------
# Right->Left warping (half-res)
# ---------------------------

def warp_right_to_left_image(imgR: torch.Tensor, disp_px: torch.Tensor, align_corners=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    imgR:   [B,3,H,W] in [0,1], half 해상도
    disp_px:[B,1,H,W] in 'half-resolution pixel' units
    """
    B, C, H, W = imgR.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=imgR.device),
        torch.linspace(-1, 1, W, device=imgR.device),
        indexing="ij"
    )
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # [B,H,W,2]
    shift_norm = 2.0 * disp_px.squeeze(1) / max(W - 1, 1)
    grid = base_grid.clone()
    grid[..., 0] = grid[..., 0] - shift_norm
    img_w = F.grid_sample(imgR, grid, mode='bilinear', padding_mode='zeros', align_corners=align_corners)
    ones = torch.ones((B, 1, H, W), device=imgR.device)
    M = F.grid_sample(ones, grid, mode='nearest', padding_mode='zeros', align_corners=align_corners)
    valid = (M > 0.5).float()
    return img_w, valid


# ---------------------------
# 방향 제약(soft disp)
# ---------------------------

@torch.no_grad()
def _per_patch_min_similarity(feat_norm: torch.Tensor, sample_k: int = 512):
    B, C, H, W = feat_norm.shape
    P = H * W
    K = min(sample_k, P)
    F_bcp = feat_norm.view(B, C, P)
    F_bpc = F_bcp.permute(0, 2, 1).contiguous()
    idx = torch.randperm(P, device=feat_norm.device)[:K]
    bank = F_bcp[:, :, idx]                     # [B,C,K]
    sims = torch.bmm(F_bpc, bank)               # [B,P,K]
    min_vals = sims.min(dim=-1).values.view(B, 1, H, W)
    return min_vals

def _rel_gate_from_sim_dynamic(sim_raw: torch.Tensor, min_vals: torch.Tensor, valid: torch.Tensor,
                               thr: float = 0.75, gamma: float = 0.0,
                               use_dynamic_thr: bool = True, dynamic_q: float = 0.7):
    eps = 1e-6
    sim_norm = (sim_raw - min_vals) / (1.0 - min_vals + eps)
    sim_norm = sim_norm.clamp(0.0, 1.0)
    if use_dynamic_thr:
        v = sim_norm[valid > 0]
        thr_eff = torch.quantile(v, dynamic_q).item() if v.numel() > 0 else thr
    else:
        thr_eff = thr
    if gamma is None or gamma <= 0.0:
        w = (sim_norm >= thr_eff).to(sim_norm.dtype)
    else:
        w = torch.sigmoid((sim_norm - thr_eff) / gamma)
    return w * valid

class DirectionalRelScaleDispLoss(nn.Module):
    def __init__(self,
                 sim_thr=0.75, sim_gamma=0.0, sample_k=512,
                 use_dynamic_thr=True, dynamic_q=0.7,
                 vert_margin=1.0, horiz_margin=0.0,
                 lambda_v=1.0, lambda_h=0.0, huber_delta=1.0):
        super().__init__()
        self.vert_pairs = [(1,0), (-1,0)]
        self.hori_pairs = [(0,1), (0,-1)]
        self.sim_thr, self.sim_gamma = sim_thr, sim_gamma
        self.sample_k = sample_k
        self.use_dynamic_thr = use_dynamic_thr
        self.dynamic_q = dynamic_q
        self.vert_margin, self.horiz_margin = vert_margin, horiz_margin
        self.lambda_v, self.lambda_h = lambda_v, lambda_h
        self.huber_delta = huber_delta

    def _accum(self, disp, feat, roi, pairs, margin):
        with torch.no_grad():
            min_vals = _per_patch_min_similarity(feat, sample_k=self.sample_k)
        loss_sum = torch.tensor(0.0, device=disp.device)
        weight_sum = torch.tensor(0.0, device=disp.device)
        for dy, dx in pairs:
            f_nb, valid_b = _shift_with_mask(feat, dy, dx)
            d_nb, _       = _shift_with_mask(disp, dy, dx)
            roi_nb, _     = _shift_with_mask(roi,  dy, dx)
            valid = valid_b * roi * roi_nb
            sim_raw = (feat * f_nb).sum(dim=1, keepdim=True)
            w = _rel_gate_from_sim_dynamic(sim_raw, min_vals, valid,
                                           self.sim_thr, self.sim_gamma,
                                           self.use_dynamic_thr, self.dynamic_q)
            diff = (disp - d_nb).abs()
            small = (diff < self.huber_delta).float()
            viol = 0.5 * (diff**2) / (self.huber_delta + 1e-6) * small + (diff - 0.5*self.huber_delta) * (1 - small)
            viol = (viol - margin).clamp(min=0.0) if margin > 0 else viol
            loss_sum   += (w * viol).sum()
            weight_sum += w.sum()
        return loss_sum / (weight_sum + 1e-6)

    def forward(self, disp, feat, roi):
        loss_v = self._accum(disp, feat, roi, self.vert_pairs, self.vert_margin)
        loss_h = self._accum(disp, feat, roi, self.hori_pairs, self.horiz_margin)
        return self.lambda_v * loss_v + self.lambda_h * loss_h


# ---------------------------
# DINO 기반 flat 패치 마스크 (입력: FL @1/8)
# ---------------------------

@torch.no_grad()
def compute_flat_patch_mask(FL: torch.Tensor, thr: float = 0.9, radius: int = 1) -> torch.Tensor:
    """
    FL: [B,C,H8,W8]  (ℓ2 정규화 특징 권장 / 모델 aux['FL'])
    thr: 코사인 유사도 임계
    radius: 방향 탐색 반경 (패치 셀 단위)
    return: [B,1,H8,W8] 0/1
    """
    FLn = F.normalize(FL, dim=1, eps=1e-6)

    def dir_ok(dy: int, dx: int) -> torch.Tensor:
        acc = None
        for r in range(1, radius + 1):
            f_nb, valid = _shift_with_mask(FLn, dy*r, dx*r)
            sim = (FLn * f_nb).sum(dim=1, keepdim=True)
            ok = (sim >= thr).to(sim.dtype) * valid
            acc = ok if acc is None else torch.maximum(acc, ok)
        return acc if acc is not None else torch.zeros_like(FLn[:, :1])

    up    = dir_ok(-1,  0)
    down  = dir_ok( 1,  0)
    left  = dir_ok( 0, -1)
    right = dir_ok( 0,  1)

    flat = (up * down * left * right)
    return flat


# ---------------------------
# Edge-aware smoothness (+ flat boost)
# ---------------------------

def edge_aware_smooth_with_flatmask(
    disp: torch.Tensor,                  # [B,1,Hh,Wh] half-res disparity (px)
    img: torch.Tensor,                   # [B,3,Hh,Wh] half-res image in [0,1]
    flat_mask_patch: Optional[torch.Tensor],  # [B,1,H8,W8] or None
    roi: Optional[torch.Tensor] = None,  # [B,1,Hh,Wh] or None
    weight_min: float = 0.05,
    use_charb: bool = True,
    charb_eps: float = 1e-3,
    flat_boost: float = 2.0,
) -> torch.Tensor:
    """Edge-aware smoothness with DINO flat-mask guided strong smoothing."""
    # disparity gradients
    grad_disp_x = torch.abs(disp[:, :, :, 1:] - disp[:, :, :, :-1])  # [B,1,H, W-1]
    grad_disp_y = torch.abs(disp[:, :, 1:, :] - disp[:, :, :-1, :])  # [B,1,H-1,W]

    # image gradients (channel-mean L1)
    grad_img_x = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]), 1, keepdim=True)

    # base weights from edges
    w_x = torch.exp(-grad_img_x).clamp_min(weight_min)
    w_y = torch.exp(-grad_img_y).clamp_min(weight_min)

    # flat mask upsample to half-res and align with gradient locations
    if flat_mask_patch is not None:
        flat_half = F.interpolate(flat_mask_patch, size=img.shape[-2:], mode="nearest")  # [B,1,H, W]
        flat_x = (flat_half[:, :, :, 1:] * flat_half[:, :, :, :-1])  # only where both pixels are flat
        flat_y = (flat_half[:, :, 1:, :] * flat_half[:, :, :-1, :])
        # (1) edge=0 -> weight=1 (최소한의 강제), (2) flat_boost로 페널티 강화
        w_x = torch.where(flat_x > 0.5, torch.ones_like(w_x), w_x)
        w_y = torch.where(flat_y > 0.5, torch.ones_like(w_y), w_y)
        if flat_boost is not None and flat_boost > 1.0:
            w_x = torch.where(flat_x > 0.5, w_x * flat_boost, w_x)
            w_y = torch.where(flat_y > 0.5, w_y * flat_boost, w_y)

    # robust (Charbonnier) penalty (optional)
    if use_charb:
        grad_disp_x = torch.sqrt(grad_disp_x**2 + charb_eps**2)
        grad_disp_y = torch.sqrt(grad_disp_y**2 + charb_eps**2)

    term_x = w_x * grad_disp_x
    term_y = w_y * grad_disp_y

    if roi is not None:
        roi_x = roi[:, :, :, 1:] * roi[:, :, :, :-1]
        roi_y = roi[:, :, 1:, :] * roi[:, :, :-1, :]
        term_x = term_x * roi_x
        term_y = term_y * roi_y
        denom = (roi_x.sum() + roi_y.sum() + 1e-6)
        return (term_x.sum() + term_y.sum()) / denom
    else:
        return term_x.mean() + term_y.mean()


# ---------------------------
# LossComputer: 모든 손실 통합 계산
# ---------------------------

class LossComputer(nn.Module):
    def __init__(self, args, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device

        # directional soft-disp loss (patch scale)
        self.dir_loss = DirectionalRelScaleDispLoss(
            sim_thr=args.sim_thr,
            sim_gamma=args.sim_gamma,
            sample_k=args.sim_sample_k,
            use_dynamic_thr=args.use_dynamic_thr,
            dynamic_q=args.dynamic_q,
            vert_margin=1.0, horiz_margin=0.0,
            lambda_v=args.lambda_v, lambda_h=args.lambda_h, huber_delta=1.0
        )

        # photometric
        self.photo = PhotometricLoss(l1_w=args.photo_l1_w, ssim_w=args.photo_ssim_w)

    def compute_losses(self,
                       prob: torch.Tensor,
                       disp_soft_patch: torch.Tensor,
                       aux: Dict[str, torch.Tensor],
                       imgL_half_enh_01: torch.Tensor,
                       imgR_half_01: torch.Tensor,
                       roi_patch: torch.Tensor,
                       roi_half: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        필수 aux 키:
          - FL: [B,C,H/8,W/8]
          - FR: [B,C,H/8,W/8] (미사용)
          - disp_half_px: [B,1,H/2,W/2]
          - raw_volume, mask, refined_masked (미사용/선택)
        """
        FL   = aux.get("FL", None)
        disp_half_px = aux["disp_half_px"]

        # photometric (half-res)
        with torch.no_grad():
            imgR_half_warp_01, valid_half = warp_right_to_left_image(imgR_half_01, disp_half_px)
        photo_map = self.photo(imgL_half_enh_01, imgR_half_warp_01)  # [B,1,H/2,W/2]
        photo_mask = roi_half * valid_half
        loss_photo = (photo_map * photo_mask).sum() / (photo_mask.sum() + 1e-6)

        # directional loss (patch scale)
        loss_dir = self.dir_loss(disp_soft_patch, FL, roi_patch)

        # DINO flat mask → smoothness
        if self.args.use_dino_flat_smooth and (FL is not None):
            flat_patch_mask = compute_flat_patch_mask(FL, thr=self.args.flat_thr, radius=self.args.flat_radius)
        else:
            flat_patch_mask = None

        loss_smooth = edge_aware_smooth_with_flatmask(
            disp=disp_half_px,
            img=imgL_half_enh_01,
            flat_mask_patch=flat_patch_mask,
            roi=roi_half,
            weight_min=self.args.smooth_weight_min,
            use_charb=self.args.smooth_use_charb,
            charb_eps=self.args.smooth_charb_eps,
            flat_boost=self.args.smooth_flat_boost
        )

        # 가중합
        total = loss_dir + self.args.w_photo * loss_photo + self.args.w_smooth * loss_smooth

        logs = {
            "dir": float(loss_dir.detach().cpu().item()),
            "photo": float(loss_photo.detach().cpu().item()),
            "smooth": float(loss_smooth.detach().cpu().item()),
            "total": float(total.detach().cpu().item()),
        }
        return total, logs
