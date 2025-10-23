import os
import glob
import argparse
import random
from typing import Tuple, List, Dict

import warnings
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from datetime import datetime, timezone, timedelta
# from modeljj import StereoModel
from modeljj_reassemble import StereoModel
from stereo_dpt import DINOv1Base8Backbone, StereoDPTHead, DPTStereoTrainCompat
from prop_utils import *
from logger import *
from sky_loss import *
# ---------------------------
# 유틸
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stem(path: str) -> str:
    s = os.path.splitext(os.path.basename(path))[0]
    return s

@torch.no_grad()
def denorm_imagenet(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B,3,H,W], imagenet norm (mean,std) 적용된 텐서
    return: [B,3,H,W] in [0,1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
    y = x * std + mean
    return y.clamp(0.0, 1.0)


# ---------------------------
# 데이터셋 (+ 하늘 마스크)
# ---------------------------

class StereoFolderDataset(Dataset):
    """
    - left_dir/right_dir: 동일 파일명 매칭
    - mask_dir: 같은 파일명(stem)이 있는 샘플만 사용, 하늘(흰색=255) 제외 마스크 제공
    """
    def __init__(self, left_dir: str, right_dir: str,
                 height: int = 384, width: int = 1224,
                 mask_dir: str = None):
        left_all  = sorted(glob.glob(os.path.join(left_dir,  "*")))
        right_all = sorted(glob.glob(os.path.join(right_dir, "*")))
        assert len(left_all) == len(right_all) and len(left_all) > 0, "좌/우 이미지 수 불일치 또는 비어 있음"
        for lp, rp in zip(left_all, right_all):
            if stem(lp) != stem(rp):
                raise ValueError(f"파일명(stem) 불일치: {lp} vs {rp}")

        self.height = height
        self.width  = width

        self.to_tensor = transforms.Compose([
            transforms.Resize((height, width), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225]),
        ])
        self.mask_resize = transforms.Resize((height, width), interpolation=InterpolationMode.NEAREST)

        self.mask_dir = mask_dir
        self.left_paths: List[str] = []
        self.right_paths: List[str] = []
        self.mask_paths: List[str] = []

        if mask_dir is None:
            self.left_paths  = left_all
            self.right_paths = right_all
            self.mask_paths  = [None] * len(self.left_paths)
        else:
            mask_all = sorted(glob.glob(os.path.join(mask_dir, "*")))
            mask_map: Dict[str, str] = {stem(p): p for p in mask_all}
            for lp, rp in zip(left_all, right_all):
                st = stem(lp)
                if st in mask_map:
                    self.left_paths.append(lp)
                    self.right_paths.append(rp)
                    self.mask_paths.append(mask_map[st])
            if len(self.left_paths) == 0:
                raise ValueError(f"mask_dir={mask_dir} 에 매칭되는 파일이 없습니다.")

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_img  = Image.open(self.left_paths[idx]).convert("RGB")
        right_img = Image.open(self.right_paths[idx]).convert("RGB")
        left_t  = self.to_tensor(left_img)
        right_t = self.to_tensor(right_img)

        if self.mask_paths[idx] is None:
            valid_full = torch.ones(1, self.height, self.width, dtype=torch.float32)
        else:
            m = Image.open(self.mask_paths[idx]).convert("L")
            m = self.mask_resize(m)
            m_t = transforms.ToTensor()(m)                  # [1,H,W], 0~1
            valid_full = (1.0 - (m_t >= 0.99).float())      # 하늘=0, 그 외=1

        return left_t, right_t, valid_full, os.path.basename(self.left_paths[idx])


def shift_with_mask(x: torch.Tensor, dy: int, dx: int):
    B, C, H, W = x.shape
    pt, pb = max(dy,0), max(-dy,0)
    pl, pr = max(dx,0), max(-dx,0)
    x_pad = F.pad(x, (pl, pr, pt, pb))
    x_shift = x_pad[:, :, pb:pb+H, pl:pl+W]
    valid = torch.ones((B,1,H,W), device=x.device, dtype=x.dtype)
    if dy > 0:   valid[:, :, :dy, :] = 0
    if dy < 0:   valid[:, :, H+dy:, :] = 0
    if dx > 0:   valid[:, :, :, :dx] = 0
    if dx < 0:   valid[:, :, :, W+dx:] = 0
    return x_shift, valid

def make_roi_patch(valid_full: torch.Tensor, patch_size: int, method: str = "avg", thr: float = 0.5):
    B, C, H, W = valid_full.shape
    assert C == 1
    H8, W8 = H // patch_size, W // patch_size
    if method == "nearest":
        roi = F.interpolate(valid_full, size=(H8, W8), mode="nearest")
    else:
        roi = F.avg_pool2d(valid_full, kernel_size=patch_size, stride=patch_size)
        roi = (roi >= thr).float()
    return roi


# ---------------------------
# 클릭식 유사도 게이트(상대 스케일 + (선택) 동적 임계)
# ---------------------------

@torch.no_grad()
def per_patch_min_similarity(feat_norm: torch.Tensor, sample_k: int = 512):
    B, C, H, W = feat_norm.shape
    P = H * W
    K = min(sample_k, P)
    F_bcp = feat_norm.view(B, C, P)             # [B,C,P]
    F_bpc = F_bcp.permute(0, 2, 1).contiguous() # [B,P,C]
    idx = torch.randperm(P, device=feat_norm.device)[:K]
    bank = F_bcp[:, :, idx]                     # [B,C,K]
    sims = torch.bmm(F_bpc, bank)               # [B,P,K]
    min_vals = sims.min(dim=-1).values.view(B, 1, H, W)
    return min_vals

def rel_gate_from_sim_dynamic(sim_raw: torch.Tensor, min_vals: torch.Tensor, valid: torch.Tensor,
                              thr: float = 0.75, gamma: float = 0.0, use_dynamic_thr: bool = True, dynamic_q: float = 0.7):
    eps = 1e-6
    sim_norm = (sim_raw - min_vals) / (1.0 - min_vals + eps)
    sim_norm = sim_norm.clamp(0.0, 1.0)
    if use_dynamic_thr:
        v = sim_norm[valid > 0]
        if v.numel() > 0:
            thr_eff = torch.quantile(v, dynamic_q).item()
        else:
            thr_eff = thr
    else:
        thr_eff = thr
    if gamma is None or gamma <= 0.0:
        w = (sim_norm >= thr_eff).to(sim_norm.dtype)
    else:
        w = torch.sigmoid((sim_norm - thr_eff) / gamma)
    return w * valid


# ---------------------------
# (A) 방향 제약(soft disp 기준)
# ---------------------------

class DirectionalRelScaleDispLoss(nn.Module):
    """
    세로(±1): |Δ|<=1, 가로(±1): |Δ|<=0 (lambda_h 기본 0.0; 샤픈 가로 일관성으로 대체 권장)
    """
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
            min_vals = per_patch_min_similarity(feat, sample_k=self.sample_k)
        loss_sum = torch.tensor(0.0, device=disp.device)
        weight_sum = torch.tensor(0.0, device=disp.device)
        for dy, dx in pairs:
            f_nb, valid_b = shift_with_mask(feat, dy, dx)
            d_nb, _       = shift_with_mask(disp, dy, dx)
            roi_nb, _     = shift_with_mask(roi,  dy, dx)
            valid = valid_b * roi * roi_nb
            sim_raw = (feat * f_nb).sum(dim=1, keepdim=True)
            w = rel_gate_from_sim_dynamic(sim_raw, min_vals, valid,
                                          self.sim_thr, self.sim_gamma,
                                          self.use_dynamic_thr, self.dynamic_q)
            diff = (disp - d_nb).abs()
            # Huber(δ)
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
# (B) 샤픈 가로 일관성 (ArgMax 근사)
# ---------------------------

class HorizontalSharpenedConsistency(nn.Module):
    def __init__(self, D, tau_sharp=0.2, huber_delta=0.25, use_fixed_denom=True,
                 sim_thr=0.75, sim_gamma=0.0, sample_k=512, use_dynamic_thr=True, dynamic_q=0.7):
        super().__init__()
        self.tau = tau_sharp
        self.delta = huber_delta
        self.use_fixed_denom = use_fixed_denom
        self.register_buffer("disp_values",
            torch.arange(0, D+1, dtype=torch.float32).view(1,1,D+1,1,1))
        # 게이트 설정
        self.sim_thr, self.sim_gamma = sim_thr, sim_gamma
        self.sample_k = sample_k
        self.use_dynamic_thr, self.dynamic_q = use_dynamic_thr, dynamic_q

    def forward(self, refined_logits_masked, feat, roi):
        # 샤픈 분포 → disp_sharp(≈ArgMax 기대값)
        p_sharp = torch.softmax(refined_logits_masked / self.tau, dim=2)      # [B,1,D+1,H,W]
        disp_sharp = (p_sharp * self.disp_values).sum(dim=2)                  # [B,1,H,W]

        # 한 번만 min_vals 계산
        with torch.no_grad():
            min_vals = per_patch_min_similarity(feat, sample_k=self.sample_k)

        loss_sum, denom = 0.0, 0.0
        for dy, dx in [(0,1),(0,-1)]:
            d_nb, valid_b = shift_with_mask(disp_sharp, dy, dx)
            roi_nb, _     = shift_with_mask(roi, dy, dx)
            valid = valid_b * roi * roi_nb

            f_nb, _  = shift_with_mask(feat, dy, dx)
            sim_raw  = (feat * f_nb).sum(dim=1, keepdim=True)
            w_sim = rel_gate_from_sim_dynamic(sim_raw, min_vals, valid,
                                              self.sim_thr, self.sim_gamma,
                                              self.use_dynamic_thr, self.dynamic_q)

            diff = (disp_sharp - d_nb).abs()
            small = (diff < self.delta).float()
            viol  = 0.5 * (diff**2) / (self.delta + 1e-6) * small + (diff - 0.5*self.delta) * (1 - small)

            loss_sum += (w_sim * viol).sum()
            denom    += (roi if self.use_fixed_denom else w_sim).sum()

        return loss_sum / (denom + 1e-6)


# ---------------------------
# (C) 분포-수준 일치(JS 유사), (D) 엔트로피 감소
# ---------------------------

def _shift5_spatial(x5, dy, dx):
    # x5: [B,1_orC,D+1,H,W]
    B,C,D,H,W = x5.shape
    pt, pb = max(dy,0), max(-dy,0)
    pl, pr = max(dx,0), max(-dx,0)
    x_pad = F.pad(x5, (pl, pr, pt, pb, 0, 0))
    return x_pad[:, :, :, pb:pb+H, pl:pl+W]

def _shift_along_disp(p, s):
    # p: [B,1,D+1,H,W]
    if s == 0: return p
    if s > 0:
        pad = torch.zeros_like(p[:, :, :s])
        return torch.cat([pad, p[:, :, :-s]], dim=2)
    else:
        s = -s
        pad = torch.zeros_like(p[:, :, :s])
        return torch.cat([p[:, :, s:], pad], dim=2)

def _sym_kl(p, q, eps=1e-8):
    p = p.clamp_min(eps); q = q.clamp_min(eps)
    return (p * (p.log() - q.log()) + q * (q.log() - p.log())).sum(dim=2, keepdim=True)

class NeighborProbConsistencyLoss(nn.Module):
    def __init__(self,
                 sim_thr=0.6, sim_gamma=0.1, sample_k=1024,
                 allow_shift_v=1, allow_shift_h=0,
                 use_dynamic_thr=True, dynamic_q=0.7,
                 conf_alpha=1.0):
        super().__init__()
        self.sim_thr, self.sim_gamma = sim_thr, sim_gamma
        self.sample_k = sample_k
        self.allow_shift_v, self.allow_shift_h = allow_shift_v, allow_shift_h
        self.use_dynamic_thr, self.dynamic_q = use_dynamic_thr, dynamic_q
        self.conf_alpha = conf_alpha

    def _gate(self, feat, roi, dy, dx):
        f_nb, valid_b = shift_with_mask(feat, dy, dx)
        roi_nb, _ = shift_with_mask(roi, dy, dx)
        valid = valid_b * roi * roi_nb
        with torch.no_grad():
            min_vals = per_patch_min_similarity(feat, sample_k=self.sample_k)
        sim_raw = (feat * f_nb).sum(dim=1, keepdim=True)
        w_sim = rel_gate_from_sim_dynamic(sim_raw, min_vals, valid,
                                          self.sim_thr, self.sim_gamma,
                                          self.use_dynamic_thr, self.dynamic_q)
        return w_sim

    def forward(self, prob, feat, roi):
        # 신뢰도 게이트(top1-top2)
        with torch.no_grad():
            pp = prob.squeeze(1).clamp_min(1e-8)                 # [B,D+1,H,W]
            topv = torch.topk(pp, k=2, dim=1).values             # [B,2,H,W]
            conf = (topv[:,0] - topv[:,1]).unsqueeze(1)          # [B,1,H,W]
            conf = conf.clamp_min(0.0).pow(self.conf_alpha)

        loss_sum = torch.tensor(0.0, device=prob.device)
        weight_sum = torch.tensor(0.0, device=prob.device)
        p = prob  # [B,1,D+1,H,W]

        for (dy,dx,allow_shift) in [(-1,0,self.allow_shift_v), (1,0,self.allow_shift_v),
                                    (0,-1,self.allow_shift_h), (0,1,self.allow_shift_h)]:
            w_sim = self._gate(feat, roi, dy, dx)                 # [B,1,H,W]
            w_tot = w_sim * roi * conf
            if w_tot.sum() < 1e-6: continue

            q = _shift5_spatial(p, dy, dx)                        # [B,1,D+1,H,W]
            costs = []
            for s in range(-allow_shift, allow_shift+1):
                q_s = _shift_along_disp(q, s)
                skl = _sym_kl(p, q_s)                             # [B,1,1,H,W]
                costs.append(skl)
            cost_min = torch.stack(costs, dim=0).min(dim=0).values

            loss_sum  += (w_tot.unsqueeze(2) * cost_min).sum()
            weight_sum+= w_tot.sum()

        return loss_sum / (weight_sum + 1e-6)

class EntropySharpnessLoss(nn.Module):
    def __init__(self, conf_alpha=1.0,
                 sim_thr=0.6, sim_gamma=0.1, sample_k=512,
                 use_dynamic_thr=True, dynamic_q=0.7):
        super().__init__()
        self.nb_gate = NeighborProbConsistencyLoss(sim_thr, sim_gamma, sample_k,
                                                   allow_shift_v=0, allow_shift_h=0,
                                                   use_dynamic_thr=use_dynamic_thr, dynamic_q=dynamic_q,
                                                   conf_alpha=conf_alpha)

    def forward(self, prob, feat, roi):
        with torch.no_grad():
            w_left  = self.nb_gate._gate(feat, roi, 0, -1)
            w_right = self.nb_gate._gate(feat, roi, 0,  1)
            w_up    = self.nb_gate._gate(feat, roi, -1, 0)
            w_down  = self.nb_gate._gate(feat, roi,  1, 0)
            w = (w_left + w_right + w_up + w_down).clamp(max=1.0)
        p = prob.squeeze(1).clamp_min(1e-8)                         # [B,D+1,H,W]
        ent = -(p * p.log()).sum(dim=1, keepdim=True)               # [B,1,H,W]
        loss = (w * roi * ent).sum() / ((w * roi).sum() + 1e-6)
        return loss


# ---------------------------
# (E) 앵커, (F) 재투영
# ---------------------------

class CorrAnchorLoss(torch.nn.Module):
    def __init__(self, tau=0.6, margin=1.0, topk=2, use_huber=True):
        super().__init__()
        self.tau = tau; self.m = margin; self.k = topk; self.use_huber = use_huber
    def forward(self, raw_vol, disp, mask=None, roi=None):
        s = raw_vol.squeeze(1).detach()  # [B,D+1,H',W']
        if mask is not None:
            m = mask.squeeze(1)
            s = s + (1.0 - m) * (-1e4)
        topv, topd = torch.topk(s, k=self.k, dim=1)  # [B,K,H',W']
        w = ((topv - self.tau) / (1.0 - self.tau)).clamp(min=0.0, max=1.0)
        if roi is not None: w = w * roi
        disp_exp = disp.repeat(1, self.k, 1, 1)
        d_anchor = topd.float()
        diff = (disp_exp - d_anchor).abs()
        viol = (diff - self.m).clamp(min=0.0)
        if self.use_huber:
            small = (viol < 1.0).float()
            viol = 0.5*(viol**2)*small + (viol - 0.5)*(1-small)
        return (w * viol).sum() / (w.sum() + 1e-6)

def warp_right_to_left_feat(FR, disp_patch, align_corners=True):
    B, C, H, W = FR.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=FR.device),
        torch.linspace(-1, 1, W, device=FR.device),
        indexing="ij"
    )
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B,1,1,1)  # [B,H,W,2]
    shift_norm = 2.0 * disp_patch.squeeze(1) / max(W-1, 1)
    grid = base_grid.clone()
    grid[..., 0] = grid[..., 0] - shift_norm
    FR_w = F.grid_sample(FR, grid, mode='bilinear', padding_mode='zeros', align_corners=align_corners)
    ones = torch.ones((B,1,H,W), device=FR.device)
    M = F.grid_sample(ones, grid, mode='nearest', padding_mode='zeros', align_corners=align_corners)
    valid = (M > 0.5).float()
    return FR_w, valid

def warp_right_to_left_image(imgR, disp_px, align_corners=True):
    """
    imgR: [B,3,H,W] in [0,1], half 해상도
    disp_px: [B,1,H,W] in 'half-resolution pixel' units
    """
    B, C, H, W = imgR.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=imgR.device),
        torch.linspace(-1, 1, W, device=imgR.device),
        indexing="ij"
    )
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B,1,1,1)  # [B,H,W,2]
    shift_norm = 2.0 * disp_px.squeeze(1) / max(W-1, 1)
    grid = base_grid.clone()
    grid[..., 0] = grid[..., 0] - shift_norm
    img_w = F.grid_sample(imgR, grid, mode='bilinear', padding_mode='zeros', align_corners=align_corners)
    ones = torch.ones((B,1,H,W), device=imgR.device)
    M = F.grid_sample(ones, grid, mode='nearest', padding_mode='zeros', align_corners=align_corners)
    valid = (M > 0.5).float()
    return img_w, valid


class FeatureReprojLoss(torch.nn.Module):
    def __init__(self): super().__init__()
    def forward(self, FL, FR, disp_patch, roi=None):
        FR_w, valid = warp_right_to_left_feat(FR, disp_patch)
        if roi is not None: valid = valid * roi
        cos = (FL * FR_w).sum(dim=1, keepdim=True)
        loss = (1.0 - cos).clamp(min=0.0) * valid
        return loss.sum() / (valid.sum() + 1e-6)


# ---------------------------
# Photometric / Smoothness (제공 코드 통합)
# ---------------------------

def enhance_low_light_bgr(img_bgr: np.ndarray,
                          enable: bool = True,
                          gamma: float = 1.8,
                          clahe_clip: float = 2.0,
                          clahe_tile: int = 8) -> np.ndarray:
    if not enable:
        return img_bgr
    img = img_bgr.copy().astype(np.float32) / 255.0
    img = np.power(np.clip(img, 0, 1), 1.0 / max(gamma, 1e-6))
    img = (img * 255.0).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(int(clahe_tile), int(clahe_tile)))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    img2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return img2

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class PhotometricLoss:
    def __init__(self, weights = [0.15,0.85]):
        self.weights = weights
        self.ssim = SSIM()
    def simple_photometric_loss(self,original_image, reconstructed_image, weights = [0.15,0.85]):
        l1_loss = torch.abs(original_image - reconstructed_image).mean(1,True)
        ssim_loss = self.ssim(original_image, reconstructed_image).mean(1,True)
        losses = [l1_loss, ssim_loss]
        weighted_loss = 0
        for i in range(len(weights)):
            weighted_loss += weights[i] * losses[i]
        return weighted_loss

class MultiScalePhotometricLoss(nn.Module):
    def __init__(self, full_scale = False):
        super(MultiScalePhotometricLoss, self).__init__()
        self.ssim = SSIM()
        self.full_scale = full_scale
    def simple_loss(self,original_image, reconstructed_image, weights = [0.15,0.85]):
        assert original_image.shape == reconstructed_image.shape
        l1_loss = torch.abs(original_image - reconstructed_image).mean(1,True)
        ssim_loss = self.ssim(original_image, reconstructed_image).mean(1,True)
        losses = [l1_loss, ssim_loss]
        weighted_loss = 0
        for i in range(len(weights)):
            weighted_loss += weights[i] * losses[i]
        return weighted_loss
    def forward(self, reconstructed_images,original_images, reduce_mean = True):
        assert len(reconstructed_images) == len(original_images)
        total_loss = 0.0
        for original, recon in zip(original_images, reconstructed_images):
            if self.full_scale:
                loss = self.simple_loss(original_images[0], recon)
            else:
                loss = self.simple_loss(original, recon)
            total_loss += loss.mean()
        total_loss = total_loss / len(original_images)
        return total_loss

def get_disparity_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def multi_scale_disparity_smooth_loss(multi_scale_disparities,
                                      multi_scale_images, weight = 1e-3):
    loss = 0
    for scale in range(len(multi_scale_disparities)):
        loss += weight * get_disparity_smooth_loss(
            multi_scale_disparities[scale], multi_scale_images[scale]
        )
    return loss

def enhance_batch_bgr_from_rgb01(img_rgb_01: torch.Tensor,
                                 enable: bool,
                                 gamma: float,
                                 clahe_clip: float,
                                 clahe_tile: int) -> torch.Tensor:
    """
    img_rgb_01: [B,3,H,W] in [0,1]
    return: enhanced RGB [B,3,H,W] in [0,1]
    """
    B, C, H, W = img_rgb_01.shape
    out_list = []
    img_cpu = img_rgb_01.detach().cpu().clamp(0,1).numpy()
    for b in range(B):
        rgb = (img_cpu[b].transpose(1,2,0) * 255.0).astype(np.uint8)  # H,W,3 RGB uint8
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr_enh = enhance_low_light_bgr(bgr, enable=enable, gamma=gamma,
                                        clahe_clip=clahe_clip, clahe_tile=clahe_tile)
        rgb_enh = cv2.cvtColor(bgr_enh, cv2.COLOR_BGR2RGB)
        out_list.append(torch.from_numpy(rgb_enh.astype(np.float32) / 255.0).permute(2,0,1))
    out = torch.stack(out_list, dim=0)
    return out.to(img_rgb_01.device)


# ---------------------------
# Convex Upsample (RAFT-style)
# ---------------------------



# ---------------------------
# 전체 모델
# ---------------------------

# ---------------------------
# 옵티마이저 & 체크포인트 유틸
# ---------------------------

def build_optimizer(params, name='adamw', lr=1e-3, weight_decay=1e-2):
    name = name.lower()
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    elif name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    else:
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)


def _move_optimizer_state_to_device(optim: torch.optim.Optimizer, device: torch.device):
    """optimizer state 텐서들을 device로 이동 (CPU ckpt 로드시 안전)"""
    for state in optim.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def save_checkpoint(path: str, epoch: int, model: nn.Module,
                    optim: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler,
                    args: argparse.Namespace):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "args": vars(args),
        "scaler": (scaler.state_dict() if scaler is not None else None),
        "rng_python": random.getstate(),
        "rng_numpy":  np.random.get_state(),
        "rng_torch":  torch.get_rng_state(),
        "rng_torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "time_saved": datetime.now().isoformat(),
    }
    torch.save(ckpt, path)

def resume_from_checkpoint(args, model: nn.Module, optim: torch.optim.Optimizer,
                           scaler: torch.cuda.amp.GradScaler, device: torch.device) -> int:
    """
    Returns:
      start_epoch: 재시작할 epoch (보통 ckpt_epoch+1)
    """
    assert args.resume is not None and os.path.isfile(args.resume), f"checkpoint 없음: {args.resume}"
    print(f"[Resume] 로드: {args.resume}")
    ckpt = torch.load(args.resume, map_location="cpu")

    # 1) 모델 가중치
    strict = not args.resume_non_strict
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)
    if not strict:
        if missing:
            warnings.warn(f"[Resume] 누락된 키 {len(missing)}개: {missing[:10]}{' ...' if len(missing)>10 else ''}")
        if unexpected:
            warnings.warn(f"[Resume] 예기치 않은 키 {len(unexpected)}개: {unexpected[:10]}{' ...' if len(unexpected)>10 else ''}")

    # 2) 옵티마이저
    if (not args.resume_reset_optim) and ("optim" in ckpt):
        optim.load_state_dict(ckpt["optim"])
        _move_optimizer_state_to_device(optim, device)
        print("[Resume] optimizer 상태 복구")
    else:
        print("[Resume] optimizer 상태 초기화(미복구)")

    # 3) AMP GradScaler
    if scaler is not None and (not args.resume_reset_scaler) and ("scaler" in ckpt) and (ckpt["scaler"] is not None):
        try:
            scaler.load_state_dict(ckpt["scaler"])
            print("[Resume] GradScaler 상태 복구")
        except Exception as e:
            warnings.warn(f"[Resume] GradScaler 상태 복구 실패 → 무시: {e}")
    else:
        print("[Resume] GradScaler 상태 초기화(미복구)")

    # 4) RNG (선택 복구: 실험 재현성 필요 시)
    try:
        if "rng_python" in ckpt: random.setstate(ckpt["rng_python"])
        if "rng_numpy"  in ckpt: np.random.set_state(ckpt["rng_numpy"])
        if "rng_torch"  in ckpt: torch.set_rng_state(ckpt["rng_torch"])
        if "rng_torch_cuda" in ckpt and ckpt["rng_torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["rng_torch_cuda"])
        print("[Resume] RNG state 복구")
    except Exception as e:
        warnings.warn(f"[Resume] RNG state 복구 실패 → 무시: {e}")

    last_epoch = int(ckpt.get("epoch", 0))
    start_epoch = last_epoch + 1
    print(f"[Resume] 마지막 epoch={last_epoch} → 재시작 epoch={start_epoch}")
    return start_epoch


# ---------------------------
# 학습 루프
# ---------------------------

def train(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = StereoFolderDataset(args.left_dir, args.right_dir,
                                  height=args.height, width=args.width,
                                  mask_dir=args.mask_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    model = StereoModel(max_disp_px=args.max_disp_px, patch_size=args.patch_size,
                        agg_base_ch=args.agg_ch, agg_depth=args.agg_depth,
                        softarg_t=args.softarg_t, norm=args.norm).to(device)
    # model = DPTStereoTrainCompat(
    #     max_disp_px=args.max_disp_px,
    #     patch_size=args.patch_size,
    #     feat_dim=256,            # DPT 내부 채널(필요시 변경 가능)
    #     readout='project',       # CLS readout 주입 방식
    #     embed_dim=768,           # DINOv1-B/8
    #     temperature=0.7          # SoftAndArgMax의 T와 유사
    # ).to(device)
    # 손실 모듈
    dir_loss_fn = DirectionalRelScaleDispLoss(
        sim_thr=args.sim_thr, sim_gamma=args.sim_gamma, sample_k=args.sim_sample_k,
        use_dynamic_thr=args.use_dynamic_thr, dynamic_q=args.dynamic_q,
        vert_margin=1.0, horiz_margin=0.0,
        lambda_v=args.lambda_v, lambda_h=args.lambda_h, huber_delta=1.0).to(device)

    hsharp_fn = HorizontalSharpenedConsistency(
        D=(args.max_disp_px//args.patch_size), tau_sharp=args.tau_sharp, huber_delta=args.huber_delta_h,
        use_fixed_denom=True,
        sim_thr=args.sim_thr, sim_gamma=args.sim_gamma, sample_k=args.sim_sample_k,
        use_dynamic_thr=args.use_dynamic_thr, dynamic_q=args.dynamic_q).to(device)

    prob_cons_fn = NeighborProbConsistencyLoss(
        sim_thr=max(0.5, args.sim_thr-0.15), sim_gamma=max(0.05, args.sim_gamma),
        sample_k=max(1024, args.sim_sample_k),
        allow_shift_v=1, allow_shift_h=0,
        use_dynamic_thr=True, dynamic_q=max(0.7, args.dynamic_q), conf_alpha=1.0).to(device)

    entropy_fn = EntropySharpnessLoss(
        conf_alpha=1.0, sim_thr=args.sim_thr, sim_gamma=args.sim_gamma,
        sample_k=args.sim_sample_k, use_dynamic_thr=True, dynamic_q=args.dynamic_q).to(device)

    anchor_loss_fn = CorrAnchorLoss(tau=args.anchor_tau, margin=args.anchor_margin,
                                    topk=args.anchor_topk, use_huber=True).to(device)
    reproj_loss_fn = FeatureReprojLoss().to(device)
    
    seed_anchor_fn = SeedAnchorHuberLoss(
        tau=args.seed_tau, huber_delta=args.seed_huber_delta
    ).to(device)


    photo_crit = PhotometricLoss(weights=[args.photo_l1_w, args.photo_ssim_w])

    sky_crit = SkyZeroLoss(
        max_disp_px=args.max_disp_px,
        patch_size=args.patch_size,
        compute_at='half',          # ★ 1/2에서 계산
        thr_px=3.0, y_max_ratio=0.6,
        w_huber=1.0, huber_delta_px=0.5,
        w_ce=0.0,                   # 원하면 0.05 정도로 살짝
        debug_dir=os.path.join(args.save_dir, "dbg_sky"),  # ★ PNG 저장 폴더
        save_every=0               # 50 스텝마다 저장 (0이면 매번)
    ).to(device)


    optim = build_optimizer([p for p in model.parameters() if p.requires_grad],
                            name=args.optim, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # ---[ Resume ]---
    if args.resume is not None:
        start_epoch = resume_from_checkpoint(args, model, optim, scaler, device)
    else:
        start_epoch = 1
        
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        log_path = save_args_as_text(
                args,
                save_dir=args.save_dir,
                filename="trainlog.txt",
                append_if_exists=True,  # 기존 파일 있으면 이어서 기록
                title="Stereo Training Arguments"
            )
        print(f"[Log] Arguments written to: {log_path}")
        
        
    model.train()
    
    for epoch in range(start_epoch, args.epochs + 1):
        running = 0.0
        for it, (imgL, imgR, valid_full, names) in enumerate(loader, start=1):
            imgL = imgL.to(device, non_blocking=True)
            imgR = imgR.to(device, non_blocking=True)
            valid_full = valid_full.to(device, non_blocking=True)  # [B,1,H,W]

            # ROI (avg-pool 기반 권장)
            roi_patch = make_roi_patch(valid_full, patch_size=args.patch_size,
                                       method=args.roi_method, thr=args.roi_thr)

            # Half-scale ROI (photometric에 사용)
            roi_half = make_roi_patch(valid_full, patch_size=2,
                                      method=args.roi_method, thr=args.roi_thr)

            # photometric용 원본 [0,1] 변환 + 1/2 스케일
            with torch.no_grad():
                imgL_01 = denorm_imagenet(imgL)
                imgR_01 = denorm_imagenet(imgR)
                imgL_half_01 = F.interpolate(imgL_01, scale_factor=0.5, mode='bilinear', align_corners=False)
                imgR_half_01 = F.interpolate(imgR_01, scale_factor=0.5, mode='bilinear', align_corners=False)

            # enhance (좌 1/2 영상만 기준으로 사용)
            imgL_half_enh_01 = enhance_batch_bgr_from_rgb01(
                imgL_half_01, enable=(not args.no_enhance),
                gamma=args.enhance_gamma,
                clahe_clip=args.enhance_clahe_clip,
                clahe_tile=args.enhance_clahe_tile
            )

            with torch.cuda.amp.autocast(enabled=args.amp):
                prob, disp_soft, aux = model(imgL, imgR)
                FL, FR   = aux["FL"], aux["FR"]
                raw_vol  = aux["raw_volume"]
                mask_d   = aux["mask"]
                refined_masked = aux["refined_masked"]

                # 1/2 해상도 disparity (픽셀 단위)
                disp_half_px = aux["disp_half_px"]
                

                # 오른쪽 half 이미지를 좌로 warp
                imgR_half_warp_01, valid_half = warp_right_to_left_image(imgR_half_01, disp_half_px)

                # Losses
                loss_dir    = dir_loss_fn(disp_soft, FL, roi_patch) * args.w_dir
                loss_hsharp = hsharp_fn(refined_masked, FL, roi_patch) * args.w_hsharp
                loss_prob   = prob_cons_fn(prob, FL, roi_patch) * args.w_probcons
                loss_ent    = entropy_fn(prob, FL, roi_patch) * args.w_entropy
                loss_anchor = anchor_loss_fn(raw_vol, disp_soft, mask=mask_d, roi=roi_patch) * args.w_anchor
                loss_reproj = reproj_loss_fn(FL, FR, disp_soft, roi=roi_patch) * args.w_reproj

                # Photometric (L1+SSIM) on half res, ROI/warp valid로 마스킹
                photo_map = photo_crit.simple_photometric_loss(imgL_half_enh_01, imgR_half_warp_01,
                                                               weights=[args.photo_l1_w, args.photo_ssim_w])  # [B,1,H/2,W/2]
                photo_mask = roi_half * valid_half
                loss_photo = (photo_map * photo_mask).sum() / (photo_mask.sum() + 1e-6)
                loss_photo = loss_photo * args.w_photo

                # Edge-aware smoothness on half res
                loss_smooth = get_disparity_smooth_loss(disp_half_px, imgL_half_enh_01) * args.w_smooth




                # 최종 loss (기존 + 새 항)
                # 필요시 다른 항도 활성화 가능
                # === 1/8: bad 마스크(팽창 없음) ===
                with torch.no_grad():
                    bad_seed_mask = build_bad_seed_mask_1of8(
                        disp_soft=disp_soft, prob_5d=prob, roi_patch=roi_patch,
                        low_idx_thr=args.seed_low_idx_thr, high_idx_thr=args.seed_high_idx_thr,
                        conf_thr=args.seed_conf_thr, road_ymin=args.seed_road_ymin,
                        use_extremes=True, use_conf=True
                    )

                    # (NEW) 정규화 사각형 범위 마스크 (패치 해상도 기준: roi_patch 크기)
                    seed_rect_mask = make_norm_rect_mask_like(
                        roi_patch, y_min=args.seed_ymin, y_max=args.seed_ymax,
                        x_min=args.seed_xmin, x_max=args.seed_xmax
                    ).bool()

                    # 행 모드 계산용 good 마스크(범위 내에서만 집계)
                    good_mask = (roi_patch > 0) & (~bad_seed_mask)
                    good_mask = good_mask & seed_rect_mask  # (NEW) 범위 제한

                    D = prob.shape[2] - 1
                    row_mode_idx, row_valid = rowwise_mode_idx_1of8(
                        disp_soft=disp_soft, good_mask=good_mask, D=D,
                        bin_size=args.seed_bin_w, min_count=args.seed_min_count
                    )
                    seed_idx_map = row_mode_idx.expand(-1, -1, -1, disp_soft.shape[-1])
                    valid_rows   = row_valid.expand_as(bad_seed_mask)

                    # bad 이면서 행 모드 유효 & (NEW) 사각형 범위 내에서만 앵커링
                    anchor_mask = bad_seed_mask & valid_rows & seed_rect_mask  # (NEW)


                # === 시드 앵커 손실(작게) ===
                loss_seed = seed_anchor_fn(
                    refined_logits_masked=refined_masked,   # [B,1,D+1,H/8,W/8]
                    seed_idx_map=seed_idx_map,              # [B,1,H/8,W/8]
                    anchor_mask=anchor_mask                 # [B,1,H/8,W/8]
                ) * args.w_seed

                loss = loss_dir + loss_photo + loss_smooth
                # if epoch <= 22: 
                loss += loss_seed
                loss_sky, sky_aux = sky_crit(
                    prob_5d=prob,
                    disp_soft=disp_soft,
                    roi_patch=roi_patch,
                    disp_half_px=aux["disp_half_px"],  # 있으면 그대로 사용
                    roi_half=roi_half,
                    names=names,                       # ★ Dataset이 반환한 파일명 리스트
                    step=(epoch-1)*len(loader)+it      # ★ 전역 step (PNG 파일명에 포함)
                )
                loss = loss + (args.w_sky * loss_sky)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # 안정화
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.agg.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(model.upmask_head.parameters(), max_norm=5.0)
            scaler.step(optim)
            scaler.update()

            running += loss.item()
            if it % args.log_every == 0:
                # 모니터링: 좌/우 soft vs WTA 평균 차이
                with torch.no_grad():
                    disp_wta = aux["disp_wta"]
                    soft_dx = (roi_patch * (disp_soft - shift_with_mask(disp_soft,0,1)[0]).abs()).sum() / (roi_patch.sum()+1e-6)
                    wta_dx  = (roi_patch * (disp_wta  - shift_with_mask(disp_wta, 0,1)[0]).abs()).sum() / (roi_patch.sum()+1e-6)
                print(f"[Epoch {epoch:03d} | Iter {it:04d}/{len(loader)}] "
                      f"loss={running/args.log_every:.4f} "
                      f"(dir={loss_dir.item():.4f}, hsharp={loss_hsharp.item():.4f}, "
                      f"prob={loss_prob.item():.4f}, ent={loss_ent.item():.4f}, "
                      f"anc={(loss_anchor/max(args.w_anchor,1e-9)).item():.4f}, rep={(loss_reproj/max(args.w_reproj,1e-9)).item():.4f}, "
                      f"photo={(loss_photo/max(args.w_photo,1e-9)).item():.4f}, smooth={(loss_smooth/max(args.w_smooth,1e-9)).item():.4f}) "
                      f"seed loss={ (loss_seed/max(args.w_seed,1e-9)).item():.4f}, "
                      f"| mean|Δx| soft={soft_dx:.3f} wta={wta_dx:.3f}")
                running = 0.0

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt_path = os.path.join(args.save_dir, f"stereo_epoch{epoch:03d}.pth")
            save_checkpoint(ckpt_path, epoch, model, optim, scaler, args)
            print(f"[Save] {ckpt_path}")
        
            


# ---------------------------
# 메인
# ---------------------------
current_time = datetime.now(tz=timezone.utc).astimezone(timezone(timedelta(hours=9))).strftime("%y%m%d_%H%M%S")

def parse_args():
    p = argparse.ArgumentParser()
    # 데이터
    p.add_argument("--left_dir", type=str, required=True)
    p.add_argument("--right_dir", type=str, required=True)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width",  type=int, default=1224)
    p.add_argument("--mask_dir", type=str, default=None, help="하늘 마스크 폴더(파일명 매칭). 흰색=하늘")

    # 모델/학습
    p.add_argument("--max_disp_px", type=int, default=80)
    p.add_argument("--patch_size",  type=int, default=8)
    p.add_argument("--agg_ch",      type=int, default=32)
    p.add_argument("--agg_depth",   type=int, default=3)
    p.add_argument("--softarg_t",   type=float, default=0.9)  # 초기 탐색↑
    p.add_argument("--norm",        type=str, default="gn", choices=["bn","gn"], help="3D conv 정규화")

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs",     type=int, default=20, help="학습을 마칠 최종 epoch (resume 시 마지막+1 ~ epochs)")
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--optim",      type=str, default="adamw", choices=["adamw","adam","sgd"])
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--amp",        action="store_true", help="mixed precision")

    # ROI 축소 옵션
    p.add_argument("--roi_method", type=str, default="avg", choices=["avg","nearest"])
    p.add_argument("--roi_thr",    type=float, default=0.5)

    # 방향 이웃 제약(soft 기준)
    p.add_argument("--w_dir",        type=float, default=0.1)
    p.add_argument("--sim_thr",      type=float, default=0.8)
    p.add_argument("--sim_gamma",    type=float, default=0.0)
    p.add_argument("--sim_sample_k", type=int,   default=1024)
    p.add_argument("--use_dynamic_thr", action="store_true")
    p.add_argument("--dynamic_q",    type=float, default=0.7)
    p.add_argument("--lambda_v",     type=float, default=1.0)
    p.add_argument("--lambda_h",     type=float, default=1.0)   # 가로는 샤픈 항으로 대체 권장
    p.add_argument("--huber_delta_h", type=float, default=0.25) # 샤픈 가로 용

    # 샤픈 가로 일관성
    p.add_argument("--w_hsharp",   type=float, default=0.0)
    p.add_argument("--tau_sharp",  type=float, default=0.2)

    # 분포-일치/엔트로피
    p.add_argument("--w_probcons", type=float, default=0.0)
    p.add_argument("--w_entropy",  type=float, default=0.00)

    # 앵커/재투영
    p.add_argument("--w_anchor",     type=float, default=0.0)
    p.add_argument("--anchor_tau",   type=float, default=0.5)
    p.add_argument("--anchor_margin",type=float, default=1.0)
    p.add_argument("--anchor_topk",  type=int,   default=2)
    p.add_argument("--w_reproj",     type=float, default=1.0)

    # Photometric / Smoothness 추가
    p.add_argument("--w_photo",    type=float, default=1.0, help="photometric loss weight")
    p.add_argument("--w_smooth",   type=float, default=0.01, help="edge-aware smoothness weight")
    p.add_argument("--photo_l1_w",   type=float, default=0.15)
    p.add_argument("--photo_ssim_w", type=float, default=0.85)

    # Enhance 옵션
    p.add_argument("--no_enhance", dest="no_enhance", action="store_true", help="저조도 보정 비활성화")
    p.set_defaults(no_enhance=False)
    p.add_argument("--enhance_gamma",      type=float, default=1.8)
    p.add_argument("--enhance_clahe_clip", type=float, default=2.0)
    p.add_argument("--enhance_clahe_tile", type=int,   default=8)

    # ---[ Resume / Checkpoint ]---
    p.add_argument("--resume", type=str, default=None,
                   help="불러올 체크포인트(.pth) 경로. 지정하면 해당 지점부터 이어서 학습")
    p.add_argument("--resume_non_strict", action="store_true",
                   help="state_dict 로드 시 strict=False (일부 키 불일치 허용)")
    p.add_argument("--resume_reset_optim", action="store_true",
                   help="체크포인트의 optimizer 상태를 무시하고 현재 설정으로 재시작")
    p.add_argument("--resume_reset_scaler", action="store_true",
                   help="체크포인트의 GradScaler 상태를 무시")

    # 로깅/저장
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--save_dir", type=str, default=f"./log/checkpoints_{current_time}")
    
    
    # --- 1/8 Seeded Prior (핀 + 고무줄) ---
    p.add_argument("--w_seed", type=float, default=0.02, help="시드 앵커 손실 가중치(작게)")
    p.add_argument("--seed_low_idx_thr",  type=float, default=1.0)
    p.add_argument("--seed_high_idx_thr", type=float, default=1.0)
    p.add_argument("--seed_conf_thr",     type=float, default=0.05)
    p.add_argument("--seed_road_ymin",    type=float, default=0.8)   # 하단만 적용하고 싶을 때
    p.add_argument("--seed_bin_w",        type=float, default=1.0)   # 행 모드 bin 폭(패치 인덱스 단위)
    p.add_argument("--seed_min_count",    type=int,   default=8)     # 행 모드 최소 표본 수
    p.add_argument("--seed_tau",          type=float, default=0.30)  # 샤프닝 온도(ArgMax 근사)
    p.add_argument("--seed_huber_delta",  type=float, default=0.50)
    
        # --- 1/8 Seeded Prior 범위(정규화 좌표) ---
    p.add_argument("--seed_ymin", type=float, default=0.7,
                   help="시드 적용 세로 시작(정규화 0~1, 상단=0)")
    p.add_argument("--seed_ymax", type=float, default=1.0,
                   help="시드 적용 세로 끝(정규화 0~1, 하단=1)")
    p.add_argument("--seed_xmin", type=float, default=0.2,
                   help="시드 적용 가로 시작(정규화 0~1, 좌측=0)")
    p.add_argument("--seed_xmax", type=float, default=0.8,
                   help="시드 적용 가로 끝(정규화 0~1, 우측=1)")

    p.add_argument("--w_sky", type=float, default=0.0,
                   help="sky weight for SkyZeroLoss")

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
