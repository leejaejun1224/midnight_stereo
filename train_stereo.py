# -*- coding: utf-8 -*-
import os
import glob
import argparse
import random
from typing import List, Optional, Tuple, Dict, Set

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
    x: [B,3,H,W], imagenet norm 적용
    return: [B,3,H,W] in [0,1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
    y = x * std + mean
    return y.clamp(0.0, 1.0)


# ---------------------------
# 데이터셋 (mask_dir 제거)
# ---------------------------

class StereoFolderDataset(Dataset):
    """
    - left_dir/right_dir: 동일 파일명 매칭
    - 마스크 관련 입력/처리는 제거
    """
    def __init__(self, left_dir: str, right_dir: str,
                 height: int = 384, width: int = 1224):
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

        self.left_paths:  List[str] = left_all
        self.right_paths: List[str] = right_all

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_img  = Image.open(self.left_paths[idx]).convert("RGB")
        right_img = Image.open(self.right_paths[idx]).convert("RGB")
        left_t  = self.to_tensor(left_img)
        right_t = self.to_tensor(right_img)
        name = os.path.basename(self.left_paths[idx])
        return left_t, right_t, name


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
    세로(±1): |Δ|<=1, 가로(±1): |Δ|<=0
    """
    def __init__(self,
                 sim_thr=0.75, sim_gamma=0.0, sample_k=512,
                 use_dynamic_thr=True, dynamic_q=0.7,
                 vert_margin=1.0, horiz_margin=0.0,
                 lambda_v=1.0, lambda_h=1.0, huber_delta=1.0):
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
    imgR: [B,3,H,W] in [0,1]  (입력 텐서 해상도와 동일)
    disp_px: [B,1,H,W] in '해당 해상도 격자 기준' pixel units
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
# Photometric / Smoothness
# ---------------------------
def enhance_batch_bgr_from_rgb01(img_rgb_01: torch.Tensor, 
                                 enable: bool, gamma: float, 
                                 clahe_clip: float, clahe_tile: int) -> torch.Tensor: 
    """ img_rgb_01: [B,3,H,W] in [0,1] return: enhanced RGB [B,3,H,W] in [0,1] """ 
    B, C, H, W = img_rgb_01.shape 
    out_list = [] 
    img_cpu = img_rgb_01.detach().cpu().clamp(0,1).numpy() 
    for b in range(B): 
        rgb = (img_cpu[b].transpose(1,2,0) * 255.0).astype(np.uint8) # H,W,3 RGB uint8 
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) 
        bgr_enh = enhance_low_light_bgr(bgr, enable=enable, gamma=gamma, 
                                        clahe_clip=clahe_clip, clahe_tile=clahe_tile) 
        rgb_enh = cv2.cvtColor(bgr_enh, cv2.COLOR_BGR2RGB) 
        out_list.append(torch.from_numpy(rgb_enh.astype(np.float32) / 255.0).permute(2,0,1)) 
    out = torch.stack(out_list, dim=0) 
    return out.to(img_rgb_01.device)

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
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3,  1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
    def forward(self, x, y):
        x = self.refl(x); y = self.refl(y)
        mu_x = self.mu_x_pool(x); mu_y = self.mu_y_pool(y)
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

def get_disparity_smooth_loss(disp, img):
    """Edge-aware smoothness for disparity"""
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


# ---------------------------
# (NEW) MS2 실시간 정량평가 유틸
# ---------------------------

def _safe_np_load(path: str):
    if not path or not os.path.isfile(path):
        return None
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.lib.npyio.NpzFile):
        return {k: obj[k] for k in obj.files}
    try:
        if hasattr(obj, "item"):
            return obj.item()
    except Exception:
        pass
    return obj

def _normalize_unit_m(b: float) -> float:
    # 2m 이상 값이 들어오면 mm로 보고 m로 변환
    return b / 1000.0 if abs(b) > 2.0 else b

def read_fx_baseline_rgb(intrinsic_left_npy: Optional[str],
                         calib_npy: Optional[str]) -> Tuple[Optional[float], Optional[float], str]:
    """
    intrinsic_left_npy(3x3)와 calib_npy(dict)에서 fx(px), baseline(m) 추출
    """
    src = []
    fx, B = None, None

    # fx from intrinsic
    if intrinsic_left_npy:
        K = _safe_np_load(intrinsic_left_npy)
        if isinstance(K, dict):
            for k in ["K", "K_left", "K_rgbL", "intrinsic", "intrinsic_left"]:
                if k in K and isinstance(K[k], np.ndarray) and K[k].shape[:2] == (3,3):
                    fx = float(K[k][0,0]); src.append(f"K:{intrinsic_left_npy}"); break
        elif isinstance(K, np.ndarray) and K.shape[:2] == (3,3):
            fx = float(K[0,0]); src.append(f"K:{intrinsic_left_npy}")

    # baseline from calib
    C = _safe_np_load(calib_npy) if calib_npy else None
    if isinstance(C, dict):
        # 1) 직접 T_lr (4x4)
        for key in ["T_lr", "T_left_right", "T_rgb_lr"]:
            if key in C and isinstance(C[key], np.ndarray) and C[key].shape == (4,4):
                B = _normalize_unit_m(abs(float(C[key][0,3]))); src.append(f"Tlr:{calib_npy}"); break
        # 2) 좌우 translation
        if B is None and ("T_rgbL" in C and "T_rgbR" in C):
            TL = np.asarray(C["T_rgbL"]).reshape(3)
            TR = np.asarray(C["T_rgbR"]).reshape(3)
            B = _normalize_unit_m(abs(float(TR[0] - TL[0]))); src.append(f"T_LR:{calib_npy}")
        # 3) P행렬만 있을 때: rectified 가정 (P_right[0,3] = -fx * B)
        if B is None:
            P_left = None; P_right = None
            for k in ["P_left", "P0", "P2", "proj_left", "P_rgbL"]:
                if k in C and isinstance(C[k], np.ndarray) and C[k].size >= 12:
                    P_left = C[k].reshape(3,4); break
            for k in ["P_right", "P1", "P3", "proj_right", "P_rgbR"]:
                if k in C and isinstance(C[k], np.ndarray) and C[k].size >= 12:
                    P_right = C[k].reshape(3,4); break
            if P_left is not None and P_right is not None:
                fx_from_P = float(P_left[0,0])
                fx_eff = fx_from_P if fx_from_P > 0 else (fx if fx else 0.0)
                if fx_eff > 0:
                    B = _normalize_unit_m(abs(float(-P_right[0,3] / fx_eff))); src.append(f"P:{calib_npy}")

    return fx, B, "+".join(src) if src else "not_found"

def _first_existing(path_without_ext: str, exts=(".png",".tiff",".tif",".exr",".npy")) -> Optional[str]:
    for e in exts:
        p = path_without_ext + e
        if os.path.isfile(p): return p
    g = glob.glob(path_without_ext + ".*")
    return g[0] if len(g) else None

@torch.no_grad()
def load_ms2_gt_depth_batch(names: List[str], gt_depth_dir: str, scale: float,
                            target_hw: Tuple[int,int], device: torch.device) -> Optional[torch.Tensor]:
    """uint16 PNG( depth[m]*scale ) 또는 npy(exr) → [B,1,H,W] (meters), nearest resize"""
    if not gt_depth_dir:
        return None
    Ht, Wt = target_hw
    outs = []
    for n in names:
        p = _first_existing(os.path.join(gt_depth_dir, os.path.splitext(n)[0]))
        if p is None:
            raise FileNotFoundError(f"[GT] depth not found for {n} under {gt_depth_dir}")
        if p.endswith(".npy"):
            arr = np.load(p).astype(np.float32)
        else:
            arr = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise FileNotFoundError(f"[GT] failed to read: {p}")
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            arr = arr.astype(np.float32)
            # PNG는 depth[m]*scale
            arr = arr / max(scale, 1e-6)
        if arr.shape != (Ht,Wt):
            arr = cv2.resize(arr, (Wt,Ht), interpolation=cv2.INTER_NEAREST)
        outs.append(torch.from_numpy(arr)[None,None])  # [1,1,H,W]
    return torch.cat(outs, dim=0).to(device)

@torch.no_grad()
def disparity_to_depth(disp_px: torch.Tensor, focal_px: float, baseline_m: float, eps: float = 1e-6) -> torch.Tensor:
    return (focal_px * baseline_m) / disp_px.clamp_min(eps)

@torch.no_grad()
def compute_ms2_disparity_metrics(pred_disp: torch.Tensor,
                                  gt_disp: torch.Tensor,
                                  valid: torch.Tensor) -> Dict[str, float]:
    """
    EPE, D1-all(AND: >3px & >5%), >1px, >2px
    pred_disp, gt_disp, valid: [B,1,H,W]
    """
    eps = 1e-6
    err = (pred_disp - gt_disp).abs()
    v = (valid > 0.5).float()
    denom = v.sum().clamp_min(eps)

    epe = (err * v).sum() / denom

    thr = torch.maximum(torch.tensor(3.0, device=gt_disp.device, dtype=gt_disp.dtype),
                        0.05 * gt_disp.abs())
    d1_mask = ((err > 3.0) & ((err / gt_disp.clamp_min(eps)) > 0.05)) * (v > 0.5)
    d1 = d1_mask.sum().float() / denom * 100.0

    bad1 = ((err > 1.0).float() * v).sum() / denom * 100.0
    bad2 = ((err > 2.0).float() * v).sum() / denom * 100.0

    return {"EPE": epe.item(), "D1_all": d1.item(), "> 1px": bad1.item(), "> 2px": bad2.item(), "valid_px": denom.item()}

@torch.no_grad()
def compute_depth_metrics(pred_depth_m: torch.Tensor,
                          gt_depth_m: torch.Tensor,
                          valid: torch.Tensor) -> Dict[str, float]:
    """AbsRel, RMSE, δ<1.25 (필수 세트)"""
    eps = 1e-6
    v = (valid > 0.5).float()
    pd = pred_depth_m.clamp_min(eps)
    gd = gt_depth_m.clamp_min(eps)
    denom = v.sum().clamp_min(eps)

    abs_rel = ((pd - gd).abs() / gd * v).sum() / denom
    rmse    = torch.sqrt((((pd - gd) ** 2) * v).sum() / denom)
    ratio   = torch.maximum(pd / gd, gd / pd)
    d1      = ((ratio < 1.25).float() * v).sum() / denom

    return {"AbsRel": abs_rel.item(), "RMSE": rmse.item(), "δ<1.25": d1.item(), "valid_px": denom.item()}

@torch.no_grad()
def compute_bin_weighted_depth(pred_depth_m: torch.Tensor,
                               gt_depth_m: torch.Tensor,
                               valid: torch.Tensor,
                               max_depth_m: float = 50.0,
                               num_bins: int = 5) -> Dict[str, float]:
    """0~max_depth_m 구간을 num_bins로 등분, bin별 metric 평균"""
    edges = torch.linspace(0.0, max_depth_m, steps=num_bins + 1, device=pred_depth_m.device)
    bins = [(edges[i], edges[i+1]) for i in range(num_bins)]
    acc_absrel = 0.0; acc_rmse = 0.0; acc_d1 = 0.0; cnt = 0
    for lo, hi in bins:
        m = (valid > 0.5) & (gt_depth_m >= lo) & (gt_depth_m < hi)
        if m.sum() < 1:
            continue
        sub = compute_depth_metrics(pred_depth_m, gt_depth_m, m.float())
        acc_absrel += sub["AbsRel"]; acc_rmse += sub["RMSE"]; acc_d1 += sub["δ<1.25"]; cnt += 1
    if cnt == 0:
        return {"W/AbsRel": float("nan"), "W/RMSE": float("nan"), "W/δ<1.25": float("nan")}
    return {"W/AbsRel": acc_absrel/cnt, "W/RMSE": acc_rmse/cnt, "W/δ<1.25": acc_d1/cnt}

def _fmt_disp(m: Dict[str,float]) -> str:
    if not m: return ""
    return f"EPE={m['EPE']:.3f}  D1={m['D1_all']:.2f}%  >1px={m['> 1px']:.2f}%  >2px={m['> 2px']:.2f}%"

def _fmt_depth(m: Dict[str,float]) -> str:
    if not m: return ""
    return f"AbsRel={m['AbsRel']:.3f}  RMSE={m['RMSE']:.3f}  δ1={m['δ<1.25']:.3f}"

def _fmt_depth_w(m: Dict[str,float]) -> str:
    if not m: return ""
    return f"W/AbsRel={m['W/AbsRel']:.3f}  W/RMSE={m['W/RMSE']:.3f}  W/δ1={m['W/δ<1.25']:.3f}"


# ---------------------------
# 학습 루프 (프리트레인 동결 지원 추가)
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

import torch, warnings
from torch import nn

def smart_load_pretrained(model: nn.Module,
                          state_dict: dict,
                          prefix_to_strip: str = "",
                          ignore_substrings = ("num_batches_tracked",),
                          verbose: bool = True):
    """
    - ckpt의 키를 model과 최대한 매칭해서 로드.
    - shape 다른 텐서는 자동 스킵.
    - 'module.' 같은 prefix 제거 지원.
    - BN의 num_batches_tracked 등은 무시.
    """
    own_state = model.state_dict()
    loaded, skipped_name, skipped_shape = [], [], []

    # 1) prefix 정리
    def _fix_key(k):
        if prefix_to_strip and k.startswith(prefix_to_strip):
            return k[len(prefix_to_strip):]
        return k

    for k, v in list(state_dict.items()):
        k2 = _fix_key(k)
        # 2) 무시 키
        if any(s in k2 for s in ignore_substrings):
            continue
        # 3) 존재 + shape 일치일 때만 채택
        if k2 in own_state and own_state[k2].shape == v.shape:
            own_state[k2].copy_(v)
            loaded.append(k2)
        else:
            # shape 불일치 또는 키 미존재
            if k2 in own_state:
                skipped_shape.append((k2, tuple(v.shape), tuple(own_state[k2].shape)))
            else:
                skipped_name.append(k2)

    if verbose:
        print(f"[smart_load] 로드 성공 {len(loaded)}개")
        if skipped_shape:
            print(f"[smart_load] shape 불일치 {len(skipped_shape)}개 (ckpt → model):")
            for n, s1, s2 in skipped_shape[:10]:
                print(f"  - {n}: {s1} → {s2}")
            if len(skipped_shape) > 10: print("  ...")
        if skipped_name:
            print(f"[smart_load] 모델에 없는 키 {len(skipped_name)}개 (예: {skipped_name[:5]})")

    return loaded, skipped_name, skipped_shape


@torch.no_grad()
def init_new_modules(model: nn.Module, loaded_param_names: set):
    """
    ckpt에서 못 불러온(=새로 추가된) 파라미터만 안전 초기화.
    - Conv/Linear: kaiming normal
    - Norm/BN: weight=1, bias=0
    - bias: 0
    """
    for name, m in model.named_modules():
        for p_name, p in m.named_parameters(recurse=False):
            full = f"{name}.{p_name}" if name else p_name
            if full in loaded_param_names:  # 이미 ckpt에서 채워진 파라미터
                continue
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                if p_name == "weight":
                    nn.init.kaiming_normal_(p)
                elif p_name == "bias":
                    nn.init.zeros_(p)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d,
                                nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if p_name == "weight":
                    nn.init.ones_(p)
                elif p_name == "bias":
                    nn.init.zeros_(p)
            else:
                # 기타 모듈은 bias만 0으로
                if p_name == "bias":
                    nn.init.zeros_(p)

def freeze_params_by_name(model: nn.Module, names_set: Set[str], verbose: bool = True):
    """
    names_set(ckpt로부터 로드된 파라미터 이름들)에 해당하는 파라미터만 동결(requires_grad=False)
    """
    num_tensors = 0
    num_elems_frozen = 0
    for n, p in model.named_parameters():
        if n in names_set:
            p.requires_grad = False
            num_tensors += 1
            num_elems_frozen += p.numel()
    if verbose:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[Freeze] 동결 텐서 수={num_tensors}, 동결 파라미터 수={num_elems_frozen:,}")
        print(f"[Freeze] 학습 가능 파라미터={trainable:,} / 전체={total:,}")

def resume_from_checkpoint(args,
                           model: nn.Module,
                           device: torch.device) -> Tuple[int, Set[str], Optional[dict]]:
    """
    체크포인트로부터 모델 가중치만 로드하고, 로드에 성공한 파라미터 이름 집합을 반환.
    - 옵티마이저/스케일러는 여기서 로드하지 않음 (동결 여부에 따라 이후 처리)
    - --pretrained_freeze=True이면 start_epoch는 1부터 시작하도록 리셋
    """
    assert args.resume is not None and os.path.isfile(args.resume), f"checkpoint 없음: {args.resume}"
    print(f"[Resume] 로드: {args.resume}")
    ckpt = torch.load(args.resume, map_location="cpu")

    # === (A) 모델 가중치 로드 (안전 모드) ===
    state = ckpt.get("model", ckpt)  # 방어적: 저장 방식 따라 키가 다를 수 있음
    # DataParallel/DistributedDataParallel로 저장된 경우 'module.' prefix 제거
    has_module_prefix = any(k.startswith("module.") for k in state.keys())
    prefix = "module." if has_module_prefix else ""
    loaded, skipped_name, skipped_shape = smart_load_pretrained(
        model, state, prefix_to_strip=prefix, verbose=True
    )
    loaded_set = set(loaded)

    # 새 모듈 안전 초기화
    init_new_modules(model, loaded_set)

    # === (B) RNG(선택) ===
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
    start_epoch = 1 if getattr(args, "pretrained_freeze", False) else last_epoch + 1
    print(f"[Resume] 마지막 epoch={last_epoch} → 재시작 epoch={start_epoch} "
          f"({'동결모드: epoch 리셋' if getattr(args,'pretrained_freeze',False) else '연속학습'})")

    # 동결은 train()에서 옵티마이저 생성 전에 수행
    return start_epoch, loaded_set, ckpt


def train(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- (NEW) fx/baseline 자동 로딩 ---
    fx, B, src = read_fx_baseline_rgb(
        intrinsic_left_npy=getattr(args, "K_left_npy", None),
        calib_npy=getattr(args, "calib_npy", None)
    )
    if getattr(args, "focal_px", 0.0) <= 0.0 and fx is not None:
        args.focal_px = float(fx)
    if getattr(args, "baseline_m", 0.0) <= 0.0 and B is not None:
        args.baseline_m = float(B)
    if getattr(args, "realtime_test", False):
        print(f"[Calib] fx(px)={getattr(args,'focal_px',0.0):.6f}  baseline(m)={getattr(args,'baseline_m',0.0):.6f}  src={src}")

    dataset = StereoFolderDataset(args.left_dir, args.right_dir,
                                  height=args.height, width=args.width)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    model = StereoModel(max_disp_px=args.max_disp_px, patch_size=args.patch_size,
                        agg_base_ch=args.agg_ch, agg_depth=args.agg_depth,
                        softarg_t=args.softarg_t, norm=args.norm).to(device)
    # model = StereoModel(max_disp_px=args.max_disp_px, patch_size=args.patch_size, volume_mode='gwc', gw_groups=8).to(device)

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

    sky_loss = SkyGridZeroLoss(
        max_disp_px=args.max_disp_px,
        patch_size=args.patch_size).to(device)

    # ---[ Resume + Pretrained Freeze 처리 ]---
    start_epoch = 1
    loaded_set: Set[str] = set()
    ckpt = None
    if args.resume is not None:
        start_epoch, loaded_set, ckpt = resume_from_checkpoint(args, model, device)
        # 동결 옵션이면: ckpt로부터 로드된 파라미터만 동결
        if args.pretrained_freeze:
            freeze_params_by_name(model, loaded_set, verbose=True)

    # === 옵티마이저/스케일러는 동결/미동결 결정 후 생성 ===
    optim = build_optimizer([p for p in model.parameters() if p.requires_grad],
                            name=args.optim, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # === 옵티마이저/스케일러 상태 복구 (동결 모드가 아닐 때만) ===
    if (ckpt is not None) and (not args.pretrained_freeze):
        # 옵티마이저
        if (not args.resume_reset_optim) and ("optim" in ckpt):
            try:
                optim.load_state_dict(ckpt["optim"])
                _move_optimizer_state_to_device(optim, device)
                print("[Resume] optimizer 상태 복구")
            except Exception as e:
                warnings.warn(f"[Resume] optimizer 복구 실패 → 초기화: {e}")
        else:
            print("[Resume] optimizer 상태 초기화(미복구)")
        # AMP GradScaler
        if scaler is not None and (not args.resume_reset_scaler) and ("scaler" in ckpt) and (ckpt["scaler"] is not None):
            try:
                scaler.load_state_dict(ckpt["scaler"])
                print("[Resume] GradScaler 상태 복구")
            except Exception as e:
                warnings.warn(f"[Resume] GradScaler 상태 복구 실패 → 무시: {e}")
        else:
            print("[Resume] GradScaler 상태 초기화(미복구)")
    elif args.pretrained_freeze and ckpt is not None:
        print("[Resume] pretrained_freeze=True → optimizer/scaler 상태 복구 생략 (파라미터 구성 변경)")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        log_path = save_args_as_text(
                args,
                save_dir=args.save_dir,
                filename="trainlog.txt",
                append_if_exists=True,
                title="Stereo Training Arguments"
            )
        print(f"[Log] Arguments written to: {log_path}")
        
    model.train()
    
    for epoch in range(start_epoch, args.epochs + 1):
        running = 0.0
        for it, (imgL, imgR, names) in enumerate(loader, start=1):  # ★ valid_full 제거
            imgL = imgL.to(device, non_blocking=True)
            imgR = imgR.to(device, non_blocking=True)

            # photometric용 원본 [0,1] 변환 + 1/2 스케일
            with torch.no_grad():
                imgL_01 = denorm_imagenet(imgL)
                imgR_01 = denorm_imagenet(imgR)
                imgL_half_01 = F.interpolate(imgL_01, scale_factor=0.5, mode='bilinear', align_corners=False)
                imgR_half_01 = F.interpolate(imgR_01, scale_factor=0.5, mode='bilinear', align_corners=False)

                # 원본 해상도에서도 photometric/smoothness를 사용할 경우만 보정 계산
                need_full_enh = (args.w_photo_fullres > 0.0) or (args.w_smooth_fullres > 0.0)
                if need_full_enh:
                    imgL_full_enh_01 = enhance_batch_bgr_from_rgb01(
                        imgL_01, enable=(not args.no_enhance),
                        gamma=args.enhance_gamma,
                        clahe_clip=args.enhance_clahe_clip,
                        clahe_tile=args.enhance_clahe_tile
                    )
                else:
                    imgL_full_enh_01 = None

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

                # 1/2 해상도 disparity (픽셀 단위, 1/2 격자 기준 px)
                disp_half_px = aux["disp_half_px"]

                # === ROI 전영역(=1) 처리 ===
                roi_patch = torch.ones_like(disp_soft)           # [B,1,H/8,W/8]
                roi_half  = torch.ones_like(disp_half_px)        # [B,1,H/2,W/2]

                # 오른쪽 half 이미지를 좌로 warp
                imgR_half_warp_01, valid_half = warp_right_to_left_image(imgR_half_01, disp_half_px)

                # Losses (원래 루틴)
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

                # ====== [추가] 원본 해상도 단위의 photometric/smoothness ======
                loss_photo_full  = torch.tensor(0.0, device=device)
                loss_smooth_full = torch.tensor(0.0, device=device)

                # (평가용) full-res disparity 확보 (항상 생성)
                if "disp_full_px" in aux and aux["disp_full_px"] is not None:
                    disp_full_px_eval = aux["disp_full_px"]                         # [B,1,H,W]
                else:
                    disp_full_grid = F.interpolate(disp_half_px, scale_factor=2.0, mode="bilinear", align_corners=False)
                    disp_full_px_eval = disp_full_grid * 2.0                        # half-px → full-px

                if (args.w_photo_fullres > 0.0) or (args.w_smooth_fullres > 0.0):
                    # photometric (full)
                    if args.w_photo_fullres > 0.0:
                        imgR_full_warp_01, valid_full = warp_right_to_left_image(imgR_01, disp_full_px_eval)
                        photo_full_map = photo_crit.simple_photometric_loss(
                            imgL_full_enh_01 if imgL_full_enh_01 is not None else imgL_01,
                            imgR_full_warp_01,
                            weights=[args.photo_l1_w, args.photo_ssim_w]
                        )  # [B,1,H,W]
                        loss_photo_full = (photo_full_map * valid_full).sum() / (valid_full.sum() + 1e-6)
                        loss_photo_full = loss_photo_full * args.w_photo_fullres

                    # smoothness (full)
                    if args.w_smooth_fullres > 0.0:
                        base_img_full = imgL_full_enh_01 if imgL_full_enh_01 is not None else imgL_01
                        loss_smooth_full = get_disparity_smooth_loss(disp_full_px_eval, base_img_full) * args.w_smooth_fullres
                # ===========================================================

                # === 1/8 seed prior 유틸 ===
                with torch.no_grad():
                    bad_seed_mask = build_bad_seed_mask_1of8(
                        disp_soft=disp_soft, prob_5d=prob, roi_patch=roi_patch,
                        low_idx_thr=args.seed_low_idx_thr, high_idx_thr=args.seed_high_idx_thr,
                        conf_thr=args.seed_conf_thr, road_ymin=args.seed_road_ymin,
                        use_extremes=True, use_conf=True
                    )

                    seed_rect_mask = make_norm_rect_mask_like(
                        roi_patch, y_min=args.seed_ymin, y_max=args.seed_ymax,
                        x_min=args.seed_xmin, x_max=args.seed_xmax
                    ).bool()

                    good_mask = (roi_patch > 0) & (~bad_seed_mask)
                    good_mask = good_mask & seed_rect_mask

                    D = prob.shape[2] - 1
                    row_mode_idx, row_valid = rowwise_mode_idx_1of8(
                        disp_soft=disp_soft, good_mask=good_mask, D=D,
                        bin_size=args.seed_bin_w, min_count=args.seed_min_count
                    )
                    seed_idx_map = row_mode_idx.expand(-1, -1, -1, disp_soft.shape[-1])
                    valid_rows   = row_valid.expand_as(bad_seed_mask)

                    anchor_mask = bad_seed_mask & valid_rows & seed_rect_mask

                loss_seed = seed_anchor_fn(
                    refined_logits_masked=refined_masked,   # [B,1,D+1,H/8,W/8]
                    seed_idx_map=seed_idx_map,              # [B,1,H/8,W/8]
                    anchor_mask=anchor_mask                 # [B,1,H/8,W/8]
                ) * args.w_seed

                # 총손실 (원본 해상도 항 추가 포함)
                loss = loss_dir + loss_photo + loss_smooth + loss_seed \
                       + loss_photo_full + loss_smooth_full

                # === Sky loss (1/2 → 1/8 grid-ALL rule, disp=0 유도) ===
                loss_sky, aux_sky = sky_loss(
                    refined_logits_masked=aux["refined_masked"],
                    disp_half_px=aux["disp_half_px"],
                    roi_half=None,
                    roi_patch=None,
                    names=names,
                    step=(epoch-1)*len(loader)+it
                )
                loss = loss + (args.w_sky * loss_sky)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # 안정화
            scaler.unscale_(optim)
            if hasattr(model, "agg"):
                torch.nn.utils.clip_grad_norm_(model.agg.parameters(), max_norm=5.0)
            if hasattr(model, "upmask_head"):
                torch.nn.utils.clip_grad_norm_(model.upmask_head.parameters(), max_norm=5.0)
            scaler.step(optim)
            scaler.update()

            running += loss.item()
            if it % args.log_every == 0:
                with torch.no_grad():
                    disp_wta = aux["disp_wta"]
                    soft_dx = (roi_patch * (disp_soft - shift_with_mask(disp_soft,0,1)[0]).abs()).sum() / (roi_patch.sum()+1e-6)
                    wta_dx  = (roi_patch * (disp_wta  - shift_with_mask(disp_wta, 0,1)[0]).abs()).sum() / (roi_patch.sum()+1e-6)

                    # --- (NEW) Realtime MS2 metrics ---
                    extra_eval = ""
                    if getattr(args, "realtime_test", False) and getattr(args, "gt_depth_dir", None):
                        # GT depth 로드 (현재 입력 해상도 기준으로 nearest 리사이즈)
                        H, W = imgL.shape[-2], imgL.shape[-1]
                        gt_depth = load_ms2_gt_depth_batch(
                            names=names,
                            gt_depth_dir=args.gt_depth_dir,
                            scale=args.gt_depth_scale,
                            target_hw=(H,W),
                            device=device
                        )
                        if gt_depth is not None:
                            valid = (gt_depth > 0).float()

                            # 예측 disp (full-res)
                            pred_disp_px = disp_full_px_eval

                            # Stereo 지표 (GT depth → disp 변환, fx/B 필요)
                            disp_msg = ""
                            has_fb = (getattr(args, "focal_px", 0.0) > 0.0) and (getattr(args, "baseline_m", 0.0) > 0.0)
                            if has_fb:
                                gt_disp_px = (args.focal_px * args.baseline_m) / gt_depth.clamp_min(1e-6)
                                disp_metrics = compute_ms2_disparity_metrics(pred_disp_px, gt_disp_px, valid)
                                disp_msg = "[Disp] " + _fmt_disp(disp_metrics)

                                # Depth 지표 (pred depth = fx*B/disp)
                                pred_depth_m = disparity_to_depth(pred_disp_px, args.focal_px, args.baseline_m)
                                depth_metrics = compute_depth_metrics(pred_depth_m, gt_depth, valid)
                                depth_msg = "[Depth] " + _fmt_depth(depth_metrics)

                                # Bin-weighted (선택)
                                depth_w = {}
                                if getattr(args, "eval_num_bins", 0) > 0 and getattr(args, "eval_max_depth_m", 0.0) > 0.0:
                                    depth_w = compute_bin_weighted_depth(
                                        pred_depth_m, gt_depth, valid,
                                        max_depth_m=args.eval_max_depth_m,
                                        num_bins=args.eval_num_bins
                                    )
                                depth_w_msg = "[Depth-W] " + _fmt_depth_w(depth_w) if depth_w else ""

                                parts = [s for s in [disp_msg, depth_msg, depth_w_msg] if s]
                                if parts:
                                    extra_eval = " || " + "  ".join(parts)
                            else:
                                extra_eval = " || [Calib] fx/baseline 미설정 → stereo/depth metric 생략"
                    # ----------------------------------

                print(f"[Epoch {epoch:03d} | Iter {it:04d}/{len(loader)}] "
                      f"loss={running/args.log_every:.4f} !!"
                      f"(dir={loss_dir.item():.4f}, hsharp={loss_hsharp.item():.4f}, "
                      f"prob={loss_prob.item():.4f}, ent={loss_ent.item():.4f}, "
                      f"anc={(loss_anchor/max(args.w_anchor,1e-9)).item():.4f}, rep={(loss_reproj/max(args.w_reproj,1e-9)).item():.4f}, "
                      f"photo={(loss_photo/max(args.w_photo,1e-9)).item():.4f}, smooth={(loss_smooth/max(args.w_smooth,1e-9)).item():.4f}, "
                      f"photoF={(loss_photo_full/max(args.w_photo_fullres,1e-9)).item():.4f}, smoothF={(loss_smooth_full/max(args.w_smooth_fullres,1e-9)).item():.4f}, "
                      f"seed={(loss_seed/max(args.w_seed,1e-9)).item():.4f}, sky={(loss_sky/max(args.w_sky,1e-9)).item():.4f}) "
                      f"| mean|Δx| soft={soft_dx:.3f} wta={wta_dx:.3f}"
                      f"{extra_eval}"
                )
                running = 0.0

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            if epoch % args.save_every == 0 or epoch == args.epochs:
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
    # ★ mask_dir 옵션 제거

    # 모델/학습
    p.add_argument("--max_disp_px", type=int, default=88)
    p.add_argument("--patch_size",  type=int, default=8)
    p.add_argument("--agg_ch",      type=int, default=64)
    p.add_argument("--agg_depth",   type=int, default=3)
    p.add_argument("--softarg_t",   type=float, default=0.9)
    p.add_argument("--norm",        type=str, default="gn", choices=["bn","gn"])

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--optim",      type=str, default="adamw", choices=["adamw","adam","sgd"])
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--amp",        action="store_true", help="mixed precision")

    # 방향 이웃 제약(soft 기준)
    p.add_argument("--w_dir",        type=float, default=1.0)
    p.add_argument("--sim_thr",      type=float, default=0.8)
    p.add_argument("--sim_gamma",    type=float, default=0.0)
    p.add_argument("--sim_sample_k", type=int,   default=1024)
    p.add_argument("--use_dynamic_thr", action="store_true")
    p.add_argument("--dynamic_q",    type=float, default=0.7)
    p.add_argument("--lambda_v",     type=float, default=1.0)
    p.add_argument("--lambda_h",     type=float, default=1.0)
    p.add_argument("--huber_delta_h", type=float, default=0.25)

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

    # Photometric / Smoothness (half)
    p.add_argument("--w_photo",    type=float, default=1.0)
    p.add_argument("--w_smooth",   type=float, default=0.01)
    p.add_argument("--photo_l1_w",   type=float, default=0.15)
    p.add_argument("--photo_ssim_w", type=float, default=0.85)

    # ★ Photometric / Smoothness (full-res 추가)
    p.add_argument("--w_photo_fullres",  type=float, default=0.0,
                   help="원본 해상도 photometric loss 가중치 (0이면 비활성)")
    p.add_argument("--w_smooth_fullres", type=float, default=0.0,
                   help="원본 해상도 edge-aware smoothness 가중치 (0이면 비활성)")
    p.add_argument("--fullres_disp_mode", type=str, default="bilinear",
                   choices=["nearest","bilinear"],
                   help="half→full disparity 업샘플 방식 (기본 bilinear)")

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

    # ★★★ 프리트레인 동결 옵션 추가 ★★★
    p.add_argument("--pretrained_freeze", action="store_true",
                   help="체크포인트에서 불러온(이미 학습된) 파라미터는 동결하고, 새 모듈만 학습")

    # 로깅/저장
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=2)
    p.add_argument("--save_dir", type=str, default=f"./log/checkpoints_{current_time}")
    
    # --- 1/8 Seeded Prior (핀 + 고무줄) ---
    p.add_argument("--w_seed", type=float, default=0.0, help="시드 앵커 손실 가중치(작게)")
    p.add_argument("--seed_low_idx_thr",  type=float, default=1.0)
    p.add_argument("--seed_high_idx_thr", type=float, default=1.0)
    p.add_argument("--seed_conf_thr",     type=float, default=0.05)
    p.add_argument("--seed_road_ymin",    type=float, default=0.8)
    p.add_argument("--seed_bin_w",        type=float, default=1.0)
    p.add_argument("--seed_min_count",    type=int,   default=8)
    p.add_argument("--seed_tau",          type=float, default=0.30)
    p.add_argument("--seed_huber_delta",  type=float, default=0.50)
    p.add_argument("--seed_ymin", type=float, default=0.7)
    p.add_argument("--seed_ymax", type=float, default=1.0)
    p.add_argument("--seed_xmin", type=float, default=0.2)
    p.add_argument("--seed_xmax", type=float, default=0.8)

    # Sky loss weight
    p.add_argument("--w_sky", type=float, default=0.0,
                   help="sky weight for SkyZeroLoss")

    # ---------- (NEW) Realtime evaluation ----------
    p.add_argument("--realtime_test", action="store_true",
                   help="배치별 실시간 정량평가(EPE/D1/>kpx + Depth/Weighted) 출력")
    p.add_argument("--gt_disp_dir", type=str, default=None,
                   help="(선택) GT disparity 디렉토리 (잘 쓰지 않음; 보통 depth GT 사용)")
    p.add_argument("--gt_depth_dir", type=str, default=None,
                   help="GT depth 디렉토리 (PNG: depth[m]*scale, 또는 NPY/EXR)")
    p.add_argument("--gt_disp_scale", type=float, default=1.0,
                   help="GT disparity 스케일 나눗값")
    p.add_argument("--gt_depth_scale", type=float, default=256.0,
                   help="GT depth PNG가 depth[m]*scale 로 저장된 경우 scale (MS2=256)")
    p.add_argument("--eval_num_bins", type=int, default=5,
                   help="bin-weighted depth metric bin 수 (0이면 비활성)")
    p.add_argument("--eval_max_depth_m", type=float, default=50.0,
                   help="bin-weighted depth metric 최대 거리(m)")

    # 캘리브(자동로딩/직접입력)
    p.add_argument("--calib_npy", type=str, default="/home/jaejun/dataset/MS2/sync_data/_2021-08-13-22-36-41/calib.npy")
    p.add_argument("--K_left_npy", type=str, default="/home/jaejun/dataset/MS2/intrinsic_left.npy")
    p.add_argument("--T_lr_npy", type=str, default=None, help="4x4 extrinsic .npy (left->right)")
    p.add_argument("--focal_px", type=float, default=764.5138549804688)
    p.add_argument("--baseline_m", type=float, default=0.29918420530585865)

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
