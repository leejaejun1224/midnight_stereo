import os
import glob
import argparse
import random
from typing import List

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
from modeljj4x4 import StereoModel  # ★ 1/4→1 업샘플 모델
from stereo_dpt import DINOv1Base8Backbone, StereoDPTHead, DPTStereoTrainCompat
from prop_utils import *
from logger import *
from sky_loss import *
# --- NEW: MS2 metrics import ---
from metric.ms2_metric import (
    MS2EvalConfig, load_ms2_gt_batch,
    compute_ms2_disparity_metrics, compute_depth_metrics, compute_bin_weighted_depth_metrics,
    disparity_to_depth, fmt_disp_metrics, fmt_depth_metrics, fmt_weighted_depth_metrics
)
from losses import (
    PhotometricLoss, get_disparity_smooth_loss, warp_right_to_left_image,
    DirectionalRelScaleDispLoss, HorizontalSharpenedConsistency,
    NeighborProbConsistencyLoss, EntropySharpnessLoss, FeatureReprojLoss,
    SeedAnchorHuberLoss, SkyGridZeroLoss
)
from calib.calib import *

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
# 데이터셋
# ---------------------------

class StereoFolderDataset(Dataset):
    """
    - left_dir/right_dir: 동일 파일명 매칭
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

        # DataLoader는 ImageNet 정규화 텐서를 내보냄.
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
# 클릭식 유사도 게이트 (동일)
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
        self.sim_thr, self.sim_gamma = sim_thr, sim_gamma
        self.sample_k = sample_k
        self.use_dynamic_thr, self.dynamic_q = use_dynamic_thr, dynamic_q

    def forward(self, refined_logits_masked, feat, roi):
        p_sharp = torch.softmax(refined_logits_masked / self.tau, dim=2)      # [B,1,D+1,H,W]
        disp_sharp = (p_sharp * self.disp_values).sum(dim=2)                  # [B,1,H,W]
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
# (C)/(D) 그대로
# ---------------------------

def _shift5_spatial(x5, dy, dx):
    B,C,D,H,W = x5.shape
    pt, pb = max(dy,0), max(-dy,0)
    pl, pr = max(dx,0), max(-dx,0)
    x_pad = F.pad(x5, (pl, pr, pt, pb, 0, 0))
    return x_pad[:, :, :, pb:pb+H, pl:pl+W]

def _shift_along_disp(p, s):
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
        with torch.no_grad():
            pp = prob.squeeze(1).clamp_min(1e-8)                 # [B,D+1,H,W]
            topv = torch.topk(pp, k=2, dim=1).values             # [B,2,H,W]
            conf = (topv[:,0] - topv[:,1]).unsqueeze(1)          # [B,1,H,W]
            conf = conf.clamp_min(0.0).pow(self.conf_alpha)

        loss_sum = torch.tensor(0.0, device=prob.device)
        weight_sum = torch.tensor(0.0, device=prob.device)
        p = prob

        for (dy,dx,allow_shift) in [(-1,0,self.allow_shift_v), (1,0,self.allow_shift_v),
                                    (0,-1,self.allow_shift_h), (0,1,self.allow_shift_h)]:
            w_sim = self._gate(feat, roi, dy, dx)
            w_tot = w_sim * roi * conf
            if w_tot.sum() < 1e-6: continue

            q = _shift5_spatial(p, dy, dx)
            costs = []
            for s in range(-allow_shift, allow_shift+1):
                q_s = _shift_along_disp(q, s)
                skl = _sym_kl(p, q_s)
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
        p = prob.squeeze(1).clamp_min(1e-8)
        ent = -(p * p.log()).sum(dim=1, keepdim=True)
        loss = (w * roi * ent).sum() / ((w * roi).sum() + 1e-6)
        return loss


# ---------------------------
# (E) 앵커, (F) 재투영, warp (동일)
# ---------------------------

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
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B,1,1,1)
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
    imgR: [B,3,H,W] in [0,1]
    disp_px: [B,1,H,W] in pixel units at the SAME resolution
    """
    B, C, H, W = imgR.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=imgR.device),
        torch.linspace(-1, 1, W, device=imgR.device),
        indexing="ij"
    )
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B,1,1,1)
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
    B, C, H, W = img_rgb_01.shape 
    out_list = [] 
    img_cpu = img_rgb_01.detach().cpu().clamp(0,1).numpy() 
    for b in range(B): 
        rgb = (img_cpu[b].transpose(1,2,0) * 255.0).astype(np.uint8)
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

import torch
import torch.nn as nn
import torch.nn.functional as F

def _gaussian_window(window_size=11, sigma=1.5, channels=3, device='cpu', dtype=torch.float32):
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(1)  # [K,1]
    kernel_2d = (g @ g.t())         # [K,K]
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel_2d = kernel_2d.view(1, 1, window_size, window_size)
    kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)  # depthwise
    return kernel_2d

def get_disparity_smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()


# ---------------------------
# 학습 루프
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
    assert args.resume is not None and os.path.isfile(args.resume), f"checkpoint 없음: {args.resume}"
    print(f"[Resume] 로드: {args.resume}")
    ckpt = torch.load(args.resume, map_location="cpu")

    strict = not args.resume_non_strict
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)
    if not strict:
        if missing:
            warnings.warn(f"[Resume] 누락된 키 {len(missing)}개: {missing[:10]}{' ...' if len(missing)>10 else ''}")
        if unexpected:
            warnings.warn(f"[Resume] 예기치 않은 키 {len(unexpected)}개: {unexpected[:10]}{' ...' if len(unexpected)>10 else ''}")

    if (not args.resume_reset_optim) and ("optim" in ckpt):
        optim.load_state_dict(ckpt["optim"])
        _move_optimizer_state_to_device(optim, device)
        print("[Resume] optimizer 상태 복구")
    else:
        print("[Resume] optimizer 상태 초기화(미복구)")

    if scaler is not None and (not args.resume_reset_scaler) and ("scaler" in ckpt) and (ckpt["scaler"] is not None):
        try:
            scaler.load_state_dict(ckpt["scaler"])
            print("[Resume] GradScaler 상태 복구")
        except Exception as e:
            warnings.warn(f"[Resume] GradScaler 상태 복구 실패 → 무시: {e}")
    else:
        print("[Resume] GradScaler 상태 초기화(미복구)")

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
def train(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- NEW: realtime metrics on/off & import ---
    enable_realtime = bool(
        getattr(args, "realtime_test", False) and
        (getattr(args, "gt_disp_dir", None) or getattr(args, "gt_depth_dir", None))
    )
    if enable_realtime:
        eval_cfg = MS2EvalConfig()


    # 인자에 값이 없으면 자동으로 채운다.
    
    fx_auto, bl_auto, src = resolve_fx_baseline(
        left_dir=args.left_dir,
        focal_px=getattr(args, "focal_px", 0.0),
        baseline_m=getattr(args, "baseline_m", 0.0),
        calib_npy=getattr(args, "calib_npy", None),
        K_left_npy=getattr(args, "K_left_npy", None),
        T_lr_npy=getattr(args, "T_lr_npy", None),
        modality="rgb"
    )
    if fx_auto is not None and getattr(args, "focal_px", 0.0) <= 0:
        args.focal_px = fx_auto
    if bl_auto is not None and getattr(args, "baseline_m", 0.0) <= 0:
        args.baseline_m = bl_auto
    if enable_realtime:
        print(f"[Calib] fx(px)={getattr(args,'focal_px',0.0):.3f}  baseline(m)={getattr(args,'baseline_m',0.0):.4f}  (src: {src})")

    # -------------------- 기존 파트 --------------------
    dataset = StereoFolderDataset(args.left_dir, args.right_dir,
                                  height=args.height, width=args.width)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    model = StereoModel(
        max_disp_px=args.max_disp_px,
        patch_size=args.patch_size,
        agg_base_ch=args.agg_ch,
        agg_depth=args.agg_depth,
        softarg_t=args.softarg_t,
        norm=args.norm,

        # === 여기만 바꾸면 새 cost-volume 입력 사용 ===
        sim_fusion_mode="learned_fused",   # ★ 새 모드
        dino_weight=0.65,
        fuse_feat_mode=None,
        sum_alpha = 0.5,
        cnn_center=True,
        spx_source="dino",
        # 1×1 conv/MLP 옵션
        cv_fuse_out_ch=768,
        cv_fuse_arch="conv1x1",            # 또는 "mlp"

        # 나머지 기존 설정
        ).to(device)

    # 손실 모듈들
    dir_loss_fn = DirectionalRelScaleDispLoss(
        sim_thr=args.sim_thr, sim_gamma=args.sim_gamma, sample_k=args.sim_sample_k,
        use_dynamic_thr=args.use_dynamic_thr, dynamic_q=args.dynamic_q,
        vert_margin=1.0, horiz_margin=0.0,
        lambda_v=args.lambda_v, lambda_h=args.lambda_h, huber_delta=1.0).to(device)

    hsharp_fn = HorizontalSharpenedConsistency(
        D=(args.max_disp_px // (args.patch_size // 2)),  # ★ 1/4 스텝 기반 D
        tau_sharp=args.tau_sharp, huber_delta=args.huber_delta_h,
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

    reproj_loss_fn = FeatureReprojLoss().to(device)
    
    seed_anchor_fn = SeedAnchorHuberLoss(
        tau=args.seed_tau, huber_delta=args.seed_huber_delta
    ).to(device)

    # PhotometricLoss 생성자 인자 수정됨
    photo_crit = PhotometricLoss(w_l1 = args.photo_l1_w, w_ssim = args.photo_ssim_w).to(device)

    sky_loss = SkyGridZeroLoss(
        max_disp_px=args.max_disp_px,
        patch_size=args.patch_size).to(device)

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
                append_if_exists=True,
                title="Stereo Training Arguments"
            )
        print(f"[Log] Arguments written to: {log_path}")
        
    model.train()
    
    for epoch in range(start_epoch, args.epochs + 1):
        running = 0.0
        for it, (imgL, imgR, names) in enumerate(loader, start=1):
            imgL = imgL.to(device, non_blocking=True)
            imgR = imgR.to(device, non_blocking=True)

            # ★ 모델/포토메트릭용 0..1 이미지
            with torch.no_grad():
                imgL_01 = denorm_imagenet(imgL)  # [B,3,H,W] in 0..1
                imgR_01 = denorm_imagenet(imgR)  # [B,3,H,W] in 0..1

            # ★ 풀 해상도에서 좌영상 향상(선택)
            imgL_enh_01 = enhance_batch_bgr_from_rgb01(
                imgL_01, enable=(not args.no_enhance),
                gamma=args.enhance_gamma,
                clahe_clip=args.enhance_clahe_clip,
                clahe_tile=args.enhance_clahe_tile
            )

            with torch.cuda.amp.autocast(enabled=args.amp):
                # ★ 모델 입력은 0..1 이미지
                prob, disp_soft, aux = model(imgL_01, imgR_01)

                # ★ loss용 특징(DINO 분기)
                FL_dino = aux["FL"]["dino"]  # [B,Cd,H/4,W/4]
                FR_dino = aux["FR"]["dino"]  # [B,Cd,H/4,W/4]

                raw_vol  = aux["raw_volume"]            # [B,1,D+1,H/4,W/4]
                mask_d   = aux["mask"]                  # [B,1,D+1,H/4,W/4]
                refined_masked = aux["refined_masked"]  # [B,1,D+1,H/4,W/4]

                # === ROI 전영역(=1) @ 1/4 & 1/1 ===
                roi_patch = torch.ones_like(disp_soft)             # [B,1,H/4,W/4]
                disp_full_px = aux["disp_full_px"]                 # ★ [B,1,H,W]
                roi_full  = torch.ones_like(disp_full_px)          # [B,1,H,W]

                # ★ 오른쪽 풀 이미지 워핑 → photometric(풀 해상도)
                imgR_full_warp_01, valid_full = warp_right_to_left_image(imgR_01, disp_full_px)
                
                photo_map = photo_crit.simple_photometric_loss(
                    imgL_enh_01, imgR_full_warp_01,
                    weights=[args.photo_l1_w, args.photo_ssim_w]
                )  # [B,1,H,W]
                
                photo_mask = roi_full * valid_full
                loss_photo = (photo_map * photo_mask).sum() / (photo_mask.sum() + 1e-6)
                loss_photo = loss_photo * args.w_photo

                # ★ Edge-aware smoothness (풀 해상도)
                loss_smooth = get_disparity_smooth_loss(disp_full_px, imgL_enh_01) * args.w_smooth

                # 1/4 손실들
                loss_dir    = dir_loss_fn(disp_soft, aux["feat_dir_L"], roi_patch) * args.w_dir
                loss_hsharp = hsharp_fn(refined_masked, FL_dino, roi_patch) * args.w_hsharp
                loss_prob   = prob_cons_fn(prob, FL_dino, roi_patch) * args.w_probcons
                loss_ent    = entropy_fn(prob, FL_dino, roi_patch) * args.w_entropy
                loss_reproj = reproj_loss_fn(FL_dino, FR_dino, disp_soft, roi=roi_patch) * args.w_reproj

                # ★ Sky loss: refined_logits 1/4 → 1/8로, disp_full_px → half로 스케일 일치
                refined_logits_masked_1of8 = F.interpolate(
                    refined_masked,           # [B,1,D+1,H/4,W/4]
                    scale_factor=(1.0, 0.5, 0.5),
                    mode='trilinear',
                    align_corners=False
                )  # -> [B,1,D+1,H/8,W/8]

                disp_half_px_for_sky = F.interpolate(
                    disp_full_px, scale_factor=0.5,
                    mode='bilinear', align_corners=False
                ) * 0.5  # ★ 해상도 절반 → 값도 절반

                loss_sky, aux_sky = sky_loss(
                    refined_logits_masked=refined_logits_masked_1of8,
                    disp_half_px=disp_half_px_for_sky,
                    roi_half=None, roi_patch=None,
                    names=names,
                    step=(epoch-1)*len(loader)+it
                )

                loss = (
                    loss_dir + loss_hsharp + loss_prob + loss_ent +
                    loss_reproj + loss_photo + loss_smooth +
                    args.w_sky * loss_sky
                )

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # 안정화
            scaler.unscale_(optim)
            if hasattr(model, "agg"):
                torch.nn.utils.clip_grad_norm_(model.agg.parameters(), max_norm=5.0)
            if hasattr(model, "spx_full"):  # ★ 업샘플 헤드 모듈명
                torch.nn.utils.clip_grad_norm_(model.spx_full.parameters(), max_norm=5.0)
            scaler.step(optim)
            scaler.update()

            running += loss.item()
            if it % args.log_every == 0:
                with torch.no_grad():
                    disp_wta = aux["disp_wta"]
                    soft_dx = (roi_patch * (disp_soft - shift_with_mask(disp_soft,0,1)[0]).abs()).sum() / (roi_patch.sum()+1e-6)
                    wta_dx  = (roi_patch * (disp_wta  - shift_with_mask(disp_wta, 0,1)[0]).abs()).sum() / (roi_patch.sum()+1e-6)

                    # --- NEW: Realtime MS2 metrics (현재 배치) ---
                    extra_eval = ""
                    if enable_realtime:
                        H, W = imgL.shape[-2], imgL.shape[-1]
                        gt = load_ms2_gt_batch(
                            names=names,
                            target_hw=(H, W),
                            device=device,
                            gt_disp_dir=getattr(args, "gt_disp_dir", None),
                            gt_depth_dir=getattr(args, "gt_depth_dir", None),
                            gt_disp_scale=getattr(args, "gt_disp_scale", 1.0),
                            gt_depth_scale=getattr(args, "gt_depth_scale", 256.0),  # RGB depth PNG 기본 256 스케일 가정
                        )

                        disp_msg, depth_msg, depth_w_msg = "", "", ""

                        # (1) Depth 지표 (fx/baseline 필요)
                        has_fb = (getattr(args, "focal_px", 0.0) > 0.0) and (getattr(args, "baseline_m", 0.0) > 0.0)

                        if gt["gt_depth"] is not None and has_fb:
                            pred_depth = disparity_to_depth(aux["disp_full_px"], args.focal_px, args.baseline_m)
                            gt_depth   = gt["gt_depth"]
                            depth_metrics = compute_depth_metrics(pred_depth, gt_depth, gt["valid"], cfg=eval_cfg)
                            depth_w_metrics = compute_bin_weighted_depth_metrics(
                                pred_depth, gt_depth, gt["valid"],
                                max_depth_m=getattr(args, "eval_max_depth_m", 50.0),
                                num_bins=getattr(args, "eval_num_bins", 5),
                                cfg=eval_cfg
                            )
                            depth_msg   = fmt_depth_metrics(depth_metrics)
                            depth_w_msg = fmt_weighted_depth_metrics(depth_w_metrics)

                        # (2) Disparity 지표
                        gt_disp_for_metrics = gt["gt_disp"]
                        # GT disp 없고, GT depth + fb 있으면 depth->disp 변환해서 EPE/D1 계산
                        if gt_disp_for_metrics is None and gt["gt_depth"] is not None and has_fb:
                            gt_disp_for_metrics = (args.focal_px * args.baseline_m) / gt["gt_depth"].clamp_min(1e-6)
                        if gt_disp_for_metrics is not None:
                            disp_metrics = compute_ms2_disparity_metrics(
                                pred_disp=aux["disp_full_px"],
                                gt_disp=gt_disp_for_metrics,
                                valid=gt["valid"],
                                cfg=eval_cfg
                            )
                            disp_msg = fmt_disp_metrics(disp_metrics)

                        parts = []
                        if disp_msg:    parts.append("[Disp] "    + disp_msg)
                        if depth_msg:   parts.append("[Depth] "   + depth_msg)
                        if depth_w_msg: parts.append("[Depth-W] " + depth_w_msg)
                        if parts:
                            extra_eval = " || " + "  ".join(parts)
                    # ------------------------------------------------

                print(f"[Epoch {epoch:03d} | Iter {it:04d}/{len(loader)}] "
                      f"loss={running/args.log_every:.4f} !!"
                      f"(dir={loss_dir.item():.4f}, hsharp={loss_hsharp.item():.4f}, "
                      f"prob={loss_prob.item():.4f}, ent={loss_ent.item():.4f}, "
                      f"rep={(loss_reproj/max(args.w_reproj,1e-9)).item():.4f}, "
                      f"photo={(loss_photo/max(args.w_photo,1e-9)).item():.4f}, smooth={(loss_smooth/max(args.w_smooth,1e-9)).item():.4f}, "
                    #   f"sky={(loss_sky/max(args.w_sky,1e-9)).item():.4f}) "
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

    # 모델/학습
    p.add_argument("--max_disp_px", type=int, default=88)   # ★ 4의 배수 권장 (1/4 스텝=4px)
    p.add_argument("--patch_size",  type=int, default=8)    # ★ 1/4 격자 스텝 = 4
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
    p.add_argument("--sim_thr",      type=float, default=0.75)
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

    # Photometric / Smoothness
    p.add_argument("--w_photo",    type=float, default=1.0)
    p.add_argument("--w_smooth",   type=float, default=0.1)
    p.add_argument("--photo_l1_w",   type=float, default=0.15)
    p.add_argument("--photo_ssim_w", type=float, default=0.85)

    # Enhance 옵션
    p.add_argument("--no_enhance", dest="no_enhance", action="store_true", help="저조도 보정 비활성화")
    p.set_defaults(no_enhance=False)
    p.add_argument("--enhance_gamma",      type=float, default=1.8)
    p.add_argument("--enhance_clahe_clip", type=float, default=2.0)
    p.add_argument("--enhance_clahe_tile", type=int,   default=8)

    # ---[ Resume / Checkpoint ]---
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--resume_non_strict", action="store_true")
    p.add_argument("--resume_reset_optim", action="store_true")
    p.add_argument("--resume_reset_scaler", action="store_true")

    # 로깅/저장
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=2)
    p.add_argument("--save_dir", type=str, default=f"./log/checkpoints_{current_time}")
    
    # --- Seeded Prior ---
    p.add_argument("--w_seed", type=float, default=0.0)
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
    p.add_argument("--w_sky", type=float, default=0.0)


    # Realtime eval
    p.add_argument("--realtime_test", action="store_true")
    p.add_argument("--gt_disp_dir", type=str, default=None)
    p.add_argument("--gt_depth_dir", type=str, default=None)
    p.add_argument("--gt_disp_scale", type=float, default=1.0)
    p.add_argument("--gt_depth_scale", type=float, default=256.0)  # RGB depth PNG 기본 256 가정
    p.add_argument("--eval_num_bins", type=int, default=5)
    p.add_argument("--eval_max_depth_m", type=float, default=50.0)

    # Calib auto-load
    p.add_argument("--calib_npy", type=str, default="/home/jaejun/dataset/MS2/sync_data/_2021-08-13-22-36-41/calib.npy")
    p.add_argument("--K_left_npy", type=str, default="/home/jaejun/dataset/MS2/intrinsic_left.npy")
    p.add_argument("--T_lr_npy", type=str, default=None, help="4x4 extrinsic .npy (left->right)")
    p.add_argument("--focal_px", type=float, default=764.5138549804688)
    p.add_argument("--baseline_m", type=float, default=0.29918420530585865)




    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
