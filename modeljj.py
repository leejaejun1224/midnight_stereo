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

# ---------------------------
# Utils
# ---------------------------

def assert_multiple(x, m, name="size"):
    if x % m != 0:
        raise ValueError(f"{name}={x} 는 {m}의 배수여야 합니다.")


def build_corr_volume_with_mask(FL: torch.Tensor, FR: torch.Tensor, D: int):
    B, C, H, W = FL.shape
    vols, masks = [], []
    for d in range(D + 1):
        if d == 0:
            FR_shift = FR
            valid = torch.ones((B, 1, H, W), device=FL.device, dtype=FL.dtype)
        else:
            FR_shift = F.pad(FR, (d, 0, 0, 0))[:, :, :, :W]
            valid = torch.ones((B, 1, H, W), device=FL.device, dtype=FL.dtype)
            valid[:, :, :, :d] = 0.0
        corr = (FL * FR_shift).sum(dim=1, keepdim=True)  # [B,1,H,W]
        vols.append(corr.squeeze(1))
        masks.append(valid.squeeze(1))
    vol  = torch.stack(vols,  dim=1).unsqueeze(1)   # [B,1,D+1,H,W]
    mask = torch.stack(masks, dim=1).unsqueeze(1)   # [B,1,D+1,H,W]
    return vol, mask

# ---------------------------
# DINO v1 ViT-B/8 features (1/8 grid)
# ---------------------------

class DINOvits8Features(nn.Module):
    def __init__(self, patch_size: int = 8):
        super().__init__()
        self.patch = patch_size
        self.backbone = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        assert_multiple(H, self.patch, "height")
        assert_multiple(W, self.patch, "width")
        tokens = self.backbone.get_intermediate_layers(x, n=1)[0]  # [B,1+P,C]
        patch_tokens = tokens[:, 1:, :]
        C = patch_tokens.shape[-1]
        H8, W8 = H // self.patch, W // self.patch
        feat = patch_tokens.transpose(1, 2).contiguous().view(B, C, H8, W8)  # [B,C,H/8,W/8]
        feat = F.normalize(feat, dim=1, eps=1e-6)
        return feat

# ---------------------------
# 3D U-Net like Aggregation (H/W만 다운)
# ---------------------------

class Conv3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm='bn', groups=8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        if norm == 'gn':
            self.bn1 = nn.GroupNorm(groups, out_ch)
            self.bn2 = nn.GroupNorm(groups, out_ch)
        else:
            self.bn1 = nn.BatchNorm3d(out_ch)
            self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class Down3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm='bn', groups=8):
        super().__init__()
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=(1,2,2), padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.GroupNorm(groups, out_ch) if norm=='gn' else nn.BatchNorm3d(out_ch)
    def forward(self, x):
        return self.relu(self.bn(self.down(x)))

class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm='bn', groups=8):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=3, stride=(1,2,2), padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.GroupNorm(groups, out_ch) if norm=='gn' else nn.BatchNorm3d(out_ch)
    def forward(self, x, out_hw=None):
        if out_hw is not None:
            B, _, D, _, _ = x.shape
            Ht, Wt = out_hw
            x = self.up(x, output_size=(B, self.up.out_channels, D, Ht, Wt))
        else:
            x = self.up(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CostAggregator3D(nn.Module):
    def __init__(self, base_ch: int = 32, depth: int = 3, norm='bn'):
        super().__init__()
        ch = base_ch
        # Encoder
        self.enc1 = Conv3DBlock(1, ch, norm)
        self.down1 = Down3D(ch, ch*2, norm)
        self.enc2 = Conv3DBlock(ch*2, ch*2, norm)
        self.down2 = Down3D(ch*2, ch*4, norm)
        self.enc3 = Conv3DBlock(ch*4, ch*4, norm)
        self.down3 = Down3D(ch*4, ch*8, norm) if depth >= 3 else nn.Identity()
        # Bottleneck
        if depth >= 3:
            self.bott = Conv3DBlock(ch*8, ch*8, norm)
            self.up3  = Up3D(ch*8, ch*4, norm)
        # Decoder
        self.dec3 = Conv3DBlock(ch*8 if depth >= 3 else ch*4, ch*4, norm)
        self.up2  = Up3D(ch*4, ch*2, norm)
        self.dec2 = Conv3DBlock(ch*4, ch*2, norm)
        self.up1  = Up3D(ch*2, ch, norm)
        self.dec1 = Conv3DBlock(ch*2, ch, norm)  # ※ concat 후 2ch → ch
        self.out  = nn.Conv3d(ch, 1, kernel_size=1)
        self.depth = depth

    def forward(self, x):
        e1 = self.enc1(x); d1 = self.down1(e1)
        e2 = self.enc2(d1); d2 = self.down2(e2)
        e3 = self.enc3(d2)
        if self.depth >= 3:
            d3 = self.down3(e3)
            b  = self.bott(d3)
            u3 = self.up3(b, out_hw=(e3.shape[-2], e3.shape[-1]))
            c3 = torch.cat([u3, e3], dim=1)
        else:
            c3 = e3
        dec3 = self.dec3(c3)
        u2   = self.up2(dec3, out_hw=(e2.shape[-2], e2.shape[-1]))
        c2   = torch.cat([u2, e2], dim=1)
        dec2 = self.dec2(c2)
        u1   = self.up1(dec2, out_hw=(e1.shape[-2], e1.shape[-1]))
        c1   = torch.cat([u1, e1], dim=1)
        dec1 = self.dec1(c1)
        out  = self.out(dec1)   # [B,1,D+1,H',W']
        return out

# ---------------------------
# Soft + ArgMax
# ---------------------------

class SoftAndArgMax(nn.Module):
    def __init__(self, D: int, temperature: float = 0.7):
        super().__init__()
        self.register_buffer("disp_values", torch.arange(0, D+1, dtype=torch.float32).view(1,1,D+1,1,1))
        self.t = temperature
    def forward(self, vol_masked: torch.Tensor):
        prob = torch.softmax(vol_masked / self.t, dim=2)              # [B,1,D+1,H',W']
        disp_soft = (prob * self.disp_values).sum(dim=2)              # [B,1,H',W']
        disp_wta  = vol_masked.argmax(dim=2, keepdim=False).float()   # [B,1,H',W']
        return prob, disp_soft, disp_wta

# ---------------------------
# (참고) 기존 convex 업샘플 함수 (미사용)
# ---------------------------

def convex_upsample_2d_scalar(disp_lo: torch.Tensor, mask: torch.Tensor, scale: int) -> torch.Tensor:
    B, C, H, W = disp_lo.shape
    assert C == 1
    mask = mask.view(B, 1, 9, scale, scale, H, W)
    mask = torch.softmax(mask, dim=2)
    unf = F.unfold(disp_lo, kernel_size=3, padding=1)
    unf = unf.view(B, 1, 9, H, W).unsqueeze(3).unsqueeze(4)
    up = torch.sum(mask * unf, dim=2)
    up = up.permute(0,1,4,2,5,3).contiguous().view(B, 1, H*scale, W*scale)
    return up

# ---------------------------
# ACVNet-style context upsample (동일 동작)
# ---------------------------

def context_upsample(depth_low, up_weights):
    """
    depth_low: [B,1,h,w]   (여기서는 h=H/8)
    up_weights: [B,9,4*h,4*w]  (여기서는 [B,9,H/2,W/2])
    return: [B, H/2, W/2]  (채널 1 생략 형식 그대로 유지)
    """
    b, c, h, w = depth_low.shape
    depth_unfold = F.unfold(depth_low.reshape(b,c,h,w), 3, 1, 1).reshape(b, 9, h, w)
    depth_unfold = F.interpolate(depth_unfold, (h*4, w*4), mode='nearest').reshape(b, 9, h*4, w*4)
    depth = (depth_unfold * up_weights).sum(1)
    return depth

# ---------------------------
# 2D 유틸 블록 (stem / spx에 사용)
# ---------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SPX2Block(nn.Module):
    """
    H/4 특징을 ConvTranspose2d로 H/2로 올리고, stem_2(H/2)와 skip concat 후 3x3로 정리.
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)  # H/4 -> H/2
        self.fuse = nn.Sequential(
            ConvBNReLU(out_ch + skip_ch, out_ch, k=3, s=1, p=1),
        )
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return x

# ---------------------------
# Stereo Model with context_upsample
# ---------------------------

class StereoModel(nn.Module):
    def __init__(self, max_disp_px: int = 64, patch_size: int = 8,
                 agg_base_ch: int = 32, agg_depth: int = 3, softarg_t: float = 0.9,
                 norm='gn'):
        super().__init__()
        self.patch = patch_size
        assert_multiple(max_disp_px, patch_size, "max_disp_px")
        if self.patch % 2 != 0:
            raise ValueError(f"patch_size={self.patch} 는 짝수여야 합니다. (1/8 → 1/2 업샘플에 필요)")
        self.D = max_disp_px // patch_size

        # 1) Feature (ViT-B/8, 1/8 grid)
        self.feat_net = DINOvits8Features(patch_size)

        # 2) 3D aggregation + soft-argmax
        self.agg  = CostAggregator3D(base_ch=agg_base_ch, depth=agg_depth, norm=norm)
        self.post = SoftAndArgMax(D=self.D, temperature=softarg_t)

        # 3) context upsample 준비 (1/8 → 1/2 == ×4)
        self.up_scale = self.patch // 2  # e.g., patch=8 → 4

        # stem 경로: 입력 이미지에서 저층 피처 (H/2, H/4)
        self.stem_2 = nn.Sequential(
            ConvBNReLU(3, 32, k=3, s=2, p=1),             # H → H/2
            ConvBNReLU(32, 32, k=3, s=1, p=1),
        )
        self.stem_4 = nn.Sequential(
            ConvBNReLU(32, 48, k=3, s=2, p=1),            # H/2 → H/4
            ConvBNReLU(48, 48, k=3, s=1, p=1),
        )

        feat_ch = getattr(self.feat_net.backbone, "embed_dim", 384)  # ViT-B/8: 768
        # H/4에서 FL(업샘플)과 stem_4을 결합해 spx 분기 입력 생성
        self.spx_4 = nn.Sequential(
            ConvBNReLU(feat_ch + 48, 64, k=3, s=1, p=1)   # [B,64,H/4,W/4]
        )
        # H/4 → H/2 업샘플 + stem_2 skip 결합
        self.spx_2 = SPX2Block(in_ch=64, skip_ch=32, out_ch=64)      # [B,64,H/2,W/2]

        # 최종 업샘플 가중치(3×3=9개) 예측 (H/2, W/2)
        self.spx_head = nn.Conv2d(64, 9, kernel_size=3, stride=1, padding=1)

    @torch.no_grad()
    def extract_feats(self, left: torch.Tensor, right: torch.Tensor):
        FL = self.feat_net(left)   # [B,C,H/8,W/8]
        FR = self.feat_net(right)  # [B,C,H/8,W/8]
        return FL, FR

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        """
        Returns:
          prob: [B,1,D+1,H/8,W/8]
          disp_soft: [B,1,H/8,W/8] (단위: patch)
          aux: dict(
             FL/FR,
             raw_volume/refined/refined_masked/mask/disp_wta,
             disp_half: [B,1,H/2,W/2] (단위: patch),
             disp_half_px: [B,1,H/2,W/2] (단위: half 픽셀)
          )
        """
        B, _, H, W = left.shape

        # 1) features
        FL, FR = self.extract_feats(left, right)                         # [B,C,H/8,W/8]

        # 2) cost volume + 3D aggregation
        vol, mask = build_corr_volume_with_mask(FL, FR, self.D)          # [B,1,D+1,H/8,W/8]
        vol_in = vol * mask
        refined = self.agg(vol_in)                                       # [B,1,D+1,H/8,W/8]
        refined_masked = refined + (1.0 - mask) * (-1e4)

        # 3) soft-arg + WTA
        prob, disp_soft, disp_wta = self.post(refined_masked)            # disp_soft: [B,1,H/8,W/8]
        raw_for_anchor = (vol + (1.0 - mask) * (-1e4)).detach()

        # 4) context upsample 가중치 예측 (입력 이미지 경로 활용)
        # stem 저층 피처
        stem2 = self.stem_2(left)                                        # [B,32,H/2,W/2]
        stem4 = self.stem_4(stem2)                                       # [B,48,H/4,W/4]

        # FL(1/8)을 H/4로 올려 stem4와 결합 → spx_4
        FL4 = F.interpolate(FL, size=(H // 4, W // 4), mode='bilinear', align_corners=False)  # [B,C,H/4,W/4]
        spx4_in = torch.cat([FL4, stem4], dim=1)                         # [B,C+48,H/4,W/4]
        xspx4 = self.spx_4(spx4_in)                                      # [B,64,H/4,W/4]

        # H/4 → H/2 업샘플(+ stem2 skip)
        xspx2 = self.spx_2(xspx4, stem2)                                 # [B,64,H/2,W/2]

        # 픽셀당 3×3 가중치 9채널 예측 (softmax로 볼록결합 보장)
        spx_logits = self.spx_head(xspx2)                                # [B,9,H/2,W/2]
        spx_pred = F.softmax(spx_logits, dim=1)                          # [B,9,H/2,W/2]
        # 안정성 체크(선택): assert spx_pred.shape[2:] == (H//2, W//2)

        # 5) ACVNet-style context_upsample: 1/8 → 1/2 (×4)
        disp_half = context_upsample(disp_soft, spx_pred).unsqueeze(1)   # [B,1,H/2,W/2] (단위: patch)
        disp_half_px = disp_half * float(self.patch / 2.0)               # half-res 'px' 단위

        return prob, disp_soft, {
            "FL": FL, "FR": FR,
            "raw_volume": raw_for_anchor,
            "refined_volume": refined,
            "mask": mask,
            "refined_masked": refined_masked,
            "disp_wta": disp_wta,
            "disp_half": disp_half,
            "disp_half_px": disp_half_px,
        }
