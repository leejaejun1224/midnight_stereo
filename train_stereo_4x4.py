import os
import glob
import argparse
import random
from typing import Tuple, List, Dict

import cv2  # NEW
import numpy as np  # NEW

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from datetime import datetime, timezone, timedelta

# ============================================================
# 유틸
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def assert_multiple(x, m, name="size"):
    if x % m != 0:
        raise ValueError(f"{name}={x} 는 {m}의 배수여야 합니다.")

def stem(path: str) -> str:
    s = os.path.splitext(os.path.basename(path))[0]
    return s


# ============================================================
# 데이터셋 (+ 하늘 마스크)
# ============================================================

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


# ============================================================
# DINO(v1) ViT-S/8 특징 추출 (feature-only)
# ============================================================

class DINOvits8Features(nn.Module):
    """
    - 입력:  [B,3,H,W]  (H,W는 8의 배수)
    - 출력:  [B,C,H/8,W/8] (L2 정규화된 마지막 레이어 패치 특징)
    - extract_tokens(x, n): neck용 마지막 n개 레이어 토큰 리스트
    """
    def __init__(self, patch_size: int = 8):
        super().__init__()
        self.patch = patch_size
        self.backbone = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        assert_multiple(H, self.patch, "height")
        assert_multiple(W, self.patch, "width")
        tokens_last = self.backbone.get_intermediate_layers(x, n=1)[0]  # [B, 1+P, C]
        patch_tokens = tokens_last[:, 1:, :]                            # [B, P, C]
        C = patch_tokens.shape[-1]
        H8, W8 = H // self.patch, W // self.patch
        feat = patch_tokens.transpose(1, 2).contiguous().view(B, C, H8, W8)
        feat = F.normalize(feat, dim=1, eps=1e-6)
        return feat

    @torch.no_grad()
    def extract_tokens(self, x: torch.Tensor, n: int = 4) -> List[torch.Tensor]:
        B, _, H, W = x.shape
        assert_multiple(H, self.patch, "height")
        assert_multiple(W, self.patch, "width")
        outs = self.backbone.get_intermediate_layers(x, n=n)  # list of [B,1+P,C]
        toks: List[torch.Tensor] = []
        for t in outs:
            toks.append(t[:, 1:, :].contiguous())  # [B,P,C]
        return toks


# ============================================================
# Neck: 1-step up (H/8 → r=2 → H/4), 절대 H까지 올리지 않음
# ============================================================

def tokens_to_map(patch_tokens: torch.Tensor, H: int, W: int, patch: int) -> torch.Tensor:
    B, P, C = patch_tokens.shape
    assert_multiple(H, patch, "height")
    assert_multiple(W, patch, "width")
    Hs, Ws = H // patch, W // patch
    if P != Hs * Ws:
        raise AssertionError(f"P must equal (H/patch)*(W/patch)={Hs*Ws}, got {P}")
    fmap = patch_tokens.view(B, Hs, Ws, C).permute(0, 3, 1, 2).contiguous()
    return fmap  # [B,C,H/patch,W/patch]

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class QuarterNeck(nn.Module):
    """
    ViT 패치 토큰(깊→얕)을 H/8에서 융합 → 한 번(r=2) 업샘플 → 정확히 1/4 특징
    출력: [B, C_out, floor(H/4), floor(W/4)]
    """
    def __init__(self, c_vit: int, c_mid: int = 256, c_out: int = 256, patch: int = 8,
                 up_kind: str = "deconv"):
        super().__init__()
        self.patch = patch
        self.up_kind = up_kind.lower()
        self.proj = nn.ModuleList([nn.Conv2d(c_vit, c_mid, 1) for _ in range(4)])
        self.fuse1 = ConvBNReLU(c_mid * 2, c_mid)
        self.fuse2 = ConvBNReLU(c_mid * 2, c_mid)
        self.fuse3 = ConvBNReLU(c_mid * 2, c_mid)
        if self.up_kind == "deconv":
            self.up = nn.Sequential(
                nn.ConvTranspose2d(c_mid, c_mid, kernel_size=2, stride=2),
                nn.BatchNorm2d(c_mid), nn.ReLU(inplace=True),
                nn.Conv2d(c_mid, c_out, 3, 1, 1),
            )
        elif self.up_kind == "bilinear":
            self.up_conv = nn.Sequential(
                nn.Conv2d(c_mid, c_out, 3, 1, 1),
                nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            )
        elif self.up_kind == "pixelshuffle":
            self.up = nn.Sequential(
                nn.Conv2d(c_mid, c_mid * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.Conv2d(c_mid, c_out, 3, 1, 1),
            )
        else:
            raise ValueError(f"Unknown up_kind: {self.up_kind}")

    def forward(self, toks: List[torch.Tensor], H: int, W: int) -> torch.Tensor:
        if len(toks) < 4:
            toks = (toks + [toks[-1]] * 4)[:4]
        elif len(toks) > 4:
            toks = toks[:4]

        fmaps: List[torch.Tensor] = []
        for i, t in enumerate(toks):
            fmap = tokens_to_map(t, H, W, self.patch)  # [B,Cvit,H/8,W/8]
            fmaps.append(self.proj[i](fmap))           # [B,c_mid,H/8,W/8]

        x = self.fuse1(torch.cat([fmaps[0], fmaps[1]], dim=1))
        x = self.fuse2(torch.cat([x,        fmaps[2]], dim=1))
        x = self.fuse3(torch.cat([x,        fmaps[3]], dim=1))

        if self.up_kind == "bilinear":
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = self.up_conv(x)
        else:
            x = self.up(x)

        H4, W4 = H // 4, W // 4
        if x.shape[-2] != H4 or x.shape[-1] != W4:
            x = F.interpolate(x, size=(H4, W4), mode="bilinear", align_corners=False)
        return x


# ============================================================
# 특징 추출기 래퍼/팩토리 (1/8 또는 1/4)
# ============================================================

class DINOFeat1by8(nn.Module):
    stride: int = 8
    def __init__(self, dino_patch: int = 8):
        super().__init__()
        self.backbone = DINOvits8Features(patch_size=dino_patch)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class DINOFeat1by4(nn.Module):
    stride: int = 4
    def __init__(self, n_layers: int = 4, c_vit: int = 384,
                 neck_mid_ch: int = 256, neck_out_ch: int = 256,
                 dino_patch: int = 8):
        super().__init__()
        self.backbone = DINOvits8Features(patch_size=dino_patch)
        self.n_layers = n_layers
        self.neck = QuarterNeck(c_vit=c_vit, c_mid=neck_mid_ch, c_out=neck_out_ch, patch=dino_patch)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        with torch.no_grad():
            toks = self.backbone.extract_tokens(x, n=self.n_layers)
        fmap_1_4 = self.neck(toks, H, W)  # [B, C_out, H/4, W/4]
        fmap_1_4 = F.normalize(fmap_1_4, dim=1, eps=1e-6)
        return fmap_1_4

def build_feature_extractor(kind: str,
                            n_layers: int = 4,
                            neck_mid_ch: int = 256,
                            neck_out_ch: int = 256,
                            dino_patch: int = 8,
                            c_vit: int = 384) -> nn.Module:
        k = kind.lower()
        if k in ["vits8_1by8", "vits8", "dino_vits8_1by8"]:
            return DINOFeat1by8(dino_patch=dino_patch)
        elif k in ["vits8_1by4", "quarter", "dino_vits8_1by4"]:
            return DINOFeat1by4(n_layers=n_layers, c_vit=c_vit,
                                neck_mid_ch=neck_mid_ch, neck_out_ch=neck_out_ch,
                                dino_patch=dino_patch)
        else:
            raise ValueError(f"Unknown feature kind: {kind}")


# ============================================================
# 상관 볼륨(+ 마스크)
# ============================================================

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


# ============================================================
# 3D Aggregation
# ============================================================

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
        self.enc1 = Conv3DBlock(1, ch, norm)
        self.down1 = Down3D(ch, ch*2, norm)
        self.enc2 = Conv3DBlock(ch*2, ch*2, norm)
        self.down2 = Down3D(ch*2, ch*4, norm)
        self.enc3 = Conv3DBlock(ch*4, ch*4, norm)
        self.down3 = Down3D(ch*4, ch*8, norm) if depth >= 3 else nn.Identity()
        if depth >= 3:
            self.bott = Conv3DBlock(ch*8, ch*8, norm)
            self.up3  = Up3D(ch*8, ch*4, norm)
        self.dec3 = Conv3DBlock(ch*8 if depth >= 3 else ch*4, ch*4, norm)
        self.up2  = Up3D(ch*4, ch*2, norm)
        self.dec2 = Conv3DBlock(ch*4, ch*2, norm)
        self.up1  = Up3D(ch*2, ch, norm)
        self.dec1 = Conv3DBlock(ch*2, ch, norm)
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


# ============================================================
# Soft + ArgMax (+ Top2 Soft-Exp)
# ============================================================

class SoftAndArgMax(nn.Module):
    def __init__(self, D: int, temperature: float = 0.7):
        super().__init__()
        self.register_buffer("disp_values", torch.arange(0, D+1, dtype=torch.float32).view(1,1,D+1,1,1))
        self.t = temperature
    def forward(self, vol_masked: torch.Tensor):
        # prob over disparity
        prob = torch.softmax(vol_masked / self.t, dim=2)            # [B,1,D+1,H',W']
        # standard soft-argmax
        disp_soft = (prob * self.disp_values).sum(dim=2)            # [B,1,H',W']
        # top-2 only (no renorm): sum_{k in top2} p_k * d_k
        topv, topi = torch.topk(prob, k=2, dim=2)                   # [B,1,2,H',W']
        disp_vals = self.disp_values.expand_as(prob)
        topd = torch.gather(disp_vals, dim=2, index=topi)           # [B,1,2,H',W']
        disp_soft_top2 = (topv * topd).sum(dim=2)                   # [B,1,H',W']
        # WTA
        disp_wta  = vol_masked.argmax(dim=2, keepdim=False).float() # [B,1,H',W']
        return prob, disp_soft, disp_wta, disp_soft_top2


# ============================================================
# 보조: 시프트/ROI
# ============================================================

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
    Hs, Ws = H // patch_size, W // patch_size
    if method == "nearest":
        roi = F.interpolate(valid_full, size=(Hs, Ws), mode="nearest")
    else:
        roi = F.avg_pool2d(valid_full, kernel_size=patch_size, stride=patch_size)
        roi = (roi >= thr).float()
    return roi


# ============================================================
# 클릭식 유사도 게이트(상대 스케일 + (선택) 동적 임계)
# ============================================================

@torch.no_grad()
def per_patch_min_similarity(feat_norm: torch.Tensor, sample_k: int = 512):
    B, C, H, W = feat_norm.shape
    P = H * W
    K = min(sample_k, P)
    F_bcp = feat_norm.view(B, C, P)
    F_bpc = F_bcp.permute(0, 2, 1).contiguous()
    idx = torch.randperm(P, device=feat_norm.device)[:K]
    bank = F_bcp[:, :, idx]
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
        thr_eff = torch.quantile(v, dynamic_q).item() if v.numel() > 0 else thr
    else:
        thr_eff = thr
    if gamma is None or gamma <= 0.0:
        w = (sim_norm >= thr_eff).to(sim_norm.dtype)
    else:
        w = torch.sigmoid((sim_norm - thr_eff) / gamma)
    return w * valid


# ============================================================
# (A) 방향 제약(soft disp 기준) + (A') 수직 단조 제약
# ============================================================

class DirectionalRelScaleDispLoss(nn.Module):
    """
    - 기존: 세로/가로 이웃과의 상대 차이를 Huber로 제한 (|Δ|<=margin)
    - 추가: 수직 단조 제약
        * dy = -1 (위): d_up <= d_cur + mono_margin
        * dy = +1 (아래): d_down >= d_cur - mono_margin
    """
    def __init__(self,
                 sim_thr=0.75, sim_gamma=0.0, sample_k=512,
                 use_dynamic_thr=True, dynamic_q=0.7,
                 vert_margin=1.0, horiz_margin=0.0,
                 lambda_v=1.0, lambda_h=1.0,
                 huber_delta=1.0,
                 # NEW (vertical monotonic):
                 lambda_v_mono: float = 1.0,
                 mono_margin: float = 0.0):
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
        self.lambda_v_mono = lambda_v_mono
        self.mono_margin = mono_margin

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

    def _accum_vertical_mono(self, disp, feat, roi):
        """
        위/아래에 대한 단조 제약:
          - 위(dy=-1): d_up <= d_cur + mono_margin  → hinge_up = relu(d_up - d_cur - mono_margin)
          - 아래(dy=+1): d_down >= d_cur - mono_margin → hinge_dn = relu(d_cur - d_down - mono_margin)
        """
        with torch.no_grad():
            min_vals = per_patch_min_similarity(feat, sample_k=self.sample_k)

        loss_sum = torch.tensor(0.0, device=disp.device)
        weight_sum = torch.tensor(0.0, device=disp.device)

        # 위쪽 이웃 (dy = -1): d_up <= d_cur + mono_margin
        dy, dx = -1, 0
        f_nb, valid_b = shift_with_mask(feat, dy, dx)
        d_nb, _       = shift_with_mask(disp, dy, dx)
        roi_nb, _     = shift_with_mask(roi,  dy, dx)
        valid = valid_b * roi * roi_nb
        sim_raw = (feat * f_nb).sum(dim=1, keepdim=True)
        w_up = rel_gate_from_sim_dynamic(sim_raw, min_vals, valid,
                                         self.sim_thr, self.sim_gamma,
                                         self.use_dynamic_thr, self.dynamic_q)
        hinge_up = (d_nb - disp - self.mono_margin).clamp(min=0.0)  # violation
        loss_sum += (w_up * hinge_up).sum()
        weight_sum += w_up.sum()

        # 아래쪽 이웃 (dy = +1): d_down >= d_cur - mono_margin
        dy, dx = +1, 0
        f_nb, valid_b = shift_with_mask(feat, dy, dx)
        d_nb, _       = shift_with_mask(disp, dy, dx)
        roi_nb, _     = shift_with_mask(roi,  dy, dx)
        valid = valid_b * roi * roi_nb
        sim_raw = (feat * f_nb).sum(dim=1, keepdim=True)
        w_dn = rel_gate_from_sim_dynamic(sim_raw, min_vals, valid,
                                         self.sim_thr, self.sim_gamma,
                                         self.use_dynamic_thr, self.dynamic_q)
        hinge_dn = (disp - d_nb - self.mono_margin).clamp(min=0.0)  # violation
        loss_sum += (w_dn * hinge_dn).sum()
        weight_sum += w_dn.sum()

        return loss_sum / (weight_sum + 1e-6)

    def forward(self, disp, feat, roi):
        # 상대 스케일 제약 (기존)
        loss_v = self._accum(disp, feat, roi, self.vert_pairs, self.vert_margin)
        loss_h = self._accum(disp, feat, roi, self.hori_pairs, self.horiz_margin)
        # 수직 단조 제약 (추가)
        loss_v_mono = self._accum_vertical_mono(disp, feat, roi)
        return self.lambda_v * loss_v + self.lambda_h * loss_h + self.lambda_v_mono * loss_v_mono


# ============================================================
# (B) 샤픈 가로 일관성 (ArgMax 근사)
# ============================================================

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


# ============================================================
# (C) 분포-수준 일치(JS 유사), (D) 엔트로피 감소
# ============================================================

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
            pp = prob.squeeze(1).clamp_min(1e-8)
            topv = torch.topk(pp, k=2, dim=1).values
            conf = (topv[:,0] - topv[:,1]).unsqueeze(1)
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


# ============================================================
# (E) 앵커, (F) 재투영
# ============================================================

class CorrAnchorLoss(torch.nn.Module):
    def __init__(self, tau=0.6, margin=1.0, topk=2, use_huber=True):
        super().__init__()
        self.tau = tau; self.m = margin; self.k = topk; self.use_huber = use_huber
    def forward(self, raw_vol, disp, mask=None, roi=None):
        s = raw_vol.squeeze(1).detach()
        if mask is not None:
            m = mask.squeeze(1)
            s = s + (1.0 - m) * (-1e4)
        topv, topd = torch.topk(s, k=self.k, dim=1)
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
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B,1,1,1)
    shift_norm = 2.0 * disp_patch.squeeze(1) / max(W-1, 1)
    grid = base_grid.clone()
    grid[..., 0] = grid[..., 0] - shift_norm
    FR_w = F.grid_sample(FR, grid, mode='bilinear', padding_mode='zeros', align_corners=align_corners)
    ones = torch.ones((B,1,H,W), device=FR.device)
    M = F.grid_sample(ones, grid, mode='nearest', padding_mode='zeros', align_corners=align_corners)
    valid = (M > 0.5).float()
    return FR_w, valid

class FeatureReprojLoss(torch.nn.Module):
    def __init__(self): super().__init__()
    def forward(self, FL, FR, disp_patch, roi=None):
        FR_w, valid = warp_right_to_left_feat(FR, disp_patch)
        if roi is not None: valid = valid * roi
        cos = (FL * FR_w).sum(dim=1, keepdim=True)
        loss = (1.0 - cos).clamp(min=0.0) * valid
        return loss.sum() / (valid.sum() + 1e-6)


# ============================================================
# NEW: 저조도 보정 + 포토메트릭/스무스 도우미
# ============================================================

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

def denorm_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    std  = x.new_tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    y = x * std + mean
    return y.clamp(0.0, 1.0)

def batch_quarter_and_lowlight(rgb01_bchw: torch.Tensor,
                               enable: bool = True,
                               gamma: float = 1.8,
                               clahe_clip: float = 2.0,
                               clahe_tile: int = 8) -> torch.Tensor:
    # 1/4 축소
    img_q = F.interpolate(rgb01_bchw, scale_factor=0.25, mode="bilinear", align_corners=False)
    # 저조도 보정(OpenCV, CPU)
    out = []
    for b in range(img_q.shape[0]):
        arr = (img_q[b].detach().cpu().permute(1,2,0).numpy() * 255.0).astype(np.uint8)  # RGB uint8
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        bgr2 = enhance_low_light_bgr(bgr, enable=enable, gamma=gamma, clahe_clip=clahe_clip, clahe_tile=clahe_tile)
        rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb2).float().permute(2,0,1) / 255.0
        out.append(t)
    out = torch.stack(out, dim=0).to(rgb01_bchw.device)
    return out

def resample_disparity_to(disp_in: torch.Tensor, target_hw: Tuple[int,int]) -> torch.Tensor:
    B, _, h, w = disp_in.shape
    Ht, Wt = target_hw
    disp_up = F.interpolate(disp_in, size=(Ht, Wt), mode="bilinear", align_corners=False)
    scale_x = Wt / float(max(w, 1))
    return disp_up * scale_x

def warp_right_to_left_img(imgR: torch.Tensor, disp_patch: torch.Tensor, align_corners: bool = True):
    return warp_right_to_left_feat(imgR, disp_patch, align_corners=align_corners)

# --- Photometric & Smoothness Losses (사용자 제공 내용 통합) ---
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images"""
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
    def identiy_photometric_loss(self, source_image, targate_image, weights = [0.15,0.85]):
        return self.simple_photometric_loss(source_image, targate_image, weights)
    def minimum_photometric_loss(self,original_image, reconstructed_images):
        losses = []
        for recon_image in reconstructed_images:
            losses.append(self.simple_photometric_loss(original_image, recon_image, self.weights))
        losses = torch.stack(losses, dim=1)
        return torch.min(losses, dim=1)[0]

def get_disparity_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image (edge-aware)"""
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


# ============================================================
# 전체 모델 (stride-aware)
# ============================================================

class StereoModel(nn.Module):
    def __init__(self,
                 feat_extractor: nn.Module,         # 1/8 또는 1/4
                 max_disp_px: int = 64,
                 agg_base_ch: int = 32, agg_depth: int = 3,
                 softarg_t: float = 0.7,
                 norm='bn'):
        super().__init__()
        self.feat_net = feat_extractor
        self.stride = getattr(self.feat_net, "stride", 8)  # 8 또는 4
        assert_multiple(max_disp_px, self.stride, "max_disp_px")
        self.D = max_disp_px // self.stride

        self.agg = CostAggregator3D(base_ch=agg_base_ch, depth=agg_depth, norm=norm)
        self.post = SoftAndArgMax(D=self.D, temperature=softarg_t)

    def extract_feats(self, left: torch.Tensor, right: torch.Tensor):
        FL = self.feat_net(left)
        FR = self.feat_net(right)
        return FL, FR

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        FL, FR = self.extract_feats(left, right)               # [B,C,H',W']
        vol, mask = build_corr_volume_with_mask(FL, FR, self.D)
        vol_in = vol * mask
        refined = self.agg(vol_in)                             # [B,1,D+1,H',W']
        refined_masked = refined + (1.0 - mask) * (-1e4)
        prob, disp_soft, disp_wta, disp_soft_top2 = self.post(refined_masked)
        raw_for_anchor = (vol + (1.0 - mask) * (-1e4)).detach()
        return prob, disp_soft, {
            "FL": FL, "FR": FR,
            "raw_volume": raw_for_anchor,
            "refined_volume": refined,
            "mask": mask,
            "refined_masked": refined_masked,
            "disp_wta": disp_wta,
            "disp_soft_top2": disp_soft_top2
        }


# ============================================================
# 학습 루프
# ============================================================

def build_optimizer(params, name='adamw', lr=1e-3, weight_decay=1e-2):
    name = name.lower()
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    elif name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    else:
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)

def train(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = StereoFolderDataset(args.left_dir, args.right_dir,
                                  height=args.height, width=args.width,
                                  mask_dir=args.mask_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    # --- 특징 추출기 선택(1/8 또는 1/4) ---
    feat_extractor = build_feature_extractor(
        kind=args.feat_type,
        n_layers=args.feat_layers,
        neck_mid_ch=args.neck_mid_ch,
        neck_out_ch=args.neck_out_ch,
        dino_patch=8,
        c_vit=384
    )

    model = StereoModel(feat_extractor=feat_extractor,
                        max_disp_px=args.max_disp_px,
                        agg_base_ch=args.agg_ch, agg_depth=args.agg_depth,
                        softarg_t=args.softarg_t, norm=args.norm).to(device)

    print(f"[Info] Feature stride = {model.stride} (feat_type={args.feat_type}), "
          f"D = {model.D} (= max_disp_px // stride).")

    # --- 게이트용 DINO 1/8 백본 (항상 사용) ---
    gate_backbone = DINOvits8Features(patch_size=8).to(device).eval()

    # 픽셀 기준 마진을 stride(=4/8)로 나눠서 내부 마진으로 사용
    vert_m = args.vert_margin_px / model.stride
    hori_m = args.horiz_margin_px / model.stride

    # 손실 모듈
    dir_loss_fn = DirectionalRelScaleDispLoss(
        sim_thr=args.sim_thr, sim_gamma=args.sim_gamma, sample_k=args.sim_sample_k,
        use_dynamic_thr=args.use_dynamic_thr, dynamic_q=args.dynamic_q,
        vert_margin=vert_m, horiz_margin=hori_m,
        lambda_v=args.lambda_v, lambda_h=args.lambda_h, huber_delta=1.0,
        lambda_v_mono=args.lambda_v_mono, mono_margin=args.mono_margin
    ).to(device)

    hsharp_fn = HorizontalSharpenedConsistency(
        D=model.D, tau_sharp=args.tau_sharp, huber_delta=args.huber_delta_h,
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

    # NEW: photometric loss 인스턴스
    photo_loss_fn = PhotometricLoss(weights=[0.15, 0.85])

    optim = build_optimizer([p for p in model.parameters() if p.requires_grad],
                            name=args.optim, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for it, (imgL, imgR, valid_full, names) in enumerate(loader, start=1):
            imgL = imgL.to(device, non_blocking=True)
            imgR = imgR.to(device, non_blocking=True)
            valid_full = valid_full.to(device, non_blocking=True)  # [B,1,H,W]

            # ROI는 특징 stride로 축소 (1/8→8, 1/4→4)
            roi_patch = make_roi_patch(valid_full, patch_size=model.stride,
                                       method=args.roi_method, thr=args.roi_thr)

            with torch.cuda.amp.autocast(enabled=args.amp):
                prob, disp_soft, aux = model(imgL, imgR)
                FL, FR   = aux["FL"], aux["FR"]
                raw_vol  = aux["raw_volume"]
                mask_d   = aux["mask"]
                refined_masked = aux["refined_masked"]
                disp_soft_top2 = aux.get("disp_soft_top2", disp_soft)

            # ---- 게이트 특징: DINO 1/8 → (FL 해상도) 업샘플 ----
            with torch.no_grad():
                F8 = gate_backbone(imgL)  # [B,384, H/8, W/8]
                feat_gate = F.interpolate(F8, size=FL.shape[-2:], mode="bilinear", align_corners=False)
                feat_gate = F.normalize(feat_gate, dim=1, eps=1e-6)  # [B,384, H', W']  (H',W' == FL/disp 해상도)

            # ============================
            # NEW: 1/4 이미지 photometric & smoothness
            # ============================
            with torch.no_grad():
                # (1) 원본 정규화 해제 → RGB[0,1]
                left_rgb01  = denorm_imagenet(imgL)   # [B,3,H,W]
                right_rgb01 = denorm_imagenet(imgR)

                # (2) 1/4 축소 후 저조도 보정
                left_q  = batch_quarter_and_lowlight(
                            left_rgb01,
                            enable=bool(args.ll_enable),
                            gamma=args.ll_gamma,
                            clahe_clip=args.ll_clahe_clip,
                            clahe_tile=args.ll_clahe_tile)
                right_q = batch_quarter_and_lowlight(
                            right_rgb01,
                            enable=bool(args.ll_enable),
                            gamma=args.ll_gamma,
                            clahe_clip=args.ll_clahe_clip,
                            clahe_tile=args.ll_clahe_tile)

                # (3) 1/4 해상도용 ROI (하늘 제외)
                roi_quarter = make_roi_patch(valid_full, patch_size=4,
                                             method=args.roi_method, thr=args.roi_thr)  # [B,1,H/4,W/4]

            with torch.cuda.amp.autocast(enabled=args.amp):
                # (4) disparity를 1/4 해상도로 보간 + 값 스케일 (grad 유지)
                Hq, Wq = left_q.shape[-2], left_q.shape[-1]
                disp_for_photo = resample_disparity_to(disp_soft, (Hq, Wq))  # [B,1,H/4,W/4]

                # (5) 오른쪽 1/4 이미지를 disparity로 왼쪽 뷰로 워핑
                right_q_warp, valid_photo = warp_right_to_left_img(right_q, disp_for_photo)  # [B,3,H/4,W/4], [B,1,H/4,W/4]

                # (6) Photometric loss (L1 + SSIM)
                photo_map = photo_loss_fn.simple_photometric_loss(left_q, right_q_warp)      # [B,1,H/4,W/4]
                photo_mask = (roi_quarter * valid_photo)                                     # 유효영역만
                loss_photo = (photo_map * photo_mask).sum() / (photo_mask.sum() + 1e-6)

                # (7) Edge-aware smoothness (1/4)
                loss_smooth = multi_scale_disparity_smooth_loss(
                    [disp_for_photo], [left_q], weight=args.smooth_weight
                )

            with torch.cuda.amp.autocast(enabled=args.amp):
                # --- 손실 ---
                # NOTE: directional loss는 Top-2 기반 disparity 사용
                loss_dir    = dir_loss_fn(disp_soft_top2, feat_gate, roi_patch)
                loss_hsharp = hsharp_fn(refined_masked, feat_gate, roi_patch) * args.w_hsharp
                loss_prob   = prob_cons_fn(prob, feat_gate, roi_patch) * args.w_probcons
                loss_ent    = entropy_fn(prob, feat_gate, roi_patch) * args.w_entropy

                loss_anchor = anchor_loss_fn(raw_vol, disp_soft, mask=mask_d, roi=roi_patch) * args.w_anchor
                loss_reproj = reproj_loss_fn(FL, FR, disp_soft, roi=roi_patch) * args.w_reproj

                # NEW: photometric & smoothness 가중치 포함
                loss_photo_w  = args.w_photo * loss_photo
                loss_smooth_w = loss_smooth  # 함수 내부에 weight 적용됨

                if args.full_losses:
                    loss = loss_dir + loss_hsharp + loss_prob + loss_ent + loss_anchor + loss_reproj + loss_photo_w + loss_smooth_w
                else:
                    loss = loss_dir + loss_anchor + loss_reproj + loss_photo_w + loss_smooth_w

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.agg.parameters(), max_norm=5.0)
            scaler.step(optim)
            scaler.update()

            running += loss.item()
            if it % args.log_every == 0:
                with torch.no_grad():
                    disp_wta = aux["disp_wta"]
                    soft_dx = (roi_patch * (disp_soft - shift_with_mask(disp_soft,0,1)[0]).abs()).sum() / (roi_patch.sum()+1e-6)
                    wta_dx  = (roi_patch * (disp_wta  - shift_with_mask(disp_wta, 0,1)[0]).abs()).sum() / (roi_patch.sum()+1e-6)
                print(f"[Epoch {epoch:03d} | Iter {it:04d}/{len(loader)}] "
                      f"loss={running/args.log_every:.4f} "
                      f"(dir={loss_dir.item():.4f}, hsharp={loss_hsharp.item():.4f}, "
                      f"prob={loss_prob.item():.4f}, ent={loss_ent.item():.4f}, "
                      f"anc={(loss_anchor/max(args.w_anchor,1e-9)).item():.4f}, "
                      f"rep={(loss_reproj/max(args.w_reproj,1e-9)).item():.4f}, "
                      f"photo={(loss_photo).item():.4f}, smooth={(loss_smooth).item():.4f}) "
                      f"| mean|Δx| soft={soft_dx:.3f} wta={wta_dx:.3f} | dir uses top2")
                running = 0.0

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.save_dir, f"stereo_epoch{epoch:03d}.pth"))


# ============================================================
# 메인
# ============================================================

current_time = datetime.now(tz=timezone.utc).astimezone(timezone(timedelta(hours=9))).strftime("%y%m%d_%H%M%S")

def parse_args():
    p = argparse.ArgumentParser()
    # 데이터
    p.add_argument("--left_dir", type=str, required=True)
    p.add_argument("--right_dir", type=str, required=True)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width",  type=int, default=1224)
    p.add_argument("--mask_dir", type=str, default=None, help="하늘 마스크 폴더(파일명 매칭). 흰색=하늘")

    # 특징/neck 선택
    p.add_argument("--feat_type", type=str, default="vits8_1by4",
                   choices=["vits8_1by8", "vits8_1by4", "quarter"],
                   help="특징 해상도 선택: 1/8 또는 1/4")
    p.add_argument("--feat_layers", type=int, default=4,
                   help="QuarterNeck에 사용할 ViT 마지막 레이어 토큰 개수")
    p.add_argument("--neck_mid_ch", type=int, default=256,
                   help="QuarterNeck 내부 중간 채널")
    p.add_argument("--neck_out_ch", type=int, default=256,
                   help="최종 특징 채널(코사인 상관의 C)")

    # 모델/학습
    p.add_argument("--max_disp_px", type=int, default=64, help="입력 픽셀 단위 최대 시차")
    p.add_argument("--agg_ch",      type=int, default=32)
    p.add_argument("--agg_depth",   type=int, default=3)
    p.add_argument("--softarg_t",   type=float, default=0.9)
    p.add_argument("--norm",        type=str, default="gn", choices=["bn","gn"], help="3D conv 정규화")

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--optim",      type=str, default="adamw", choices=["adamw","adam","sgd"])
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--amp",        action="store_true", help="mixed precision")

    # ROI 축소 옵션
    p.add_argument("--roi_method", type=str, default="avg", choices=["avg","nearest"])
    p.add_argument("--roi_thr",    type=float, default=0.5)

    # 방향/게이트
    p.add_argument("--sim_thr",      type=float, default=0.75)
    p.add_argument("--sim_gamma",    type=float, default=0.0)
    p.add_argument("--sim_sample_k", type=int,   default=1024)
    p.add_argument("--use_dynamic_thr", action="store_true")
    p.add_argument("--dynamic_q",    type=float, default=0.7)
    p.add_argument("--lambda_v",     type=float, default=1.0)
    p.add_argument("--lambda_h",     type=float, default=1.0)
    p.add_argument("--huber_delta_h", type=float, default=0.25)

    # 픽셀-마진 (이웃 허용치)
    p.add_argument("--vert_margin_px", type=float, default=8.0,
                   help="세로 이웃 허용치(픽셀). 내부에서는 stride로 나눠 격자 단위로 사용")
    p.add_argument("--horiz_margin_px", type=float, default=0.0,
                   help="가로 이웃 허용치(픽셀). 내부에서는 stride로 나눠 격자 단위로 사용")

    # 수직 단조 제약 (NEW)
    p.add_argument("--lambda_v_mono", type=float, default=1.0,
                   help="세로 단조 제약 가중치 (위<=현재<=아래)")
    p.add_argument("--mono_margin", type=float, default=0.0,
                   help="단조 제약 여유(margin), 격자 단위")

    # 샤픈 가로 일관성
    p.add_argument("--w_hsharp",   type=float, default=0.0)
    p.add_argument("--tau_sharp",  type=float, default=0.2)

    # 분포-일치/엔트로피
    p.add_argument("--w_probcons", type=float, default=0.0)
    p.add_argument("--w_entropy",  type=float, default=0.00)

    # 앵커/재투영
    p.add_argument("--w_anchor",     type=float, default=1.0)
    p.add_argument("--anchor_tau",   type=float, default=0.5)
    p.add_argument("--anchor_margin",type=float, default=1.0)
    p.add_argument("--anchor_topk",  type=int,   default=2)
    p.add_argument("--w_reproj",     type=float, default=1.0)

    # 로깅/저장
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--save_dir", type=str, default=f"./checkpoints_{current_time}")

    # 손실 조합 스위치
    p.add_argument("--full_losses", action="store_true",
                   help="모든 보조 손실(hsharp/prob/entropy)까지 합산")

    # --- NEW: photometric / smoothness / low-light args ---
    p.add_argument("--w_photo", type=float, default=1.0,
                   help="photometric loss 가중치 (1/4 해상도)")
    p.add_argument("--smooth_weight", type=float, default=1e-3,
                   help="edge-aware smoothness 내부 weight (multi_scale 함수에 전달)")
    p.add_argument("--ll_enable", type=int, default=1,
                   help="저조도 보정 적용(1) / 미적용(0)")
    p.add_argument("--ll_gamma", type=float, default=1.8)
    p.add_argument("--ll_clahe_clip", type=float, default=2.0)
    p.add_argument("--ll_clahe_tile", type=int, default=8)

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
