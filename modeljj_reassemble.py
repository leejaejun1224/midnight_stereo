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
from upsampler import FinalUpsample2x

# -----------------------------------------
# Utils
# -----------------------------------------
def spatial_upsample_volume_4d(vol4d: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """
    vol4d: [B, Cv, D, Hs, Ws]  (여기서 D는 disparity 축 길이)
    target_hw: (Ht, Wt)  (공간(H,W)만 업샘플; D는 그대로 유지)
    return: [B, Cv, D, Ht, Wt]
    """
    B, Cv, Dp, Hs, Ws = vol4d.shape
    # 각 disparity slice별로 2D bilinear
    v = vol4d.permute(0, 2, 1, 3, 4).contiguous().view(B*Dp, Cv, Hs, Ws)     # [B*D, Cv, Hs, Ws]
    v = F.interpolate(v, size=target_hw, mode='bilinear', align_corners=False)
    v = v.view(B, Dp, Cv, target_hw[0], target_hw[1]).permute(0, 2, 1, 3, 4).contiguous()
    return v

def assert_multiple(x, m, name="size"):
    if x % m != 0:
        raise ValueError(f"{name}={x} 는 {m}의 배수여야 합니다.")


def build_corr_volume_with_mask(FL: torch.Tensor, FR: torch.Tensor, D: int, mode: str = 'gwc', groups: int = 8):
    """
    Build cost volume between left/right feature maps at 1/patch stride.

    Args:
        FL, FR : [B,C,H,W]  (assumed channel-normalized)
        D      : max disparity in feature pixels (inclusive; typically max_disp/patch)
        mode   : 'sum' (legacy sum correlation -> [B,1,D+1,H,W])
                 'gwc' (group-wise correlation -> [B,G,D+1,H,W])
        groups : number of groups when mode == 'gwc'

    Returns:
        vol  : [B,Cv,D+1,H,W]  (Cv=1 for 'sum', Cv=groups for 'gwc')
        mask : [B,1,D+1,H,W]   (1.0 valid, 0.0 invalid)
    """
    B, C, H, W = FL.shape
    vols, masks = [], []
    if mode not in ['sum', 'gwc']:
        raise ValueError("mode must be 'sum' or 'gwc'")
    if mode == 'gwc':
        assert C % groups == 0, f"C={C} 는 groups={groups}로 나누어 떨어져야 합니다."
        Cg = C // groups
        FLg = FL.view(B, groups, Cg, H, W)

    for d in range(D + 1):
        if d == 0:
            FR_shift = FR
            valid = torch.ones((B, 1, H, W), device=FL.device, dtype=FL.dtype)
        else:
            FR_shift = F.pad(FR, (d, 0, 0, 0))[:, :, :, :W]
            valid = torch.ones((B, 1, H, W), device=FL.device, dtype=FL.dtype)
            valid[:, :, :, :d] = 0.0

        if mode == 'sum':
            corr = (FL * FR_shift).sum(dim=1, keepdim=True)  # [B,1,H,W]
        else:
            FRg = FR_shift.view(B, groups, Cg, H, W)
            corr = (FLg * FRg).mean(dim=2)  # [B,G,H,W]

        vols.append(corr)
        masks.append(valid)

    vol = torch.stack(vols, dim=2)   # [B,Cv,D+1,H,W]; Cv=1 or G
    mask = torch.stack(masks, dim=2) # [B,1, D+1,H,W]
    return vol, mask

# -----------------------------------------
# DINO v1 ViT-B/8 feature extractor (tokens->grid baseline)
# -----------------------------------------

class DINOvits8Features(nn.Module):
    def __init__(self, patch_size: int = 8):
        super().__init__()
        self.patch = patch_size
        # dino_vitb8 로드 (torch.hub)
        self.backbone = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        assert_multiple(H, self.patch, "height")
        assert_multiple(W, self.patch, "width")
        tokens = self.backbone.get_intermediate_layers(x, n=1)[0]  # [B, 1+P, C]
        patch_tokens = tokens[:, 1:, :]                            # [B, P, C]
        C = patch_tokens.shape[-1]
        H8, W8 = H // self.patch, W // self.patch
        feat = patch_tokens.transpose(1, 2).contiguous().view(B, C, H8, W8)  # [B,C,H',W']
        feat = F.normalize(feat, dim=1, eps=1e-6)
        return feat

# -----------------------------------------
# Reassemble (extracted from DPT head pre-decoder): tokens -> grid -> 1x1 -> norm+GELU -> resample
# -----------------------------------------

class ReadoutMerge(nn.Module):
    """Merge ViT [CLS] token into patch tokens.
    mode: 'ignore' | 'add' | 'project'  (default 'project')
    input : x (B, N+1, D)
    output: y (B, N,   D)
    """
    def __init__(self, embed_dim: int, mode: str = 'project'):
        super().__init__()
        assert mode in ['ignore', 'add', 'project']
        self.mode = mode
        if mode == 'project':
            self.proj = nn.Sequential(
                nn.LayerNorm(2*embed_dim),
                nn.Linear(2*embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls = x[:, :1, :]          # (B,1,D)
        patch = x[:, 1:, :]        # (B,N,D)
        if self.mode == 'ignore':
            return patch
        elif self.mode == 'add':
            return patch + cls
        else:
            B, N, D = patch.shape
            cls_expand = cls.expand(B, N, D)
            z = torch.cat([patch, cls_expand], dim=-1)  # (B,N,2D)
            return self.proj(z)                         # (B,N,D)


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for BCHW feature maps."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var  = x.var(dim=1, keepdim=True, unbiased=False)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return xhat * self.weight + self.bias


class UpDown2dBilinear(nn.Module):
    """Resample BCHW from stride s_in to stride s_out using bilinear + 2 conv refine."""
    def __init__(self, ch: int, s_in: int, s_out: int):
        super().__init__()
        self.s_in, self.s_out = s_in, s_out
        self.refine = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        Ht = int(round(H * self.s_in / self.s_out))
        Wt = int(round(W * self.s_in / self.s_out))
        x = F.interpolate(x, size=(Ht, Wt), mode='bilinear', align_corners=False)
        return self.refine(x)


class ReassembleStage(nn.Module):
    """DPT-style Reassemble for a single scale.
    Steps: readout merge -> tokens->grid -> 1x1 -> norm+GELU -> resample(to out_stride)
    tokens: (B, N+1, D), image_size=(H,W), patch_size=p
    return: (B, C_hat, H/out_stride, W/out_stride)
    """
    def __init__(self,
                 embed_dim: int,
                 out_channels: int = 192,
                 readout: str = 'project',
                 out_stride: int = 8,
                 patch_size: int = 8,
                 norm: str = 'bn'):
        super().__init__()
        self.out_stride = out_stride
        self.patch_size = patch_size
        self.readout = ReadoutMerge(embed_dim, mode=readout)
        self.proj1x1 = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'ln2d':
            self.norm = LayerNorm2d(out_channels)
        else:
            raise ValueError("norm must be 'bn' or 'ln2d'")
        self.resample = UpDown2dBilinear(out_channels, s_in=self.patch_size, s_out=self.out_stride)

    def _tokens_to_grid(self, x: torch.Tensor, image_size: Tuple[int,int]) -> torch.Tensor:
        B, Np, D = x.shape
        H, W = image_size
        Hp, Wp = H // self.patch_size, W // self.patch_size
        assert Hp * Wp == Np, f"Token count {Np} != (H/p)*(W/p)={Hp*Wp}. Pad 이미지 크기를 p={self.patch_size} 배수로 맞추세요."
        x = x.transpose(1, 2).contiguous().view(B, D, Hp, Wp)
        return x

    def forward(self, tokens: torch.Tensor, image_size: Tuple[int,int]) -> torch.Tensor:
        x = self.readout(tokens)                       # (B, N, D)
        x = self._tokens_to_grid(x, image_size)        # (B, D, H/p, W/p)
        x = self.proj1x1(x)                            # (B, C_hat, H/p, W/p)
        x = F.gelu(self.norm(x))
        x = self.resample(x)                           # (B, C_hat, H/out, W/out)
        return x

# -----------------------------------------
# Modernized 3D Aggregation (GWC + Separable 3D UNet + ASPP3D + StripPooling)
# -----------------------------------------

class ChannelSE3D(nn.Module):
    """Squeeze-and-Excitation for 3D tensors (B,C,D,H,W)."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        B, C, D, H, W = x.shape
        s = x.mean(dim=(2,3,4))                # [B,C]
        w = self.mlp(s).view(B, C, 1, 1, 1)    # [B,C,1,1,1]
        return x * w

class SepRes3DBlock(nn.Module):
    """Depthwise-separable residual 3D block with GroupNorm."""
    def __init__(self, ch: int, norm='gn', groups_gn: int = 8):
        super().__init__()
        self.dw = nn.Conv3d(ch, ch, kernel_size=3, padding=1, groups=ch)
        self.pw = nn.Conv3d(ch, ch, kernel_size=1)
        if norm == 'gn':
            self.bn1 = nn.GroupNorm(num_groups=min(groups_gn, ch), num_channels=ch)
            self.bn2 = nn.GroupNorm(num_groups=min(groups_gn, ch), num_channels=ch)
        else:
            self.bn1 = nn.BatchNorm3d(ch)
            self.bn2 = nn.BatchNorm3d(ch)
        self.act = nn.GELU()
        self.se = ChannelSE3D(ch)

    def forward(self, x):
        res = x
        x = self.dw(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.bn2(x)
        x = self.se(x)
        x = self.act(x + res)
        return x

class ASPP3D(nn.Module):
    """Atrous Spatial Pyramid over disparity + spatial dims."""
    def __init__(self, ch: int, rates: List[Tuple[int,int,int]] = ((1,1,1),(1,2,2),(2,2,2),(3,4,4))):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv3d(ch, ch, kernel_size=3, padding=(r[0], r[1], r[2]), dilation=(r[0], r[1], r[2]))
            for r in rates
        ])
        self.merge = nn.Conv3d(ch*len(self.branches), ch, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        x = self.merge(x)
        return self.act(x)

class StripPooling3D(nn.Module):
    """Capture long-range context along H and W using large-kernel separable convs on each disparity slice."""
    def __init__(self, ch: int, kh: int = 1, kw: int = 9):
        super().__init__()
        # apply conv with kernels (1,1,kw) and (1,kh,1)
        self.conv_w = nn.Conv3d(ch, ch, kernel_size=(1,1,kw), padding=(0,0,kw//2), groups=ch)
        self.conv_h = nn.Conv3d(ch, ch, kernel_size=(1,kh,1), padding=(0,kh//2,0), groups=ch)
        self.pw = nn.Conv3d(ch, ch, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        xw = self.conv_w(x)
        xh = self.conv_h(x)
        x = self.pw(xw + xh)
        return self.act(x)

class Down3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm='gn', groups_gn: int = 8):
        super().__init__()
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=(1,2,2), padding=1)
        if norm == 'gn':
            self.bn = nn.GroupNorm(num_groups=min(groups_gn, out_ch), num_channels=out_ch)
        else:
            self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.bn(self.down(x)))
    
class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm='gn', groups_gn: int = 8):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_ch, out_ch, kernel_size=3, stride=(1,2,2), padding=1
        )
        if norm == 'gn':
            self.bn = nn.GroupNorm(num_groups=min(groups_gn, out_ch), num_channels=out_ch)
        else:
            self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.GELU()

    def forward(self, x, out_hw=None):
        if out_hw is not None:
            B, _, D, _, _ = x.shape
            Ht, Wt = int(out_hw[0]), int(out_hw[1])
            x = self.up(x, output_size=(B, self.up.out_channels, D, Ht, Wt))
        else:
            x = self.up(x)
        x = self.bn(x)
        return self.act(x)


class GWCAggregator3D(nn.Module):
    """
    Input : [B,Cv,D+1,H,W]
    Output: [B,1,D+1,H,W]
    """
    def __init__(self, in_ch: int, base_ch: int = 32, depth: int = 3, norm='gn'):
        super().__init__()
        ch = base_ch
        self.stem = nn.Conv3d(in_ch, ch, kernel_size=3, padding=1)

        # Encoder
        self.e1 = SepRes3DBlock(ch, norm=norm)
        self.d1 = Down3D(ch, ch*2, norm=norm)
        self.e2 = SepRes3DBlock(ch*2, norm=norm)
        self.d2 = Down3D(ch*2, ch*4, norm=norm)
        self.e3 = SepRes3DBlock(ch*4, norm=norm)

        self.depth = depth
        if depth >= 3:
            self.d3 = Down3D(ch*4, ch*8, norm=norm)
            self.b_aspp  = ASPP3D(ch*8)
            self.b_strip = StripPooling3D(ch*8, kh=9, kw=9)
            self.b_res   = SepRes3DBlock(ch*8, norm=norm)
            self.u3      = Up3D(ch*8, ch*4, norm=norm)   # in: ch*8 -> out: ch*4

        # Decoder
        self.dec3 = SepRes3DBlock(ch*8 if depth >= 3 else ch*4, norm=norm)

        # 🔧 핵심 수정: u2는 dec3 출력 채널(= ch*8 또는 ch*4)을 입력으로 받도록
        self.u2   = Up3D(ch*8 if depth >= 3 else ch*4, ch*2, norm=norm)  # in: ch*8 → out: ch*2

        self.dec2 = SepRes3DBlock(ch*4, norm=norm)

        # 🔧 핵심 수정: u1은 dec2 출력 채널(= ch*4)을 입력으로 받도록
        self.u1   = Up3D(ch*4, ch, norm=norm)  # in: ch*4 → out: ch

        # 🔧 핵심 수정: dec1에서 채널을 ch*2 → ch로 축소(그 후 SE/Out)
        if norm == 'gn':
            _norm = lambda c: nn.GroupNorm(num_groups=min(8, c), num_channels=c)
        else:
            _norm = lambda c: nn.BatchNorm3d(c)

        self.dec1 = nn.Sequential(
            SepRes3DBlock(ch*2, norm=norm),
            nn.Conv3d(ch*2, ch, kernel_size=1, bias=False),
            _norm(ch),
            nn.GELU()
        )

        self.se1  = ChannelSE3D(ch)
        self.out  = nn.Conv3d(ch, 1, kernel_size=1)

    def forward(self, x):
        x  = self.stem(x)
        e1 = self.e1(x); d1 = self.d1(e1)
        e2 = self.e2(d1); d2 = self.d2(e2)
        e3 = self.e3(d2)

        if self.depth >= 3:
            d3 = self.d3(e3)
            b  = self.b_aspp(d3)
            b  = self.b_strip(b)
            b  = self.b_res(b)
            u3 = self.u3(b, out_hw=(e3.shape[-2], e3.shape[-1]))
            c3 = torch.cat([u3, e3], dim=1)  # ch*4 + ch*4 = ch*8
        else:
            c3 = e3

        dec3 = self.dec3(c3)  # ch*8 (또는 ch*4)

        u2   = self.u2(dec3, out_hw=(e2.shape[-2], e2.shape[-1]))  # out: ch*2
        # 안전가드(디버깅시 유용)
        # assert u2.shape[-2:] == e2.shape[-2:]
        c2   = torch.cat([u2, e2], dim=1)                          # ch*2 + ch*2 = ch*4

        dec2 = self.dec2(c2)                                       # ch*4
        u1   = self.u1(dec2, out_hw=(e1.shape[-2], e1.shape[-1]))  # out: ch
        # assert u1.shape[-2:] == e1.shape[-2:]

        c1   = torch.cat([u1, e1], dim=1)                          # ch + ch = ch*2
        dec1 = self.dec1(c1)                                       # → ch
        dec1 = self.se1(dec1)                                      # ch
        out  = self.out(dec1)                                      # 1
        return out


# -----------------------------------------
# Soft + ArgMax (둘 다 계산)
# -----------------------------------------

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

def convex_upsample_2d_scalar(disp_lo: torch.Tensor, mask: torch.Tensor, scale: int) -> torch.Tensor:
    """
    disp_lo: [B,1,H',W']  (scalar field)
    mask:    [B, 9*(scale*scale), H', W']
    return:  [B,1,H'*scale, W'*scale]
    """
    B, C, H, W = disp_lo.shape
    assert C == 1
    # mask reshape & softmax over 9 neighbors
    mask = mask.view(B, 1, 9, scale, scale, H, W)
    mask = torch.softmax(mask, dim=2)  # convex weights over 9

    # unfold 3x3 neighborhood
    unf = F.unfold(disp_lo, kernel_size=3, padding=1)  # [B, 9, H*W]
    unf = unf.view(B, 1, 9, H, W).unsqueeze(3).unsqueeze(4)  # [B,1,9,1,1,H,W]

    up = torch.sum(mask * unf, dim=2)  # [B,1,scale,scale,H,W]
    up = up.permute(0,1,4,2,5,3).contiguous().view(B, 1, H*scale, W*scale)
    return up

# -----------------------------------------
# Stereo Model (Reassemble after DINOvits8Features)
# -----------------------------------------

class StereoModel(nn.Module):
    def __init__(self,
                 max_disp_px: int = 64,
                 patch_size: int = 8,
                 agg_base_ch: int = 32,
                 agg_depth: int = 3,
                 softarg_t: float = 0.9,
                 norm='gn',
                 reassem_ch: int = 192,
                 readout: str = 'project',
                 reassem_norm: str = 'bn',
                 volume_mode: str = 'gwc',   # 'gwc' | 'sum'
                 gw_groups: int = 8):
        super().__init__()
        self.patch = patch_size
        assert_multiple(max_disp_px, patch_size, "max_disp_px")
        if self.patch % 2 != 0:
            raise ValueError(f"patch_size={self.patch} 는 짝수여야 합니다. (1/8 → 1/2 업샘플에 필요)")
        self.D = max_disp_px // patch_size

        # 1) Load DINO v1 ViT-B/8 (frozen, eval)
        self.feat_net = DINOvits8Features(patch_size)  # we reuse its backbone
        self.backbone = self.feat_net.backbone
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # DINO embed dim (ViT-B/8 default 768)
        embed_dim = int(getattr(self.backbone, "embed_dim", 768))

        # 2) Reassemble -> 1/8 features (Ĉ channels)
        self.reassemble = ReassembleStage(
            embed_dim=embed_dim,
            out_channels=reassem_ch,
            readout=readout,
            out_stride=self.patch,  # 1/8
            patch_size=self.patch,
            norm=reassem_norm
        )
        self.reassem_ch = reassem_ch

        # 3) Volume mode
        self.volume_mode = volume_mode
        self.gw_groups = gw_groups
        in_ch = gw_groups if volume_mode == 'gwc' else 1

        # 4) 3D aggregator and soft-argmax
        self.agg = GWCAggregator3D(in_ch=in_ch, base_ch=agg_base_ch, depth=agg_depth, norm=norm)
        self.post = SoftAndArgMax(D=self.D, temperature=softarg_t)

        # 5) Convex upsample head (1/8 → 1/2 == ×(patch/2))
        self.up_scale = self.patch // 2  # e.g., patch=8 → scale=4
        self.upmask_head = nn.Sequential(
            nn.Conv2d(self.reassem_ch, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, (9)*(self.up_scale**2), kernel_size=1)
        )
        
        # self.final_up = FinalUpsample2x(
        #     dino_ch=reassem_ch,        # DINO 채널(예: 768)
        #     guide_ch=64,
        #     fuse_ch=96,
        #     refine_ch=64,
        #     softmax_t=softarg_t,    # volume softmax 온도 동일 사용
        #     res_limit=1.5,
        #     use_edge_head=False     # 필요시 True로 켜고 보조손실 추가 가능
        # )

    @torch.no_grad()
    def _get_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return DINO tokens (B, N+1, D) from ViT-B/8.
        """
        tokens = self.backbone.get_intermediate_layers(x, n=1)[0]  # [B, 1+P, D]
        return tokens

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        """
        Returns:
          prob: [B,1,D+1,H/8,W/8]
          disp_soft: [B,1,H/8,W/8] (단위: patch)
          aux: dict(
             FL/FR: reassembled 1/8 features,
             left_tokens/right_tokens: DINO tokens (L2-normalized along D) for cosine prior,
             raw_volume/refined/refined_masked/mask/disp_wta,
             disp_half: [B,1,H/2,W/2] (단위: patch),
             disp_half_px: [B,1,H/2,W/2] (단위: half 픽셀)
          )
        """
        B, _, H, W = left.shape
        assert_multiple(H, self.patch, "height")
        assert_multiple(W, self.patch, "width")

        # 1) DINO tokens (for prior and reassemble)
        with torch.no_grad():  # backbone frozen
            left_tokens  = self._get_tokens(left)   # (B, N+1, D)
            right_tokens = self._get_tokens(right)  # (B, N+1, D)

        # L2-normalize tokens for cosine-prior (if you need grid, reshape outside)
        left_tokens_n  = F.normalize(left_tokens,  p=2, dim=-1)
        right_tokens_n = F.normalize(right_tokens, p=2, dim=-1)

        # 2) Reassemble -> 1/8 features (Ĉ channels)
        FL = self.reassemble(left_tokens,  image_size=(H, W))  # (B, Ĉ, H/8, W/8)
        FR = self.reassemble(right_tokens, image_size=(H, W))  # (B, Ĉ, H/8, W/8)

        # 3) Normalize channel-wise so correlation behaves like cosine
        FL = F.normalize(FL, dim=1, eps=1e-6)
        FR = F.normalize(FR, dim=1, eps=1e-6)

        # 4) Correlation volume at 1/8
        vol, mask = build_corr_volume_with_mask(FL, FR, self.D, mode=self.volume_mode, groups=self.gw_groups)
        vol_in = vol * mask  # broadcast mask [B,1,D+1,H,W] → [B,Cv,D+1,H,W]

        # 5) 3D aggregation
        refined = self.agg(vol_in)          # [B,1,D+1,H',W']
        refined_masked = refined + (1.0 - mask) * (-1e4)

        # 6) Prob / soft-arg / WTA at 1/8
        prob, disp_soft, disp_wta = self.post(refined_masked)

        # raw_volume for visualization/anchor (keep legacy shape [B,1,...] using channel-mean if GWC)
        raw_single = vol.mean(dim=1, keepdim=True) if vol.shape[1] > 1 else vol
        raw_for_anchor = (raw_single + (1.0 - mask) * (-1e4)).detach()

        # 7) Guided convex upsample: 1/8 → 1/2
        upmask = self.upmask_head(FL)  # [B,9*s*s,H',W']
        disp_half = convex_upsample_2d_scalar(disp_soft, upmask, self.up_scale)  # patch 단위
        disp_half_px = disp_half * float(self.patch / 2.0)  # half-res px 단위

        # ----- (NEW) 최종 2× 업샘플 (½ → 1) -----
        # 4D volume (logits) 를 H/2로 공간 업샘플 (disparity 축은 유지)
        vol4d_half = spatial_upsample_volume_4d(refined_masked, target_hw=(H // 2, W // 2))  # [B,1,D+1,H/2,W/2]

        # DINO(feature)도 H/2로 보간해 upsampler에 전달(선택)
        dino_half = F.interpolate(FL, size=(H // 2, W // 2), mode='bilinear', align_corners=False)  # [B,feat_ch,H/2,W/2]

        # 최종 업샘플 실행
        # disp_full, aux_up = self.final_up(disp_half, left, vol4d_half, dino_half)   # [B,1,H,W]
        # disp_full_px = disp_full * float(self.patch)  # 'patch' 단위를 원본 px로 변환

        # aux 업데이트
        aux = {
            "FL": FL, "FR": FR,
            "raw_volume": raw_for_anchor,
            "refined_volume": refined,
            "mask": mask,
            "refined_masked": refined_masked,
            "disp_wta": disp_wta,
            "disp_half": disp_half,
            "disp_half_px": disp_half_px,
            # (NEW)
            # "disp_full": disp_full,
            # "disp_full_px": disp_full_px,
        }
        # aux.update(aux_up)

        return prob, disp_soft, aux

        # return prob, disp_soft, {
        #     "FL": FL, "FR": FR,
        #     "left_tokens": left_tokens_n, "right_tokens": right_tokens_n,
        #     "raw_volume": raw_for_anchor,
        #     "refined_volume": refined,
        #     "mask": mask,
        #     "refined_masked": refined_masked,
        #     "disp_wta": disp_wta,
        #     "disp_half": disp_half,
        #     "disp_half_px": disp_half_px,
        # }
