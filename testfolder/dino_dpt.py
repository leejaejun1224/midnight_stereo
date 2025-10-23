import math
from typing import List, Tuple, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Utils: 2D sine-cos positional embedding (for ViT)
# =========================================================
def build_2d_sincos_position_embedding(h: int, w: int, dim: int, cls_token: bool = True) -> torch.Tensor:
    """
    Return [1, (h*w + 1), dim] if cls_token else [1, h*w, dim]
    """
    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        # pos: [M]
        omega = torch.arange(embed_dim // 2, dtype=torch.float32)
        omega = 1. / (10000 ** (omega / (embed_dim / 2)))
        out = torch.einsum('m,d->md', pos, omega)  # [M, embed_dim//2]
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        return torch.cat([emb_sin, emb_cos], dim=1)  # [M, embed_dim]

    grid_y = torch.arange(h, dtype=torch.float32)
    grid_x = torch.arange(w, dtype=torch.float32)
    grid = torch.meshgrid(grid_y, grid_x, indexing='ij')
    grid = torch.stack(grid, dim=0).reshape(2, 1, h, w)
    grid_y = grid[0].reshape(-1)
    grid_x = grid[1].reshape(-1)

    assert (dim % 4) == 0, "Position embedding dim must be divisible by 4."
    dim_half = dim // 2
    pos_embed_y = get_1d_sincos_pos_embed_from_grid(dim_half, grid_y)  # [hw, dim/2]
    pos_embed_x = get_1d_sincos_pos_embed_from_grid(dim_half, grid_x)  # [hw, dim/2]
    pos_embed = torch.cat([pos_embed_y, pos_embed_x], dim=1)           # [hw, dim]
    if cls_token:
        cls = torch.zeros(1, dim, dtype=torch.float32)                 # zeros for CLS
        pos_embed = torch.cat([cls, pos_embed], dim=0)                  # [hw+1, dim]
    return pos_embed.unsqueeze(0)                                       # [1, hw(+1), dim]


# =========================================================
# Minimal ViT backbone (DINOv1-Base/8 style, 12 layers)
# =========================================================
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=proj_drop)

    def forward(self, x):
        # x: [B, N, D]
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = h + x
        h = x
        x = self.norm2(x)
        x = h + self.mlp(x)
        return x

class DINOv1Base8Backbone(nn.Module):
    """
    ViT-B/8 style backbone (12 blocks, dim=768, heads=12).
    Outputs token sequences from selected layers (e.g., [2,5,8,11]).
    """
    def __init__(self, img_channels=3, patch_size=8, embed_dim=768, depth=12, num_heads=12,
                 select_layers=(2, 5, 8, 11)):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.select_layers = tuple(select_layers)

        self.patch_embed = nn.Conv2d(img_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads) for _ in range(depth)
        ])
        self.pos_cache = {}  # (h,w)->pos_embed device-wise

    def _pos_embed(self, B, h, w, device):
        key = (h, w, device)
        if key not in self.pos_cache:
            pe = build_2d_sincos_position_embedding(h, w, self.embed_dim, cls_token=True).to(device)
            self.pos_cache[key] = pe
        return self.pos_cache[key].expand(B, -1, -1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        x: [B,3,H,W], H,W must be divisible by patch_size
        return: list of tokens from selected layers, each [B, N+1, D]
        """
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "H,W must be multiples of patch_size=8"
        h, w = H // self.patch_size, W // self.patch_size

        feats = self.patch_embed(x)                  # [B, D, h, w]
        tokens = feats.flatten(2).transpose(1, 2)    # [B, N, D], N=h*w
        pos = self._pos_embed(B, h, w, x.device)     # [B, N+1, D]
        cls = self.cls_token.expand(B, -1, -1)       # [B,1,D]
        x = torch.cat([cls, tokens], dim=1) + pos    # [B, N+1, D]

        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.select_layers:
                outs.append(x)
        # expected len=4
        return outs

# =========================================================
# DPT building blocks (Reassemble, Fusion, etc.)
# =========================================================
class ProjectReadout(nn.Module):
    def __init__(self, dim: int, mode: Literal['project', 'add', 'ignore'] = 'project'):
        super().__init__()
        self.mode = mode
        if mode == 'project':
            self.proj = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
        elif mode == 'add':
            self.proj = nn.Linear(dim, dim)
        elif mode == 'ignore':
            self.proj = nn.Identity()
        else:
            raise ValueError(f"Unsupported readout mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        if self.mode == 'project':
            cls_exp = cls_token.expand(-1, patch_tokens.size(1), -1)
            out = self.proj(torch.cat([patch_tokens, cls_exp], dim=-1))
        elif self.mode == 'add':
            out = patch_tokens + self.proj(cls_token).expand_as(patch_tokens)
        else:
            out = patch_tokens
        return out  # [B, N, D]

class ResidualConvUnit(nn.Module):
    def __init__(self, channels: int, gn_groups: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=min(gn_groups, channels), num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=min(gn_groups, channels), num_channels=channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x); x = self.gn1(x); x = self.act(x)
        x = self.conv2(x); x = self.gn2(x)
        return self.act(x + residual)
    
class FeatureFusionBlock(nn.Module):
    def __init__(self, channels: int, gn_groups: int = 32):
        super().__init__()
        self.rcu1 = ResidualConvUnit(channels, gn_groups)
        self.rcu2 = ResidualConvUnit(channels, gn_groups)

    def forward(self,
                x: Optional[torch.Tensor],            # up path
                lateral: Optional[torch.Tensor] = None  # higher-res lateral
                ) -> torch.Tensor:
        # 핵심: 업샘플을 '×2'가 아니라 'lateral의 실제 크기'에 맞추기
        if x is not None:
            if lateral is not None:
                # lateral과 같은 (H,W)로 정확히 맞춤
                x = F.interpolate(x, size=lateral.shape[-2:], mode='bilinear', align_corners=False)
            else:
                # lateral이 없으면 기존처럼 ×2 (예비용)
                x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)

        if lateral is not None:
            x = x + lateral if x is not None else lateral

        x = self.rcu1(x)
        x = self.rcu2(x)
        return x

class Reassemble(nn.Module):
    """
    ViT tokens -> 2D feature at target stride (1/32,1/16,1/8,1/4).
    """
    def __init__(self, in_dim: int, out_dim: int = 256,
                 readout: Literal['project', 'add', 'ignore'] = 'project',
                 target_stride: int = 32, patch_size: int = 8):
        super().__init__()
        assert target_stride in (4, 8, 16, 32)
        self.target_stride = target_stride
        self.patch_size = patch_size
        self.readout = ProjectReadout(in_dim, readout)
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(num_groups=min(32, out_dim), num_channels=out_dim)
        self.act = nn.GELU()

    @staticmethod
    def _resize(x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        Ht, Wt = size_hw
        H, W = x.shape[-2:]
        if Ht > H or Wt > W:
            return F.interpolate(x, size=(Ht, Wt), mode='bilinear', align_corners=False)
        elif Ht < H or Wt < W:
            return F.interpolate(x, size=(Ht, Wt), mode='area')
        else:
            return x

    def forward(self, tokens: torch.Tensor, img_hw: Tuple[int, int]) -> torch.Tensor:
        B, Lp1, D = tokens.shape
        H, W = img_hw
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        h = H // self.patch_size
        w = W // self.patch_size
        x = self.readout(tokens)                       # [B, N, D]
        x = x.transpose(1, 2).reshape(B, D, h, w)      # [B, D, h, w]
        x = self.proj(x); x = self.gn(x); x = self.act(x)
        target_h = math.ceil(H / self.target_stride)
        target_w = math.ceil(W / self.target_stride)
        x = self._resize(x, (target_h, target_w))
        return x  # [B, out_dim, H/stride, W/stride]


# =========================================================
# Stereo: 1D horizontal correlation cost volume + 2D encoder
# =========================================================
def cost_volume_1d_corr(left: torch.Tensor, right: torch.Tensor, max_disp: int) -> torch.Tensor:
    """
    left, right: [B, C, H, W] (features at a given scale)
    Return cost volume: [B, D_disp, H, W], where D_disp = max_disp + 1
    Correlation is cosine (channel-wise dot after L2 norm). Invalid columns are zeroed.
    """
    B, C, H, W = left.shape
    assert right.shape == left.shape

    Ln = F.normalize(left, dim=1)
    Rn = F.normalize(right, dim=1)

    vols = []
    device = left.device
    for d in range(max_disp + 1):
        if d == 0:
            shifted = Rn
            corr = (Ln * shifted).sum(1, keepdim=True)  # [B,1,H,W]
        else:
            # shift right features to the RIGHT by d (equiv. align R[x-d] with L[x])
            shifted = F.pad(Rn, pad=(d, 0, 0, 0), mode='constant', value=0.0)[:, :, :, :W]
            corr = (Ln * shifted).sum(1, keepdim=True)
            # zero-out invalid columns [0:d)
            if W >= d:
                corr[..., :d] = 0.0
        vols.append(corr)
    vol = torch.cat(vols, dim=1)  # [B, max_disp+1, H, W]
    return vol


class StereoLateralEncoder(nn.Module):
    """
    Encodes [cost volume (B,Dd,H,W) concatenated with left feature (B,Cf,H,W)]
    into a common feature width (feat_dim) with small conv stack.
    """
    def __init__(self, left_feat_ch: int, max_disp_s: int, out_ch: int):
        super().__init__()
        in_ch = (max_disp_s + 1) + left_feat_ch
        mid = max(out_ch, in_ch // 2)
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, mid), num_channels=mid),
            nn.GELU(),
            nn.Conv2d(mid, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
            nn.GELU(),
        )

    def forward(self, cost_vol: torch.Tensor, left_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cost_vol, left_feat], dim=1)
        return self.enc(x)  # [B, out_ch, H, W]


# =========================================================
# Stereo DPT Head: cost volume -> lateral -> DPT fusion -> 1/4 output
# =========================================================
class StereoDPTHead(nn.Module):
    """
    Build cost volumes at 1/16, 1/8, 1/4; encode to lateral features; fuse top-down; predict at 1/4.

    Args:
        embed_dim:   ViT token dim (DINOv1 Base = 768)
        feat_dim:    internal feature width for DPT (default 256)
        out_channels: output channels (e.g., disparity=1)
        readout:     'project' | 'add' | 'ignore' (readout injection of CLS)
        patch_size:  ViT patch size (8 for DINOv1 Base/8)
        max_disp:    full-resolution maximum disparity in pixels (e.g., 192)
    """
    def __init__(self,
                 embed_dim: int = 768,
                 feat_dim: int = 256,
                 out_channels: int = 1,
                 readout: Literal['project', 'add', 'ignore'] = 'project',
                 patch_size: int = 8,
                 max_disp: int = 192):
        super().__init__()
        self.max_disp = max_disp
        self.feat_dim = feat_dim

        # Reassemble for left/right at four target strides
        self.reass32_L = Reassemble(embed_dim, feat_dim, readout, target_stride=32, patch_size=patch_size)
        self.reass16_L = Reassemble(embed_dim, feat_dim, readout, target_stride=16, patch_size=patch_size)
        self.reass8_L  = Reassemble(embed_dim, feat_dim, readout, target_stride=8,  patch_size=patch_size)
        self.reass4_L  = Reassemble(embed_dim, feat_dim, readout, target_stride=4,  patch_size=patch_size)

        self.reass32_R = Reassemble(embed_dim, feat_dim, readout, target_stride=32, patch_size=patch_size)
        self.reass16_R = Reassemble(embed_dim, feat_dim, readout, target_stride=16, patch_size=patch_size)
        self.reass8_R  = Reassemble(embed_dim, feat_dim, readout, target_stride=8,  patch_size=patch_size)
        self.reass4_R  = Reassemble(embed_dim, feat_dim, readout, target_stride=4,  patch_size=patch_size)

        # Max disparity per scale (rounded up)
        self.d16 = math.ceil(max_disp / 16)
        self.d8  = math.ceil(max_disp / 8)
        self.d4  = math.ceil(max_disp / 4)

        # Cost-volume -> lateral encoders (concat with left feature)
        self.enc16 = StereoLateralEncoder(left_feat_ch=feat_dim, max_disp_s=self.d16, out_ch=feat_dim)
        self.enc8  = StereoLateralEncoder(left_feat_ch=feat_dim, max_disp_s=self.d8,  out_ch=feat_dim)
        self.enc4  = StereoLateralEncoder(left_feat_ch=feat_dim, max_disp_s=self.d4,  out_ch=feat_dim)

        # DPT fusion path
        self.fuse_32_16 = FeatureFusionBlock(feat_dim)
        self.fuse_16_8  = FeatureFusionBlock(feat_dim)
        self.fuse_8_4   = FeatureFusionBlock(feat_dim)

        # Task head at 1/4
        self.head = nn.Sequential(
            ResidualConvUnit(feat_dim),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, feat_dim), num_channels=feat_dim),
            nn.GELU(),
            nn.Conv2d(feat_dim, out_channels, kernel_size=1, bias=True),
        )

    def forward(self,
                tokens_L: List[torch.Tensor],  # [t4, t8, t16, t32] or any 4 layers shallow->deep
                tokens_R: List[torch.Tensor],
                img_hw: Tuple[int, int]) -> torch.Tensor:
        assert len(tokens_L) == 4 and len(tokens_R) == 4, "Need 4 token layers for L/R"
        t4L, t8L, t16L, t32L = tokens_L
        t4R, t8R, t16R, t32R = tokens_R

        # 1) Reassemble to 2D features
        f32L = self.reass32_L(t32L, img_hw)  # [B,C,H/32,W/32]
        f16L = self.reass16_L(t16L, img_hw)  # [B,C,H/16,W/16]
        f8L  = self.reass8_L (t8L,  img_hw)  # [B,C,H/8,W/8]
        f4L  = self.reass4_L (t4L,  img_hw)  # [B,C,H/4,W/4]

        f32R = self.reass32_R(t32R, img_hw)
        f16R = self.reass16_R(t16R, img_hw)
        f8R  = self.reass8_R (t8R,  img_hw)
        f4R  = self.reass4_R (t4R,  img_hw)

        # 2) Cost volumes at 1/16, 1/8, 1/4 (concat with left feats -> lateral)
        cv16 = cost_volume_1d_corr(f16L, f16R, self.d16)  # [B, d16+1, H/16, W/16]
        lat16 = self.enc16(cv16, f16L)                    # [B, C, H/16, W/16]

        cv8  = cost_volume_1d_corr(f8L, f8R, self.d8)     # [B, d8+1,  H/8,  W/8 ]
        lat8 = self.enc8(cv8, f8L)                        # [B, C, H/8,  W/8 ]

        cv4  = cost_volume_1d_corr(f4L, f4R, self.d4)     # [B, d4+1,  H/4,  W/4 ]
        lat4 = self.enc4(cv4, f4L)                        # [B, C, H/4,  W/4 ]

        # 3) DPT top-down fusion: start from 1/32 (left only), then lateral at each step
        y = self.fuse_32_16(f32L, lateral=lat16)          # -> H/16
        y = self.fuse_16_8 (y,    lateral=lat8)           # -> H/8
        y = self.fuse_8_4  (y,    lateral=lat4)           # -> H/4

        # 4) Head at 1/4
        out = self.head(y)  # [B, out_channels, H/4, W/4]
        return out


# =========================================================
# Full Stereo model wrapper: backbone + head
# =========================================================
class StereoDPTModel(nn.Module):
    def __init__(self,
                 patch_size: int = 8,
                 embed_dim: int = 768,
                 feat_dim: int = 256,
                 out_channels: int = 1,
                 readout: str = 'project',
                 max_disp: int = 192,
                 select_layers=(2, 5, 8, 11)):
        super().__init__()
        self.backbone = DINOv1Base8Backbone(
            img_channels=3, patch_size=patch_size, embed_dim=embed_dim,
            depth=12, num_heads=12, select_layers=select_layers
        )
        self.head = StereoDPTHead(
            embed_dim=embed_dim, feat_dim=feat_dim, out_channels=out_channels,
            readout=readout, patch_size=patch_size, max_disp=max_disp
        )

    def forward(self, img_left: torch.Tensor, img_right: torch.Tensor) -> torch.Tensor:
        """
        img_left/right: [B,3,H,W] (H,W multiples of patch_size=8)
        return: [B, out_channels, H/4, W/4]
        """
        B, C, H, W = img_left.shape
        tokens_L = self.backbone(img_left)
        tokens_R = self.backbone(img_right)
        out = self.head(tokens_L, tokens_R, img_hw=(H, W))
        return out


# =========================================================
# main: dummy inputs & shape print
# =========================================================
def main():
    # --------- YOU CAN EDIT THESE ----------
    B, C, H, W = 1, 3, 384, 1224   # H,W must be divisible by 8
    feat_dim = 256
    out_channels = 1              # disparity/depth -> 1
    max_disp_fullres = 64        # typical stereo setting
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # --------------------------------------

    torch.manual_seed(0)
    model = StereoDPTModel(
        patch_size=8, embed_dim=768, feat_dim=feat_dim, out_channels=out_channels,
        readout='project', max_disp=max_disp_fullres
    ).to(device)

    left  = torch.randn(B, C, H, W, device=device)
    right = torch.randn(B, C, H, W, device=device)

    with torch.no_grad():
        out = model(left, right)

    print(f"Input:  left/right = [{B}, {C}, {H}, {W}]")
    print(f"Output: disparity/depth map = {tuple(out.shape)}  # [B, {out_channels}, H/4, W/4]")

if __name__ == "__main__":
    main()
