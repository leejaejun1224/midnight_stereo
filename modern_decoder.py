# modern_decoder.py
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# 작은 유틸
# ---------------------------

def _get_norm3d(num_channels: int, kind: str = "ln") -> nn.Module:
    kind = (kind or "ln").lower()
    if kind == "bn":
        return nn.BatchNorm3d(num_channels)
    if kind.startswith("gn"):
        # 예: "gn8" → 8그룹, 기본은 8
        try:
            g = int(kind[2:])
        except Exception:
            g = 8
        g = max(1, min(g, num_channels))
        return nn.GroupNorm(g, num_channels)
    # "ln": 채널축 LayerNorm과 유사한 효과 (GroupNorm with 1 group)
    return nn.GroupNorm(1, num_channels)


class DropPath(nn.Module):
    """Stochastic depth. timm 없이 간단 버전."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = x.new_empty(shape).bernoulli_(keep)
        return x / keep * rand


# ---------------------------
# 블록 구성요소
# ---------------------------

class DSConv3d(nn.Module):
    """
    3D depthwise separable conv
      - (1,3,3) depthwise → 1x1x1 pointwise
      - (3,1,1) depthwise → 1x1x1 pointwise
    """
    def __init__(self, in_ch, out_ch, act=True, norm="ln"):
        super().__init__()
        self.dw_hw = nn.Conv3d(in_ch, in_ch, kernel_size=(1,3,3), padding=(0,1,1),
                               groups=in_ch, bias=False)
        self.pw_hw = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.dw_d  = nn.Conv3d(out_ch, out_ch, kernel_size=(3,1,1), padding=(1,0,0),
                               groups=out_ch, bias=False)
        self.pw_d  = nn.Conv3d(out_ch, out_ch, kernel_size=1, bias=False)
        self.bn    = _get_norm3d(out_ch, norm)
        self.act   = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        x = self.pw_hw(self.dw_hw(x))
        x = self.pw_d(self.dw_d(x))
        x = self.bn(x)
        x = self.act(x)
        return x


class AxialAttentionD(nn.Module):
    """
    시차축(D) 전용 Multi-Head Self-Attention.
    각 (h,w) 위치에서 길이 D 시퀀스에 대해 MHSA 수행.
    입력/출력: [B,C,D,H,W]
    """
    def __init__(self, channels: int, num_heads: int = 4, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert channels % num_heads == 0, "channels는 num_heads의 배수여야 합니다."
        self.h = num_heads
        self.dim = channels // num_heads
        self.scale = self.dim ** -0.5

        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1, bias=False)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.proj = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        # 간단한 사인/코사인 위치인코딩 (D축)
        self.register_buffer("_pe_cache", None, persistent=False)

    # modern_decoder.py 의 AxialAttentionD 클래스 내부

    def _sinusoidal_pe(self, D: int, device, dtype):
        """
        반환 shape: [1, 1, dim, D, 1, 1]
        - 1(배치), 1(헤드 브로드캐스트), dim(헤드당 채널), D(시차 길이), 1, 1
        """
        # 캐시 유효성: device/dtype/D 모두 일치할 때만 재사용
        if (self._pe_cache is not None and
            self._pe_cache.device == device and
            self._pe_cache.dtype  == dtype  and
            self._pe_cache.shape[3] == D):    # <-- D 차원은 index 3
            return self._pe_cache

        dim  = self.dim                   # per-head channel dim
        half = (dim + 1) // 2             # 홀수일 때 마지막 채널은 sin으로 채움

        # 표준 inverse-frequency (간단/안정형)
        # 주의: half==1인 극단 케이스도 안전하게 동작
        inv_freq = torch.exp(
            torch.arange(0, half, device=device, dtype=dtype) *
            (-math.log(10000.0) / max(1, half - 1))
        )                                           # [half]

        pos = torch.arange(D, device=device, dtype=dtype)    # [D]
        angles = pos[:, None] * inv_freq[None, :]            # [D, half]

        sin = angles.sin().transpose(0, 1).contiguous()      # [half, D]
        cos = angles.cos().transpose(0, 1).contiguous()      # [half, D]

        pe = torch.zeros(1, 1, dim, D, 1, 1, device=device, dtype=dtype)
        even = dim // 2
        if even > 0:
            pe[:, :, 0:2*even:2, :, 0, 0] = sin[:even]       # 짝수 채널에 sin
            pe[:, :, 1:2*even:2, :, 0, 0] = cos[:even]       # 홀수 채널에 cos
        if dim % 2 == 1:
            pe[:, :, -1, :, 0, 0] = sin[-1]                  # 남는 한 채널은 sin

        self._pe_cache = pe
        return pe


    def forward(self, x):
        B, C, D, H, W = x.shape
        qkv = self.qkv(x)  # [B,3C,D,H,W]
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # [B, h, dim, D, H, W]
        q = q.view(B, self.h, self.dim, D, H, W)
        k = k.view(B, self.h, self.dim, D, H, W)
        v = v.view(B, self.h, self.dim, D, H, W)

        # D축 위치 인코딩(간단) 추가
        pe = self._sinusoidal_pe(D, x.device, x.dtype)  # [1,1,dim,D,1,1]
        q = q + pe
        k = k + pe

        # 각 (H,W) 위치마다 D 토큰 self-attention
        # reshape: [B,h,HW,D,dim]
        HW = H * W
        q = q.permute(0,1,4,5,3,2).contiguous().view(B, self.h, HW, D, self.dim)
        k = k.permute(0,1,4,5,3,2).contiguous().view(B, self.h, HW, D, self.dim)
        v = v.permute(0,1,4,5,3,2).contiguous().view(B, self.h, HW, D, self.dim)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,h,HW,D,D]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)  # [B,h,HW,D,dim]

        # 복원: [B,C,D,H,W]
        out = out.view(B, self.h, H, W, D, self.dim).permute(0,1,5,4,2,3).contiguous()
        out = out.view(B, -1, D, H, W)
        out = self.proj_drop(self.proj(out))
        return out


class AxialBlock(nn.Module):
    """
    [Norm → AxialAttentionD → DropPath + Skip] → [Norm → DSConv3d(FFN) → DropPath + Skip]
    """
    def __init__(self, ch, num_heads=4, mlp_ratio=2.0, drop_path=0.0, norm="ln"):
        super().__init__()
        hidden = int(ch * mlp_ratio)
        self.n1 = _get_norm3d(ch, norm)
        self.attn = AxialAttentionD(ch, num_heads=num_heads)
        self.dp1 = DropPath(drop_path)

        self.n2 = _get_norm3d(ch, norm)
        self.ffn = nn.Sequential(
            nn.Conv3d(ch, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            DSConv3d(hidden, ch, act=True, norm=norm)
        )
        self.dp2 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.dp1(self.attn(self.n1(x)))
        x = x + self.dp2(self.ffn(self.n2(x)))
        return x


class Down3D_HW(nn.Module):
    """H/W만 다운: stride=(1,2,2)"""
    def __init__(self, in_ch, out_ch, norm="ln"):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=(1,2,2), padding=1, bias=False)
        self.bn   = _get_norm3d(out_ch, norm)
        self.act  = nn.GELU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Up3D_HW(nn.Module):
    """H/W만 업: ConvTranspose3d, stride=(1,2,2)"""
    def __init__(self, in_ch, out_ch, norm="ln"):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=3, stride=(1,2,2), padding=1, bias=False)
        self.bn   = _get_norm3d(out_ch, norm)
        self.act  = nn.GELU()
    def forward(self, x, out_hw=None):
        if out_hw is not None:
            B, C, D, _, _ = x.shape
            Ht, Wt = out_hw
            x = self.deconv(x, output_size=(B, self.deconv.out_channels, D, Ht, Wt))
        else:
            x = self.deconv(x)
        return self.act(self.bn(x))


class CrossScaleFuse(nn.Module):
    """업샘플된 x와 스킵 s를 게이트로 융합"""
    def __init__(self, ch, norm="ln"):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(ch*2, ch, kernel_size=1, bias=False),
            _get_norm3d(ch, norm),
            nn.GELU(),
            nn.Conv3d(ch, ch, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.mix  = nn.Conv3d(ch*2, ch, kernel_size=1, bias=False)

    def forward(self, x_up, skip):
        g = self.gate(torch.cat([x_up, skip], dim=1))
        x = g * x_up + (1 - g) * skip
        return self.mix(torch.cat([x, skip], dim=1))


# ---------------------------
# ModernStereoDecoder
# ---------------------------

class ModernStereoDecoder(nn.Module):
    """
    입력:  [B,1,D+1,H,W]   (시차 비용볼륨 로짓/스코어)
    출력:  [B,1,D+1,H,W]   (정제된 로짓)
    - H,W는 1/4 해상도 기준을 가정(모델 외부에서 보장)
    - D는 forward 시 결정되며, axial attention이 자동 적용
    """
    def __init__(self,
                 base_ch: int = 32,
                 depth: int = 3,
                 num_heads: int = 4,
                 mlp_ratio: float = 2.0,
                 blocks_per_stage=(2, 2, 2, 2),
                 drop_path: float = 0.1,
                 norm: str = "ln"):
        super().__init__()
        assert depth in (2, 3, 4), "depth는 2~4만 지원합니다."
        self.depth = depth
        ch = base_ch

        # stage별 블록 개수
        if isinstance(blocks_per_stage, int):
            b1 = b2 = b3 = b4 = int(blocks_per_stage)
        else:
            b = list(blocks_per_stage) + [2,2,2,2]
            b1, b2, b3, b4 = b[:4]

        # DropPath 비율을 선형 분배
        def dpr_seq(n, dp):
            return [dp * i / max(1, n-1) for i in range(n)]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(1, ch, kernel_size=3, padding=1, bias=False),
            _get_norm3d(ch, norm),
            nn.GELU()
        )

        # Encoder 1
        self.enc1 = nn.Sequential(*[
            AxialBlock(ch, num_heads, mlp_ratio, dpr_seq(b1, drop_path)[i], norm)
            for i in range(b1)
        ])
        self.down1 = Down3D_HW(ch, ch*2, norm)
        c2 = ch*2

        # Encoder 2
        self.enc2 = nn.Sequential(*[
            AxialBlock(c2, num_heads, mlp_ratio, dpr_seq(b2, drop_path)[i], norm)
            for i in range(b2)
        ])
        self.down2 = Down3D_HW(c2, c2*2, norm)
        c3 = c2*2

        # Encoder 3 (선택)
        if depth >= 3:
            self.enc3 = nn.Sequential(*[
                AxialBlock(c3, num_heads, mlp_ratio, dpr_seq(b3, drop_path)[i], norm)
                for i in range(b3)
            ])
            if depth >= 4:
                self.down3 = Down3D_HW(c3, c3*2, norm)
                c4 = c3*2
                self.bott = nn.Sequential(*[
                    AxialBlock(c4, num_heads, mlp_ratio, dpr_seq(b4, drop_path)[i], norm)
                    for i in range(b4)
                ])
                self.up3 = Up3D_HW(c4, c3, norm)
            else:
                # depth==3: 보틀넥만 enc3로 사용
                self.bott = nn.Sequential(*[
                    AxialBlock(c3, num_heads, mlp_ratio, dpr_seq(b3, drop_path)[i], norm)
                    for i in range(b3)
                ])
        else:
            # depth==2: enc3 생략
            self.enc3 = None
            self.bott = nn.Sequential(*[
                AxialBlock(c3, num_heads, mlp_ratio, dpr_seq(b3, drop_path)[i], norm)
                for i in range(b3)
            ])

        # Decoder path
        if depth >= 3:
            self.fuse3 = CrossScaleFuse(c3, norm)
        self.up2 = Up3D_HW(c3, c2, norm)
        self.fuse2 = CrossScaleFuse(c2, norm)
        self.up1 = Up3D_HW(c2, ch, norm)
        self.fuse1 = CrossScaleFuse(ch, norm)

        # 출력 헤드
        self.head = nn.Conv3d(ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # x: [B,1,D+1,H,W]
        B, C, D, H, W = x.shape
        assert C == 1, "입력 채널은 1이어야 합니다. (cost volume)"
        # Stem
        s1 = self.stem(x)     # [B,ch,D,H,W]

        # Encoder
        e1 = self.enc1(s1)
        d1 = self.down1(e1)   # [B,2ch,D,H/2,W/2]

        e2 = self.enc2(d1)
        d2 = self.down2(e2)   # [B,4ch,D,H/4,W/4]

        if self.depth >= 3:
            e3 = self.enc3(d2)
            if self.depth >= 4:
                d3  = self.down3(e3)
                b   = self.bott(d3)
                u3  = self.up3(b, out_hw=(e3.shape[-2], e3.shape[-1]))
                m3  = self.fuse3(u3, e3)
            else:
                m3  = self.bott(e3)
        else:
            m3 = self.bott(d2)

        # Decoder
        u2 = self.up2(m3, out_hw=(e2.shape[-2], e2.shape[-1]))
        m2 = self.fuse2(u2, e2)

        u1 = self.up1(m2, out_hw=(e1.shape[-2], e1.shape[-1]))
        m1 = self.fuse1(u1, e1)

        out = self.head(m1)   # [B,1,D,H,W]
        return out
