import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from agg.aggregator import *



# -----------------------------
# 공통 상수
# -----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# -----------------------------
# DINO ViT-S/8 로드 유틸
# -----------------------------
def load_dino_vits8(device: torch.device, eval_mode: bool = True):
    """
    facebookresearch/dino 의 dino_vits8을 우선 로드.
    (인터넷/허브 문제시 timm fallback 시도)
    """
    model = None
    err_msgs = []
    # 1) torch.hub (공식)
    try:
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
    except Exception as e:
        err_msgs.append(f"torch.hub dino_vits8 load failed: {e}")

    # 2) timm fallback (환경에 따라 미설치일 수 있음)
    if model is None:
        try:
            import timm
            # timm의 DINO 사전학습 모델 이름은 환경마다 다를 수 있음
            # 가장 가까운 후보를 시도
            for name in ["vit_small_patch8_224.dino", "vit_small_patch8_224", "vit_small_patch8_224.augreg"]:
                try:
                    model = timm.create_model(name, pretrained=True)
                    break
                except Exception:
                    continue
            if model is None:
                raise RuntimeError("timm fallback couldn't create a ViT-S/8 model.")
        except Exception as e:
            err_msgs.append(f"timm fallback load failed: {e}")

    if model is None:
        raise RuntimeError("Failed to load DINO ViT-S/8.\n" + "\n".join(err_msgs))

    if eval_mode:
        model.eval()
    model.to(device)
    return model


# -----------------------------
# ViT 토큰 추출 유틸
# -----------------------------
@torch.no_grad()
def _vit_tokens_hw_last_layer(vit: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    마지막 레이어의 패치 토큰을 [B,H8,W8,C]로 반환 (CLS 제거).
    vit: DINO ViT-S/8 (facebookresearch/dino) 기준으로 get_intermediate_layers 지원.
    x: [B,3,Hpad,Wpad], Hpad/Wpad는 8의 배수.
    """
    if not hasattr(vit, "get_intermediate_layers"):
        raise RuntimeError("ViT model must provide get_intermediate_layers (DINO style).")

    out = vit.get_intermediate_layers(x, n=1)[0]  # [B, N(+cls), C]
    B, N, C = out.shape
    Hpad, Wpad = x.shape[-2:]
    h8 = Hpad // 8
    w8 = Wpad // 8

    # CLS 제거
    if N == h8 * w8 + 1:
        tokens = out[:, 1:, :]
    elif N == h8 * w8:
        tokens = out
    else:
        raise RuntimeError(f"Unexpected tokens: N={N}, expected {h8*w8} (±1 for CLS).")

    tokens_hw = tokens.reshape(B, h8, w8, C).contiguous()  # [B,H8,W8,C]
    return tokens_hw


def _uniform_pad_for_shift(H: int, W: int, dx: int, dy: int, grid: int = 8) -> Tuple[int, int, int, int, int, int]:
    """
    모든 시프트에서 동일한 최종 캔버스 크기를 쓰도록 패딩량을 결정.
    반환: (left, right, top, bottom, Hpad_target, Wpad_target)
    """
    # 4px 시프트를 고려해, (H+4), (W+4)를 grid 배수로 올림
    Hpad_target = math.ceil((H + 4) / grid) * grid
    Wpad_target = math.ceil((W + 4) / grid) * grid

    left   = dx
    top    = dy
    right  = Wpad_target - (W + dx)
    bottom = Hpad_target - (H + dy)

    assert right >= 0 and bottom >= 0
    return left, right, top, bottom, Hpad_target, Wpad_target


@torch.no_grad()
def _shift_tokens_once(
    vit: nn.Module,
    x: torch.Tensor,
    dx: int,
    dy: int,
    pad_mode: str = "replicate",
) -> torch.Tensor:
    """
    단일 시프트 (dx, dy)에서 ViT 패치 토큰을 [B,H/8,W/8,C]로 추출.
    - 네 시프트 모두 같은 타깃 캔버스 크기를 사용(컨텍스트 차이 감소).
    """
    B, _, H, W = x.shape
    left, right, top, bottom, Hpad_t, Wpad_t = _uniform_pad_for_shift(H, W, dx, dy, grid=8)

    xpad = F.pad(x, (left, right, top, bottom), mode=pad_mode)  # [B,3,Hpad_t,Wpad_t]
    tokens_hw = _vit_tokens_hw_last_layer(vit, xpad)            # [B, Hpad_t/8, Wpad_t/8, C]

    # 원본 유효 영역 크기만 유지
    H8 = H // 8
    W8 = W // 8
    tokens_hw = tokens_hw[:, :H8, :W8, :].contiguous()          # [B,H/8,W/8,C]
    return tokens_hw


@torch.no_grad()
def build_interleaved_quarter_features(
    vit: nn.Module,
    x: torch.Tensor,
    pad_mode: str = "replicate",
    amp: bool = True,
) -> torch.Tensor:
    """
    인터리빙으로 1/4 해상도 피처를 생성.
    반환: [B, H/4, W/4, C], 채널 마지막 & L2 정규화됨.
    """
    assert x.dim() == 4 and x.size(2) % 8 == 0 and x.size(3) % 8 == 0, \
        "Input H,W must be multiples of 8."

    B, _, H, W = x.shape
    H4, W4 = H // 4, W // 4

    with autocast(enabled=amp):
        f00 = _shift_tokens_once(vit, x, dx=0, dy=0, pad_mode=pad_mode)  # [B,H/8,W/8,C]
        f40 = _shift_tokens_once(vit, x, dx=4, dy=0, pad_mode=pad_mode)
        f04 = _shift_tokens_once(vit, x, dx=0, dy=4, pad_mode=pad_mode)
        f44 = _shift_tokens_once(vit, x, dx=4, dy=4, pad_mode=pad_mode)

    C = f00.shape[-1]
    Fq = torch.empty((B, H4, W4, C), device=x.device, dtype=f00.dtype)

    # 인터리빙(체스판 배치)
    Fq[:, 1::2, 1::2, :] = f00
    Fq[:, 1::2, 0::2, :] = f40
    Fq[:, 0::2, 1::2, :] = f04
    Fq[:, 0::2, 0::2, :] = f44

    # L2 정규화 → 내적 == cosine
    Fq = F.normalize(Fq, dim=-1)
    return Fq  # [B,H/4,W/4,C]


# ========== 최소주의 3x3 스택 유틸 ==========
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Conv3x3GNAct(nn.Module):
    """3x3 conv -> (GN or BN) -> GELU"""
    def __init__(self, in_ch, out_ch, stride=1, norm='gn', gn_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(gn_groups, out_ch)
        else:
            self.norm = nn.Identity()
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResBlock3x3(nn.Module):
    """(3x3, gn, gelu) x2 + 잔차"""
    def __init__(self, ch, norm='gn', gn_groups=8):
        super().__init__()
        self.conv1 = Conv3x3GNAct(ch, ch, stride=1, norm=norm, gn_groups=gn_groups)
        # 두 번째는 활성화 직전에 합치기 위해 act 없이 norm만
        self.conv2_conv = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        if norm == 'bn':
            self.conv2_norm = nn.BatchNorm2d(ch)
        elif norm == 'gn':
            self.conv2_norm = nn.GroupNorm(gn_groups, ch)
        else:
            self.conv2_norm = nn.Identity()
        self.act = nn.GELU()
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2_norm(self.conv2_conv(y))
        return self.act(x + y)

# ========== Conv 1/4 분기 (3x3-only) ==========
class ConvQuarter(nn.Module):
    """
    순수 3x3 스택:
      - 1/2: Conv3x3 s=2 -> (ResBlock3x3)*N_half
      - 1/4: Conv3x3 s=2 -> (ResBlock3x3)*N_quarter
    out: [B, cc, H/4, W/4]
    """
    def __init__(
        self,
        in_ch: int = 3,
        cc: int = 256,          # 출력 채널 (기본 256)
        norm: str = 'gn',
        gn_groups: int = 8,
        c_half: int = 128,      # H/2 단계 채널
        n_blocks_half: int = 2, # 1/2 스케일 블록 수
        n_blocks_quarter: int = 3, # 1/4 스케일 블록 수
    ):
        super().__init__()
        # H/2
        self.s2 = Conv3x3GNAct(in_ch, c_half, stride=2, norm=norm, gn_groups=gn_groups)
        self.stage_half = nn.Sequential(*[
            ResBlock3x3(c_half, norm=norm, gn_groups=gn_groups)
            for _ in range(n_blocks_half)
        ])
        # H/4
        self.s4 = Conv3x3GNAct(c_half, cc, stride=2, norm=norm, gn_groups=gn_groups)
        self.stage_quarter = nn.Sequential(*[
            ResBlock3x3(cc, norm=norm, gn_groups=gn_groups)
            for _ in range(n_blocks_quarter)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.s2(x)            # [B,c_half,H/2,W/2]
        x = self.stage_half(x)
        x = self.s4(x)            # [B,cc,H/4,W/4]
        x = self.stage_quarter(x) # [B,cc,H/4,W/4]
        return x

# ========== Fusion 1/4 (3x3-only) ==========
class Fusion1_4(nn.Module):
    """
    1/4에서 Conv분기(cc)와 ViT분기(cv)를 concat 후,
    3x3-only 네트로 Cf로 축소·융합.
    - 첫 forward 때 v8_up 채널을 읽어 in_conv를 동적 생성(런타임 적응)
    """
    def __init__(
        self,
        cc: int = 256,
        cv: int = 0,           # 무시(런타임에서 자동 적응)
        cf: int = 320,         # 최종 융합 출력 채널 (디코더 fused_in_ch와 일치)
        norm: str = 'gn',
        gn_groups: int = 8,
        n_blocks: int = 2,     # 융합 후 ResBlock 개수
    ):
        super().__init__()
        self.cc = cc
        self.cf = cf
        self.norm = norm
        self.gn_groups = gn_groups
        self.n_blocks = n_blocks

        self.in_conv: Optional[nn.Module] = None  # 첫 forward에서 생성
        self.blocks: Optional[nn.Sequential] = None

    def _build_once(self, in_ch_total: int, device: torch.device):
        # 입력(cat) -> cf 로 바로 축소 (3x3)
        self.in_conv = Conv3x3GNAct(
            in_ch_total, self.cf, stride=1, norm=self.norm, gn_groups=self.gn_groups
        ).to(device)
        # 융합 후 3x3 잔차 블록들
        self.blocks = nn.Sequential(*[
            ResBlock3x3(self.cf, norm=self.norm, gn_groups=self.gn_groups)
            for _ in range(self.n_blocks)
        ]).to(device)

    def forward(self, c4: torch.Tensor, v8_up: torch.Tensor) -> torch.Tensor:
        if self.in_conv is None:
            in_ch_total = c4.shape[1] + v8_up.shape[1]
            self._build_once(in_ch_total, c4.device)
        x = torch.cat([c4, v8_up], dim=1)
        x = self.in_conv(x)
        x = self.blocks(x)
        return x


# -----------------------------
# 유틸: ViT 1/8 토큰 추출 (학습/동결 제어)
# -----------------------------
def extract_vit_1_8_tokens(
    vit: nn.Module,
    x: torch.Tensor,
    enable_grad: bool = False,
    amp: bool = True,
) -> torch.Tensor:
    """
    ViT 마지막 레이어 패치 토큰 [B,H/8,W/8,C] 반환.
    (enable_grad=False면 no_grad/EVAL, True면 grad 허용)
    """
    ctx = torch.enable_grad() if enable_grad else torch.no_grad()
    with ctx:
        with autocast(enabled=amp):
            tokens_hw = _vit_tokens_hw_last_layer(vit, x)  # [B,H/8,W/8,C]
    return tokens_hw


# -----------------------------
# 패딩/언패딩 유틸
# -----------------------------
def pad_right_bottom_to_multiple_of_8(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    입력을 우/하로만 패딩하여 H,W를 8의 배수로 맞춤.
    반환: (x_pad, (pad_b, pad_r))
    """
    _, _, H, W = x.shape
    pad_r = (-W) % 8
    pad_b = (-H) % 8
    if pad_r or pad_b:
        x = F.pad(x, (0, pad_r, 0, pad_b), mode="replicate")
    return x, (pad_b, pad_r)


def unpad_1_4(x: torch.Tensor, pad_b: int, pad_r: int) -> torch.Tensor:
    """ [B,C,H/4,W/4] 텐서의 우/하 패딩 제거 """
    if pad_b == 0 and pad_r == 0:
        return x
    Hb4 = pad_b // 4
    Wr4 = pad_r // 4
    H, W = x.shape[-2:]
    return x[..., : H - Hb4, : W - Wr4].contiguous()


def unpad_1_8(x: torch.Tensor, pad_b: int, pad_r: int) -> torch.Tensor:
    """ [B,C,H/8,W/8] 텐서의 우/하 패딩 제거 """
    if pad_b == 0 and pad_r == 0:
        return x
    Hb8 = pad_b // 8
    Wr8 = pad_r // 8
    H, W = x.shape[-2:]
    return x[..., : H - Hb8, : W - Wr8].contiguous()


def unpad_1_4_lastdim(x: torch.Tensor, pad_b: int, pad_r: int) -> torch.Tensor:
    """ [B,H/4,W/4,C] 텐서의 우/하 패딩 제거 """
    if pad_b == 0 and pad_r == 0:
        return x
    Hb4 = pad_b // 4
    Wr4 = pad_r // 4
    H, W = x.shape[1:3]
    return x[:, : H - Hb4, : W - Wr4, :].contiguous()


# -----------------------------
# StereoModel (백본 교체/인터리빙 포함)
# -----------------------------
class StereoModel(nn.Module):
    """
    제안된 백본:
      - CosSim 전용: DINO ViT-S/8 4시프트 인터리빙으로 1/4 피처 (L2 normalized, [B,H/4,W/4,C])
      - 다운스트림: ViT 1/8 토큰 업샘플(×2) + Conv 1/4 피처 concat 후 가벼운 융합 → [B,Cf,H/4,W/4]

    forward(imgL, imgR) -> dict:
      {
        'left': {
          'fused_1_4':        [B,Cf,H/4,W/4],
          'cossim_feat_1_4':  [B,H/4,W/4,C],  # L2 normalized (cos 전용)
          'vit_1_8':          [B,Cv,H/8,W/8],
          'conv_1_4':         [B,Cc,H/4,W/4],
        },
        'right': { ... 동일 ... },
        'meta': {
          'valid_hw': (H, W),   # 언패드된 원본 해상도
          'pad':      (pad_b, pad_r)
        }
      }
    """
    def __init__(
        self,
        device: Optional[torch.device] = None,
        freeze_vit: bool = True,
        amp: bool = True,
        cc: int = 192,
        cv: int = 384+384,
        cf: int = 256,
        autopad_to_8: bool = True,
        pad_mode_for_interleave: str = "replicate",
        vit_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = amp
        self.freeze_vit = freeze_vit
        self.autopad_to_8 = autopad_to_8
        self.pad_mode_for_interleave = pad_mode_for_interleave

        # ViT (DINO ViT-S/8)
        self.vit = vit_model if vit_model is not None else load_dino_vits8(self.device, eval_mode=freeze_vit)
        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False

        # Conv 1/4 & Fusion
        self.conv14 = ConvQuarter(in_ch=3, cc=256)
        self.fuse14 = Fusion1_4(cc=256, cf=320) 

        self.to(self.device)

    # ---- 단일 이미지에 대한 백본 ----
    def _backbone_single(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        img: [B,3,H,W] (RGB, ImageNet 정규화 가정)
        반환: dict (왼/오 공통 구조)
        """
        assert img.dim() == 4 and img.size(1) == 3
        img = img.to(self.device, non_blocking=True)

        # 1) 필요시 우/하 패딩으로 8의 배수 정렬
        pad_b = pad_r = 0
        if self.autopad_to_8:
            img, (pad_b, pad_r) = pad_right_bottom_to_multiple_of_8(img)
        B, _, Hpad, Wpad = img.shape
        H, W = Hpad - pad_b, Wpad - pad_r

        # 2) CosSim 전용 1/4 피처 (인터리빙, L2 normalized)
        #    - 비학습(no_grad), AMP, 4시프트 순차 실행
        with torch.no_grad():
            Fq = build_interleaved_quarter_features(
                self.vit, img, pad_mode=self.pad_mode_for_interleave, amp=self.amp
            )  # [B,H/4,W/4,C]
        # 언패드 반영
        Fq = unpad_1_4_lastdim(Fq, pad_b, pad_r)  # [B,h4,w4,C]

        # 3) ViT 1/8 토큰 (학습/동결 옵션)
        tokens_hw = extract_vit_1_8_tokens(
            self.vit, img, enable_grad=not self.freeze_vit, amp=self.amp
        )  # [B,H/8,W/8,Cv]
        V8 = tokens_hw.permute(0, 3, 1, 2).contiguous()  # [B,Cv,H/8,W/8]
        V8 = unpad_1_8(V8, pad_b, pad_r)                 # 언패드

        # 4) Conv 1/4
        C4 = self.conv14(img)                            # [B,Cc,H/4,W/4]
        C4 = unpad_1_4(C4, pad_b, pad_r)

        # 5) 융합 @ 1/4
        V8_up = F.interpolate(V8, scale_factor=2, mode="bilinear", align_corners=False)  # [B,Cv,H/4,W/4]
        F4 = self.fuse14(C4, V8_up)                                                          # [B,Cf,H/4,W/4]

        return {
            "fused_1_4":       F4,   # 다운스트림용
            "cossim_feat_1_4": Fq,   # L2 normalized, cos 전용 (채널 마지막)
            "vit_1_8":         V8,   # 필요시 사용
            "conv_1_4":        C4,   # 필요시 사용
        }

    # ---- 스테레오 입력 (좌/우) ----
    def forward(self, imgL: torch.Tensor, imgR: torch.Tensor) -> Dict[str, Dict]:
        """
        imgL, imgR: [B,3,H,W], ImageNet 정규화된 텐서
        """
        # 좌/우 동일 파이프라인
        outL = self._backbone_single(imgL)
        outR = self._backbone_single(imgR)

        H4, W4 = outL["fused_1_4"].shape[-2:]
        H8, W8 = outL["vit_1_8"].shape[-2:]
        H  = H4 * 4
        W  = W4 * 4

        return {
            "left":  outL,
            "right": outR,
            "meta": {
                "valid_hw": (H, W),  # 언패드 후 원래 해상도
                "fused_shape":  (H4, W4),
                "vit_1_8_shape": (H8, W8),
            }
        }


# -----------------------------
# (선택) 전처리 헬퍼
# -----------------------------
class ImgPreprocess(nn.Module):
    """
    PIL 대신 텐서 입력만 쓸 경우를 대비한 간단 전처리:
      - [0,1] 범위 텐서 → ImageNet 정규화
    """
    def __init__(self):
        super().__init__()
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std",  std,  persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std
    
    
def pad_to_multiple(x: torch.Tensor, mult: int = 16, mode: str = "replicate"):
    """
    x: [B,3,H,W]
    오른쪽/아래로만 mult의 배수에 맞게 패딩
    return: x_pad, (pad_b, pad_r)
    """
    B, C, H, W = x.shape
    pad_r = (-W) % mult
    pad_b = (-H) % mult
    if pad_r or pad_b:
        x = F.pad(x, (0, pad_r, 0, pad_b), mode=mode)
    return x, (pad_b, pad_r)

def unpad(x: torch.Tensor, pad: tuple):
    """우/하 패딩 제거 (임의 채널수/스케일에도 사용 가능: 마지막 두 축 기준)"""
    pad_b, pad_r = pad
    if pad_b == 0 and pad_r == 0:
        return x
    H, W = x.shape[-2:]
    return x[..., :H - pad_b, :W - pad_r].contiguous()

# if __name__=="__main__":
    
#     left  = torch.randn(1, 3, 384, 1224, device="cuda")  # 주어진 크기
#     right = torch.randn(1, 3, 384, 1224, device="cuda")

#     # 1) 16의 배수로 패딩
#     left_p,  pad = pad_to_multiple(left,  mult=16, mode="replicate")  # W: 1224 -> 1232
#     right_p, _   = pad_to_multiple(right, mult=16, mode="replicate")
#     # 2) 백본/디코더 실행 (StereoModel은 autopad_to_8=False 권장)
#     print(left_p.shape)
#     print(right_p.shape)
#     stereo = StereoModel(autopad_to_8=False).cuda().eval()
#     backbone_out = stereo(left_p, right_p)
#     decoder = SOTAStereoDecoder(
#         max_disp_px=192, fused_in_ch=256, red_ch=48, base3d=32,
#         use_motif=True, two_stage=False
#     ).cuda().eval()
#     pred = decoder(backbone_out)

#     # 3) 언패드 (원본 해상도 복원)
#     disp_full = unpad(pred["disp_full"], pad)     # [B,1,384,1224]
#     disp_q     = unpad(pred["disp_1_4"], (pad[0]//4, pad[1]//4))  # [B,1,96,306]
#     print(disp_full.shape)