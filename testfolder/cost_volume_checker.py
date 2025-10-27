#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cost_volume_checker.py
----------------------
양안(left/right) 이미지로부터 1/4 해상도 특징을 추출하고,
다양한 결합 모드(dino_only, late_weighted, concat, sum, cnn_only)로
0..max disparity 범위의 dot-product correlation cost volume을 계산한 뒤
각 픽셀에서 유사도가 최대인 disparity(WTA)를 시각화합니다.

필요 패키지:
  - torch, torchvision
  - numpy
  - pillow (PIL)
  - matplotlib (색상 매핑용)

예시 사용법:
  python cost_volume_checker.py \
    --left path/to/left.png \
    --right path/to/right.png \
    --out ./cv_outputs \
    --modes dino_only,late_weighted,concat,sum \
    --max-disp 64 \
    --patch 8 \
    --dino-weight 0.9 \
    --sum-alpha 0.5 \
    --device auto \
    --save-volume

주의:
  - DINO 가중치는 torch.hub를 통해 github에서 받아옵니다.
    네트워크/권한 이슈로 로드가 실패할 수 있습니다. 그 경우 'cnn_only' 모드를 사용하세요.
"""
from __future__ import annotations
import os
import sys
import math
import argparse
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Utils
# ---------------------------

def assert_multiple(x: int, m: int, name: str = "size"):
    if x % m != 0:
        raise ValueError(f"{name}={x} 는 {m}의 배수여야 합니다.")

@torch.no_grad()
def build_corr_volume_with_mask(FL: torch.Tensor, FR: torch.Tensor, D: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    dot-product correlation cost volume.
    FL, FR: [B,C,H,W]  (여기서는 H=W=1/4 해상도)
    return:
      vol:  [B,1,D+1,H,W]
      mask: [B,1,D+1,H,W] (시프트 유효영역)
    """
    B, C, H, W = FL.shape
    vols, masks = [], []
    for d in range(D + 1):
        if d == 0:
            FR_shift = FR
            valid = torch.ones((B, 1, H, W), device=FL.device, dtype=FL.dtype)
        else:
            # 오른쪽 이미지를 왼쪽으로 d 셀 만큼 시프트(좌측 d 픽셀은 invalid)
            FR_shift = F.pad(FR, (d, 0, 0, 0))[:, :, :, :W]
            valid = torch.ones((B, 1, H, W), device=FL.device, dtype=FL.dtype)
            valid[:, :, :, :d] = 0.0
        corr = (FL * FR_shift).sum(dim=1, keepdim=True)  # [B,1,H,W]
        vols.append(corr.squeeze(1))
        masks.append(valid.squeeze(1))
    vol  = torch.stack(vols,  dim=1).unsqueeze(1)   # [B,1,D+1,H,W]
    mask = torch.stack(masks, dim=1).unsqueeze(1)   # [B,1,D+1,H,W]
    return vol, mask


# =========================================================
# 1/4 특징 추출기 (Frozen) — DINO(1/8→1/4) + MobileNetV2(1/4)
# =========================================================

class MobileNetV2_S4(nn.Module):
    """
    stride 4 CNN 특징 (stage 0~3), 채널 24, 위치별 채널 L2 정규화.
    입력은 0..1 또는 uint8; 내부에서 ImageNet 정규화.
    """
    def __init__(self, pretrained: bool = True, out_norm: bool = True, center: bool = True):
        super().__init__()
        from torchvision import models
        try:
            from torchvision.models import MobileNet_V2_Weights
            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            m = models.mobilenet_v2(weights=weights)
            mean = (0.485, 0.456, 0.406) if weights is None else weights.meta["mean"]
            std  = (0.229, 0.224, 0.225) if weights is None else weights.meta["std"]
        except Exception:
            m = models.mobilenet_v2(pretrained=pretrained)
            mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)

        self.encoder = nn.Sequential(*list(m.features[:4]))  # 0,1,2,3  => stride=4
        self.register_buffer("mean", torch.tensor(mean).view(1,3,1,1), persistent=False)
        self.register_buffer("std",  torch.tensor(std).view(1,3,1,1),  persistent=False)
        self.out_norm = out_norm
        self.center = center
        self.out_channels = 24

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W], float(0..1) or uint8
        if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            x = x.float().div_(255.0)
        B, C, H, W = x.shape
        # 오른쪽/아래 reflect pad로 4의 배수 보장
        need_h = (4 - (H % 4)) % 4; need_w = (4 - (W % 4)) % 4
        if need_h or need_w:
            x = F.pad(x, (0, need_w, 0, need_h), mode="reflect")
        # ImageNet 정규화
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        y = self.encoder(x)                      # [B,24,ceil(H/4),ceil(W/4)]
        y = y[:, :, :H//4, :W//4].contiguous()  # [B,24,H/4,W/4]
        if self.center:
            mu  = y.mean(dim=(2,3), keepdim=True)
            std = y.std(dim=(2,3), keepdim=True) + 1e-6
            y = (y - mu) / std
        if self.out_norm:
            y = F.normalize(y, dim=1, eps=1e-6)
        return y


@torch.no_grad()
def _dino_tokens_to_grid(backbone, x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    facebookresearch/dino hub 모델의 get_intermediate_layers → [B,C,H/patch,W/patch]
    (CLS 제거 후 그리드로 reshape)
    """
    tokens = backbone.get_intermediate_layers(x, n=1)[0]  # [B,1+P,C]
    patch_tokens = tokens[:, 1:, :]                       # (CLS 제거)
    B, P, C = patch_tokens.shape
    H, W = x.shape[-2], x.shape[-1]
    h, w = H // patch_size, W // patch_size
    assert h*w == P, f"Token {P} != h*w {h*w}. 입력 H,W는 patch_size({patch_size})의 배수여야 함."
    grid = patch_tokens.transpose(1, 2).contiguous().view(B, C, h, w)  # [B,C,H/8,W/8]
    return grid


class QuarterFeatureExtractor(nn.Module):
    """
    1/4 해상도 특징 추출기 (Frozen)
      - DINO ViT-B/8: [B,Cd,H/8,W/8] → NN 2× → [B,Cd,H/4,W/4] → L2 norm
      - MobileNetV2_S4: [B,24,H/4,W/4] → (옵션 중심화) → L2 norm
    입력은 0..1 또는 uint8.
    """
    def __init__(self, patch_size: int = 8, cnn_center: bool = True, try_load_dino: bool = True):
        super().__init__()
        self.patch = patch_size
        self.dino = None
        self.dino_dim = 0
        if try_load_dino:
            try:
                self.dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
                self.dino.eval()
                for p in self.dino.parameters():
                    p.requires_grad = False
                self.dino_dim = getattr(self.dino, "embed_dim", 768)
            except Exception as e:
                print(f"[WARN] DINO 로드 실패: {e}\n       'cnn_only' 모드만 동작합니다.")
                self.dino = None
                self.dino_dim = 0

        self.cnn = MobileNetV2_S4(pretrained=True, out_norm=True, center=cnn_center)
        self.cnn_dim  = self.cnn.out_channels

        # ImageNet 정규화 (DINO 입력용)
        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1), persistent=False)
        self.register_buffer("std",  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1),  persistent=False)

        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def channels(self) -> Dict[str, int]:
        return {"dino": self.dino_dim, "cnn": self.cnn_dim}

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B,3,H,W] (float 0..1 또는 uint8)
        return dict: {'dino': [B,Cd,H/4,W/4] or None, 'cnn': [B,24,H/4,W/4]}
        """
        B, _, H, W = x.shape
        assert_multiple(H, self.patch, "height")
        assert_multiple(W, self.patch, "width")

        out = {}

        # DINO 입력 정규화
        if self.dino is not None:
            x_dino = x
            if x_dino.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                x_dino = x_dino.float().div_(255.0)
            x_dino = (x_dino - self.mean.to(x.device)) / self.std.to(x.device)
            Fd8 = _dino_tokens_to_grid(self.dino, x_dino, self.patch)      # [B,Cd,H/8,W/8]
            Fd4 = F.interpolate(Fd8, scale_factor=2, mode='nearest')       # [B,Cd,H/4,W/4]
            Fd4 = F.normalize(Fd4, dim=1, eps=1e-6)
            out["dino"] = Fd4
        else:
            out["dino"] = None

        # CNN 1/4
        Fc4 = self.cnn(x)                                              # [B,24,H/4,W/4]
        out["cnn"] = Fc4

        return out


# ---------------------------
# I/O + 시각화 유틸
# ---------------------------

def load_image_as_tensor(path: str, device: torch.device) -> torch.Tensor:
    """이미지를 [1,3,H,W] float32(0..1) 텐서로 로드"""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0   # [H,W,3], 0..1
    ten = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
    return ten.to(device)

def pad_to_multiple_of(x: torch.Tensor, m: int) -> Tuple[torch.Tensor, Tuple[int,int]]:
    """오른쪽/아래 reflect pad로 H,W를 m의 배수로 만듦. returns (x_pad, (pad_h, pad_w))"""
    _, _, H, W = x.shape
    need_h = (m - (H % m)) % m
    need_w = (m - (W % m)) % m
    if need_h or need_w:
        x = F.pad(x, (0, need_w, 0, need_h), mode="reflect")
    return x, (need_h, need_w)

def crop_to_original(x: torch.Tensor, orig_hw: Tuple[int,int]) -> torch.Tensor:
    """오른쪽/아래 패딩을 제거하여 원 해상도로 자르기"""
    _, _, H, W = x.shape
    H0, W0 = orig_hw
    return x[:, :, :H0, :W0]

def to_colormap_image(x: np.ndarray, vmin: float, vmax: float, cmap: str = "turbo") -> np.ndarray:
    """스칼라 맵을 0..255 RGB 컬러맵 이미지로 변환"""
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    x_clip = np.clip((x - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    mapped = cm.get_cmap(cmap)(x_clip)[..., :3]  # [H,W,3], 0..1
    img = (mapped * 255.0).astype(np.uint8)
    return img

def save_image(array_uint8_hw3: np.ndarray, path: str):
    Image.fromarray(array_uint8_hw3).save(path)


# ---------------------------
# 메인 로직
# ---------------------------

@torch.no_grad()
def run(args):
    # 디바이스 선택
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.out, exist_ok=True)

    # 입력 로드 & 패딩
    left = load_image_as_tensor(args.left, device)
    right = load_image_as_tensor(args.right, device)
    H0, W0 = left.shape[-2], left.shape[-1]
    left, (pad_h, pad_w) = pad_to_multiple_of(left, args.patch)
    right, _ = pad_to_multiple_of(right, args.patch)

    # 특징 추출기
    feat_net = QuarterFeatureExtractor(patch_size=args.patch, cnn_center=True, try_load_dino=True).to(device)
    chans = feat_net.channels()
    print(f"[INFO] Channels: DINO={chans['dino']}, CNN={chans['cnn']}")

    # 1/4 특징
    FL = feat_net(left)
    FR = feat_net(right)
    FdL, FdR = FL["dino"], FR["dino"]
    FcL, FcR = FL["cnn"],  FR["cnn"]

    # 파생 설정
    grid_stride = args.patch // 2                  # 1/4 셀 한 칸 == full-res 픽셀 grid_stride
    assert args.patch % 2 == 0, "patch size는 짝수여야 합니다."
    assert_multiple(args.max_disp, grid_stride, "max_disp")
    D = args.max_disp // grid_stride

    # 사용할 모드 파싱
    req_modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    valid_modes = {"dino_only", "late_weighted", "concat", "sum", "cnn_only"}
    for m in req_modes:
        if m not in valid_modes:
            raise ValueError(f"알 수 없는 mode: {m}. 사용 가능: {sorted(valid_modes)}")

    # DINO 미가용 시 모드 필터링
    if FdL is None or FdR is None:
        req_modes = [m for m in req_modes if m in ("cnn_only",)]
        if not req_modes:
            print("[WARN] DINO가 없어 'cnn_only'만 실행 가능합니다. --modes cnn_only 로 다시 시도하세요.")
            return

    # 모드별 실행
    for mode in req_modes:
        print(f"[RUN] mode = {mode}")
        if mode == "cnn_only":
            FLm, FRm = FcL, FcR
            vol, mask = build_corr_volume_with_mask(FLm, FRm, D)
        elif mode == "dino_only":
            FLm, FRm = FdL, FdR
            vol, mask = build_corr_volume_with_mask(FLm, FRm, D)
        elif mode == "late_weighted":
            vol_d, mask_d = build_corr_volume_with_mask(FdL, FdR, D)
            vol_c, mask_c = build_corr_volume_with_mask(FcL, FcR, D)
            a = float(args.dino_weight)
            vol  = a * vol_d + (1.0 - a) * vol_c
            mask = mask_d * mask_c
        elif mode == "concat":
            FfL = F.normalize(torch.cat([FdL, FcL], dim=1), dim=1, eps=1e-6)
            FfR = F.normalize(torch.cat([FdR, FcR], dim=1), dim=1, eps=1e-6)
            vol, mask = build_corr_volume_with_mask(FfL, FfR, D)
        elif mode == "sum":
            Cd, Cc = FdL.shape[1], FcL.shape[1]  # Cd should be multiple of Cc (=24)
            assert Cd % Cc == 0, "DINO 채널 수는 24의 배수여야 sum 가능"
            g = Cd // Cc
            # DINO 채널 → 24로 그룹 평균 축소
            def reduce_dino(Fd):
                B, _, h, w = Fd.shape
                return Fd.view(B, Cc, g, h, w).mean(dim=2).contiguous()
            Fd_red_L = F.normalize(reduce_dino(FdL), dim=1, eps=1e-6)
            Fd_red_R = F.normalize(reduce_dino(FdR), dim=1, eps=1e-6)
            a = float(args.sum_alpha)
            FfL = F.normalize(a * Fd_red_L + (1.0 - a) * FcL, dim=1, eps=1e-6)
            FfR = F.normalize(a * Fd_red_R + (1.0 - a) * FcR, dim=1, eps=1e-6)
            vol, mask = build_corr_volume_with_mask(FfL, FfR, D)
        else:
            raise RuntimeError("unreachable")

        # 마스킹 + WTA
        vol_masked = vol + (1.0 - mask) * (-1e4)
        disp_idx_q4 = vol_masked.argmax(dim=2).float()   # [B,1,H/4,W/4], 값 범위 0..D

        # 시각화: 1/4 해상도 disparity index
        disp_idx_q4_np = disp_idx_q4[0,0].detach().cpu().numpy()  # [H/4,W/4]
        disp_idx_img = to_colormap_image(disp_idx_q4_np, vmin=0.0, vmax=float(D), cmap=args.cmap)
        # save_image(disp_idx_img, os.path.join(args.out, f"disp_idx_q4_{mode}.png"))

        # 풀 해상도: 최근접 업샘플 + 픽셀 단위로 변환 (grid_stride px/1step)
        disp_full = F.interpolate(disp_idx_q4, scale_factor=4, mode="nearest") * float(grid_stride)  # [B,1,H_pad,W_pad]
        disp_full = crop_to_original(disp_full, (H0, W0))
        disp_full_np = disp_full[0,0].detach().cpu().numpy()  # [H,W]
        disp_full_img = to_colormap_image(disp_full_np, vmin=0.0, vmax=float(args.max_disp), cmap=args.cmap)
        save_image(disp_full_img, os.path.join(args.out, f"disp_px_full_{mode}.png"))

        # 옵션: 볼륨 저장 (메모리/용량 주의)
        if args.save_volume:
            np.save(os.path.join(args.out, f"cost_volume_{mode}.npy"),
                    vol_masked[0,0].detach().cpu().numpy().astype(np.float32))  # [D+1,H/4,W/4]

        print(f"[DONE] mode={mode}: "
              f"saved -> disp_idx_q4_{mode}.png, disp_px_full_{mode}.png"
              + (", cost_volume .npy" if args.save_volume else ""))

    print(f"[FINISHED] outputs saved in: {args.out}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cost volume checker with multiple fusion modes.")
    p.add_argument("--left", required=True, help="좌측 이미지 경로")
    p.add_argument("--right", required=True, help="우측 이미지 경로")
    p.add_argument("--out", default="./cv_outputs", help="출력 디렉터리")
    p.add_argument("--modes", default="late_weighted,concat",
                   help="실행할 모드들(콤마구분): dino_only,late_weighted,concat,sum,cnn_only")
    p.add_argument("--max-disp", dest="max_disp", type=int, default=80, help="최대 시차(px 단위, full-res)")
    p.add_argument("--patch", type=int, default=8, help="DINO patch size (보통 8). 입력 H,W는 이 값의 배수여야 함")
    p.add_argument("--dino-weight", dest="dino_weight", type=float, default=0.8, help="late_weighted에서 DINO 가중치")
    p.add_argument("--sum-alpha", dest="sum_alpha", type=float, default=0.5, help="sum 결합에서 DINO 비중")
    p.add_argument("--device", default="auto", help="'auto' | 'cpu' | 'cuda' | 'cuda:0' ...")
    p.add_argument("--save-volume", action="store_true", help="마스킹된 cost volume을 .npy로 저장")
    p.add_argument("--cmap", default="turbo", help="시각화 컬러맵 (matplotlib colormap 이름)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    torch.set_grad_enabled(False)
    run(args)
