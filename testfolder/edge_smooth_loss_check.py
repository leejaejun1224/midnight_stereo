#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
from typing import Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ===============================
# 이미지 로드/전처리 유틸
# ===============================

def load_image_as_tensor(path: str, to_float01: bool = True) -> torch.Tensor:
    """
    path 이미지를 [1,C,H,W] float32 텐서로 반환.
    RGB로 강제 변환. to_float01이면 [0,1] 스케일.
    """
    img = Image.open(path).convert("RGB")
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()  # [C,H,W], 0..255
    if to_float01:
        x = x / 255.0
    return x.unsqueeze(0)  # [1,C,H,W]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ===============================
# 엣지 강도/가중치 계산
# ===============================

@torch.no_grad()
def edge_strength_and_weight(img: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    img: [B,C,H,W], float, [0,1] 권장
    returns:
      edge_strength: [B,1,H,W]   = mean_c(|∂x I|)+mean_c(|∂y I|)
      edge_weight:   [B,1,H,W]   = exp(-edge_strength)
      edge_vis:      [B,1,H,W]   = edge_strength의 [0,1] 정규화(시각화용)
      weight_vis:    [B,1,H,W]   = edge_weight의 [0,1] 정규화(시각화용)
    """
    # 1-픽셀 차분
    gx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])   # [B,C,H,W-1]
    gy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])   # [B,C,H-1,W]
    gx = gx.mean(1, keepdim=True)  # [B,1,H,W-1]
    gy = gy.mean(1, keepdim=True)  # [B,1,H-1,W]

    # 해상도 복원(경계 패딩)
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))

    edge_strength = gx + gy                    # [B,1,H,W]
    edge_weight   = torch.exp(-edge_strength)  # [B,1,H,W]

    # [0,1] 정규화(이미지별)
    emax = edge_strength.flatten(2).amax(dim=2, keepdim=True) + eps
    edge_vis = edge_strength / emax

    w_min = edge_weight.flatten(2).amin(dim=2, keepdim=True)
    w_max = edge_weight.flatten(2).amax(dim=2, keepdim=True)
    weight_vis = (edge_weight - w_min) / (w_max - w_min + eps)

    return edge_strength, edge_weight, edge_vis, weight_vis


def sobel_edge_strength(img: torch.Tensor) -> torch.Tensor:
    """
    Sobel 기반 엣지 강도 (채널 평균 L1).
    img: [B,C,H,W] in [0,1]
    returns: [B,1,H,W]
    """
    B, C, H, W = img.shape
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
    kx = kx.repeat(C, 1, 1, 1)
    ky = ky.repeat(C, 1, 1, 1)
    gx = F.conv2d(img, kx, padding=1, groups=C)  # [B,C,H,W]
    gy = F.conv2d(img, ky, padding=1, groups=C)
    edge = (gx.abs().mean(1, keepdim=True) + gy.abs().mean(1, keepdim=True))
    return edge


@torch.no_grad()
def edge_strength_and_weight_sobel(img: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sobel로 edge_strength를 만들고 동일한 방식으로 가중치/정규화를 계산.
    """
    edge_strength = sobel_edge_strength(img)          # [B,1,H,W]
    edge_weight   = torch.exp(-edge_strength)         # [B,1,H,W]

    emax = edge_strength.flatten(2).amax(dim=2, keepdim=True) + eps
    edge_vis = edge_strength / emax

    w_min = edge_weight.flatten(2).amin(dim=2, keepdim=True)
    w_max = edge_weight.flatten(2).amax(dim=2, keepdim=True)
    weight_vis = (edge_weight - w_min) / (w_max - w_min + eps)

    return edge_strength, edge_weight, edge_vis, weight_vis


# ===============================
# 패치 요약(평균/강한 엣지 비율)
# ===============================

@torch.no_grad()
def edge_per_patch(edge_strength: torch.Tensor, patch: int = 32, threshold: Optional[float] = None):
    """
    edge_strength: [B,1,H,W]
    returns:
      avg_edge:  [B,1,H/patch,W/patch]  (패치 평균 |∇I|)
      edge_ratio: (옵션) 임계값 초과 비율 맵
    """
    avg_edge = F.avg_pool2d(edge_strength, kernel_size=patch, stride=patch)
    if threshold is None:
        return avg_edge, None
    strong = (edge_strength > threshold).float()
    edge_ratio = F.avg_pool2d(strong, kernel_size=patch, stride=patch)
    return avg_edge, edge_ratio


# ===============================
# 시각화/저장
# ===============================

def save_overlay(img: torch.Tensor, heat: torch.Tensor, out_path: str, title: str):
    """
    img:  [1,C,H,W] in [0,1]
    heat: [1,1,H,W] in [0,1]
    """
    img_np  = img[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    heat_np = heat[0, 0].detach().cpu().clamp(0, 1).numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    plt.imshow(heat_np, alpha=0.5)
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_gray(heat: torch.Tensor, out_path: str, title: str):
    """
    heat: [1,1,H,W] in arbitrary scale; 내부에서 [0,1]로 맞춰 그림.
    """
    h = heat[0, 0].detach().cpu()
    h = (h - h.min()) / (h.max() - h.min() + 1e-6)
    plt.figure(figsize=(6, 6))
    plt.imshow(h.numpy())
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_patch_map(t: torch.Tensor, out_path: str, title: str):
    """
    t: [1,1,Hp,Wp]  (패치 요약 맵)
    """
    x = t[0, 0].detach().cpu()
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    plt.figure(figsize=(6, 6))
    plt.imshow(x.numpy())
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ===============================
# 메인 루틴
# ===============================

def process_one_image(img_path: str,
                      out_dir: str,
                      use_sobel: bool = False,
                      patch: int = 32,
                      strong_ratio_thr: Optional[float] = None):
    """
    img_path 한 장 처리하여 결과 저장.
    strong_ratio_thr:
        None이면 강한 엣지 비율 미계산.
        값이 있으면 edge_strength > strong_ratio_thr 기준으로 비율 계산.
        (일반적으로 edge_strength.mean() 또는 quantile 사용 추천)
    """
    stem = os.path.splitext(os.path.basename(img_path))[0]
    img = load_image_as_tensor(img_path, to_float01=True)  # [1,3,H,W]

    if use_sobel:
        edge_strength, edge_weight, edge_vis, weight_vis = edge_strength_and_weight_sobel(img)
        tag = "sobel"
    else:
        edge_strength, edge_weight, edge_vis, weight_vis = edge_strength_and_weight(img)
        tag = "diff"

    # 저장 경로
    ensure_dir(out_dir)

    # 1) 맵 자체 저장
    save_gray(edge_strength, os.path.join(out_dir, f"{stem}_edge_strength_{tag}.png"), f"edge_strength ({tag})")
    save_gray(edge_weight,   os.path.join(out_dir, f"{stem}_edge_weight_{tag}.png"),   f"edge_weight exp(-|∇I|) ({tag})")

    # 2) 원본 오버레이
    save_overlay(img, edge_vis,   os.path.join(out_dir, f"{stem}_overlay_edge_{tag}.png"),   f"Overlay edge ({tag})")
    save_overlay(img, weight_vis, os.path.join(out_dir, f"{stem}_overlay_weight_{tag}.png"), f"Overlay weight ({tag})")

    # 3) 패치 요약
    avg_edge, edge_ratio = edge_per_patch(edge_strength, patch=patch,
                                          threshold=None if strong_ratio_thr is None else strong_ratio_thr)
    save_patch_map(avg_edge, os.path.join(out_dir, f"{stem}_patch_avg_edge_{tag}_p{patch}.png"),
                   f"Avg edge per {patch}x{patch} patch ({tag})")
    if edge_ratio is not None:
        save_patch_map(edge_ratio, os.path.join(out_dir, f"{stem}_patch_strong_ratio_{tag}_p{patch}.png"),
                       f"Strong-edge ratio per patch ({tag})")


def collect_images(image: Optional[str], dir_path: Optional[str]) -> list:
    if image:
        return [image]
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(dir_path, e)))
    files.sort()
    return files


def parse_args():
    p = argparse.ArgumentParser(description="Edge strength visualization and patch summary")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=str, help="단일 이미지 경로")
    src.add_argument("--dir", type=str, help="이미지 폴더 경로(하위 png/jpg 등 일괄 처리)")
    p.add_argument("--out", type=str, default="./edge_vis_out", help="출력 폴더")
    p.add_argument("--sobel", action="store_true", help="Sobel 기반 엣지 강도 사용")
    p.add_argument("--patch", type=int, default=32, help="패치 요약 크기(기본 32)")
    p.add_argument("--strong_thr", type=float, default=None,
                   help="강한 엣지 비율 임계값(없으면 비율 미계산). 예: --strong_thr 0.1")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out)

    if args.image:
        paths = collect_images(args.image, None)
    else:
        paths = collect_images(None, args.dir)

    if len(paths) == 0:
        print("처리할 이미지가 없습니다.")
        return

    for path in paths:
        print(f"[process] {path}")
        process_one_image(
            img_path=path,
            out_dir=args.out,
            use_sobel=args.sobel,
            patch=args.patch,
            strong_ratio_thr=args.strong_thr
        )

    print(f"완료. 결과는 '{args.out}' 폴더에 저장되었습니다.")


if __name__ == "__main__":
    main()
