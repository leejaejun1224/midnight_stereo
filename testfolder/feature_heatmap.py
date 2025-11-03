#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_change_heatmaps_dino_s8.py

주요 기능
- DINO v1 ViT-S/8 패치 특징으로 현재 프레임의 각 패치가
  과거 N프레임(예: 5~6장) 대비 얼마나 변했는지(=특징 거리)를 히트맵으로 시각화/저장
- 작은 에고모션을 보상하기 위해 ±r(패치 그리드) 내에서 최소 거리를 사용 (local motion compensation)
- 여러 과거 프레임과의 거리들을 mean/median/max/min 등으로 집계
- 히트맵 PNG 및 원본에 반투명 오버레이 PNG 저장(옵션)
- 메모리 절약을 위해 과거 특징은 CPU float16으로 보관

의존성
- torch, torchvision, PIL, matplotlib

사용 예시
python make_change_heatmaps_dino_s8.py \
  --image_dir /path/to/frames \
  --history 6 \
  --search_radius 1 \
  --agg mean \
  --out_dir ./heatmaps \
  --save_overlay 1 --alpha 0.6 --cmap magma
"""

import os
import re
import argparse
from collections import deque
from typing import List, Tuple, Deque

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# headless 저장을 위해 Agg 백엔드
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm
import numpy as np

# -----------------------------
# 유틸
# -----------------------------
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_images(image_dir: str, exts: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')):
    files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
             if os.path.splitext(f.lower())[1] in exts]
    files.sort(key=natural_key)
    return files

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

# -----------------------------
# DINO v1 ViT-S/8 로더 & 특징 추출
# -----------------------------
def load_dino_vits8(device: torch.device):
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    model.eval()
    model.to(device)
    return model

@torch.no_grad()
def extract_dino_patch_tokens(model, img_t: torch.Tensor, patch_size: int = 8) -> torch.Tensor:
    """
    - img_t: (1, 3, Hpad, Wpad) ImageNet 정규화 텐서
    - 반환: (H_p, W_p, C)  (CLS 제외, 마지막 레이어)
    """
    feats = model.get_intermediate_layers(img_t, n=1)[0]  # (1, N+1, C)
    feats = feats[:, 1:, :]  # remove CLS
    B, N, C = feats.shape
    _, _, Hpad, Wpad = img_t.shape
    H_p = Hpad // patch_size
    W_p = Wpad // patch_size
    assert N == H_p * W_p, f"N={N} != H_p*W_p={H_p*W_p}"
    feats = feats.reshape(B, H_p, W_p, C)
    return feats[0]

# -----------------------------
# 전처리: 패딩/정규화
# -----------------------------
IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD  = [0.229, 0.224, 0.225]

def load_and_prepare(path: str, device: torch.device, pad_to_multiple: int = 8):
    """
    - 반환:
      x_imnet : (1,3,Hpad,Wpad) ImageNet 정규화 텐서
      (H, W)  : 원본 크기
      (Hpad,Wpad) : 패딩 후 크기
    """
    img = Image.open(path).convert('RGB')
    W, H = img.size

    pad_r = (pad_to_multiple - (W % pad_to_multiple)) % pad_to_multiple
    pad_b = (pad_to_multiple - (H % pad_to_multiple)) % pad_to_multiple

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMNET_MEAN, std=IMNET_STD),
    ])
    x = transform(img)  # (3,H,W)
    if pad_r != 0 or pad_b != 0:
        x = F.pad(x, (0, pad_r, 0, pad_b), value=0.0)

    x = x.unsqueeze(0).to(device)  # (1,3,Hpad,Wpad)
    Hpad, Wpad = x.shape[-2], x.shape[-1]
    return x, (H, W), (Hpad, Wpad)

# -----------------------------
# Local motion-compensated distance (±r 탐색)
# -----------------------------
def min_dist_with_local_search(feats_a: torch.Tensor,
                               feats_b: torch.Tensor,
                               r: int = 1):
    """
    feats_*: (Hp, Wp, C) [L2 정규화 완료]
    r: 패치 그리드에서 탐색 반경(0이면 동일 위치 비교)
    반환: dist_map (Hp, Wp), 최소 거리
    """
    Hp, Wp, C = feats_a.shape
    device = feats_a.device

    if r <= 0:
        sim = (feats_a * feats_b).sum(dim=-1)         # (Hp, Wp)
        return 1.0 - sim

    # 큰 값으로 초기화
    best = torch.full((Hp, Wp), 2.0, device=device)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            y0, y1 = max(0,  dy), min(Hp, Hp + dy)
            x0, x1 = max(0,  dx), min(Wp, Wp + dx)
            yy0, yy1 = max(0, -dy), min(Hp, Hp - dy)
            xx0, xx1 = max(0, -dx), min(Wp, Wp - dx)
            if y0 >= y1 or x0 >= x1:
                continue
            sim = (feats_a[yy0:yy1, xx0:xx1] * feats_b[y0:y1, x0:x1]).sum(dim=-1)
            dist = 1.0 - sim
            sub = best[yy0:yy1, xx0:xx1]
            best[yy0:yy1, xx0:xx1] = torch.minimum(sub, dist)
    return best

# -----------------------------
# 히트맵 유틸
# -----------------------------
def normalize_per_frame(x: torch.Tensor, vmin_q: float = 0.02, vmax_q: float = 0.98) -> torch.Tensor:
    """
    x: (Hp, Wp) 거리 맵 -> 0..1로 정규화
    분위수로 범위를 잡아 outlier 영향 축소
    """
    flat = x.flatten()
    vmin = torch.quantile(flat, vmin_q)
    vmax = torch.quantile(flat, vmax_q)
    # 수치 안전장치
    if float(vmax - vmin) < 1e-6:
        return torch.zeros_like(x)
    return ((x - vmin) / (vmax - vmin)).clamp_(0.0, 1.0)

def tensor_to_colormap_img(x01: torch.Tensor, cmap_name: str = "magma") -> Image.Image:
    """
    x01: (H, W) 0..1 텐서 -> RGB heatmap PIL Image
    """
    arr = x01.detach().cpu().numpy()
    rgba = cm.get_cmap(cmap_name)(arr)  # (H,W,4), 0..1
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return Image.fromarray(rgb)

def overlay_on_image(base_img: Image.Image, heat_img: Image.Image, alpha: float = 0.6) -> Image.Image:
    """
    base_img, heat_img: 동일 크기 PIL Image (RGB)
    alpha: heat 가중
    """
    base = np.array(base_img).astype(np.float32)
    heat = np.array(heat_img).astype(np.float32)
    over = (alpha * heat + (1 - alpha) * base).clip(0, 255).astype(np.uint8)
    return Image.fromarray(over)

# -----------------------------
# 메인
# -----------------------------
@torch.no_grad()
def run(image_dir: str,
        out_dir: str,
        patch_size: int = 8,
        history: int = 6,
        search_radius: int = 1,
        agg: str = "mean",           # mean/median/max/min
        save_overlay: bool = True,
        alpha: float = 0.6,
        cmap_name: str = "magma",
        vmin_q: float = 0.02,
        vmax_q: float = 0.98):

    device = get_device()
    print(f"[Info] Device: {device}")

    img_paths = list_images(image_dir)
    if len(img_paths) < 2:
        raise ValueError("시퀀스가 2장 미만입니다. 최소 2장 이상이 필요합니다.")
    print(f"[Info] Found {len(img_paths)} frames")

    # 출력 폴더
    heat_dir = os.path.join(out_dir, "heat")
    ovly_dir = os.path.join(out_dir, "overlay")
    os.makedirs(heat_dir, exist_ok=True)
    if save_overlay:
        os.makedirs(ovly_dir, exist_ok=True)

    print("[Info] Loading DINO v1 ViT-S/8 (torch.hub: facebookresearch/dino:main, dino_vits8)")
    model = load_dino_vits8(device)

    # 과거 특징 버퍼(최대 history개). 메모리 절약을 위해 CPU float16으로 보관.
    # 원소: dict(feats, Hp, Wp, H, W, Hpad, Wpad)
    past_buf: Deque[dict] = deque(maxlen=max(1, history))

    def feats_to_cpu_half(t: torch.Tensor) -> torch.Tensor:
        # (Hp, Wp, C) -> CPU half
        return t.detach().to("cpu", dtype=torch.float16).contiguous()

    def cpu_half_to_device_fp32(t: torch.Tensor) -> torch.Tensor:
        return t.to(device=device, dtype=torch.float32, non_blocking=True)

    # 도우미: 거리 집계 방식
    def reduce_stack(stack: torch.Tensor, how: str) -> torch.Tensor:
        # stack: (K, Hp, Wp)
        if how == "mean":
            return stack.mean(dim=0)
        elif how == "median":
            return stack.median(dim=0).values
        elif how == "max":
            return stack.max(dim=0).values
        elif how == "min":
            return stack.min(dim=0).values
        else:
            raise ValueError(f"Unknown agg: {how}")

    # 메인 루프
    for idx, p in enumerate(img_paths):
        # 1) 특징 추출
        x_t, (H, W), (Hpad, Wpad) = load_and_prepare(p, device, pad_to_multiple=patch_size)
        feats_t = extract_dino_patch_tokens(model, x_t, patch_size)
        feats_t = l2_normalize(feats_t, dim=-1)   # (Hp, Wp, C)
        Hp, Wp, C = feats_t.shape

        # 2) 현재 프레임의 변화량 히트맵 계산 (과거 버퍼와 비교)
        dist_maps: List[torch.Tensor] = []
        valid_past = 0
        for item in past_buf:
            # 해상도/그리드가 다르면 스킵
            if item["Hp"] != Hp or item["Wp"] != Wp or item["Hpad"] != Hpad or item["Wpad"] != Wpad:
                continue
            feats_past = cpu_half_to_device_fp32(item["feats"])  # (Hp, Wp, C) fp32 on device
            # 로컬 모션 보상 최소 거리
            dist = min_dist_with_local_search(feats_past, feats_t, r=search_radius)  # (Hp, Wp)
            dist_maps.append(dist)
            valid_past += 1

        if valid_past == 0:
            print(f"[Info] {idx+1}/{len(img_paths)} | {os.path.basename(p)} : 과거 비교 불가(초기 구간). 스킵.")
        else:
            stack = torch.stack(dist_maps, dim=0)  # (K, Hp, Wp)
            change_map = reduce_stack(stack, agg)  # (Hp, Wp)

            # 3) 0..1 정규화(프레임별 분위수 기반) → 업샘플 → 저장
            cm01 = normalize_per_frame(change_map, vmin_q=vmin_q, vmax_q=vmax_q)  # (Hp, Wp)
            cm01_up = F.interpolate(cm01[None, None].float(), size=(Hpad, Wpad),
                                    mode='bilinear', align_corners=False)[0, 0]  # (Hpad,Wpad)
            cm01_up = cm01_up[:H, :W]  # 패딩 제거

            # 색맵 이미지
            heat_img = tensor_to_colormap_img(cm01_up, cmap_name=cmap_name)

            # 저장 경로
            base = os.path.basename(p)
            heat_path = os.path.join(heat_dir, base)
            heat_img.save(heat_path)

            if save_overlay:
                orig_img = Image.open(p).convert("RGB")
                overlay_img = overlay_on_image(orig_img, heat_img, alpha=alpha)
                ovly_path = os.path.join(ovly_dir, base)
                overlay_img.save(ovly_path)

            print(f"[Save] {base} | heat → {os.path.relpath(heat_path)}"
                  f"{' | overlay → ' + os.path.relpath(ovly_path) if save_overlay else ''}")

        # 4) 현재 프레임을 버퍼에 push (마지막에)
        past_buf.append({
            "feats": feats_to_cpu_half(feats_t),
            "Hp": Hp, "Wp": Wp,
            "H": H, "W": W,
            "Hpad": Hpad, "Wpad": Wpad
        })

    print(f"\n[Done] Heatmaps saved to: {heat_dir}")
    if save_overlay:
        print(f"[Done] Overlays saved to: {ovly_dir}")

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="DINO ViT-S/8 change heatmaps with multi-frame history and local motion compensation.")
    ap.add_argument("--image_dir", type=str, default='/home/jaejun/dataset/MS2/sync_data/tester/rgb/img_left', help="입력 시퀀스 이미지 폴더 경로")
    ap.add_argument("--out_dir", type=str, default="./heatmap_out", help="히트맵/오버레이 저장 폴더")
    ap.add_argument("--patch_size", type=int, default=8, help="DINO 패치 크기(8)")
    ap.add_argument("--history", type=int, default=6, help="과거 프레임 개수 (윈도우 크기)")
    ap.add_argument("--search_radius", type=int, default=1, help="로컬 모션 보상 탐색 반경 r (0이면 동일 위치 비교)")
    ap.add_argument("--agg", type=str, default="mean", choices=["mean", "median", "max", "min"],
                    help="과거 프레임들과의 거리 집계 방식")
    ap.add_argument("--save_overlay", type=int, default=1, help="원본 위 오버레이 저장(1) / 저장 안함(0)")
    ap.add_argument("--alpha", type=float, default=0.6, help="오버레이 알파(heat 가중치)")
    ap.add_argument("--cmap", type=str, default="magma", help="matplotlib 컬러맵 이름(예: magma, viridis, turbo, jet)")
    ap.add_argument("--vmin_q", type=float, default=0.02, help="정규화 하위 분위수 (0~1)")
    ap.add_argument("--vmax_q", type=float, default=0.98, help="정규화 상위 분위수 (0~1)")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.set_grad_enabled(False)
    _ = run(
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        patch_size=args.patch_size,
        history=max(1, args.history),
        search_radius=max(0, args.search_radius),
        agg=args.agg,
        save_overlay=bool(args.save_overlay),
        alpha=args.alpha,
        cmap_name=args.cmap,
        vmin_q=args.vmin_q,
        vmax_q=args.vmax_q,
    )
