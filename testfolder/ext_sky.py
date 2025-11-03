#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_change_heatmaps_dino_s8_accum_ema.py

기능 요약
- DINO ViT-S/8 패치 특징으로 프레임별 "변화량" 히트맵을 생성
- 로컬 모션 보상(±r 셀 탐색)으로 min-Δ 거리 사용
- 즉시 변화량(instantaneous) 히트맵 저장 (옵션)
- 전 과거 누적(무한 누적) EMA 히트맵 저장 (옵션)
  M_t = α M_{t-1} + (1-α) D_t
  * D_t는 (a) 현재 vs 직전, 또는 (b) 현재 vs 과거 K개 집계(mean/median/max/min)

사용 예시
python make_change_heatmaps_dino_s8_accum_ema.py \
  --image_dir /path/to/frames \
  --out_dir ./heatmaps \
  --search_radius 1 \
  --accum_ema 1 --accum_alpha 0.95 --accum_source prev \
  --history 6 --agg mean \
  --save_inst 1 --save_ema 1 --save_overlay 1 --alpha 0.6 --cmap magma
"""

import os
import re
import argparse
from collections import deque
from typing import List, Tuple, Deque, Dict, Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# headless 저장
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
    files.sort(key=natural_key)  # ✅ 핵심 수정: key=natural_key
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
    - img_t: (1,3,Hpad,Wpad) ImageNet 정규화 텐서
    - 반환: (Hp, Wp, C)  (CLS 제외, 마지막 레이어)
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
    반환:
      x_imnet : (1,3,Hpad,Wpad) ImageNet 정규화 텐서
      (H,W)   : 원본 크기
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
    x: (Hp, Wp) 거리 맵 -> 0..1
    분위수로 범위를 잡아 outlier 영향 축소
    """
    flat = x.flatten()
    vmin = torch.quantile(flat, vmin_q)
    vmax = torch.quantile(flat, vmax_q)
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
        search_radius: int = 1,
        # D_t 구성 방식
        accum_source: str = "prev",     # 'prev' or 'hist_mean'
        history: int = 6,               # accum_source='hist_mean'일 때 과거 창 크기
        agg: str = "mean",              # mean/median/max/min for hist_mean
        # 저장 옵션
        save_inst: bool = True,
        save_ema: bool = True,
        save_overlay: bool = True,
        alpha: float = 0.6,
        cmap_name: str = "magma",
        vmin_q: float = 0.02,
        vmax_q: float = 0.98,
        # 시간 EMA
        accum_ema: bool = True,
        accum_alpha: float = 0.95):

    device = get_device()
    print(f"[Info] Device: {device}")

    img_paths = list_images(image_dir)
    if len(img_paths) < 2:
        raise ValueError("시퀀스가 2장 미만입니다.")
    print(f"[Info] Found {len(img_paths)} frames")

    # 출력 폴더 구성
    inst_dir = os.path.join(out_dir, "inst_heat")
    inst_ov_dir = os.path.join(out_dir, "inst_overlay")
    ema_dir = os.path.join(out_dir, "ema_heat")
    ema_ov_dir = os.path.join(out_dir, "ema_overlay")
    os.makedirs(out_dir, exist_ok=True)
    if save_inst:
        os.makedirs(inst_dir, exist_ok=True)
        if save_overlay:
            os.makedirs(inst_ov_dir, exist_ok=True)
    if accum_ema and save_ema:
        os.makedirs(ema_dir, exist_ok=True)
        if save_overlay:
            os.makedirs(ema_ov_dir, exist_ok=True)

    print("[Info] Loading DINO v1 ViT-S/8 (torch.hub: facebookresearch/dino:main, dino_vits8)")
    model = load_dino_vits8(device)

    # 과거 특징 버퍼 (CPU half로 저장)
    past_buf: Deque[Dict[str, Any]] = deque(maxlen=max(1, history))
    prev_item: Dict[str, Any] = {}  # 직전 프레임 저장

    def feats_to_cpu_half(t: torch.Tensor) -> torch.Tensor:
        return t.detach().to("cpu", dtype=torch.float16).contiguous()

    def cpu_half_to_device_fp32(t: torch.Tensor) -> torch.Tensor:
        return t.to(device=device, dtype=torch.float32, non_blocking=True)

    def reduce_stack(stack: torch.Tensor, how: str) -> torch.Tensor:
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

    ema_map = None  # (Hp, Wp) on device

    for idx, p in enumerate(img_paths):
        base = os.path.basename(p)

        # 1) 특징 추출
        x_t, (H, W), (Hpad, Wpad) = load_and_prepare(p, device, pad_to_multiple=patch_size)
        feats_t = extract_dino_patch_tokens(model, x_t, patch_size)
        feats_t = l2_normalize(feats_t, dim=-1)   # (Hp, Wp, C)
        Hp, Wp, C = feats_t.shape

        # 2) D_t (instantaneous change map) 계산
        have_D = False
        D_map = None

        if accum_source == "prev":
            if prev_item:
                pi = prev_item
                if (pi["Hp"] == Hp and pi["Wp"] == Wp and pi["Hpad"] == Hpad and pi["Wpad"] == Wpad):
                    feats_prev = cpu_half_to_device_fp32(pi["feats"])
                    D_map = min_dist_with_local_search(feats_prev, feats_t, r=search_radius)  # (Hp,Wp)
                    have_D = True
                else:
                    print(f"[Warn] {base}: grid mismatch vs previous. Skipped D_t (prev).")
            else:
                print(f"[Info] {base}: no previous frame yet. Skipped D_t (prev).")

        elif accum_source == "hist_mean":
            dist_maps: List[torch.Tensor] = []
            valid_past = 0
            for item in past_buf:
                if item["Hp"] != Hp or item["Wp"] != Wp or item["Hpad"] != Hpad or item["Wpad"] != Wpad:
                    continue
                feats_past = cpu_half_to_device_fp32(item["feats"])
                dist = min_dist_with_local_search(feats_past, feats_t, r=search_radius)
                dist_maps.append(dist)
                valid_past += 1
            if valid_past > 0:
                stack = torch.stack(dist_maps, dim=0)  # (K,Hp,Wp)
                D_map = reduce_stack(stack, agg)       # (Hp,Wp)
                have_D = True
            else:
                print(f"[Info] {base}: no valid past window. Skipped D_t (hist_mean).")
        else:
            raise ValueError("--accum_source must be 'prev' or 'hist_mean'")

        # 3) 즉시 변화량 저장(옵션)
        if have_D and save_inst:
            D01 = normalize_per_frame(D_map, vmin_q=vmin_q, vmax_q=vmax_q)
            D01_up = F.interpolate(D01[None, None].float(), size=(Hpad, Wpad),
                                   mode='bilinear', align_corners=False)[0, 0]
            D01_up = D01_up[:H, :W]
            heat_img = tensor_to_colormap_img(D01_up, cmap_name=cmap_name)
            heat_path = os.path.join(inst_dir, base)
            heat_img.save(heat_path)
            if save_overlay:
                orig_img = Image.open(p).convert("RGB")
                ov = overlay_on_image(orig_img, heat_img, alpha=alpha)
                ov_path = os.path.join(inst_ov_dir, base)
                ov.save(ov_path)
            print(f"[Save][inst] {base}")

        # 4) 시간 EMA 누적(옵션)
        if accum_ema and have_D:
            if (ema_map is None) or (ema_map.shape[0] != Hp or ema_map.shape[1] != Wp):
                ema_map = torch.zeros((Hp, Wp), device=device)
                print(f"[Info] EMA map initialized @ {Hp}x{Wp}")
            ema_map = accum_alpha * ema_map + (1.0 - accum_alpha) * D_map  # on device

            if save_ema:
                E01 = normalize_per_frame(ema_map, vmin_q=vmin_q, vmax_q=vmax_q)
                E01_up = F.interpolate(E01[None, None].float(), size=(Hpad, Wpad),
                                       mode='bilinear', align_corners=False)[0, 0]
                E01_up = E01_up[:H, :W]
                ema_img = tensor_to_colormap_img(E01_up, cmap_name=cmap_name)
                ema_path = os.path.join(ema_dir, base)
                ema_img.save(ema_path)
                if save_overlay:
                    orig_img = Image.open(p).convert("RGB")
                    ov = overlay_on_image(orig_img, ema_img, alpha=alpha)
                    ov_path = os.path.join(ema_ov_dir, base)
                    ov.save(ov_path)
                print(f"[Save][ema]  {base}")

        # 5) 버퍼 갱신 (마지막에)
        item = {
            "feats": feats_t.detach().to("cpu", dtype=torch.float16).contiguous(),
            "Hp": Hp, "Wp": Wp, "Hpad": Hpad, "Wpad": Wpad,
        }
        prev_item = item  # 직전 프레임
        if accum_source == "hist_mean":
            past_buf.append(item)

    print("\n[Done] Outputs in:", os.path.abspath(out_dir))

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="DINO ViT-S/8 change heatmaps with infinite-time EMA accumulation.")
    ap.add_argument("--image_dir", type=str, default='/home/jaejun/dataset/MS2/sync_data/tester/rgb/img_left', help="입력 시퀀스 이미지 폴더")
    ap.add_argument("--out_dir", type=str, default="../log/heatmap_out", help="출력 폴더(하위에 inst/ema 등 생성)")

    ap.add_argument("--patch_size", type=int, default=8, help="DINO 패치 크기(8)")
    ap.add_argument("--search_radius", type=int, default=1, help="로컬 모션 보상 탐색 반경 r (0이면 동일 위치 비교)")

    ap.add_argument("--accum_source", type=str, default="hist_mean", choices=["prev", "hist_mean"],
                    help="D_t 구성 방식: prev=현재vs직전, hist_mean=현재vs과거K 집계")
    ap.add_argument("--history", type=int, default=6, help="accum_source=hist_mean일 때 과거 창 크기")
    ap.add_argument("--agg", type=str, default="mean", choices=["mean", "median", "max", "min"],
                    help="hist_mean의 거리 집계 방식")

    ap.add_argument("--accum_ema", type=int, default=1, help="시간 EMA 누적 저장 여부(1/0)")
    ap.add_argument("--accum_alpha", type=float, default=0.95, help="EMA α (0~1, 클수록 과거 가중↑)")

    ap.add_argument("--save_inst", type=int, default=1, help="즉시 변화량 히트맵 저장(1/0)")
    ap.add_argument("--save_ema", type=int, default=1, help="EMA 누적 히트맵 저장(1/0)")
    ap.add_argument("--save_overlay", type=int, default=1, help="원본 위 오버레이 저장(1/0)")
    ap.add_argument("--alpha", type=float, default=0.6, help="오버레이 알파")
    ap.add_argument("--cmap", type=str, default="magma",
                    help="matplotlib 컬러맵 이름(예: magma, viridis, turbo, jet)")
    ap.add_argument("--vmin_q", type=float, default=0.02, help="정규화 하위 분위수")
    ap.add_argument("--vmax_q", type=float, default=0.98, help="정규화 상위 분위수")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.set_grad_enabled(False)
    _ = run(
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        patch_size=args.patch_size,
        search_radius=max(0, args.search_radius),
        accum_source=args.accum_source,
        history=max(1, args.history),
        agg=args.agg,
        save_inst=bool(args.save_inst),
        save_ema=bool(args.save_ema),
        save_overlay=bool(args.save_overlay),
        alpha=args.alpha,
        cmap_name=args.cmap,
        vmin_q=args.vmin_q,
        vmax_q=args.vmax_q,
        accum_ema=bool(args.accum_ema),
        accum_alpha=args.accum_alpha,
    )
