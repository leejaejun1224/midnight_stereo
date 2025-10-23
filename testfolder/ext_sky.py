#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
find_sky_patch_dino_s8.py
- DINO v1 ViT-S/8 패치 특징으로 시퀀스 상단 절반(하늘 후보)에서
  프레임 간 feature distance 하위 5% 패치를 골라 EMA 프로토타입을 만들고,
  첫 프레임에서 가장 유사한 패치 위치를 찾습니다.
"""

import os
import re
import math
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# -----------------------------
# 유틸
# -----------------------------
def natural_key(s: str):
    # 문자열 내 숫자를 정수로 변환해 자연 정렬
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
    """
    facebookresearch/dino:main 의 dino_vits8 모델을 torch.hub로 로드.
    """
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    model.eval()
    model.to(device)
    return model

@torch.no_grad()
def extract_dino_patch_tokens(model, img_t: torch.Tensor, patch_size: int = 8) -> torch.Tensor:
    """
    모델에서 마지막 레이어의 패치 토큰을 추출하여 (H_p, W_p, C)로 반환.
    - img_t: (1, 3, Hpad, Wpad) [0-1] 정규화 후 ImageNet 정규화된 텐서
    - 반환: feats_hw: (H_p, W_p, C) L2 정규화 전(여기서 정규화는 호출부에서 수행)
    """
    # facebook DINO 모델은 get_intermediate_layers 제공
    feats = model.get_intermediate_layers(img_t, n=1)[0]  # (1, N+1, C)
    feats = feats[:, 1:, :]  # CLS 제거 -> (1, N, C)

    B, N, C = feats.shape
    _, _, Hpad, Wpad = img_t.shape
    H_p = Hpad // patch_size
    W_p = Wpad // patch_size
    assert N == H_p * W_p, f"Token 수 N={N} != H_p*W_p={H_p*W_p}. 입력 크기/패치 크기를 확인하세요."

    feats = feats.reshape(B, H_p, W_p, C)
    return feats[0]  # (H_p, W_p, C)

# -----------------------------
# 전처리: 패딩/정규화
# -----------------------------
IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD  = [0.229, 0.224, 0.225]

def load_and_prepare(path: str, device: torch.device, pad_to_multiple: int = 8):
    """
    이미지를 로드하여 오른쪽/아래로 8의 배수가 되도록 0 패딩하고,
    ToTensor + ImageNet 정규화를 적용해 (1,3,Hpad,Wpad) 텐서와
    원본 크기(H,W)를 함께 반환.
    """
    img = Image.open(path).convert('RGB')
    W, H = img.size

    # pad right/bottom to multiple of 8
    pad_r = (pad_to_multiple - (W % pad_to_multiple)) % pad_to_multiple
    pad_b = (pad_to_multiple - (H % pad_to_multiple)) % pad_to_multiple

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMNET_MEAN, std=IMNET_STD),
    ])
    x = transform(img)  # (3,H,W)
    if pad_r != 0 or pad_b != 0:
        # pad format: (left, right, top, bottom)
        x = F.pad(x, (0, pad_r, 0, pad_b), value=0.0)
    x = x.unsqueeze(0).to(device)  # (1,3,Hpad,Wpad)
    Hpad, Wpad = x.shape[-2], x.shape[-1]
    return x, (H, W), (Hpad, Wpad)

def build_top_half_valid_mask(H: int, W: int, Hpad: int, Wpad: int, patch: int = 8, device=None):
    """
    원본 H,W와 패딩된 Hpad,Wpad가 주어졌을 때,
    패치 그리드(H_p, W_p) 상에서 '원본 영역 안'이면서 '상단 절반(하늘 후보)'인 위치 마스크를 생성.
    - 유효 기준: 패치 중심 (y= row*patch + patch/2, x= col*patch + patch/2)
                 가 각각 y < H, x < W (원본 내부) 이고, y < H/2 (상단 절반)
    반환: mask: (H_p, W_p) bool
    """
    H_p = Hpad // patch
    W_p = Wpad // patch
    ys = torch.arange(H_p, device=device) * patch + patch // 2  # 패치 중심 y
    xs = torch.arange(W_p, device=device) * patch + patch // 2  # 패치 중심 x
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    inside = (yy < H) & (xx < W)
    top_half = yy < (H // 2)
    return (inside & top_half)

# -----------------------------
# 메인 로직
# -----------------------------
@torch.no_grad()
def run(image_dir: str, ema_decay: float = 0.9, low_pct: float = 5.0, patch_size: int = 8):
    device = get_device()
    print(f"[Info] Device: {device}")

    # 1) 이미지 목록
    img_paths = list_images(image_dir)
    if len(img_paths) < 2:
        raise ValueError("시퀀스가 2장 미만입니다. 최소 2장 이상의 연속 프레임이 필요합니다.")
    print(f"[Info] Found {len(img_paths)} frames")

    # 2) DINO v1 ViT-S/8 로드
    print("[Info] Loading DINO v1 ViT-S/8 (torch.hub: facebookresearch/dino:main, dino_vits8)")
    model = load_dino_vits8(device)

    # 3) 첫 프레임 특징 미리 계산 (슬라이딩 윈도우용)
    x0, (H0, W0), (H0pad, W0pad) = load_and_prepare(img_paths[0], device, pad_to_multiple=patch_size)
    feats_prev = extract_dino_patch_tokens(model, x0, patch_size)  # (H_p, W_p, C)
    feats_prev = l2_normalize(feats_prev, dim=-1)

    # 첫 프레임의 상단 절반 유효 마스크 (최종 위치 검색에 재사용)
    top_mask_first = build_top_half_valid_mask(H0, W0, H0pad, W0pad, patch=patch_size, device=device)

    ema_vec = None
    Cdim = feats_prev.shape[-1]

    # 4) 시퀀스 순회: (t, t+1) 쌍으로 진행
    for idx in range(len(img_paths) - 1):
        p_t   = img_paths[idx]
        p_t1  = img_paths[idx + 1]

        # 현재 쌍 로드/특징
        if idx == 0:
            feats_t = feats_prev  # 이미 계산
            Hpad_t, Wpad_t = x0.shape[-2], x0.shape[-1]
            H_t, W_t = H0, W0
        else:
            xt, (H_t, W_t), (Hpad_t, Wpad_t) = load_and_prepare(p_t, device, pad_to_multiple=patch_size)
            feats_t = extract_dino_patch_tokens(model, xt, patch_size)
            feats_t = l2_normalize(feats_t, dim=-1)

        xt1, (H_t1, W_t1), (Hpad_t1, Wpad_t1) = load_and_prepare(p_t1, device, pad_to_multiple=patch_size)
        feats_t1 = extract_dino_patch_tokens(model, xt1, patch_size)
        feats_t1 = l2_normalize(feats_t1, dim=-1)

        # 안전성: 패치 그리드 크기가 달라지면(이상 케이스) 현재 쌍을 스킵
        if feats_t.shape != feats_t1.shape:
            print(f"[Warn] Patch grid mismatch at {os.path.basename(p_t)} -> {os.path.basename(p_t1)}. Skip pair.")
            feats_prev = feats_t1
            continue

        Hp, Wp, C = feats_t.shape
        assert C == Cdim, "특징 차원이 일관되지 않습니다."

        # 상단 절반 유효 마스크 (현재 프레임 기준)
        top_mask = build_top_half_valid_mask(H_t, W_t, Hpad_t, Wpad_t, patch=patch_size, device=device)
        # 유효 영역 이외는 선택에서 제외하기 위해 mask 이용
        valid_idx = top_mask.view(-1).nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            print(f"[Warn] No valid top-half tokens at frame {idx}. Skip EMA update.")
            feats_prev = feats_t1
            continue

        # 5) 동일 위치 cosine distance
        # sim(i,j) = dot(f_t, f_t1)  -> dist = 1 - sim
        sim_map = (feats_t * feats_t1).sum(dim=-1)  # (Hp, Wp)
        dist_map = 1.0 - sim_map  # (Hp, Wp)

        # 상단 절반 유효 영역만 펼쳐서 하위 5% 선택
        dist_vec_valid = dist_map.view(-1)[valid_idx]  # (M,)
        M = dist_vec_valid.numel()
        K = max(1, math.ceil(M * (low_pct / 100.0)))

        # 거리 작은 K개 인덱스 (ascending)
        vals, topk_idx_local = torch.topk(dist_vec_valid, k=K, largest=False)
        chosen_flat_idx = valid_idx[topk_idx_local]  # 원래 플랫 인덱스

        # 관측 벡터: t+1 프레임의 선택 패치 평균
        feats_t1_flat = feats_t1.view(-1, C)
        obs = feats_t1_flat.index_select(0, chosen_flat_idx)  # (K, C)
        obs = obs.mean(dim=0)  # (C,)
        obs = l2_normalize(obs, dim=0)

        # 6) EMA 업데이트
        if ema_vec is None:
            ema_vec = obs.clone()
        else:
            ema_vec = ema_decay * ema_vec + (1.0 - ema_decay) * obs
            ema_vec = l2_normalize(ema_vec, dim=0)

        feats_prev = feats_t1  # 슬라이딩 윈도우 갱신

        if (idx + 1) % 10 == 0 or (idx + 1) == (len(img_paths) - 1):
            print(f"[Info] Processed pair {idx}->{idx+1}, EMA updated. K={K}, mean dist={float(vals.mean()):.4f}")

    if ema_vec is None:
        raise RuntimeError("EMA가 생성되지 않았습니다. 유효한 프레임 쌍이 없었을 수 있습니다.")

    # 7) 첫 프레임에서 ema와 최유사 패치 위치 검색 (상단 절반 권장)
    # 첫 프레임 특징은 feats_prev가 아님에 주의. 다시 계산/사용.
    feats_first = extract_dino_patch_tokens(model, x0, patch_size)
    feats_first = l2_normalize(feats_first, dim=-1)  # (Hp0, Wp0, C)
    Hp0, Wp0, _ = feats_first.shape

    ema_vec_ = ema_vec.view(1, 1, -1)  # (1,1,C)
    sim0 = (feats_first * ema_vec_).sum(dim=-1)  # (Hp0, Wp0), cosine similarity
    # 상단 절반만 고려 (명시적 요구는 선택 단계였지만, 하늘 위치를 찾는 목적이므로 동일 제약 적용 권장)
    sim0_masked = sim0.clone()
    sim0_masked[~top_mask_first] = -1e9  # 배제

    best_idx = torch.argmax(sim0_masked.view(-1)).item()
    best_r = best_idx // Wp0
    best_c = best_idx % Wp0

    # 패치 중심 기준 원본 좌표 (패딩 전)
    best_x = int(best_c * patch_size + patch_size // 2)
    best_y = int(best_r * patch_size + patch_size // 2)
    best_x = min(best_x, W0 - 1)
    best_y = min(best_y, H0 - 1)

    print("\n========== RESULT ==========")
    print(f"Best patch (row, col) on first frame: ({best_r}, {best_c})  [grid {Hp0}x{Wp0}]")
    print(f"Pixel coordinate (approx. patch center): (x={best_x}, y={best_y}) within original size (W={W0}, H={H0})")
    print(f"Cosine similarity at best location: {float(sim0[best_r, best_c]):.6f}")
    print("============================\n")

    return {
        "grid_rc": (best_r, best_c),
        "pixel_xy": (best_x, best_y),
        "similarity": float(sim0[best_r, best_c]),
        "grid_size": (Hp0, Wp0),
        "first_image": img_paths[0],
    }

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Find sky-like stable patch with DINO v1 ViT-S/8 over a sequence.")
    ap.add_argument("--image_dir", type=str,default='/home/jaejun/dataset/MS2/sync_data/_2021-08-13-22-36-41/rgb/img_left', help="입력 시퀀스 이미지 폴더 경로")
    ap.add_argument("--ema_decay", type=float, default=0.9, help="EMA decay (0~1, 클수록 과거 가중↑)")
    ap.add_argument("--low_pct", type=float, default=5.0, help="프레임 쌍 distance 하위 백분율(%)")
    ap.add_argument("--patch_size", type=int, default=8, help="DINO 패치 크기(여기서는 8)")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    _ = run(
        image_dir=args.image_dir,
        ema_decay=args.ema_decay,
        low_pct=args.low_pct,
        patch_size=args.patch_size,
    )
