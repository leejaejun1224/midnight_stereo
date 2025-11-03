#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DINO v1 ViT-S/8 기반 패치 유사도 마스크 생성 도구 (원본 해상도 유지)

사용법:
  python dino_patch_mask_native.py /path/to/images --threshold 0.7 --out-dir ./masks

조작:
  - 마우스 클릭: 기준 패치 선택
  - ↑ / ↓: threshold ±0.05
  - n: 현재 선택 결과 저장 후 다음 이미지로
  - c: 현재 이미지 선택 초기화
  - q: 종료
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

# ------------------------------
# 유틸
# ------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_images(folder: Path):
    files = []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            files.append(p)
    return files

def load_dino_vits8(device: torch.device):
    """
    facebookresearch/dino 의 torch.hub 엔트리포인트 사용.
    - 모델: dino_vits8 (ViT-Small, patch size 8)
    """
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    model.eval()
    model.to(device)
    return model

def get_patch_tokens(model, x: torch.Tensor):
    """
    x: (1,3,H,W) 텐서 (정규화 포함)
    반환: (N, C) 패치 토큰 (cls 토큰 제거)
    """
    if hasattr(model, "get_intermediate_layers"):
        with torch.no_grad():
            outs = model.get_intermediate_layers(x, n=1)
            feat = outs[-1] if isinstance(outs, (list, tuple)) else outs
            # feat: (B, 1+N, C) -> cls 제외
            tokens = feat[:, 1:, :].squeeze(0).contiguous()
            return tokens  # (N, C)
    else:
        raise RuntimeError(
            "모델에 get_intermediate_layers가 없습니다. "
            "torch.hub의 facebookresearch/dino:main, dino_vits8 을 사용하세요."
        )

def minmax_norm(arr: np.ndarray, eps: float = 1e-8):
    a_min, a_max = arr.min(), arr.max()
    return (arr - a_min) / (a_max - a_min + eps)

def expand_to_pixels(patch_map: np.ndarray, patch_size: int):
    """
    (Hp, Wp) 패치 맵을 (Hp*ps, Wp*ps) 픽셀 맵으로 최근접 복제
    """
    return np.repeat(np.repeat(patch_map, patch_size, axis=0), patch_size, axis=1)

def save_binary_mask(mask_01: np.ndarray, out_path: Path):
    """
    mask_01: (H,W) 0/1 uint8 배열을 PNG로 저장
    """
    if mask_01.dtype != np.uint8:
        mask_01 = mask_01.astype(np.uint8)
    Image.fromarray(mask_01, mode="L").save(out_path)

def pad_to_multiple_of(x_1chw: torch.Tensor, multiple: int = 8):
    """
    입력: (1, C, H, W)
    오른쪽/아래 방향으로만 replicate 패딩하여 H,W를 multiple의 배수로 맞춤.
    반환: x_pad, (pad_bottom, pad_right)
    """
    _, _, H, W = x_1chw.shape
    pad_r = (multiple - (W % multiple)) % multiple
    pad_b = (multiple - (H % multiple)) % multiple
    if pad_r == 0 and pad_b == 0:
        return x_1chw, (0, 0)
    x_pad = F.pad(x_1chw, (0, pad_r, 0, pad_b), mode='replicate')
    return x_pad, (pad_b, pad_r)

# ------------------------------
# 메인 로직
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="DINO v1 ViT-S/8 패치 유사도 기반 이진 마스크 생성 도구 (원본 해상도 유지)"
    )
    parser.add_argument("--image_dir", type=str, default="/home/jaejun/dataset/MS2/sync_data/tester/rgb/img_left", help="이미지 폴더 경로")
    parser.add_argument("--threshold", "-t", type=float, default=0.75, help="선택 임계값 (0~1, 기본 0.7)")
    parser.add_argument("--out-dir", type=str, default="./log/mask", help="마스크 저장 폴더(지정 시 여기에 저장). 미지정이면 원본 폴더.")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists() or not image_dir.is_dir():
        print(f"[오류] 이미지 폴더를 찾을 수 없습니다: {image_dir}")
        sys.exit(1)

    files = list_images(image_dir)
    if not files:
        print(f"[오류] 이미지가 없습니다: {image_dir}")
        sys.exit(1)

    out_base = Path(args.out_dir) if args.out_dir else None
    if out_base:
        out_base.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[정보] 장치: {device}")
    print("[정보] 모델 로딩 중 (facebookresearch/dino:main, dino_vits8)...")
    model = load_dino_vits8(device)

    # 리사이즈 없음: 원본 크기 그대로
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    patch_size = 8
    print(f"[정보] 총 {len(files)}개 이미지 처리. ↑/↓로 threshold 조절, 마우스 클릭, 'n' 저장/다음, 'c' 초기화, 'q' 종료.")

    for idx, f in enumerate(files, 1):
        # --------------------------
        # 이미지 준비 (원본 해상도 유지)
        # --------------------------
        img_pil = Image.open(f).convert("RGB")
        orig_w, orig_h = img_pil.size

        x = transform(img_pil).unsqueeze(0).to(device)  # (1,3,H,W) - 원본 크기
        # 패치 크기의 배수가 아니면 오른쪽/아래로 replicate 패딩
        x_pad, (pad_b, pad_r) = pad_to_multiple_of(x, multiple=patch_size)
        _, _, H_pad, W_pad = x_pad.shape
        Hp, Wp = H_pad // patch_size, W_pad // patch_size  # 패치 그리드

        with torch.no_grad():
            tokens = get_patch_tokens(model, x_pad)  # (N, C), N=Hp*Wp
            # 코사인 유사도 계산을 위해 L2 정규화 (CPU로 보관해도 충분)
            tokens = tokens.cpu().numpy()
            norms = np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-8
            tokens_unit = tokens / norms  # (N, C)

        # 상태
        state = {
            "threshold": float(args.threshold),
            "click_idx": None,       # 선택한 패치 인덱스
            "sim_norm": None,        # (Hp, Wp) 0-1 유사도 맵
        }

        # --------------------------
        # 시각화 (원본 해상도)
        # --------------------------
        fig, ax = plt.subplots()
        ax.imshow(img_pil)
        ax.set_axis_off()
        title_text = ax.set_title(
            f"[{idx}/{len(files)}] {f.name}\n"
            f"Click=select patch | ↑/↓ threshold={state['threshold']:.2f} | n=save&next | c=clear | q=quit",
            fontsize=10
        )

        # 오버레이(원본 크기)
        sim_im = ax.imshow(
            np.zeros((orig_h, orig_w), dtype=np.float32),
            vmin=0.0, vmax=1.0, alpha=0.45, cmap='viridis'
        )
        sel_outline_im = ax.imshow(
            np.zeros((orig_h, orig_w), dtype=np.float32),
            vmin=0.0, vmax=1.0, alpha=0.35, cmap='gray'
        )
        click_marker = ax.scatter([], [], s=30, marker='x')

        def update_overlay():
            """현재 state를 기반으로 화면 오버레이 업데이트 (항상 원본 해상도에 맞춤)"""
            if state["sim_norm"] is None:
                sim_im.set_data(np.zeros((orig_h, orig_w), dtype=np.float32))
                sel_outline_im.set_data(np.zeros((orig_h, orig_w), dtype=np.float32))
                click_marker.set_offsets(np.empty((0, 2)))
            else:
                # 유사도(0-1) -> 패치 그리드 -> 픽셀로 확장 (패딩 포함) -> 원본 영역으로 크롭
                sim_pix_pad = expand_to_pixels(state["sim_norm"], patch_size)  # (H_pad, W_pad)
                sim_pix = sim_pix_pad[:orig_h, :orig_w]
                sim_im.set_data(sim_pix)

                # threshold 이상인 패치 선택 (1), 나머지 (0)
                sel_patch = (state["sim_norm"] >= state["threshold"]).astype(np.float32)  # (Hp, Wp)
                sel_pix_pad = expand_to_pixels(sel_patch, patch_size)
                sel_pix = sel_pix_pad[:orig_h, :orig_w]
                sel_outline_im.set_data(sel_pix)

                # 클릭 패치 중심 좌표 (원본 기준)
                if state["click_idx"] is not None:
                    py, px = divmod(state["click_idx"], Wp)
                    cx = min(orig_w - 1, px * patch_size + patch_size / 2.0)
                    cy = min(orig_h - 1, py * patch_size + patch_size / 2.0)
                    click_marker.set_offsets(np.array([[cx, cy]]))

            title_text.set_text(
                f"[{idx}/{len(files)}] {f.name}\n"
                f"Click=select patch | ↑/↓ threshold={state['threshold']:.2f} | n=save&next | c=clear | q=quit"
            )
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes != ax:
                return
            if event.xdata is None or event.ydata is None:
                return
            # 클릭 좌표 -> 원본 픽셀 -> 패치 인덱스
            px_orig = int(np.clip(event.xdata, 0, orig_w - 1))
            py_orig = int(np.clip(event.ydata, 0, orig_h - 1))
            px = px_orig // patch_size
            py = py_orig // patch_size
            patch_idx = py * Wp + px  # 패딩 포함 그리드에서의 인덱스
            state["click_idx"] = int(patch_idx)

            # 기준 패치와 모든 패치의 cos sim
            q = tokens_unit[patch_idx]        # (C,)
            sims = (tokens_unit @ q)          # (N,)
            sims = sims.reshape(Hp, Wp)
            sim01 = minmax_norm(sims)
            state["sim_norm"] = sim01
            update_overlay()

        def on_key(event):
            if event.key in ("up", "down"):
                step = 0.05
                if event.key == "up":
                    state["threshold"] = float(np.clip(state["threshold"] + step, 0.0, 1.0))
                else:
                    state["threshold"] = float(np.clip(state["threshold"] - step, 0.0, 1.0))
                update_overlay()

            elif event.key == "c":
                state["click_idx"] = None
                state["sim_norm"] = None
                update_overlay()

            elif event.key == "n":
                if state["sim_norm"] is None:
                    print("[알림] 아직 패치를 선택하지 않았습니다. 마우스로 클릭해주세요.")
                    return
                sel_patch = (state["sim_norm"] >= state["threshold"]).astype(np.uint8)
                # 선택된 패치는 0, 나머지는 1
                patch_mask01 = np.zeros((Hp, Wp), dtype=np.uint8)
                patch_mask01[sel_patch == 1] = 1
                # 픽셀 해상도로 확장 (패딩 포함) 후 원본 영역으로 크롭
                pix_mask01_pad = expand_to_pixels(patch_mask01, patch_size)  # (H_pad, W_pad)
                pix_mask01 = pix_mask01_pad[:orig_h, :orig_w]*255.0  # (H, W)

                # 저장 경로
                out_dir = out_base if out_base else f.parent
                out_name = f.stem + "_mask.png"
                out_path = out_dir / out_name
                save_binary_mask(pix_mask01, out_path)
                print(f"[저장] {out_path}")
                plt.close(fig)  # 다음 이미지로

            elif event.key == "q":
                plt.close('all')
                print("[종료]")
                sys.exit(0)

        cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
        cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

        update_overlay()
        plt.tight_layout()
        plt.show()

    print("[완료] 모든 이미지를 처리했습니다.")

if __name__ == "__main__":
    main()
