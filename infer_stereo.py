# -*- coding: utf-8 -*-
"""
학습된 체크포인트로 좌/우 이미지(파일 또는 디렉토리)를 추론해 시차맵 저장.

출력(.npy):
- *_disp_patch.npy     : 패치 단위 시차 (H/8 x W/8)  ← ArgMax(WTA)
- *_disp_px_lo.npy     : 픽셀 단위 시차(풀해상도 픽셀 기준), H/8 그리드 (disp_patch * patch_size)
- *_disp_px_half.npy   : 픽셀 단위 시차(풀해상도 픽셀 기준), H/2 그리드 (convex upsample 결과 × 2)

시각화/저장 대상 선택:
- --pred_scale {half,lo}로 선택한 시차를 기반으로 다음 옵션 수행
  - (옵션) 입력 해상도 업샘플(--upsample_to_input)
  - (옵션) 16bit PNG 저장(--save_uint16)
  - (옵션) 컬러맵 PNG 저장(--save_color)
      · 색상 범위 고정: [--color_min, --color_max] 기본 [0,40]
      · 오른쪽에 세로 컬러바 + 숫자 눈금 표시
  - (옵션) 원본 좌측 이미지에 반투명 overlay 저장(--save_overlay)  ← 기존 동작 유지(컬러바/숫자 없음)

추가(adhoc):
- --roi_fix 1: 입력 해상도 기준 ROI(x1,y1,x2,y2) 안에서 0/최대 시차를 같은 행의 유효 시차 평균으로 채움.
- --mask_bad_white: ROI 안의 0/최대 시차 픽셀을 컬러 시차 이미지에서 '흰색'으로 칠함(별도 저장 없음).
"""

import os, glob, argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from stereo_dpt import DINOv1Base8Backbone, StereoDPTHead, DPTStereoTrainCompat
from modeljj import StereoModel, assert_multiple
# from modeljj_reassemble import StereoModel, assert_multiple
from outlier_masking import compute_sky_mask_from_disp
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_image(path, H, W):
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((H, W), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return tfm(img).unsqueeze(0)

# ---- overlay 전용 (컬러바 없음) ----
def colorize_disp(disp_px, max_disp_px=64):
    if not HAS_CV2: return None
    x = np.clip(disp_px / float(max_disp_px), 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    return cv2.applyColorMap(x, cv2.COLORMAP_TURBO)  # BGR

def _to_8bit_norm(arr: np.ndarray, vmin: float, vmax: float) -> np.uint8:
    eps = 1e-8
    r = (arr - vmin) / max(vmax - vmin, eps)
    r = np.clip(r, 0.0, 1.0)
    return (r * 255.0).astype(np.uint8)

# ---- 컬러바 포함 컬러맵 (여기에 마스킹 반영 추가) ----
def colorize_with_vertical_bar(disp_px: np.ndarray,
                               vmin: float = 0.0,
                               vmax: float = 64.0,
                               pad: int = 8,
                               bar_width: int = 24,
                               label_width: int = 48,
                               tick_step: float = 8.0,
                               tick_count: int = 9,
                               decimals: int = 0,
                               font_scale: float = 0.5,
                               thickness: int = 1,
                               colormap: int = cv2.COLORMAP_TURBO,
                               mask_white: np.ndarray = None) -> np.ndarray:
    """
    disp_px:    (H,W) float32 disparity (full-res px)
    mask_white: (H,W) bool or uint8, True(>0)인 위치는 본체 영역을 '흰색'으로 칠함
    반환: (H, W + pad + bar_width + label_width, 3) BGR
    """
    if not HAS_CV2:
        return None

    H, W = disp_px.shape[:2]
    gray_img = _to_8bit_norm(disp_px, vmin, vmax)                # (H,W)
    cm_img   = cv2.applyColorMap(gray_img, colormap)             # (H,W,3) BGR

    # --- 여기서 '흰색 masking' 반영 ---
    if mask_white is not None:
        m = mask_white.astype(bool)
        if m.shape != (H, W):
            # 입력 해상도에 맞게 최근접 리사이즈
            m = cv2.resize(mask_white.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        cm_img[m] = (255, 255, 255)

    # 세로 컬러바와 라벨
    grad  = np.linspace(vmax, vmin, num=H, dtype=np.float32)[:, None]
    grad8 = _to_8bit_norm(grad, vmin, vmax)
    bar   = cv2.applyColorMap(grad8, colormap)
    bar   = np.repeat(bar, repeats=max(1, int(bar_width)), axis=1)

    label_canvas = np.full((H, max(0, int(label_width)), 3), 255, dtype=np.uint8)
    bar_full = np.concatenate([bar, label_canvas], axis=1)
    bw = bar_width; lw = label_canvas.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_color = (0,0,0); text_color = (0,0,0)

    if (tick_step is not None) and (tick_step > 0):
        ticks = np.arange(vmin, vmax + 1e-6, tick_step, dtype=np.float32)
    else:
        tick_count = max(2, int(tick_count) if tick_count is not None else 9)
        ticks = np.linspace(vmin, vmax, num=tick_count, dtype=np.float32)
    fmt = "{:." + str(max(0, int(decimals))) + "f}"

    for t in ticks:
        r = (t - vmin) / max(vmax - vmin, 1e-8)
        y = int(round((1.0 - r) * (H - 1)))
        y = max(0, min(H - 1, y))
        tick_len = min(8, max(4, lw // 4))
        x0 = bw - 1; x1 = bw + tick_len
        cv2.line(bar_full, (x0, y), (x1, y), line_color, max(1, int(thickness)))
        label = fmt.format(float(t))
        (tw, th), base = cv2.getTextSize(label, font, font_scale, thickness)
        tx = bw + 6; ty = int(np.clip(y + th // 2, th, H - 1))
        cv2.putText(bar_full, label, (tx, ty), font, font_scale, text_color, max(1, int(thickness)), cv2.LINE_AA)

    if pad > 0:
        pad_img = np.zeros((H, pad, 3), dtype=np.uint8)
        out = np.concatenate([cm_img, pad_img, bar_full], axis=1)
    else:
        out = np.concatenate([cm_img, bar_full], axis=1)
    return out

def save_uint16_png(path, disp_px, scale=256.0):
    disp_u16 = np.clip(disp_px * scale, 0, 65535).astype(np.uint16)
    Image.fromarray(disp_u16).save(path)

# ---- overlay 저장(여기에 마스킹 반영 추가) ----
def save_color_and_overlay(left_img_path: str,
                           disp_px: np.ndarray,
                           out_dir: str,
                           name: str,
                           max_disp_px: int,
                           save_color: bool,
                           save_overlay: bool,
                           alpha: float = 0.35,
                           mask_white: np.ndarray = None):
    """
    overlay 전용(컬러바 없음). mask_white가 주어지면 본체 컬러맵에서 흰색으로 칠함.
    """
    if not (save_color or save_overlay):
        return
    if not HAS_CV2:
        print("[WARN] OpenCV 없음 → overlay 건너뜀")
        return

    img0 = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
    if img0 is None:
        print(f"[WARN] 원본 이미지를 열 수 없어 overlay를 건너뜁니다: {left_img_path}")
        return
    H0, W0 = img0.shape[:2]

    disp_px_full = cv2.resize(disp_px, (W0, H0), interpolation=cv2.INTER_NEAREST)
    cm = colorize_disp(disp_px_full, max_disp_px=max_disp_px)  # BGR

    # --- 흰색 마스킹 반영 ---
    if mask_white is not None:
        m = mask_white.astype(bool)
        if m.shape != (H0, W0):
            m = cv2.resize(mask_white.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST) > 0
        cm[m] = (255, 255, 255)

    if save_color and cm is not None:
        cv2.imwrite(os.path.join(out_dir, f"{name}_disp_color.png"), cm)

    if save_overlay and cm is not None:
        a = max(0.0, min(1.0, float(alpha)))
        overlay = cv2.addWeighted(img0, 1.0 - a, cm, a, 0.0)
        cv2.imwrite(os.path.join(out_dir, f"{name}_disp_overlay.png"), overlay)

def match_files(left, right):
    lp = sorted(glob.glob(os.path.join(left, "*")))
    rp = sorted(glob.glob(os.path.join(right, "*")))
    assert len(lp) == len(rp) and len(lp) > 0, "좌/우 이미지 수 불일치 또는 비어 있음"
    for a, b in zip(lp, rp):
        assert os.path.basename(a) == os.path.basename(b), f"파일명 다름: {a} vs {b}"
    return lp, rp

# -----------------------------
# ADHOC: 도로 ROI 행 평균 치환 + bad mask 생성
# -----------------------------
def _parse_xyxy(s: str):
    try:
        xs = [int(v.strip()) for v in s.split(",")]
        if len(xs) != 4: raise ValueError
        x1, y1, x2, y2 = xs
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        return x1, y1, x2, y2
    except Exception:
        raise ValueError(f"--roi_xyxy 형식 오류: '{s}' (예: 200,200,1000,384)")

def _map_roi_to_disp(res_in, res_disp, xyxy):
    W_in, H_in = res_in
    W_d, H_d   = res_disp
    x1, y1, x2, y2 = xyxy
    x1 = int(np.floor(np.clip(x1 * (W_d / float(W_in)), 0, W_d - 1)))
    x2 = int(np.ceil (np.clip(x2 * (W_d / float(W_in)), 0, W_d    )))
    y1 = int(np.floor(np.clip(y1 * (H_d / float(H_in)), 0, H_d - 1)))
    y2 = int(np.ceil (np.clip(y2 * (H_d / float(H_in)), 0, H_d    )))
    x2 = max(x1 + 1, x2)
    y2 = max(y1 + 1, y2)
    return x1, y1, x2, y2

def _compute_badmask_in_roi(disp: np.ndarray,
                            roi_xyxy_in_input: tuple,
                            input_size: tuple,
                            max_disp: float) -> np.ndarray:
    """ROI 내 (0 또는 >= max_disp) 위치 True인 full 해상도 불리언 마스크"""
    H_d, W_d = disp.shape[:2]
    x1, y1, x2, y2 = _map_roi_to_disp(input_size, (W_d, H_d), roi_xyxy_in_input)
    eps0 = 1e-6; epsM = 1e-3
    bad_roi = (disp[y1:y2, x1:x2] <= eps0) | (disp[y1:y2, x1:x2] >= (max_disp - epsM))
    bad_full = np.zeros_like(disp, dtype=bool)
    bad_full[y1:y2, x1:x2] = bad_roi
    return bad_full

def _robust_mode_1d(vals: np.ndarray, max_disp: float, bin_width: float = 0.5) -> float:
    """
    연속값 배열에서 히스토그램 기반 최빈값을 구하고,
    최빈 bin 안의 값들 '중앙값'을 반환 (노이즈/꼬리 완화).
    vals: 1D float array (필터된 유효값만 넣는 걸 권장)
    max_disp: 히스토그램 상한
    bin_width: bin 폭(px). 너무 작으면 빈 bin 증가, 너무 크면 분해능 저하.
    """
    if vals.size == 0:
        return float("nan")
    bw = max(1e-6, float(bin_width))
    num_bins = max(1, int(np.ceil(max_disp / bw)))
    # 히스토그램
    hist, edges = np.histogram(vals, bins=num_bins, range=(0.0, max_disp))
    if hist.sum() == 0:
        return float("nan")
    k = int(hist.argmax())
    lo, hi = edges[k], edges[k+1]
    in_bin = (vals >= lo) & (vals < hi) if k < len(edges)-2 else (vals >= lo) & (vals <= hi)
    if not np.any(in_bin):
        # 드물게 bin이 비는 경우, bin 중앙 반환
        return float(0.5*(lo+hi))
    return float(np.median(vals[in_bin]))

def _row_mode_fill_roi(disp: np.ndarray,
                       roi_xyxy_in_input: tuple,
                       input_size: tuple,
                       max_disp: float,
                       search_radius: int = 5,
                       bin_width: float = 0.5) -> tuple:
    """
    ROI 내 0 또는 >= max_disp 픽셀을 '행 단위 최빈값'으로 채움.
    우선순위: (1) ROI 같은 행의 최빈값 → (2) 전체 행 최빈값 → (3) 주변 행(±r) 최빈값 → (4) ROI 전체 최빈값
    Returns:
      (disp_filled, stats_dict, badmask_full)
    """
    H_d, W_d = disp.shape[:2]
    x1, y1, x2, y2 = _map_roi_to_disp(input_size, (W_d, H_d), roi_xyxy_in_input)

    roi = disp[y1:y2, x1:x2]
    if roi.size == 0:
        badmask_full = np.zeros_like(disp, dtype=bool)
        return disp, {"roi": (x1,y1,x2,y2), "changed": 0, "total_bad": 0}, badmask_full

    eps0 = 0.5; epsM = 3
    bad_mask = (roi <= eps0) | (roi >= (max_disp - epsM))
    total_bad = int(bad_mask.sum())

    badmask_full = np.zeros_like(disp, dtype=bool)
    badmask_full[y1:y2, x1:x2] = bad_mask

    if total_bad == 0:
        return disp, {"roi": (x1,y1,x2,y2), "changed": 0, "total_bad": 0}, badmask_full

    out = disp.copy()
    roi_out = out[y1:y2, x1:x2]

    # 미리 ROI 전체 유효값의 모드도 계산 (최후의 보루)
    valid_roi_all = (roi > eps0) & (roi < (max_disp - epsM))
    roi_all_mode = _robust_mode_1d(roi[valid_roi_all].ravel(), max_disp, bin_width) if np.any(valid_roi_all) else float("nan")

    changed = 0
    for r in range(roi.shape[0]):
        bad = bad_mask[r]
        if not np.any(bad):
            continue

        y_abs = y1 + r
        row_roi = roi[r]
        valid_row_roi = (~bad) & (row_roi > eps0) & (row_roi < (max_disp - epsM))

        fill_val = float("nan")

        # (1) ROI 같은 행 최빈값
        if np.any(valid_row_roi):
            fill_val = _robust_mode_1d(row_roi[valid_row_roi], max_disp, bin_width)
        else:
            # (2) 전체 행 최빈값
            row_full = out[y_abs, :]
            valid_full = (row_full > eps0) & (row_full < (max_disp - epsM))
            if np.any(valid_full):
                fill_val = _robust_mode_1d(row_full[valid_full], max_disp, bin_width)
            else:
                # (3) 주변 행 최빈값
                for d in range(1, search_radius + 1):
                    up = y_abs - d
                    dn = y_abs + d
                    if up >= 0:
                        rf = out[up, :]; vf = (rf > eps0) & (rf < (max_disp - epsM))
                        if np.any(vf):
                            fill_val = _robust_mode_1d(rf[vf], max_disp, bin_width); break
                    if dn < H_d:
                        rf = out[dn, :]; vf = (rf > eps0) & (rf < (max_disp - epsM))
                        if np.any(vf):
                            fill_val = _robust_mode_1d(rf[vf], max_disp, bin_width); break
                # (4) ROI 전체 최빈값
                if not np.isfinite(fill_val) and np.any(valid_roi_all):
                    fill_val = roi_all_mode

        if not np.isfinite(fill_val):
            # 정말 대체할 값이 없으면 스킵
            continue

        roi_out[r, bad] = fill_val
        changed += int(bad.sum())

    out[y1:y2, x1:x2] = roi_out
    return out, {"roi": (x1,y1,x2,y2), "changed": changed, "total_bad": total_bad}, badmask_full


def _row_mean_fill_roi(disp: np.ndarray,
                       roi_xyxy_in_input: tuple,
                       input_size: tuple,
                       max_disp: float,
                       search_radius: int = 5) -> tuple:
    """
    Returns:
      (disp_filled, stats_dict, badmask_full)
    """
    H_d, W_d = disp.shape[:2]
    x1, y1, x2, y2 = _map_roi_to_disp(input_size, (W_d, H_d), roi_xyxy_in_input)

    roi = disp[y1:y2, x1:x2]
    if roi.size == 0:
        badmask_full = np.zeros_like(disp, dtype=bool)
        return disp, {"roi": (x1,y1,x2,y2), "changed": 0, "total_bad": 0}, badmask_full

    eps0 = 1; epsM = 1
    bad_mask = (roi <= eps0) | (roi >= (max_disp - epsM))
    total_bad = int(bad_mask.sum())

    badmask_full = np.zeros_like(disp, dtype=bool)
    badmask_full[y1:y2, x1:x2] = bad_mask

    if total_bad == 0:
        return disp, {"roi": (x1,y1,x2,y2), "changed": 0, "total_bad": 0}, badmask_full

    out = disp.copy()
    roi_out = out[y1:y2, x1:x2]

    changed = 0
    for r in range(roi.shape[0]):
        bad = bad_mask[r]
        if not np.any(bad):
            continue
        y_abs = y1 + r
        row_roi = roi[r]
        valid_roi = (~bad) & (row_roi > eps0) & (row_roi < (max_disp - epsM))
        if np.any(valid_roi):
            mean_val = float(row_roi[valid_roi].mean())
        else:
            row_full = out[y_abs, :]
            valid_full = (row_full > eps0) & (row_full < (max_disp - epsM))
            if np.any(valid_full):
                mean_val = float(row_full[valid_full].mean())
            else:
                mean_val = None
                for d in range(1, search_radius + 1):
                    up = y_abs - d
                    dn = y_abs + d
                    if up >= 0:
                        rf = out[up, :]; vf = (rf > eps0) & (rf < (max_disp - epsM))
                        if np.any(vf): mean_val = float(rf[vf].mean()); break
                    if dn < H_d:
                        rf = out[dn, :]; vf = (rf > eps0) & (rf < (max_disp - epsM))
                        if np.any(vf): mean_val = float(rf[vf].mean()); break
                if mean_val is None:
                    vr = (roi > eps0) & (roi < (max_disp - epsM))
                    if np.any(vr): mean_val = float(roi[vr].mean())
                    else: continue
        roi_out[r, bad] = mean_val
        changed += int(bad.sum())

    out[y1:y2, x1:x2] = roi_out
    return out, {"roi": (x1,y1,x2,y2), "changed": changed, "total_bad": total_bad}, badmask_full

@torch.no_grad()
def run_one(model, left_path, right_path, args, device, patch_size, max_disp_px):
    imgL = load_image(left_path, args.height, args.width).to(device)
    imgR = load_image(right_path, args.height, args.width).to(device)
    model.eval()

    _, _, aux = model(imgL, imgR)

    disp_patch_wta = aux["disp_wta"][0, 0].cpu().numpy().astype(np.float32)
    disp_px_lo = (disp_patch_wta * float(patch_size)).astype(np.float32)

    disp_half_px_halfunit = aux["disp_half_px"][0, 0].cpu().numpy().astype(np.float32)
    disp_px_half = (disp_half_px_halfunit * 2.0).astype(np.float32)

    if args.pred_scale == "half":
        disp_px_base = disp_px_half
    else:
        disp_px_base = disp_px_lo

    if args.upsample_to_input:
        if HAS_CV2:
            disp_px_out = cv2.resize(disp_px_base, (args.width, args.height), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        else:
            disp_px_out = np.array(Image.fromarray(disp_px_base).resize((args.width, args.height), resample=Image.NEAREST)).astype(np.float32)
    else:
        disp_px_out = disp_px_base

    # ---- ROI 보정 및 bad mask 생성 ----
    badmask_full = None
    x1, y1, x2, y2 = _parse_xyxy(args.roi_xyxy)
    if args.roi_fix:
        disp_px_out, stats, badmask_full = _row_mode_fill_roi(
            disp=disp_px_out,
            roi_xyxy_in_input=(x1, y1, x2, y2),
            input_size=(args.width, args.height),
            max_disp=max_disp_px,
            search_radius=5,
            bin_width=args.mode_bin_width
        )
        print(f"[ROI-FIX/MODE] rect={stats['roi']}  changed={stats['changed']}/{stats['total_bad']}")
    elif args.mask_bad_white:
        badmask_full = _compute_badmask_in_roi(
            disp=disp_px_out,
            roi_xyxy_in_input=(x1, y1, x2, y2),
            input_size=(args.width, args.height),
            max_disp=max_disp_px
        )

    return {
        "disp_patch": disp_patch_wta,
        "disp_px_lo": disp_px_lo,
        "disp_px_half": disp_px_half,
        "disp_px_out": disp_px_out,
        "badmask_full": badmask_full,   # 추가
        "max_disp_px": max_disp_px,
        "patch_size": patch_size,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--left_dir", type=str, required=True, help="좌 이미지 파일 또는 디렉터리")
    ap.add_argument("--right_dir", type=str, required=True, help="우 이미지 파일 또는 디렉터리")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width",  type=int, default=1224)
    ap.add_argument("--patch_size", type=int, default=8)

    ap.add_argument("--pred_scale", type=str, default="half", choices=["half","lo"])
    ap.add_argument("--upsample_to_input", action="store_true")
    ap.add_argument("--save_color", action="store_true")
    ap.add_argument("--save_uint16", action="store_true")
    ap.add_argument("--uint16_scale", type=float, default=256.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_overlay", action="store_true")
    ap.add_argument("--overlay_alpha", type=float, default=0.35)
    ap.add_argument("--no_save_npy", action="store_true")

    # colorbar options
    ap.add_argument("--color_min", type=float, default=0.0)
    ap.add_argument("--color_max", type=float, default=40.0)
    ap.add_argument("--colorbar_width", type=int, default=24)
    ap.add_argument("--colorbar_pad", type=int, default=8)
    ap.add_argument("--colorbar_label_width", type=int, default=48)
    ap.add_argument("--colorbar_tick_step", type=float, default=8.0)
    ap.add_argument("--colorbar_tick_count", type=int, default=9)
    ap.add_argument("--colorbar_decimals", type=int, default=0)
    ap.add_argument("--colorbar_font_scale", type=float, default=0.5)
    ap.add_argument("--colorbar_thickness", type=int, default=1)
    ap.add_argument("--mode_bin_width", type=float, default=1.0,
                help="최빈값 히스토그램 bin 폭(px)")

    # ADHOC ROI & mask-to-white
    ap.add_argument("--roi_fix", type=int, default=0, help="도로 ROI 내 0/최대 시차를 행 평균으로 치환 (1:활성, 0:비활성)")
    ap.add_argument("--roi_xyxy", type=str, default="200,300,1000,376", help="입력 해상도 기준 ROI: x1,y1,x2,y2")
    ap.add_argument("--mask_bad_white", action="store_true",
                    help="ROI 내 0/최대 시차 픽셀을 컬러 시차 이미지에서 '흰색'으로 칠함")
    
    
    # Sky mask 저장 옵션
    ap.add_argument("--save_skymask", action="store_true",
                    help="near-max disparity 기반 sky 마스크({name}_skymask.png) 저장")
    ap.add_argument("--sky_thr", type=float, default=3.0,
                    help="disp >= (max_disp - sky_thr) 를 sky 후보로 간주 (px)")
    ap.add_argument("--sky_kernel", type=int, default=11,
                    help="모폴로지 클로징 커널 크기 (odd 권장)")
    ap.add_argument("--sky_min_area", type=int, default=1,
                    help="sky 컨투어 최소 면적(픽셀)")
    ap.add_argument("--sky_vmax_ratio", type=float, default=0.7,
                    help="컨투어 y_max < H*ratio 여야 sky 인정 (또한 y_min ≤ 1 이어야 함)")


    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ck_args = ckpt.get("args", {})
    max_disp_px = ck_args.get("max_disp_px", 40)
    patch_size  = ck_args.get("patch_size", args.patch_size)
    agg_ch      = ck_args.get("agg_ch", 32)
    agg_depth   = ck_args.get("agg_depth", 3)
    softarg_t   = ck_args.get("softarg_t", 0.9)
    norm        = ck_args.get("norm", "gn")

    assert_multiple(args.height, patch_size, "height")
    assert_multiple(args.width,  patch_size, "width")

    model = StereoModel(max_disp_px=max_disp_px, patch_size=patch_size,
                        agg_base_ch=agg_ch, agg_depth=agg_depth,
                        softarg_t=softarg_t, norm=norm).to(device)
    # DPT 대안 사용 시: 주석 해제
    # model = DPTStereoTrainCompat(max_disp_px=max_disp_px, patch_size=patch_size,
    #                              feat_dim=256, readout='project', embed_dim=768, temperature=0.7).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    if os.path.isdir(args.left_dir):
        left_files, right_files = match_files(args.left_dir, args.right_dir)
    else:
        left_files, right_files = [args.left_dir], [args.right_dir]

    for lp, rp in zip(left_files, right_files):
        name = os.path.splitext(os.path.basename(lp))[0]
        out = run_one(model, lp, rp, args, device, patch_size, max_disp_px)

        # npy 저장
        if not args.no_save_npy:
            np.save(os.path.join(args.out_dir, f"{name}_disp_patch.npy"), out["disp_patch"].astype(np.float32))
            np.save(os.path.join(args.out_dir, f"{name}_disp_px_lo.npy"), out["disp_px_lo"].astype(np.float32))
            np.save(os.path.join(args.out_dir, f"{name}_disp_px_half.npy"), out["disp_px_half"].astype(np.float32))

        # 16bit 저장
        if args.save_uint16:
            save_uint16_png(os.path.join(args.out_dir, f"{name}_disp16.png"), out["disp_px_out"], scale=args.uint16_scale)

        # 컬러맵 PNG (컬러바 포함) — 여기서 흰색 마스킹 반영
        if args.save_color and HAS_CV2:
            cm_with_bar = colorize_with_vertical_bar(
                out["disp_px_out"],
                vmin=args.color_min, vmax=args.color_max,
                pad=args.colorbar_pad, bar_width=args.colorbar_width, label_width=args.colorbar_label_width,
                tick_step=args.colorbar_tick_step, tick_count=args.colorbar_tick_count,
                decimals=args.colorbar_decimals, font_scale=args.colorbar_font_scale,
                thickness=args.colorbar_thickness, colormap=cv2.COLORMAP_TURBO,
                mask_white=(out["badmask_full"] if args.mask_bad_white else None)
            )
            if cm_with_bar is not None:
                cv2.imwrite(os.path.join(args.out_dir, f"{name}_disp_color.png"), cm_with_bar)

        # overlay 저장 (컬러바 없음) — 여기서도 흰색 마스킹 반영
        if args.save_overlay:
            save_color_and_overlay(
                left_img_path=lp,
                disp_px=out["disp_px_out"],
                out_dir=args.out_dir,
                name=name,
                max_disp_px=max_disp_px,
                save_color=False,
                save_overlay=True,
                alpha=args.overlay_alpha,
                mask_white=(out["badmask_full"] if args.mask_bad_white else None)
            )
            
        if args.save_skymask and HAS_CV2:
            sky_mask = compute_sky_mask_from_disp(
                out["disp_px_out"],             # 현재 선택/업샘플 반영된 시차 지도
                max_disp_px=max_disp_px,
                thr_px=args.sky_thr,
                vmax_ratio=args.sky_vmax_ratio,
                morph_kernel=args.sky_kernel,
                min_area=args.sky_min_area
            )
            if sky_mask is not None:
                cv2.imwrite(os.path.join(args.out_dir, f"{name}_skymask.png"), sky_mask)

        print(f"[OK] {name} -> saved in {args.out_dir} "
              f"(pred_scale={args.pred_scale}, up_to_input={args.upsample_to_input}, "
              f"mask_bad_white={'Y' if args.mask_bad_white else 'N'})")

if __name__ == "__main__":
    main()
