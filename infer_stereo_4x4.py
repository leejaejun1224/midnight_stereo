# -*- coding: utf-8 -*-
"""
학습된 체크포인트로 좌/우 이미지(파일 또는 디렉토리)를 추론해 시차맵 저장.
- *_disp_patch.npy : 특징 격자 단위 시차 (H' x W'), H' = H/stride
- *_disp_px.npy    : 픽셀 단위 시차 (× stride 환산)
- (옵션) 입력 해상도 업샘플, 16bit PNG, overlay PNG
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

# train_stereo.py와 동일 디렉토리에 두고 아래를 임포트하세요.
from train_stereo_4x4 import StereoModel, build_feature_extractor, assert_multiple

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_image(path, H, W):
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((H, W), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return tfm(img).unsqueeze(0)  # [1,3,H,W]


def colorize_disp(disp_px, max_disp_px=64):
    if not HAS_CV2:
        return None
    x = np.clip(disp_px / float(max_disp_px), 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    cm = cv2.applyColorMap(x, cv2.COLORMAP_TURBO)  # BGR
    return cm


def save_uint16_png(path, disp_px, scale=256.0):
    disp_u16 = np.clip(disp_px * scale, 0, 65535).astype(np.uint16)
    Image.fromarray(disp_u16).save(path)


def save_color_and_overlay(left_img_path: str,
                           disp_px: np.ndarray,
                           out_dir: str,
                           name: str,
                           max_disp_px: int,
                           save_color: bool,
                           save_overlay: bool,
                           alpha: float = 0.35):
    """
    - disp_px: (h,w) float32 시차(픽셀 단위). 이 해상도가 원본과 달라도 됨.
    - left_img_path: 원본 좌측 이미지 경로 (overlay의 바탕으로 사용)
    """
    if not (save_color or save_overlay):
        return
    if not HAS_CV2:
        print("[WARN] OpenCV가 없어 color/overlay 저장을 건너뜁니다. --save_color/--save_overlay 무시됨.")
        return

    # 1) 원본 좌측 이미지 불러오기 (BGR)
    img0 = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
    if img0 is None:
        print(f"[WARN] 원본 이미지를 열 수 없어 overlay를 건너뜁니다: {left_img_path}")
        return
    H0, W0 = img0.shape[:2]

    # 2) 시차를 원본 해상도로 리사이즈 (이산 시차 보존을 위해 NEAREST 권장)
    disp_px_full = cv2.resize(disp_px, (W0, H0), interpolation=cv2.INTER_NEAREST)

    # 3) 컬러맵 생성 (TURBO)
    cm = colorize_disp(disp_px_full, max_disp_px=max_disp_px)  # BGR
    if cm is None:
        return

    # 4) 저장
    if save_color:
        cv2.imwrite(os.path.join(out_dir, f"{name}_disp_color.png"), cm)
    if save_overlay:
        a = max(0.0, min(1.0, float(alpha)))  # 클램프
        overlay = cv2.addWeighted(img0, 1.0 - a, cm, a, 0.0)  # 얕게 덮기
        cv2.imwrite(os.path.join(out_dir, f"{name}_disp_overlay.png"), overlay)


def match_files(left, right):
    lp = sorted(glob.glob(os.path.join(left, "*")))
    rp = sorted(glob.glob(os.path.join(right, "*")))
    assert len(lp) == len(rp) and len(lp) > 0, "좌/우 이미지 개수 불일치 또는 비어 있음"
    for a, b in zip(lp, rp):
        assert os.path.basename(a) == os.path.basename(b), f"파일명 다름: {a} vs {b}"
    return lp, rp


@torch.no_grad()
def run_one(model: StereoModel, left_path, right_path, args, device, stride: int):
    imgL = load_image(left_path, args.height, args.width).to(device)
    imgR = load_image(right_path, args.height, args.width).to(device)

    model.eval()
    # model.forward()는 (prob, disp_soft, aux) 반환. aux에 WTA가 들어있음.
    _, _, aux = model(imgL, imgR)
    disp_wta = aux["disp_wta"][0, 0].cpu().numpy()  # [H',W']  ← ArgMax 시차(격자 단위)

    # 픽셀 단위 변환: × stride (stride=4 or 8)
    disp_px_lo = disp_wta * float(stride)          # (H',W')

    if args.upsample_to_input:
        if HAS_CV2:
            disp_px = cv2.resize(disp_px_lo, (args.width, args.height), interpolation=cv2.INTER_NEAREST)
        else:
            disp_px = np.array(
                Image.fromarray(disp_px_lo).resize((args.width, args.height), resample=Image.NEAREST)
            )
    else:
        disp_px = disp_px_lo

    return disp_wta, disp_px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--left", type=str, required=True, help="좌 이미지 파일 또는 디렉터리")
    ap.add_argument("--right", type=str, required=True, help="우 이미지 파일 또는 디렉터리")
    ap.add_argument("--out_dir", type=str, required=True)

    # 입력 리사이즈(학습과 동일 전처리)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width",  type=int, default=1224)

    # 출력/옵션
    ap.add_argument("--upsample_to_input", action="store_true", help="격자 시차를 입력 해상도로 NEAREST 업샘플")
    ap.add_argument("--save_color", action="store_true", help="컬러맵 PNG 저장")
    ap.add_argument("--save_overlay", action="store_true", help="원본 위에 컬러맵 반투명 overlay 저장")
    ap.add_argument("--overlay_alpha", type=float, default=0.35, help="overlay 투명도 (0~1)")
    ap.add_argument("--save_uint16", action="store_true", help="16bit PNG 저장")
    ap.add_argument("--uint16_scale", type=float, default=256.0, help="16bit PNG 저장시 스케일")

    ap.add_argument("--save_npy", action="store_true", help="*.npy (patch, px) 저장")
    ap.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- 체크포인트 로드 ---
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ck_args = ckpt.get("args", {})  # train_stereo.py에서 저장한 argparse dict

    # 학습 시 설정 복원
    feat_type   = ck_args.get("feat_type", "vits8_1by4")  # "vits8_1by4" or "vits8_1by8"
    feat_layers = ck_args.get("feat_layers", 4)
    neck_mid_ch = ck_args.get("neck_mid_ch", 256)
    neck_out_ch = ck_args.get("neck_out_ch", 256)

    max_disp_px = ck_args.get("max_disp_px", 64)
    agg_ch      = ck_args.get("agg_ch", 32)
    agg_depth   = ck_args.get("agg_depth", 3)
    softarg_t   = ck_args.get("softarg_t", 0.9)
    norm        = ck_args.get("norm", "gn")

    # DINO(vits8) 토큰 추출은 패치 8을 요구 → 입력은 8의 배수 권장
    assert_multiple(args.height, 8, "height")
    assert_multiple(args.width,  8, "width")

    # --- 특징 추출기 / 모델 구성 (학습과 동일) ---
    feat_extractor = build_feature_extractor(
        kind=feat_type,
        n_layers=feat_layers,
        neck_mid_ch=neck_mid_ch,
        neck_out_ch=neck_out_ch,
        dino_patch=8,
        c_vit=384
    )

    model = StereoModel(
        feat_extractor=feat_extractor,
        max_disp_px=max_disp_px,
        agg_base_ch=agg_ch,
        agg_depth=agg_depth,
        softarg_t=softarg_t,
        norm=norm
    ).to(device)

    # 파라미터 로드
    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except Exception as e:
        print("[WARN] strict=True 로드 실패, strict=False로 재시도합니다.\n", str(e))
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print("  - missing keys:", missing)
        print("  - unexpected keys:", unexpected)

    model.eval()
    stride = getattr(model, "stride", 8)
    D = max_disp_px // stride
    print(f"[Info] Restored model: feat_type={feat_type}, stride={stride}, D={D}, max_disp_px={max_disp_px}")

    # --- 입력 목록 구성 ---
    if os.path.isdir(args.left):
        left_files, right_files = match_files(args.left, args.right)
    else:
        left_files, right_files = [args.left], [args.right]

    # --- 추론 ---
    with torch.no_grad():
        for lp, rp in zip(left_files, right_files):
            name = os.path.splitext(os.path.basename(lp))[0]
            disp_patch_wta, disp_px = run_one(model, lp, rp, args, device, stride=stride)

            # 저장
            if args.save_npy:
                np.save(os.path.join(args.out_dir, f"{name}_disp_patch.npy"),
                        disp_patch_wta.astype(np.float32))
                np.save(os.path.join(args.out_dir, f"{name}_disp_px.npy"),
                        disp_px.astype(np.float32))

            if args.save_uint16:
                save_uint16_png(os.path.join(args.out_dir, f"{name}_disp16.png"),
                                disp_px, scale=args.uint16_scale)

            if args.save_color or args.save_overlay:
                save_color_and_overlay(
                    left_img_path=lp,
                    disp_px=disp_px,
                    out_dir=args.out_dir,
                    name=name,
                    max_disp_px=max_disp_px,
                    save_color=args.save_color,
                    save_overlay=args.save_overlay,
                    alpha=args.overlay_alpha
                )

            print(f"[OK] {name} -> saved in {args.out_dir}")


if __name__ == "__main__":
    main()
