import numpy as np
import os, glob, argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from stereo_dpt import DINOv1Base8Backbone, StereoDPTHead, DPTStereoTrainCompat
from modeljj import StereoModel, assert_multiple

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# -----------------------------
# Sky mask from near-max disparity
# -----------------------------


def compute_sky_mask_from_disp(disp_px: np.ndarray,
                               max_disp_px: float,
                               thr_px: float = 3.0,
                               vmax_ratio: float = 0.6,
                               morph_kernel: int = 11,
                               min_area: int = 1000) -> np.ndarray:
    """
    disp_px:     (H,W) float32, 이 파일에서 사용하는 'disp_px_out' (선택 스케일/업샘플 반영)
    max_disp_px: 체크포인트에서 읽은 최대 시차(px)
    thr_px:      sky 후보 임계. disp >= max_disp_px - thr_px 인 곳을 후보로 잡음
    vmax_ratio:  컨투어의 y_max < H * vmax_ratio 여야 sky 로 인정 (기본 0.6)
    morph_kernel:모폴로지 클로징 커널 크기(odd 권장). 후보 영역을 한 덩어리로 묶음
    min_area:    너무 작은 잡영역 제거(픽셀 수)

    반환: (H,W) uint8 마스크 (sky=255, else=0)
    """
    if not HAS_CV2:
        return None
    
    
    
    H, W = disp_px.shape[:2]
    # 1) near-max 이진화
    thr_val = float(max_disp_px) - float(thr_px)
    bin0 = (disp_px >= thr_val).astype(np.uint8) * 255  # 0/255

    # 2) 모폴로지 클로징으로 "하나로 묶기"
    k = max(3, int(morph_kernel))
    if k % 2 == 0: k += 1  # odd 권장
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bin1 = cv2.morphologyEx(bin0, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 3) 컨투어 추출 (OpenCV 버전 호환)
    _cont = cv2.findContours(bin1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = _cont[0] if len(_cont) == 2 else _cont[1]

    # 4) 조건(y_min ≤ 1, y_max < H*ratio, 면적) 만족하는 컨투어만 채우기
    sky_mask = np.zeros((H, W), dtype=np.uint8)
    y_max_lim = int(H * float(vmax_ratio))

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < float(min_area):
            continue
        ys = cnt[:, :, 1].reshape(-1)
        y_min = int(ys.min()) if ys.size else H
        y_max = int(ys.max()) if ys.size else -1

        if (y_min <= 1) and (y_max < y_max_lim):
            cv2.drawContours(sky_mask, [cnt], contourIdx=-1, color=255, thickness=-1)

    return sky_mask
