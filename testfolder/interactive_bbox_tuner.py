# interactive_bbox_tuner.py
# -*- coding: utf-8 -*-
"""
이미지 위에서 바운딩 박스를 클릭으로 지정하고,
해당 영역에만 실시간으로 효과(대비/채도/CLAHE)를 적용.
화살표 키로 강도 조절, s로 저장, m으로 모드 전환.

사용법:
    python interactive_bbox_tuner.py --image /path/to/image.jpg
또는:
    python interactive_bbox_tuner.py /path/to/image.jpg
"""

import cv2
import numpy as np
import argparse
import os
from datetime import datetime

# ----- 유틸 -----
def clamp(arr, low=0, high=255):
    return np.clip(arr, low, high).astype(np.uint8)

def sorted_box(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

# ----- 효과 적용 -----
def apply_contrast_roi(roi_bgr, alpha=1.2, beta=0.0):
    # alpha: 대비(곱), beta: 밝기(더하기, 0~±100 권장)
    out = roi_bgr.astype(np.float32) * float(alpha) + float(beta)
    return clamp(out)

def apply_saturation_roi(roi_bgr, sat_scale=1.2, beta=0.0):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    # 채도 증감
    hsv[..., 1] *= float(sat_scale)
    # 밝기 보정은 V 채널에 더하기
    hsv[..., 2] += float(beta)
    # 범위 클램프
    hsv[..., 0] = np.clip(hsv[..., 0], 0, 179)
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_clahe_roi(roi_bgr, clip_limit=2.0, beta=0.0):
    # LAB의 L 채널에 CLAHE 적용
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # clipLimit: 1.0~8.0 권장
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR).astype(np.float32)
    out += float(beta)  # 전체 밝기 미세 조정
    return clamp(out)

# ----- 상태 및 콜백 -----
class BBoxEditor:
    def __init__(self, img):
        self.img = img
        self.h, self.w = img.shape[:2]
        self.points = []            # [(x1,y1), (x2,y2)]
        self.bbox = None            # (x1,y1,x2,y2)
        self.mode = 'contrast'      # 'contrast' | 'saturation' | 'clahe'
        self.strength = 1.2         # 대비/채도/CLAHE 강도 공용
        self.brightness = 0.0       # 추가 밝기(상/하 방향키)
        self.step = 0.05            # 강도 증감 단위
        self.b_step = 2.0           # 밝기 증감 단위
        self.window = "Interactive BBox Tuner"
        self.help_lines = [
            "왼쪽 클릭 두 번: 박스(왼쪽 위 → 오른쪽 아래)",
            "좌/우 화살표: 강도 - / +",
            "상/하 화살표: 밝기 - / +",
            "m: 모드 전환 (Contrast ↔ Saturation ↔ CLAHE)",
            "r: 박스 초기화, s: 저장, q 또는 ESC: 종료",
        ]
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window, self.on_mouse)

    def on_mouse(self, event, x, y, flags, userdata=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) == 2:
                # 이미 지정된 경우 새로 시작
                self.points = []
                self.bbox = None
            self.points.append((x, y))
            if len(self.points) == 2:
                x1, y1, x2, y2 = sorted_box(self.points[0], self.points[1])
                # 최소 크기 보정
                if x2 == x1: x2 = min(x1 + 1, self.w - 1)
                if y2 == y1: y2 = min(y1 + 1, self.h - 1)
                self.bbox = (x1, y1, x2, y2)

    def cycle_mode(self):
        order = ['contrast', 'saturation', 'clahe']
        i = order.index(self.mode)
        self.mode = order[(i + 1) % len(order)]
        # 모드별 초기 추천값
        if self.mode == 'contrast':
            self.strength = 1.2  # alpha
        elif self.mode == 'saturation':
            self.strength = 1.3  # sat scale
        elif self.mode == 'clahe':
            self.strength = 2.0  # clipLimit
        # 밝기는 유지

    def adjust_strength(self, delta):
        # 모드별 적절한 범위
        if self.mode == 'contrast':
            self.strength = float(np.clip(self.strength + delta, 0.1, 5.0))
        elif self.mode == 'saturation':
            self.strength = float(np.clip(self.strength + delta, 0.0, 4.0))
        elif self.mode == 'clahe':
            self.strength = float(np.clip(self.strength + delta, 0.5, 8.0))

    def adjust_brightness(self, delta):
        self.brightness = float(np.clip(self.brightness + delta, -100.0, 100.0))

    def render(self):
        # 원본 기준으로 매 프레임 재합성(누적 적용 방지)
        vis = self.img.copy()

        # 박스가 있으면 ROI만 효과 적용
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            roi = vis[y1:y2, x1:x2]

            if self.mode == 'contrast':
                proc = apply_contrast_roi(roi, alpha=self.strength, beta=self.brightness)
            elif self.mode == 'saturation':
                proc = apply_saturation_roi(roi, sat_scale=self.strength, beta=self.brightness)
            elif self.mode == 'clahe':
                proc = apply_clahe_roi(roi, clip_limit=self.strength, beta=self.brightness)
            else:
                proc = roi

            vis[y1:y2, x1:x2] = proc
            # 박스 테두리
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)

        # 안내 텍스트 오버레이
        self.draw_hud(vis)
        cv2.imshow(self.window, vis)
        return vis

    def draw_hud(self, img):
        pad = 8
        lines = [
            f"Mode: {self.mode.upper()} | Strength: {self.strength:.2f} | Brightness: {self.brightness:.1f}",
        ] + self.help_lines
        # 반투명 패널
        block_h = 20 * len(lines) + pad * 2
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (min(560, img.shape[1]), block_h), (0, 0, 0), -1)
        img[:] = cv2.addWeighted(overlay, 0.45, img, 0.55, 0.0)

        y = pad + 16
        for i, t in enumerate(lines):
            cv2.putText(img, t, (pad, y + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser(description="Interactive BBox Contrast/Saturation/CLAHE Tuner")
    parser.add_argument("image", nargs="?", help="이미지 경로")
    parser.add_argument("--image", help="이미지 경로 (동일)", dest="image_opt")
    args = parser.parse_args()

    image_path = args.image_opt or args.image
    if not image_path or not os.path.exists(image_path):
        print("[!] 이미지 경로를 확인하세요.")
        return

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print("[!] 이미지를 불러올 수 없습니다.")
        return

    editor = BBoxEditor(img=img)

    # 화살표 키 코드(플랫폼별 대응)
    KEY_LEFTS  = {81, 2424832}
    KEY_RIGHTS = {83, 2555904}
    KEY_UPS    = {82, 2490368}
    KEY_DOWNS  = {84, 2621440}

    # 대체 키(WASD)
    ALT_LEFT, ALT_RIGHT, ALT_UP, ALT_DOWN = ord('a'), ord('d'), ord('w'), ord('s')

    last_vis = None
    while True:
        last_vis = editor.render()
        key = cv2.waitKeyEx(30)

        if key == -1:
            continue
        # 종료
        if key in (27, ord('q')):  # ESC or 'q'
            break
        # 모드 전환
        if key in (ord('m'), ord('M')):
            editor.cycle_mode()
            continue
        # 박스 리셋
        if key in (ord('r'), ord('R')):
            editor.points = []
            editor.bbox = None
            continue
        # 저장
        if key in (ord('c'), ord('C')):
            # 현재 화면 저장(오버레이 포함)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = os.path.splitext(os.path.basename(image_path))[0]
            out_path = f"{base}_interactive_{ts}.png"
            if last_vis is not None:
                cv2.imwrite(out_path, last_vis)
                print(f"[+] 저장: {out_path}")
            continue

        # 강도/밝기 조절
        if key in KEY_LEFTS or key == ALT_LEFT:
            editor.adjust_strength(-editor.step)
        elif key in KEY_RIGHTS or key == ALT_RIGHT:
            editor.adjust_strength(+editor.step)
        elif key in KEY_UPS or key == ALT_UP:
            editor.adjust_brightness(+editor.b_step)
        elif key in KEY_DOWNS or key == ALT_DOWN:
            editor.adjust_brightness(-editor.b_step)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
