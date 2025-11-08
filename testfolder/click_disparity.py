import argparse
import cv2
import numpy as np

def to_display_bgr(img):
    """
    이미지를 화면 표시용 BGR(8-bit)로 변환.
    - 8/16-bit 그레이스케일/컬러 모두 지원
    - 16-bit이면 단순 선형 스케일로 8-bit 변환
    """
    if img is None:
        raise ValueError("이미지를 읽지 못했습니다.")

    if img.ndim == 2:  # grayscale
        if img.dtype != np.uint8:
            vmax = int(img.max()) if img.max() > 0 else 1
            disp = cv2.convertScaleAbs(img, alpha=255.0 / vmax)
        else:
            disp = img.copy()
        return cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

    # color
    if img.dtype != np.uint8:
        vmax = int(img.max()) if img.max() > 0 else 1
        disp = cv2.convertScaleAbs(img, alpha=255.0 / vmax)
        # cv2.imread 색상 순서는 이미 BGR
        return disp
    return img.copy()

def main(left_path, right_path):
    # 원본은 변경하지 않기 위해 UNCHANGED로 읽음(16-bit 등 유지)
    left_raw  = cv2.imread(left_path,  cv2.IMREAD_UNCHANGED)
    right_raw = cv2.imread(right_path, cv2.IMREAD_UNCHANGED)
    if left_raw is None or right_raw is None:
        raise SystemExit("이미지 경로를 확인하세요.")

    # 해상도 체크
    if left_raw.shape[:2] != right_raw.shape[:2]:
        raise SystemExit(f"해상도가 다릅니다: left={left_raw.shape[:2]}, right={right_raw.shape[:2]}")

    # 화면 표시용 8-bit BGR로 변환
    left_disp  = to_display_bgr(left_raw)
    right_disp = to_display_bgr(right_raw)

    # 인터랙션을 위한 상태
    state = {"left_pt": None, "right_pt": None, "pair_idx": 0}
    visL = left_disp.copy()
    visR = right_disp.copy()

    def redraw():
        # 항상 원본 표시이미지에서 다시 그린다
        nonlocal visL, visR
        visL = left_disp.copy()
        visR = right_disp.copy()
        if state["left_pt"] is not None:
            x, y = state["left_pt"]
            cv2.circle(visL, (x, y), 4, (255, 255, 255), -1)
            cv2.line(visL, (0, y), (visL.shape[1]-1, y), (255, 255, 255), 1)
        if state["right_pt"] is not None:
            x, y = state["right_pt"]
            cv2.circle(visR, (x, y), 4, (255, 255, 255), -1)
            cv2.line(visR, (0, y), (visR.shape[1]-1, y), (255, 255, 255), 1)

    def on_mouse_left(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["left_pt"] = (int(x), int(y))
            print("left pixel : ", state["left_pt"])
            redraw()

    def on_mouse_right(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["right_pt"] = (int(x), int(y))
            print("right pixel : ", state["right_pt"])
            redraw()

    cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Right", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Left", on_mouse_left)
    cv2.setMouseCallback("Right", on_mouse_right)

    print("Left 창에서 한 점 클릭 → Right 창에서 대응점 클릭")
    print("키보드: r=리셋, q=종료")

    while True:
        # 두 점이 모두 선택되면 disparity 계산
        if state["left_pt"] is not None and state["right_pt"] is not None:
            uL, vL = state["left_pt"]
            uR, vR = state["right_pt"]
            disp = float(uL - uR)  # 왼쪽 기준 disparity
            dv = float(vL - vR)    # 세로 오차(정렬 확인용)

            msg = f"[{state['pair_idx']}] disparity = uL - uR = {disp:.2f} px   (|Δv|={abs(dv):.2f})"
            print(msg)

            # 왼쪽 이미지에 결과 간단히 표시
            org = (10, 30 + 25 * (state["pair_idx"] % 10))
            cv2.putText(visL, msg, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # 다음 측정을 위해 포인트 리셋(표시문자는 남김)
            state["pair_idx"] += 1
            state["left_pt"] = None
            state["right_pt"] = None

        cv2.imshow("Left", visL)
        cv2.imshow("Right", visR)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:  # q 또는 ESC
            break
        elif key == ord('r'):
            state["left_pt"] = None
            state["right_pt"] = None
            state["pair_idx"] = 0
            redraw()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Click corresponding points to measure disparity (uL - uR).")
    parser.add_argument("--left", default="/home/jaejun/dataset/MS2/sync_data/_2021-08-13-22-36-41/rgb/img_left/000000.png", help="Left image path")
    parser.add_argument("--right", default="/home/jaejun/dataset/MS2/sync_data/_2021-08-13-22-36-41/rgb/img_right/000000.png", help="Right image path")
    args = parser.parse_args()
    main(args.left, args.right)
