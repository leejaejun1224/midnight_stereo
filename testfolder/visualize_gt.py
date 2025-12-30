# -*- coding: utf-8 -*-
import os
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 카메라 파라미터 ----
FOCAL_PX   = 764.5138549804688
BASELINE_M = 0.29918420530585865
FB = FOCAL_PX * BASELINE_M   # 네가 말한 그대로 fb / 픽셀값

def visualize_gt_points(gt_path, out_path, thr=0, query_coords=None):
    """
    gt_path: 원본 GT depth PNG 경로
    out_path: 결과 PNG 저장 경로
    thr    : 이 값보다 큰 픽셀을 'valid'로 간주 (기본 0)
    query_coords : (x, y) 또는 [(x1, y1), (x2, y2), ...] 형태의 좌표들 (0-based)

    valid 픽셀에 대해
        disp(px) = FOCAL_PX * BASELINE_M / 픽셀값
    을 계산해서 컬러로 점 찍어서 저장.

    + query_coords가 주어지면 각 좌표에서 가장 가까운 valid disparity 값을 찾아
      그 위치와 값(disp)을 출력하고, 그림 위에 X 표시로 찍어 줌.
    """
    # 1) PNG 로드 (float32)
    raw = np.array(Image.open(gt_path)).astype(np.float32)
    if raw.ndim == 3:
        # 혹시 3채널이면 첫 채널만 사용
        raw = raw[..., 0]

    H, W = raw.shape

    # 2) valid 마스크 (raw > thr)
    valid = raw > float(thr)
    ys, xs = np.where(valid)
    if xs.size == 0:
        print(f"[WARN] no valid gt points in {gt_path}")
        return

    # 3) disparity(px) 계산: fb / 픽셀값
    disp_vals = FB / raw[valid] * 256.0

    # 4) 시각화 (scatter + colormap)
    dpi = 200
    plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = plt.gca()
    ax.set_facecolor("black")

    vmin = float(np.nanmin(disp_vals))
    vmax = float(np.nanmax(disp_vals))
    sc = ax.scatter(
        xs, ys,
        s=1,
        c=disp_vals,
        cmap="magma",
        vmin=vmin,
        vmax=vmax,      # 원래 코드 그대로 유지
        marker=".",
        linewidths=0,
    )

    # ---------- 여기서부터 좌표 질의 기능 ----------
    if query_coords is not None:
        # (x, y) 한 쌍만 들어온 경우를 편하게 처리
        if isinstance(query_coords, tuple):
            query_coords = [query_coords]

        for (qx, qy) in query_coords:
            # 이미지 범위 체크
            if not (0 <= qx < W and 0 <= qy < H):
                print(
                    f"[WARN] {gt_path}: 입력 좌표 ({qx}, {qy}) 가 "
                    f"이미지 크기 (W={W}, H={H}) 범위를 벗어났습니다."
                )
                continue

            # 해당 픽셀이 valid 인지 확인
            if valid[qy, qx]:
                nx, ny = qx, qy
            else:
                # valid 픽셀들 중에서 (qx, qy) 와의 거리 최소인 픽셀 찾기
                d2 = (xs - qx) ** 2 + (ys - qy) ** 2
                idx = np.argmin(d2)
                nx, ny = int(xs[idx]), int(ys[idx])

            # 최근접 픽셀의 disparity 계산
            disp_here = FB / raw[ny, nx] * 256.0

            # print(
            #     f"[INFO] {gt_path}: 입력 좌표 ({qx}, {qy}) -> "
            #     f"최근접 valid 픽셀 ({nx}, {ny}), disparity(px) = {disp_here:.4f}"
            # )

            # 그림 위에도 X 표시로 시각화
            ax.scatter([nx], [ny], s=15, c="cyan", marker="x")

    # ----------------------------------------------

    ax.invert_yaxis()   # 이미지 좌표계와 맞추기
    ax.axis("off")

    cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
    cbar.set_label("GT disparity (px)", rotation=270, labelpad=10)

    plt.tight_layout(pad=0.0)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.0, facecolor="white")
    plt.close()
    print(f"[OK] saved: {out_path}")
    return vmax

def process_folder(in_dir, out_dir, thr=0, query_coords=None):
    """
    in_dir : GT PNG들이 있는 폴더
    out_dir: 결과를 저장할 새 폴더
    query_coords : 모든 이미지에 대해 검사할 좌표 리스트
                   (x, y) 또는 [(x1, y1), (x2, y2), ...]
    """
    in_dir  = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(in_dir.glob("*.png"))
    if len(png_files) == 0:
        print(f"[WARN] no *.png in {in_dir}")
        return
    
    vmaxlist = []
    for gt_path in png_files:
        out_path = out_dir / gt_path.name  # 같은 이름으로 저장
        vamx = visualize_gt_points(
                str(gt_path),
                str(out_path),
                thr=thr,
                query_coords=query_coords,
            )
        vmaxlist.append(vamx)

    print(vmaxlist)
    
if __name__ == "__main__":
    # 여기만 네 폴더에 맞게 바꿔서 쓰면 됨
    in_dir  = "/home/jaejun/dataset/MS2/proj_depth/tester/rgb/depth_filtered"
    out_dir = "./debug_gt_points_disp_262x182"   # 새 폴더

    # 조회할 좌표들 (x: 가로, y: 세로, 0-based index)
    # 예시) 한 점만 보고 싶으면 query_coords = (320, 240)
    # 여러 점이면 [(x1, y1), (x2, y2), ...] 형태
    query_coords = [(262, 182)]  # 필요에 맞게 수정

    process_folder(in_dir, out_dir, thr=0, query_coords=query_coords)
