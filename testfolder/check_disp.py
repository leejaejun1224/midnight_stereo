from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

fx = 764.5138549804688
baseline_m = 0.29918420530585865

raw = np.array(Image.open("/home/jaejun/dataset/MS2/proj_depth/tester/rgb/depth_filtered/000100.png")).astype(np.float32)
depth_m = raw / 256.0                        # mm -> m
depth_m[(raw <= 0) | ~np.isfinite(raw)] = np.nan

disp_px = (fx * baseline_m) / np.clip(depth_m, 1e-6, None)  # depth->disparity
disp_px[~np.isfinite(disp_px)] = np.nan

# (보기용) splat
def splat(arr, r=1):
    if r <= 0: return arr
    H, W = arr.shape
    out = np.full_like(arr, np.nan, np.float32)
    ys, xs = np.where(np.isfinite(arr))
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y-r), min(H, y+r+1)
        x0, x1 = max(0, x-r), min(W, x+r+1)
        out[y0:y1, x0:x1] = arr[y, x]
    return out

disp_vis = splat(disp_px, r=1)

# 표시 범위(유효 1~99%)
valid = np.isfinite(disp_vis)
vmin = float(np.nanpercentile(disp_vis, 1.0)) if valid.any() else 0.0
vmax = float(np.nanpercentile(disp_vis, 99.0)) if valid.any() else 1.0
if vmax <= vmin: vmax = vmin + 1e-3

plt.figure(figsize=(12,4))
im = plt.imshow(disp_vis, vmin=vmin, vmax=vmax)      # 기본 colormap 사용
plt.title("GT Disparity (pixels) — splatted points")
plt.axis("off")
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label("disparity (px)")
plt.tight_layout()
plt.savefig("gt_disparity_pixels_splatted.png", dpi=150)
plt.close()
