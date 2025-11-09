from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def splat_for_viz(depth_m: np.ndarray, r: int = 1) -> np.ndarray:
    if r <= 0: return depth_m
    H, W = depth_m.shape
    out = np.full_like(depth_m, np.nan, dtype=np.float32)
    ys, xs = np.where(np.isfinite(depth_m))
    for y, x in zip(ys, xs):
        y0, y1 = max(0,y-r), min(H,y+r+1)
        x0, x1 = max(0,x-r), min(W,x+r+1)
        out[y0:y1, x0:x1] = depth_m[y, x]
    return out

# ---- 파일 경로 ----
img_path = "/home/jaejun/dataset/MS2/proj_depth/tester/rgb/depth_filtered/000096.png"   # 16-bit depth PNG
gt_depth_scale = 256.0            # mm → m (데이터에 맞게)
invalid_values = [0]               # 무효값(센티넬)

# ---- 로드 & 변환 ----
raw = np.array(Image.open(img_path)).astype(np.float32)
for iv in invalid_values:
    raw[raw == float(iv)] = 0.0

depth_m = raw / gt_depth_scale
depth_m[~np.isfinite(depth_m) | (depth_m <= 0)] = np.nan

# ---- 보기 좋게 splat ----
depth_vis = splat_for_viz(depth_m, r=1)

# ---- 표시 구간(유효 픽셀의 1~99%) ----
finite = np.isfinite(depth_vis)
vmin = float(np.nanpercentile(depth_vis, 1.0)) if finite.any() else 0.0
vmax = float(np.nanpercentile(depth_vis, 99.0)) if finite.any() else 1.0
if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
    vmin, vmax = 0.0, (float(np.nanmax(depth_vis)) if finite.any() else 1.0)

plt.figure(figsize=(12,4))
im = plt.imshow(depth_vis, vmin=vmin, vmax=vmax)  # colormap 기본 사용
plt.title("GT Depth (meters)")
plt.axis("off")
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label("Depth (m)")
plt.tight_layout()
plt.savefig("gt_depth_meters_splatted.png", dpi=150)
plt.close()
