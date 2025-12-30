# -*- coding: utf-8 -*-
import os
import glob
import math
import argparse
from typing import Optional, Tuple, List, Dict

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ==== 프로젝트 유틸 (학습/추론 코드와 동일 모듈 재사용) ====
from tools import (
    StereoFolderDataset,
    denorm_imagenet,
    warp_right_to_left_image,
    PhotometricLoss,
    load_ms2_gt_depth_batch,
    compute_ms2_disparity_metrics,
)

# -------------------------------
# 공용 저장/시각화 유틸
# -------------------------------
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def _basename_wo_ext(path):
    return os.path.splitext(os.path.basename(path))[0]

def save_npy(path, np_array):
    _ensure_dir(os.path.dirname(path))
    np.save(path, np_array.astype(np.float32), allow_pickle=False)

def save_gray_png(path, np_array_uint8):
    from PIL import Image
    _ensure_dir(os.path.dirname(path))
    Image.fromarray(np_array_uint8.astype(np.uint8), mode="L").save(path)

def save_colormap_png_with_colorbar_auto_range(
    path, np_array, vmin: Optional[float] = None, vmax: Optional[float] = None,
    cmap_name: str = "magma", label: str = "", bg_color: str = "#1e1e1e"
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, to_rgba

    _ensure_dir(os.path.dirname(path))
    arr = np.array(np_array, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        vmin_eff = 0.0 if vmin is None else float(vmin)
        vmax_eff = 1.0 if vmax is None else float(vmax)
    else:
        vmin_eff = float(np.nanmin(arr[finite])) if vmin is None else float(vmin)
        vmax_eff = float(np.nanmax(arr[finite])) if vmax is None else float(vmax)
        vmax_eff = max(vmax_eff, vmin_eff + 1e-6)

    import numpy as _np
    base = plt.get_cmap(cmap_name)
    cmap = ListedColormap(base(_np.linspace(0,1,256)))
    cmap.set_bad(to_rgba(bg_color))

    H, W = arr.shape; dpi = 200.0; figsize = (W/dpi, H/dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(bg_color); ax.set_facecolor(bg_color)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin_eff, vmax=vmax_eff); ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if label: cbar.set_label(label, rotation=270, labelpad=12)
    plt.tight_layout(pad=0.1)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1, facecolor=fig.get_facecolor())
    plt.close(fig)

def annotate_png_top_left(path: str, text: str, margin: int = 5):
    from PIL import Image, ImageDraw, ImageFont
    try:
        img = Image.open(path).convert("RGBA")
    except Exception:
        return
    W, H = img.size
    fs = max(10, int(min(W, H) * 0.025))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", fs)
    except Exception:
        try: font = ImageFont.truetype("arial.ttf", fs)
        except Exception: font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)

    # 멀티라인 텍스트 크기 계산
    try:
        bbox = draw.multiline_textbbox((0,0), text, font=font, stroke_width=2, spacing=2)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        try:
            bbox = draw.textbbox((0,0), text, font=font, stroke_width=2)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        except Exception:
            tw, th = draw.textsize(text, font=font)

    bg = Image.new("RGBA", (tw+2*margin, th+2*margin), (0,0,0,100))
    img.paste(bg, (margin, margin), bg)
    draw.multiline_text((margin*2, margin*2), text, font=font,
                        fill=(255,255,255,255), stroke_width=2, stroke_fill=(0,0,0,255), spacing=2)
    img = img.convert("RGB"); img.save(path)

# -------------------------------
# Photometric error 계산 (시각화/마스킹용: L1+SSIM)
# -------------------------------
@torch.no_grad()
def compute_pth_error_map(imgL_01: torch.Tensor, imgR_01: torch.Tensor, disp_px: torch.Tensor,
                          w_l1=0.15, w_ssim=0.85) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    반환:
      pth_map: [1,1,H,W]  (invalid은 NaN)
      valid:   [1,1,H,W]  (0/1)
    disp_px는 주어진 이미지 해상도의 픽셀 단위여야 함(1/4일 때는 px@1/4).
    """
    imgR_warp, valid = warp_right_to_left_image(imgR_01, disp_px)  # [1,3,H,W], [1,1,H,W]
    pth = PhotometricLoss([w_l1, w_ssim]).simple_photometric_loss(imgL_01, imgR_warp, weights=[w_l1, w_ssim])
    pth = torch.where(valid>0.5, pth, torch.full_like(pth, float("nan")))
    return pth, valid

# -------------------------------
# 파일 매칭 유틸
# -------------------------------
def find_file_by_stem(folder: str, stem: str, exts: List[str]) -> Optional[str]:
    # 정확 이름 우선
    for ext in exts:
        cand = os.path.join(folder, f"{stem}{ext}")
        if os.path.isfile(cand): return cand
    # fallback: stem*ext
    for ext in exts:
        ms = glob.glob(os.path.join(folder, f"{stem}*{ext}"))
        if len(ms) > 0: return ms[0]
    return None

# -------------------------------
# 워핑 유틸 (그레이/컬러 공용)
# -------------------------------
def _warp_right_to_left_gray_cv2(right_gray_u8: np.ndarray, disp_px: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H, W = right_gray_u8.shape
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mapx = xs - disp_px.astype(np.float32)
    mapy = ys
    right_warp = cv2.remap(right_gray_u8, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = (mapx >= 0) & (mapx <= (W-1)) & (mapy >= 0) & (mapy <= (H-1))
    return right_warp, valid

def _warp_right_to_left_bgr_cv2(right_bgr_u8: np.ndarray, disp_px: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H, W, _ = right_bgr_u8.shape
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mapx = xs - disp_px.astype(np.float32)
    mapy = ys
    right_warp = cv2.remap(right_bgr_u8, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = (mapx >= 0) & (mapx <= (W-1)) & (mapy >= 0) & (mapy <= (H-1))
    return right_warp, valid

# -------------------------------
# Residual-SGM 핵심 유틸
# -------------------------------
def _build_sgbm_for_residual(r_px: float, block_size: int, uniqueness: int, speckle_win: int,
                             speckle_range: int, mode: str, use_color: bool = False):
    """
    r_px: 잔차 탐색 반경(px). SGBM은 numDisp가 16의 배수여야 하므로 반올림.
    use_color: True면 입력을 3채널로 넣고, P1/P2를 채널 수에 맞춰 스케일.
    """
    r = max(1, int(round(float(r_px))))
    min_disp = -r
    num_disp = int(np.ceil(2*r / 16.0) * 16)  # 2r을 16배수로
    if num_disp < 16:
        num_disp = 16

    block_size = int(block_size) if int(block_size) % 2 == 1 else int(block_size) + 1
    block_size = max(3, min(block_size, 11))

    # 채널 수에 맞춘 P1/P2 스케일
    cn = 3 if use_color else 1
    P1 = 8  * cn * (block_size ** 2)
    P2 = 32 * cn * (block_size ** 2)

    if mode.upper() == "HH":
        sgbm_mode = cv2.STEREO_SGBM_MODE_HH
    elif mode.upper() == "SGBM":
        sgbm_mode = cv2.STEREO_SGBM_MODE_SGBM
    else:
        sgbm_mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY

    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1, P2=P2,
        disp12MaxDiff=1,
        preFilterCap=31,
        uniquenessRatio=int(uniqueness),
        speckleWindowSize=int(speckle_win),
        speckleRange=int(speckle_range),
        mode=sgbm_mode
    )
    return matcher, min_disp, num_disp

def _apply_wls_generic_if_available(disp_px: np.ndarray, guide_bgr_u8: np.ndarray, lam: float, sigma_col: float) -> np.ndarray:
    if not (hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createDisparityWLSFilterGeneric")):
        return disp_px  # WLS generic 미존재 → 원본 반환
    wls = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
    wls.setLambda(float(lam))
    wls.setSigmaColor(float(sigma_col))
    d16 = np.clip(disp_px, 0, None).astype(np.float32) * 16.0
    d16 = np.clip(d16, -32768, 32767).astype(np.int16)
    out16 = wls.filter(d16, guide_bgr_u8)  # CV_16S
    out = (out16.astype(np.float32) / 16.0)
    out = np.clip(out, 0, None)
    return out

# -------------------------------
# EPE/D1 계산 (px@full-res 또는 px@scaled)
# -------------------------------
@torch.no_grad()
def compute_epe_d1_from_gt_full(
    pred_disp_full_px: torch.Tensor,  # [1,1,H,W]  px@full-res
    gt_depth_full_m: torch.Tensor,    # [1,1,H,W]
    focal_px: float, baseline_m: float
) -> Tuple[Optional[float], Optional[float]]:
    valid = (gt_depth_full_m > 0).float()
    if valid.sum() <= 0: return None, None
    gt_disp_full_px = (float(focal_px) * float(baseline_m)) / gt_depth_full_m.clamp_min(1e-6)
    m = compute_ms2_disparity_metrics(pred_disp_full_px, gt_disp_full_px, valid)
    epe = float(m.get("EPE", float("nan"))); d1 = float(m.get("D1_all", float("nan")))
    if not math.isfinite(epe): epe = None
    if not math.isfinite(d1): d1 = None
    return epe, d1

@torch.no_grad()
def compute_epe_d1_from_gt_scaled(
    pred_disp_scaled_px: torch.Tensor,   # [1,1,H,W]  px@scaled (예: 1/4)
    gt_depth_scaled_m: torch.Tensor,     # [1,1,H,W]
    focal_px: float, baseline_m: float,
    scale: float                         # = 0.25 이면 px@1/4
) -> Tuple[Optional[float], Optional[float]]:
    valid = (gt_depth_scaled_m > 0).float()
    if valid.sum() <= 0: return None, None
    fb = float(focal_px) * float(baseline_m)
    gt_disp_scaled_px = (fb / gt_depth_scaled_m.clamp_min(1e-6)) * float(scale)
    m = compute_ms2_disparity_metrics(pred_disp_scaled_px, gt_disp_scaled_px, valid)
    epe = float(m.get("EPE", float("nan"))); d1 = float(m.get("D1_all", float("nan")))
    if not math.isfinite(epe): epe = None
    if not math.isfinite(d1): d1 = None
    return epe, d1

# -------------------------------
# Depth metrics 계산 유틸 (AbsRel, SqRel, RMSE, LogRMSE, δ들)
# -------------------------------
@torch.no_grad()
def compute_depth_metrics_from_disp(
    pred_disp_full_px: torch.Tensor,   # [1,1,H,W]  px@full-res
    gt_depth_full_m: torch.Tensor,     # [1,1,H,W]
    focal_px: float, baseline_m: float
) -> Optional[Dict[str, float]]:
    if gt_depth_full_m is None or pred_disp_full_px is None:
        return None
    valid = (gt_depth_full_m > 0) & torch.isfinite(gt_depth_full_m) & torch.isfinite(pred_disp_full_px) & (pred_disp_full_px > 0)
    if valid.sum() == 0:
        return None
    fb = float(focal_px) * float(baseline_m)
    pred_depth = fb / pred_disp_full_px.clamp_min(1e-6)

    pd = pred_depth[valid].reshape(-1)
    gd = gt_depth_full_m[valid].reshape(-1)

    diff = pd - gd
    absrel = torch.mean(torch.abs(diff) / gd).item()
    sqrel  = torch.mean((diff ** 2) / gd).item()
    rmse   = torch.sqrt(torch.mean(diff ** 2)).item()

    eps = 1e-6
    rmselog = torch.sqrt(torch.mean((torch.log(pd + eps) - torch.log(gd + eps)) ** 2)).item()

    ratio = torch.maximum(pd / gd, gd / pd)
    d1 = torch.mean((ratio < (1.25)).float()).item()
    d2 = torch.mean((ratio < (1.25 ** 2)).float()).item()
    d3 = torch.mean((ratio < (1.25 ** 3)).float()).item()

    return {"AbsRel": absrel, "SqRel": sqrel, "RMSE": rmse, "LogRMSE": rmselog, "delta1": d1, "delta2": d2, "delta3": d3}

@torch.no_grad()
def compute_depth_metrics_from_disp_scaled(
    pred_disp_scaled_px: torch.Tensor,  # [1,1,H,W]  px@scaled (예: 1/4)
    gt_depth_scaled_m: torch.Tensor,    # [1,1,H,W]
    focal_px: float, baseline_m: float,
    scale: float                        # = 0.25 이면 px@1/4
) -> Optional[Dict[str, float]]:
    if gt_depth_scaled_m is None or pred_disp_scaled_px is None:
        return None
    valid = (gt_depth_scaled_m > 0) & torch.isfinite(gt_depth_scaled_m) & torch.isfinite(pred_disp_scaled_px) & (pred_disp_scaled_px > 0)
    if valid.sum() == 0:
        return None
    fb = float(focal_px) * float(baseline_m)
    # d_full = d_scaled / scale  → depth = fb / d_full = fb * scale / d_scaled
    pred_depth = (fb * float(scale)) / pred_disp_scaled_px.clamp_min(1e-6)

    pd = pred_depth[valid].reshape(-1)
    gd = gt_depth_scaled_m[valid].reshape(-1)

    diff = pd - gd
    absrel = torch.mean(torch.abs(diff) / gd).item()
    sqrel  = torch.mean((diff ** 2) / gd).item()
    rmse   = torch.sqrt(torch.mean(diff ** 2)).item()

    eps = 1e-6
    rmselog = torch.sqrt(torch.mean((torch.log(pd + eps) - torch.log(gd + eps)) ** 2)).item()

    ratio = torch.maximum(pd / gd, gd / pd)
    d1 = torch.mean((ratio < (1.25)).float()).item()
    d2 = torch.mean((ratio < (1.25 ** 2)).float()).item()
    d3 = torch.mean((ratio < (1.25 ** 3)).float()).item()

    return {"AbsRel": absrel, "SqRel": sqrel, "RMSE": rmse, "LogRMSE": rmselog, "delta1": d1, "delta2": d2, "delta3": d3}

# -------------------------------
# 요약 파일 쓰기 유틸
# -------------------------------
def _fmt(v: Optional[float], spec: str) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "N/A"
    try:
        return format(float(v), spec)
    except Exception:
        return str(v)

def _delta_str(bef: Optional[float], aft: Optional[float], spec: str) -> str:
    if (bef is None) or (aft is None) or any([isinstance(x, float) and (math.isnan(x) or math.isinf(x)) for x in [bef, aft]]):
        return "N/A"
    return f"{_fmt(bef,spec)} → {_fmt(aft,spec)} (Δ{_fmt(aft-bef, spec)})"

def write_metrics_summary(out_root: str, records: List[Dict[str, Dict[str, float]]], note_quarter: bool = False):
    if len(records) == 0:
        return
    path = os.path.join(out_root, "metrics_refine_summary.txt")
    keys = ["EPE_px", "D1_pct", "AbsRel", "SqRel", "RMSE_m", "LogRMSE", "delta1", "delta2", "delta3"]

    def collect_mean(k: str, which: str) -> Optional[float]:
        vals = []
        for r in records:
            v = r[which].get(k)
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                continue
            vals.append(float(v))
        if len(vals) == 0: return None
        return float(np.mean(vals))

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Residual-SGM refine metrics summary\n")
        f.write("# Columns: EPE(px), D1(%), AbsRel, SqRel, RMSE(m), LogRMSE, δ<1.25, δ<1.25^2, δ<1.25^3\n")
        if note_quarter:
            f.write("# NOTE: --half enabled → disparity metrics are in 'px @ 1/4-res' (full-res px × 0.25).\n\n")
        else:
            f.write("\n")

        for r in records:
            name = r.get("name", "unknown")
            bef, aft = r["before"], r["after"]
            line = (
                f"{name} | "
                f"EPE(px) {_delta_str(bef.get('EPE_px'), aft.get('EPE_px'), '.3f')} | "
                f"D1(%) {_delta_str(bef.get('D1_pct'), aft.get('D1_pct'), '.2f')} | "
                f"AbsRel {_delta_str(bef.get('AbsRel'), aft.get('AbsRel'), '.4f')} | "
                f"SqRel {_delta_str(bef.get('SqRel'),  aft.get('SqRel'),  '.4f')} | "
                f"RMSE(m) {_delta_str(bef.get('RMSE_m'), aft.get('RMSE_m'), '.3f')} | "
                f"LogRMSE {_delta_str(bef.get('LogRMSE'),aft.get('LogRMSE'),'.4f')} | "
                f"δ1 {_delta_str(bef.get('delta1'), aft.get('delta1'), '.4f')} | "
                f"δ2 {_delta_str(bef.get('delta2'), aft.get('delta2'), '.4f')} | "
                f"δ3 {_delta_str(bef.get('delta3'), aft.get('delta3'), '.4f')}\n"
            )
            f.write(line)

        # 평균
        f.write("\n# Averages over images (before → after, Δafter-before)\n")
        avg_bef = {k: collect_mean(k, "before") for k in keys}
        avg_aft = {k: collect_mean(k, "after")  for k in keys}

        def avg_line(label, b, a, spec): return f"{label} {_delta_str(b, a, spec)}\n"

        f.write(avg_line("EPE(px)", avg_bef["EPE_px"], avg_aft["EPE_px"], ".3f"))
        f.write(avg_line("D1(%)",   avg_bef["D1_pct"], avg_aft["D1_pct"], ".2f"))
        f.write(avg_line("AbsRel",  avg_bef["AbsRel"], avg_aft["AbsRel"], ".4f"))
        f.write(avg_line("SqRel",   avg_bef["SqRel"],  avg_aft["SqRel"],  ".4f"))
        f.write(avg_line("RMSE(m)", avg_bef["RMSE_m"], avg_aft["RMSE_m"], ".3f"))
        f.write(avg_line("LogRMSE", avg_bef["LogRMSE"], avg_aft["LogRMSE"], ".4f"))
        f.write(avg_line("δ1",      avg_bef["delta1"], avg_aft["delta1"], ".4f"))
        f.write(avg_line("δ2",      avg_bef["delta2"], avg_aft["delta2"], ".4f"))
        f.write(avg_line("δ3",      avg_bef["delta3"], avg_aft["delta3"], ".4f"))

    print(f"[Summary] metrics written → {path}")

# -------------------------------
# 메인 루프 (Residual-SGM 정제)
# -------------------------------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # 출력 폴더
    out_root     = _ensure_dir(args.output_dir)
    out_delta    = _ensure_dir(os.path.join(out_root, "delta"))
    out_disp_ref = _ensure_dir(os.path.join(out_root, "disp_refined"))
    out_pth_bef  = _ensure_dir(os.path.join(out_root, "pth_before"))
    out_pth_aft  = _ensure_dir(os.path.join(out_root, "pth_after"))

    # 데이터 로더(인퍼런스 전처리/리사이즈 동일)
    dataset = StereoFolderDataset(args.left_dir, args.right_dir, height=args.height, width=args.width)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    # Photometric error 계산기(시각화용)
    pth_loss = PhotometricLoss([args.pth_l1_w, args.pth_ssim_w])

    # SGBM 하이퍼(고정)
    matcher, min_disp, num_disp = _build_sgbm_for_residual(
        r_px=args.residual_px,
        block_size=args.block_size,
        uniqueness=args.uniqueness,
        speckle_win=args.speckle_win,
        speckle_range=args.speckle_range,
        mode=args.mode,
        use_color=bool(args.color)
    )

    # 전/후 메트릭 요약을 저장할 리스트
    all_records: List[Dict[str, Dict[str, float]]] = []

    # (실행) 스케일
    scale = 0.5 if args.half else 1.0
    unit_px = "px@1/4" if args.half else "px"

    for it, (imgL, imgR, names) in enumerate(loader, start=1):
        name = names[0] if isinstance(names, (list, tuple)) else names
        stem = _basename_wo_ext(name)

        # --- 초기 disparity 파일(NPY) 로드 ---
        disp_path = find_file_by_stem(args.disp_dir, stem, [".npy"])
        if disp_path is None:
            print(f"[Skip] init disp npy not found for stem={stem}")
            continue
        D0_full = np.load(disp_path).astype(np.float32)  # (H,W) px
        # 원 이미지 크기
        H_t, W_t = int(imgL.shape[-2]), int(imgL.shape[-1])
        # 실제 실행 해상도
        H_run = int(round(H_t * scale))
        W_run = int(round(W_t * scale))

        # --- 토치→넘파이 이미지 변환 (OpenCV용) + 스케일 반영 ---
        imgL01_full = denorm_imagenet(imgL.to(device))  # [1,3,H,W]
        imgR01_full = denorm_imagenet(imgR.to(device))
        if args.half:
            imgL01 = F.interpolate(imgL01_full, size=(H_run, W_run), mode="bilinear", align_corners=False)
            imgR01 = F.interpolate(imgR01_full, size=(H_run, W_run), mode="bilinear", align_corners=False)
        else:
            imgL01, imgR01 = imgL01_full, imgR01_full

        L_rgb = (imgL01[0].detach().cpu().permute(1,2,0).numpy() * 255.0).astype(np.uint8)
        R_rgb = (imgR01[0].detach().cpu().permute(1,2,0).numpy() * 255.0).astype(np.uint8)
        L_bgr = cv2.cvtColor(L_rgb, cv2.COLOR_RGB2BGR)
        R_bgr = cv2.cvtColor(R_rgb, cv2.COLOR_RGB2BGR)
        L_gray = cv2.cvtColor(L_bgr, cv2.COLOR_BGR2GRAY)
        R_gray = cv2.cvtColor(R_bgr, cv2.COLOR_BGR2GRAY)

        # --- 초기 D0를 실행 해상도에 맞춤(가로배율) + 스케일 값 반영 ---
        if D0_full.shape != (H_run, W_run):
            scale_x = W_run / D0_full.shape[1]
            D0_np = cv2.resize(D0_full, (W_run, H_run), interpolation=cv2.INTER_LINEAR) * scale_x
        else:
            scale_x = W_run / D0_full.shape[1]  # = 1.0 또는 0.25 등
            D0_np = D0_full * scale_x * 2.0

        # --- Photometric error (before) ---
        D0_t = torch.from_numpy(D0_np).to(device=device, dtype=torch.float32).view(1,1,H_run,W_run)
        pth_before, _ = compute_pth_error_map(imgL01, imgR01, D0_t, w_l1=args.pth_l1_w, w_ssim=args.pth_ssim_w)
        pth_bef_np = pth_before[0,0].detach().cpu().numpy()
        save_colormap_png_with_colorbar_auto_range(
            os.path.join(out_pth_bef, f"{stem}_pth_before.png"),
            pth_bef_np, vmin=0.0, vmax=None, cmap_name=args.err_cmap,
            label="Photometric error", bg_color=args.bg_color
        )

        # --- Residual-SGM: 오른쪽을 D0로 warp → 잔차 SGBM ---
        if args.color:
            Rw_bgr, _ = _warp_right_to_left_bgr_cv2(R_bgr, D0_np)  # (H_run,W_run,3)
            delta16 = matcher.compute(L_bgr, Rw_bgr)
        else:
            Rw_gray, _ = _warp_right_to_left_gray_cv2(R_gray, D0_np)  # (H_run,W_run)
            delta16 = matcher.compute(L_gray, Rw_gray)

        delta = (delta16.astype(np.float32) / 16.0)
        r = int(round(float(args.residual_px)))
        delta = np.clip(delta, -r, r).astype(np.float32)

        # --- 최종 D = D0 + Δ ---
        D_ref = (D0_np + delta).astype(np.float32)

        # (선택) WLS/FGS 정제
        if args.wls:
            D_ref = _apply_wls_generic_if_available(D_ref, L_bgr, args.wls_lambda, args.wls_sigma)
        if args.clip_nonneg:
            D_ref = np.clip(D_ref, 0, None)

        # --- 저장 ---
        save_npy(os.path.join(out_delta, f"{stem}_delta.npy"), delta)
        save_colormap_png_with_colorbar_auto_range(
            os.path.join(out_delta, f"{stem}_delta_cb.png"),
            delta, vmin=-r, vmax=+r, cmap_name="RdBu_r",
            label=f"Δ disparity ({unit_px})", bg_color=args.bg_color
        )

        save_npy(os.path.join(out_disp_ref, f"{stem}.npy"), D_ref)
        disp_png = os.path.join(out_disp_ref, f"{stem}_disp_refined_cb.png")
        save_colormap_png_with_colorbar_auto_range(
            disp_png, D_ref*2.0, vmin=0.0, vmax=args.vmax_disp,
            cmap_name=args.disp_cmap, label=f"Disparity ({unit_px})", bg_color=args.bg_color
        )

        # --- Photometric error (after) ---
        D_ref_t = torch.from_numpy(D_ref).to(device=device, dtype=torch.float32).view(1,1,H_run,W_run)
        pth_after, _ = compute_pth_error_map(imgL01, imgR01, D_ref_t, w_l1=args.pth_l1_w, w_ssim=args.pth_ssim_w)
        pth_after_np = pth_after[0,0].detach().cpu().numpy()
        pth_png_after = os.path.join(out_pth_aft, f"{stem}_pth_after.png")
        save_colormap_png_with_colorbar_auto_range(
            pth_png_after, pth_after_np, vmin=0.0, vmax=None,
            cmap_name=args.err_cmap, label="Photometric error", bg_color=args.bg_color
        )

        # --- (선택) GT 메트릭: 이미지별 GT가 있을 때만 계산 ---
        record = {"name": stem, "before": {}, "after": {}}
        if args.gt_depth_dir and args.focal_px > 0 and args.baseline_m > 0:
            has_gt = False
            gt_depth = None
            try:
                gt_depth = load_ms2_gt_depth_batch(
                    names=[name], gt_depth_dir=args.gt_depth_dir,
                    scale=args.gt_depth_scale, target_hw=(H_run, W_run), device=device
                )
                if gt_depth is not None:
                    valid_gt = torch.isfinite(gt_depth) & (gt_depth > 0)
                    has_gt = bool(valid_gt.sum().item() > 0)
            except Exception as _:
                has_gt = False

            if has_gt:
                # reshape or half → scaled 경로, 아니면 full 경로
                metrics_use_scaled = bool(args.reshape or args.half)
                scale_for_metrics = float(scale) if metrics_use_scaled else 1.0

                if metrics_use_scaled:
                    epe_bef, d1_bef = compute_epe_d1_from_gt_scaled(D0_t,   gt_depth, args.focal_px, args.baseline_m, scale_for_metrics)
                    epe_aft, d1_aft = compute_epe_d1_from_gt_scaled(D_ref_t, gt_depth, args.focal_px, args.baseline_m, scale_for_metrics)
                    depth_bef = compute_depth_metrics_from_disp_scaled(D0_t,   gt_depth, args.focal_px, args.baseline_m, scale_for_metrics)
                    depth_aft = compute_depth_metrics_from_disp_scaled(D_ref_t, gt_depth, args.focal_px, args.baseline_m, scale_for_metrics)
                else:
                    epe_bef, d1_bef = compute_epe_d1_from_gt_full(D0_t,   gt_depth, args.focal_px, args.baseline_m)
                    epe_aft, d1_aft = compute_epe_d1_from_gt_full(D_ref_t, gt_depth, args.focal_px, args.baseline_m)
                    depth_bef = compute_depth_metrics_from_disp(D0_t,   gt_depth, args.focal_px, args.baseline_m)
                    depth_aft = compute_depth_metrics_from_disp(D_ref_t, gt_depth, args.focal_px, args.baseline_m)

                record["before"].update({
                    "EPE_px": epe_bef,
                    "D1_pct": d1_bef,
                    "AbsRel": None if depth_bef is None else depth_bef["AbsRel"],
                    "SqRel":  None if depth_bef is None else depth_bef["SqRel"],
                    "RMSE_m": None if depth_bef is None else depth_bef["RMSE"],
                    "LogRMSE":None if depth_bef is None else depth_bef["LogRMSE"],
                    "delta1": None if depth_bef is None else depth_bef["delta1"],
                    "delta2": None if depth_bef is None else depth_bef["delta2"],
                    "delta3": None if depth_bef is None else depth_bef["delta3"],
                })
                record["after"].update({
                    "EPE_px": epe_aft,
                    "D1_pct": d1_aft,
                    "AbsRel": None if depth_aft is None else depth_aft["AbsRel"],
                    "SqRel":  None if depth_aft is None else depth_aft["SqRel"],
                    "RMSE_m": None if depth_aft is None else depth_aft["RMSE"],
                    "LogRMSE":None if depth_aft is None else depth_aft["LogRMSE"],
                    "delta1": None if depth_aft is None else depth_aft["delta1"],
                    "delta2": None if depth_aft is None else depth_aft["delta2"],
                    "delta3": None if depth_aft is None else depth_aft["delta3"],
                })
                all_records.append(record)
            else:
                # 이미지별 GT 없음 → 메트릭 계산/오버레이 모두 생략
                print(f"[Info] GT depth not found or empty for '{stem}', skipping metrics for this image.")
        else:
            # 전체적으로 GT 설정이 없는 경우에만 안내 오버레이
            annotate_png_top_left(disp_png, f"Metrics: N/A (no GT/focal/baseline)  |  unit={unit_px}")
            annotate_png_top_left(pth_png_after, f"Metrics: N/A (no GT/focal/baseline)  |  unit={unit_px}")

        print(f"[{it:04d}] residual-SGM refined & saved: {stem} (Δ in [-{r}, +{r}])  |  mode={'COLOR' if args.color else 'GRAY'}  |  scale={'1/4' if args.half else '1/1'}")

    # 요약 텍스트 저장 (GT가 있었던 이미지들만 포함)
    write_metrics_summary(out_root, all_records, note_quarter=bool(args.half))
    print(f"[Done] outputs → {out_root}")

# argparse
# -------------------------------
def get_args():
    p = argparse.ArgumentParser("Residual-SGM refine: warp right by D0 → SGBM on ±r → D = D0+Δ (WLS/FGS optional)")

    # 입력 경로
    p.add_argument("--left_dir",  type=str, required=True)
    p.add_argument("--right_dir", type=str, required=True)
    p.add_argument("--disp_dir",  type=str, required=True, help="*.npy (full-res px) initial disparity per image stem")

    # 데이터 크기(인퍼런스와 동일하게 맞추기)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--width",  type=int, default=1280)

    # 출력
    p.add_argument("--output_dir", type=str, default="./log/rsgm_out_robotcar")

    # Residual-SGM 파라미터
    p.add_argument("--residual_px", type=float, default=1.0, help="잔차 탐색 반경 r (px) → minDisp=-r, numDisp≈2r(16의 배수)")
    p.add_argument("--block_size", type=int, default=5, help="SGBM block size (odd 3..11)")
    p.add_argument("--uniqueness", type=int, default=5, help="uniqueness ratio (5~15)")
    p.add_argument("--speckle_win", type=int, default=200, help="speckle window size")
    p.add_argument("--speckle_range", type=int, default=2, help="speckle range")
    p.add_argument("--mode", type=str, default="HH", choices=["HH","3WAY","SGBM"], help="SGBM mode")
    p.add_argument("--color", action="store_true", help="컬러(BGR, 3채널)로 SGBM 수행")

    # (선택) 후처리
    p.add_argument("--wls", action="store_true", help="WLS(Generic) 정제 사용 (opencv-contrib 필요)")
    p.add_argument("--wls_lambda", type=float, default=4000.0, help="WLS lambda")
    p.add_argument("--wls_sigma",  type=float, default=0.8,    help="WLS sigmaColor")
    p.add_argument("--clip_nonneg", action="store_true", help="최종 disparity를 [0,∞)로 클립")

    # Photometric error 가중
    p.add_argument("--pth_l1_w",   type=float, default=0.15)
    p.add_argument("--pth_ssim_w", type=float, default=0.85)

    # (선택) GT 메트릭
    p.add_argument("--gt_depth_dir",  type=str, default=None, help="GT depth root (파일명 기준 매칭)")
    p.add_argument("--gt_depth_scale", type=float, default=1000.0)
    p.add_argument("--focal_px", type=float, default=983.044006)
    p.add_argument("--baseline_m", type=float, default=0.239)

    # 시각화
    p.add_argument("--vmax_disp", type=float, default=64.0, help="disparity 시각화 상한(px)")
    p.add_argument("--disp_cmap", type=str, default="magma")
    p.add_argument("--err_cmap",  type=str, default="magma")
    p.add_argument("--bg_color",  type=str, default="#1e1e1e")

    # 스케일 플래그
    p.add_argument("--half", action="store_true", help="전체 과정을 1/4 해상도/단위(px@1/4)로 수행")
    p.add_argument("--reshape", action="store_true",
                   help="정량 지표 계산 시 실행 스케일을 반영해 disparity→depth 변환을 수행하고 "
                        "GT depth도 실행 해상도로 리쉐이프하여 평가")

    # 기타
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    run(args)
