# -*- coding: utf-8 -*-
import os
import glob
import math
import argparse
from typing import Optional, Tuple, List

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
    try:
        bbox = draw.textbbox((0,0), text, font=font, stroke_width=2)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)
    bg = Image.new("RGBA", (tw+2*margin, th+2*margin), (0,0,0,100))
    img.paste(bg, (margin, margin), bg)
    draw.text((margin*2, margin*2), text, font=font,
              fill=(255,255,255,255), stroke_width=2, stroke_fill=(0,0,0,255))
    img = img.convert("RGB"); img.save(path)

# -------------------------------
# Photometric error 계산 (시각화/마스킹용: L1+SSIM)
# -------------------------------
@torch.no_grad()
def compute_pth_error_map(imgL_01: torch.Tensor, imgR_01: torch.Tensor, disp_full_px: torch.Tensor,
                          w_l1=0.15, w_ssim=0.85) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    반환:
      pth_map: [1,1,H,W]  (invalid은 NaN)
      valid:   [1,1,H,W]  (0/1)
    """
    imgR_warp, valid = warp_right_to_left_image(imgR_01, disp_full_px)  # [1,3,H,W], [1,1,H,W]
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
    """
    right_gray_u8: (H,W) uint8
    disp_px: (H,W) float32, left-view disparity (px)
    반환: (right_warp_gray_u8, valid_mask_bool)
    """
    H, W = right_gray_u8.shape
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mapx = xs - disp_px.astype(np.float32)
    mapy = ys
    right_warp = cv2.remap(right_gray_u8, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = (mapx >= 0) & (mapx <= (W-1)) & (mapy >= 0) & (mapy <= (H-1))
    return right_warp, valid

def _warp_right_to_left_bgr_cv2(right_bgr_u8: np.ndarray, disp_px: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    right_bgr_u8: (H,W,3) uint8
    disp_px    : (H,W) float32
    반환: (right_warp_bgr_u8, valid_mask_bool)
    """
    H, W, _ = right_bgr_u8.shape
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mapx = xs - disp_px.astype(np.float32)
    mapy = ys
    right_warp = cv2.remap(right_bgr_u8, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = (mapx >= 0) & (mapx <= (W-1)) & (mapy >= 0) & (mapy <= (H-1))
    return right_warp, valid

# -------------------------------
# Residual-SGM 핵심 유틸 (참조용: 인터페이스 유지, 미사용 가능)
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
    """
    disp_px: (H,W) float32
    guide_bgr_u8: (H,W,3) uint8
    """
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
# TGV^2 정제 유틸 (Chambolle–Pock)
# -------------------------------
def _fwd_grad(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # t: [B,1,H,W]
    dx = torch.zeros_like(t); dy = torch.zeros_like(t)
    dx[..., :, :-1] = t[..., :, 1:] - t[..., :, :-1]
    dy[..., :-1, :] = t[..., 1:, :] - t[..., :-1, :]
    return dx, dy

def _bwd_div(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    # px, py: [B,1,H,W] → div: [B,1,H,W]
    div = torch.zeros_like(px)
    # backward diff in x
    px_left = torch.zeros_like(px); px_left[..., :, 1:] = px[..., :, :-1]
    # backward diff in y
    py_up   = torch.zeros_like(py); py_up  [..., 1:, :] = py[..., :-1, :]
    div = (px - px_left) + (py - py_up)
    return div

def _symgrad(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # v: [B,2,H,W]  (v1,v2)
    v1 = v[:, 0:1, ...]; v2 = v[:, 1:2, ...]
    dv1x, dv1y = _fwd_grad(v1)
    dv2x, dv2y = _fwd_grad(v2)
    q1 = dv1x
    q2 = 0.5 * (dv1y + dv2x)
    q3 = dv2y
    return q1, q2, q3

def _symgrad_adj(q1: torch.Tensor, q2: torch.Tensor, q3: torch.Tensor) -> torch.Tensor:
    # qk: [B,1,H,W] → return [B,2,H,W]
    # backward (adjoint)
    def _bwd_dx(a):
        a_left = torch.zeros_like(a); a_left[..., :, 1:] = a[..., :, :-1]
        return a - a_left
    def _bwd_dy(a):
        a_up = torch.zeros_like(a); a_up[..., 1:, :] = a[..., :-1, :]
        return a - a_up
    v1 = _bwd_dx(q1) + 0.5 * _bwd_dy(q2)
    v2 = 0.5 * _bwd_dx(q2) + _bwd_dy(q3)
    return torch.cat([v1, v2], dim=1)

def _soft(x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - lam, min=0.0)

@torch.no_grad()
def tgv2_refine_chambolle_pock(
    d0: torch.Tensor,              # [1,1,H,W] 초기 disparity
    w_data: torch.Tensor,          # [1,1,H,W] 데이터 항 가중 (0..1)
    g_edge: torch.Tensor,          # [1,1,H,W] 엣지‑어웨어 평활 가중 (작을수록 경계)
    iters: int = 200,
    alpha0: float = 0.6,           # 2차(곡률) 가중
    alpha1: float = 1.2,           # 1차(경사) 가중
    tau: float = 0.125,
    sigma: float = 0.125,
    theta: float = 1.0,
) -> torch.Tensor:
    """
    간략 TGV^2(가이드 가중) 정련. 데이터 항: 가중 L1  |u-d0|.
    """
    device = d0.device
    B, C, H, W = d0.shape
    u  = d0.clone()
    v  = torch.zeros((B, 2, H, W), device=device, dtype=d0.dtype)
    p  = torch.zeros_like(v)           # dual for ∇u - v
    q1 = torch.zeros((B, 1, H, W), device=device, dtype=d0.dtype)  # dual for Ev (3 comps)
    q2 = torch.zeros_like(q1)
    q3 = torch.zeros_like(q1)

    u_bar = u.clone()
    v_bar = v.clone()

    # 엣지‑어웨어 반경 (dual 프로젝션 반지름)
    # 경계에서 작게 → 더 날카롭게
    eps = 1e-12
    rad_p = torch.clamp(alpha1 * g_edge, min=1e-3)   # [B,1,H,W]
    rad_q = torch.clamp(alpha0 * g_edge, min=1e-3)   # [B,1,H,W]

    # 데이터 항 스케일(안정성 목적) : w_data ∈ [0,1] → λ_data = w_data
    lam_data = torch.clamp(w_data, 0.0, 1.0)         # [B,1,H,W]

    for _ in range(iters):
        # --- Dual ascent ---
        # p ← Π_{||·||≤rad_p} ( p + σ (∇ū - v̄) )
        gux, guy = _fwd_grad(u_bar)
        px = p[:, 0:1, ...] + sigma * (gux - v_bar[:, 0:1, ...])
        py = p[:, 1:1+1, ...] + sigma * (guy - v_bar[:, 1:1+1, ...])
        p_norm = torch.sqrt(px * px + py * py + eps)                    # [B,1,H,W]
        scale_p = torch.clamp(p_norm / (rad_p + eps), min=1.0)
        px = px / scale_p
        py = py / scale_p
        p = torch.cat([px, py], dim=1)

        # q ← Π_{||·||≤rad_q} ( q + σ E v̄ )
        ev1, ev2, ev3 = _symgrad(v_bar)
        q1n = q1 + sigma * ev1
        q2n = q2 + sigma * ev2
        q3n = q3 + sigma * ev3
        # Frobenius norm for symmetric 2x2: sqrt(q1^2 + 2*q2^2 + q3^2)
        q_norm = torch.sqrt(q1n*q1n + 2.0*q2n*q2n + q3n*q3n + eps)
        scale_q = torch.clamp(q_norm / (rad_q + eps), min=1.0)
        q1 = q1n / scale_q
        q2 = q2n / scale_q
        q3 = q3n / scale_q

        # --- Primal descent ---
        u_prev = u.clone()
        v_prev = v.clone()

        # u ← prox_{τ * |·-d0| (가중)} ( u + τ div p )
        div_p = _bwd_div(px, py)
        z = u + tau * div_p
        # prox of weighted L1: d0 + soft(z-d0, τ * λ_i)
        u = d0 + _soft(z - d0, tau * lam_data)

        # v ← v + τ ( -p + E* q )
        e_adj = _symgrad_adj(q1, q2, q3)  # [B,2,H,W]
        v = v + tau * (-p + e_adj)

        # --- Over-relaxation ---
        u_bar = u + theta * (u - u_prev)
        v_bar = v + theta * (v - v_prev)

    return u

def _edge_weight_from_img(imgL_01: torch.Tensor) -> torch.Tensor:
    """
    imgL_01: [1,3,H,W] (0..1) → 경계에서 작은 값, 내부에서 1에 가까운 가중
    """
    with torch.no_grad():
        # luminance
        R, G, B = imgL_01[:, 0:1], imgL_01[:, 1:2], imgL_01[:, 2:3]
        Y = 0.2989*R + 0.5870*G + 0.1140*B
        gx, gy = _fwd_grad(Y)
        mag = torch.sqrt(gx*gx + gy*gy + 1e-12)
        # 평균 대비 스케일
        s = mag.mean() * 4.0 + 1e-6
        g = torch.exp(-mag / s)
        g = torch.clamp(g, 0.05, 1.0)
        return g

def _confidence_from_photometric(imgL_01: torch.Tensor, imgR_01: torch.Tensor, d0: torch.Tensor,
                                 w_l1: float, w_ssim: float) -> torch.Tensor:
    """
    photometric 잔차 기반 confidence ∈ [0,1], invalid는 0
    """
    with torch.no_grad():
        pth, valid = compute_pth_error_map(imgL_01, imgR_01, d0, w_l1=w_l1, w_ssim=w_ssim)
        # 유효 영역만 통계
        v = (valid > 0.5)
        if v.any():
            med = torch.nanmedian(pth[v]).item()
        else:
            finite = torch.isfinite(pth)
            med = torch.nanmedian(pth[finite]).item() if finite.any() else 1.0
        med = max(med, 1e-6)
        conf = torch.exp(-pth / (3.0 * med)) * v.float()
        conf = torch.clamp(conf, 0.0, 1.0)
        return conf

# -------------------------------
# EPE/D1 계산 (full-res px 기준)
# -------------------------------
@torch.no_grad()
def compute_epe_d1_from_gt_full(
    pred_disp_full_px: torch.Tensor,  # [1,1,H,W]
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

# -------------------------------
# 메인 루프 (TGV^2 정제)
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

    # (참조) SGBM 하이퍼(이 코드는 TGV 정제만 사용)
    matcher, min_disp, num_disp = _build_sgbm_for_residual(
        r_px=args.residual_px,
        block_size=args.block_size,
        uniqueness=args.uniqueness,
        speckle_win=args.speckle_win,
        speckle_range=args.speckle_range,
        mode=args.mode,
        use_color=bool(args.color)
    )

    for it, (imgL, imgR, names) in enumerate(loader, start=1):
        name = names[0] if isinstance(names, (list, tuple)) else names
        stem = _basename_wo_ext(name)

        # --- 초기 disparity 파일(NPY) 로드 ---
        disp_path = find_file_by_stem(args.disp_dir, stem, [".npy"])
        if disp_path is None:
            print(f"[Skip] init disp npy not found for stem={stem}")
            continue
        D0_np = np.load(disp_path).astype(np.float32)  # (H,W) px (left view)

        # 해상도 체크/보정
        H_t, W_t = int(imgL.shape[-2]), int(imgL.shape[-1])
        if D0_np.shape != (H_t, W_t):
            # disparity는 픽셀 단위 → 가로 배율로 스케일
            scale_x = W_t / D0_np.shape[1]
            D0_np = cv2.resize(D0_np, (W_t, H_t), interpolation=cv2.INTER_LINEAR) * scale_x

        # --- 토치→넘파이 이미지 변환 (OpenCV용)
        imgL01 = denorm_imagenet(imgL.to(device))
        imgR01 = denorm_imagenet(imgR.to(device))
        L_rgb = (imgL01[0].detach().cpu().permute(1,2,0).numpy() * 255.0).astype(np.uint8)
        R_rgb = (imgR01[0].detach().cpu().permute(1,2,0).numpy() * 255.0).astype(np.uint8)
        L_bgr = cv2.cvtColor(L_rgb, cv2.COLOR_RGB2BGR)
        R_bgr = cv2.cvtColor(R_rgb, cv2.COLOR_RGB2BGR)
        L_gray = cv2.cvtColor(L_bgr, cv2.COLOR_BGR2GRAY)
        R_gray = cv2.cvtColor(R_bgr, cv2.COLOR_BGR2GRAY)

        # --- Photometric error (before) ---
        D0_t = torch.from_numpy(D0_np).to(device=device, dtype=torch.float32).view(1,1,H_t,W_t)
        pth_before, _ = compute_pth_error_map(imgL01, imgR01, D0_t, w_l1=args.pth_l1_w, w_ssim=args.pth_ssim_w)
        pth_bef_np = pth_before[0,0].detach().cpu().numpy()
        save_colormap_png_with_colorbar_auto_range(
            os.path.join(out_pth_bef, f"{stem}_pth_before.png"),
            pth_bef_np, vmin=0.0, vmax=None, cmap_name=args.err_cmap,
            label="Photometric error", bg_color=args.bg_color
        )

        # ============================================================
        # --------------------- TGV^2 정제 파트 ----------------------
        # ============================================================
        # 신뢰도(데이터 항) & 엣지‑어웨어 평활 가중 계산
        w_data = _confidence_from_photometric(imgL01, imgR01, D0_t, args.pth_l1_w, args.pth_ssim_w)  # [1,1,H,W]
        g_edge = _edge_weight_from_img(imgL01)                                                       # [1,1,H,W]

        # 파라미터 (고정값; 외부 구조/인자 변경 없이 내부에서만 사용)
        # disparity 스케일에 따라 살짝 보정
        scale_disp = max(1.0, float(args.vmax_disp) / 56.0)
        alpha0 = 0.6 * scale_disp   # 2차 (곡률)
        alpha1 = 1.2 * scale_disp   # 1차 (경사)
        iters  = 200
        tau, sigma, theta = 0.125, 0.125, 1.0

        # TGV^2 정제 실행
        D_ref_t = tgv2_refine_chambolle_pock(
            d0=D0_t, w_data=w_data, g_edge=g_edge,
            iters=iters, alpha0=alpha0, alpha1=alpha1,
            tau=tau, sigma=sigma, theta=theta
        )
        D_ref = D_ref_t[0,0].detach().cpu().numpy().astype(np.float32)

        # (선택) WLS/FGS 정제
        if args.wls:
            D_ref = _apply_wls_generic_if_available(D_ref, L_bgr, args.wls_lambda, args.wls_sigma)
        if args.clip_nonneg:
            D_ref = np.clip(D_ref, 0, None)

        # --- Δ, D_ref 저장/시각화 ---
        delta = (D_ref - D0_np).astype(np.float32)
        r = int(round(float(args.residual_px)))
        save_npy(os.path.join(out_delta, f"{stem}_delta.npy"), delta)
        save_colormap_png_with_colorbar_auto_range(
            os.path.join(out_delta, f"{stem}_delta_cb.png"),
            np.clip(delta, -r, +r), vmin=-r, vmax=+r, cmap_name="RdBu_r",
            label="Δ disparity (px)", bg_color=args.bg_color
        )

        save_npy(os.path.join(out_disp_ref, f"{stem}.npy"), D_ref)
        disp_png = os.path.join(out_disp_ref, f"{stem}_disp_refined_cb.png")
        save_colormap_png_with_colorbar_auto_range(
            disp_png, D_ref, vmin=0.0, vmax=args.vmax_disp,
            cmap_name=args.disp_cmap, label="Disparity (px)", bg_color=args.bg_color
        )

        # --- Photometric error (after) ---
        D_ref_t_vis = torch.from_numpy(D_ref).to(device=device, dtype=torch.float32).view(1,1,H_t,W_t)
        pth_after, _ = compute_pth_error_map(imgL01, imgR01, D_ref_t_vis, w_l1=args.pth_l1_w, w_ssim=args.pth_ssim_w)
        pth_after_np = pth_after[0,0].detach().cpu().numpy()
        pth_png_after = os.path.join(out_pth_aft, f"{stem}_pth_after.png")
        save_colormap_png_with_colorbar_auto_range(
            pth_png_after, pth_after_np, vmin=0.0, vmax=None,
            cmap_name=args.err_cmap, label="Photometric error", bg_color=args.bg_color
        )

        # --- (선택) GT 메트릭: EPE/D1 오버레이 ---
        if args.gt_depth_dir and args.focal_px > 0 and args.baseline_m > 0:
            gt_depth = load_ms2_gt_depth_batch(
                names=[name], gt_depth_dir=args.gt_depth_dir,
                scale=args.gt_depth_scale, target_hw=(H_t, W_t), device=device
            )
            epe, d1 = compute_epe_d1_from_gt_full(D_ref_t_vis, gt_depth, args.focal_px, args.baseline_m)
            if (epe is not None) and (d1 is not None):
                annotate_png_top_left(disp_png,     f"EPE {epe:.3f} px | D1 {d1:.2f}%")
                annotate_png_top_left(pth_png_after,f"EPE {epe:.3f} px | D1 {d1:.2f}%")
            else:
                annotate_png_top_left(disp_png, "EPE/D1 : N/A")
                annotate_png_top_left(pth_png_after, "EPE/D1 : N/A")
        else:
            annotate_png_top_left(disp_png, "EPE/D1 : N/A (no GT)")
            annotate_png_top_left(pth_png_after, "EPE/D1 : N/A (no GT)")

        print(f"[{it:04d}] TGV^2 refined & saved: {stem}  |  alpha0={alpha0:.3f}, alpha1={alpha1:.3f}")

    print(f"[Done] outputs → {out_root}")

# argparse
# -------------------------------
def get_args():
    p = argparse.ArgumentParser("TGV^2 refine: confidence‑weighted L1 data + edge‑aware TGV^2 smoothness (WLS/FGS optional)")

    # 입력 경로
    p.add_argument("--left_dir",  type=str, required=True)
    p.add_argument("--right_dir", type=str, required=True)
    p.add_argument("--disp_dir",  type=str, required=True, help="*.npy (full-res px) initial disparity per image stem")

    # 데이터 크기(인퍼런스와 동일하게 맞추기)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width",  type=int, default=1224)

    # 출력
    p.add_argument("--output_dir", type=str, default="./log/rsgm_out")

    # (이전 호환) Residual‑SGM 파라미터 (본 TGV 버전에서는 내부적으로 사용하지 않음 / delta 시각화 범위 등에만 활용)
    p.add_argument("--residual_px", type=float, default=1.0, help="잔차 탐색 반경 r (px) → delta 시각화 vmin/vmax 설정에 사용")
    p.add_argument("--block_size", type=int, default=5, help="(호환) SGBM block size")
    p.add_argument("--uniqueness", type=int, default=10, help="(호환) uniqueness ratio")
    p.add_argument("--speckle_win", type=int, default=200, help="(호환) speckle window size")
    p.add_argument("--speckle_range", type=int, default=2, help="(호환) speckle range")
    p.add_argument("--mode", type=str, default="HH", choices=["HH","3WAY","SGBM"], help="(호환) SGBM mode")
    p.add_argument("--color", action="store_true", help="(호환) 컬러로 SGBM 수행")

    # (선택) 후처리
    p.add_argument("--wls", action="store_true", help="WLS(Generic) 정제 사용 (opencv-contrib 필요)")
    p.add_argument("--wls_lambda", type=float, default=8000.0, help="WLS lambda")
    p.add_argument("--wls_sigma",  type=float, default=0.8,    help="WLS sigmaColor")
    p.add_argument("--clip_nonneg", action="store_true", help="최종 disparity를 [0,∞)로 클립")

    # Photometric error 가중
    p.add_argument("--pth_l1_w",   type=float, default=0.15)
    p.add_argument("--pth_ssim_w", type=float, default=0.85)

    # (선택) GT 메트릭
    p.add_argument("--gt_depth_dir",  type=str, default=None, help="GT depth root (파일명 기준 매칭)")
    p.add_argument("--gt_depth_scale", type=float, default=256.0)
    p.add_argument("--focal_px", type=float, default=764.5138549804688)
    p.add_argument("--baseline_m", type=float, default=0.29918420530585865)

    # 시각화
    p.add_argument("--vmax_disp", type=float, default=56.0, help="disparity 시각화 상한(px)")
    p.add_argument("--disp_cmap", type=str, default="magma")
    p.add_argument("--err_cmap",  type=str, default="magma")
    p.add_argument("--bg_color",  type=str, default="#1e1e1e")

    # 기타
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    run(args)
