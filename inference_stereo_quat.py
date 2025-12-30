# -*- coding: utf-8 -*-
import os
import glob
import argparse
from typing import Tuple, Optional, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2  # ★ 추가: overlay 저장용

# =========================
# (환경에 맞게 경로 조정)
# =========================
from vit_cn import StereoModel
# from vit_cn_L import StereoModel
from agg.aggregator import SOTAStereoDecoder
# from agg.aggregator_plus import SOTAStereoDecoder

from tools import (
    StereoFolderDataset,
    read_fx_baseline_rgb,
    disparity_to_depth,
    # --- metrics & GT loader (학습 코드와 동일 유틸) ---
    load_ms2_gt_depth_batch,
    compute_ms2_disparity_metrics,
    # --- photometric error 시각화용 추가 ---
    denorm_imagenet,
    warp_right_to_left_image,
    PhotometricLoss,
)

# =========================================================
# padding 유틸 (학습 코드와 동일)
# =========================================================
def pad_to_multiple(x: torch.Tensor, mult: int = 16, mode: str = "replicate"):
    H, W = x.shape[-2], x.shape[-1]
    pad_r = (-W) % mult
    pad_b = (-H) % mult
    if pad_r or pad_b:
        x = F.pad(x, (0, pad_r, 0, pad_b), mode=mode)
    return x, (pad_b, pad_r)

def unpad_last2(x: torch.Tensor, pad: Tuple[int, int]):
    pad_b, pad_r = pad
    if pad_b == 0 and pad_r == 0:
        return x
    H, W = x.shape[-2], x.shape[-1]
    return x[..., :H - pad_b, :W - pad_r].contiguous()

# =========================================================
# 저장 & 시각화 유틸
# =========================================================
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _basename_wo_ext(path):
    return os.path.splitext(os.path.basename(path))[0]

def save_npy(path, np_array):
    _ensure_dir(os.path.dirname(path))
    np.save(path, np_array.astype(np.float32), allow_pickle=False)

def save_png_16u(path, np_array, scale: float = 1.0):
    """
    disparity/깊이를 16-bit PNG로 저장.
    - np_array: float32 HxW (px 또는 m 단위)
    - scale: 저장시 배율(예: KITTI 호환 256 배율 등)
    """
    from PIL import Image
    _ensure_dir(os.path.dirname(path))
    arr = np_array * float(scale)
    arr = np.clip(arr, 0, 65535).astype(np.uint16)
    Image.fromarray(arr, mode="I;16").save(path)

def save_colormap_png(path, np_array, vmax=None, cmap_name="magma"):
    """
    시각화용 컬러 PNG 저장(8-bit). vmax 미지정 시 자동 min-max.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(os.path.dirname(path))
    arr = np_array
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        vmin, vmax_eff = 0.0, 1.0 if vmax is None else float(vmax)
    else:
        vmin = float(arr[finite_mask].min())
        vmax_eff = float(arr[finite_mask].max()) if vmax is None else float(vmax)
        vmax_eff = max(vmax_eff, vmin + 1e-6)

    norm = (arr - vmin) / (vmax_eff - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    colored = (cmap(norm)[..., :3] * 255.0).astype("uint8")

    from PIL import Image
    Image.fromarray(colored).save(path)

def save_colormap_png_with_colorbar(
    path,
    np_array,
    vmax=None,
    cmap_name="magma",
    label: str = "",
    bg_color: str = "#1e1e1e",
):
    """
    오른쪽 colorbar 포함 버전(에러맵 등). NaN은 배경색으로 렌더.
    vmin=0 고정(오류맵 가독성), vmax 미지정시 99퍼센타일.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, to_rgba

    _ensure_dir(os.path.dirname(path))
    arr = np.array(np_array, dtype=np.float32)

    finite_mask = np.isfinite(arr)
    vmin = 0.0
    if not finite_mask.any():
        vmax_eff = 1.0 if vmax is None else float(vmax)
    else:
        if vmax is None:
            vmax_eff = float(np.nanpercentile(arr[finite_mask], 99.0))
            vmax_eff = max(vmax_eff, vmin + 1e-6)
        else:
            vmax_eff = float(vmax)

    import numpy as _np
    import matplotlib.pyplot as plt
    base_cmap = plt.get_cmap(cmap_name)
    cmap = ListedColormap(base_cmap(_np.linspace(0, 1, 256)))
    bg_rgba = to_rgba(bg_color)
    cmap.set_bad(bg_rgba)

    H, W = arr.shape
    dpi = 200.0
    figsize = (W / dpi, H / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax_eff)
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if label:
        cbar.set_label(label, rotation=270, labelpad=12)

    plt.tight_layout(pad=0.1)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1, facecolor=fig.get_facecolor())
    plt.close(fig)

def save_colormap_png_with_colorbar_auto_range(
    path,
    np_array,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_name: str = "magma",
    label: str = "",
    bg_color: str = "#1e1e1e",
):
    """
    일반 수치맵(예: disparity)용 컬러 PNG 저장 + colorbar.
    vmin/vmax 미지정 시 데이터의 finite min/max로 자동 설정.
    NaN은 배경색으로 렌더.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, to_rgba

    _ensure_dir(os.path.dirname(path))
    arr = np.array(np_array, dtype=np.float32)
    finite_mask = np.isfinite(arr)

    if not finite_mask.any():
        vmin_eff = 0.0 if (vmin is None) else float(vmin)
        vmax_eff = 1.0 if (vmax is None) else float(vmax)
    else:
        vmin_eff = float(np.nanmin(arr[finite_mask])) if (vmin is None) else float(vmin)
        vmax_eff = float(np.nanmax(arr[finite_mask])) if (vmax is None) else float(vmax)
        vmax_eff = max(vmax_eff, vmin_eff + 1e-6)

    import numpy as _np
    base_cmap = plt.get_cmap(cmap_name)
    cmap = ListedColormap(base_cmap(_np.linspace(0, 1, 256)))
    cmap.set_bad(to_rgba(bg_color))

    H, W = arr.shape
    dpi = 200.0
    figsize = (W / dpi, H / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    im = ax.imshow(arr, cmap=cmap, vmin=vmin_eff, vmax=vmax_eff)
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if label:
        cbar.set_label(label, rotation=270, labelpad=12)

    plt.tight_layout(pad=0.1)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1, facecolor=fig.get_facecolor())
    plt.close(fig)

def save_gray_png(path, np_array_uint8):
    from PIL import Image
    _ensure_dir(os.path.dirname(path))
    Image.fromarray(np_array_uint8.astype(np.uint8), mode="L").save(path)

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
        try:
            font = ImageFont.truetype("arial.ttf", fs)
        except Exception:
            font = ImageFont.load_default()

    draw = ImageDraw.Draw(img)
    try:
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=2)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)

    bg = Image.new("RGBA", (tw + margin * 2, th + margin * 2), (0, 0, 0, 100))
    img.paste(bg, (margin, margin), bg)
    draw.text((margin * 2, margin * 2), text, font=font,
              fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    img = img.convert("RGB")
    img.save(path)

def upsample_np(arr: np.ndarray, scale: int = 4, mode: str = "bilinear") -> np.ndarray:
    """
    수치맵을 scale배로 upsample (시각화 확대용).
    mode: 'bilinear' 권장(연속 수치), 'nearest'는 계단형 보존.
    """
    t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    if mode in ("bilinear", "bicubic", "trilinear"):
        t2 = F.interpolate(t, scale_factor=scale, mode=mode, align_corners=False)
    else:
        t2 = F.interpolate(t, scale_factor=scale, mode=mode)
    return t2.squeeze(0).squeeze(0).cpu().numpy()

# =========================================================
# ★ NEW: disparity를 이미지 위에 overlay하는 유틸
# =========================================================
def save_disp_overlay_on_image(
    path: str,
    base_bgr_u8: np.ndarray,   # HxWx3, BGR, uint8
    disp_np: np.ndarray,       # HxW, float
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_name: str = "magma",
    alpha: float = 0.6,
):
    """
    base_bgr_u8 위에 disparity colormap을 alpha blending해서 저장.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    _ensure_dir(os.path.dirname(path))

    H, W, _ = base_bgr_u8.shape
    arr = np.array(disp_np, dtype=np.float32)

    # disp 크기가 이미지랑 다르면 강제로 맞춤
    if arr.shape != (H, W):
        arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)

    mask = np.isfinite(arr)

    if mask.any():
        vmin_eff = float(np.nanmin(arr[mask])) if vmin is None else float(vmin)
        vmax_eff = float(np.nanmax(arr[mask])) if vmax is None else float(vmax)
        vmax_eff = max(vmax_eff, vmin_eff + 1e-6)
    else:
        vmin_eff = 0.0 if vmin is None else float(vmin)
        vmax_eff = 1.0 if vmax is None else float(vmax)

    norm = (arr - vmin_eff) / (vmax_eff - vmin_eff)
    norm = np.clip(norm, 0.0, 1.0)

    cmap = cm.get_cmap(cmap_name)
    color = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    color[~mask] = 0

    base = base_bgr_u8.astype(np.float32)
    disp_color = color.astype(np.float32)
    out = (1.0 - float(alpha)) * base + float(alpha) * disp_color
    out = np.clip(out, 0, 255).astype(np.uint8)

    cv2.imwrite(path, out)

# =========================================================
# 체크포인트 로더(다양한 포맷에 견고)
# =========================================================
def load_checkpoint_robust(ckpt_path: str,
                           stereo: torch.nn.Module,
                           decoder: torch.nn.Module,
                           device,
                           verbose: bool = True):
    import types
    import torch

    def _strip_prefix_once(name: str, prefix: str) -> str:
        return name[len(prefix):] if name.startswith(prefix) else name

    def _strip_common_prefixes(sd: dict) -> dict:
        prefixes = ["module.", "model.", "ckpt_model.", "ema.", "net.", "nets."]
        out = {}
        for k, v in sd.items():
            nk = k
            for p in prefixes:
                nk = _strip_prefix_once(nk, p)
            out[nk] = v
        return out

    def _try_load_from_sd(sd: dict) -> bool:
        if not isinstance(sd, dict) or len(sd) == 0:
            return False
        sd = _strip_common_prefixes(sd)

        stereo_sd = { _strip_prefix_once(k, "stereo."): v for k, v in sd.items() if k.startswith("stereo.") }
        decoder_sd = { _strip_prefix_once(k, "decoder."): v for k, v in sd.items() if k.startswith("decoder.") }
        loaded = False
        try:
            if len(stereo_sd) > 0:
                stereo.load_state_dict(stereo_sd, strict=False); loaded = True
            if len(decoder_sd) > 0:
                decoder.load_state_dict(decoder_sd, strict=False); loaded = True
            if loaded:
                return True
        except Exception:
            pass

        try:
            stereo.load_state_dict(sd, strict=False); loaded = True
        except Exception:
            pass
        try:
            decoder.load_state_dict(sd, strict=False); loaded = True
        except Exception:
            pass
        return loaded

    # 1) 학습 시 사용한 resume_from_checkpoint 우선
    try:
        from tools import resume_from_checkpoint
        Args = types.SimpleNamespace
        dummy_args = Args(resume=ckpt_path,
                          resume_reset_optim=True,
                          resume_reset_scaler=True)
        mdict = torch.nn.ModuleDict({"stereo": stereo, "decoder": decoder})
        start_epoch, _, _ = resume_from_checkpoint(dummy_args, mdict, device)
        if verbose:
            print(f"[Load] via tools.resume_from_checkpoint (epoch={start_epoch})")
        return
    except Exception as e:
        if verbose:
            print(f"[Load] tools.resume_from_checkpoint 경로 실패 → 수동 로더 시도: {e}")

    # 2) 수동 파싱
    obj = torch.load(ckpt_path, map_location=device)

    if isinstance(obj, torch.nn.Module):
        sd = obj.state_dict()
        if _try_load_from_sd(sd):
            if verbose: print("[Load] from nn.Module.state_dict()")
            return
        raise RuntimeError("모듈 state_dict 파싱 실패")

    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        if _try_load_from_sd(obj):
            if verbose: print("[Load] from raw state_dict")
            return

    if isinstance(obj, dict):
        if "stereo" in obj or "decoder" in obj:
            ok = False
            if isinstance(obj.get("stereo"), dict):
                try:
                    stereo.load_state_dict(_strip_common_prefixes(obj["stereo"]), strict=False); ok = True
                except Exception:
                    pass
            if isinstance(obj.get("decoder"), dict):
                try:
                    decoder.load_state_dict(_strip_common_prefixes(obj["decoder"]), strict=False); ok = True
                except Exception:
                    pass
            if ok:
                if verbose: print("[Load] from {'stereo':..., 'decoder':...}")
                return

        candidate_keys = ["state_dict", "model_state_dict", "model", "ckpt_model", "models", "ema"]
        for k in candidate_keys:
            if k in obj:
                sd = obj[k]
                if hasattr(sd, "state_dict"):
                    sd = sd.state_dict()
                if isinstance(sd, dict) and _try_load_from_sd(sd):
                    if verbose: print(f"[Load] from '{k}'")
                    return

    top_keys = []
    try:
        top_keys = list(obj.keys()) if isinstance(obj, dict) else [type(obj).__name__]
    except Exception:
        pass
    raise RuntimeError(f"지원하지 않는 체크포인트 형식: {ckpt_path}  (top-level keys: {top_keys})")

# =========================================================
# per-image metric 계산 (1/4 격자, 단위 px@full-res)
# =========================================================
def compute_epe_d1_per_item(
    pred_disp_q_px: torch.Tensor,   # [1,1,Hq,Wq], full-res px 단위 (1/4 격자)
    gt_depth_q_m: torch.Tensor,     # [1,1,Hq,Wq], meters
    focal_px: float,
    baseline_m: float,
) -> Tuple[Optional[float], Optional[float]]:
    valid = (gt_depth_q_m > 0).float()
    if valid.sum() <= 0:
        return None, None
    gt_disp_q_px = (float(focal_px) * float(baseline_m)) / gt_depth_q_m.clamp_min(1e-6)
    m = compute_ms2_disparity_metrics(pred_disp_q_px, gt_disp_q_px, valid)
    try:
        epe = float(m.get("EPE", float("nan")))
    except Exception:
        epe = None
    try:
        d1 = float(m.get("D1_all", float("nan")))
    except Exception:
        d1 = None
    if epe != epe: epe = None
    if d1 != d1: d1 = None
    return epe, d1

def compute_epe_d1_per_item_disp(
    pred_disp_px: torch.Tensor,   # [1,1,H,W], unit: px @ full-res (격자 크기는 H/W 임의)
    gt_disp_px: torch.Tensor,     # [1,1,H,W], unit: px (full-res)
) -> Tuple[Optional[float], Optional[float]]:
    """
    GT가 disparity(px)인 경우 metric 계산.
    """
    if gt_disp_px is None:
        return None, None
    valid = torch.isfinite(gt_disp_px).float() * (gt_disp_px > 0).float()
    if valid.sum() <= 0:
        return None, None
    m = compute_ms2_disparity_metrics(pred_disp_px, gt_disp_px, valid)
    epe = float(m.get("EPE", float("nan"))) if isinstance(m, dict) else None
    d1  = float(m.get("D1_all", float("nan"))) if isinstance(m, dict) else None
    if epe != epe: epe = None
    if d1 != d1: d1 = None
    return epe, d1

# =========================================================
# disparity gradient (|∂x d|, |∂y d|) 계산 유틸
# =========================================================
@torch.no_grad()
def disparity_gradients_abs(disp: torch.Tensor, keep_size: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    gx = disp[:, :, :, 1:] - disp[:, :, :, :-1]
    gy = disp[:, :, 1:, :] - disp[:, :, :-1, :]
    grad_x_abs = gx.abs()
    grad_y_abs = gy.abs()
    if keep_size:
        grad_x_abs = F.pad(grad_x_abs, (0, 1, 0, 0), mode="replicate")
        grad_y_abs = F.pad(grad_y_abs, (0, 0, 0, 1), mode="replicate")
    return grad_y_abs, grad_x_abs

# =========================================================
# 최종(Stage‑2) argmax(px) 맵 계산 유틸
# =========================================================
@torch.no_grad()
def final_argmax_px_from_pred(pred: dict,
                              pad_q: Tuple[int, int],
                              max_disp_px: float,
                              local_radius: int,
                              verbose: bool = False) -> Optional[torch.Tensor]:
    """
    반환: argmax(px) at 1/4 (shape [B,1,Hq,Wq]) or None
    우선순위:
      1) prob_volume_local_1_4 → 로컬 argmax → 전역 재정렬(idx + (d0 - r)) → *4(px)
      2) logits_1_4_ref(= -logits2) → 위와 동일
      3) 폴백: prob_volume_1_4 / logits_1_4(= -logits) → 전역 argmax → *4(px)
    """
    # --- Stage-2 local volume ---
    if "prob_volume_local_1_4" in pred and torch.is_tensor(pred["prob_volume_local_1_4"]):
        prob2 = unpad_last2(pred["prob_volume_local_1_4"], pad_q)       # [B, Dloc, Hq, Wq]
        idx_local = prob2.argmax(dim=1, keepdim=True).float()           # [B,1,Hq,Wq]
        if "disp_1_4_stage1" in pred and torch.is_tensor(pred["disp_1_4_stage1"]):
            d0_cell = unpad_last2(pred["disp_1_4_stage1"], pad_q) / 4.0 # [B,1,Hq,Wq]
        else:
            # 안전 폴백: 최종 disp_1_4 존재 시 사용 (완전 일치하진 않음)
            d0_cell = unpad_last2(pred["disp_1_4"], pad_q) / 4.0
        r = float(local_radius)
        idx_abs_cell = idx_local + (d0_cell - r)
        D_cells = float(max_disp_px) / 4.0
        idx_abs_cell = idx_abs_cell.clamp(0.0, max(D_cells - 1e-6, 0.0))
        return idx_abs_cell * 4.0                                       # px

    if "logits_1_4_ref" in pred and torch.is_tensor(pred["logits_1_4_ref"]):
        lg2 = unpad_last2(pred["logits_1_4_ref"], pad_q)                # [B, Dloc, Hq, Wq] = -logits2
        idx_local = lg2.argmax(dim=1, keepdim=True).float()
        if "disp_1_4_stage1" in pred and torch.is_tensor(pred["disp_1_4_stage1"]):
            d0_cell = unpad_last2(pred["disp_1_4_stage1"], pad_q) / 4.0
        else:
            d0_cell = unpad_last2(pred["disp_1_4"], pad_q) / 4.0
        r = float(local_radius)
        idx_abs_cell = idx_local + (d0_cell - r)
        D_cells = float(max_disp_px) / 4.0
        idx_abs_cell = idx_abs_cell.clamp(0.0, max(D_cells - 1e-6, 0.0))
        return idx_abs_cell * 4.0

    # --- Fallback: Stage-1 global volume ---
    if "prob_volume_1_4" in pred and torch.is_tensor(pred["prob_volume_1_4"]):
        prob = unpad_last2(pred["prob_volume_1_4"], pad_q)              # [B, D, Hq, Wq]
        idx_cell = prob.argmax(dim=1, keepdim=True).float()             # [B,1,Hq,Wq] (cell)
        return idx_cell * 4.0                                           # px
    if "logits_1_4" in pred and torch.is_tensor(pred["logits_1_4"]):
        lg = unpad_last2(pred["logits_1_4"], pad_q)                     # [B, D, Hq, Wq] = -logits
        idx_cell = lg.argmax(dim=1, keepdim=True).float()
        return idx_cell * 4.0

    if verbose:
        print("[final_argmax] no suitable volume found in pred")
    return None

# =========================================================
# GT disparity 로더 (basename 매칭 + 크롭/리사이즈)
# =========================================================
def _load_single_disp_gt(path: str,
                         scale: float) -> Optional[np.ndarray]:
    """
    단일 파일에서 disparity(px) 읽기.
    - 정수형 PNG/TIFF 등 → scale로 나눔(예: 256.0)
    - float 포맷(.npy/.npz/float PNG 등) → 값 그대로 사용
    반환: float32 HxW (px), 실패 시 None
    """
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            arr = np.load(path, allow_pickle=False)
            arr = np.array(arr, dtype=np.float32)
            return arr
        if ext == ".npz":
            data = np.load(path)
            # 우선순위 키 탐색
            for k in ["disp", "disparity", "arr", "data", "depth", "arr_0"]:
                if k in data:
                    arr = np.array(data[k], dtype=np.float32)
                    return arr
            # 첫 키 사용
            keys = list(data.keys())
            if len(keys) > 0:
                return np.array(data[keys[0]], dtype=np.float32)
            return None
        # 이미지 계열
        from PIL import Image
        img = Image.open(path)
        arr = np.array(img)
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / float(scale)
        else:
            arr = arr.astype(np.float32)
        return arr
    except Exception:
        return None

def _find_gt_file_for_stem(gt_root: str, name_like: str) -> Optional[str]:
    """
    주어진 입력 이미지 이름(name_like)의 stem을 기준으로
    gt_root 내부에서 같은 stem을 갖는 파일을 탐색.
    우선순위: .npy > .npz > .png > .tif > .tiff
    """
    stem = _basename_wo_ext(name_like)
    # 1) 루트 바로 아래에서 확장자별 우선 검색
    exts = [".npy", ".npz", ".png", ".tif", ".tiff"]
    for ext in exts:
        cand = os.path.join(gt_root, stem + ext)
        if os.path.isfile(cand):
            return cand
    # 2) 입력 경로 구조를 유지한 상대 경로 시도
    base_noext = os.path.splitext(name_like)[0]
    for ext in exts:
        cand = os.path.join(gt_root, base_noext + ext)
        if os.path.isfile(cand):
            return cand
    # 3) 재귀 검색 (basename 매칭)
    matches = []
    for ext in exts:
        matches.extend(glob.glob(os.path.join(gt_root, "**", stem + ext), recursive=True))
    if len(matches) > 0:
        # 확장자 우선순위 정렬
        def _rank(p):
            e = os.path.splitext(p)[1].lower()
            return exts.index(e) if e in exts else 999
        matches.sort(key=_rank)
        return matches[0]
    return None

def _align_to_target_hw(arr: np.ndarray,
                        target_hw: Tuple[int, int]) -> np.ndarray:
    """
    GT 맵을 타깃 해상도(H, W)에 정렬.
    - (H0>=H and W0>=W)면 좌상단(top-left) 크롭 → 요청 반영
    - 그 외(확대 포함)에는 최근접/양선형 보간으로 격자에 맞춤 (값 스케일은 유지)
    """
    Ht, Wt = int(target_hw[0]), int(target_hw[1])
    H0, W0 = int(arr.shape[0]), int(arr.shape[1])
    if H0 == Ht and W0 == Wt:
        return arr

    if H0 >= Ht and W0 >= Wt:
        return arr[:Ht, :Wt].copy()

    # 리사이즈 (값 스케일 유지, 단위는 여전히 full-res px)
    t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    # disparity는 연속량 → bilinear가 일반적으로 더 자연스러움
    t2 = F.interpolate(t, size=(Ht, Wt), mode="bilinear", align_corners=False)
    return t2.squeeze(0).squeeze(0).cpu().numpy()

def load_ms2_gt_disp_batch(
    names: Union[List[str], Tuple[str, ...], str],
    gt_disp_dir: str,
    scale: float,
    target_hw: Tuple[int, int],
    device: torch.device
) -> Optional[torch.Tensor]:
    """
    GT disparity(px) 배치를 로드.
    - 파일 탐색: stem 기반으로 gt_disp_dir 내에서 .npy/.npz/.png/.tif/.tiff 탐색
    - 정수형 포맷은 `scale`로 나눠 float(px)로 환산
    - 타깃 해상도에 **좌상단 크롭**을 우선 적용(더 큰 경우)하고,
      그 외에는 보간으로 맞춤(값 스케일은 유지, 즉 계속 full-res px 단위)
    """
    if gt_disp_dir is None:
        return None

    if isinstance(names, (str,)):
        name_list = [names]
    else:
        name_list = list(names)

    batch = []
    for nm in name_list:
        path = _find_gt_file_for_stem(gt_disp_dir, nm)
        if path is None:
            # 못 찾으면 invalid로 채움 (모두 0 → valid=0)
            Ht, Wt = int(target_hw[0]), int(target_hw[1])
            arr = np.zeros((Ht, Wt), dtype=np.float32)
        else:
            arr = _load_single_disp_gt(path, scale=scale)
            if arr is None:
                Ht, Wt = int(target_hw[0]), int(target_hw[1])
                arr = np.zeros((Ht, Wt), dtype=np.float32)
            else:
                # HxW로 정렬
                arr = _align_to_target_hw(arr, target_hw)

        batch.append(torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0))

    return torch.cat(batch, dim=0).to(device, non_blocking=True)

# =========================================================
# 추론 루프
# =========================================================
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp) and torch.cuda.is_available()

    # --- fx / baseline 자동 로딩(학습 코드와 동일 로직) ---
    fx, B, src = read_fx_baseline_rgb(
        intrinsic_left_npy=getattr(args, "K_left_npy", None),
        calib_npy=getattr(args, "calib_npy", None)
    )
    if getattr(args, "focal_px", 0.0) <= 0.0 and fx is not None:
        args.focal_px = float(fx)
    if getattr(args, "baseline_m", 0.0) <= 0.0 and B is not None:
        args.baseline_m = float(B)
    if args.verbose:
        print(f"[Calib] fx(px)={getattr(args,'focal_px',0.0):.6f}  baseline(m)={getattr(args,'baseline_m',0.0):.6f}  src={src}")

    # --- 데이터셋/로더(학습과 동일한 전처리 가정) ---
    dataset = StereoFolderDataset(
        args.left_dir, args.right_dir,
        height=args.height, width=args.width
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )

    # --- 모델 구성(학습 하이퍼와 동일하게 맞추세요) ---
    stereo = StereoModel(
        freeze_vit=True,
        cf=args.fused_ch,
        amp=amp_enabled,
        autopad_to_8=False,   # 입력 패딩을 외부에서 처리(학습과 동일)
    ).to(device).eval()

    decoder = SOTAStereoDecoder(
        max_disp_px=args.max_disp_px,
        fused_in_ch=args.fused_ch,
        red_ch=args.acv_red_ch,
        base3d=args.agg_ch,
        use_motif=args.use_motif,
        two_stage=args.two_stage,
        local_radius_cells=args.local_radius
    ).to(device).eval()

    # Photometric error용 손실(시각화에 map만 사용)
    photo_crit = PhotometricLoss([args.photo_l1_w, args.photo_ssim_w])

    # --- 가중치 로드 ---
    if args.ckpt is None or not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"--ckpt 경로가 올바르지 않습니다: {args.ckpt}")
    load_checkpoint_robust(args.ckpt, stereo, decoder, device, verbose=args.verbose)
    if args.verbose:
        print(f"[Load] checkpoint: {args.ckpt}")

    # --- 출력 폴더 ---
    out_root = _ensure_dir(args.output_dir)
    out_disp_qpx = _ensure_dir(os.path.join(out_root, "disp_1_4_qpx"))
    out_disp_qpx_px = _ensure_dir(os.path.join(out_root, "disp_1_4_px"))
    out_disp_full_px = _ensure_dir(os.path.join(out_root, "disp_full_px"))
    out_depth_m = _ensure_dir(os.path.join(out_root, "depth_m")) if (args.focal_px > 0 and args.baseline_m > 0) else None
    out_vis = _ensure_dir(os.path.join(out_root, "vis"))          # 시각화 폴더
    out_error = _ensure_dir(os.path.join(out_root, "error"))      # EPE 시각화 폴더
    out_pth   = _ensure_dir(os.path.join(out_root, "photometric_error"))  # Photometric error 폴더
    out_arg   = _ensure_dir(os.path.join(out_root, "argmax_1_4"))         # 최종 argmax(1/4) 폴더

    # ★ NEW: 디버그용 이미지 / overlay 폴더
    out_dbg_imgL    = _ensure_dir(os.path.join(out_root, "debug_imgL"))
    out_dbg_overlay = _ensure_dir(os.path.join(out_root, "debug_overlay"))

    # --- metrics 로깅 CSV (선택)
    metrics_csv_path = os.path.join(out_root, "metrics_per_image.csv")
    metrics_csv_fp = open(metrics_csv_path, "w", encoding="utf-8") if args.write_csv else None
    if metrics_csv_fp is not None:
        metrics_csv_fp.write("name,EPE_px,D1_all_percent\n")

    # --- 추론 ---
    torch.set_grad_enabled(False)
    context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad

    # 어떤 GT를 쓸지 결정
    use_gt_disp = args.gt_disp_dir is not None
    if use_gt_disp and (args.gt_depth_dir is not None) and args.verbose:
        print("[GT] --gt_disp_dir과 --gt_depth_dir이 모두 주어졌습니다. disparity GT를 우선 사용합니다.")

    has_fb = (args.focal_px > 0.0 and args.baseline_m > 0.0)

    with context():
        for it, (imgL, imgR, names) in enumerate(loader, start=1):
            imgL = imgL.to(device, non_blocking=True)  # ImageNet 정규화 가정
            imgR = imgR.to(device, non_blocking=True)

            # Photometric용 [0,1] 복원
            imgL_01 = denorm_imagenet(imgL)
            imgR_01 = denorm_imagenet(imgR)

            # 1) 입력 ×16 패딩
            imgL_pad, pad = pad_to_multiple(imgL, mult=16, mode="replicate")
            imgR_pad, _   = pad_to_multiple(imgR, mult=16, mode="replicate")
            assert pad[0] % 4 == 0 and pad[1] % 4 == 0, "pad must be divisible by 4 (학습 코드와 동일 전제)"

            # 2) 모델 추론
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                
                bb_out = stereo(imgL_pad, imgR_pad)
                pred   = decoder(bb_out)
                # 3) 출력 언패드 — 모두 1/4 해상도 좌표계
                pad_q = (pad[0] // 4, pad[1] // 4)

                # 디코더가 1/4 격자에서 "full-res px 단위" disparity를 예측(학습 코드와 동일)
                disp_q_pad_full_px = pred["disp_1_4"]                           # [B,1,Hq_pad,Wq_pad], unit: px @ full-res
                disp_q_full_px     = unpad_last2(disp_q_pad_full_px, pad_q)     # [B,1,Hq,Wq], unit: px
                disp_q_qpx         = disp_q_full_px / 4.0                       # [B,1,Hq,Wq], unit: 1/4‑px

                # Full-res(px) 업샘플 (디코더가 disp_full 제공 시 해당 경로 사용)
                if "disp_full" in pred:
                    disp_full_px = unpad_last2(pred["disp_full"], pad)          # [B,1,H,W], unit: px
                else:
                    disp_full_px = F.interpolate(disp_q_full_px, scale_factor=4, mode="bilinear", align_corners=False)

                # === NEW: 최종(Stage-2) argmax(px) 맵 @1/4 ===
                argmax_px_q = final_argmax_px_from_pred(
                    pred=pred, pad_q=pad_q, max_disp_px=float(args.max_disp_px),
                    local_radius=int(args.local_radius), verbose=args.verbose
                )  # [B,1,Hq,Wq] or None

            Bsz = disp_q_full_px.shape[0]

            # 4) (선택) GT 로딩 -> per-image metrics 계산
            gt_depth_q = None
            gt_depth = None
            gt_disp_q = None
            gt_disp = None

            if use_gt_disp:
                gt_disp_q = load_ms2_gt_disp_batch(
                    names=names,
                    gt_disp_dir=args.gt_disp_dir,
                    scale=args.gt_disp_scale,
                    target_hw=disp_q_full_px.shape[-2:],  # (Hq, Wq)
                    device=imgL.device
                )
                gt_disp = load_ms2_gt_disp_batch(
                    names=names,
                    gt_disp_dir=args.gt_disp_dir,
                    scale=args.gt_disp_scale,
                    target_hw=disp_full_px.shape[-2:],     # (H, W)
                    device=imgL.device
                )
            elif args.gt_depth_dir is not None:
                # 기존 depth GT 경로
                gt_depth_q = load_ms2_gt_depth_batch(
                    names=names,
                    gt_depth_dir=args.gt_depth_dir,
                    scale=args.gt_depth_scale,
                    target_hw=disp_q_full_px.shape[-2:],  # (Hq, Wq)
                    device=imgL.device
                )
                gt_depth = load_ms2_gt_depth_batch(
                    names=names,
                    gt_depth_dir=args.gt_depth_dir,
                    scale=args.gt_depth_scale,
                    target_hw=disp_full_px.shape[-2:],  # (H, W)
                    device=imgL.device
                )

            # 5) 저장 및 오버레이
            for bi in range(Bsz):
                name = names[bi] if isinstance(names, (list, tuple)) else names
                stem = _basename_wo_ext(name)

                # numpy 변환
                disp_q_qpx_np     = disp_q_qpx[bi, 0].detach().cpu().numpy()
                disp_q_full_px_np = disp_q_full_px[bi, 0].detach().cpu().numpy()
                disp_full_px_np   = disp_full_px[bi, 0].detach().cpu().numpy()

                # ★ NEW: 이 disparity가 실제로 만들어진 left 이미지 자체를 저장
                imgL01_np = imgL_01[bi].detach().cpu().permute(1, 2, 0).numpy()  # H,W,3 [0,1]
                imgL01_np = np.clip(imgL01_np, 0.0, 1.0)
                imgL_u8 = (imgL01_np * 255.0).astype(np.uint8)                   # RGB
                imgL_bgr = cv2.cvtColor(imgL_u8, cv2.COLOR_RGB2BGR)

                debug_img_path = os.path.join(out_dbg_imgL, f"{stem}_left_used.png")
                cv2.imwrite(debug_img_path, imgL_bgr)

                # ★ NEW: 이 left_used에 disp_full_px를 바로 overlay
                debug_overlay_path = os.path.join(out_dbg_overlay, f"{stem}_disp_overlay.png")
                vmax_disp = args.vmax if args.vmax is not None else args.max_disp_px
                save_disp_overlay_on_image(
                    debug_overlay_path,
                    imgL_bgr,
                    disp_full_px_np,
                    vmin=0.0,
                    vmax=vmax_disp,
                    cmap_name=args.disp_cmap,
                    alpha=args.overlay_alpha,
                )

                # 5-1) *.npy 저장
                if args.save_npy:
                    save_npy(os.path.join(out_disp_full_px, f"{stem}.npy"), disp_full_px_np)
                    save_npy(os.path.join(out_disp_qpx_px, f"{stem}.npy"), disp_q_qpx_np)

                # 5-2) 16-bit PNG(선택)
                if args.save_png16:
                    scale = float(args.png16_scale)
                    save_png_16u(os.path.join(out_disp_full_px, f"{stem}.png"), disp_full_px_np, scale=scale)

                # 5-3) 깊이(m) (선택)
                depth_m_np = None
                if out_depth_m is not None:
                    depth_m_np = disparity_to_depth(
                        torch.from_numpy(disp_full_px_np).unsqueeze(0).unsqueeze(0),
                        float(args.focal_px), float(args.baseline_m)
                    ).squeeze(0).squeeze(0).numpy()
                    # 필요 시 depth 저장/시각화 로직을 추가하세요.

                # 5-4) 시각화 PNG (disparity/깊이 + 오버레이)
                epe_full, d1_full = None, None
                text_overlay = None
                text_overlay_q = None

                # --- 1/4 해상도 metric 텍스트 ---
                if use_gt_disp and (gt_disp_q is not None):
                    epe_q, d1_q = compute_epe_d1_per_item_disp(
                        pred_disp_px=disp_q_full_px[bi:bi+1],
                        gt_disp_px=gt_disp_q[bi:bi+1],
                    )
                    text_overlay_q = f"EPE 1/4 {epe_q:.3f} px | D1 1/4 {d1_q:.2f}%" if (epe_q is not None and d1_q is not None) else "EPE/D1 1/4 : N/A"
                elif (gt_depth_q is not None) and has_fb:
                    epe_q, d1_q = compute_epe_d1_per_item(
                        pred_disp_q_px=disp_q_full_px[bi:bi+1],
                        gt_depth_q_m=gt_depth_q[bi:bi+1],
                        focal_px=float(args.focal_px),
                        baseline_m=float(args.baseline_m),
                    )
                    text_overlay_q = f"EPE 1/4 {epe_q:.3f} px | D1 1/4 {d1_q:.2f}%" if (epe_q is not None and d1_q is not None) else "EPE/D1 1/4 : N/A"
                elif args.overlay_always:
                    text_overlay_q = "EPE/D1 1/4 : N/A"

                # --- full 해상도 metric 텍스트 ---
                if use_gt_disp and (gt_disp is not None):
                    epe_full, d1_full = compute_epe_d1_per_item_disp(
                        pred_disp_px=disp_full_px[bi:bi+1],
                        gt_disp_px=gt_disp[bi:bi+1],
                    )
                    text_overlay = f"EPE {epe_full:.3f} px | D1 {d1_full:.2f}%" if (epe_full is not None and d1_full is not None) else "EPE/D1 : N/A"
                elif (gt_depth is not None) and has_fb:
                    epe_full, d1_full = compute_epe_d1_per_item(
                        pred_disp_q_px=disp_full_px[bi:bi+1],
                        gt_depth_q_m=gt_depth[bi:bi+1],
                        focal_px=float(args.focal_px),
                        baseline_m=float(args.baseline_m),
                    )
                    text_overlay = f"EPE {epe_full:.3f} px | D1 {d1_full:.2f}%"
                elif args.overlay_always:
                    text_overlay = "EPE/D1 : N/A"

                # disparity 시각화 저장
                if args.save_color:
                    # 1/4 해상도(q‑px) 보조 시각화 (공간 ×4 확대)
                    p2_q = os.path.join(out_disp_qpx_px, f"{stem}_disp_1_4_px.png")
                    save_colormap_png(p2_q, upsample_np(disp_q_qpx_np, scale=4, mode="bilinear"),
                                      vmax=args.max_disp_px / 4.0, cmap_name=args.disp_cmap)
                    if text_overlay_q:
                        annotate_png_top_left(p2_q, text_overlay_q)

                    # === NEW: 최종(Stage-2) argmax(px) 저장 (공간 ×4 확대) ===
                    if argmax_px_q is not None:
                        arg_px_np = argmax_px_q[bi, 0].detach().cpu().numpy()             # px @1/4
                        arg_px_np_up = upsample_np(arg_px_np, scale=4, mode="bilinear")    # 보기용 확대

                        # (2) 컬러바 포함
                        p_arg_cb = os.path.join(out_arg, f"{stem}_argmax_1_4_px_cb.png")
                        save_colormap_png_with_colorbar_auto_range(
                            p_arg_cb,
                            arg_px_np_up,
                            vmin=0.0, vmax=args.max_disp_px,
                            cmap_name=args.disp_cmap,
                            label="Argmax Stage-2 (px)",
                            bg_color=args.disp_bg_color
                        )
                        annotate_png_top_left(p_arg_cb, "Argmax Stage-2 (px)" + (f"  |  {text_overlay_q}" if text_overlay_q else ""))

                        if args.save_npy:
                            save_npy(os.path.join(out_arg, f"{stem}_argmax_1_4_px.npy"), arg_px_np)

                    # === full-res(px) + colorbar 버전 저장 ===
                    p2_cb = os.path.join(out_disp_full_px, f"{stem}_disp_px_cb.png")
                    save_colormap_png_with_colorbar_auto_range(
                        p2_cb,
                        disp_full_px_np,
                        vmin=None,
                        vmax=args.max_disp_px,
                        cmap_name=args.disp_cmap,
                        label="Disparity (px)",
                        bg_color=args.disp_bg_color
                    )
                    if text_overlay:
                        annotate_png_top_left(p2_cb, text_overlay)

                # === disparity gradient 시각화 저장 ===
                if args.save_color and args.save_disp_grads:
                    if args.grad_on in ("full", "both"):
                        disp_full_curr = disp_full_px[bi:bi+1]  # [1,1,H,W]
                        gy_full, gx_full = disparity_gradients_abs(disp_full_curr, keep_size=True)
                        gy_full_np = gy_full.squeeze(0).squeeze(0).detach().cpu().numpy()
                        gx_full_np = gx_full.squeeze(0).squeeze(0).detach().cpu().numpy()
                        pvy = os.path.join(out_vis, f"{stem}_gradV_full.png")
                        pvx = os.path.join(out_vis, f"{stem}_gradH_full.png")
                        save_colormap_png(pvy, gy_full_np, vmax=args.vmax_grad)
                        save_colormap_png(pvx, gx_full_np, vmax=args.vmax_grad)
                        annotate_png_top_left(pvy, "Vertical |∂y disp|")
                        annotate_png_top_left(pvx, "Horizontal |∂x disp|")

                    if args.grad_on in ("q", "both"):
                        disp_q_curr = disp_q_full_px[bi:bi+1]  # [1,1,Hq,Wq]
                        gy_q, gx_q = disparity_gradients_abs(disp_q_curr, keep_size=True)
                        gy_q_np = gy_q.squeeze(0).squeeze(0).detach().cpu().numpy()
                        gx_q_np = gx_q.squeeze(0).squeeze(0).detach().cpu().numpy()
                        pvy_q = os.path.join(out_vis, f"{stem}_gradV_1_4.png")
                        pvx_q = os.path.join(out_vis, f"{stem}_gradH_1_4.png")
                        save_colormap_png(pvy_q, gy_q_np, vmax=args.vmax_grad)
                        save_colormap_png(pvx_q, gx_q_np, vmax=args.vmax_grad)
                        annotate_png_top_left(pvy_q, "Vertical |∂y disp| @1/4")
                        annotate_png_top_left(pvx_q, "Horizontal |∂x disp| @1/4")

                # --- Error maps (GT가 있을 때) ---
                if use_gt_disp and (gt_disp is not None):
                    valid_full = (gt_disp[bi:bi+1] > 0).float() * torch.isfinite(gt_disp[bi:bi+1]).float()
                    epe_map_full = torch.abs(disp_full_px[bi:bi+1] - gt_disp[bi:bi+1]) * valid_full  # [1,1,H,W]
                    epe_map_full_np  = epe_map_full.squeeze(0).squeeze(0).detach().cpu().numpy()

                    if args.save_npy:
                        save_npy(os.path.join(out_vis, f"{stem}_err_epe_full_px.npy"), epe_map_full_np)

                    if args.save_color:
                        vmax_err = args.vmax_err if (args.vmax_err is not None and args.vmax_err > 0) else None
                        p_err_full_png = os.path.join(out_error, f"{stem}_error map.png")
                        save_colormap_png_with_colorbar(
                            p_err_full_png,
                            epe_map_full_np,
                            vmax=vmax_err,
                            cmap_name=args.err_cmap,
                            label="EPE (px)",
                            bg_color=args.err_bg_color
                        )
                        if text_overlay:
                            annotate_png_top_left(p_err_full_png, text_overlay)

                elif (gt_depth is not None) and has_fb:
                    valid_full = (gt_depth[bi:bi+1] > 0).float()  # [1,1,H,W]
                    gt_disp_full_px = (float(args.focal_px) * float(args.baseline_m)) / \
                                      torch.clamp(gt_depth[bi:bi+1], min=1e-6)  # [1,1,H,W]
                    epe_map_full = torch.abs(disp_full_px[bi:bi+1] - gt_disp_full_px) * valid_full  # [1,1,H,W]
                    epe_map_full_np  = epe_map_full.squeeze(0).squeeze(0).detach().cpu().numpy()

                    if args.save_npy:
                        save_npy(os.path.join(out_vis, f"{stem}_err_epe_full_px.npy"), epe_map_full_np)

                    if args.save_color:
                        vmax_err = args.vmax_err if (args.vmax_err is not None and args.vmax_err > 0) else None
                        p_err_full_png = os.path.join(out_error, f"{stem}_error map.png")
                        save_colormap_png_with_colorbar(
                            p_err_full_png,
                            epe_map_full_np,
                            vmax=vmax_err,
                            cmap_name=args.err_cmap,
                            label="EPE (px)",
                            bg_color=args.err_bg_color
                        )
                        if text_overlay:
                            annotate_png_top_left(p_err_full_png, text_overlay)

                # --- Photometric error map 저장 ---
                if args.save_color:
                    imgL_b = imgL_01[bi:bi+1]
                    imgR_b = imgR_01[bi:bi+1]
                    imgR_warp, valid_w = warp_right_to_left_image(imgR_b, disp_full_px[bi:bi+1])
                    pth_map = photo_crit.simple_photometric_loss(
                        imgL_b, imgR_warp,
                        weights=[args.photo_l1_w, args.photo_ssim_w]
                    )  # [1,1,H,W] 가정

                    pth_map_np = pth_map.squeeze(0).squeeze(0).detach().cpu().numpy()
                    valid_w_np = valid_w.squeeze(0).squeeze(0).detach().cpu().numpy().astype(bool)
                    pth_map_np[~valid_w_np] = np.nan

                    p_pth_png = os.path.join(out_pth, f"{stem}_pth_error.png")
                    save_colormap_png_with_colorbar(
                        p_pth_png,
                        pth_map_np,
                        vmax=None,                  # 99th percentile 자동
                        cmap_name=args.err_cmap,    # 에러맵과 동일 팔레트 사용
                        label="Photometric error",
                        bg_color=args.err_bg_color
                    )
                    if args.save_npy:
                        save_npy(os.path.join(out_pth, f"{stem}_pth_error.npy"), pth_map_np)

                # CSV 로깅
                if metrics_csv_fp is not None:
                    if (epe_full is not None) and (d1_full is not None):
                        metrics_csv_fp.write(f"{stem},{epe_full},{d1_full}\n")
                    else:
                        metrics_csv_fp.write(f"{stem},,\n")

            if args.verbose and (it % args.log_every == 0):
                print(f"[Infer {it:05d}/{len(loader)}] saved batch of {Bsz}")

    if metrics_csv_fp is not None:
        metrics_csv_fp.close()
    if args.verbose:
        print(f"[Done] outputs → {out_root}")

# =========================================================
# argparse
# =========================================================
def get_args():
    p = argparse.ArgumentParser("Stereo Inference — pad ×16, outputs unpadded @1/4 + full-res, per-image EPE/D1 overlay + error maps, FINAL Stage-2 argmax saving")

    # 데이터
    p.add_argument("--left_dir",  type=str, required=True)
    p.add_argument("--right_dir", type=str, required=True)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width",  type=int, default=1224)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--workers",    type=int, default=4)

    # 모델/디코더 (학습과 동일하게 맞춰야 정확)
    p.add_argument("--max_disp_px", type=int, default=28)
    p.add_argument("--fused_ch",    type=int, default=512)
    p.add_argument("--acv_red_ch",  type=int, default=128)
    p.add_argument("--agg_ch",      type=int, default=128)
    p.add_argument("--use_motif",   type=bool, default=True)
    p.add_argument("--two_stage",   type=bool, default=True)
    p.add_argument("--local_radius", type=int, default=8)

    # 실행
    p.add_argument("--ckpt",       type=str, required=True, help="학습에서 저장한 .pth")
    p.add_argument("--amp",        action="store_true")
    p.add_argument("--output_dir", type=str, default="./infer_out")
    p.add_argument("--log_every",  type=int, default=10)
    p.add_argument("--verbose",    action="store_true")

    # 저장 옵션
    p.add_argument("--save_npy",    action="store_true", help="*.npy 저장")
    p.add_argument("--save_png16",  action="store_true", help="16-bit PNG 저장")
    p.add_argument("--png16_scale", type=float, default=1.0, help="disparity PNG 저장 배율(예: KITTI 256)")
    p.add_argument("--depth_png16_scale", type=float, default=1000.0, help="깊이[m]→mm로 16-bit 저장 등")
    p.add_argument("--save_color",  action="store_true", help="컬러맵 PNG 시각화 저장")
    p.add_argument("--vmax",        type=float, default=None, help="disparity 시각화 상한(px)")
    p.add_argument("--vmax_depth",  type=float, default=None, help="depth 시각화 상한(m)")
    p.add_argument("--vmax_err",    type=float, default=6.0, help="EPE heatmap 시각화 상한(px)")
    p.add_argument("--write_csv",   action="store_true", help="per-image EPE/D1 CSV 저장")
    p.add_argument("--overlay_always", action="store_true", help="GT 없을 때도 'N/A' 오버레이")

    # ★ NEW: overlay 강도 조절
    p.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.85,
        help="debug overlay 강도 (0.0=안보임, 1.0=컬러맵만 보임)",
    )

    # 캘리브(자동/수동)
    p.add_argument("--calib_npy", type=str, default=None)
    p.add_argument("--K_left_npy", type=str, default=None)
    p.add_argument("--focal_px", type=float, default=764.5138549804688)
    p.add_argument("--baseline_m", type=float, default=0.29918420530585865)

    # per-image metrics용 GT depth (선택)
    p.add_argument("--gt_depth_dir",  type=str, default=None, help="GT depth root 디렉토리(파일명 기준 매칭)")
    p.add_argument("--gt_depth_scale", type=float, default=256.0, help="GT depth 스케일(예: 256.0)")

    # === NEW: per-image metrics용 GT disparity (선택) ===
    p.add_argument("--gt_disp_dir",  type=str, default=None, help="GT disparity root 디렉토리(파일명 기준 매칭). 주어지면 depth 대신 이것으로 metric 계산")
    p.add_argument("--gt_disp_scale", type=float, default=256.0, help="정수형 GT disparity 스케일(예: 256.0). float 파일은 스케일 적용 안 함")

    # --- disparity gradient 저장 옵션 ---
    p.add_argument("--save_disp_grads", action="store_true",
                   help="|∂y disp| (vertical)과 |∂x disp| (horizontal) 히트맵 저장")
    p.add_argument("--grad_on", choices=["full", "q", "both"], default="full",
                   help="기울기 계산 대상: full(기본), q(1/4 격자), both")
    p.add_argument("--vmax_grad", type=float, default=None,
                   help="gradient 히트맵 컬러 상한(px/pixel). None이면 이미지별 min-max 자동")

    # --- error/photometric 시각화 옵션 ---
    p.add_argument("--err_bg_color", type=str, default="#1e1e1e",
                   help="에러맵 배경색 (예: '#1e1e1e', 'black', '#f0f0f0')")
    p.add_argument("--err_cmap", type=str, default="magma",
                   help="에러맵 컬러맵 이름(예: 'magma', 'inferno', 'viridis', 'plasma')")

    # --- disparity 컬러바 시각화 옵션 ---
    p.add_argument("--disp_bg_color", type=str, default="#1e1e1e",
                   help="disparity 컬러맵 배경색(유효하지 않은 픽셀 표현)")
    p.add_argument("--disp_cmap", type=str, default="magma",
                   help="disparity 컬러맵 이름(예: 'magma', 'inferno', 'viridis', 'plasma')")

    # --- Photometric 가중치 (학습과 동일 기본값) ---
    p.add_argument("--photo_l1_w",   type=float, default=0.15)
    p.add_argument("--photo_ssim_w", type=float, default=0.85)

    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    run_inference(args)
