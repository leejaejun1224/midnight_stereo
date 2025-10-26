# ms2_metrics.py
import os
import glob
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import cv2
import torch
import torch.nn.functional as F


# ---------------------------
# Config & Utilities
# ---------------------------

@dataclass
class MS2EvalConfig:
    # D1-all = err > max(px_thr, rel_thr * gt)
    d1_px_thr: float = 3.0
    d1_rel_thr: float = 0.05
    # error-n > k px
    bad_px_thrs: Tuple[float, ...] = (1.0, 2.0, 3.0)
    eps: float = 1e-6

    # Bin-weighted depth metrics (DuskTillDawn'24)
    num_bins: int = 5
    max_depth_m: float = 50.0


def _to_float32_gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr.astype(np.float32)


def _first_existing(path_without_ext: str, exts=(".png", ".tiff", ".tif", ".exr", ".npy")) -> Optional[str]:
    # exact match
    for e in exts:
        p = path_without_ext + e
        if os.path.isfile(p): return p
    # wildcard for unknown original ext
    g = glob.glob(path_without_ext + ".*")
    return g[0] if len(g) else None


def _resize_disp_like(disp: np.ndarray, target_hw: Tuple[int,int]) -> np.ndarray:
    """Resize disparity map to (H,W). IMPORTANT: disparity values scale with width ratio."""
    Ht, Wt = target_hw
    H0, W0 = disp.shape
    if (H0, W0) == (Ht, Wt):
        return disp
    # nearest to preserve sparsity, then scale horizontally
    disp_r = cv2.resize(disp, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
    sx = float(Wt) / max(W0, 1)
    return disp_r * sx


def _resize_depth_like(depth: np.ndarray, target_hw: Tuple[int,int]) -> np.ndarray:
    """Depth is metric; value 자체는 스케일 불필요. 최근접으로 크기만 맞춤."""
    Ht, Wt = target_hw
    H0, W0 = depth.shape
    if (H0, W0) == (Ht, Wt):
        return depth
    return cv2.resize(depth, (Wt, Ht), interpolation=cv2.INTER_NEAREST)


def disparity_to_depth(disp_px: torch.Tensor, focal_px: float, baseline_m: float, eps: float = 1e-6) -> torch.Tensor:
    # depth = f * B / disparity
    return (focal_px * baseline_m) / (disp_px.clamp_min(eps))


# ---------------------------
# Batch GT loader (file name match)
# ---------------------------

def load_ms2_gt_batch(
    names: List[str],
    target_hw: Tuple[int,int],
    device: torch.device,
    gt_disp_dir: Optional[str] = None,
    gt_depth_dir: Optional[str] = None,
    gt_disp_scale: float = 1.0,
    gt_depth_scale: float = 1.0,
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Returns:
      dict with keys in {"gt_disp","gt_depth","valid"} where each is [B,1,H,W] tensor or None.
      valid = (gt>0).float() (둘 다 있으면 OR)
    """
    Ht, Wt = target_hw
    bs = len(names)
    disp_list, depth_list = [], []

    for n in names:
        stem = os.path.splitext(n)[0]
        disp, depth = None, None

        if gt_disp_dir:
            p = _first_existing(os.path.join(gt_disp_dir, stem))
            if p is None:
                raise FileNotFoundError(f"[GT] disparity not found for {n} under {gt_disp_dir}")
            arr = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise FileNotFoundError(f"[GT] failed to read: {p}")
            arr = _to_float32_gray(arr) / max(gt_disp_scale, 1e-6)
            arr = _resize_disp_like(arr, (Ht, Wt))
            disp = torch.from_numpy(arr)

        if gt_depth_dir:
            p = _first_existing(os.path.join(gt_depth_dir, stem))
            if p is None:
                raise FileNotFoundError(f"[GT] depth not found for {n} under {gt_depth_dir}")
            if p.endswith(".npy"):
                arr = np.load(p).astype(np.float32)
            else:
                arr = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if arr is None:
                    raise FileNotFoundError(f"[GT] failed to read: {p}")
                arr = _to_float32_gray(arr)
            arr = arr / max(gt_depth_scale, 1e-6)
            arr = _resize_depth_like(arr, (Ht, Wt))
            depth = torch.from_numpy(arr)

        disp_list.append(disp)
        depth_list.append(depth)

    gt_disp = None
    gt_depth = None
    valid   = None

    if any([d is not None for d in disp_list]):
        gt_disp = torch.stack([d if d is not None else torch.zeros((Ht, Wt), dtype=torch.float32) for d in disp_list], dim=0).unsqueeze(1).to(device)
        valid_d = (gt_disp > 0).float()
        valid = valid_d if valid is None else torch.maximum(valid, valid_d)

    if any([d is not None for d in depth_list]):
        gt_depth = torch.stack([d if d is not None else torch.zeros((Ht, Wt), dtype=torch.float32) for d in depth_list], dim=0).unsqueeze(1).to(device)
        valid_z = (gt_depth > 0).float()
        valid = valid_z if valid is None else torch.maximum(valid, valid_z)

    return {"gt_disp": gt_disp, "gt_depth": gt_depth, "valid": valid}


# ---------------------------
# Disparity metrics (torch)
# ---------------------------

def compute_ms2_disparity_metrics(
    pred_disp: torch.Tensor,   # [B,1,H,W]
    gt_disp: torch.Tensor,     # [B,1,H,W]
    valid: torch.Tensor,       # [B,1,H,W] in {0,1}
    cfg: MS2EvalConfig = MS2EvalConfig()
) -> Dict[str, float]:
    with torch.no_grad():
        err = (pred_disp - gt_disp).abs()
        v = (valid > 0.5).float()
        denom = v.sum().clamp_min(cfg.eps)

        epe = (err * v).sum() / denom

        # D1-all: |err| > max(3 px, 0.05 * |gt|)
        thr = torch.maximum(torch.tensor(cfg.d1_px_thr, device=gt_disp.device, dtype=gt_disp.dtype),
                            cfg.d1_rel_thr * gt_disp.abs())
        d1 = ((err > thr).float() * v).sum() / denom * 100.0

        out = {
            "EPE": epe.item(),
            "D1_all": d1.item(),
        }
        for k in cfg.bad_px_thrs:
            bad = ((err > k).float() * v).sum() / denom * 100.0
            out[f"> {int(k)}px"] = bad.item()
        out["valid_px"] = denom.item()
        return out


# ---------------------------
# Depth metrics (torch)
# ---------------------------

def compute_depth_metrics(
    pred_depth_m: torch.Tensor,  # [B,1,H,W]
    gt_depth_m: torch.Tensor,    # [B,1,H,W]
    valid: torch.Tensor,         # [B,1,H,W]
    cfg: MS2EvalConfig = MS2EvalConfig()
) -> Dict[str, float]:
    with torch.no_grad():
        v = (valid > 0.5).float()
        pd = pred_depth_m.clamp_min(cfg.eps)
        gd = gt_depth_m.clamp_min(cfg.eps)
        denom = v.sum().clamp_min(cfg.eps)

        abs_rel = ((pd - gd).abs() / gd * v).sum() / denom
        sq_rel  = (((pd - gd) ** 2) / gd * v).sum() / denom
        rmse    = torch.sqrt((((pd - gd) ** 2) * v).sum() / denom)
        rmse_log= torch.sqrt((((pd.log() - gd.log()) ** 2) * v).sum() / denom)

        ratio = torch.maximum(pd / gd, gd / pd)
        d1  = ((ratio < 1.25).float()  * v).sum() / denom
        d2  = ((ratio < 1.25**2).float()* v).sum() / denom
        d3  = ((ratio < 1.25**3).float()* v).sum() / denom

        return {
            "AbsRel": abs_rel.item(),
            "SqRel":  sq_rel.item(),
            "RMSE":   rmse.item(),
            "RMSElog":rmse_log.item(),
            "δ<1.25": d1.item(),
            "δ<1.25²":d2.item(),
            "δ<1.25³":d3.item(),
            "valid_px": denom.item()
        }


def compute_bin_weighted_depth_metrics(
    pred_depth_m: torch.Tensor, gt_depth_m: torch.Tensor, valid: torch.Tensor,
    max_depth_m: float = 50.0, num_bins: int = 5, cfg: MS2EvalConfig = MS2EvalConfig()
) -> Dict[str, float]:
    """
    Dusk Till Dawn(2024) 제안: 깊이 구간별 metric 계산 후 평균. (기본 0–50 m, 5 bins)
    """
    edges = torch.linspace(0.0, max_depth_m, steps=num_bins + 1, device=pred_depth_m.device)
    bins = [(edges[i], edges[i+1]) for i in range(num_bins)]
    acc = {"AbsRel":0.0, "RMSE":0.0, "δ<1.25":0.0}
    cnt = 0
    for lo, hi in bins:
        m = (valid > 0.5) & (gt_depth_m >= lo) & (gt_depth_m < hi)
        if m.sum() < 1:  # skip empty
            continue
        sub = compute_depth_metrics(pred_depth_m, gt_depth_m, m.float(), cfg)
        acc["AbsRel"] += sub["AbsRel"]; acc["RMSE"] += sub["RMSE"]; acc["δ<1.25"] += sub["δ<1.25"]; cnt += 1
    if cnt == 0:  # no valid bins
        return {"W/AbsRel": float("nan"), "W/RMSE": float("nan"), "W/δ<1.25": float("nan")}
    return { "W/AbsRel": acc["AbsRel"]/cnt, "W/RMSE": acc["RMSE"]/cnt, "W/δ<1.25": acc["δ<1.25"]/cnt }


# ---------------------------
# Formatting helper
# ---------------------------

def fmt_disp_metrics(m: Dict[str,float]) -> str:
    if not m: return ""
    base = f"EPE={m['EPE']:.3f}  D1={m['D1_all']:.2f}%"
    extras = "  ".join([f"{k}={m[k]:.2f}%" for k in m.keys() if k.startswith('> ')])
    return base + (("  " + extras) if extras else "")


def fmt_depth_metrics(m: Dict[str,float]) -> str:
    if not m: return ""
    return f"AbsRel={m['AbsRel']:.3f}  RMSE={m['RMSE']:.3f}  δ1={m['δ<1.25']:.3f}"


def fmt_weighted_depth_metrics(m: Dict[str,float]) -> str:
    if not m: return ""
    return f"W/AbsRel={m['W/AbsRel']:.3f}  W/RMSE={m['W/RMSE']:.3f}  W/δ1={m['W/δ<1.25']:.3f}"
