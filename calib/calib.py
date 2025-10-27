# ms2_calib.py
import os
import numpy as np
from typing import Optional, Tuple

def _safe_load(path: Optional[str]):
    if not path or not os.path.isfile(path):
        return None
    obj = np.load(path, allow_pickle=True)
    # npz면 dict처럼 동작
    if isinstance(obj, np.lib.npyio.NpzFile):
        data = {k: obj[k] for k in obj.files}
        return data
    # npy면 배열 또는 dict(allow_pickle)일 수 있음
    try:
        if hasattr(obj, "item"):
            return obj.item()
    except Exception:
        pass
    return obj  # ndarray or raw

def _extract_from_dict(d, modality="rgb") -> Tuple[Optional[float], Optional[float]]:
    """d: dict 계열에서 fx, baseline을 추출"""
    if not isinstance(d, dict):
        return None, None
    node = d.get(modality, d)  # 'rgb' 노드가 있으면 우선
    if not isinstance(node, dict):
        node = d

    # 후보 키들
    k_candidates = ["K_left", "K", "intrinsic_left", "intrinsic"]
    t_candidates = ["T_lr", "T", "extrinsic_lr", "extrinsic", "T_left_right"]

    K = None
    for k in k_candidates:
        if k in node:
            K = node[k]
            break
    T = None
    for t in t_candidates:
        if t in node:
            T = node[t]
            break

    fx = None
    if isinstance(K, np.ndarray) and K.ndim >= 2 and K.shape[0] >= 1 and K.shape[1] >= 1:
        fx = float(K[0, 0])

    baseline = None
    if isinstance(T, np.ndarray):
        if T.shape == (4, 4):
            baseline = abs(float(T[0, 3]))
        elif T.shape in [(3,), (3, 1)]:
            baseline = abs(float(T[0]))

    return fx, baseline

def _find_near_calib(left_dir: str) -> Optional[str]:
    # left_dir, 부모, 조부모 폴더에서 calib 파일 탐색
    candidates = ["calib.npy", "calib.npz", "calib_rgb.npy"]
    ups = [left_dir, os.path.dirname(left_dir), os.path.dirname(os.path.dirname(left_dir))]
    for up in ups:
        for name in candidates:
            p = os.path.join(up, name)
            if os.path.isfile(p):
                return p
    return None

def resolve_fx_baseline(
    left_dir: str,
    focal_px: float = 0.0,
    baseline_m: float = 0.0,
    calib_npy: Optional[str] = None,
    K_left_npy: Optional[str] = None,
    T_lr_npy: Optional[str] = None,
    modality: str = "rgb",
) -> Tuple[Optional[float], Optional[float], str]:
    """
    우선순위에 따라 fx, baseline을 찾아 반환.
    returns: fx, baseline, source_message
    """
    # 1) 이미 제공된 값
    if focal_px and baseline_m:
        return float(focal_px), float(baseline_m), "args(focal_px, baseline_m)"

    # 2) calib 파일에서 읽기
    for cand in [calib_npy, _find_near_calib(left_dir)]:
        obj = _safe_load(cand)
        if obj is None:
            continue
        fx, bl = None, None
        if isinstance(obj, dict):
            fx, bl = _extract_from_dict(obj, modality=modality)
        elif isinstance(obj, np.ndarray):
            # npy에 dict가 안 담겼고, (예외적으로) 4x4, 3x3 단일 행렬일 수도 있음
            if obj.shape == (3, 3) and not focal_px:
                fx = float(obj[0, 0])
            if obj.shape == (4, 4) and not baseline_m:
                bl = abs(float(obj[0, 3]))
        if (fx and bl):
            return fx, bl, f"calib:{cand}"
        # 하나만 있어도 기록
        if fx and not focal_px:
            focal_px = fx
            src_fx = f"calib:{cand}"
        if bl and not baseline_m:
            baseline_m = bl
            src_bl = f"calib:{cand}"
        if focal_px and baseline_m:
            return focal_px, baseline_m, f"calib:{cand}"

    # 3) 개별 파일
    if K_left_npy and not focal_px:
        K = _safe_load(K_left_npy)
        if isinstance(K, np.ndarray) and K.shape == (3, 3):
            focal_px = float(K[0, 0])
    if T_lr_npy and not baseline_m:
        T = _safe_load(T_lr_npy)
        if isinstance(T, np.ndarray) and T.shape == (4, 4):
            baseline_m = abs(float(T[0, 3]))

    src = "K_left_npy/T_lr_npy"
    if focal_px and baseline_m:
        return focal_px, baseline_m, src

    # 4) 부분만 구해진 경우라도 반환
    if focal_px and not baseline_m:
        return focal_px, None, src
    if baseline_m and not focal_px:
        return None, baseline_m, src

    return None, None, "not_found"
