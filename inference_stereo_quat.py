# -*- coding: utf-8 -*-
import os
import argparse
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# =========================
# (환경에 맞게 경로 조정)
# =========================
from vit_cn import StereoModel
from agg.aggregator import SOTAStereoDecoder

from tools import (
    StereoFolderDataset,
    read_fx_baseline_rgb,
    disparity_to_depth,
    # --- metrics & GT loader (학습 코드와 동일 유틸) ---
    load_ms2_gt_depth_batch,
    compute_ms2_disparity_metrics,
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
    bg_color: str = "#1e1e1e",  # ✅ 배경색(진한 회색)
):
    """
    오른쪽 colorbar가 포함된 컬러 PNG 저장. (에러맵 시각화용)
    - NaN/invalid은 배경색으로 채워 가독성 향상.
    - vmax 미지정 시 99th-percentile로 자동 설정하여 이상치에 덜 민감.
    - vmin은 0으로 고정(오류 맵 가독성).
    - figure/axes facecolor를 저장 시에도 강제로 반영하여 흰 테두리 문제 방지.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, to_rgba

    _ensure_dir(os.path.dirname(path))
    arr = np.array(np_array, dtype=np.float32)

    # 범위 설정
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

    # colormap 준비(❗NaN을 배경색으로 렌더)
    import numpy as _np
    base_cmap = plt.get_cmap(cmap_name)
    cmap = ListedColormap(base_cmap(_np.linspace(0, 1, 256)))
    bg_rgba = to_rgba(bg_color)
    cmap.set_bad(bg_rgba)  # NaN → 배경색

    # 이미지 크기에 맞춰 DPI/figsize 설정(픽셀 보존)
    H, W = arr.shape
    dpi = 200.0
    figsize = (W / dpi, H / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # ✅ 배경색 지정
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax_eff)
    ax.axis("off")

    # 오른쪽 colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if label:
        cbar.set_label(label, rotation=270, labelpad=12)

    plt.tight_layout(pad=0.1)
    # ✅ 저장 시 facecolor 강제 반영
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1, facecolor=fig.get_facecolor())
    plt.close(fig)

def save_gray_png(path, np_array_uint8):
    """
    0/255 같은 8-bit 단일 채널 이미지를 PNG로 저장.
    """
    from PIL import Image
    _ensure_dir(os.path.dirname(path))
    Image.fromarray(np_array_uint8.astype(np.uint8), mode="L").save(path)

def annotate_png_top_left(path: str, text: str, margin: int = 5):
    """
    PNG에 좌측 상단 텍스트 오버레이(작게). 가독성을 위해 반투명 배경 + 외곽선.
    """
    from PIL import Image, ImageDraw, ImageFont

    try:
        img = Image.open(path).convert("RGBA")
    except Exception:
        return  # 이미지가 없으면 스킵

    W, H = img.size
    # 글자 크기: 작은 텍스트(짧은 쪽의 2.5%)
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

    # 반투명 배경 박스
    bg = Image.new("RGBA", (tw + margin * 2, th + margin * 2), (0, 0, 0, 100))
    img.paste(bg, (margin, margin), bg)

    # 흰색 텍스트 + 검은 외곽선
    draw.text((margin * 2, margin * 2), text, font=font,
              fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))

    img = img.convert("RGB")
    img.save(path)

# =========================================================
# 체크포인트 로더(다양한 포맷에 견고)
# =========================================================
def load_checkpoint_robust(ckpt_path: str,
                           stereo: torch.nn.Module,
                           decoder: torch.nn.Module,
                           device,
                           verbose: bool = True):
    """
    - 1순위: tools.resume_from_checkpoint 사용 (학습과 동일 경로)
    - 실패 시: 다양한 키 패턴('state_dict', 'model', 'ckpt_model', 'stereo/decoder' 등) 자동 감지 로드
    """
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
    """
    EPE(px), D1_all(%) 반환. 유효 GT 없으면 (None, None).
    """
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

# =========================================================
# disparity gradient (|∂x d|, |∂y d|) 계산 유틸
# =========================================================
@torch.no_grad()
def disparity_gradients_abs(disp: torch.Tensor, keep_size: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    disp: [B,1,H,W] float (px 단위 권장)
    returns:
      grad_y_abs: [B,1,H,W] = |∂y d| (세로 방향 변화량의 크기)
      grad_x_abs: [B,1,H,W] = |∂x d| (가로 방향 변화량의 크기)
    """
    # forward difference
    gx = disp[:, :, :, 1:] - disp[:, :, :, :-1]   # [B,1,H,W-1]
    gy = disp[:, :, 1:, :] - disp[:, :, :-1, :]   # [B,1,H-1,W]
    grad_x_abs = gx.abs()
    grad_y_abs = gy.abs()

    if keep_size:
        # (left,right,top,bottom) 순서. 우/하 방향으로 한 칸 패딩해 크기 복원.
        grad_x_abs = F.pad(grad_x_abs, (0, 1, 0, 0), mode="replicate")
        grad_y_abs = F.pad(grad_y_abs, (0, 0, 0, 1), mode="replicate")

    return grad_y_abs, grad_x_abs

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
    out_vis = _ensure_dir(os.path.join(out_root, "vis"))  # 시각화 폴더
    out_error = _ensure_dir(os.path.join(out_root, "error"))  # 시각화 폴더

    # --- metrics 로깅 CSV (선택)
    metrics_csv_path = os.path.join(out_root, "metrics_per_image.csv")
    metrics_csv_fp = open(metrics_csv_path, "w", encoding="utf-8") if args.write_csv else None
    if metrics_csv_fp is not None:
        metrics_csv_fp.write("name,EPE_px,D1_all_percent\n")

    # --- 추론 ---
    torch.set_grad_enabled(False)
    context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad

    with context():
        for it, (imgL, imgR, names) in enumerate(loader, start=1):
            imgL = imgL.to(device, non_blocking=True)  # ImageNet 정규화 가정
            imgR = imgR.to(device, non_blocking=True)

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

            Bsz = disp_q_full_px.shape[0]
            # 4) (선택) GT depth 로딩 -> per-image metrics 계산
            gt_depth_q = None
            gt_depth = None
            has_fb = (args.focal_px > 0.0 and args.baseline_m > 0.0)
            if args.gt_depth_dir is not None:
                gt_depth_q = load_ms2_gt_depth_batch(
                    names=names,
                    gt_depth_dir=args.gt_depth_dir,
                    scale=args.gt_depth_scale,
                    target_hw=disp_q_full_px.shape[-2:],  # (Hq, Wq)
                    device=imgL.device
                )
            if args.gt_depth_dir is not None:
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

                # 5-1) *.npy 저장
                if args.save_npy:
                    save_npy(os.path.join(out_disp_qpx,     f"{stem}.npy"), disp_q_qpx_np)
                    save_npy(os.path.join(out_disp_qpx_px,  f"{stem}.npy"), disp_q_full_px_np)
                    save_npy(os.path.join(out_disp_full_px, f"{stem}.npy"), disp_full_px_np)

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
                    if args.save_npy:
                        save_npy(os.path.join(out_depth_m, f"{stem}.npy"), depth_m_np)
                    if args.save_png16:
                        save_png_16u(os.path.join(out_depth_m, f"{stem}.png"), depth_m_np, scale=float(args.depth_png16_scale))

                # 5-4) 시각화 PNG (disparity/깊이 + 오버레이)
                epe_full, d1_full = None, None
                text_overlay = None
                text_overlay_q = None
                if gt_depth_q is not None and has_fb:
                    epe_q, d1_q = compute_epe_d1_per_item(
                        pred_disp_q_px=disp_q_full_px[bi:bi+1],
                        gt_depth_q_m=gt_depth_q[bi:bi+1],
                        focal_px=float(args.focal_px),
                        baseline_m=float(args.baseline_m),
                    )
                    text_overlay_q = f"EPE 1/4 {epe_q:.3f} px | D1 1/4 {d1_q:.2f}%" if (epe_q is not None and d1_q is not None) else "EPE/D1 1/4 : N/A"
                elif args.overlay_always:
                    text_overlay_q = "EPE/D1 1/4 : N/A"

                if (gt_depth is not None) and has_fb:
                    epe_full, d1_full = compute_epe_d1_per_item(
                        pred_disp_q_px=disp_full_px[bi:bi+1],
                        gt_depth_q_m=gt_depth[bi:bi+1],
                        focal_px=float(args.focal_px),
                        baseline_m=float(args.baseline_m),
                    )
                    text_overlay = f"EPE {epe_full:.3f} px | D1 {d1_full:.2f}%" if (epe_full is not None and d1_full is not None) else "EPE/D1 : N/A"
                elif args.overlay_always:
                    text_overlay = "EPE/D1 : N/A"

                # disparity full-res(px)
                if args.save_color:
                    # 1/4 해상도(px) 보조 시각화
                    p2 = os.path.join(out_disp_qpx_px, f"{stem}_disp_1_4_px.png")
                    save_colormap_png(p2, disp_q_full_px_np, vmax=args.vmax)
                    if text_overlay_q:
                        annotate_png_top_left(p2, text_overlay_q)

                    p2 = os.path.join(out_disp_full_px, f"{stem}_disp_px.png")
                    save_colormap_png(p2, disp_full_px_np, vmax=args.vmax)
                    if text_overlay:
                        annotate_png_top_left(p2, text_overlay)

                # === NEW: disparity gradient 시각화 저장 ===
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
                        if args.save_npy:
                            save_npy(os.path.join(out_vis, f"{stem}_gradV_full.npy"), gy_full_np)
                            save_npy(os.path.join(out_vis, f"{stem}_gradH_full.npy"), gx_full_np)

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
                        if args.save_npy:
                            save_npy(os.path.join(out_vis, f"{stem}_gradV_1_4.npy"), gy_q_np)
                            save_npy(os.path.join(out_vis, f"{stem}_gradH_1_4.npy"), gx_q_np)

                # --- Error maps (GT가 있고 fx/B가 있을 때) ---
                if (gt_depth is not None) and has_fb:
                    # 유효 mask (full-res)
                    valid_full = (gt_depth[bi:bi+1] > 0).float()  # [1,1,H,W]

                    # GT disparity(px@full-res)
                    gt_disp_full_px = (float(args.focal_px) * float(args.baseline_m)) / \
                                      torch.clamp(gt_depth[bi:bi+1], min=1e-6)  # [1,1,H,W]

                    # EPE map (px) — FULL RES 직접 계산
                    epe_map_full = torch.abs(disp_full_px[bi:bi+1] - gt_disp_full_px) * valid_full  # [1,1,H,W]

                    # numpy 변환 + invalid=NaN 처리(EPE)
                    epe_map_full_np  = epe_map_full.squeeze(0).squeeze(0).detach().cpu().numpy()
                    valid_full_np    = valid_full.squeeze(0).squeeze(0).detach().cpu().numpy().astype(bool)
                    # epe_map_full_np[~valid_full_np] = np.nan  # NaN은 함수에서 배경색으로 채움

                    if args.save_npy:
                        save_npy(os.path.join(out_vis, f"{stem}_err_epe_full_px.npy"), epe_map_full_np)

                    # === 요청 기능 — {stem}_error map.png 저장(오른쪽 color bar, 진한 배경) ===
                    if args.save_color:
                        vmax_err = args.vmax_err if (args.vmax_err is not None and args.vmax_err > 0) else None
                        p_err_full_png = os.path.join(out_error, f"{stem}_error map.png")
                        save_colormap_png_with_colorbar(
                            p_err_full_png,
                            epe_map_full_np,
                            vmax=vmax_err,
                            cmap_name=args.err_cmap,   # <- 선택 컬러맵
                            label="EPE (px)",
                            bg_color=args.err_bg_color # <- 진한 배경
                        )
                        if text_overlay:
                            annotate_png_top_left(p_err_full_png, text_overlay)

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
    p = argparse.ArgumentParser("Stereo Inference — pad ×16, outputs unpadded @1/4 + full-res, per-image EPE/D1 overlay + error maps (with colorbar & dark bg)")

    # 데이터
    p.add_argument("--left_dir",  type=str, required=True)
    p.add_argument("--right_dir", type=str, required=True)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width",  type=int, default=1224)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--workers",    type=int, default=4)

    # 모델/디코더 (학습과 동일하게 맞춰야 정확)
    p.add_argument("--max_disp_px", type=int, default=60)
    p.add_argument("--fused_ch",    type=int, default=320)
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
    p.add_argument("--vmax_err",    type=float, default=None, help="EPE heatmap 시각화 상한(px)")
    p.add_argument("--write_csv",   action="store_true", help="per-image EPE/D1 CSV 저장")
    p.add_argument("--overlay_always", action="store_true", help="GT 없을 때도 'N/A' 오버레이")

    # 캘리브(자동/수동)
    p.add_argument("--calib_npy", type=str, default=None)
    p.add_argument("--K_left_npy", type=str, default=None)
    p.add_argument("--focal_px", type=float, default=764.5138549804688)
    p.add_argument("--baseline_m", type=float, default=0.29918420530585865)

    # per-image metrics용 GT depth
    p.add_argument("--gt_depth_dir",  type=str, default=None, help="GT depth root 디렉토리(파일명 기준 매칭)")
    p.add_argument("--gt_depth_scale", type=float, default=256.0, help="GT depth 스케일(예: 256.0)")

    # --- disparity gradient 저장 옵션 ---
    p.add_argument("--save_disp_grads", action="store_true",
                   help="|∂y disp| (vertical)과 |∂x disp| (horizontal) 히트맵 저장")
    p.add_argument("--grad_on", choices=["full", "q", "both"], default="full",
                   help="기울기 계산 대상: full(기본), q(1/4 격자), both")
    p.add_argument("--vmax_grad", type=float, default=None,
                   help="gradient 히트맵 컬러 상한(px/pixel). None이면 이미지별 min-max 자동")

    # --- NEW: error map 시각화 옵션 ---
    p.add_argument("--err_bg_color", type=str, default="#1e1e1e",
                   help="에러맵 배경색 (예: '#1e1e1e', 'black', '#f0f0f0')")
    p.add_argument("--err_cmap", type=str, default="magma",
                   help="에러맵 컬러맵 이름(예: 'magma', 'inferno', 'viridis', 'plasma')")

    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    run_inference(args)
