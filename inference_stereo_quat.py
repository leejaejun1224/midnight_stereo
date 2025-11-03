# -*- coding: utf-8 -*-
import os
import argparse
import math
from typing import Tuple

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
)

# =========================================================
# padding 유틸 (학습 코드와 동일한 동작)
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
# 저장 유틸
# =========================================================
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _basename_wo_ext(path):
    return os.path.splitext(os.path.basename(path))[0]

def save_npy(path, np_array):
    import numpy as np
    _ensure_dir(os.path.dirname(path))
    np.save(path, np_array.astype(np.float32), allow_pickle=False)

def save_png_16u(path, np_array, scale: float = 1.0):
    """
    disparity를 16-bit PNG로 저장.
    - np_array: float32 HxW (px 또는 m 단위)
    - scale: 저장시 배율(예: KITTI 호환 256 배율 등). 기본 1.0
    """
    import numpy as np
    from PIL import Image
    _ensure_dir(os.path.dirname(path))
    arr = np_array * float(scale)
    arr = np.clip(arr, 0, 65535).astype(np.uint16)
    Image.fromarray(arr, mode="I;16").save(path)

def save_colormap_png(path, np_array, vmax=None, cmap_name="magma"):
    """
    시각화용 컬러 PNG 저장(8-bit). vmax 미지정 시 자동 min-max.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(os.path.dirname(path))
    vmin = float(np_array[np.isfinite(np_array)].min()) if np.isfinite(np_array).any() else 0.0
    if vmax is None:
        vmax = float(np_array[np.isfinite(np_array)].max()) if np.isfinite(np_array).any() else 1.0
    vmax = max(vmax, vmin + 1e-6)

    norm = (np_array - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    colored = (cmap(norm)[..., :3] * 255.0).astype("uint8")

    from PIL import Image
    Image.fromarray(colored).save(path)

# =========================================================
# 체크포인트 로더(학습 저장 형식 모두 대응)
# =========================================================
def load_checkpoint(ckpt_path: str, stereo: torch.nn.Module, decoder: torch.nn.Module, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    def _strip_module(sd):
        return {k.replace("module.", ""): v for k, v in sd.items()}

    loaded = False

    # 1) 학습 코드의 기본 저장 형식
    if isinstance(ckpt, dict) and "stereo" in ckpt and "decoder" in ckpt:
        stereo.load_state_dict(_strip_module(ckpt["stereo"]), strict=False)
        decoder.load_state_dict(_strip_module(ckpt["decoder"]), strict=False)
        loaded = True

    # 2) 기타 포맷 대비
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = _strip_module(ckpt["state_dict"])
        # stereo.* / decoder.*로 나뉘어 있을 수도, 하나로 붙어 있을 수도 있음
        stereo_sd = {k.replace("stereo.", ""): v for k, v in sd.items() if k.startswith("stereo.")}
        decoder_sd = {k.replace("decoder.", ""): v for k, v in sd.items() if k.startswith("decoder.")}
        if len(stereo_sd) > 0:
            stereo.load_state_dict(stereo_sd, strict=False); loaded = True
        if len(decoder_sd) > 0:
            decoder.load_state_dict(decoder_sd, strict=False); loaded = True
        # 만약 하나로 합쳐져 있으면 전부 stereo에 넣었다가 실패하면 decoder에도 시도
        if not loaded:
            try:
                stereo.load_state_dict(sd, strict=False); loaded = True
            except Exception:
                try:
                    decoder.load_state_dict(sd, strict=False); loaded = True
                except Exception:
                    pass

    if not loaded:
        raise RuntimeError(f"지원하지 않는 체크포인트 형식: {ckpt_path}")

# =========================================================
# 추론 루프
# =========================================================
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp) and torch.cuda.is_available()

    # --- fx / baseline 자동 로딩(학습 코드와 동일 로직 사용) ---
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

    # --- 모델 구성(학습과 동일 하이퍼) ---
    stereo = StereoModel(
        freeze_vit=True,           # inference에 영향 없음
        amp=amp_enabled,
        autopad_to_8=False,        # 입력 패딩을 외부에서 처리(학습과 동일)
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
    load_checkpoint(args.ckpt, stereo, decoder, device)
    if args.verbose:
        print(f"[Load] checkpoint: {args.ckpt}")

    # --- 출력 폴더 구성 ---
    out_root = _ensure_dir(args.output_dir)
    out_disp_qpx = _ensure_dir(os.path.join(out_root, "disp_1_4_qpx"))
    out_disp_qpx_px = _ensure_dir(os.path.join(out_root, "disp_1_4_px"))
    out_disp_full_px = _ensure_dir(os.path.join(out_root, "disp_full_px"))
    out_depth_m = _ensure_dir(os.path.join(out_root, "depth_m")) if (args.focal_px > 0 and args.baseline_m > 0) else None
    out_vis = _ensure_dir(os.path.join(out_root, "vis")) if args.save_color else None

    # --- 추론 ---
    torch.set_grad_enabled(False)
    if hasattr(torch, "inference_mode"):
        context = torch.inference_mode
    else:
        context = torch.no_grad

    with context():
        for it, (imgL, imgR, names) in enumerate(loader, start=1):
            imgL = imgL.to(device, non_blocking=True)  # ImageNet 정규화 가정(tools.StereoFolderDataset)
            imgR = imgR.to(device, non_blocking=True)

            # 1) 입력 16배수 패딩
            imgL_pad, pad = pad_to_multiple(imgL, mult=16, mode="replicate")
            imgR_pad, _   = pad_to_multiple(imgR, mult=16, mode="replicate")

            assert pad[0] % 4 == 0 and pad[1] % 4 == 0, "pad must be divisible by 4 (학습 코드와 동일 전제)"

            # 2) 모델 추론
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                bb_out = stereo(imgL_pad, imgR_pad)
                pred   = decoder(bb_out)

                # 3) 출력 언패드(모두 1/4 해상도 좌표계)
                pad_q = (pad[0] // 4, pad[1] // 4)

                # 디코더가 full‑px 기준으로 낸다고 가정(학습 코드와 동일):
                #   1/4 해상도에서 warp 등에 쓰기 위해서는 /4.0(→ 1/4‑px 단위)
                disp_q_pad_full_px = pred["disp_1_4"]              # [B,1,Hq_pad,Wq_pad], unit: px @ full-res
                disp_q_full_px     = unpad_last2(disp_q_pad_full_px, pad_q)   # [B,1,Hq,Wq], unit: px
                disp_q_qpx         = disp_q_full_px / 4.0                     # [B,1,Hq,Wq], unit: 1/4‑px

                # 원본 해상도(px) 업샘플
                disp_full_px = F.interpolate(
                    disp_q_full_px, scale_factor=4, mode="bilinear", align_corners=False
                )  # [B,1,H,W], unit: px

            # 4) 저장
            # 배치 차원 제거
            disp_q_qpx_np     = disp_q_qpx.squeeze(0).squeeze(0).detach().cpu().numpy()
            disp_q_full_px_np = disp_q_full_px.squeeze(0).squeeze(0).detach().cpu().numpy()
            disp_full_px_np   = disp_full_px.squeeze(0).squeeze(0).detach().cpu().numpy()

            name = names[0] if isinstance(names, (list, tuple)) else names
            stem = _basename_wo_ext(name)

            # 4-1) NPY 저장
            if args.save_npy:
                save_npy(os.path.join(out_disp_qpx,     f"{stem}.npy"), disp_q_qpx_np)
                save_npy(os.path.join(out_disp_qpx_px,  f"{stem}.npy"), disp_q_full_px_np)
                save_npy(os.path.join(out_disp_full_px, f"{stem}.npy"), disp_full_px_np)

            # 4-2) 16-bit PNG(선택) — 보관/타툴 호환
            if args.save_png16:
                # 보통 disparity PNG는 특정 배율(예: KITTI 256x)로 저장하기도 함
                scale = float(args.png16_scale)
                save_png_16u(os.path.join(out_disp_full_px, f"{stem}.png"), disp_full_px_np, scale=scale)

            # 4-3) 깊이(m) (선택)
            if out_depth_m is not None:
                # Z = fx * B / d(px)
                depth_m_np = disparity_to_depth(
                    torch.from_numpy(disp_full_px_np).unsqueeze(0).unsqueeze(0),
                    fx=float(args.focal_px), baseline=float(args.baseline_m)
                ).squeeze(0).squeeze(0).numpy()
                if args.save_npy:
                    save_npy(os.path.join(out_depth_m, f"{stem}.npy"), depth_m_np)
                if args.save_png16:
                    save_png_16u(os.path.join(out_depth_m, f"{stem}.png"), depth_m_np, scale=float(args.depth_png16_scale))

            # 4-4) 시각화(컬러 PNG) (선택)
            if out_vis is not None:
                # disparity full-res(px) 컬러
                save_colormap_png(os.path.join(out_vis, f"{stem}_disp_full_px.png"), disp_full_px_np, vmax=args.vmax)
                # 1/4 해상도(px)도 참고 시각화
                save_colormap_png(os.path.join(out_vis, f"{stem}_disp_1_4_px.png"), disp_q_full_px_np, vmax=args.vmax)
                # fx/B 있으면 depth도 시각화
                if out_depth_m is not None:
                    # 너무 큰 값 outlier 억제를 위해 vmax_depth가 있으면 사용
                    vmax_depth = args.vmax_depth if (args.vmax_depth is not None and args.vmax_depth > 0) else None
                    save_colormap_png(os.path.join(out_vis, f"{stem}_depth_m.png"),
                                      depth_m_np if out_depth_m is not None else disp_full_px_np,
                                      vmax=vmax_depth)

            if args.verbose and (it % args.log_every == 0):
                print(f"[Infer {it:05d}/{len(loader)}] saved: {stem}")

    if args.verbose:
        print(f"[Done] outputs → {out_root}")

# =========================================================
# argparse
# =========================================================
def get_args():
    p = argparse.ArgumentParser("Stereo Inference — inputs padded to ×16, outputs unpadded @1/4 + full-res upsample")

    # 데이터
    p.add_argument("--left_dir",  type=str, required=True)
    p.add_argument("--right_dir", type=str, required=True)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width",  type=int, default=1224)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--workers",    type=int, default=4)

    # 모델/디코더 (학습과 동일하게 맞춰야 함)
    p.add_argument("--max_disp_px", type=int, default=192)
    p.add_argument("--fused_ch",    type=int, default=256)
    p.add_argument("--acv_red_ch",  type=int, default=48)
    p.add_argument("--agg_ch",      type=int, default=32)
    p.add_argument("--use_motif",   type=bool, default=True)
    p.add_argument("--two_stage",   type=bool, default=False)
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
    p.add_argument("--png16_scale", type=float, default=1.0, help="PNG 저장 배율(예: KITTI 호환 256)")
    p.add_argument("--depth_png16_scale", type=float, default=1000.0, help="깊이[m]→mm로 16-bit 저장 등")
    p.add_argument("--save_color",  action="store_true", help="컬러맵 PNG 시각화 저장")
    p.add_argument("--vmax",        type=float, default=None, help="disparity 시각화 상한(px)")
    p.add_argument("--vmax_depth",  type=float, default=None, help="depth 시각화 상한(m)")

    # 캘리브(자동/수동) — 있으면 깊이 산출
    p.add_argument("--calib_npy", type=str, default=None)
    p.add_argument("--K_left_npy", type=str, default=None)
    p.add_argument("--focal_px", type=float, default=0.0)
    p.add_argument("--baseline_m", type=float, default=0.0)

    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    run_inference(args)
