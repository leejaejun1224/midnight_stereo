# -*- coding: utf-8 -*-
import os
import math
import csv
import argparse
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from logger import save_args_as_text
# =========================
# (환경에 맞게 경로 조정)
# =========================
from vit_cn import StereoModel
from agg.aggregator import SOTAStereoDecoder
from agg.decoder_selective_igev import IGEVStereo
from logger import save_args_txt_dynamic

from tools import (
    set_seed,
    StereoFolderDataset,
    denorm_imagenet,
    warp_right_to_left_image,
    get_disparity_smooth_loss,
    # --- 실시간 평가 ---
    read_fx_baseline_rgb,
    load_ms2_gt_depth_batch,
    compute_ms2_disparity_metrics,
    compute_depth_metrics,
    compute_bin_weighted_depth,
    disparity_to_depth,
    PhotometricLoss
)
from losses import (
    DirectionalRelScaleDispLoss,   # (세로 [-1,0]/[0,1] + cossim_feat 버전)
    FeatureReprojLoss,
    AdaptiveWindowDistillLoss,
    # PhotometricLoss,
)
torch.backends.cudnn.benchmark = True

# (선택) 체크포인트 유틸
try:
    from tools import save_checkpoint, resume_from_checkpoint, _move_optimizer_state_to_device
    HAS_CKPT = True
except Exception:
    HAS_CKPT = False


# =========================================================
# RealtimeEvalLogger
# =========================================================
class RealtimeEvalLogger:
    def __init__(self, save_dir: str,
                 filename_txt: str = "realtime_eval.txt",
                 filename_csv: str = "realtime_eval.csv"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.txt_path = os.path.join(self.save_dir, filename_txt)
        self.csv_path = os.path.join(self.save_dir, filename_csv)
        self.history = []
        self._load_history_if_exists()
        self._epoch_reset()

    def _load_history_if_exists(self):
        if not os.path.isfile(self.csv_path): return
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                def f32(x):
                    try: return float(x)
                    except: return float("nan")
                try: ep = int(row.get("epoch", "0"))
                except: ep = 0
                self.history.append({
                    "epoch": ep,
                    "Disp_EPE":      f32(row.get("Disp_EPE", "nan")),
                    "Disp_D1_all":   f32(row.get("Disp_D1_all", "nan")),
                    "Disp_gt1px":    f32(row.get("Disp_gt1px", "nan")),
                    "Disp_gt2px":    f32(row.get("Disp_gt2px", "nan")),
                    "Depth_AbsRel":  f32(row.get("Depth_AbsRel", "nan")),
                    "Depth_RMSE":    f32(row.get("Depth_RMSE", "nan")),
                    "Depth_delta1":  f32(row.get("Depth_delta1", "nan")),
                    "W_AbsRel":      f32(row.get("W_AbsRel", "nan")),
                    "W_RMSE":        f32(row.get("W_RMSE", "nan")),
                    "W_delta1":      f32(row.get("W_delta1", "nan")),
                })

    def _epoch_reset(self):
        self._acc = {
            "Disp_EPE":     [],
            "Disp_D1_all":  [],
            "Disp_gt1px":   [],
            "Disp_gt2px":   [],
            "Depth_AbsRel": [],
            "Depth_RMSE":   [],
            "Depth_delta1": [],
            "W_AbsRel":     [],
            "W_RMSE":       [],
            "W_delta1":     [],
        }

    def start_epoch(self): self._epoch_reset()

    def add_batch(self, disp_metrics: Optional[Dict], depth_metrics: Optional[Dict], depth_w: Optional[Dict] = None):
        def _add(key, val):
            if val is None: return
            try:
                v = float(val)
                if math.isfinite(v):
                    self._acc[key].append(v)
            except: pass

        if disp_metrics:
            _add("Disp_EPE",    disp_metrics.get("EPE"))
            _add("Disp_D1_all", disp_metrics.get("D1_all"))
            _add("Disp_gt1px",  disp_metrics.get("> 1px"))
            _add("Disp_gt2px",  disp_metrics.get("> 2px"))

        if depth_metrics:
            _add("Depth_AbsRel", depth_metrics.get("AbsRel"))
            _add("Depth_RMSE",   depth_metrics.get("RMSE"))
            _add("Depth_delta1", depth_metrics.get("δ<1.25"))

        if depth_w:
            _add("W_AbsRel", depth_w.get("W/AbsRel"))
            _add("W_RMSE",   depth_w.get("W/RMSE"))
            _add("W_delta1", depth_w.get("W/δ<1.25"))

    def _mean(self, key: str) -> float:
        arr = self._acc.get(key, [])
        return float(sum(arr)/len(arr)) if len(arr) > 0 else float("nan")

    def end_epoch(self, epoch: int):
        means = {k: self._mean(k) for k in self._acc.keys()}
        entry = {"epoch": epoch, **means}
        self.history.append(entry)
        self._append_txt(entry)
        self._append_csv(entry)
        self._save_plots()

    def _append_txt(self, e: Dict):
        line = (
            f"[Epoch {e['epoch']:03d}] "
            f"[Disp] EPE={e['Disp_EPE']:.3f}  D1={e['Disp_D1_all']:.2f}%  "
            f">1px={e['Disp_gt1px']:.2f}%  >2px={e['Disp_gt2px']:.2f}%  "
            f"[Depth] AbsRel={e['Depth_AbsRel']:.3f}  RMSE={e['Depth_RMSE']:.3f}  "
            f"δ1={e['Depth_delta1']:.3f}  "
            f"[Depth-W] W/AbsRel={e['W_AbsRel']:.3f}  W/RMSE={e['W_RMSE']:.3f}  "
            f"W/δ1={e['W_delta1']:.3f}\n"
        )
        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(line)

    def _append_csv(self, e: Dict):
        header = [
            "epoch",
            "Disp_EPE","Disp_D1_all","Disp_gt1px","Disp_gt2px",
            "Depth_AbsRel","Depth_RMSE","Depth_delta1",
            "W_AbsRel","W_RMSE","W_delta1"
        ]
        is_new = not os.path.isfile(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if is_new: w.writeheader()
            row = {k: e.get(k, float("nan")) for k in header}
            w.writerow(row)

    def _save_plots(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys = [
            ("Disp_EPE",    "EPE (px)"),
            ("Disp_D1_all", "D1-all (%)"),
            ("Disp_gt1px",  "> 1 px (%)"),
            ("Disp_gt2px",  "> 2 px (%)"),
            ("Depth_AbsRel","AbsRel"),
            ("Depth_RMSE",  "RMSE (m)"),
            ("Depth_delta1","δ<1.25"),
            ("W_AbsRel",    "Weighted AbsRel"),
            ("W_RMSE",      "Weighted RMSE (m)"),
            ("W_delta1",    "Weighted δ<1.25"),
        ]
        for key, ylabel in keys:
            xs, ys = [], []
            for e in self.history:
                v = e.get(key, float("nan"))
                if isinstance(v, (int, float)) and math.isfinite(v):
                    xs.append(e["epoch"]); ys.append(v)
            if len(xs) == 0: continue
            plt.figure()
            plt.plot(xs, ys, marker="o")
            plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(key)
            plt.grid(True, linestyle="--", alpha=0.4)
            out = os.path.join(self.save_dir, f"plot_{key}.png")
            plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


# =========================
# padding 유틸 (16의 배수 권장)
# =========================
def pad_to_multiple(x: torch.Tensor, mult: int = 16, mode: str = "replicate"):
    H, W = x.shape[-2], x.shape[-1]
    pad_r = (-W) % mult
    pad_b = (-H) % mult
    if pad_r or pad_b:
        x = F.pad(x, (0, pad_r, 0, pad_b), mode=mode)
    return x, (pad_b, pad_r)

def unpad_last2(x: torch.Tensor, pad: Tuple[int, int]):
    pad_b, pad_r = pad
    if pad_b == 0 and pad_r == 0: return x
    H, W = x.shape[-2], x.shape[-1]
    return x[..., :H - pad_b, :W - pad_r].contiguous()

def unpad_bhwc(x: torch.Tensor, pad_q: Tuple[int, int]) -> torch.Tensor:
    """
    x: [B, H, W, C] (BHWC)
    pad_q: (pad_b//4, pad_r//4) — 아래/오른쪽 패딩 길이 (1/4 격자 기준)
    """
    pad_b, pad_r = pad_q
    if pad_b > 0:
        x = x[:, :-pad_b, :, :]
    if pad_r > 0:
        x = x[:, :, :-pad_r, :]
    return x.contiguous()


# =========================
# 학습 루프 (입력 16배수 패딩 → 모델 → 출력 언패드, 모든 손실 @1/4 + Full-res 추가)
# =========================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # --- fx / baseline 자동 로딩 (실시간 평가용) ---
    fx, B, src = read_fx_baseline_rgb(
        intrinsic_left_npy=getattr(args, "K_left_npy", None),
        calib_npy=getattr(args, "calib_npy", None)
    )
    if getattr(args, "focal_px", 0.0) <= 0.0 and fx is not None:
        args.focal_px = float(fx)
    if getattr(args, "baseline_m", 0.0) <= 0.0 and B is not None:
        args.baseline_m = float(B)
    if args.realtime_test:
        print(f"[Calib] fx(px)={getattr(args,'focal_px',0.0):.6f}  baseline(m)={getattr(args,'baseline_m',0.0):.6f}  src={src}")

    os.makedirs(args.save_dir, exist_ok=True)
    args_txt = save_args_txt_dynamic(args, save_dir=args.save_dir, filename="args.txt")
    print(f"[Args] Saved to {args_txt}")

    # --- 데이터셋/로더 ---
    dataset = StereoFolderDataset(args.left_dir, args.right_dir,
                                  height=args.height, width=args.width)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True)

    # --- 모델/디코더 ---
    stereo = StereoModel(
        freeze_vit=True,
        amp=args.amp,
        autopad_to_8=False,   # << 입력은 외부에서 16배수 패딩
    ).to(device).train()

    decoder = SOTAStereoDecoder(
        max_disp_px=args.max_disp_px,
        fused_in_ch=args.fused_ch,
        red_ch=args.acv_red_ch,
        base3d=args.agg_ch,
        use_motif=args.use_motif,
        two_stage=args.two_stage,
        local_radius_cells=args.local_radius
    ).to(device).train()

    # decoder = IGEVStereo(args).to(device).train()

    ckpt_model = torch.nn.ModuleDict({
        "stereo": stereo,
        "decoder": decoder,
    })

    # --- 손실 (모두 1/4 + Full-res 추가) ---
    dir_loss_fn = DirectionalRelScaleDispLoss(
        sim_thr=args.sim_thr, sim_gamma=args.sim_gamma,
        sample_k=args.sim_sample_k,
        use_dynamic_thr=args.use_dynamic_thr, dynamic_q=args.dynamic_q,
        vert_up_allow=args.vert_up_allow,         # 위쪽: Δ ∈ [-1,0]
        vert_down_allow=args.vert_down_allow,     # 아래: Δ ∈ [0,+1]
        horiz_margin=args.horiz_margin,           # 가로 대칭 허용
        lambda_v=args.lambda_v, lambda_h=args.lambda_h,
        huber_delta=args.huber_delta_h
    ).to(device)
    criterion = AdaptiveWindowDistillLoss(
            max_disp=args.max_disp_px / 4.0,
            roi_mode="frac", roi_u0=0.3, roi_u1=7.0, roi_v0=0.7, roi_v1=1.0,
            ent_T=0.1, ent_vis_thr=0.6, train_ent_thr=None,  # after-entropy gating off
            max_half=12, T_teacher=0.1, T_student=1.0,
            lambda_kl=1.0, lambda_reg=0.5,
            use_peak_weight=True, weight_alpha=1.0, weight_beta=1.0,
            weight_gamma=0.5, weight_delta=1.0,
            gap_min=1.0, gap_norm=4.0, L0=8.0
        ).to(device)
    
    # (선택) Feature reprojection loss — args.w_reproj 있을 때만 켜기
    w_reproj = float(getattr(args, "w_reproj", 0.0))
    reprog_loss_fn = None
    if w_reproj > 0.0:
        try:
            reprog_loss_fn = FeatureReprojLoss(charbonnier_eps=1e-3).to(device)
        except Exception as e:
            print(f"[Warn] FeatureReprojLoss unavailable ({e}) → w_reproj=0으로 강등")
            w_reproj = 0.0

    # photo_crit = PhotometricLoss(w_l1 = args.photo_l1_w, w_ssim=args.photo_ssim_w)
    photo_crit = PhotometricLoss([args.photo_l1_w, args.photo_ssim_w])

    # --- 옵티마/스케일러 ---
    params = list(stereo.parameters()) + list(decoder.parameters())
    params = [p for p in params if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    # scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # --- 체크포인트 재개 ---
    start_epoch = 1

    if args.resume and HAS_CKPT:
        start_epoch, _, ckpt = resume_from_checkpoint(args,ckpt_model, device)
        if (ckpt is not None) and ("optim" in ckpt) and (not args.resume_reset_optim):
            try:
                optim.load_state_dict(ckpt["optim"]); _move_optimizer_state_to_device(optim, device)
                print("[Resume] optimizer 상태 복구")
            except Exception as e:
                print(f"[Resume] optimizer 복구 실패 → 초기화: {e}")
        # if (ckpt is not None) and ("scaler" in ckpt) and (ckpt["scaler"] is not None) and (not args.resume_reset_scaler):
        #     try:
        #         scaler.load_state_dict(ckpt["scaler"]); print("[Resume] GradScaler 상태 복구")
        #     except Exception as e:
        #         print(f"[Resume] GradScaler 상태 복구 실패 → 무시: {e}")

    # --- RealtimeEvalLogger ---
    os.makedirs(args.save_dir, exist_ok=True)
    rtlog = RealtimeEvalLogger(args.save_dir)

    # --- 학습 루프 ---
    stereo.train(); decoder.train()
    decayed = False
    for epoch in range(start_epoch, args.epochs + 1):
        rtlog.start_epoch()
        running = 0.0
        if (not decayed) and (epoch >= args.decay_epoch):
            for g in optim.param_groups:
                g["lr"] *= 0.1
            decayed = True
            print(f"[Epoch {epoch}] LR decayed x0.1 → {optim.param_groups[0]['lr']:.3e}")
        for it, (imgL, imgR, names) in enumerate(loader, start=1):
            imgL = imgL.to(device, non_blocking=True)  # ImageNet 정규화 가정
            imgR = imgR.to(device, non_blocking=True)

            # Photometric용 [0,1] 복원 (언패드 원본 크기)
            with torch.no_grad():
                imgL_01 = denorm_imagenet(imgL)
                imgR_01 = denorm_imagenet(imgR)

            # === 1) 입력 16배수 패딩 ===
            imgL_pad, pad = pad_to_multiple(imgL,  mult=16, mode="replicate")
            imgR_pad, _   = pad_to_multiple(imgR,  mult=16, mode="replicate")

            # pad = (pad_b, pad_r) 이고, 16배수라서 pad_b%4==0, pad_r%4==0 이 보장됨
            assert pad[0] % 4 == 0 and pad[1] % 4 == 0, "pad must be divisible by 4"

            with torch.cuda.amp.autocast(enabled=args.amp):
                # 2) 백본/디코더 실행
                bb_out = stereo(imgL_pad, imgR_pad)
                # for sota 어쩌고
                pred   = decoder(bb_out)
                
                # 이건 igev 디코더용
                # pred   = decoder(imgL_pad, imgR_pad, bb_out["left"]["fused_1_4"], bb_out["right"]["fused_1_4"], iters=args.igev_iters)

                # 3) 출력 언패드 — 모두 1/4 해상도 좌표계(+Full-res)
                pad_q = (pad[0] // 4, pad[1] // 4)

                # 예측 disparity 두 버전 유지:
                # (a) 1/4‑px 단위(photometric/warp/깊이용)
                disp_q_qpx_padded = pred["disp_1_4"] / 4.0                 # [B,1,Hq_pad,Wq_pad], unit: 1/4‑px
                disp_q_qpx        = unpad_last2(disp_q_qpx_padded, pad_q)  # [B,1,Hq,Wq],      unit: 1/4‑px

                # (b) px 단위(EPE/D1용)
                disp_q_px = unpad_last2(pred["disp_1_4"], pad_q)           # [B,1,Hq,Wq],      unit: px

                # (c) feature들 언패드
                CS_1_4_padded = bb_out["left"]["cossim_feat_1_4"]  # [B,Hq_pad,Wq_pad,C] (BHWC)
                CS_1_4        = unpad_bhwc(CS_1_4_padded, pad_q)   # [B,Hq,Wq,C]
                feat_L = unpad_last2(bb_out["left"]["fused_1_4"],  pad_q)  # [B,C,Hq,Wq]
                feat_R = unpad_last2(bb_out["right"]["fused_1_4"], pad_q)  # [B,C,Hq,Wq]

                FqL = CS_1_4
                FqR_padded = bb_out["right"]["cossim_feat_1_4"]  # [B,Hq_pad,Wq_pad,C] (BHWC)
                FqR        = unpad_bhwc(FqR_padded, pad_q)

                # (d) full-res disparity(px) — 디코더 제공 or 업샘플 fallback
                if "disp_full" in pred:
                    disp_full_px_padded = pred["disp_full"]                # [B,1,H_pad,W_pad]
                    disp_full_px        = unpad_last2(disp_full_px_padded, pad)  # [B,1,H,W]
                else:
                    # fallback: 1/4 지도(px)를 ×4 업샘플
                    disp_full_px = F.interpolate(disp_q_px, scale_factor=4, mode="bilinear", align_corners=False)

                # --- Losses @ 1/4 ---
                roi_1_4 = torch.ones_like(disp_q_qpx)
                loss_dir    = dir_loss_fn(disp_q_qpx, CS_1_4, roi_1_4) * args.w_dir
                loss_reproj = (reprog_loss_fn(feat_L, feat_R, disp_q_qpx) * w_reproj) if (w_reproj > 0 and reprog_loss_fn is not None) else 0.0

                Hq, Wq = disp_q_qpx.shape[-2:]
                imgL_q01 = F.interpolate(imgL_01, size=(Hq, Wq), mode="area")
                imgR_q01 = F.interpolate(imgR_01, size=(Hq, Wq), mode="area")

                # 우→좌 warp (1/4 격자, 1/4‑px 단위)
                imgR_qwarp, valid_q = warp_right_to_left_image(imgR_q01, disp_q_qpx)

                # Photometric @1/4
                photo_map_q  = photo_crit.simple_photometric_loss(imgL_q01, imgR_qwarp, weights=[args.photo_l1_w, args.photo_ssim_w])
                loss_photo_q = (photo_map_q * valid_q).sum() / (valid_q.sum() + 1e-6)
                loss_photo_q = loss_photo_q * args.w_photo_qres

                # Smoothness @1/4
                loss_smooth_q = get_disparity_smooth_loss(disp_q_qpx, imgL_q01) * args.w_smooth_qres

                # --- 추가: Full-res Photometric & Smoothness ---
                # 우→좌 warp (원본 해상도, px 단위)
                imgR_full_warp, valid_full = warp_right_to_left_image(imgR_01, disp_full_px)

                # Photometric @Full-res
                photo_map_full  = photo_crit.simple_photometric_loss(imgL_01, imgR_full_warp, weights=[args.photo_l1_w, args.photo_ssim_w])
                loss_photo_full = (photo_map_full * valid_full).sum() / (valid_full.sum() + 1e-6)
                loss_photo_full = loss_photo_full * args.w_photo_fullres
                # logits = unpad_last2(pred["logits_1_4"], pad_q)
                # distillloss, info = criterion(FqL, FqR, logits, return_debug=True)
                
                # Smoothness @Full-res
                loss_smooth_full = get_disparity_smooth_loss(disp_full_px, imgL_01) * args.w_smooth_fullres

                # 총손실
                loss = (
                    loss_dir
                    + loss_reproj if isinstance(loss_reproj, torch.Tensor) else loss_dir + torch.tensor(loss_reproj, device=device, dtype=loss_dir.dtype)
                )
                loss = loss + loss_photo_q + loss_smooth_q + loss_photo_full + loss_smooth_full # distillloss*args.w_distill

            # 최적화
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            # scaler.scale(loss).backward()

            # torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            # scaler.step(optim); scaler.update()
            running += float(loss.item())

            # --- 실시간 정량평가 ---
            extra_eval = ""
            disp_metrics = None
            depth_metrics = None
            depth_w = None

            if args.realtime_test and (args.gt_depth_dir is not None):
                with torch.no_grad():
                    # GT depth를 1/4 언패드 크기(Hq,Wq)로 로드 (단위: m)
                    gt_depth_q = load_ms2_gt_depth_batch(
                        names=names,
                        gt_depth_dir=args.gt_depth_dir,
                        scale=args.gt_depth_scale,
                        target_hw=(Hq, Wq),
                        device=imgL.device
                    )
                    if gt_depth_q is not None:
                        valid_mask = (gt_depth_q > 0).float()
                        has_fb = (getattr(args, "focal_px", 0.0) > 0.0) and (getattr(args, "baseline_m", 0.0) > 0.0)
                        if has_fb:
                            # --- Disparity metrics (px 기준) ---
                            gt_disp_q_px   = (args.focal_px * args.baseline_m) / gt_depth_q.clamp_min(1e-6) / 4.0  # px
                            pred_disp_q_px = disp_q_px  / 4.0                                                     # px
                            disp_metrics   = compute_ms2_disparity_metrics(pred_disp_q_px, gt_disp_q_px, valid_mask)

                            # --- Depth metrics (@1/4 격자) ---
                            fx_q = args.focal_px / 4.0
                            pred_depth_q = disparity_to_depth(disp_q_qpx, fx_q, args.baseline_m)  # disp: 1/4‑px, fx: fx/4
                            depth_metrics = compute_depth_metrics(pred_depth_q, gt_depth_q, valid_mask)

                            # --- Bin-weighted depth (선택) ---
                            if (args.eval_num_bins > 0) and (args.eval_max_depth_m > 0.0):
                                depth_w = compute_bin_weighted_depth(
                                    pred_depth_q, gt_depth_q, valid_mask,
                                    max_depth_m=args.eval_max_depth_m,
                                    num_bins=args.eval_num_bins
                                )

                            def _get(d, k, fmt="{:.3f}"):
                                if d is None: return "nan"
                                v = d.get(k, float("nan"))
                                try: return fmt.format(float(v))
                                except: return "nan"

                            extra_eval = (
                                f" || [Disp(px)] EPE={_get(disp_metrics,'EPE')}  D1={_get(disp_metrics,'D1_all','{:.2f}')}%  "
                                f">1px={_get(disp_metrics,'> 1px','{:.2f}')}%  >2px={_get(disp_metrics,'> 2 px','{:.2f}')}%  "
                                f"[Depth@1/4] AbsRel={_get(depth_metrics,'AbsRel')}  RMSE={_get(depth_metrics,'RMSE')}  "
                                f"δ1={_get(depth_metrics,'δ<1.25')}"
                            )
                        else:
                            extra_eval = " || [Calib] fx/baseline 미설정 → stereo/depth metric 생략"

            # 콘솔 로그
            if it % args.log_every == 0:
                lrp = (loss_reproj.item() if isinstance(loss_reproj, torch.Tensor) else float(loss_reproj)) if (w_reproj > 0) else 0.0
                print(f"[Epoch {epoch:03d} | Iter {it:04d}/{len(loader)}] "
                      f"loss={running/args.log_every:.4f} "
                      f"(dir={float(loss_dir):.4f}, reproj={lrp:.4f}, "
                      f"photoQ={float(loss_photo_q):.4f}, smoothQ={float(loss_smooth_q):.4f}, "
                      f"photoF={float(loss_photo_full):.4f}, smoothF={float(loss_smooth_full):.4f})"
                      f"{extra_eval}")
                running = 0.0

            # RealtimeEvalLogger 누적
            if args.realtime_test and (disp_metrics is not None or depth_metrics is not None or depth_w is not None):
                rtlog.add_batch(disp_metrics, depth_metrics, depth_w)

        # --- 에포크 종료: 로그 파일/CSV/플롯 쓰기 ---
        rtlog.end_epoch(epoch)

        # --- 체크포인트 저장 ---
        if (epoch % args.save_every == 0) or (epoch == args.epochs):
            ckpt_path = os.path.join(args.save_dir, f"dir_all_qres_epoch{epoch:03d}.pth")
            if HAS_CKPT:
                save_checkpoint(
                    ckpt_path, epoch,
                    ckpt_model, optim, args
                )
            else:
                torch.save({
                    "epoch": epoch,
                    "stereo": stereo.state_dict(),
                    "decoder": decoder.state_dict(),
                    "optim": optim.state_dict(),
                    "args": vars(args)
                }, ckpt_path)
            print(f"[Save] {ckpt_path}")


# =========================
# argparse
# =========================

from datetime import datetime, timezone, timedelta
current_time = datetime.now(tz=timezone.utc).astimezone(timezone(timedelta(hours=9))).strftime("%y%m%d_%H%M%S")
def get_args():
    p = argparse.ArgumentParser("Stereo — All losses @1/4 + Full-res photometric/smooth (inputs padded to ×16, outputs unpadded)")

    # 데이터
    p.add_argument("--left_dir",  type=str, required=True)
    p.add_argument("--right_dir", type=str, required=True)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width",  type=int, default=1224)

    # 모델/디코더
    p.add_argument("--max_disp_px", type=int, default=60)
    p.add_argument("--fused_ch",    type=int, default=320)
    p.add_argument("--acv_red_ch",  type=int, default=128)
    p.add_argument("--agg_ch",      type=int, default=128)
    p.add_argument("--use_motif",   type=bool, default=True)
    p.add_argument("--two_stage",   type=bool, default=True)
    p.add_argument("--local_radius", type=int, default=8)
    
    
    p.add_argument("--hidden_dims", default=[128, 128, 128])
    p.add_argument("--n_downsample", type=int, default=2)
    p.add_argument("--n_gru_layers", type=int, default=3)
    p.add_argument("--corr_radius",type=int, default=4)
    p.add_argument("--corr_levels",type=int, default=4)
    p.add_argument("--mixed_precision",type=bool, default=True)
    p.add_argument("--precision_dtype",type=str, default="float16")
    p.add_argument("--igev_iters",type=int, default=16)
    
    # 학습
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--decay_epoch", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--amp",        type=bool, default=False)
    p.add_argument("--seed",       type=int, default=42)

    # 손실 가중치
    p.add_argument("--w_dir",              type=float, default=1.0)
    p.add_argument("--w_reproj",           type=float, default=0.0)
    p.add_argument("--w_distill",           type=float, default=0.5)
    # 1/4 해상도
    p.add_argument("--w_photo_qres",       type=float, default=1.0,   help="Photometric @1/4")
    p.add_argument("--w_smooth_qres",      type=float, default=0.01,  help="Smoothness  @1/4")
    # Full-res 추가
    p.add_argument("--w_photo_fullres",    type=float, default=1.0,   help="Photometric @Full-res")
    p.add_argument("--w_smooth_fullres",   type=float, default=0.1, help="Smoothness  @Full-res")
    # photometric 내부 가중(공통)
    p.add_argument("--photo_l1_w",         type=float, default=0.15)
    p.add_argument("--photo_ssim_w",       type=float, default=0.85)
    

    # DirectionalRelScaleDispLoss 하이퍼 (cossim + 세로 [-1,0]/[0,1])
    p.add_argument("--sim_thr",      type=float, default=0.8)
    p.add_argument("--sim_gamma",    type=float, default=0.0)
    p.add_argument("--sim_sample_k", type=int,   default=1024)  # (호환성 유지)
    p.add_argument("--use_dynamic_thr", action="store_true")
    p.add_argument("--dynamic_q",    type=float, default=0.7)
    p.add_argument("--vert_up_allow",   type=float, default=0.4)
    p.add_argument("--vert_down_allow", type=float, default=0.4)
    p.add_argument("--horiz_margin",    type=float, default=0.0)
    p.add_argument("--lambda_v",     type=float, default=1.0)
    p.add_argument("--lambda_h",     type=float, default=1.0)
    p.add_argument("--huber_delta_h", type=float, default=1.0)

    # 로깅/저장/재개
    p.add_argument("--log_every",   type=int, default=10)
    p.add_argument("--save_every",  type=int, default=1)
    p.add_argument("--save_dir",    type=str, default=f"./log/checkpoints_{current_time}")
    p.add_argument("--resume",      type=str, default=None)
    p.add_argument("--resume_reset_optim",  action="store_true")
    p.add_argument("--resume_reset_scaler", action="store_true")

    # --- 실시간 정량평가 옵션 (모두 1/4 해상도 기준) ---
    p.add_argument("--realtime_test", action="store_true")
    p.add_argument("--gt_depth_dir",  type=str, default=None)
    p.add_argument("--gt_depth_scale", type=float, default=256.0)
    p.add_argument("--eval_num_bins",  type=int, default=5)
    p.add_argument("--eval_max_depth_m", type=float, default=50.0)

    # 캘리브(자동/수동)
    p.add_argument("--calib_npy", type=str, default=None)
    p.add_argument("--K_left_npy", type=str, default=None)
    p.add_argument("--focal_px", type=float, default=764.5138549804688)
    p.add_argument("--baseline_m", type=float, default=0.29918420530585865)

    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    # save_args_as_text(args, args.save_dir)
    train(args)
