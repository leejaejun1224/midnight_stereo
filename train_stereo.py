# -*- coding: utf-8 -*-
import os
import warnings
import argparse
from typing import Set
from datetime import datetime, timezone, timedelta

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 외부 모듈 (사용자 환경 그대로)
from modeljj_reassemble import StereoModel
from stereo_dpt import DINOv1Base8Backbone, StereoDPTHead, DPTStereoTrainCompat
from prop_utils import *      # SeedAnchorHuberLoss, build_bad_seed_mask_1of8, rowwise_mode_idx_1of8, make_norm_rect_mask_like, ...
from logger import *
from sky_loss import *       # SkyGridZeroLoss

# 분리한 유틸/손실/워핑/리줌/프리즈
from tools import (
    set_seed, denorm_imagenet,
    StereoFolderDataset,
    DirectionalRelScaleDispLoss, HorizontalSharpenedConsistency,
    NeighborProbConsistencyLoss, EntropySharpnessLoss,
    CorrAnchorLoss, FeatureReprojLoss,
    shift_with_mask, warp_right_to_left_image,
    PhotometricLoss, get_disparity_smooth_loss,
    read_fx_baseline_rgb, load_ms2_gt_depth_batch,
    disparity_to_depth, compute_ms2_disparity_metrics, compute_depth_metrics, compute_bin_weighted_depth,
    _fmt_disp, _fmt_depth, _fmt_depth_w,
    build_optimizer, _move_optimizer_state_to_device, save_checkpoint,
    resume_from_checkpoint,
    freeze_params_by_name, unfreeze_by_name_contains, unfreeze_last_k_params, dump_param_lists,
    enhance_batch_bgr_from_rgb01
)

# ---------------------------
# 학습 루프
# ---------------------------
# train_stereo.py 상단 import 아래에 추가 (csv, math)
import csv
import math

# ... (기존 import 들 유지) ...

# ==== NEW: RealtimeEvalLogger =================================================
class RealtimeEvalLogger:
    """
    - 매 에포크 동안 add_batch(...)로 배치별 지표를 누적
    - end_epoch(epoch)에서 평균을 TXT/CSV로 저장하고 PNG 그래프를 갱신
    - 저장 위치: <save_dir>/realtime_eval.txt, realtime_eval.csv, plot_*.png
    """
    def __init__(self, save_dir: str,
                 filename_txt: str = "realtime_eval.txt",
                 filename_csv: str = "realtime_eval.csv"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.txt_path = os.path.join(self.save_dir, filename_txt)
        self.csv_path = os.path.join(self.save_dir, filename_csv)
        self.history = []           # [{'epoch': int, 'Disp_EPE': ..., ...}, ...]
        self._load_history_if_exists()
        self._epoch_reset()

    def _load_history_if_exists(self):
        if not os.path.isfile(self.csv_path):
            return
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
        # 배치 누적 버퍼
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

    def start_epoch(self):
        self._epoch_reset()

    def add_batch(self, disp_metrics: dict, depth_metrics: dict, depth_w: dict = None):
        # 안전하게 추가 (nan/inf 무시)
        def _add(key, val):
            if val is None: return
            try:
                v = float(val)
                if math.isfinite(v):
                    self._acc[key].append(v)
            except:
                pass

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

    def _append_txt(self, e: dict):
        # 한 줄 요약 (로그와 유사한 형식)
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

    def _append_csv(self, e: dict):
        header = [
            "epoch",
            "Disp_EPE","Disp_D1_all","Disp_gt1px","Disp_gt2px",
            "Depth_AbsRel","Depth_RMSE","Depth_delta1",
            "W_AbsRel","W_RMSE","W_delta1"
        ]
        is_new = not os.path.isfile(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if is_new:
                w.writeheader()
            row = {k: e.get(k, float("nan")) for k in header}
            w.writerow(row)

    def _save_plots(self):
        # 파일 저장 전용 백엔드
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
                    xs.append(e["epoch"])
                    ys.append(v)
            if len(xs) == 0:
                continue
            plt.figure()
            plt.plot(xs, ys, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(key)
            plt.grid(True, linestyle="--", alpha=0.4)
            out = os.path.join(self.save_dir, f"plot_{key}.png")
            plt.tight_layout()
            plt.savefig(out, dpi=150)
            plt.close()
# ==== /RealtimeEvalLogger =====================================================

def train(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- (NEW) fx/baseline 자동 로딩 ---
    fx, B, src = read_fx_baseline_rgb(
        intrinsic_left_npy=getattr(args, "K_left_npy", None),
        calib_npy=getattr(args, "calib_npy", None)
    )
    if getattr(args, "focal_px", 0.0) <= 0.0 and fx is not None:
        args.focal_px = float(fx)
    if getattr(args, "baseline_m", 0.0) <= 0.0 and B is not None:
        args.baseline_m = float(B)
    if getattr(args, "realtime_test", False):
        print(f"[Calib] fx(px)={getattr(args,'focal_px',0.0):.6f}  baseline(m)={getattr(args,'baseline_m',0.0):.6f}  src={src}")

    dataset = StereoFolderDataset(args.left_dir, args.right_dir,
                                  height=args.height, width=args.width)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    model = StereoModel(max_disp_px=args.max_disp_px, patch_size=args.patch_size,
                        agg_base_ch=args.agg_ch, agg_depth=args.agg_depth,
                        softarg_t=args.softarg_t, norm=args.norm).to(device)

    # 손실 모듈
    dir_loss_fn = DirectionalRelScaleDispLoss(
        sim_thr=args.sim_thr, sim_gamma=args.sim_gamma, sample_k=args.sim_sample_k,
        use_dynamic_thr=args.use_dynamic_thr, dynamic_q=args.dynamic_q,
        vert_margin=1.0, horiz_margin=0.0,
        lambda_v=args.lambda_v, lambda_h=args.lambda_h, huber_delta=1.0).to(device)

    hsharp_fn = HorizontalSharpenedConsistency(
        D=(args.max_disp_px//args.patch_size), tau_sharp=args.tau_sharp, huber_delta=args.huber_delta_h,
        use_fixed_denom=True,
        sim_thr=args.sim_thr, sim_gamma=args.sim_gamma, sample_k=args.sim_sample_k,
        use_dynamic_thr=args.use_dynamic_thr, dynamic_q=args.dynamic_q).to(device)

    prob_cons_fn = NeighborProbConsistencyLoss(
        sim_thr=max(0.5, args.sim_thr-0.15), sim_gamma=max(0.05, args.sim_gamma),
        sample_k=max(1024, args.sim_sample_k),
        allow_shift_v=1, allow_shift_h=0,
        use_dynamic_thr=True, dynamic_q=max(0.7, args.dynamic_q), conf_alpha=1.0).to(device)

    entropy_fn = EntropySharpnessLoss(
        conf_alpha=1.0, sim_thr=args.sim_thr, sim_gamma=args.sim_gamma,
        sample_k=args.sim_sample_k, use_dynamic_thr=True, dynamic_q=args.dynamic_q).to(device)

    anchor_loss_fn = CorrAnchorLoss(tau=args.anchor_tau, margin=args.anchor_margin,
                                    topk=args.anchor_topk, use_huber=True).to(device)
    reproj_loss_fn = FeatureReprojLoss().to(device)
    
    seed_anchor_fn = SeedAnchorHuberLoss(
        tau=args.seed_tau, huber_delta=args.seed_huber_delta
    ).to(device)

    photo_crit = PhotometricLoss(weights=[args.photo_l1_w, args.photo_ssim_w])

    sky_loss = SkyGridZeroLoss(
        max_disp_px=args.max_disp_px,
        patch_size=args.patch_size).to(device)

    # ---[ Resume + Pretrained Freeze 처리 ]---
    start_epoch = 1
    loaded_set: Set[str] = set()
    ckpt = None
    if args.resume is not None:
        start_epoch, loaded_set, ckpt = resume_from_checkpoint(args, model, device)
        if args.pretrained_freeze:
            # 1) 체크포인트로 로드된 파라미터 동결
            freeze_params_by_name(model, loaded_set, verbose=True)

            # 2) 예외 문자열 적용
            except_list = [s.strip() for s in (getattr(args, "freeze_except", "") or "").split(",") if s.strip()]
            if except_list:
                unfreeze_by_name_contains(model, except_list, verbose=True)

            # 3) 마지막 K개 파라미터 해제
            if getattr(args, "freeze_last_k", 0) > 0:
                unfreeze_last_k_params(model, args.freeze_last_k, verbose=True)

            # 4) 자동 폴백: 여전히 학습 파라미터 0개면 헤드/디코더/업마스크/agg 등 일반적인 말단을 우선 해제,
            #    그래도 0개면 마지막 32개 텐서를 해제
            if not any(p.requires_grad for p in model.parameters()):
                print("[Freeze] 학습 가능한 파라미터가 0개입니다. 자동 폴백을 시도합니다.")
                unfreeze_by_name_contains(model, ["upmask", "head", "refine", "decoder", "agg", "out", "classifier"], verbose=True)
                if not any(p.requires_grad for p in model.parameters()):
                    unfreeze_last_k_params(model, 32, verbose=True)
            # 최종 확인
            if not any(p.requires_grad for p in model.parameters()):
                raise RuntimeError(
                    "동결 이후 학습 가능한 파라미터가 없습니다. "
                    "다음 중 하나를 사용하세요: "
                    "--freeze_except upmask_head,agg  또는  --freeze_last_k 32  "
                    "혹은 --pretrained_freeze 옵션을 제거하세요."
                )

    # === 옵티마이저/스케일러는 동결/미동결 결정 후 생성 ===
    optim = build_optimizer([p for p in model.parameters() if p.requires_grad],
                            name=args.optim, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # === 옵티마이저/스케일러 상태 복구 (동결 모드가 아닐 때만) ===
    if (ckpt is not None) and (not args.pretrained_freeze):
        if (not args.resume_reset_optim) and ("optim" in ckpt):
            try:
                optim.load_state_dict(ckpt["optim"])
                _move_optimizer_state_to_device(optim, device)
                print("[Resume] optimizer 상태 복구")
            except Exception as e:
                warnings.warn(f"[Resume] optimizer 복구 실패 → 초기화: {e}")
        else:
            print("[Resume] optimizer 상태 초기화(미복구)")
        if scaler is not None and (not args.resume_reset_scaler) and ("scaler" in ckpt) and (ckpt["scaler"] is not None):
            try:
                scaler.load_state_dict(ckpt["scaler"])
                print("[Resume] GradScaler 상태 복구")
            except Exception as e:
                warnings.warn(f"[Resume] GradScaler 상태 복구 실패 → 무시: {e}")
        else:
            print("[Resume] GradScaler 상태 초기화(미복구)")
    elif args.pretrained_freeze and ckpt is not None:
        print("[Resume] pretrained_freeze=True → optimizer/scaler 상태 복구 생략 (파라미터 구성 변경)")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        log_path = save_args_as_text(
                args,
                save_dir=args.save_dir,
                filename="trainlog.txt",
                append_if_exists=True,
                title="Stereo Training Arguments"
            )
        print(f"[Log] Arguments written to: {log_path}")
        dump_param_lists(args.save_dir, model)
        
    model.train()
    
    for epoch in range(start_epoch, args.epochs + 1):
        running = 0.0
        for it, (imgL, imgR, names) in enumerate(loader, start=1):
            imgL = imgL.to(device, non_blocking=True)
            imgR = imgR.to(device, non_blocking=True)

            # photometric용 원본 [0,1] 변환 + 1/2 스케일
            with torch.no_grad():
                imgL_01 = denorm_imagenet(imgL)
                imgR_01 = denorm_imagenet(imgR)
                imgL_half_01 = F.interpolate(imgL_01, scale_factor=0.5, mode='bilinear', align_corners=False)
                imgR_half_01 = F.interpolate(imgR_01, scale_factor=0.5, mode='bilinear', align_corners=False)

                # 원본 해상도에서도 photometric/smoothness를 사용할 경우만 보정 계산
                need_full_enh = (args.w_photo_fullres > 0.0) or (args.w_smooth_fullres > 0.0)
                if need_full_enh:
                    imgL_full_enh_01 = enhance_batch_bgr_from_rgb01(
                        imgL_01, enable=(not args.no_enhance),
                        gamma=args.enhance_gamma,
                        clahe_clip=args.enhance_clahe_clip,
                        clahe_tile=args.enhance_clahe_tile
                    )
                else:
                    imgL_full_enh_01 = None

            # enhance (좌 1/2 영상만 기준으로 사용)
            imgL_half_enh_01 = enhance_batch_bgr_from_rgb01(
                imgL_half_01, enable=(not args.no_enhance),
                gamma=args.enhance_gamma,
                clahe_clip=args.enhance_clahe_clip,
                clahe_tile=args.enhance_clahe_tile
            )

            with torch.cuda.amp.autocast(enabled=args.amp):
                prob, disp_soft, aux = model(imgL, imgR)
                FL, FR   = aux["FL"], aux["FR"]
                raw_vol  = aux["raw_volume"]
                mask_d   = aux["mask"]
                refined_masked = aux["refined_masked"]

                # 1/2 해상도 disparity (픽셀 단위, 1/2 격자 기준 px)
                disp_half_px = aux["disp_half_px"]

                # === ROI 전영역(=1) 처리 ===
                roi_patch = torch.ones_like(disp_soft)           # [B,1,H/8,W/8]
                roi_half  = torch.ones_like(disp_half_px)        # [B,1,H/2,W/2]

                # 오른쪽 half 이미지를 좌로 warp
                imgR_half_warp_01, valid_half = warp_right_to_left_image(imgR_half_01, disp_half_px)

                # Losses
                loss_dir    = dir_loss_fn(disp_soft, FL, roi_patch) * args.w_dir
                loss_hsharp = hsharp_fn(refined_masked, FL, roi_patch) * args.w_hsharp
                loss_prob   = prob_cons_fn(prob, FL, roi_patch) * args.w_probcons
                loss_ent    = entropy_fn(prob, FL, roi_patch) * args.w_entropy
                loss_anchor = anchor_loss_fn(raw_vol, disp_soft, mask=mask_d, roi=roi_patch) * args.w_anchor
                loss_reproj = reproj_loss_fn(FL, FR, disp_soft, roi=roi_patch) * args.w_reproj

                # Photometric (L1+SSIM) on half res
                photo_map = photo_crit.simple_photometric_loss(
                    imgL_half_enh_01, imgR_half_warp_01,
                    weights=[args.photo_l1_w, args.photo_ssim_w]
                )  # [B,1,H/2,W/2]
                photo_mask = roi_half * valid_half
                loss_photo = (photo_map * photo_mask).sum() / (photo_mask.sum() + 1e-6)
                loss_photo = loss_photo * args.w_photo

                # Edge-aware smoothness on half res
                loss_smooth = get_disparity_smooth_loss(disp_half_px, imgL_half_enh_01) * args.w_smooth

                # ====== [추가] 원본 해상도 photometric/smoothness ======
                loss_photo_full  = torch.tensor(0.0, device=device)
                loss_smooth_full = torch.tensor(0.0, device=device)

                # (평가용) full-res disparity 확보 (항상 생성)
                if "disp_full_px" in aux and aux["disp_full_px"] is not None:
                    disp_full_px_eval = aux["disp_full_px"]                         # [B,1,H,W]
                else:
                    disp_full_grid = F.interpolate(disp_half_px, scale_factor=2.0, mode="bilinear", align_corners=False)
                    disp_full_px_eval = disp_full_grid * 2.0                        # half-px → full-px

                if (args.w_photo_fullres > 0.0) or (args.w_smooth_fullres > 0.0):
                    # photometric (full)
                    if args.w_photo_fullres > 0.0:
                        imgR_full_warp_01, valid_full = warp_right_to_left_image(imgR_01, disp_full_px_eval)
                        photo_full_map = photo_crit.simple_photometric_loss(
                            imgL_full_enh_01 if imgL_full_enh_01 is not None else imgL_01,
                            imgR_full_warp_01,
                            weights=[args.photo_l1_w, args.photo_ssim_w]
                        )  # [B,1,H,W]
                        loss_photo_full = (photo_full_map * valid_full).sum() / (valid_full.sum() + 1e-6)
                        loss_photo_full = loss_photo_full * args.w_photo_fullres

                    # smoothness (full)
                    if args.w_smooth_fullres > 0.0:
                        base_img_full = imgL_full_enh_01 if imgL_full_enh_01 is not None else imgL_01
                        loss_smooth_full = get_disparity_smooth_loss(disp_full_px_eval, base_img_full) * args.w_smooth_fullres
                # ===========================================================

                # === 1/8 seed prior 유틸 ===
                with torch.no_grad():
                    bad_seed_mask = build_bad_seed_mask_1of8(
                        disp_soft=disp_soft, prob_5d=prob, roi_patch=roi_patch,
                        low_idx_thr=args.seed_low_idx_thr, high_idx_thr=args.seed_high_idx_thr,
                        conf_thr=args.seed_conf_thr, road_ymin=args.seed_road_ymin,
                        use_extremes=True, use_conf=True
                    )

                    seed_rect_mask = make_norm_rect_mask_like(
                        roi_patch, y_min=args.seed_ymin, y_max=args.seed_ymax,
                        x_min=args.seed_xmin, x_max=args.seed_xmax
                    ).bool()

                    good_mask = (roi_patch > 0) & (~bad_seed_mask)
                    good_mask = good_mask & seed_rect_mask

                    D = prob.shape[2] - 1
                    row_mode_idx, row_valid = rowwise_mode_idx_1of8(
                        disp_soft=disp_soft, good_mask=good_mask, D=D,
                        bin_size=args.seed_bin_w, min_count=args.seed_min_count
                    )
                    seed_idx_map = row_mode_idx.expand(-1, -1, -1, disp_soft.shape[-1])
                    valid_rows   = row_valid.expand_as(bad_seed_mask)

                    anchor_mask = bad_seed_mask & valid_rows & seed_rect_mask

                loss_seed = seed_anchor_fn(
                    refined_logits_masked=refined_masked,   # [B,1,D+1,H/8,W/8]
                    seed_idx_map=seed_idx_map,              # [B,1,H/8,W/8]
                    anchor_mask=anchor_mask                 # [B,1,H/8,W/8]
                ) * args.w_seed

                # 총손실
                loss = loss_dir + loss_photo + loss_smooth + loss_seed \
                       + loss_photo_full + loss_smooth_full

                # === Sky loss ===
                loss_sky, _ = sky_loss(
                    refined_logits_masked=aux["refined_masked"],
                    disp_half_px=aux["disp_half_px"],
                    roi_half=None,
                    roi_patch=None,
                    names=names,
                    step=(epoch-1)*len(loader)+it
                )
                loss = loss + (args.w_sky * loss_sky)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            if hasattr(model, "agg"):
                torch.nn.utils.clip_grad_norm_(model.agg.parameters(), max_norm=5.0)
            if hasattr(model, "upmask_head"):
                torch.nn.utils.clip_grad_norm_(model.upmask_head.parameters(), max_norm=5.0)
            scaler.step(optim)
            scaler.update()

            running += loss.item()
            if it % args.log_every == 0:
                with torch.no_grad():
                    disp_wta = aux["disp_wta"]
                    soft_dx = (roi_patch * (disp_soft - shift_with_mask(disp_soft,0,1)[0]).abs()).sum() / (roi_patch.sum()+1e-6)
                    wta_dx  = (roi_patch * (disp_wta  - shift_with_mask(disp_wta, 0,1)[0]).abs()).sum() / (roi_patch.sum()+1e-6)

                    # --- (NEW) Realtime MS2 metrics ---
                    extra_eval = ""
                    if getattr(args, "realtime_test", False) and getattr(args, "gt_depth_dir", None):
                        H, W = imgL.shape[-2], imgL.shape[-1]
                        gt_depth = load_ms2_gt_depth_batch(
                            names=names,
                            gt_depth_dir=args.gt_depth_dir,
                            scale=args.gt_depth_scale,
                            target_hw=(H,W),
                            device=device
                        )
                        if gt_depth is not None:
                            valid = (gt_depth > 0).float()

                            # 예측 disp (full-res)
                            pred_disp_px = disp_full_px_eval

                            # Stereo 지표 (GT depth → disp 변환, fx/B 필요)
                            has_fb = (getattr(args, "focal_px", 0.0) > 0.0) and (getattr(args, "baseline_m", 0.0) > 0.0)
                            if has_fb:
                                gt_disp_px = (args.focal_px * args.baseline_m) / gt_depth.clamp_min(1e-6)
                                disp_metrics = compute_ms2_disparity_metrics(pred_disp_px, gt_disp_px, valid)
                                disp_msg = "[Disp] " + _fmt_disp(disp_metrics)

                                # Depth 지표 (pred depth = fx*B/disp)
                                pred_depth_m = disparity_to_depth(pred_disp_px, args.focal_px, args.baseline_m)
                                depth_metrics = compute_depth_metrics(pred_depth_m, gt_depth, valid)
                                depth_msg = "[Depth] " + _fmt_depth(depth_metrics)

                                # Bin-weighted (선택)
                                depth_w_msg = ""
                                if getattr(args, "eval_num_bins", 0) > 0 and getattr(args, "eval_max_depth_m", 0.0) > 0.0:
                                    depth_w = compute_bin_weighted_depth(
                                        pred_depth_m, gt_depth, valid,
                                        max_depth_m=args.eval_max_depth_m,
                                        num_bins=args.eval_num_bins
                                    )
                                    depth_w_msg = "[Depth-W] " + _fmt_depth_w(depth_w)

                                parts = [s for s in [disp_msg, depth_msg, depth_w_msg] if s]
                                if parts:
                                    extra_eval = " || " + "  ".join(parts)
                            else:
                                extra_eval = " || [Calib] fx/baseline 미설정 → stereo/depth metric 생략"
                    # ----------------------------------

                print(f"[Epoch {epoch:03d} | Iter {it:04d}/{len(loader)}] "
                      f"loss={running/args.log_every:.4f} !!"
                      f"(dir={loss_dir.item():.4f}, hsharp={loss_hsharp.item():.4f}, "
                      f"prob={loss_prob.item():.4f}, ent={loss_ent.item():.4f}, "
                      f"anc={(loss_anchor/max(args.w_anchor,1e-9)).item():.4f}, rep={(loss_reproj/max(args.w_reproj,1e-9)).item():.4f}, "
                      f"photo={(loss_photo/max(args.w_photo,1e-9)).item():.4f}, smooth={(loss_smooth/max(args.w_smooth,1e-9)).item():.4f}, "
                      f"photoF={(loss_photo_full/max(args.w_photo_fullres,1e-9)).item():.4f}, smoothF={(loss_smooth_full/max(args.w_smooth_fullres,1e-9)).item():.4f}, "
                      f"seed={(loss_seed/max(args.w_seed,1e-9)).item():.4f}, sky={(loss_sky/max(args.w_sky,1e-9)).item():.4f}) "
                      f"| mean|Δx| soft={soft_dx:.3f} wta={wta_dx:.3f}"
                      f"{extra_eval}"
                )
                running = 0.0

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            if epoch % args.save_every == 0 or epoch == args.epochs:
                ckpt_path = os.path.join(args.save_dir, f"stereo_epoch{epoch:03d}.pth")
                save_checkpoint(ckpt_path, epoch, model, optim, scaler, args)
                print(f"[Save] {ckpt_path}")


# ---------------------------
# 메인
# ---------------------------
current_time = datetime.now(tz=timezone.utc).astimezone(timezone(timedelta(hours=9))).strftime("%y%m%d_%H%M%S")

def parse_args():
    p = argparse.ArgumentParser()
    # 데이터
    p.add_argument("--left_dir", type=str, required=True)
    p.add_argument("--right_dir", type=str, required=True)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width",  type=int, default=1224)

    # 모델/학습
    p.add_argument("--max_disp_px", type=int, default=88)
    p.add_argument("--patch_size",  type=int, default=8)
    p.add_argument("--agg_ch",      type=int, default=64)
    p.add_argument("--agg_depth",   type=int, default=3)
    p.add_argument("--softarg_t",   type=float, default=0.9)
    p.add_argument("--norm",        type=str, default="gn", choices=["bn","gn"])

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--optim",      type=str, default="adamw", choices=["adamw","adam","sgd"])
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--amp",        action="store_true", help="mixed precision")

    # 방향 이웃 제약(soft 기준)
    p.add_argument("--w_dir",        type=float, default=1.0)
    p.add_argument("--sim_thr",      type=float, default=0.8)
    p.add_argument("--sim_gamma",    type=float, default=0.0)
    p.add_argument("--sim_sample_k", type=int,   default=1024)
    p.add_argument("--use_dynamic_thr", action="store_true")
    p.add_argument("--dynamic_q",    type=float, default=0.7)
    p.add_argument("--lambda_v",     type=float, default=1.0)
    p.add_argument("--lambda_h",     type=float, default=1.0)
    p.add_argument("--huber_delta_h", type=float, default=0.25)

    # 샤픈 가로 일관성
    p.add_argument("--w_hsharp",   type=float, default=0.0)
    p.add_argument("--tau_sharp",  type=float, default=0.2)

    # 분포-일치/엔트로피
    p.add_argument("--w_probcons", type=float, default=0.0)
    p.add_argument("--w_entropy",  type=float, default=0.00)

    # 앵커/재투영
    p.add_argument("--w_anchor",     type=float, default=0.0)
    p.add_argument("--anchor_tau",   type=float, default=0.5)
    p.add_argument("--anchor_margin",type=float, default=1.0)
    p.add_argument("--anchor_topk",  type=int,   default=2)
    p.add_argument("--w_reproj",     type=float, default=1.0)

    # Photometric / Smoothness (half)
    p.add_argument("--w_photo",    type=float, default=1.0)
    p.add_argument("--w_smooth",   type=float, default=0.01)
    p.add_argument("--photo_l1_w",   type=float, default=0.15)
    p.add_argument("--photo_ssim_w", type=float, default=0.85)

    # Photometric / Smoothness (full-res 추가)
    p.add_argument("--w_photo_fullres",  type=float, default=0.0,
                   help="원본 해상도 photometric loss 가중치 (0이면 비활성)")
    p.add_argument("--w_smooth_fullres", type=float, default=0.0,
                   help="원본 해상도 edge-aware smoothness 가중치 (0이면 비활성)")
    p.add_argument("--fullres_disp_mode", type=str, default="bilinear",
                   choices=["nearest","bilinear"],
                   help="half→full disparity 업샘플 방식 (기본 bilinear)")

    # Enhance 옵션
    p.add_argument("--no_enhance", dest="no_enhance", action="store_true", help="저조도 보정 비활성화")
    p.set_defaults(no_enhance=False)
    p.add_argument("--enhance_gamma",      type=float, default=1.8)
    p.add_argument("--enhance_clahe_clip", type=float, default=2.0)
    p.add_argument("--enhance_clahe_tile", type=int,   default=8)

    # ---[ Resume / Checkpoint ]---
    p.add_argument("--resume", type=str, default=None,
                   help="불러올 체크포인트(.pth) 경로. 지정하면 해당 지점부터 이어서 학습")
    p.add_argument("--resume_non_strict", action="store_true",
                   help="state_dict 로드 시 strict=False (일부 키 불일치 허용)")
    p.add_argument("--resume_reset_optim", action="store_true",
                   help="체크포인트의 optimizer 상태를 무시하고 현재 설정으로 재시작")
    p.add_argument("--resume_reset_scaler", action="store_true",
                   help="체크포인트의 GradScaler 상태를 무시")

    # ★ 프리트레인 동결 옵션 + 예외/폴백
    p.add_argument("--pretrained_freeze", action="store_true",
                   help="체크포인트에서 불러온(이미 학습된) 파라미터는 동결하고, 새 모듈만 학습")
    p.add_argument("--freeze_except", type=str, default="",
                   help="콤마로 구분한 파라미터명 부분문자열 리스트. 매칭되면 동결에서 제외(학습). 예: 'upmask_head,agg,refine'")
    p.add_argument("--freeze_last_k", type=int, default=0,
                   help="동결 이후 마지막 K개 파라미터 텐서는 학습으로 유지")

    # 로깅/저장
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=2)
    p.add_argument("--save_dir", type=str, default=f"./log/checkpoints_{current_time}")
    
    # --- 1/8 Seeded Prior (핀 + 고무줄) ---
    p.add_argument("--w_seed", type=float, default=0.0, help="시드 앵커 손실 가중치(작게)")
    p.add_argument("--seed_low_idx_thr",  type=float, default=1.0)
    p.add_argument("--seed_high_idx_thr", type=float, default=1.0)
    p.add_argument("--seed_conf_thr",     type=float, default=0.05)
    p.add_argument("--seed_road_ymin",    type=float, default=0.8)
    p.add_argument("--seed_bin_w",        type=float, default=1.0)
    p.add_argument("--seed_min_count",    type=int,   default=8)
    p.add_argument("--seed_tau",          type=float, default=0.30)
    p.add_argument("--seed_huber_delta",  type=float, default=0.50)
    p.add_argument("--seed_ymin", type=float, default=0.7)
    p.add_argument("--seed_ymax", type=float, default=1.0)
    p.add_argument("--seed_xmin", type=float, default=0.2)
    p.add_argument("--seed_xmax", type=float, default=0.8)

    # Sky loss weight
    p.add_argument("--w_sky", type=float, default=0.0,
                   help="sky weight for SkyZeroLoss")

    # ---------- (NEW) Realtime evaluation ----------
    p.add_argument("--realtime_test", action="store_true",
                   help="배치별 실시간 정량평가(EPE/D1/>kpx + Depth/Weighted) 출력")
    p.add_argument("--gt_disp_dir", type=str, default=None,
                   help="(선택) GT disparity 디렉토리")
    p.add_argument("--gt_depth_dir", type=str, default=None,
                   help="GT depth 디렉토리 (PNG: depth[m]*scale, 또는 NPY/EXR)")
    p.add_argument("--gt_disp_scale", type=float, default=1.0,
                   help="GT disparity 스케일 나눗값")
    p.add_argument("--gt_depth_scale", type=float, default=256.0,
                   help="GT depth PNG가 depth[m]*scale 로 저장된 경우 scale (MS2=256)")
    p.add_argument("--eval_num_bins", type=int, default=5,
                   help="bin-weighted depth metric bin 수 (0이면 비활성)")
    p.add_argument("--eval_max_depth_m", type=float, default=50.0,
                   help="bin-weighted depth metric 최대 거리(m)")

    # 캘리브(자동로딩/직접입력)
    p.add_argument("--calib_npy", type=str, default="/home/jaejun/dataset/MS2/sync_data/_2021-08-13-22-36-41/calib.npy")
    p.add_argument("--K_left_npy", type=str, default="/home/jaejun/dataset/MS2/intrinsic_left.npy")
    p.add_argument("--T_lr_npy", type=str, default=None, help="4x4 extrinsic .npy (left->right)")
    p.add_argument("--focal_px", type=float, default=764.5138549804688)
    p.add_argument("--baseline_m", type=float, default=0.29918420530585865)

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
