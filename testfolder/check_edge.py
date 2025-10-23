# edge_smooth_viz.py
# ------------------------------------------------------------
# Edge-aware smoothness 시각화 + DINO ViT-B/8 기반 "평탄 패치" 마스킹
# - edge_weight_from_image: 에지 강도와 smooth 가중치 계산
# - save_edge_smooth_viz_batch: heatmap/overlay 저장
# - DINO ViT-B/8 특징으로 flat patch(8x8) 판단 → 해당 영역 edge=0 반영한 시각화 추가
# - CLI: 이미지 파일/디렉토리 입력 → PNG 저장
# ------------------------------------------------------------

import os
import sys
import glob
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


# ---------------------------
# Core: edge/weight 계산
# ---------------------------

@torch.no_grad()
def edge_weight_from_image(img_rgb01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    img_rgb01: [B,3,H,W], float32, 값 범위 [0,1]
    Returns:
      edge:   [B,1,H,W] = 0.5*(|∂x I| + |∂y I|), 채널 평균 절대 그래디언트
      weight: [B,1,H,W] = exp(-edge)  (edge-aware smoothness에서 쓰는 가중치)
    """
    assert img_rgb01.dim() == 4 and img_rgb01.size(1) == 3, "입력은 [B,3,H,W] 이어야 합니다."
    x = img_rgb01.clamp(0, 1)

    # 채널 평균 절대 그래디언트 (좌우/상하 1차 차분)
    gx = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]), dim=1, keepdim=True)
    gy = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]), dim=1, keepdim=True)

    # 가장자리 크기 보존을 위해 복제 패딩해서 원래 크기로 복원
    gx = F.pad(gx, (0, 1, 0, 0), mode='replicate')  # pad right
    gy = F.pad(gy, (0, 0, 0, 1), mode='replicate')  # pad bottom

    edge = 0.5 * (gx + gy)
    weight = torch.exp(-edge)
    return edge, weight


# ---------------------------
# DINO ViT-B/8 특징 추출 & flat patch 마스크
# ---------------------------

class DinoVITB8Features(torch.nn.Module):
    """
    facebookresearch/dino:main 의 dino_vitb8 (ViT-B/8) 로드
    입력: RGB [0,1] → ImageNet 정규화 → 토큰 → [B,C,H/8,W/8] ℓ2 정규화 특징
    """
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        try:
            self.backbone = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
        except Exception as e:
            raise RuntimeError(
                f"DINO ViT-B/8 로드 실패: {e}\n"
                "인터넷 연결 또는 torch.hub 캐시를 확인하세요."
            )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval().to(self.device)

        # ImageNet 통계
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    @torch.no_grad()
    def forward(self, img_rgb01: torch.Tensor) -> torch.Tensor:
        """
        img_rgb01 : [B,3,H,W] in [0,1]
        return    : [B,C,H8,W8], ℓ2 정규화
        """
        x = img_rgb01.to(self.device)
        B, _, H, W = x.shape

        # H,W를 8의 배수로 크롭 (오른쪽/아래)
        Hm = (H // 8) * 8
        Wm = (W // 8) * 8
        if Hm != H or Wm != W:
            x = x[:, :, :Hm, :Wm]
        # 정규화
        x = (x - self.mean) / self.std

        # DINO 토큰 → 패치 특징
        tokens = self.backbone.get_intermediate_layers(x, n=1)[0]  # [B,1+P,C]
        patch = tokens[:, 1:, :]                                   # [B,P,C]
        C = patch.shape[-1]
        H8, W8 = Hm // 8, Wm // 8
        feat = patch.transpose(1, 2).contiguous().view(B, C, H8, W8)  # [B,C,H8,W8]
        feat = F.normalize(feat, dim=1, eps=1e-6)
        return feat, (Hm, Wm)


@torch.no_grad()
def compute_flat_patch_mask(FL: torch.Tensor, thr: float = 0.9, radius: int = 1) -> torch.Tensor:
    """
    FL:   [B,C,H8,W8]  (ℓ2 정규화 특징)
    thr:  코사인 유사도 임계 (기본 0.9)
    r:    각 방향 최대 r칸 이내에 thr 이상 이웃이 '하나라도' 있으면 해당 방향 OK
    return: [B,1,H8,W8] (0/1), 네 방향 모두 OK일 때 1
    """
    FLn = F.normalize(FL, dim=1, eps=1e-6)

    def dir_ok(dy: int, dx: int) -> torch.Tensor:
        acc = None
        for r in range(1, radius + 1):
            f_nb, valid = shift_with_mask_feat(FLn, dy*r, dx*r)  # [B,C,H8,W8], [B,1,H8,W8]
            sim = (FLn * f_nb).sum(dim=1, keepdim=True)          # [B,1,H8,W8]
            ok = (sim >= thr).to(sim.dtype) * valid
            acc = ok if acc is None else torch.maximum(acc, ok)  # 동일 방향 내 any
        return acc if acc is not None else torch.zeros_like(FLn[:, :1])

    up    = dir_ok(-1,  0)
    down  = dir_ok( 1,  0)
    left  = dir_ok( 0, -1)
    right = dir_ok( 0,  1)
    flat = (up * down * left * right)                            # 네 방향 모두 만족
    return flat


def shift_with_mask_feat(x: torch.Tensor, dy: int, dx: int):
    """특징 텐서용 시프트 + 유효 마스크"""
    B, C, H, W = x.shape
    pt, pb = max(dy,0), max(-dy,0)
    pl, pr = max(dx,0), max(-dx,0)
    x_pad = F.pad(x, (pl, pr, pt, pb))
    x_shift = x_pad[:, :, pb:pb+H, pl:pl+W]
    valid = torch.ones((B,1,H,W), device=x.device, dtype=x.dtype)
    if dy > 0:   valid[:, :, :dy, :] = 0
    if dy < 0:   valid[:, :, H+dy:, :] = 0
    if dx > 0:   valid[:, :, :, :dx] = 0
    if dx < 0:   valid[:, :, :, W+dx:] = 0
    return x_shift, valid


@torch.no_grad()
def build_flat_mask_fullres(
    img_rgb01: torch.Tensor,
    dino: DinoVITB8Features,
    thr: float = 0.9,
    radius: int = 1
) -> torch.Tensor:
    """
    입력 해상도와 동일한 [B,1,H,W] flat 마스크 생성 (8x8 패치 판정 → nearest 업샘플 + 패딩)
    """
    B, _, H, W = img_rgb01.shape
    FL, (Hm, Wm) = dino(img_rgb01)                        # [B,C,H8,W8], (Hm,Wm)=8의 배수
    flat_patch = compute_flat_patch_mask(FL, thr=thr, radius=radius)  # [B,1,H8,W8]
    flat_full  = F.interpolate(flat_patch, size=(Hm, Wm), mode="nearest")  # [B,1,Hm,Wm]

    if Hm != H or Wm != W:
        # 아래/오른쪽 0-padding으로 원본 해상도 맞춤
        pad = (0, W - Wm, 0, H - Hm)
        flat_full = F.pad(flat_full, pad, mode="constant", value=0.0)
    return flat_full  # [B,1,H,W]


# ---------------------------
# 시각화 유틸
# ---------------------------

def _apply_colormap01(x01: np.ndarray, cmap: int = None) -> np.ndarray:
    """
    x01: [H,W] float in [0,1] → BGR uint8 heatmap (OpenCV 컬러맵)
    """
    if not HAS_CV2:
        raise RuntimeError("OpenCV(cv2)가 필요합니다. pip install opencv-python 로 설치하세요.")
    x = (np.clip(x01, 0.0, 1.0) * 255.0).astype(np.uint8)
    if cmap is None:
        cmap = cv2.COLORMAP_TURBO
    try:
        cm = cv2.applyColorMap(x, cmap)
    except Exception:
        cm = cv2.applyColorMap(x, cv2.COLORMAP_JET)
    return cm


def save_edge_smooth_viz_batch(img_rgb01: torch.Tensor,
                               names: List[str],
                               out_dir: str,
                               edge_map: torch.Tensor = None,
                               weight_map: torch.Tensor = None,
                               alpha: float = 0.4,
                               edge_quantile: float = 0.8,
                               prefix: str = "viz") -> None:
    """
    img_rgb01: [B,3,H,W] in [0,1]
    names:    길이 B의 파일명(또는 샘플명) 리스트
    out_dir:  결과 저장 폴더

    저장 파일 (샘플별):
      {name}_{prefix}_edge_heat.png             : 에지 강도 heatmap (TURBO)
      {name}_{prefix}_edge_overlay.png          : 에지 heatmap 오버레이 (원본 위 반투명)
      {name}_{prefix}_smooth_weight.png         : exp(-edge) heatmap (VIRIDIS, 밝을수록 smooth)
      {name}_{prefix}_edge_mask_overlay.png     : 상위 quantile 에지 바이너리 마스크 오버레이(빨강)
    """
    if not HAS_CV2:
        raise RuntimeError("OpenCV(cv2)가 필요합니다. pip install opencv-python 로 설치하세요.")
    os.makedirs(out_dir, exist_ok=True)

    B = img_rgb01.shape[0]
    # 이름 정리
    if not isinstance(names, (list, tuple)) or len(names) != B:
        names = [f"sample_{i:03d}" for i in range(B)]
    else:
        names = [os.path.splitext(str(n))[0] for n in names]

    # 필요 시 edge/weight 계산
    if edge_map is None or weight_map is None:
        edge_map, weight_map = edge_weight_from_image(img_rgb01)

    # numpy 변환
    imgs = (img_rgb01.detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)  # [B,3,H,W]
    edges = edge_map.detach().cpu().clamp(0, 1).numpy()  # [B,1,H,W]
    wmaps = weight_map.detach().cpu().clamp(0, 1).numpy()

    a = float(np.clip(alpha, 0.0, 1.0))
    q = float(np.clip(edge_quantile, 0.0, 1.0))

    for i in range(B):
        name = names[i]
        rgb = np.transpose(imgs[i], (1, 2, 0))  # H,W,3
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        E = edges[i, 0]  # [H,W], 0~1
        W = wmaps[i, 0]  # [H,W], 0~1

        # 1) Edge heatmap (TURBO)
        heat = _apply_colormap01(E, cv2.COLORMAP_TURBO)
        cv2.imwrite(os.path.join(out_dir, f"{name}_{prefix}_edge_heat.png"), heat)

        # 2) Edge overlay
        edge_overlay = cv2.addWeighted(bgr, 1.0 - a, heat, a, 0.0)
        cv2.imwrite(os.path.join(out_dir, f"{name}_{prefix}_edge_overlay.png"), edge_overlay)

        # 3) Smoothness weight heatmap (VIRIDIS; 밝을수록 smooth)
        w_heat = _apply_colormap01(W, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(os.path.join(out_dir, f"{name}_{prefix}_smooth_weight.png"), w_heat)

        # 4) Edge binary mask overlay (상위 quantile 빨강)
        thr = float(np.quantile(E, q))
        edge_mask = (E >= thr)  # shape: (H, W), bool

        red = np.zeros_like(bgr)
        red[..., 2] = 255

        # 전체 오버레이를 먼저 만든 뒤, 2D 마스크로 픽셀만 교체
        overlay_full = cv2.addWeighted(bgr, 1.0 - a, red, a, 0.0)

        mask_overlay = bgr.copy()
        mask_overlay[edge_mask] = overlay_full[edge_mask]

        cv2.imwrite(os.path.join(out_dir, f"{name}_{prefix}_edge_mask_overlay.png"), mask_overlay)


def save_flatmask_overlay(img_rgb01: torch.Tensor,
                          flat_full: torch.Tensor,
                          names: List[str],
                          out_dir: str,
                          alpha: float = 0.35,
                          prefix: str = "viz") -> None:
    """
    DINO 평탄 패치 full-res 마스크를 원본에 오버레이로 저장
    """
    if not HAS_CV2:
        raise RuntimeError("OpenCV(cv2)가 필요합니다. pip install opencv-python 로 설치하세요.")
    os.makedirs(out_dir, exist_ok=True)

    imgs = (img_rgb01.detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    masks = (flat_full.detach().cpu().numpy() > 0.5)

    for i in range(img_rgb01.shape[0]):
        name = os.path.splitext(str(names[i]))[0] if i < len(names) else f"sample_{i:03d}"
        rgb = np.transpose(imgs[i], (1, 2, 0))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        ms = masks[i, 0]  # H,W bool
        red = np.zeros_like(bgr); red[..., 2] = 255
        overlay_full = cv2.addWeighted(bgr, 1.0 - alpha, red, alpha, 0.0)

        out = bgr.copy()
        out[ms] = overlay_full[ms]
        cv2.imwrite(os.path.join(out_dir, f"{name}_{prefix}_flatmask_overlay.png"), out)


# ---------------------------
# (선택) 간단한 CLI
# ---------------------------

def _gather_images(input_path: str, recursive: bool = False) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    paths: List[str] = []
    if os.path.isdir(input_path):
        pattern = "**/*" if recursive else "*"
        for ext in exts:
            paths.extend(glob.glob(os.path.join(input_path, pattern, ext), recursive=recursive))
    else:
        # 단일 파일 or glob 패턴
        if any(ch in input_path for ch in ["*", "?", "["]):
            paths.extend(glob.glob(input_path))
        else:
            paths = [input_path]
    paths = [p for p in paths if os.path.isfile(p)]
    paths.sort()
    return paths


def _load_as_tensor(batch_files: List[str], device: torch.device) -> Tuple[torch.Tensor, List[str]]:
    """
    OpenCV로 이미지를 읽어 RGB [0,1]로 변환해 [B,3,H,W] 텐서로 반환.
    다양한 해상도가 섞여 있으면 가장 작은 H,W로 중앙 크롭/리사이즈를 고려할 수 있지만,
    여기서는 각 이미지를 '자체 해상도' 그대로 보존하기 위해 배치 불가 → 개별 처리 권장.
    본 유틸은 동일 해상도 이미지 배치에만 적합.
    """
    if not HAS_CV2:
        raise RuntimeError("OpenCV(cv2)가 필요합니다. pip install opencv-python 로 설치하세요.")
    imgs = []
    names = []
    H, W = None, None
    for p in batch_files:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] 이미지를 열 수 없음: {p}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if H is None and W is None:
            H, W = h, w
        if h != H or w != W:
            # 간단히 리사이즈로 맞춤 (원본 유지 원하면 개별 처리 권장)
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        imgs.append(img)
        names.append(os.path.basename(p))
    if len(imgs) == 0:
        raise RuntimeError("로딩된 이미지가 없습니다.")
    img_tensor = torch.stack(imgs, dim=0).to(device)
    return img_tensor, names


def main():
    ap = argparse.ArgumentParser(description="Edge-aware smoothness 시각화 + DINO flat patch mask")
    ap.add_argument("--input", type=str, default="/home/jaejun/dataset/MS2/sync_data/_2021-08-13-22-36-41/rgb/img_left/000128.png",
                    help="입력 이미지 파일/디렉토리/글롭 패턴")
    ap.add_argument("--out_dir", type=str, required=True, help="결과 저장 폴더")
    ap.add_argument("--recursive", action="store_true", help="디렉토리 검색 시 하위 폴더까지")
    ap.add_argument("--batch_size", type=int, default=8, help="동일 해상도 이미지 배치 크기")
    ap.add_argument("--alpha", type=float, default=0.4, help="오버레이 투명도(0~1)")
    ap.add_argument("--quantile", type=float, default=0.98, help="edge로 간주할 상위 quantile (0~1)")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="연산 디바이스")
    ap.add_argument("--prefix", type=str, default="viz", help="파일명 접미사 태그")

    # --- DINO flat patch 옵션 ---
    ap.add_argument("--use_dino_flat", action="store_true", help="DINO ViT-B/8로 평탄 패치 판정 및 edge=0 반영 시각화")
    ap.add_argument("--flat_thr", type=float, default=0.7, help="flat 판단용 코사인 유사도 임계")
    ap.add_argument("--flat_radius", type=int, default=1, help="방향별 탐색 반경 (패치 셀 단위)")

    args = ap.parse_args()

    if not HAS_CV2:
        print("ERROR: OpenCV(cv2)가 필요합니다. pip install opencv-python 로 설치하세요.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    paths = _gather_images(args.input, recursive=args.recursive)
    if len(paths) == 0:
        print("ERROR: 입력과 일치하는 이미지가 없습니다.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # DINO 백본 준비 (옵션)
    dino = None
    if args.use_dino_flat:
        try:
            dino = DinoVITB8Features(device)
        except Exception as e:
            print(f"[WARN] DINO 로드 실패 → flat 패치 기능 비활성화: {e}", file=sys.stderr)
            dino = None

    # 동일 해상도끼리 묶어서 처리
    by_size = {}
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] 이미지를 열 수 없음: {p}")
            continue
        h, w = img.shape[:2]
        by_size.setdefault((h, w), []).append(p)

    for (h, w), files in by_size.items():
        print(f"[INFO] 처리중 해상도 {h}x{w}, 파일 수={len(files)}")
        for s in range(0, len(files), args.batch_size):
            batch_files = files[s:s + args.batch_size]
            try:
                imgs, names = _load_as_tensor(batch_files, device)
            except Exception as e:
                print(f"[WARN] 배치 로드 실패({e}) → 개별 처리 시도")
                for pth in batch_files:
                    img1, n1 = _load_as_tensor([pth], device)
                    edge, weight = edge_weight_from_image(img1)
                    save_edge_smooth_viz_batch(
                        img_rgb01=img1, names=n1, out_dir=args.out_dir,
                        edge_map=edge, weight_map=weight,
                        alpha=args.alpha, edge_quantile=args.quantile, prefix=args.prefix
                    )

                    # DINO flat mask 적용 시각화
                    if args.use_dino_flat and dino is not None:
                        flat_full = build_flat_mask_fullres(img1, dino, thr=args.flat_thr, radius=args.flat_radius)
                        # edge=0 반영
                        edge_flat = edge.clone()
                        edge_flat[flat_full > 0.5] = 0.0
                        weight_flat = torch.exp(-edge_flat)

                        save_flatmask_overlay(img1, flat_full, n1, args.out_dir, alpha=0.35, prefix=args.prefix)
                        save_edge_smooth_viz_batch(
                            img_rgb01=img1, names=n1, out_dir=args.out_dir,
                            edge_map=edge_flat, weight_map=weight_flat,
                            alpha=args.alpha, edge_quantile=args.quantile, prefix=f"{args.prefix}_flat"
                        )
                continue

            # 기본 edge/weight
            edge, weight = edge_weight_from_image(imgs)
            save_edge_smooth_viz_batch(
                img_rgb01=imgs, names=names, out_dir=args.out_dir,
                edge_map=edge, weight_map=weight,
                alpha=args.alpha, edge_quantile=args.quantile, prefix=args.prefix
            )

            # DINO flat mask 적용
            if args.use_dino_flat and dino is not None:
                flat_full = build_flat_mask_fullres(imgs, dino, thr=args.flat_thr, radius=args.flat_radius)
                edge_flat = edge.clone()
                edge_flat[flat_full > 0.5] = 0.0
                weight_flat = torch.exp(-edge_flat)

                save_flatmask_overlay(imgs, flat_full, names, args.out_dir, alpha=0.35, prefix=args.prefix)
                save_edge_smooth_viz_batch(
                    img_rgb01=imgs, names=names, out_dir=args.out_dir,
                    edge_map=edge_flat, weight_map=weight_flat,
                    alpha=args.alpha, edge_quantile=args.quantile, prefix=f"{args.prefix}_flat"
                )

    print(f"[DONE] 저장 폴더: {args.out_dir}")


if __name__ == "__main__":
    main()
