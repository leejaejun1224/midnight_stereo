import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import cm

# ===========================================
# 0) DINO 로드 & 전처리
# ===========================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

def load_dino(device: torch.device):
    """
    facebookresearch/dino의 ViT-B/8 (dino_vitb8) 모델 로드.
    - 가변 입력 크기 지원(포지셔널 임베딩 내부 보간)
    """
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
    model.eval().to(device)
    return model

def pil_to_tensor(img_pil: Image.Image) -> torch.Tensor:
    """
    PIL → Tensor [1,3,H,W] + ImageNet 정규화
    """
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    x = tfm(img_pil).unsqueeze(0)  # [1,3,H,W]
    return x


# ===========================================
# 1) ViT-S/8 패치 토큰 추출 + 1/4 격자 생성(학습 없음)
#    (0/4 px 시프트-패딩 4회 → 인터리빙)
# ===========================================
@torch.no_grad()
def _extract_patch_tokens_dino(model, x: torch.Tensor) -> torch.Tensor:
    """
    DINO의 get_intermediate_layers를 사용해 마지막 레이어 패치 토큰을 [H8,W8,C]로 반환.
    입력 x: [1,3,Hpad,Wpad] (이미 패딩 완료본)
    반환: [H8, W8, C]  (CLS 토큰 제거)
    """
    out = model.get_intermediate_layers(x, n=1)[0]  # [1, N(+cls), C]
    B, N, C = out.shape
    Hpad, Wpad = x.shape[-2:]
    h8 = Hpad // 8
    w8 = Wpad // 8

    if N == h8 * w8 + 1:
        tokens = out[:, 1:, :]
    elif N == h8 * w8:
        tokens = out
    else:
        raise RuntimeError(f"Unexpected token count: N={N}, expected {h8*w8} or {h8*w8+1}")

    tokens_hw = tokens.reshape(B, h8, w8, C).squeeze(0).contiguous()  # [H8,W8,C]
    return tokens_hw


@torch.no_grad()
def _shift_once(model, img_tensor: torch.Tensor, dx: int, dy: int, pad_mode='replicate') -> torch.Tensor:
    """
    (dx,dy) ∈ {0,4} 시프트-패딩 한 입력에서 패치 토큰을 추출.
    오른쪽/아래는 stride=8 정합을 위해 자동 패딩.
    반환: [H//8, W//8, C] (원본 영역 기준으로 잘라냄)
    """
    _, _, H, W = img_tensor.shape
    right  = (8 - (W + dx) % 8) % 8
    bottom = (8 - (H + dy) % 8) % 8

    # F.pad: (left, right, top, bottom)
    x_pad = F.pad(img_tensor, (dx, right, dy, bottom), mode=pad_mode)
    tokens = _extract_patch_tokens_dino(model, x_pad)  # [H8',W8',C]

    H8 = H // 8
    W8 = W // 8
    tokens = tokens[:H8, :W8, :]  # 원본 범위만 사용
    return tokens


@torch.no_grad()
def build_quarter_features(model, img_tensor: torch.Tensor) -> torch.Tensor:
    """
    4개 오프셋 (0,0), (4,0), (0,4), (4,4) → 인터리빙하여
    최종 1/4 해상도 [H//4, W//4, C] 피처맵을 생성 (L2 정규화 포함).
    """
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device, non_blocking=True)

    _, _, H, W = img_tensor.shape
    assert H % 8 == 0 and W % 8 == 0, \
        "입력 H,W는 8의 배수여야 합니다. (--pad_to_8 옵션으로 자동 패딩 가능)"

    H4, W4 = H // 4, W // 4

    f00 = _shift_once(model, img_tensor, dx=0, dy=0)  # [H8,W8,C]
    f40 = _shift_once(model, img_tensor, dx=4, dy=0)
    f04 = _shift_once(model, img_tensor, dx=0, dy=4)
    f44 = _shift_once(model, img_tensor, dx=4, dy=4)

    C = f00.shape[-1]
    Fq = torch.empty((H4, W4, C), device=device, dtype=f00.dtype)

    # 인터리빙 규칙:
    Fq[1::2, 1::2, :] = f00  # (0,0)
    Fq[1::2, 0::2, :] = f40  # (4,0)
    Fq[0::2, 1::2, :] = f04  # (0,4)
    Fq[0::2, 0::2, :] = f44  # (4,4)

    # 코사인 유사도 계산을 위한 L2 정규화
    Fq = F.normalize(Fq, dim=-1)
    return Fq  # [H4,W4,C]


# ===========================================
# 2) Cost Volume 구축 (좌→우, 수평 시차만)
# ===========================================
@torch.no_grad()
def build_cost_volume(featL: torch.Tensor, featR: torch.Tensor, max_disp: int) -> torch.Tensor:
    """
    featL, featR: [H4, W4, C] (L2 정규화됨)
    max_disp: 1/4 격자 단위의 최대 disparity (포함, 즉 0..max_disp)

    반환: cost_vol [D+1, H4, W4] (각 d에서 L·Rshift 내적 = cos sim)
    - 정의: E(d)[y,x] = dot( featL[y,x], featR[y,x-d] )  (x-d<0는 invalid)
    """
    assert featL.shape == featR.shape
    H4, W4, C = featL.shape
    device = featL.device
    D = int(max_disp)

    cost_vol = torch.full((D + 1, H4, W4), float('-inf'), device=device, dtype=featL.dtype)

    for d in range(D + 1):
        if d == 0:
            sim = (featL * featR).sum(dim=-1)  # [H4,W4]
            cost_vol[0] = sim
        else:
            left_slice  = featL[:, d:, :]      # [H4, W4-d, C]
            right_slice = featR[:, :-d, :]     # [H4, W4-d, C]
            sim = (left_slice * right_slice).sum(dim=-1)  # [H4, W4-d]
            cost_vol[d, :, d:] = sim  # invalid 구간(0..d-1)은 -inf 유지

    return cost_vol  # [D+1,H4,W4]


@torch.no_grad()
def argmax_disparity(cost_vol: torch.Tensor):
    """
    cost_vol: [D+1, H4, W4]
    반환:
      - disp_map: [H4,W4] (long), 각 위치에서 argmax disparity
      - peak_sim: [H4,W4] (float), 해당 disparity에서의 유사도 값
    """
    peak_sim, disp_map = cost_vol.max(dim=0)  # over disparity axis
    return disp_map, peak_sim


# ===========================================
# 2-1) Entropy map from cost volume
# ===========================================
@torch.no_grad()
def build_entropy_map(cost_vol: torch.Tensor,
                      T: float = 0.1,
                      eps: float = 1e-8,
                      normalize: bool = True) -> torch.Tensor:
    """
    cost_vol: [D+1, H4, W4]  (invalid는 -inf)
    T: temperature (작을수록 분포가 날카로워져 엔트로피가 낮아짐)
    반환: ent_map [H4, W4], (정규화 시 0..1)
    """
    # per-pixel max로 중심 이동
    m = torch.amax(cost_vol, dim=0, keepdim=True)              # [1,H4,W4]
    logits = (cost_vol - m) / max(T, eps)                      # [D+1,H4,W4]

    prob = torch.softmax(logits, dim=0)                        # [D+1,H4,W4]
    p = prob.clamp_min(eps)
    ent = -(p * p.log()).sum(dim=0)                            # [H4,W4]

    if normalize:
        valid = torch.isfinite(cost_vol)                       # [D+1,H4,W4]
        Deff  = valid.sum(dim=0).clamp_min(1).to(p.dtype)      # [H4,W4]
        ent = torch.where(Deff > 1, ent / (Deff.log() + eps), torch.zeros_like(ent))
        ent = ent.clamp_(0.0, 1.0)
    return ent


# ===========================================
# 2-2) Top-2 disparity index gap from cost volume
# ===========================================
@torch.no_grad()
def build_top2_gap_map(cost_vol: torch.Tensor) -> torch.Tensor:
    """
    cost_vol: [D+1, H4, W4]  (invalid는 -inf)
    반환: gap_map [H4, W4] (float)
      - 각 픽셀에서 상위 2개 후보 disparity 인덱스의 절대 차이 |d1 - d2|
      - 유효 후보가 2개 미만이면 NaN으로 둬서 시각화 시 투명 처리
    """
    Dp1 = cost_vol.shape[0]
    if Dp1 < 2:
        # disparity 후보가 1개뿐이면 전부 NaN
        H4, W4 = cost_vol.shape[1:]
        return torch.full((H4, W4), float('nan'), device=cost_vol.device, dtype=cost_vol.dtype)

    valid = torch.isfinite(cost_vol)                 # [D+1,H4,W4]
    Deff  = valid.sum(dim=0)                         # [H4,W4]

    # 상위 2개 값/인덱스
    vals, idxs = torch.topk(cost_vol, k=2, dim=0)   # [2,H4,W4], [2,H4,W4]
    d1 = idxs[0].to(torch.float32)
    d2 = idxs[1].to(torch.float32)
    gap = (d1 - d2).abs()                           # [H4,W4]

    # 유효 후보가 2개 미만인 곳은 NaN
    gap = torch.where(Deff >= 2, gap, torch.full_like(gap, float('nan')))
    return gap


# ===========================================
# 3) 시각화 (개별 패널 저장용 + 기존 4-up)
# ===========================================
def upsample_nearest_4x(map_2d: np.ndarray) -> np.ndarray:
    """
    [H4,W4] → [H4*4, W4*4] 간단 최근접 업샘플(시각화용)
    """
    return np.kron(map_2d, np.ones((4, 4), dtype=map_2d.dtype))

def _get_transparent_disp_cmap():
    # turbo가 없거나 with_extremes 미지원일 때를 대비해 안전 처리
    try:
        cmap = cm.get_cmap('turbo')
    except Exception:
        cmap = cm.get_cmap('plasma')
    if hasattr(cmap, 'with_extremes'):
        cmap = cmap.with_extremes(bad=(0, 0, 0, 0))
    else:
        try:
            cmap = cmap.copy()
        except Exception:
            pass
        try:
            cmap.set_bad((0, 0, 0, 0))
        except Exception:
            pass
    return cmap

def visualize_results(left_img_pil: Image.Image,
                      disp_map: np.ndarray,
                      peak_sim: np.ndarray,
                      max_disp: int,
                      save_path: Path = None,
                      entropy_map: np.ndarray = None,
                      ent_vis_thr: float = None,
                      top2_gap_map: np.ndarray = None,
                      gap_min: float = 2.0):
    """
    (기존) 4-up(또는 3-up) Figure를 화면/파일로 출력
    + top2_gap_map 추가 시, 패널 1개 추가(최대 5-up)
    """
    img_np = np.asarray(left_img_pil)

    # 엔트로피 마스크 적용 (disparity에만)
    disp_for_vis = disp_map.copy()
    if entropy_map is not None and ent_vis_thr is not None:
        mask_q = (entropy_map <= float(ent_vis_thr))
        disp_for_vis = np.where(mask_q, disp_for_vis, np.nan)

    # 업샘플
    disp_up = upsample_nearest_4x(disp_for_vis).astype(np.float32)
    sim_up  = upsample_nearest_4x(peak_sim).astype(np.float32)
    ent_up  = upsample_nearest_4x(entropy_map).astype(np.float32) if entropy_map is not None else None
    gap_up  = upsample_nearest_4x(top2_gap_map).astype(np.float32) if top2_gap_map is not None else None

    has_ent = ent_up is not None
    has_gap = gap_up is not None
    num_cols = 3 + int(has_ent) + int(has_gap)

    fig, axes = plt.subplots(1, num_cols, figsize=(6 * num_cols, 6))
    if num_cols == 1:
        axes = [axes]
    else:
        axes = np.ravel(axes).tolist()

    fig.suptitle(f"Stereo Cost Volume (D=0..{max_disp}) from DINO ViT-B/8 (1/4-grid)")
    cmap_disp = _get_transparent_disp_cmap()

    # (1) 오버레이
    ax0 = axes[0]
    ax0.imshow(img_np)
    im0 = ax0.imshow(disp_up, vmin=0, vmax=max_disp, alpha=0.55, cmap=cmap_disp)
    ax0.set_title("Left image + disparity overlay (masked by entropy)")
    ax0.axis("off")
    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    cbar0.set_label("disparity (grid units; ~pixels = disp*4)")

    # (2) disparity
    ax1 = axes[1]
    im1 = ax1.imshow(disp_up, vmin=0, vmax=max_disp, cmap=cmap_disp)
    ax1.set_title("Disparity map (entropy-masked; nearest x4)")
    ax1.axis("off")
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("disparity")

    # (3) peak similarity
    ax2 = axes[2]
    im2 = ax2.imshow((sim_up + 1.0) / 2.0, vmin=0.0, vmax=1.0, cmap='viridis')
    ax2.set_title("Peak cosine similarity (0..1)")
    ax2.axis("off")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("cosine similarity")

    col_idx = 3

    # (4) entropy (옵션)
    if has_ent:
        ax = axes[col_idx]; col_idx += 1
        im = ax.imshow(ent_up, vmin=0.0, vmax=1.0, cmap='magma')
        ax.set_title("Entropy of disparity candidates (0..1)")
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("normalized entropy")

    # (5) top-2 disparity gap (옵션) — 차이<gap_min 은 투명
    if has_gap:
        ax = axes[col_idx]
        gap_vis = np.where(gap_up >= float(gap_min), gap_up, np.nan)
        cmap_gap = _get_transparent_disp_cmap()
        im = ax.imshow(gap_vis, vmin=float(gap_min), vmax=max_disp, cmap=cmap_gap)
        ax.set_title(f"Top-2 disparity index gap (>= {gap_min}; grid units)")
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("abs(d1 - d2)")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[Saved figure] {save_path}")

    plt.show()

def save_viz_panels(left_img_pil: Image.Image,
                    disp_map: np.ndarray,
                    peak_sim: np.ndarray,
                    max_disp: int,
                    entropy_map: np.ndarray,
                    ent_vis_thr: float,
                    out_dir: Path,
                    stem: str,
                    top2_gap_map: np.ndarray = None,
                    gap_min: float = 2.0):
    """
    현재 5개의 시각화 결과를 개별 PNG로 저장:
      - overlay/<stem>.png
      - disp/<stem>.png
      - peak_sim/<stem>.png
      - entropy/<stem>.png
      - top2_gap/<stem>.png
    """
    out_dir = Path(out_dir)
    (out_dir / "overlay").mkdir(parents=True, exist_ok=True)
    (out_dir / "disp").mkdir(parents=True, exist_ok=True)
    (out_dir / "peak_sim").mkdir(parents=True, exist_ok=True)
    (out_dir / "entropy").mkdir(parents=True, exist_ok=True)
    if top2_gap_map is not None:
        (out_dir / "top2_gap").mkdir(parents=True, exist_ok=True)

    img_np = np.asarray(left_img_pil)
    H, W = img_np.shape[:2]

    # ---- 엔트로피 마스크 적용(1/4 그리드) ----
    disp_for_vis = disp_map.copy()
    if entropy_map is not None and ent_vis_thr is not None:
        mask_q = (entropy_map <= float(ent_vis_thr))
        disp_for_vis = np.where(mask_q, disp_for_vis, np.nan)

    # 1/4 → 원본 크기
    disp_up = upsample_nearest_4x(disp_for_vis).astype(np.float32)  # [H,W]
    sim_up  = upsample_nearest_4x(peak_sim).astype(np.float32)      # [H,W]
    ent_up  = upsample_nearest_4x(entropy_map).astype(np.float32)   # [H,W]

    cmap_disp = _get_transparent_disp_cmap()

    # (1) overlay
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.imshow(img_np)
    im = ax.imshow(disp_up, vmin=0, vmax=max_disp, alpha=0.55, cmap=cmap_disp)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("disparity (grid units; ~pixels = disp*4)")
    overlay_path = out_dir / "overlay" / f"{stem}.png"
    fig.savefig(overlay_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    # (2) disparity
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(disp_up, vmin=0, vmax=max_disp, cmap=cmap_disp)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("disparity")
    disp_path = out_dir / "disp" / f"{stem}.png"
    fig.savefig(disp_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    # (3) peak similarity (0..1)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow((sim_up + 1.0) / 2.0, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cosine similarity")
    sim_path = out_dir / "peak_sim" / f"{stem}.png"
    fig.savefig(sim_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    # (4) entropy (0..1)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(ent_up, vmin=0.0, vmax=1.0, cmap="magma")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("normalized entropy")
    ent_path = out_dir / "entropy" / f"{stem}.png"
    fig.savefig(ent_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    # (5) top-2 disparity gap (>= gap_min만 보이도록)
    gap_path = None
    if top2_gap_map is not None:
        gap_up = upsample_nearest_4x(top2_gap_map).astype(np.float32)  # [H,W]
        gap_vis = np.where(gap_up >= float(gap_min), gap_up, np.nan)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        cmap_gap = _get_transparent_disp_cmap()
        im = ax.imshow(gap_vis, vmin=float(gap_min), vmax=max_disp, cmap=cmap_gap)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("top-2 disparity gap (grid units)")
        gap_path = out_dir / "top2_gap" / f"{stem}.png"
        fig.savefig(gap_path, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)

    print(f"[Saved panels] {overlay_path}, {disp_path}, {sim_path}, {ent_path}" + \
          (f", {gap_path}" if gap_path is not None else ""))


# ===========================================
# 4) 보조: 오른쪽/아래 패딩으로 8의 배수 맞추기
# ===========================================
def pad_right_bottom_to_multiple(img_pil: Image.Image, mult: int = 8) -> Image.Image:
    """
    입력 이미지를 오른쪽/아래 방향으로만 패딩해서 (H,W)가 mult의 배수가 되도록 맞춤.
    """
    W, H = img_pil.size
    pad_w = (mult - (W % mult)) % mult
    pad_h = (mult - (H % mult)) % mult
    if pad_w == 0 and pad_h == 0:
        return img_pil
    new_img = Image.new("RGB", (W + pad_w, H + pad_h))
    new_img.paste(img_pil, (0, 0))
    return new_img


# ===========================================
# 5) 메인
# ===========================================
def main():
    parser = argparse.ArgumentParser(description="Stereo cost-volume with DINO ViT-B/8 (1/4 grid)")
    # 단일 파일 모드(기존)
    parser.add_argument("--left",  type=str, default=None, help="Left image path")
    parser.add_argument("--right", type=str, default=None, help="Right image path")
    # 디렉터리 모드(신규)
    parser.add_argument("--left_dir",  type=str, default=None, help="Directory of left images")
    parser.add_argument("--right_dir", type=str, default=None, help="Directory of right images")
    parser.add_argument("--glob", type=str, default="*.png", help="Glob pattern for left images in left_dir")

    parser.add_argument("--max_disp", type=int, default=22, help="Max disparity on 1/4 grid (inclusive)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pad_to_8", action="store_true",
                        help="Pad right/bottom so H,W become multiples of 8 (both left & right).")
    parser.add_argument("--save_fig", type=str, default=None,
                        help="(Optional) path to save multi-panel visualization (png) for single-file mode")
    parser.add_argument("--out_dir", type=str, default="./out_dir",
                        help="(Directory mode) Root output directory for panels")
    parser.add_argument("--save_numpy", action="store_true",
                        help="(Directory mode) Also save disp/peak_sim/entropy/top2_gap as npz under out_dir/npy")

    parser.add_argument("--ent_T", type=float, default=0.1,
                        help="Temperature for entropy; smaller → sharper distribution (try 0.05~0.2)")
    parser.add_argument("--ent_vis_thr", type=float, default=0.9,
                        help="Only plot disparity where entropy <= this threshold; others are hidden.")

    # NEW: gap 시각화 임계
    parser.add_argument("--gap_min", type=float, default=1.0,
                        help="Visualize top-2 disparity index gap >= this threshold (grid units; ~pixels = gap*4)")

    args = parser.parse_args()
    device = torch.device(args.device)

    # 모드 판별
    dir_mode = (args.left_dir is not None and args.right_dir is not None)
    file_mode = (args.left is not None and args.right is not None)

    assert dir_mode or file_mode, \
        "하나를 선택하세요: (1) --left/--right (단일 파일) 또는 (2) --left_dir/--right_dir (디렉터리)."
    if dir_mode and file_mode:
        raise AssertionError("단일 파일 모드와 디렉터리 모드를 동시에 사용할 수 없습니다.")

    # DINO 모델은 한 번만 로드
    model = load_dino(device)

    if file_mode:
        # ===== 단일 파일 모드 =====
        left_path  = Path(args.left);  right_path = Path(args.right)
        assert left_path.exists(),  f"Left image not found: {left_path}"
        assert right_path.exists(), f"Right image not found: {right_path}"

        L_pil = Image.open(str(left_path)).convert("RGB")
        R_pil = Image.open(str(right_path)).convert("RGB")
        assert L_pil.size == R_pil.size, \
            f"Left/Right size mismatch: left={L_pil.size}, right={R_pil.size}"

        if args.pad_to_8:
            L_pil = pad_right_bottom_to_multiple(L_pil, mult=8)
            R_pil = pad_right_bottom_to_multiple(R_pil, mult=8)

        W, H = L_pil.size
        assert (H % 8 == 0) and (W % 8 == 0), \
            "H,W must be multiples of 8. Use --pad_to_8 to pad automatically."

        xL = pil_to_tensor(L_pil)  # [1,3,H,W]
        xR = pil_to_tensor(R_pil)

        with torch.no_grad():
            featL = build_quarter_features(model, xL)  # [H//4,W//4,C]
            featR = build_quarter_features(model, xR)

            cost_vol = build_cost_volume(featL, featR, args.max_disp)
            disp_map_t, peak_sim_t = argmax_disparity(cost_vol)
            entropy_t  = build_entropy_map(cost_vol, T=args.ent_T, normalize=True)
            top2_gap_t = build_top2_gap_map(cost_vol)  # NEW

        disp_map = disp_map_t.detach().cpu().numpy().astype(np.int32)
        peak_sim = peak_sim_t.detach().cpu().numpy().astype(np.float32)
        entropy  = entropy_t.detach().cpu().numpy().astype(np.float32)
        top2_gap = top2_gap_t.detach().cpu().numpy().astype(np.float32)  # NEW

        visualize_results(
            L_pil, disp_map, peak_sim, args.max_disp,
            save_path=Path(args.save_fig) if args.save_fig else None,
            entropy_map=entropy,
            ent_vis_thr=args.ent_vis_thr,
            top2_gap_map=top2_gap,        # NEW
            gap_min=args.gap_min          # NEW
        )
        return

    # ===== 디렉터리 모드 =====
    left_dir  = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    out_dir   = Path(args.out_dir)

    assert left_dir.is_dir(),  f"--left_dir not found: {left_dir}"
    assert right_dir.is_dir(), f"--right_dir not found: {right_dir}"

    # left_dir에서 glob으로 왼쪽 이미지 목록
    left_files = sorted(left_dir.glob(args.glob))
    assert len(left_files) > 0, f"No files matching {args.glob} in {left_dir}"

    # npz 저장 폴더(옵션)
    if args.save_numpy:
        (out_dir / "npy").mkdir(parents=True, exist_ok=True)

    # 오른쪽 이미지 매칭 함수(동일 stem 우선)
    def find_right_for_left(left_path: Path) -> Path:
        stem = left_path.stem
        for ext in IMG_EXTS:
            cand = right_dir / f"{stem}{ext}"
            if cand.exists():
                return cand
        # 확장자/이름이 다르면 실패
        return None

    processed, skipped = 0, 0
    for lp in left_files:
        rp = find_right_for_left(lp)
        if rp is None:
            print(f"[Skip] Right not found for left={lp.name}")
            skipped += 1
            continue

        try:
            # 1) 로드
            L_pil = Image.open(str(lp)).convert("RGB")
            R_pil = Image.open(str(rp)).convert("RGB")

            # 2) 크기 체크
            if L_pil.size != R_pil.size:
                print(f"[Skip] size mismatch: {lp.name} vs {rp.name} => {L_pil.size} != {R_pil.size}")
                skipped += 1
                continue

            # 3) 패딩
            if args.pad_to_8:
                L_pil = pad_right_bottom_to_multiple(L_pil, mult=8)
                R_pil = pad_right_bottom_to_multiple(R_pil, mult=8)

            W, H = L_pil.size
            if (H % 8 != 0) or (W % 8 != 0):
                print(f"[Skip] not multiple of 8 even after padding: {lp.name}")
                skipped += 1
                continue

            # 4) 텐서 변환
            xL = pil_to_tensor(L_pil)
            xR = pil_to_tensor(R_pil)

            # 5) 특징/코스트/추정
            with torch.no_grad():
                featL = build_quarter_features(model, xL)
                featR = build_quarter_features(model, xR)

                cost_vol = build_cost_volume(featL, featR, args.max_disp)
                disp_map_t, peak_sim_t = argmax_disparity(cost_vol)
                entropy_t  = build_entropy_map(cost_vol, T=args.ent_T, normalize=True)
                top2_gap_t = build_top2_gap_map(cost_vol)  # NEW

            # 6) NumPy 변환
            disp_map = disp_map_t.detach().cpu().numpy().astype(np.int32)
            peak_sim = peak_sim_t.detach().cpu().numpy().astype(np.float32)
            entropy  = entropy_t.detach().cpu().numpy().astype(np.float32)
            top2_gap = top2_gap_t.detach().cpu().numpy().astype(np.float32)  # NEW

            # 7) 패널 저장 (최대 5개 PNG)
            stem = lp.stem  # 파일이름(확장자 제외)
            save_viz_panels(
                L_pil, disp_map, peak_sim, args.max_disp,
                entropy_map=entropy,
                ent_vis_thr=args.ent_vis_thr,
                out_dir=out_dir,
                stem=stem,
                top2_gap_map=top2_gap,     # NEW
                gap_min=args.gap_min       # NEW
            )

            # 8) (옵션) npz 저장
            if args.save_numpy:
                npz_path = out_dir / "npy" / f"{stem}.npz"
                np.savez_compressed(
                    npz_path,
                    disp=disp_map,      # int32
                    peak_sim=peak_sim,  # float32 (-1..1)
                    entropy=entropy,    # float32 (0..1)
                    top2_gap=top2_gap   # float32 (grid units; NaN allowed)
                )
            processed += 1

        except Exception as e:
            print(f"[Error] {lp.name}: {e}")
            skipped += 1

    print(f"[Done] processed={processed}, skipped={skipped}, out_dir={out_dir.resolve()}")

if __name__ == "__main__":
    main()
