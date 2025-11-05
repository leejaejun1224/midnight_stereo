import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt


# ===========================================
# 0) DINO 로드 & 전처리
# ===========================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_dino(device: torch.device):
    """
    facebookresearch/dino의 ViT-S/8 (dino_vits8) 모델 로드.
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

    # 인터리빙 규칙 (중요):
    # dy=4 → even rows(0,2,...), dy=0 → odd rows(1,3,...)
    # dx=4 → even cols(0,2,...), dx=0 → odd cols(1,3,...)
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
            # 전체 유효
            sim = (featL * featR).sum(dim=-1)  # [H4,W4]
            cost_vol[0] = sim
        else:
            # x in [d, W4-1]만 유효
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
# 3) 시각화
# ===========================================
def upsample_nearest_4x(map_2d: np.ndarray) -> np.ndarray:
    """
    [H4,W4] → [H4*4, W4*4] 간단 최근접 업샘플(시각화용)
    """
    return np.kron(map_2d, np.ones((4, 4), dtype=map_2d.dtype))

from matplotlib import cm  # 파일 상단 import 근처에 있어도 됩니다.

def visualize_results(left_img_pil: Image.Image,
                      disp_map: np.ndarray,
                      peak_sim: np.ndarray,
                      max_disp: int,
                      save_path: Path = None,
                      entropy_map: np.ndarray = None,
                      ent_vis_thr: float = None):
    """
    - 좌측 이미지 위에 disparity 오버레이
    - disparity 단독 히트맵
    - peak similarity 히트맵
    - (옵션) entropy 맵 (0..1)
    - (옵션) ent_vis_thr: entropy <= thr인 픽셀만 disparity 표시 (나머지는 투명)
    """
    img_np = np.asarray(left_img_pil)
    H, W = img_np.shape[:2]

    # ---- (추가) 엔트로피 임계값으로 마스크 적용 (1/4 그리드에서) ----
    disp_for_vis = disp_map.copy()
    if entropy_map is not None and ent_vis_thr is not None:
        mask_q = (entropy_map <= float(ent_vis_thr))             # True: 표시, False: 숨김
        disp_for_vis = np.where(mask_q, disp_for_vis, np.nan)    # 숨김 픽셀은 NaN

    # 1/4 → 원본 크기로 최근접 업샘플
    disp_up  = upsample_nearest_4x(disp_for_vis).astype(np.float32)  # [H,W] (NaN 유지)
    sim_up   = upsample_nearest_4x(peak_sim).astype(np.float32)
    ent_up   = upsample_nearest_4x(entropy_map).astype(np.float32) if entropy_map is not None else None

    num_cols = 4 if ent_up is not None else 3
    fig, axes = plt.subplots(1, num_cols, figsize=(6 * num_cols, 6))
    if num_cols == 3:
        ax0, ax1, ax2 = axes
    else:
        ax0, ax1, ax2, ax3 = axes

    fig.suptitle(f"Stereo Cost Volume (D=0..{max_disp}) from DINO ViT-S/8 (1/4-grid)")

    # ---- (중요) NaN을 투명하게 만들 컬러맵 준비 ----
    cmap_disp = cm.get_cmap('turbo').with_extremes(bad=(0, 0, 0, 0))  # NaN -> fully transparent

    # (1) 좌: 오리지널 + disparity 오버레이 (NaN은 투명)
    ax0.imshow(img_np)
    im0 = ax0.imshow(disp_up, vmin=0, vmax=max_disp, alpha=0.55, cmap=cmap_disp)
    ax0.set_title("Left image + disparity overlay (masked by entropy)")
    ax0.axis("off")
    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    cbar0.set_label("disparity (grid units; ~pixels = disp*4)")

    # (2) 중: disparity 히트맵 단독 (NaN은 투명)
    im1 = ax1.imshow(disp_up, vmin=0, vmax=max_disp, cmap=cmap_disp)
    ax1.set_title("Disparity map (entropy-masked; nearest x4)")
    ax1.axis("off")
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("disparity")

    # (3) 우: peak similarity
    im2 = ax2.imshow((sim_up + 1.0) / 2.0, vmin=0.0, vmax=1.0, cmap='viridis')
    ax2.set_title("Peak cosine similarity (0..1)")
    ax2.axis("off")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("cosine similarity")

    # (4) 옵션: entropy 맵
    if ent_up is not None:
        im3 = ax3.imshow(ent_up, vmin=0.0, vmax=1.0, cmap='magma')
        ax3.set_title("Entropy of disparity candidates (0..1)")
        ax3.axis("off")
        cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label("normalized entropy")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[Saved figure] {save_path}")

    plt.show()

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
    # per-pixel max로 중심 이동(안정성 + 대비 향상)
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
# 5) 메인
# ===========================================
def main():
    parser = argparse.ArgumentParser(description="Stereo cost-volume with DINO ViT-S/8 (1/4 grid)")
    parser.add_argument("--left",  type=str,default="/home/jaejun/dataset/MS2/sync_data/_2021-08-13-22-36-41/rgb/img_left/000096.png", help="Left image path")
    parser.add_argument("--right", type=str, default="/home/jaejun/dataset/MS2/sync_data/_2021-08-13-22-36-41/rgb/img_right/000096.png", help="Right image path")
    parser.add_argument("--max_disp", type=int, default=22, help="Max disparity on 1/4 grid (inclusive)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pad_to_8", action="store_true",
                        help="Pad right/bottom so H,W become multiples of 8 (both left & right).")
    parser.add_argument("--save_fig", type=str, default=None, help="Optional: path to save visualization (png)")
    parser.add_argument("--save_npy", type=str, default=None,
                        help="Optional: path to save npy (dict with 'disp', 'peak_sim')")
    parser.add_argument(
        "--ent_T", type=float, default=0.1,
        help="Temperature for entropy; smaller → sharper distribution (try 0.05~0.2)"
    )
    parser.add_argument(
        "--ent_vis_thr", type=float, default=0.9,
        help="Only plot disparity where entropy <= this threshold; others are hidden."
    )
    args = parser.parse_args()

    left_path  = Path(args.left);  right_path = Path(args.right)
    assert left_path.exists(),  f"Left image not found: {left_path}"
    assert right_path.exists(), f"Right image not found: {right_path}"

    # 1) 이미지 로드 (RGB)
    L_pil = Image.open(str(left_path)).convert("RGB")
    R_pil = Image.open(str(right_path)).convert("RGB")

    # 2) 크기 일치 검사
    assert L_pil.size == R_pil.size, \
        f"Left/Right size mismatch: left={L_pil.size}, right={R_pil.size}"

    # 3) (선택) 8의 배수로 패딩 (오른쪽/아래)
    if args.pad_to_8:
        L_pil = pad_right_bottom_to_multiple(L_pil, mult=8)
        R_pil = pad_right_bottom_to_multiple(R_pil, mult=8)

    W, H = L_pil.size
    assert (H % 8 == 0) and (W % 8 == 0), \
        "H,W must be multiples of 8. Use --pad_to_8 to pad automatically."

    # 4) 텐서 변환
    xL = pil_to_tensor(L_pil)  # [1,3,H,W]
    xR = pil_to_tensor(R_pil)

    # 5) DINO 로드
    device = torch.device(args.device)
    model = load_dino(device)

    # 6) 1/4 해상도 피처맵 생성
    with torch.no_grad():
        featL = build_quarter_features(model, xL)  # [H//4,W//4,C], L2-norm
        featR = build_quarter_features(model, xR)  # [H//4,W//4,C], L2-norm

    # 7) Cost volume & argmax disparity
    # 7) Cost volume & argmax disparity
    with torch.no_grad():
        cost_vol = build_cost_volume(featL, featR, args.max_disp)
        disp_map_t, peak_sim_t = argmax_disparity(cost_vol)
        # 온도 적용한 엔트로피
        entropy_t = build_entropy_map(cost_vol, T=args.ent_T, normalize=True)
     # [H4,W4]


    # 8) NumPy 변환
    disp_map  = disp_map_t.detach().cpu().numpy().astype(np.int32)       # [H4,W4]
    peak_sim  = peak_sim_t.detach().cpu().numpy().astype(np.float32)     # [H4,W4]
    entropy   = entropy_t.detach().cpu().numpy().astype(np.float32) 
    # 9) 시각화
    visualize_results(
        L_pil, disp_map, peak_sim, args.max_disp,
        save_path=Path(args.save_fig) if args.save_fig else None,
        entropy_map=entropy,
        ent_vis_thr=args.ent_vis_thr     # ★ 추가
    )

    # 10) 저장 옵션 (npy)
    if args.save_npy:
        out = {
            "disp": disp_map,           # 1/4-grid disparity (int32)
            "peak_sim": peak_sim,       # cosine similarity of best disp (float32, -1..1)
        }
        np.save(args.save_npy, out, allow_pickle=True)
        print(f"[Saved npy] {args.save_npy} (keys: disp, peak_sim)")


if __name__ == "__main__":
    main()
