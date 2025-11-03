import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt


# -----------------------------
# 0) 유틸: DINO 로드 & 전처리
# -----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_dino(device):
    """
    facebookresearch/dino(공식)에서 ViT-S/8 (dino_vits8) 모델을 로드.
    """
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    model.eval().to(device)
    return model

def preprocess_pil(img_pil):
    """
    PIL 이미지를 Tensor(BCHW)로 변환 + ImageNet 정규화.
    """
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    x = tfm(img_pil).unsqueeze(0)  # [1,3,H,W]
    return x


# ----------------------------------------------
# 1) 핵심: 시프트(0/4px)로 4개 피처맵 만들고 1/4 격자 생성
# ----------------------------------------------
def _extract_patch_tokens_dino(model, x):
    """
    DINO의 get_intermediate_layers를 사용해 마지막 레이어 출력(패치 토큰)을 [H8, W8, C]로 뽑아낸다.
    입력 x: [1,3,Hpad,Wpad]
    반환: tokens_hw: [H8, W8, C] (CLS 토큰 제거)
    """
    with torch.no_grad():
        # DINO는 유연한 입력 크기를 처리함(포지셔널 임베딩 보간 내부 지원).
        out = model.get_intermediate_layers(x, n=1)[0]  # [1, N(+cls), C]
        B, N, C = out.shape
        Hpad, Wpad = x.shape[-2:]
        h8 = Hpad // 8
        w8 = Wpad // 8

        # CLS 토큰 유무에 따라 분기
        if N == h8 * w8 + 1:
            tokens = out[:, 1:, :]  # CLS 제외
        elif N == h8 * w8:
            tokens = out
        else:
            raise RuntimeError(f"Unexpected token count: N={N}, grid={h8}x{w8}")

        tokens_hw = tokens.reshape(B, h8, w8, C).squeeze(0).contiguous()  # [H8,W8,C]
        return tokens_hw


def _shift_features_once(model, img_tensor, dx, dy, pad_mode='replicate'):
    """
    하나의 시프트 오프셋(dx, dy)에서 패치 피처맵을 추출하고,
    (H//8, W//8, C) 크기로 '원본 영역'만 잘라 리턴.
    - dx, dy ∈ {0, 4}
    - 오른쪽/아래는 stride=8에 맞추기 위해 자동으로 더 패딩됨.
    """
    _, _, H, W = img_tensor.shape
    # 오른쪽/아래는 (W+dx)%(8)==0 이 되도록 최소 패딩
    right  = (- (W + dx)) % 8
    bottom = (- (H + dy)) % 8

    # PyTorch F.pad의 인자 순서: (left, right, top, bottom)
    x_pad = F.pad(img_tensor, (dx, right, dy, bottom), mode=pad_mode)
    tokens_hw = _extract_patch_tokens_dino(model, x_pad)  # [H8',W8',C]

    # 원본 기준으로 유지해야 할 유효 토큰 수
    H8 = H // 8
    W8 = W // 8
    tokens_hw = tokens_hw[:H8, :W8, :]  # (dx,dy)==(4,*)일 때 생기는 가장자리 여분 제거
    return tokens_hw  # [H8,W8,C]


def build_quarter_res_features(model, img_tensor):
    """
    4개 시프트 (0,0), (4,0), (0,4), (4,4)를 통해
    최종 1/4 해상도 [H//4, W//4, C] 피처맵을 생성.
    짝수/홀수 행열에 각각 배치(interleave).
    """
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device, non_blocking=True)

    _, _, H, W = img_tensor.shape
    assert H % 8 == 0 and W % 8 == 0, \
        "입력 H,W는 8의 배수여야 합니다. (필요하면 외부에서 오른쪽/아래로 패딩하세요)"

    H8, W8 = H // 8, W // 8
    H4, W4 = H // 4, W // 4

    # 네 오프셋에서 피처 수집
    f00 = _shift_features_once(model, img_tensor, dx=0, dy=0)  # → (odd row, odd col)
    f40 = _shift_features_once(model, img_tensor, dx=4, dy=0)  # → (odd row, even col)
    f04 = _shift_features_once(model, img_tensor, dx=0, dy=4)  # → (even row, odd col)
    f44 = _shift_features_once(model, img_tensor, dx=4, dy=4)  # → (even row, even col)

    C = f00.shape[-1]
    Fq = torch.empty((H4, W4, C), device=device, dtype=f00.dtype)

    # 인터리빙 규칙 (중요):
    # - dy=4 → even rows (0,2,4,...), dy=0 → odd rows (1,3,5,...)
    # - dx=4 → even cols (0,2,4,...), dx=0 → odd cols (1,3,5,...)
    Fq[1::2, 1::2, :] = f00  # (0,0)
    Fq[1::2, 0::2, :] = f40  # (4,0)
    Fq[0::2, 1::2, :] = f04  # (0,4)
    Fq[0::2, 0::2, :] = f44  # (4,4)

    # 코사인 유사도 계산을 위한 L2 정규화
    Fq = F.normalize(Fq, dim=-1)
    return Fq  # [H4,W4,C] (L2 normalized)


# ---------------------------------------------------
# 2) 인터랙티브 뷰어: 클릭 → 코사인 유사도 히트맵 갱신
# ---------------------------------------------------
def interactive_view(img_pil, feats_q):
    """
    img_pil: 원본 이미지 (PIL)
    feats_q: [H//4, W//4, C] (torch, L2 정규화됨)
    - 좌: 원본 + 유사도 오버레이
    - 우: 1/4 격자 히트맵(원본 크기로 보간 표시)
    """
    img_np = np.asarray(img_pil)
    H, W = img_np.shape[:2]

    feats_np = feats_q.detach().cpu().numpy()  # [H4,W4,C]
    H4, W4, C = feats_np.shape

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Click on the left image to visualize cosine similarity (DINO ViT-S/8, 1/4 grid)")

    # 좌: 원본 이미지
    ax0.imshow(img_np)
    ax0.set_title("Image (click anywhere)")
    ax0.axis("off")

    # 좌 오버레이(초기값은 0)
    sim_init = np.zeros((H4, W4), dtype=np.float32)
    overlay_im = ax0.imshow(sim_init, vmin=0, vmax=1, alpha=0.5, cmap="jet",
                            interpolation="bilinear", extent=(0, W, H, 0))

    # 우: 히트맵(원본 크기 비율로 보여주기 위해 extent 지정)
    hm = ax1.imshow(sim_init, vmin=0, vmax=1, cmap="jet",
                    interpolation="nearest", extent=(0, W, H, 0))
    ax1.set_title("Cosine similarity heatmap (1/4 grid shown at image scale)")
    ax1.axis("off")
    cbar = fig.colorbar(hm, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("cosine similarity")

    click_marker, = ax0.plot([], [], marker='x', markersize=10, color='white', mew=2)

    def on_click(event):
        if event.inaxes != ax0:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)

        # 원본 픽셀 (x,y) → 1/4 격자 인덱스 (j,i)
        # 격자 i,j의 중심은 정확히 4*i, 4*j 에 대응됩니다.
        i = int(round(y / 4.0))
        j = int(round(x / 4.0))
        i = np.clip(i, 0, H4 - 1)
        j = np.clip(j, 0, W4 - 1)

        q = feats_np[i, j, :]                     # [C]
        sim = (feats_np * q[None, None, :]).sum(axis=2)  # [H4,W4] cos sim (이미 L2 정규화됨)
        sim = np.clip(sim, -1.0, 1.0)

        # 좌 오버레이/우 히트맵 업데이트
        overlay_im.set_data(sim)
        hm.set_data(sim)
        click_marker.set_data([x], [y])

        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show()


# -----------------------------
# 3) 실행 진입점
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="입력 이미지 경로")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pad_right_bottom_to_8", action="store_true",
                        help="입력 H,W가 8의 배수가 아니면 오른쪽/아래로 8의 배수까지 패딩합니다.")
    args = parser.parse_args()

    img_path = Path(args.image)
    assert img_path.exists(), f"Image not found: {img_path}"

    img_pil = Image.open(str(img_path)).convert("RGB")
    W, H = img_pil.size

    # 필요 시 오른쪽/아래 패딩(입력 자체를 8의 배수로 맞추기 위함)
    if args.pad_right_bottom_to_8:
        pad_r = (- W) % 8
        pad_b = (- H) % 8
        if pad_r or pad_b:
            # PIL에선 간단히 캔버스 확장(검정 패딩)
            new_img = Image.new("RGB", (W + pad_r, H + pad_b))
            new_img.paste(img_pil, (0, 0))
            img_pil = new_img
            W, H = img_pil.size

    # 전처리 & 모델 로드
    x = preprocess_pil(img_pil)  # [1,3,H,W] (이제 H,W는 8의 배수라고 가정)
    assert x.shape[-2] % 8 == 0 and x.shape[-1] % 8 == 0, \
        "입력 해상도가 8의 배수가 아닙니다. --pad_right_bottom_to_8 옵션을 써서 맞춰주세요."

    device = torch.device(args.device)
    model = load_dino(device)

    # 1/4 해상도 피처맵 만들기
    with torch.no_grad():
        feats_q = build_quarter_res_features(model, x)  # [H//4,W//4,C], L2 정규화 끝남

    # 인터랙티브 뷰어 실행
    interactive_view(img_pil, feats_q)


if __name__ == "__main__":
    main()
