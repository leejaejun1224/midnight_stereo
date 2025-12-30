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
    DINO ViT-Base/8 (dino_vitb8) 모델 로드.
    """
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
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
# 1) ViT-B/8 패치 토큰 → 1/8 해상도 feature
# ----------------------------------------------
def _extract_patch_tokens_dino(model, x):
    """
    DINO get_intermediate_layers로 마지막 레이어 패치 토큰을 [H8, W8, C]로 뽑는다.
    입력 x: [1,3,Hpad,Wpad]
    반환: tokens_hw: [H8, W8, C] (CLS 토큰 제거)
    """
    with torch.no_grad():
        out = model.get_intermediate_layers(x, n=1)[0]  # [1, N(+cls), C]
        B, N, C = out.shape
        Hpad, Wpad = x.shape[-2:]

        h8 = Hpad // 8
        w8 = Wpad // 8

        if N == h8 * w8 + 1:
            tokens = out[:, 1:, :]  # CLS 제외
        elif N == h8 * w8:
            tokens = out
        else:
            raise RuntimeError(f"Unexpected token count: N={N}, grid={h8}x{w8}")

        tokens_hw = tokens.reshape(B, h8, w8, C).squeeze(0).contiguous()  # [H8,W8,C]
        return tokens_hw


def build_eighth_res_features(model, img_tensor):
    """
    ViT-B/8에서 바로 나온 1/8 해상도 feature:
      - 입력: img_tensor [1,3,H,W] (H,W는 8의 배수)
      - 출력: F8 [H//8, W//8, C], L2 정규화
    """
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device, non_blocking=True)

    _, _, H, W = img_tensor.shape
    assert H % 8 == 0 and W % 8 == 0, \
        "입력 H,W는 8의 배수여야 합니다. (--pad_right_bottom_to_8 옵션으로 맞춰주세요)"

    tokens_hw = _extract_patch_tokens_dino(model, img_tensor)  # [H8,W8,C]
    F8 = F.normalize(tokens_hw, dim=-1)  # cos similarity용 L2 정규화
    return F8  # [H8,W8,C]


# ---------------------------------------------------
# 2) 1/8 feature → stereo cost volume
# ---------------------------------------------------
def build_cost_volume_from_features(feats_l, feats_r, max_disp, downsample=8):
    """
    feats_l, feats_r: [Hf, Wf, C] (1/8 해상도, L2 normalized)
    max_disp: 원본 해상도 기준 최대 disparity (px)
    downsample: 8  (1/8 해상도)

    반환:
      cost_volume: [1, D, Hf, Wf]
        D = max_disp // downsample + 1
        disparity 후보 = 0, downsample, 2*downsample, ...
      score(d,y,x) = dot( feat_l(y,x), feat_r(y,x-d) )
        (x-d < 0 는 invalid → 아주 낮은 score로 채움)
    """
    assert feats_l.shape == feats_r.shape, "좌/우 feature 크기가 다릅니다."
    Hf, Wf, C = feats_l.shape

    max_d_f = max_disp // downsample
    D = max_d_f + 1

    # [1,C,Hf,Wf]로 변환
    feat_l = feats_l.permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
    feat_r = feats_r.permute(2, 0, 1).unsqueeze(0)
    _, C, H, W = feat_l.shape

    # invalid disparity는 아주 작은 score로 (softmax에서 확률 ~0 되도록)
    cost_volume = feat_l.new_full((1, D, H, W), fill_value=-1e9)  # [1,D,H,W]

    for d in range(D):
        if d == 0:
            sim = (feat_l * feat_r).sum(dim=1)        # [1,H,W]
            cost_volume[0, 0, :, :] = sim[0]
        else:
            # left(x,y) ↔ right(x-d,y)
            sim = (feat_l[:, :, :, d:] * feat_r[:, :, :, :W-d]).sum(dim=1)  # [1,H,W-d]
            cost_volume[0, d, :, d:] = sim[0]

    return cost_volume  # [1,D,Hf,Wf]


# ---------------------------------------------------
# 3) 클릭 → disparity 확률 분포 그래프 저장
# ---------------------------------------------------
def interactive_disp_view(left_img_pil, cost_volume, downsample=8, save_dir=Path("./disp_plots")):
    """
    left_img_pil: 왼쪽 원본 이미지 (PIL)
    cost_volume: [1, D, Hf, Wf] (torch, CPU 권장)
    downsample: 8 (1/8 해상도)
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    img_np = np.asarray(left_img_pil)
    H, W = img_np.shape[:2]

    cv = cost_volume.squeeze(0)  # [D,Hf,Wf]
    D, Hf, Wf = cv.shape

    fig, ax = plt.subplots()
    ax.imshow(img_np)
    ax.set_title("click")
    ax.axis("off")

    click_marker, = ax.plot([], [], marker='x', markersize=10, color='white', mew=2)

    def onclick(event, cv=cv, factor=downsample, sdir=save_dir):
        if event.inaxes is not ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x_full = int(round(event.xdata))
        y_full = int(round(event.ydata))
        x_full = np.clip(x_full, 0, W - 1)
        y_full = np.clip(y_full, 0, H - 1)

        # 원본 → feature grid index (1/8)
        i = int(round(y_full / factor))   # row
        j = int(round(x_full / factor))   # col
        i = np.clip(i, 0, Hf - 1)
        j = np.clip(j, 0, Wf - 1)

        # 해당 픽셀의 disparity score 벡터 [D]
        scores = cv[:, i, j]  # torch [D]
        T=0.1
        # softmax(score) → 각 disparity 후보의 확률
        probs = torch.softmax(scores/T, dim=0)

        # disparity 후보 (px 단위)
        disp_vals = torch.arange(D, dtype=torch.float32) * factor  # [0, 8, 16, ...]

        # 그래프 그리기 + 저장
        fig2, ax2 = plt.subplots()
        ax2.plot(disp_vals.numpy(), probs.numpy(), marker='o')
        ax2.set_xlabel("Disparity (pixels)")
        ax2.set_ylabel("Probability")
        ax2.set_title(f"Pixel ({x_full}, {y_full}) disparity distribution")
        ax2.grid(True)

        out_path = sdir / f"disp_prob_x{x_full}_y{y_full}.png"
        fig2.savefig(out_path, dpi=150)
        plt.close(fig2)

        click_marker.set_data([x_full], [y_full])
        fig.canvas.draw_idle()

        print(f"[INFO] 픽셀 ({x_full}, {y_full}) → disparity 확률 그래프 저장: {out_path}")

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()


# -----------------------------
# 4) 실행 진입점 (stereo)
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    num=137
    parser.add_argument("--left",  type=str, default="/home/jaejun/dataset/MS2/sync_data/tester/rgb/img_left/000137.png", help="왼쪽 이미지 경로")
    parser.add_argument("--right", type=str, default="/home/jaejun/dataset/MS2/sync_data/tester/rgb/img_right/000137.png", help="오른쪽 이미지 경로")
    parser.add_argument("--max_disp", type=int, default=56, help="최대 disparity (원본 px 단위)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pad_right_bottom_to_8", action="store_true",
                        help="입력 H,W가 8의 배수가 아니면 오른쪽/아래로 8의 배수까지 패딩(좌/우 둘 다).")
    parser.add_argument("--out_dir", type=str, default=f"./log/disp_plots{num}",
                        help="disparity 확률 그래프 저장 폴더")
    args = parser.parse_args()

    left_path  = Path(args.left)
    right_path = Path(args.right)
    assert left_path.exists(),  f"Left image not found: {left_path}"
    assert right_path.exists(), f"Right image not found: {right_path}"

    left_img_pil  = Image.open(str(left_path)).convert("RGB")
    right_img_pil = Image.open(str(right_path)).convert("RGB")

    W_l, H_l = left_img_pil.size
    W_r, H_r = right_img_pil.size

    # 필요 시 오른쪽/아래 패딩 (좌/우 둘 다 동일하게 8의 배수로)
    if args.pad_right_bottom_to_8:
        pad_r_l = (-W_l) % 8
        pad_b_l = (-H_l) % 8
        if pad_r_l or pad_b_l:
            new_img = Image.new("RGB", (W_l + pad_r_l, H_l + pad_b_l))
            new_img.paste(left_img_pil, (0, 0))
            left_img_pil = new_img
            W_l, H_l = left_img_pil.size

        pad_r_r = (-W_r) % 8
        pad_b_r = (-H_r) % 8
        if pad_r_r or pad_b_r:
            new_img = Image.new("RGB", (W_r + pad_r_r, H_r + pad_b_r))
            new_img.paste(right_img_pil, (0, 0))
            right_img_pil = new_img
            W_r, H_r = right_img_pil.size

    assert (W_l, H_l) == (W_r, H_r), "좌/우 이미지 크기가 서로 다릅니다."
    assert H_l % 8 == 0 and W_l % 8 == 0, \
        "입력 해상도가 8의 배수가 아닙니다. --pad_right_bottom_to_8 옵션을 써서 맞춰주세요."

    # 전처리
    x_left  = preprocess_pil(left_img_pil)   # [1,3,H,W]
    x_right = preprocess_pil(right_img_pil)  # [1,3,H,W]

    device = torch.device(args.device)
    model = load_dino(device)

    x_left  = x_left.to(device)
    x_right = x_right.to(device)

    # 1/8 해상도 feature 추출 (ViT-B/8)
    with torch.no_grad():
        feats_l = build_eighth_res_features(model, x_left)   # [Hf,Wf,C]
        feats_r = build_eighth_res_features(model, x_right)  # [Hf,Wf,C]

        # stereo cost volume 생성: [1,D,Hf,Wf]
        cost_volume = build_cost_volume_from_features(
            feats_l, feats_r, max_disp=args.max_disp, downsample=8
        )
        cost_volume = cost_volume.cpu()

    # 클릭 → disparity 확률 그래프 저장
    interactive_disp_view(
        left_img_pil,
        cost_volume,
        downsample=8,
        save_dir=Path(args.out_dir)
    )


if __name__ == "__main__":
    main()
