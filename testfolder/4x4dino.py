import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

# --------------------- 유틸 ---------------------

def assert_multiple(n: int, m: int, name: str = "size"):
    assert n % m == 0, f"{name} ({n}) must be a multiple of {m}"

def shift_reflect(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """
    정수 픽셀 평행이동 (경계 reflect). 크기는 그대로 유지.
    +dx: 내용이 오른쪽으로 dx만큼 이동(좌측에 패드), +dy: 내용이 아래로 dy만큼 이동(상단에 패드)
    """
    B, C, H, W = x.shape
    left   = dx if dx > 0 else 0
    right  = -dx if dx < 0 else 0
    top    = dy if dy > 0 else 0
    bottom = -dy if dy < 0 else 0
    y0 = 0 if dy >= 0 else bottom
    x0 = 0 if dx >= 0 else right
    y1 = y0 + H
    x1 = x0 + W
    x_pad = F.pad(x, (left, right, top, bottom), mode="reflect")
    return x_pad[:, :, y0:y1, x0:x1]


# --------------------- DINO ViT-B/8: 패치 특징 추출 ---------------------

class DINOvits8Features(nn.Module):
    """
    입력 x: [B, 3, H, W]  (H,W는 patch_size의 배수; 정규화는 사용자가 책임짐)
    출력:   [B, C, H/8, W/8] (C=768 for ViT-Base)
    """
    def __init__(self, patch_size: int = 8):
        super().__init__()
        self.patch = patch_size
        # facebookresearch/dino torch.hub: "dino_vitb8" (ViT-Base/8)
        self.backbone = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        assert_multiple(H, self.patch, "height")
        assert_multiple(W, self.patch, "width")

        # 마지막 레이어 토큰: [B, 1+P, C]
        tokens = self.backbone.get_intermediate_layers(x, n=1)[0]
        patch_tokens = tokens[:, 1:, :]  # CLS 제거
        C = patch_tokens.shape[-1]
        H8, W8 = H // self.patch, W // self.patch

        feat = patch_tokens.transpose(1, 2).contiguous().view(B, C, H8, W8)  # [B,C,H/8,W/8]
        feat = F.normalize(feat, dim=1, eps=1e-6)  # 선택적: 채널 L2 노멀라이즈
        return feat


# --------------------- 2× 업샘플(= stride 4) Interleave ---------------------
class DINOvits8Stride4(nn.Module):
    def __init__(self, feature_extractor: Optional[nn.Module] = None, patch_size: int = 8):
        super().__init__()
        self.patch = patch_size
        self.shift = patch_size // 2  # 4px for patch=8
        assert self.shift * 2 == self.patch, "patch_size must be even"
        self.features = feature_extractor if feature_extractor is not None else DINOvits8Features(patch_size)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, sequential: bool = True) -> torch.Tensor:
        B, _, H, W = x.shape
        assert_multiple(H, self.patch, "height")
        assert_multiple(W, self.patch, "width")

        shifts = [(0, 0), (self.shift, 0), (0, self.shift), (self.shift, self.shift)]

        if sequential:
            x0 = shift_reflect(x, *shifts[0]); F00 = self.features(x0)
            B, C, h, w = F00.shape
            O = F00.new_zeros(B, C, 2*h, 2*w)
            O[:, :, 0::2, 0::2] = F00; del x0, F00

            x1 = shift_reflect(x, *shifts[1]); F10 = self.features(x1); O[:, :, 0::2, 1::2] = F10; del x1, F10
            x2 = shift_reflect(x, *shifts[2]); F01 = self.features(x2); O[:, :, 1::2, 0::2] = F01; del x2, F01
            x3 = shift_reflect(x, *shifts[3]); F11 = self.features(x3); O[:, :, 1::2, 1::2] = F11; del x3, F11
            return O.contiguous()
        else:
            xs = [shift_reflect(x, dx, dy) for (dx, dy) in shifts]
            x_cat = torch.cat(xs, dim=0)            # [4B,3,H,W]
            F8 = self.features(x_cat)               # [4B,C,h,w]
            _, C, h, w = F8.shape
            F8 = F8.view(4, B, C, h, w)
            O = F8.new_zeros(B, C, 2*h, 2*w)
            O[:, :, 0::2, 0::2] = F8[0]
            O[:, :, 0::2, 1::2] = F8[1]
            O[:, :, 1::2, 0::2] = F8[2]
            O[:, :, 1::2, 1::2] = F8[3]
            return O.contiguous()



# --------------------- 사용 예시 ---------------------
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, H, W = 1, 384, 1224  # H,W는 8의 배수
    x = torch.randn(B, 3, H, W, device=device)

    feat_extractor = DINOvits8Features(patch_size=8).to(device)
    stride4 = DINOvits8Stride4(feature_extractor=feat_extractor, patch_size=8).to(device)

    # (1) 순차 처리 (메모리 효율)
    O_seq = stride4(x, sequential=True)
    print("sequential output:", tuple(O_seq.shape))  # [B, C(=768), H/4, W/4]

    # (2) 병렬 처리 (빠름, 메모리↑)
    # O_par = stride4(x, sequential=False)
    # print("parallel output:  ", tuple(O_par.shape))
