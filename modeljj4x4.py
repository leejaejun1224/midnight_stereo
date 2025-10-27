import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Utils
# ---------------------------

def assert_multiple(x, m, name="size"):
    if x % m != 0:
        raise ValueError(f"{name}={x} 는 {m}의 배수여야 합니다.")

def build_corr_volume_with_mask(FL: torch.Tensor, FR: torch.Tensor, D: int):
    """
    dot-product correlation cost volume.
    FL, FR: [B,C,H,W]  (여기서는 H=W=1/4 해상도)
    return:
      vol:  [B,1,D+1,H,W]
      mask: [B,1,D+1,H,W] (시프트 유효영역)
    """
    B, C, H, W = FL.shape
    vols, masks = [], []
    for d in range(D + 1):
        if d == 0:
            FR_shift = FR
            valid = torch.ones((B, 1, H, W), device=FL.device, dtype=FL.dtype)
        else:
            # 오른쪽 이미지를 왼쪽으로 d 셀 만큼 시프트(좌측 d 픽셀은 invalid)
            FR_shift = F.pad(FR, (d, 0, 0, 0))[:, :, :, :W]
            valid = torch.ones((B, 1, H, W), device=FL.device, dtype=FL.dtype)
            valid[:, :, :, :d] = 0.0
        corr = (FL * FR_shift).sum(dim=1, keepdim=True)  # [B,1,H,W]
        vols.append(corr.squeeze(1))
        masks.append(valid.squeeze(1))
    vol  = torch.stack(vols,  dim=1).unsqueeze(1)   # [B,1,D+1,H,W]
    mask = torch.stack(masks, dim=1).unsqueeze(1)   # [B,1,D+1,H,W]
    return vol, mask


# =========================================================
# 1/4 특징 추출기 (Frozen) — DINO(1/8→1/4) + MobileNetV2(1/4)
# =========================================================

class MobileNetV2_S4(nn.Module):
    """
    stride 4 CNN 특징 (stage 0~3), 채널 24, 위치별 채널 L2 정규화.
    입력은 0..1 또는 uint8; 내부에서 ImageNet 정규화.
    """
    def __init__(self, pretrained=True, out_norm=True, center=True):
        super().__init__()
        from torchvision import models
        try:
            from torchvision.models import MobileNet_V2_Weights
            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            m = models.mobilenet_v2(weights=weights)
            mean = (0.485, 0.456, 0.406) if weights is None else weights.meta["mean"]
            std  = (0.229, 0.224, 0.225) if weights is None else weights.meta["std"]
        except Exception:
            m = models.mobilenet_v2(pretrained=pretrained)
            mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)

        self.encoder = nn.Sequential(*list(m.features[:4]))  # 0,1,2,3  => stride=4
        self.register_buffer("mean", torch.tensor(mean).view(1,3,1,1), persistent=False)
        self.register_buffer("std",  torch.tensor(std).view(1,3,1,1),  persistent=False)
        self.out_norm = out_norm
        self.center = center
        self.out_channels = 24

        for p in self.parameters(): p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W], float(0..1) or uint8
        if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            x = x.float().div_(255.0)
        B, C, H, W = x.shape
        # 오른쪽/아래 reflect pad로 4의 배수 보장
        need_h = (4 - (H % 4)) % 4; need_w = (4 - (W % 4)) % 4
        if need_h or need_w:
            x = F.pad(x, (0, need_w, 0, need_h), mode="reflect")
        # ImageNet 정규화
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        y = self.encoder(x)                      # [B,24,ceil(H/4),ceil(W/4)]
        y = y[:, :, :H//4, :W//4].contiguous()  # [B,24,H/4,W/4]
        if self.center:
            mu  = y.mean(dim=(2,3), keepdim=True)
            std = y.std(dim=(2,3), keepdim=True) + 1e-6
            y = (y - mu) / std
        if self.out_norm:
            y = F.normalize(y, dim=1, eps=1e-6)
        return y


@torch.no_grad()
def _dino_tokens_to_grid(backbone, x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    facebookresearch/dino hub 모델의 get_intermediate_layers → [B,C,H/patch,W/patch]
    (CLS 제거 후 그리드로 reshape)
    """
    tokens = backbone.get_intermediate_layers(x, n=1)[0]  # [B,1+P,C]
    patch_tokens = tokens[:, 1:, :]                       # (CLS 제거)
    B, P, C = patch_tokens.shape
    H, W = x.shape[-2], x.shape[-1]
    h, w = H // patch_size, W // patch_size
    assert h*w == P, f"Token {P} != h*w {h*w}. 입력 H,W는 patch_size({patch_size})의 배수여야 함."
    grid = patch_tokens.transpose(1, 2).contiguous().view(B, C, h, w)  # [B,C,H/8,W/8]
    return grid


class FusedQuarterFeatures(nn.Module):
    """
    1/4 해상도 특징 추출기 (Frozen)
      - DINO ViT-B/8: [B,Cd,H/8,W/8] → NN 2× → [B,Cd,H/4,W/4] → L2 norm
      - MobileNetV2_S4: [B,24,H/4,W/4] → (옵션 중심화) → L2 norm
      - (옵션) 특징 레벨 결합: concat | sum(가중합, DINO 채널을 24로 그룹평균 축소)
    입력은 0..1 또는 uint8.
    """
    def __init__(self, patch_size=8, fuse_feat_mode=None, sum_alpha=0.5, cnn_center=True):
        super().__init__()
        self.patch = patch_size
        # DINO hub
        self.dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
        for p in self.dino.parameters(): p.requires_grad = False
        self.dino.eval()
        # CNN
        self.cnn = MobileNetV2_S4(pretrained=True, out_norm=True, center=cnn_center)
        # ImageNet 정규화 (DINO 입력용)
        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1), persistent=False)
        self.register_buffer("std",  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1),  persistent=False)

        # channel dims
        self.dino_dim = getattr(self.dino, "embed_dim", 768)
        self.cnn_dim  = self.cnn.out_channels
        self.fuse_feat_mode = fuse_feat_mode  # None, 'concat', 'sum'
        self.sum_alpha = float(sum_alpha)

        self.eval()

    def channels(self, which="dino"):
        if which == "dino": return self.dino_dim
        if which == "cnn":  return self.cnn_dim
        if which == "fused":
            if self.fuse_feat_mode == "concat": return self.dino_dim + self.cnn_dim
            if self.fuse_feat_mode == "sum":    return self.cnn_dim
            raise ValueError("fused channels requested but fuse_feat_mode is None")
        raise ValueError(f"unknown which={which}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        x: [B,3,H,W] (float 0..1 또는 uint8)
        return dict:
          'dino':  [B,Cd,H/4,W/4]
          'cnn':   [B,24,H/4,W/4]
          'fused': [B,Cf,H/4,W/4] or None
        """
        B, _, H, W = x.shape
        assert_multiple(H, self.patch, "height")
        assert_multiple(W, self.patch, "width")

        # DINO 입력 정규화
        x_dino = x
        if x_dino.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            x_dino = x_dino.float().div_(255.0)
        x_dino = (x_dino - self.mean.to(x.device)) / self.std.to(x.device)

        # DINO 1/8 → 1/4
        Fd8 = _dino_tokens_to_grid(self.dino, x_dino, self.patch)      # [B,Cd,H/8,W/8]
        Fd4 = F.interpolate(Fd8, scale_factor=2, mode='bilinear', align_corners=True)       # [B,Cd,H/4,W/4]
        Fd4 = F.normalize(Fd4, dim=1, eps=1e-6)

        # CNN 1/4
        Fc4 = self.cnn(x)                                              # [B,24,H/4,W/4]

        # (옵션) 특징 결합
        Ff4 = None
        if self.fuse_feat_mode == "concat":
            Ff4 = torch.cat([Fd4, Fc4], dim=1)                         # [B,Cd+24,H/4,W/4]
            Ff4 = F.normalize(Ff4, dim=1, eps=1e-6)
        elif self.fuse_feat_mode == "sum":
            # DINO 채널 → 24로 그룹평균 축소 후 가중합
            g = Fd4.shape[1] // Fc4.shape[1]
            assert Fd4.shape[1] % Fc4.shape[1] == 0, "DINO 채널 수는 24의 배수여야 sum 가능"
            Fd_red = Fd4.view(B, Fc4.shape[1], g, Fd4.shape[2], Fd4.shape[3]).mean(dim=2)
            Fd_red = F.normalize(Fd_red, dim=1, eps=1e-6)
            Ff4 = self.sum_alpha * Fd_red + (1.0 - self.sum_alpha) * Fc4
            Ff4 = F.normalize(Ff4, dim=1, eps=1e-6)

        return {"dino": Fd4, "cnn": Fc4, "fused": Ff4}


# ---------------------------
# 3D U-Net like Aggregation (H/W만 다운)
# ---------------------------

class Conv3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm='bn', groups=8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        if norm == 'gn':
            self.bn1 = nn.GroupNorm(groups, out_ch)
            self.bn2 = nn.GroupNorm(groups, out_ch)
        else:
            self.bn1 = nn.BatchNorm3d(out_ch)
            self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class Down3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm='bn', groups=8):
        super().__init__()
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=(1,2,2), padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.GroupNorm(groups, out_ch) if norm=='gn' else nn.BatchNorm3d(out_ch)
    def forward(self, x):
        return self.relu(self.bn(self.down(x)))

class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm='bn', groups=8):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=3, stride=(1,2,2), padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.GroupNorm(groups, out_ch) if norm=='gn' else nn.BatchNorm3d(out_ch)
    def forward(self, x, out_hw=None):
        if out_hw is not None:
            B, _, D, _, _ = x.shape
            Ht, Wt = out_hw
            x = self.up(x, output_size=(B, self.up.out_channels, D, Ht, Wt))
        else:
            x = self.up(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CostAggregator3D(nn.Module):
    def __init__(self, base_ch: int = 32, depth: int = 3, norm='bn'):
        super().__init__()
        ch = base_ch
        # Encoder
        self.enc1 = Conv3DBlock(1, ch, norm)
        self.down1 = Down3D(ch, ch*2, norm)
        self.enc2 = Conv3DBlock(ch*2, ch*2, norm)
        self.down2 = Down3D(ch*2, ch*4, norm)
        self.enc3 = Conv3DBlock(ch*4, ch*4, norm)
        self.down3 = Down3D(ch*4, ch*8, norm) if depth >= 3 else nn.Identity()
        # Bottleneck
        if depth >= 3:
            self.bott = Conv3DBlock(ch*8, ch*8, norm)
            self.up3  = Up3D(ch*8, ch*4, norm)
        # Decoder
        self.dec3 = Conv3DBlock(ch*8 if depth >= 3 else ch*4, ch*4, norm)
        self.up2  = Up3D(ch*4, ch*2, norm)
        self.dec2 = Conv3DBlock(ch*4, ch*2, norm)
        self.up1  = Up3D(ch*2, ch, norm)
        self.dec1 = Conv3DBlock(ch*2, ch, norm)  # ※ concat 후 2ch → ch
        self.out  = nn.Conv3d(ch, 1, kernel_size=1)
        self.depth = depth

    def forward(self, x):
        e1 = self.enc1(x); d1 = self.down1(e1)
        e2 = self.enc2(d1); d2 = self.down2(e2)
        e3 = self.enc3(d2)
        if self.depth >= 3:
            d3 = self.down3(e3)
            b  = self.bott(d3)
            u3 = self.up3(b, out_hw=(e3.shape[-2], e3.shape[-1]))
            c3 = torch.cat([u3, e3], dim=1)
        else:
            c3 = e3
        dec3 = self.dec3(c3)
        u2   = self.up2(dec3, out_hw=(e2.shape[-2], e2.shape[-1]))
        c2   = torch.cat([u2, e2], dim=1)
        dec2 = self.dec2(c2)
        u1   = self.up1(dec2, out_hw=(e1.shape[-2], e1.shape[-1]))
        c1   = torch.cat([u1, e1], dim=1)
        dec1 = self.dec1(c1)
        out  = self.out(dec1)   # [B,1,D+1,H',W']
        return out


# ---------------------------
# Soft + ArgMax
# ---------------------------

class SoftAndArgMax(nn.Module):
    def __init__(self, D: int, temperature: float = 0.7):
        super().__init__()
        self.register_buffer("disp_values", torch.arange(0, D+1, dtype=torch.float32).view(1,1,D+1,1,1))
        self.t = temperature
    def forward(self, vol_masked: torch.Tensor):
        prob = torch.softmax(vol_masked / self.t, dim=2)              # [B,1,D+1,H',W']
        disp_soft = (prob * self.disp_values).sum(dim=2)              # [B,1,H',W']
        disp_wta  = vol_masked.argmax(dim=2, keepdim=False).float()   # [B,1,H',W']
        return prob, disp_soft, disp_wta


# ---------------------------
# ACVNet-style context upsample (scale 인자화)
# ---------------------------

def context_upsample_scaled(depth_low, up_weights, scale: int):
    """
    depth_low:  [B,1,h,w]          (여기서는 h=H/4)
    up_weights:[B,9,scale*h,scale*w]  (여기서는 [B,9,H,W], scale=4)
    return:     [B, scale*h, scale*w] (여기서는 [B,H,W])
    """
    b, c, h, w = depth_low.shape
    assert c == 1
    depth_unfold = F.unfold(depth_low.reshape(b,c,h,w), 3, 1, 1).reshape(b, 9, h, w)
    depth_unfold = F.interpolate(depth_unfold, (h*scale, w*scale), mode='nearest').reshape(b, 9, h*scale, w*scale)
    depth = (depth_unfold * up_weights).sum(1)
    return depth


# ---------------------------
# 2D 유틸 블록 (stem / spx에 사용)
# ---------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SPX4xHead(nn.Module):
    """
    H/4 입력(feature)을 ConvTranspose2d(×4)로 한 번에 H로 올려서,
    full-res stem skip과 concat 후 9채널 up-weights를 예측.
    """
    def __init__(self, in_ch, skip_ch, mid_ch=64):
        super().__init__()
        # ×4 업샘플: kernel=8, stride=4, padding=2  → 정수 매핑
        self.up4  = nn.ConvTranspose2d(in_ch, mid_ch, kernel_size=8, stride=4, padding=2)
        self.fuse = ConvBNReLU(mid_ch + skip_ch, mid_ch, k=3, s=1, p=1)
        self.head = nn.Conv2d(mid_ch, 9, kernel_size=3, stride=1, padding=1)
    def forward(self, x_q4, skip_full):
        x = self.up4(x_q4)                     # [B,mid,H,W]
        x = torch.cat([x, skip_full], dim=1)   # + full-res stem
        x = self.fuse(x)
        w = self.head(x)                       # [B,9,H,W]
        return F.softmax(w, dim=1)


# ---------------------------
# Stereo Model (1/4 cost + direct 4× upsample to full)
# ---------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ... (위쪽 유틸/블록/함수들은 기존 그대로) ...


class StereoModel(nn.Module):
    def __init__(self,
                 max_disp_px: int = 64,
                 patch_size: int = 8,
                 agg_base_ch: int = 32,
                 agg_depth: int = 3,
                 softarg_t: float = 0.9,
                 norm: str = 'gn',
                 # --- 기존 옵션 ---
                 sim_fusion_mode: str = "late_weighted",    # 'late_weighted' | 'dino_only' | 'concat' | 'sum' | 'learned_fused'
                 dino_weight: float = 0.90,                 # late_weighted 가중치
                 fuse_feat_mode: str = None,                # FusedQuarterFeatures 내부용 (그대로)
                 sum_alpha: float = 0.5,
                 cnn_center: bool = True,
                 spx_source: str = "dino",
                 # --- [NEW] cost volume용 학습 결합기 설정 ---
                 cv_fuse_out_ch: int = 96,                  # learned_fused 출력 채널 수
                 cv_fuse_arch: str = "conv1x1"              # "conv1x1" | "mlp"
                 ):
        super().__init__()
        self.patch = patch_size
        self.grid_stride = self.patch // 2  # 1/4 그리드 스텝(px) = 4
        if self.patch % 2 != 0:
            raise ValueError(f"patch_size={self.patch} 는 짝수여야 합니다. (1/4 그리드 스텝용)")
        assert_multiple(max_disp_px, self.grid_stride, "max_disp_px")
        self.D = max_disp_px // self.grid_stride

        # 1) Feature (Frozen 1/4)
        self.feat_net = FusedQuarterFeatures(
            patch_size=self.patch,
            fuse_feat_mode=fuse_feat_mode,
            sum_alpha=sum_alpha,
            cnn_center=cnn_center
        )
        for p in self.feat_net.parameters(): p.requires_grad = False
        self.feat_net.eval()

        self.sim_fusion_mode = sim_fusion_mode
        self.dino_weight = float(dino_weight)
        self.spx_source = spx_source

        # 2) 3D aggregation + soft-argmax (1/4 해상도)
        self.agg  = CostAggregator3D(base_ch=agg_base_ch, depth=agg_depth, norm=norm)
        self.post = SoftAndArgMax(D=self.D, temperature=softarg_t)

        # 3) stem 피처 (full-res, quarter-res)
        self.stem_1 = nn.Sequential(
            ConvBNReLU(3, 32, k=3, s=1, p=1),
            ConvBNReLU(32, 32, k=3, s=1, p=1),
        )
        self.stem_4 = nn.Sequential(
            ConvBNReLU(3, 32, k=3, s=2, p=1),        # H → H/2
            ConvBNReLU(32, 48, k=3, s=2, p=1),       # H/2 → H/4
            ConvBNReLU(48, 48, k=3, s=1, p=1),
        )

        # --- [NEW] cost volume용 학습 결합기 (concat → 1×1 conv or 1×1 MLP) ---
        self.cv_fuse = None
        self.cv_out_norm = False
        sum_ch = self.feat_net.channels("dino") + self.feat_net.channels("cnn")
        self.cv_fuse_out_ch = int(cv_fuse_out_ch)
        self.cv_fuse_arch = str(cv_fuse_arch).lower()

        if self.sim_fusion_mode in ("learned_fused",):  # 새 모드에서만 사용
            if self.cv_fuse_arch == "mlp":
                mid_ch = max(self.cv_fuse_out_ch, sum_ch // 2)
                self.cv_fuse = nn.Sequential(
                    nn.Conv2d(sum_ch, mid_ch, kernel_size=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_ch, self.cv_fuse_out_ch, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=8, num_channels=self.cv_fuse_out_ch),
                    nn.ReLU(inplace=True)
                )
            else:  # "conv1x1"
                self.cv_fuse = nn.Sequential(
                    nn.Conv2d(sum_ch, self.cv_fuse_out_ch, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=8, num_channels=self.cv_fuse_out_ch),
                    nn.ReLU(inplace=True)
                )
            self.cv_out_norm = True  # dot corr 전에 L2 정규화

        # spx 입력 채널 계산 (그대로)
        if self.spx_source == "dino":
            feat_ch_spx = self.feat_net.channels("dino")
        elif self.spx_source == "cnn":
            feat_ch_spx = self.feat_net.channels("cnn")
        elif self.spx_source == "fused":
            feat_ch_spx = self.feat_net.channels("fused")
        else:
            raise ValueError("spx_source must be 'dino'|'cnn'|'fused'")

        self.spx_4 = nn.Sequential(ConvBNReLU(feat_ch_spx + 48, 64, k=3, s=1, p=1))
        self.spx_full = SPX4xHead(in_ch=64, skip_ch=32, mid_ch=64)

    @torch.no_grad()
    def extract_feats(self, left: torch.Tensor, right: torch.Tensor):
        FL = self.feat_net(left)
        FR = self.feat_net(right)
        return FL, FR

    def _build_dir_feat(self, FL_dict, FR_dict):
        """
        [NEW] directional loss 용 feature 생성
        정의: f_dir = [ sqrt(a)*u  ||  sqrt(1-a)*w ],  (u,w는 채널 L2 정규화 완료)
        내적이 a*cos_dino + (1-a)*cos_cnn 와 정확히 일치.
        """
        a = float(self.dino_weight)
        sa = math.sqrt(max(a, 0.0))
        sb = math.sqrt(max(1.0 - a, 0.0))

        uL, uR = FL_dict["dino"], FR_dict["dino"]   # [B,Cd,H/4,W/4], L2 normalized
        wL, wR = FL_dict["cnn"],  FR_dict["cnn"]    # [B,24,H/4,W/4],  L2 normalized

        fdirL = torch.cat([uL * sa, wL * sb], dim=1)
        fdirR = torch.cat([uR * sa, wR * sb], dim=1)
        # 이론상 이미 ‖fdir‖=1 이지만, 수치 안정 위해 얕게 normalize
        fdirL = F.normalize(fdirL, dim=1, eps=1e-6)
        fdirR = F.normalize(fdirR, dim=1, eps=1e-6)
        return fdirL, fdirR

    def _build_cost_volume_inputs(self, FL_dict, FR_dict):
        """
        [NEW] cost volume 입력 feature 쌍 생성
        - learned_fused: concat(DINO,CNN) → 1×1 conv/MLP → L2 norm
        - 그 외 모드는 기존 로직(아래 forward에서 처리)
        """
        assert self.cv_fuse is not None, "learned_fused 모드에서만 호출됩니다."
        FdL, FdR = FL_dict["dino"], FR_dict["dino"]
        FcL, FcR = FL_dict["cnn"],  FR_dict["cnn"]

        inL = torch.cat([FdL, FcL], dim=1)
        inR = torch.cat([FdR, FcR], dim=1)
        outL = self.cv_fuse(inL)
        outR = self.cv_fuse(inR)
        if self.cv_out_norm:
            outL = F.normalize(outL, dim=1, eps=1e-6)
            outR = F.normalize(outR, dim=1, eps=1e-6)
        return outL, outR

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        B, _, H, W = left.shape
        assert_multiple(H, self.patch, "height")
        assert_multiple(W, self.patch, "width")

        # ---- 1) 특징 (1/4) ----
        FL_dict, FR_dict = self.extract_feats(left, right)

        # ---- [NEW] directional loss용 feature (가중합 cos-sim 재현) ----
        feat_dir_L, feat_dir_R = self._build_dir_feat(FL_dict, FR_dict)

        # ---- 2) cost volume (모드별) ----
        mode = self.sim_fusion_mode

        if mode == "dino_only":
            FL, FR = FL_dict["dino"], FR_dict["dino"]
            vol, mask = build_corr_volume_with_mask(FL, FR, self.D)

        elif mode == "late_weighted":
            FLd, FRd = FL_dict["dino"], FR_dict["dino"]
            FLc, FRc = FL_dict["cnn"],  FR_dict["cnn"]
            vol_d, mask_d = build_corr_volume_with_mask(FLd, FRd, self.D)
            vol_c, mask_c = build_corr_volume_with_mask(FLc, FRc, self.D)
            a = self.dino_weight
            vol  = a * vol_d + (1.0 - a) * vol_c
            mask = (mask_d * mask_c)  # 동일 마스크

        elif mode in ("concat", "sum"):
            # 기존 FusedQuarterFeatures 경로 그대로 사용
            FfL, FfR = FL_dict["fused"], FR_dict["fused"]
            assert FfL is not None, "fuse_feat_mode가 필요합니다."
            vol, mask = build_corr_volume_with_mask(FfL, FfR, self.D)

        elif mode == "learned_fused":
            # [NEW] concat → 1×1 conv(or 1×1 MLP) → L2 norm → cost volume
            FL_fused, FR_fused = self._build_cost_volume_inputs(FL_dict, FR_dict)
            vol, mask = build_corr_volume_with_mask(FL_fused, FR_fused, self.D)

        else:
            raise ValueError("sim_fusion_mode must be 'dino_only'|'late_weighted'|'concat'|'sum'|'learned_fused'")

        vol_in = vol * mask

        # ---- 3) 3D aggregation @ 1/4 ----
        refined = self.agg(vol_in)
        refined_masked = refined + (1.0 - mask) * (-1e4)

        # ---- 4) soft-arg + WTA ----
        prob, disp_soft, disp_wta = self.post(refined_masked)
        raw_for_anchor = (vol + (1.0 - mask) * (-1e4)).detach()

        # ---- 5) full-res up-weights 예측 ----
        stem1 = self.stem_1(left)          # [B,32,H,W]
        stem4 = self.stem_4(left)          # [B,48,H/4,W/4]

        if self.spx_source == "dino":
            spx_feat_L = FL_dict["dino"]
        elif self.spx_source == "cnn":
            spx_feat_L = FL_dict["cnn"]
        else:  # 'fused'
            spx_feat_L = FL_dict["fused"]
            if spx_feat_L is None:
                raise ValueError("spx_source='fused'지만 fused features가 없음 (fuse_feat_mode 설정 확인)")

        spx4_in = torch.cat([spx_feat_L, stem4], dim=1)
        xspx4   = self.spx_4(spx4_in)                    # [B,64,H/4,W/4]
        up_w_full = self.spx_full(xspx4, stem1)          # [B,9,H,W]

        # ---- 6) ACVNet-style context_upsample ----
        disp_full = context_upsample_scaled(disp_soft, up_w_full, scale=4).unsqueeze(1)
        disp_full_px = disp_full * float(self.grid_stride)

        return prob, disp_soft, {
            "FL": FL_dict, "FR": FR_dict,
            "raw_volume": raw_for_anchor,
            "mask": mask,
            "refined_volume": refined,
            "refined_masked": refined_masked,
            "disp_wta": disp_wta,
            "disp_full": disp_full,
            "disp_full_px": disp_full_px,
            "up_w_full": up_w_full,
            # [NEW] directional loss용 feature (좌/우 둘 다 제공; 보통 좌만 사용)
            "feat_dir_L": feat_dir_L,
            "feat_dir_R": feat_dir_R,
            # [NEW] learned_fused 모드 사용 시 디버깅용
            "cv_mode": mode,
        }
