# upsampler.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 공용 유틸
# -----------------------------
class DepthwiseSeparable2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.bn(x)

def convex_upsample_2d_scalar(disp_lo: torch.Tensor, mask: torch.Tensor, scale: int) -> torch.Tensor:
    """
    disp_lo: [B,1,H',W']   scalar field
    mask:    [B, 9*(scale*scale), H', W']
    return:  [B,1,H'*scale, W'*scale]
    """
    B, C, H, W = disp_lo.shape
    assert C == 1
    mask = mask.view(B, 1, 9, scale, scale, H, W)
    mask = torch.softmax(mask, dim=2)  # convex weights
    unf = F.unfold(disp_lo, kernel_size=3, padding=1)                    # [B,9,H*W]
    unf = unf.view(B, 1, 9, H, W).unsqueeze(3).unsqueeze(4)              # [B,1,9,1,1,H,W]
    up = torch.sum(mask * unf, dim=2)                                    # [B,1,scale,scale,H,W]
    up = up.permute(0,1,4,2,5,3).contiguous().view(B, 1, H*scale, W*scale)
    return up

# -----------------------------
# 최종 2× 업샘플러 (½ → 1)
#   inputs:
#     disp_half   : [B,1,H/2,W/2]   (disp 단위는 그대로 유지; patch 또는 px)
#     img_full    : [B,3,H,W]
#     vol4d_half  : [B,Cv,D,H/2,W/2]  (logits/cost)
#     dino_half   : [B,Cd,H/2,W/2] or None
#   returns:
#     disp_full   : [B,1,H,W]
#     aux         : dict(upmask2, confidence_full, delta_full, edge_half, edge_full)
# -----------------------------
class FinalUpsample2x(nn.Module):
    def __init__(self,
                 dino_ch: int = 0,
                 guide_ch: int = 64,
                 fuse_ch: int = 96,
                 refine_ch: int = 64,
                 softmax_t: float = 0.9,
                 res_limit: float = 1.5,
                 use_edge_head: bool = False):
        super().__init__()
        self.softmax_t = softmax_t
        self.res_limit = res_limit
        self.use_edge_head = use_edge_head

        # H/2 이미지 피처 인코더
        self.img_half_enc = nn.Sequential(
            nn.Conv2d(3, guide_ch//2, 3, padding=1), nn.GELU(),
            nn.Conv2d(guide_ch//2, guide_ch, 3, padding=1), nn.GELU()
        )

        # (옵션) 학습형 엣지 헤드
        if use_edge_head:
            self.edge_head_half = nn.Sequential(
                DepthwiseSeparable2d(guide_ch + (dino_ch if dino_ch>0 else 0), guide_ch),
                nn.GELU(),
                nn.Conv2d(guide_ch, 1, 3, padding=1)
            )

        # upmask 입력 채널: img_half_feat + disp + (disp_dx,disp_dy) + (conf,std) + edge_half + (옵션)dino
        upmask_in_ch = guide_ch + 1 + 2 + 2 + 1 + (dino_ch if dino_ch>0 else 0)
        self.upmask_fuse = nn.Sequential(
            nn.Conv2d(upmask_in_ch, fuse_ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(fuse_ch, fuse_ch, 3, padding=1), nn.GELU()
        )
        self.upmask_head2 = nn.Conv2d(fuse_ch, 9*(2**2), kernel_size=1)  # 36ch

        # 1× 리파이너 입력: disp_full0 + edge_full + conf_full + img_full_feat
        self.img_full_enc = nn.Sequential(
            nn.Conv2d(3, refine_ch//2, 3, padding=1), nn.GELU(),
            nn.Conv2d(refine_ch//2, refine_ch, 3, padding=1), nn.GELU()
        )
        refine_in_ch = 1 + 1 + 1 + refine_ch
        self.refine = nn.Sequential(
            DepthwiseSeparable2d(refine_in_ch, refine_ch), nn.GELU(),
            DepthwiseSeparable2d(refine_ch, refine_ch), nn.GELU(),
            nn.Conv2d(refine_ch, 1, 3, padding=1)
        )

    @staticmethod
    def _finite_diff_xy(x: torch.Tensor):
        """전방 차분: x->[B,*,H,W] → (dx,dy)"""
        dx = x[..., :, 1:] - x[..., :, :-1]; dx = F.pad(dx, (0,1,0,0))
        dy = x[..., 1:, :] - x[..., :-1, :]; dy = F.pad(dy, (0,0,0,1))
        return dx, dy

    @staticmethod
    def _feat_grad_mag(f: torch.Tensor):
        """특징공간 기울기 크기 1채널화"""
        dx, dy = FinalUpsample2x._finite_diff_xy(f)
        return torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-6).mean(dim=1, keepdim=True)

    def _volume_stats(self, vol: torch.Tensor):
        """
        vol: [B,Cv,D,Hs,Ws] (logits/cost)
        return: conf, std  [B,1,Hs,Ws]
        """
        B, Cv, Dp, Hs, Ws = vol.shape
        logits = vol.mean(dim=1, keepdim=True)                     # [B,1,D,Hs,Ws]
        prob = torch.softmax(logits / self.softmax_t, dim=2)
        conf, _ = prob.max(dim=2)                                  # [B,1,Hs,Ws]
        disp_vals = torch.arange(Dp, dtype=vol.dtype, device=vol.device).view(1,1,Dp,1,1)
        mean = (prob * disp_vals).sum(dim=2)
        var  = (prob * (disp_vals - mean.unsqueeze(2))**2).sum(dim=2)
        std  = torch.sqrt(var + 1e-6)
        return conf, std

    def forward(self, disp_half, img_full, vol4d_half, dino_half=None):
        B, _, H, W = img_full.shape
        Hh, Wh = disp_half.shape[-2:]
        assert (H == Hh*2) and (W == Wh*2), "img_full 해상도는 disp_half의 정확히 2배여야 합니다."

        # ----- H/2 가이드 피처 & 힌트 -----
        img_half = F.interpolate(img_full, size=(Hh, Wh), mode='bilinear', align_corners=False)
        img_half_feat = self.img_half_enc(img_half)
        disp_dx, disp_dy = self._finite_diff_xy(disp_half)
        conf_half, std_half = self._volume_stats(vol4d_half)
        edge_half = self._feat_grad_mag(dino_half if dino_half is not None else img_half_feat)
        if getattr(self, "use_edge_head", False):
            base = img_half_feat if dino_half is None else torch.cat([img_half_feat, dino_half], dim=1)
            edge_half_pred = torch.sigmoid(self.edge_head_half(base))
            edge_half = 0.5 * (edge_half + edge_half_pred)

        upmask_in = torch.cat([img_half_feat, disp_half, disp_dx, disp_dy, conf_half, std_half, edge_half] +
                              ([dino_half] if dino_half is not None else []), dim=1)

        # ----- ½→1: Convex upsample -----
        z = self.upmask_fuse(upmask_in)
        upmask2 = self.upmask_head2(z)                                   # [B,36,H/2,W/2]
        disp_full0 = convex_upsample_2d_scalar(disp_half, upmask2, scale=2)  # [B,1,H,W]

        # ----- 1×: Residual refine -----
        img_full_feat = self.img_full_enc(img_full)
        edge_full = self._feat_grad_mag(img_full_feat)
        conf_full = F.interpolate(conf_half, size=(H, W), mode='bilinear', align_corners=False)

        ref_in = torch.cat([disp_full0, edge_full, conf_full, img_full_feat], dim=1)
        delta_full = self.refine(ref_in)
        delta_full = self.res_limit * torch.tanh(delta_full)             # 안정화

        disp_full = disp_full0 + delta_full
        aux = {
            "upmask2": upmask2,
            "confidence_full": conf_full,
            "delta_full": delta_full,
            "edge_half": edge_half,
            "edge_full": edge_full
        }
        return disp_full, aux
