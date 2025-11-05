# decoder_sota.py
# -*- coding: utf-8 -*-
"""
SOTA-지향 Stereo Decoder:
- 1/4 cos-sim correlation volume  (from StereoModel['cossim_feat_1_4'])
- 1/4 concat volume (from StereoModel['fused_1_4'], channel-reduced)
- ACV attention (corr -> 3D conv -> weights -> weight * concat volume)  [ACVNet, TPAMI'24]
- (opt) MCCV-style motif gating to emphasize edge/structure correlation   [MoCha-Stereo, CVPR'24]
- 3D hourglass aggregation -> soft-argmin disparity at 1/4
- (opt) 2-stage: local-range refine around coarse disparity (IGEV++ spirit)
- upsample x4 + light 2D refine -> full-res disparity

Inputs:
  backbone_out = StereoModel.forward(...)
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# utils
# ---------------------------
def soft_argmin(prob: torch.Tensor) -> torch.Tensor:
    B, D, H, W = prob.shape
    disp_values = torch.arange(D, device=prob.device, dtype=prob.dtype).view(1, D, 1, 1)
    return torch.sum(prob * disp_values, dim=1, keepdim=True)

def entropy_map(prob: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    B, D, H, W = prob.shape
    p = prob.clamp_min(eps)
    ent = -(p * p.log()).sum(dim=1, keepdim=True) / (torch.log(torch.tensor(float(D), device=prob.device)) + eps)
    return ent.clamp(0, 1)

def _shift_right_feats(feat: torch.Tensor, d: int) -> torch.Tensor:
    # shift right image to left by d cells (i.e., pad left by d)
    if d == 0:
        return feat
    return F.pad(feat, (d, 0, 0, 0))[:, :, :, : feat.shape[-1]]

def _down2(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=2, stride=2)


# ---------------------------
# 1) Correlation volume @ 1/4 (cosine)  [channel-last -> channel-first]
# ---------------------------
class CorrVolume1_4(nn.Module):
    def __init__(self, max_disp_px: int):
        super().__init__()
        self.D = max(1, max_disp_px // 4)

    def forward(self, L_corr: torch.Tensor, R_corr: torch.Tensor) -> torch.Tensor:
        """
        L_corr, R_corr: [B, H4, W4, C], L2-normalized
        return: corr volume ~ cosine similarity → [B, D, H4, W4]
        """
        B, H4, W4, C = L_corr.shape
        L = L_corr.permute(0, 3, 1, 2).contiguous()   # [B,C,H4,W4]
        R = R_corr.permute(0, 3, 1, 2).contiguous()
        vols = []
        for d in range(self.D):
            R_shift = _shift_right_feats(R, d)
            sim = (L * R_shift).sum(dim=1, keepdim=False)  # [B,H4,W4]
            vols.append(sim)
        corr = torch.stack(vols, dim=1)  # [B,D,H4,W4]
        return corr


# ---------------------------
# 2) Concat volume @ 1/4 with channel reduction  [ACV: concat + attention]
# ---------------------------
class ConcatVolume1_4(nn.Module):
    def __init__(self, in_ch: int, red_ch: int = 48):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, red_ch, 1)  # compress fused_1_4 channels

    def forward(self, L_fused: torch.Tensor, R_fused: torch.Tensor, D: int) -> torch.Tensor:
        """
        L_fused, R_fused: [B,Cf,H4,W4]  (channel-first)
        return: concat volume [B, 2*Cr, D, H4, W4]
        """
        Lr = self.reduce(L_fused)
        Rr = self.reduce(R_fused)
        vols = []
        for d in range(D):
            R_shift = _shift_right_feats(Rr, d)
            vols.append(torch.cat([Lr, R_shift], dim=1))  # [B,2*Cr,H4,W4]
        vol = torch.stack(vols, dim=2)  # [B,2*Cr,D,H4,W4]
        return vol, Lr.shape[1]  # Cr


# ---------------------------
# 3) ACV attention from corr volume  [ACVNet, TPAMI'24]
#     corr[B, D, H, W] -> weights[B, 2*Cr, D, H, W]  (sigmoid)
# ---------------------------
class ACVAttention(nn.Module):
    def __init__(self, out_ch: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, hidden, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv3d(hidden, out_ch, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, corr: torch.Tensor) -> torch.Tensor:
        x = corr.unsqueeze(1)          # [B,1,D,H,W]
        w = self.net(x)                # [B,out_ch,D,H,W]
        return w


# ---------------------------
# 4) (opt) MCCV-style motif gating  [MoCha-Stereo, CVPR'24]
#     간단 projector로 motif 채널 생성 후, motif correlation로 채널 게이트 생성
# ---------------------------
class MotifProjector(nn.Module):
    def __init__(self, in_ch: int, m_ch: int = 24):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, m_ch, 1), nn.ReLU(inplace=True),
            nn.Conv2d(m_ch, m_ch, 3, 1, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)  # [B,m,H,W]

class MCCVGating(nn.Module):
    def __init__(self, m_ch: int, out_ch: int):
        super().__init__()
        self.map = nn.Sequential(
            nn.Conv3d(1, out_ch, 3, 1, 1),
            nn.Sigmoid()
        )
        self.m_ch = m_ch

    def forward(self, ML: torch.Tensor, MR: torch.Tensor, D: int) -> torch.Tensor:
        """
        ML, MR: motif features [B,m,H4,W4]
        return: weights [B,out_ch,D,H4,W4]
        """
        B, m, H4, W4 = ML.shape
        vols = []
        for d in range(D):
            MR_shift = _shift_right_feats(MR, d)
            # motif correlation (cos-normalization for stability)
            MLn = F.normalize(ML, dim=1)
            MRn = F.normalize(MR_shift, dim=1)
            sim = (MLn * MRn).sum(dim=1, keepdim=True)  # [B,1,H4,W4]
            vols.append(sim)
        mccv = torch.stack(vols, dim=2)  # [B,1,D,H4,W4]
        return self.map(mccv)            # [B,out_ch,D,H4,W4]


# ---------------------------
# 5) 3D aggregation (hourglass)
# ---------------------------
class Basic3D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class Hourglass3D(nn.Module):
    def __init__(self, in_ch: int, base: int = 32):
        super().__init__()
        c = base
        self.in_block = Basic3D(in_ch, c)
        self.down1 = Basic3D(c, c, s=2)
        self.down2 = Basic3D(c, c*2, s=2)
        self.mid   = Basic3D(c*2, c*2)
        self.up1   = nn.ConvTranspose3d(c*2, c, 3, 2, 1, 1)
        self.up2   = nn.ConvTranspose3d(c, c, 3, 2, 1, 1)
        self.post  = Basic3D(c, c)
        self.out   = nn.Conv3d(c, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.in_block(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        xm = self.mid(x2)
        y  = self.up1(xm) + x1
        y  = self.up2(y)  + x0
        y  = self.post(y)
        return self.out(y).squeeze(1)  # [B,D,H,W]


# ---------------------------
# 6) Full-res refine (x4 upsample + light 2D)
# ---------------------------
class FullRefine(nn.Module):
    def __init__(self, cf: int = 256, mid: int = 64):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(1 + cf, mid, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 3, 1, 1)
        )

    def forward(self, disp_q: torch.Tensor, fused_q: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        H, W = out_hw
        disp_up  = F.interpolate(disp_q, size=(H, W), mode="bilinear", align_corners=False)
        fused_up = F.interpolate(fused_q, size=(H, W), mode="bilinear", align_corners=False)
        return disp_up + self.refine(torch.cat([disp_up, fused_up], dim=1))


# ---------------------------
# 7) SOTA-inspired Decoder (ACV + MCCV + 3D agg + opt. 2-stage)
# ---------------------------
class SOTAStereoDecoder(nn.Module):
    def __init__(self,
                 max_disp_px: int = 192,
                 fused_in_ch: int = 256,   # StereoModel fused_1_4 channels (Cf)
                 red_ch: int = 48,         # channel after 1x1 reduction for concat volume (Cr)
                 base3d: int = 32,
                 use_motif: bool = True,
                 two_stage: bool = False,  # (opt) local-range refine stage
                 local_radius_cells: int = 8):
        super().__init__()
        self.D = max(1, max_disp_px // 4)
        self.concat = ConcatVolume1_4(in_ch=fused_in_ch, red_ch=red_ch)
        self.acv = ACVAttention(out_ch=2 * red_ch, hidden=16)
        self.agg = Hourglass3D(in_ch=2 * red_ch, base=base3d)
        self.refine_full = FullRefine(cf=fused_in_ch)

        self.use_motif = use_motif
        if use_motif:
            self.motif_proj_L = MotifProjector(in_ch=fused_in_ch, m_ch=24)
            self.motif_proj_R = MotifProjector(in_ch=fused_in_ch, m_ch=24)
            self.mccv_gate     = MCCVGating(m_ch=24, out_ch=2 * red_ch)

        # 2-stage (IGEV++ spirit)
        self.two_stage = two_stage
        self.local_r = local_radius_cells
        if two_stage:
            # second aggregation head for local window
            self.acv_local = ACVAttention(out_ch=2 * red_ch, hidden=8)
            self.agg_local = Hourglass3D(in_ch=2 * red_ch, base=base3d)

        self.corr_builder = CorrVolume1_4(max_disp_px=max_disp_px)

    @torch.no_grad()
    def _check_inputs(self, out: Dict):
        assert "left" in out and "right" in out and "meta" in out
        assert "cossim_feat_1_4" in out["left"] and "fused_1_4" in out["left"]
        assert "cossim_feat_1_4" in out["right"]
        assert "valid_hw" in out["meta"]

    def _build_acv_volume(self, Lcorr, Rcorr, Lfuse, Rfuse) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # corr volume (cos-sim)
        corr = self.corr_builder(Lcorr, Rcorr)                              # [B,D,H4,W4]
        # concat volume (channel-reduced)
        concat_vol, Cr = self.concat(Lfuse, Rfuse, self.D)                  # [B,2*Cr,D,H4,W4]
        # ACV weights from corr
        w_corr = self.acv(corr)                                             # [B,2*Cr,D,H4,W4]
        vol_acv = concat_vol * w_corr                                       # [B,2*Cr,D,H4,W4]
        return vol_acv, corr, Cr

    def forward(self, backbone_out: Dict) -> Dict[str, torch.Tensor]:
        self._check_inputs(backbone_out)

        # fetch features
        Lcorr = backbone_out["left"]["cossim_feat_1_4"]    # [B,H4,W4,C] (L2 norm)
        Rcorr = backbone_out["right"]["cossim_feat_1_4"]
        Lfuse = backbone_out["left"]["fused_1_4"]          # [B,Cf,H4,W4]
        Rfuse = backbone_out["right"]["fused_1_4"]
        H, W  = backbone_out["meta"]["valid_hw"]

        # --- Stage-1: ACV ( + MCCV gating ) ---
        vol_acv, corr_vol, Cr = self._build_acv_volume(Lcorr, Rcorr, Lfuse, Rfuse)  # [B,2Cr,D,H4,W4], [B,D,H4,W4]

        if self.use_motif:
            ML = self.motif_proj_L(Lfuse)     # [B,m,H4,W4]
            MR = self.motif_proj_R(Rfuse)
            w_mccv = self.mccv_gate(ML, MR, self.D)          # [B,2Cr,D,H4,W4]
            vol_acv = vol_acv * (0.5 + 0.5 * w_mccv)         # gentle gating

        # 3D aggregation -> logits
        logits = self.agg(vol_acv)                           # [B,D,H4,W4]
        prob   = F.softmax(-logits, dim=1)
        disp_q = soft_argmin(prob) * 4.0                     # pixel units at 1/4

        out = {
            "prob_volume_1_4": prob,
            "corr_volume_1_4": corr_vol,
            "disp_1_4_stage1": disp_q,
        }

        # --- (optional) Stage-2: local-range refine (IGEV++ spirit) ---
        if self.two_stage and self.local_r > 0:
            B, _, H4, W4 = disp_q.shape
            d0 = (disp_q / 4.0).detach()  # center in cell units
            # build local window indices [ -r ... +r ]
            Dloc = 2 * self.local_r + 1
            offs = torch.arange(-self.local_r, self.local_r + 1, device=disp_q.device, dtype=disp_q.dtype).view(1, Dloc, 1, 1)
            # clamp to [0, D-1]
            d_idx = (d0 + offs).clamp_(0, float(self.D - 1))  # [B,Dloc,H4,W4]
            # gather corr weights at local offsets
            # (bilinear gather along disparity dim)
            d_floor = d_idx.floor().long()
            d_ceil  = (d_floor + 1).clamp(max=self.D - 1)
            alpha   = (d_idx - d_floor.float())
            # corr: [B,D,H4,W4] -> local corr [B,Dloc,H4,W4]
            corr_f = torch.gather(corr_vol, 1, d_floor)
            corr_c = torch.gather(corr_vol, 1, d_ceil)
            corr_loc = (1 - alpha) * corr_f + alpha * corr_c  # [B,Dloc,H4,W4]

            # local concat volume (same channel reduce & shifting by nearest offset)
            Lr = self.concat.reduce(Lfuse)  # [B,Cr,H4,W4]
            Rr = self.concat.reduce(Rfuse)

            vols = []
            # use nearest integer shift for concat (lightweight)
            for i in range(Dloc):
                di = d_floor[:, i, :, :].view(B, 1, H4, W4)  # integer shift per-pixel
                # approximate by uniform shift with median di to stay efficient
                # (full per-pixel warp is heavy; this is a pragmatic trade-off)
                d_med = torch.median(di.float()).item()
                d_med = int(max(0, min(self.D - 1, round(d_med))))
                R_shift = _shift_right_feats(Rr, d_med)
                vols.append(torch.cat([Lr, R_shift], dim=1))
            vol_concat_loc = torch.stack(vols, dim=2)        # [B,2Cr,Dloc,H4,W4]

            # ACV local attention
            w_loc = self.acv_local(corr_loc)                  # [B,2Cr,Dloc,H4,W4]
            vol_loc = vol_concat_loc * w_loc
            logits2 = self.agg_local(vol_loc)                 # [B,Dloc,H4,W4]
            prob2   = F.softmax(-logits2, dim=1)
            disp_ref_cell = soft_argmin(prob2) + (d0 - self.local_r)  # re-center
            disp_q = disp_ref_cell * 4.0

            out.update({
                "prob_volume_local_1_4": prob2,
                "disp_1_4_stage2": disp_q
            })

        # --- Full-res refine ---
        disp_full = self.refine_full(disp_q, Lfuse, (H, W))
        out["disp_full"] = disp_full
        out["confidence_1_4"] = 1.0 - entropy_map(out["prob_volume_1_4"])
        out["disp_1_4"] = disp_q
        return out
