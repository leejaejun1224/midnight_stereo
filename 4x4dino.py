# click_similarity_viz.py (Semantic-preserving cosine sim)
# DINO(1/8→1/4) + MobileNetV2(1/4), but similarity is fused at MAP-level to keep semantics.
import os, sys, argparse, cv2, random, colorsys, requests, inspect
from io import BytesIO
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as pth_transforms
from torchvision import models as tv_models

import vision_transformer as vits


# ----------------------------
# utils
# ----------------------------
def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / max(N,1), 1, brightness) for i in range(max(N,1))]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

# ----------------------------
# MobileNetV2 stride-4 feature extractor (24ch)
# ----------------------------
class MobileNetV2_S4(nn.Module):
    def __init__(self, pretrained=True, out_norm=True):
        super().__init__()
        try:
            from torchvision.models import MobileNet_V2_Weights
            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            m = tv_models.mobilenet_v2(weights=weights)
            mean = (0.485, 0.456, 0.406) if weights is None else weights.meta["mean"]
            std  = (0.229, 0.224, 0.225) if weights is None else weights.meta["std"]
        except Exception:
            m = tv_models.mobilenet_v2(pretrained=pretrained)
            mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)

        self.encoder = nn.Sequential(*list(m.features[:4]))  # stride 4
        self.register_buffer("mean", torch.tensor(mean).view(1,3,1,1), persistent=False)
        self.register_buffer("std",  torch.tensor(std).view(1,3,1,1),  persistent=False)
        self.out_norm = out_norm
        for p in self.parameters(): p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        # x: [B,3,H,W], uint8 or float
        if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            x = x.float().div_(255.0)
        B,C,H,W = x.shape
        need_h = (4 - (H % 4)) % 4; need_w = (4 - (W % 4)) % 4
        if need_h or need_w:
            x = F.pad(x, (0, need_w, 0, need_h), mode="reflect")
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        y = self.encoder(x)                  # [B,24,ceil(H/4),ceil(W/4)]
        y = y[:, :, :H//4, :W//4].contiguous()
        if self.out_norm: y = F.normalize(y, dim=1, eps=1e-6)
        return y                             # [B,24,H/4,W/4]

# ----------------------------
# DINO tokens → grid [B,Cd,H/8,W/8]
# ----------------------------
@torch.no_grad()
def dino_tokens_to_grid(model, x_bchw, patch_size):
    sig = inspect.signature(model.get_intermediate_layers)
    if "return_class_token" in sig.parameters:
        out = model.get_intermediate_layers(x_bchw, n=1, return_class_token=True)[0]
        if isinstance(out, (tuple, list)): patch_tokens = out[1]
        else:                               patch_tokens = out[:, 1:, :]
    else:
        out = model.get_intermediate_layers(x_bchw, n=1)[0]
        patch_tokens = out[:, 1:, :]

    B,_,H,W = x_bchw.shape
    B2,P,Cd = patch_tokens.shape
    assert B2==B
    h, w = H//patch_size, W//patch_size
    assert h*w==P, f"Token {P} != h*w {h*w}"
    grid = patch_tokens.transpose(1,2).contiguous().view(B, Cd, h, w)
    return grid  # [B,Cd,H/8,W/8]

def reduce_channels_group_mean(feat, out_channels):
    B,C,H,W = feat.shape
    assert C % out_channels == 0, f"C({C}) % out_channels({out_channels}) != 0"
    g = C // out_channels
    return feat.view(B, out_channels, g, H, W).mean(dim=2)

# ----------------------------
# Main visualizer (cos-sim only)
# ----------------------------
class ClickSimilarityVisualizer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # DINO backbone
        self.model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        for p in self.model.parameters(): p.requires_grad = False
        self.model.eval().to(self.device)

        # weights
        if os.path.isfile(args.pretrained_weights):
            state_dict = torch.load(args.pretrained_weights, map_location="cpu")
            if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
                print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[args.checkpoint_key]
            state_dict = {k.replace("module.", ""): v for k,v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k,v in state_dict.items()}
            msg = self.model.load_state_dict(state_dict, strict=False)
            print(f'Loaded {args.pretrained_weights} with msg: {msg}')
        else:
            print("No --pretrained_weights provided. Loading reference DINO weights.")
            url = None
            if args.arch=="vit_small" and args.patch_size==16:
                url="dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            elif args.arch=="vit_small" and args.patch_size==8:
                url="dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
            elif args.arch=="vit_base" and args.patch_size==16:
                url="dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            elif args.arch=="vit_base" and args.patch_size==8:
                url="dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            if url is not None:
                state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/"+url)
                self.model.load_state_dict(state_dict, strict=True)
            else:
                print("No reference weights; using random (not recommended).")

        # image
        if args.image_path is None:
            print("No --image_path provided. Download sample.")
            response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
            img_pil = Image.open(BytesIO(response.content)).convert('RGB')
        elif os.path.isfile(args.image_path):
            with open(args.image_path, 'rb') as f:
                img_pil = Image.open(f).convert('RGB')
        else:
            print(f"Invalid --image_path: {args.image_path}"); sys.exit(1)

        # resize
        if isinstance(args.image_size, (list,tuple)) and len(args.image_size)==2:
            target_size = (args.image_size[0], args.image_size[1])  # (H,W)
        else:
            target_size = (args.image_size, args.image_size)
        img_resized_pil = img_pil.resize(target_size, Image.BILINEAR)
        img_np = np.array(img_resized_pil)

        # transforms
        tfm_dino = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])
        tfm_raw = pth_transforms.ToTensor()

        x_dino = tfm_dino(img_resized_pil)      # (3,H,W) normalized
        x_raw  = tfm_raw(img_resized_pil)       # (3,H,W) 0..1

        # crop to patch multiple (8) → stride 4 auto satisfied
        H = x_dino.shape[1] - x_dino.shape[1] % args.patch_size
        W = x_dino.shape[2] - x_dino.shape[2] % args.patch_size
        self.img_vis = img_np[:H, :W, :].copy()
        self.x_dino = x_dino[:,:H,:W].unsqueeze(0).to(self.device)
        self.x_raw  = x_raw[:,:H,:W].unsqueeze(0).to(self.device)

        self.Hc, self.Wc = H, W
        self.grid_stride = args.patch_size // 2  # 8 → 4
        self.Hg, self.Wg = H//self.grid_stride, W//self.grid_stride
        self.P = self.Hg * self.Wg

        # --- build features ---
        print("Extracting features...")
        with torch.no_grad():
            # DINO 1/8 → 1/4
            dino_1over8 = dino_tokens_to_grid(self.model, self.x_dino, args.patch_size)  # [1,Cd,H/8,W/8]
            dino_1over4 = F.interpolate(dino_1over8, scale_factor=2, mode='nearest')     # [1,Cd,H/4,W/4]
            dino_1over4 = F.normalize(dino_1over4, dim=1, eps=1e-6)

            # CNN 1/4
            cnn_extractor = MobileNetV2_S4(pretrained=True, out_norm=True).to(self.device)
            cnn_1over4 = cnn_extractor(self.x_raw)                                      # [1,24,H/4,W/4]

            # 공간 중심화(옵션): 전역 명암/색 편향 제거
            if args.cnn_center:
                mu = cnn_1over4.mean(dim=(2,3), keepdim=True)
                std = cnn_1over4.std(dim=(2,3), keepdim=True) + 1e-6
                cnn_1over4 = (cnn_1over4 - mu) / std
                cnn_1over4 = F.normalize(cnn_1over4, dim=1, eps=1e-6)

            # shape 정합
            H4 = min(dino_1over4.shape[2], cnn_1over4.shape[2], self.Hg)
            W4 = min(dino_1over4.shape[3], cnn_1over4.shape[3], self.Wg)
            dino_1over4 = dino_1over4[:, :, :H4, :W4].contiguous()
            cnn_1over4  = cnn_1over4[:,  :, :H4, :W4].contiguous()
            self.Hg, self.Wg = H4, W4; self.P = H4*W4

            # 보관 (torch)
            self.feat_dino = dino_1over4        # [1,Cd,H4,W4]  L2 norm per-location
            self.feat_cnn  = cnn_1over4         # [1,24,H4,W4]  centered + L2 norm per-location

            # 필요 시 특징 레벨 fuse도 유지(참조용)
            if args.fuse_mode == "concat":
                fused = torch.cat([dino_1over4, cnn_1over4], dim=1)
                fused = F.normalize(fused, dim=1, eps=1e-6)
            elif args.fuse_mode == "sum":
                # DINO 채널을 24로 축소 후 가중합
                dino_red = reduce_channels_group_mean(dino_1over4, cnn_1over4.shape[1])  # [1,24,H4,W4]
                dino_red = F.normalize(dino_red, dim=1, eps=1e-6)
                fused = args.sum_alpha * dino_red + (1.0 - args.sum_alpha) * cnn_1over4
                fused = F.normalize(fused, dim=1, eps=1e-6)
            else:
                fused = None
            self.feat_fused = fused  # (optional)

        # figure
        os.makedirs(args.output_dir, exist_ok=True)
        torchvision.utils.save_image(
            torchvision.utils.make_grid(self.x_dino.cpu(), normalize=True, scale_each=True),
            os.path.join(args.output_dir, "img_model_space.png")
        )
        cv2.imwrite(os.path.join(args.output_dir, "img_resized.png"),
                    cv2.cvtColor(self.img_vis, cv2.COLOR_RGB2BGR))

        self.fig, self.ax1 = plt.subplots(1, 1, figsize=(6, 6))
        self.ax1.set_title(f"Cosine similarity — sim_mode={args.sim_mode}, dino_w={args.dino_weight}")
        self.im1 = self.ax1.imshow(self.img_vis)
        self.overlay1 = self.ax1.imshow(np.zeros((self.Hc, self.Wc)), alpha=0.0, cmap='jet', vmin=0.0, vmax=1.0)
        self.marker1 = self.ax1.scatter([], [], s=30, c='white', marker='x')
        divider1 = make_axes_locatable(self.ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        self.cbar1 = self.fig.colorbar(self.overlay1, cax=cax1)
        self.cbar1.set_label("Similarity (0–1)", rotation=270, labelpad=12)
        self.cbar1.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        self.cbar1.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        self.ax1.set_axis_off()

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.click_idx = 0

    # ---------- helpers ----------
    def _xy_to_patch_index(self, x, y):
        pj = int(x) // self.grid_stride
        pi = int(y) // self.grid_stride
        pi = np.clip(pi, 0, self.Hg - 1); pj = np.clip(pj, 0, self.Wg - 1)
        return pi * self.Wg + pj, (pi, pj)

    def _upsample_gridmap(self, grid_map_hw):
        return cv2.resize(grid_map_hw, (self.Wc, self.Hc), interpolation=cv2.INTER_NEAREST)

    @torch.no_grad()
    def _cos_map_from_feat(self, feat_1over4, pidx):
        """
        feat_1over4: [1, C, H4, W4], L2-normalized per location
        return: (Hc, Wc) numpy in [0,1]
        """
        B,C,H4,W4 = feat_1over4.shape
        f = feat_1over4.flatten(start_dim=2).transpose(1,2).contiguous().squeeze(0)  # [P,C]
        ref = f[pidx:pidx+1, :]                            # [1,C]
        sim = torch.matmul(f, ref.t()).squeeze(1)          # [P]
        sim01 = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        grid = sim01.view(H4, W4).detach().cpu().numpy()
        up = self._upsample_gridmap(grid)
        return up

    def onclick(self, event):
        if event.inaxes not in (self.ax1,): return
        if event.xdata is None or event.ydata is None: return

        x, y = int(event.xdata), int(event.ydata)
        pidx, (pi, pj) = self._xy_to_patch_index(x, y)

        # --- compute similarity map according to sim_mode ---
        mode = self.args.sim_mode
        if mode == "dino_only":
            sim_map01 = self._cos_map_from_feat(self.feat_dino, pidx)

        elif mode == "late_weighted":
            sim_d = self._cos_map_from_feat(self.feat_dino, pidx)
            sim_c = self._cos_map_from_feat(self.feat_cnn,  pidx)
            a = float(self.args.dino_weight)
            sim = a * sim_d + (1.0 - a) * sim_c
            smin, smax = sim.min(), sim.max()
            sim_map01 = (sim - smin) / (smax - smin + 1e-8)

        elif mode == "late_and":
            # 기하 평균: DINO가 게이트처럼 작동
            sim_d = self._cos_map_from_feat(self.feat_dino, pidx)
            sim_c = self._cos_map_from_feat(self.feat_cnn,  pidx)
            a = float(self.args.dino_weight)  # 0.0~1.0
            sim = np.power(np.maximum(sim_d, 1e-8), a) * np.power(np.maximum(sim_c, 1e-8), 1.0 - a)
            smin, smax = sim.min(), sim.max()
            sim_map01 = (sim - smin) / (smax - smin + 1e-8)

        elif mode == "fused_concat":
            assert self.feat_fused is not None, "fused features not built"
            sim_map01 = self._cos_map_from_feat(self.feat_fused, pidx)

        elif mode == "fused_sum":
            assert self.feat_fused is not None, "fused features not built"
            sim_map01 = self._cos_map_from_feat(self.feat_fused, pidx)

        else:
            raise ValueError(f"Unknown sim_mode: {mode}")

        # update view
        self.overlay1.set_data(sim_map01); self.overlay1.set_alpha(self.args.alpha)
        self.marker1.set_offsets([[x, y]])
        self.cbar1.update_normal(self.overlay1)
        self.fig.canvas.draw_idle()

        # save
        base = f"click_{self.click_idx:03d}_x{x}_y{y}"
        plt.imsave(os.path.join(self.args.output_dir, base + "_sim_heat.png"),
                   sim_map01, cmap='jet', vmin=0, vmax=1)
        sim_overlay = (self.img_vis * 0.6 + (plt.get_cmap('jet')(sim_map01)[..., :3] * 255) * 0.4).astype(np.uint8)
        cv2.imwrite(os.path.join(self.args.output_dir, base + "_sim_overlay.png"),
                    cv2.cvtColor(sim_overlay, cv2.COLOR_RGB2BGR))

        print(f"[{self.click_idx}] (x={x}, y={y}) -> grid 1/4 (row={pi}, col={pj}, idx={pidx}) saved.")
        self.click_idx += 1

    def run(self):
        print("Click anywhere: cosine similarity. Close window to end.")
        plt.tight_layout(); plt.show()


def parse_args():
    p = argparse.ArgumentParser('Cosine similarity with DINO semantics preserved')
    p.add_argument('--arch', default='vit_base', type=str,
                   choices=['vit_tiny','vit_small','vit_base'])
    p.add_argument('--patch_size', default=8, type=int)
    p.add_argument('--pretrained_weights', default='./pretrained/dino_vitbase8_pretrain.pth', type=str)
    p.add_argument("--checkpoint_key", default="teacher", type=str)
    p.add_argument("--image_path", default="/home/jaejun/dataset/MS2/sync_data/_2021-08-13-22-36-41/rgb/img_left/000128.png", type=str)
    p.add_argument("--image_size", default=(1224,384), type=int, nargs="+")
    p.add_argument('--output_dir', default='./output_clickviz')
    p.add_argument("--alpha", type=float, default=0.5)
    # ---- Key options to keep semantics ----
    p.add_argument("--sim_mode", type=str, default="late_weighted",
                   choices=["dino_only","late_weighted","late_and","fused_concat","fused_sum"],
                   help="Where to compute cosine similarity. Prefer late_* to preserve semantics.")
    p.add_argument("--dino_weight", type=float, default=0.85,
                   help="Weight for DINO similarity in late fusion (0~1). Higher => more semantic.")
    p.add_argument("--cnn_center", action="store_true", default=True,
                   help="Spatially center & whiten CNN features to remove global color/intensity bias.")
    p.add_argument("--fuse_mode", type=str, default="concat", choices=["concat","sum"],
                   help="(Optional) build a fused feature map too (for fused_* sim_mode).")
    p.add_argument("--sum_alpha", type=float, default=0.5,
                   help="When fuse_mode=sum: alpha*DINOred + (1-alpha)*CNN")
    args = p.parse_args()
    if isinstance(args.image_size, list) and len(args.image_size)==1:
        args.image_size = args.image_size[0]
    return args


if __name__ == "__main__":
    args = parse_args()
    vis = ClickSimilarityVisualizer(args)
    vis.run()
