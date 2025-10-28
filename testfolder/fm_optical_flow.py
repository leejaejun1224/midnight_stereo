#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fm_optical_flow.py
------------------
- Vision foundation model: DINOv2 (via HuggingFace Transformers)
- Method: patch-token features (ViT) + local cosine-similarity search (sparse grid) → arrow overlay
- Outputs: flow-overlaid images (and optional .npz feature dumps)

Install:
    pip install -U torch torchvision pillow opencv-python transformers timm tqdm

Usage:
    python fm_optical_flow.py \
        --input_dir /path/to/frames \
        --output_dir /path/to/out \
        --model facebook/dinov2-base \
        --grid_step 2 \
        --search_radius 6 \
        --min_mag_px 2 \
        --save_features

Notes:
- This computes **sparse flow** at the ViT patch-grid resolution and draws arrows on the original RGB frame.
- The search radius is in **token** units. With patch size P (e.g., 14), max displacement covered ≈ search_radius * P pixels.
- For large motions, increase --search_radius or downsample frames externally.
- If you prefer denser arrows, reduce --grid_step (but it will be slower).
"""

import os
import re
import math
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from transformers import AutoImageProcessor, AutoModel

# -------------------------
# Utilities
# -------------------------

def natural_key(path: str):
    """Natural sort key (numbers in filenames sorted numerically)."""
    base = os.path.basename(path)
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', base)]

def load_rgb(path: str) -> np.ndarray:
    """Load image as RGB uint8 (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    return np.array(img)

def pad_to_multiple(img: np.ndarray, multiple: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad bottom/right to make H and W multiples of 'multiple'. Returns padded image and (pad_h, pad_w)."""
    h, w = img.shape[:2]
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    # Pad with black
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded, (pad_h, pad_w)

def draw_arrows(
    img_bgr: np.ndarray,
    flows: List[Tuple[float, float, float, float]],
    color=(0, 255, 0),
    thickness=1,
    tipLength=0.25,
):
    """Draw arrows (x0,y0 -> x1,y1) onto BGR image in-place."""
    h, w = img_bgr.shape[:2]
    for (x0, y0, x1, y1) in flows:
        # Clip to image
        x0i = int(np.clip(x0, 0, w - 1))
        y0i = int(np.clip(y0, 0, h - 1))
        x1i = int(np.clip(x1, 0, w - 1))
        y1i = int(np.clip(y1, 0, h - 1))
        if x0i == x1i and y0i == y1i:
            continue
        cv2.arrowedLine(img_bgr, (x0i, y0i), (x1i, y1i), color, thickness=thickness, tipLength=tipLength)

# -------------------------
# Feature extractor (DINOv2 via HF)
# -------------------------

class Dinov2FeatureExtractor:
    def __init__(self, model_name: str = "facebook/dinov2-base", device: Optional[str] = None):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval().to(self.device)
        # Try to read patch size from config; default to 14 (DINOv2 ViT-B/14)
        self.patch = getattr(getattr(self.model, "config", None), "patch_size", 14)

    @torch.no_grad()
    def image_to_tokens(self, img_rgb_uint8: np.ndarray) -> Tuple[torch.Tensor, int, int, Tuple[int, int]]:
        """
        Convert RGB uint8 image to L2-normalized token feature grid.
        Returns:
            feats (Htok, Wtok, C)  float32 (on CPU)
            Htok, Wtok (ints)
            pad_hw (pad_h, pad_w) used before feature extraction
        """
        # Pad to multiple of patch so that conv patch embedding aligns cleanly
        img_padded, (pad_h, pad_w) = pad_to_multiple(img_rgb_uint8, self.patch)
        # HF processor: disable resizing/cropping; rely only on normalization
        inputs = self.processor(
            img_padded,
            do_resize=False,
            do_center_crop=False,
            return_tensors="pt"
        )
        pixel_values = inputs["pixel_values"].to(self.device)  # (1,3,H,W)

        outputs = self.model(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state  # (1, N, C) typically includes CLS

        # Infer token grid size
        H, W = img_padded.shape[:2]
        Htok = H // self.patch
        Wtok = W // self.patch
        expected = Htok * Wtok

        # Remove CLS if present
        N = tokens.shape[1]
        if N == expected + 1:
            tokens = tokens[:, 1:, :]  # drop CLS
        elif N != expected:
            # Fallback: try to infer grid (square-ish) — but this should rarely happen.
            # We still attempt a best-effort reshape.
            root = int(math.sqrt(N))
            Htok = root
            Wtok = N // root

        # Reshape to (Htok, Wtok, C)
        feats = tokens.reshape(Htok, Wtok, -1).contiguous()
        # L2 normalize
        feats = torch.nn.functional.normalize(feats, dim=-1)
        # Move to CPU float32 for safety
        feats = feats.detach().cpu().float()
        return feats, Htok, Wtok, (pad_h, pad_w)

# -------------------------
# Sparse flow on token grid
# -------------------------

def local_search_displacement(
    F1: torch.Tensor,
    F2: torch.Tensor,
    y: int,
    x: int,
    radius: int,
    mutual_check: bool = True
) -> Tuple[int, int, float]:
    """
    For a token at (y, x) in F1, find the best match in F2 within a square window of 'radius' (token units).
    Returns (dy, dx, best_sim). If mutual_check is True, only keep matches that are reciprocal.
    """
    Htok, Wtok, C = F2.shape
    y0 = max(0, y - radius)
    y1 = min(Htok - 1, y + radius)
    x0 = max(0, x - radius)
    x1 = min(Wtok - 1, x + radius)

    q = F1[y, x]  # (C,)

    # Extract candidate patch from F2: (hwin, wwin, C)
    cand = F2[y0 : y1 + 1, x0 : x1 + 1, :]
    # Flatten to (K, C)
    cand_flat = cand.reshape(-1, C)
    sims = torch.matmul(cand_flat, q)  # cosine sim since already normalized
    best_idx = int(torch.argmax(sims).item())
    best_sim = float(sims[best_idx].item())
    # Map back to (yy, xx)
    hwin = y1 - y0 + 1
    wwin = x1 - x0 + 1
    yy = best_idx // wwin
    xx = best_idx % wwin
    y2 = y0 + yy
    x2 = x0 + xx

    if mutual_check:
        # Best match back from F2(y2,x2) into F1 local neighborhood
        y0r = max(0, y2 - radius)
        y1r = min(Htok - 1, y2 + radius)
        x0r = max(0, x2 - radius)
        x1r = min(Wtok - 1, x2 + radius)

        q2 = F2[y2, x2]
        cand_r = F1[y0r : y1r + 1, x0r : x1r + 1, :]
        cand_r_flat = cand_r.reshape(-1, C)
        sims_r = torch.matmul(cand_r_flat, q2)
        best_idx_r = int(torch.argmax(sims_r).item())
        hwin_r = y1r - y0r + 1
        wwin_r = x1r - x0r + 1
        yy_r = best_idx_r // wwin_r
        xx_r = best_idx_r % wwin_r
        y1best = y0r + yy_r
        x1best = x0r + xx_r
        if not (y1best == y and x1best == x):
            return 0, 0, -1.0  # reject (not mutual)

    return (y2 - y), (x2 - x), best_sim

def compute_sparse_arrows(
    F1: torch.Tensor,
    F2: torch.Tensor,
    patch: int,
    Horig: int,
    Worig: int,
    grid_step: int = 2,
    search_radius: int = 6,
    min_mag_px: float = 1.0,
    min_sim: float = 0.2,
    mutual_check: bool = True,
) -> List[Tuple[float, float, float, float]]:
    """
    Compute sparse arrow list between two frames given token feature grids.
    Returns list of (x0, y0, x1, y1) in pixel coordinates of the **original** image (before padding).
    """
    Htok, Wtok, _ = F1.shape
    arrows = []
    for y in range(0, Htok, grid_step):
        for x in range(0, Wtok, grid_step):
            dy, dx, sim = local_search_displacement(F1, F2, y, x, radius=search_radius, mutual_check=mutual_check)
            if sim < min_sim:
                continue
            # Token center in *padded* pixel coords
            x0 = x * patch + patch / 2.0
            y0 = y * patch + patch / 2.0
            x1 = x0 + dx * patch
            y1 = y0 + dy * patch
            # Filter by magnitude
            mag = math.hypot(dx * patch, dy * patch)
            if mag < min_mag_px:
                continue
            # Discard arrows that end outside original image (if the frame was padded)
            if x0 >= Worig or y0 >= Horig or x1 < 0 or y1 < 0:
                continue
            # Clip end-point to original bounds (just to be safe)
            x1 = min(max(x1, 0.0), Worig - 1.0)
            y1 = min(max(y1, 0.0), Horig - 1.0)
            # Keep only arrows whose start is inside original area
            if x0 < Worig and y0 < Horig:
                arrows.append((x0, y0, x1, y1))
    return arrows

# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="DINOv2-feature optical flow (sparse arrows)")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with sequential frames (images)")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save flow overlays")
    parser.add_argument("--model", type=str, default="facebook/dinov2-base", help="HF model id (e.g., facebook/dinov2-base, -small, -large)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda, cpu (default: auto)")
    parser.add_argument("--grid_step", type=int, default=2, help="Token grid stride for arrows (larger = fewer arrows)")
    parser.add_argument("--search_radius", type=int, default=6, help="Search radius in token units")
    parser.add_argument("--min_mag_px", type=float, default=2.0, help="Minimum arrow length in pixels")
    parser.add_argument("--min_sim", type=float, default=0.2, help="Minimum cosine similarity to accept a match")
    parser.add_argument("--mutual_check", action="store_true", help="Enable mutual (forward-backward) check to drop outliers")
    parser.add_argument("--save_features", action="store_true", help="Save per-frame token features as .npz")
    parser.add_argument("--exts", type=str, default="jpg,jpeg,png,bmp", help="Image extensions to include")

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect and sort frames
    exts = tuple(("." + e.strip().lower()) for e in args.exts.split(","))
    frame_paths = [str(p) for p in input_dir.iterdir() if p.suffix.lower() in exts]
    frame_paths = sorted(frame_paths, key=natural_key)
    assert len(frame_paths) >= 2, "Need at least 2 images in the input folder."

    # Prepare model
    feat_extractor = Dinov2FeatureExtractor(model_name=args.model, device=args.device)
    patch = feat_extractor.patch

    # Extract features & process pairs
    feats_cache = {}  # simple cache to reuse F(t) as F(t+1) in next step
    Horig_cache = {}
    Worig_cache = {}

    for i in tqdm(range(len(frame_paths) - 1), desc="Processing pairs", ncols=100):
        p0 = frame_paths[i]
        p1 = frame_paths[i + 1]

        if p0 in feats_cache:
            F0, Htok0, Wtok0, pad0 = feats_cache[p0]
            H0, W0 = Horig_cache[p0]
        else:
            img0 = load_rgb(p0)
            H0, W0 = img0.shape[:2]
            F0, Htok0, Wtok0, pad0 = feat_extractor.image_to_tokens(img0)
            feats_cache[p0] = (F0, Htok0, Wtok0, pad0)
            Horig_cache[p0] = (H0, W0)

        img1 = load_rgb(p1)
        H1, W1 = img1.shape[:2]
        F1, Htok1, Wtok1, pad1 = feat_extractor.image_to_tokens(img1)
        feats_cache[p1] = (F1, Htok1, Wtok1, pad1)
        Horig_cache[p1] = (H1, W1)

        # Compute sparse arrows (F0 -> F1). We use H0,W0 to clip within original (unpadded) area.
        arrows = compute_sparse_arrows(
            F0, F1,
            patch=patch,
            Horig=H0,
            Worig=W0,
            grid_step=args.grid_step,
            search_radius=args.search_radius,
            min_mag_px=args.min_mag_px,
            min_sim=args.min_sim,
            mutual_check=args.mutual_check,
        )

        # Draw on original image (BGR for OpenCV)
        img0_bgr = cv2.cvtColor(load_rgb(p0), cv2.COLOR_RGB2BGR)
        draw_arrows(img0_bgr, arrows, color=(0, 255, 0), thickness=1, tipLength=0.25)

        out_name = f"flow_{i:05d}.jpg"
        cv2.imwrite(str(output_dir / out_name), img0_bgr)

        # Optionally save features
        if args.save_features:
            # Save as npz: feats (Htok,Wtok,C), patch, Horig,Worig
            np.savez_compressed(
                str(output_dir / f"features_{i:05d}.npz"),
                feats=F0.numpy(),
                patch=patch,
                Horig=H0,
                Worig=W0
            )

    print(f"Done. Saved overlays to: {str(output_dir)}")

if __name__ == "__main__":
    main()
