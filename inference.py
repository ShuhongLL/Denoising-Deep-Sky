#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from astropy.io import fits

from dataset import FitsPairDataset  # (x:(1,H,W), y:(1,H,W), bg_mask:bool, os_path, gt_path)
from model import UNet, PMNUNet
from metric import evaluate_loader


def resolve_ckpt_path(ckpt_arg: str) -> Path:
    p = Path(ckpt_arg).expanduser().resolve()
    if p.is_dir():
        p = p / "final.pth"
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


def save_comp_fits_fz_single(out_path: Path, img2d: np.ndarray, header: Optional[fits.Header] = None) -> None:
    """
    Save a single 2D float32 image at HDU[1] as CompImageHDU (GZIP_1, 32x32 tiles).
    """
    if img2d.ndim != 2:
        raise ValueError(f"Expected 2D image (H,W), got {img2d.shape}")
    data32 = img2d.astype("float32", copy=False)
    primary = fits.PrimaryHDU()
    comp = fits.CompImageHDU(
        data=data32,
        header=header,
        compression_type="GZIP_1",
        tile_shape=(32, 32),
    )
    hdul = fits.HDUList([primary, comp])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdul.writeto(out_path, overwrite=True, output_verify="silentfix")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Run inference on first N samples of train/test and save each prediction as its own fits.fz.")
    ap.add_argument("-c", "--checkpoint", required=True, help="Path to checkpoint dir or final.pth")
    ap.add_argument("-d", "--data", required=True, help="Root dir containing train_test_split.json and subfolders")
    ap.add_argument("-x", "--input-suffix", default="os", help="input npy suffix")
    ap.add_argument("-y", "--label-suffix", default="calib", help="label npy suffix")
    split = ap.add_mutually_exclusive_group()
    split.add_argument("--train", action="store_true", help="Use train split")
    split.add_argument("--test", action="store_true", help="Use test split (default)")
    ap.add_argument("--json", default="train_test_split.json", help="Split JSON filename")
    ap.add_argument("-o", "--output", default="", help="Output directory; default: <ckpt_dir>/results/<split>")
    ap.add_argument("--arch", default="unet", choices=["unet", "pmn_unet"], help="model architecture")
    ap.add_argument("--format", default="fits.fz", choices=["fits.fz", "npy"], help="Output format for predictions")
    ap.add_argument("--batch", type=int, default=1, help="Batch size for inference")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="cuda or cpu")
    ap.add_argument("--eval", action="store_true", help="Also run metrics on the same subset")

    args = ap.parse_args()

    ckpt_path = resolve_ckpt_path(args.checkpoint)
    ckpt_dir = ckpt_path.parent
    root = Path(args.data).expanduser().resolve()
    json_path = root / args.json
    if not json_path.is_file():
        raise FileNotFoundError(f"Split json not found: {json_path}")

    split_name = "train" if args.train else "test"

    # Dataset
    ds = FitsPairDataset(
        root=str(root),
        split_json=args.json,
        split=split_name,
        input_suffix=args.input_suffix,
        label_suffix=args.label_suffix,
    )
    # Output dir
    if args.output:
        out_dir = Path(args.output).expanduser().resolve()
    else:
        out_dir = ckpt_dir / "results" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model
    device = torch.device(args.device)
    if args.arch == "unet":
        model = UNet(in_nc=1, out_nc=1, nf=32).to(device)
    else:
        model = PMNUNet(in_nc=1, out_nc=1, nf=32, res=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    torch.set_grad_enabled(False)

    loader = DataLoader(
        ds,
        batch_size=max(1, int(args.batch)),
        shuffle=False,
        num_workers=max(0, int(args.workers)),
        pin_memory=True,
        persistent_workers=(int(args.workers) > 0),
        drop_last=False,
    )

    saved = 0
    printed_debug = False
    for batch in loader:
        x, _, _, os_paths, gt_paths = batch
        if not printed_debug:
            print("[DEBUG] os_path:", os_paths[0])
            print("[DEBUG] gt_path:", gt_paths[0])
            printed_debug = True
        x = x.to(device, non_blocking=True)                 # (B,1,H,W)
        pred = model(x).detach().cpu().numpy()              # (B,1,H,W)

        B = pred.shape[0]
        for b in range(B):
            img2d = pred[b, 0]                              # (H,W)
            subdir_name = Path(os_paths[b]).parent.name
            if args.format == "npy":
                out_fp = out_dir / f"{subdir_name}_pred.npy"
                np.save(out_fp, img2d)
            else:
                out_fp = out_dir / f"{subdir_name}_pred.fits.fz"
                save_comp_fits_fz_single(out_fp, img2d)
            print(f"[OK] {out_fp}")
            saved += 1

    print(f"[DONE] saved {saved} files to {out_dir}")

    if args.eval:
        metrics = evaluate_loader(model, loader, device, criterion=None)
        print("-" * 70)
        print(
            f"[EVAL on {split_name}] "
            f"PSNR={metrics['psnr']:.4f} dB, SSIM={metrics['ssim']:.6f} | "
            # f"ASINH-PSNR={metrics['psnr_asinh']:.4f} dB, ASINH-SSIM={metrics['ssim_asinh']:.6f} | "
            f"LOSS={metrics['loss']:.6g} | "
            f"BG: NMAD(pred)={metrics.get('bg_nmad_pred', float('nan')):.6g}, "
            f"σ(pred)={metrics.get('bg_sigma_pred', float('nan')):.6g}"
        )


if __name__ == "__main__":
    main()
