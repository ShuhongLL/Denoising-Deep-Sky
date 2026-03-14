#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import math
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import FitsPairDataset
from model import UNet, PMNUNet
from metric import psnr, evaluate_loader, LO, HI

# -----------------------------
#   LOSSES (for training)
# -----------------------------
class RawL2Loss(nn.Module):
    """Tone-curved / reweighted L2 loss"""
    def __init__(self, eps: float = 1e-3, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
        with torch.no_grad():
            w = 1.0 / (pred.clamp(min=0.0).detach() + self.eps)
        diff2 = (pred - target) ** 2
        if mask is not None:
            diff2 = diff2 * mask
            w = w * mask
        loss = diff2 * w
        if self.reduction == "mean":
            denom = w.sum()
            return loss.sum() / (denom + 1e-12)
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class RawL1Loss(nn.Module):
    """Tone-curved / reweighted L1 loss"""
    def __init__(self, eps: float = 1e-3, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
        with torch.no_grad():
            w = 1.0 / (pred.clamp(min=0.0).detach() + self.eps)
        diff = (pred - target).abs()
        if mask is not None:
            diff = diff * mask
            w = w * mask
        loss = diff * w
        if self.reduction == "mean":
            denom = w.sum()
            return loss.sum() / (denom + 1e-12)
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# -----------------------------
# Dataloaders
# -----------------------------
def make_loaders(root: str, json_name: str, batch_size: int, num_workers: int, input_suffix: str, label_suffix: str):
    ds_train = FitsPairDataset(root, split_json=json_name, split="train",
                               input_suffix=input_suffix, label_suffix=label_suffix)
    ds_test  = FitsPairDataset(root, split_json=json_name, split="test",
                           input_suffix=input_suffix, label_suffix=label_suffix)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    dl_test  = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return dl_train, dl_test


# -----------------------------
# LR schedule: piecewise constant factor
# 0-99: 1.0 (1e-4), 100-179: 0.5 (5e-5), 180-199: 0.1 (1e-5)
# -----------------------------
def lr_factor(epoch: int) -> float:
    # epoch is 0-based: 0..119
    if epoch < 60:
        return 1.0
    elif epoch < 110:
        return 0.5
    else:
        return 0.1


# -----------------------------
# Validation loop (uses RawL1Loss by default)
# -----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    psnrs, losses = [], []
    crit = RawL1Loss()
    for x, y, _, *_ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = crit(pred, y)
        losses.append(loss.item())
        psnrs.append(psnr(pred, y))
    val_psnr = float(np.nanmean(psnrs)) if psnrs else float("nan")
    val_loss = float(np.mean(losses)) if losses else float("nan")
    return val_psnr, val_loss


# -----------------------------
# Main training
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True, help="root dir containing subdirs + train_test_split.json")
    ap.add_argument("--json", default="train_test_split.json")
    ap.add_argument("--arch", default="unet", choices=["unet", "pmn_unet"], help="model architecture")
    ap.add_argument("-x", "--input-suffix", default="syn")
    ap.add_argument("-y", "--label-suffix", default="mean")
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save", default="experiments", help="directory to save checkpoints")
    ap.add_argument("--resume", default="", help="path to a checkpoint .pth (state_dict)")
    ap.add_argument("--name", default="", help="prefix for output folder name")
    ap.add_argument("--amp", action="store_true", help="enable mixed precision (torch.cuda.amp)")
    ap.add_argument("--grad-clip", type=float, default=0.0, help="0 to disable")
    ap.add_argument("--smoke", action="store_true", help="run 1 epoch on a small subset")
    ap.add_argument("--smoke-train", type=int, default=8)
    ap.add_argument("--smoke-test", type=int, default=8)
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_suffix = args.input_suffix
    label_suffix = args.label_suffix

    # Load dataset
    dl_train, dl_test = make_loaders(args.data, args.json, args.batch_size, args.workers,
                                     input_suffix=input_suffix, label_suffix=label_suffix)

    if args.smoke:
        args.epochs = 1
        train_ds = dl_train.dataset
        test_ds = dl_test.dataset
        ntr = min(int(args.smoke_train), len(train_ds))
        nts = min(int(args.smoke_test), len(test_ds))
        bs_tr = max(1, min(int(args.batch_size), ntr))
        bs_te = max(1, min(int(args.batch_size), nts))
        dl_train = DataLoader(Subset(train_ds, list(range(ntr))), batch_size=bs_tr, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
        dl_test = DataLoader(Subset(test_ds, list(range(nts))), batch_size=bs_te, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # Model (single-channel)
    if args.arch == "unet":
        model = UNet(in_nc=1, out_nc=1, nf=32).to(device)
    elif args.arch == "pmn_unet":
        model = PMNUNet(in_nc=1, out_nc=1, nf=32, res=False).to(device)
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")
    
    if args.resume and Path(args.resume).is_file():
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state)
        print(f"[INFO] Resumed weights from: {args.resume}")

    # Optim & sched
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda e: lr_factor(e))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Training criterion (tone-curved L1)
    criterion = RawL1Loss()

    # Output dir
    out_name = f"{args.name}_{Path(args.data).name}_{timestamp}"
    out_dir = Path(args.save) / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_psnr = -1.0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for x, y, _, *_ in dl_train:
            x = x.to(device, non_blocking=True)     # (B,1,H,W)
            y = y.to(device, non_blocking=True)     # (B,1,H,W)

            x = torch.nan_to_num(x, nan=0.0, posinf=float(HI), neginf=float(LO)).clamp(float(LO), float(HI))
            y = torch.nan_to_num(y, nan=0.0, posinf=float(HI), neginf=float(LO)).clamp(float(LO), float(HI))

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(x)                     # (B,1,H,W)
                pred = torch.nan_to_num(pred, nan=0.0, posinf=float(HI), neginf=float(LO)).clamp(float(LO), float(HI))
                loss = criterion(pred, y)

            scaler.scale(loss).backward()

            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))

            scaler.step(optim)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        
        denom_steps = max(1, len(dl_train))
        if (epoch + 1) % 20 == 0 or (epoch + 1) == args.epochs:
            stats = evaluate_loader(model, dl_test, device, criterion=RawL1Loss())
            print(f"[Epoch {epoch+1:03d}] train_loss={epoch_loss/denom_steps:.6f} | "
                f"val_loss={stats['loss']:.6f} | "
                f"PSNR={stats['psnr']:.3f}  SSIM={stats['ssim']:.4f} | "
                # f"ASINH-PSNR={stats['psnr_asinh']:.3f}  ASINH-SSIM={stats['ssim_asinh']:.4f} | "
                f"NMAD={stats['bg_nmad_pred']:.6f}  SIGMA={stats['bg_sigma_pred']:.6f} | "
                f"lr={optim.param_groups[0]['lr']:.2e}")

            if stats['psnr'] > best_psnr:
                best_psnr = stats['psnr']
                ckpt = out_dir / f"best_psnr_{best_psnr:.2f}dB_epoch{epoch+1}.pth"
                torch.save(model.state_dict(), ckpt)
        else:
            print(f"[Epoch {epoch+1:03d}] train_loss={epoch_loss/denom_steps:.6f} | "
                  f"lr={optim.param_groups[0]['lr']:.2e}")

    # Final save
    save_path = out_dir / "final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Model saved to: {save_path}")


if __name__ == "__main__":
    main()