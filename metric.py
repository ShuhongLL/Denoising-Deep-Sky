#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metric.py — evaluate a checkpoint on the test split using the npy-based dataset.

- Requires: --checkpoint/-c (MANDATORY)
- Loads test split from dataset.FitsPairDataset (returns (x, y, bg_mask, os_path, gt_path))
- Runs inference and reports averaged metrics:
    psnr, ssim, psnr_asinh, ssim_asinh, loss,
    bg_nmad_pred, bg_sigma_pred

Changes from previous version:
- Linear-domain PSNR/SSIM now evaluated after clipping both pred/gt to [-200, 2500].
- "log" metrics replaced by asinh-domain metrics with fixed k=5.
- All data_range / MAX use global constants for comparability across images.

Public API:
    evaluate_loader(model, loader, device, criterion=None) -> Dict[str, float]
    evaluate_checkpoint(root, split_json, checkpoint_dir, device,
                        batch_size=1, num_workers=4, verbose=False,
                        criterion=None) -> Dict[str, float]

`train.py` 可以 `from metric import evaluate_checkpoint` 直接调用。
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Dict

import math
import numpy as np
import torch
import torch.nn as nn

from model import UNet                   # single-channel UNet
from dataset import FitsPairDataset     # returns (x:(1,H,W), y:(1,H,W), bg_mask:(1,H,W)[bool], os, gt)

# ---------- Optional SSIM dependency ----------
try:
    from skimage.metrics import structural_similarity as ssim_skimage
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False

# -----------------------------
#   Global evaluation constants
# -----------------------------
LO = -200.0        # global lower bound for linear-domain evaluation
HI = 2500.0        # global upper bound for linear-domain evaluation
DR = HI - LO       # linear data range (used for PSNR/SSIM)
K_ASINH = 5.0      # fixed k for asinh-domain evaluation
LO_T = np.arcsinh(K_ASINH * LO)
HI_T = np.arcsinh(K_ASINH * HI)
DR_T = HI_T - LO_T # asinh-domain data range

# -----------------------------
#   Utils
# -----------------------------
def _torch_clip(x: torch.Tensor, lo: float = LO, hi: float = HI) -> torch.Tensor:
    return torch.clamp(x, min=lo, max=hi)

def _np_clip(a: np.ndarray, lo: float = LO, hi: float = HI) -> np.ndarray:
    return np.clip(a, lo, hi).astype(np.float32, copy=False)

def _np_asinh_k(a: np.ndarray, k: float = K_ASINH) -> np.ndarray:
    # assume input already clipped to [LO, HI] for comparable range
    return np.arcsinh(k * a).astype(np.float32, copy=False)

def robust_nmad(x: np.ndarray) -> float:
    """NMAD = 1.4826 * median(|x - median(x)|); empty -> nan."""
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return 1.4826 * float(np.median(np.abs(x - med)))

def _masked_values_np(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract finite values from arr under boolean mask.
    Accepts arr shapes (H,W) or (1,H,W), mask shapes (H,W) or (1,H,W).
    Returns 1D float64 array.
    """
    a = np.asarray(arr, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    if a.shape != m.shape:
        raise ValueError(f"Mask shape {m.shape} != image shape {a.shape}")
    m = m & np.isfinite(a)
    return a[m]

def background_stats_pred(pred_1chw: np.ndarray, bg_mask_1chw: np.ndarray) -> Dict[str, float]:
    """
    Compute NMAD and sigma over background (mask=True) on prediction.
    Inputs are numpy arrays with shape (1,H,W) or (H,W).
    Returns dict: {'nmad':..., 'std':..., 'mean':..., 'N':...}
    """
    vals = _masked_values_np(pred_1chw, bg_mask_1chw)
    if vals.size == 0:
        return {"nmad": float("nan"), "std": float("nan"), "mean": float("nan"), "N": 0}
    return {
        "nmad": robust_nmad(vals),
        "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        "mean": float(np.mean(vals)),
        "N": int(vals.size),
    }

# -----------------------------
#   PSNR (linear tensor; clipped; global DR)
# -----------------------------
def psnr(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Linear-domain PSNR on tensors (B,1,H,W) or (1,H,W); AFTER clipping to [LO, HI].
    Uses global DR (=HI-LO). If mask is given, compute over masked pixels only.
    """
    pred_c = _torch_clip(pred)
    tgt_c  = _torch_clip(target)
    if mask is not None:
        valid = mask if mask.dtype == torch.bool else (mask > 0.5)
        if not torch.any(valid):
            return float("nan")
        diff = (pred_c - tgt_c)[valid]
        mse = torch.mean(diff * diff).item()
    else:
        mse = torch.mean((pred_c - tgt_c) ** 2).item()
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(DR) - 10.0 * math.log10(mse)

# -----------------------------
#   NumPy helpers (SSIM/PSNR; linear/asinh)
# -----------------------------
def _finite_fill(a: np.ndarray, fill: Optional[float] = None) -> np.ndarray:
    """
    Replace non-finite with mean (or given fill); returns a copy.
    """
    out = a.astype(np.float32, copy=True)
    if fill is None:
        fill = float(np.nanmean(out)) if np.isfinite(out).any() else 0.0
    m = ~np.isfinite(out)
    if m.any():
        out[m] = fill
    return out

def _ssim2d(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    """
    SSIM for 2D arrays; non-finite replaced by global mean.
    """
    a2 = _finite_fill(a)
    b2 = _finite_fill(b)
    if HAVE_SKIMAGE:
        return float(ssim_skimage(b2, a2, data_range=data_range))
    # Coarse fallback (global SSIM proxy)
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    xmu = float(np.mean(a2)); ymu = float(np.mean(b2))
    xsig2 = float(np.var(a2, ddof=1)); ysig2 = float(np.var(b2, ddof=1))
    cov = float(np.mean((a2 - xmu) * (b2 - ymu)))
    num = (2 * xmu * ymu + C1) * (2 * cov + C2)
    den = (xmu**2 + ymu**2 + C1) * (xsig2 + ysig2 + C2)
    return float(num / den) if den != 0.0 else float("nan")

def compute_psnr_linear_np(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Linear-domain PSNR after clipping to [LO, HI]; uses global DR=HI-LO.
    Shapes: (H,W) or (C,H,W) with C=1.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {gt.shape}")
    pr = _np_clip(pred); g = _np_clip(gt)
    mask = np.isfinite(pr) & np.isfinite(g)
    if not np.any(mask):
        return float("nan")
    diff = (pr - g)[mask]
    mse = float(np.mean(diff * diff, dtype=np.float64))
    if mse == 0.0:
        return float("inf")
    return 20.0 * np.log10(DR) - 10.0 * np.log10(mse)

def compute_ssim_linear_np(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Linear-domain SSIM after clipping to [LO, HI]; uses global data_range DR=HI-LO.
    Supports 2D or (C,H,W); averages channels if C>1.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {gt.shape}")
    pr = _np_clip(pred); g = _np_clip(gt)
    if g.ndim == 2:
        return _ssim2d(pr, g, DR)
    elif g.ndim == 3:
        C = g.shape[0]
        vals = []
        for c in range(C):
            vals.append(_ssim2d(pr[c], g[c], DR))
        return float(np.nanmean(np.array(vals, dtype=np.float64)))
    else:
        raise ValueError(f"Unsupported ndim for SSIM: {gt.ndim}")

def compute_psnr_asinh_unit_np(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Asinh-domain PSNR after clipping to [LO,HI] then asinh(k·x),
    followed by linear normalization to [0,1] using global LO_T, HI_T.
    Uses MAX=1.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {gt.shape}")
    pr = _np_asinh_k(_np_clip(pred)); g = _np_asinh_k(_np_clip(gt))
    pr_n = (pr - LO_T) / (DR_T + 1e-12)
    g_n  = (g  - LO_T) / (DR_T + 1e-12)
    mask = np.isfinite(pr_n) & np.isfinite(g_n)
    if not np.any(mask):
        return float("nan")
    diff = (pr_n - g_n)[mask]
    mse = float(np.mean(diff * diff, dtype=np.float64))
    if mse == 0.0:
        return float("inf")
    # MAX in [0,1] is 1
    return 10.0 * np.log10(1.0 / (mse + 1e-12))

def compute_ssim_asinh_unit_np(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Asinh-domain SSIM after clipping and asinh(k·x), then normalize to [0,1].
    data_range=1.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {gt.shape}")
    pr = _np_asinh_k(_np_clip(pred)); g = _np_asinh_k(_np_clip(gt))
    pr_n = (pr - LO_T) / (DR_T + 1e-12)
    g_n  = (g  - LO_T) / (DR_T + 1e-12)

    def _ssim_any(a: np.ndarray, b: np.ndarray) -> float:
        if a.ndim == 2:
            return _ssim2d(a, b, 1.0)
        elif a.ndim == 3:
            C = a.shape[0]
            vals = []
            for c in range(C):
                vals.append(_ssim2d(a[c], b[c], 1.0))
            return float(np.nanmean(np.array(vals, dtype=np.float64)))
        else:
            raise ValueError(f"Unsupported ndim for SSIM: {a.ndim}")

    return _ssim_any(pr_n, g_n)

# -----------------------------
#   Reusable: evaluate on loader (for training/validation)
# -----------------------------
@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: Optional[nn.Module] = None
) -> Dict[str, float]:
    """
    Evaluate on a torch DataLoader (expects (x,y,bg_mask,*_)).
    Returns dict with averaged metrics:
      {
        'psnr','ssim','psnr_asinh','ssim_asinh','loss',
        'bg_nmad_pred','bg_sigma_pred'
      }
    """
    model.eval()
    ps_list, ss_list = [], []
    ps_asinh_list, ss_asinh_list = [], []
    loss_list = []
    bg_nmad_pred_list, bg_sigma_pred_list = [], []

    for x, y, bg_mask, *_ in loader:
        x = x.to(device, non_blocking=True)          # (B,1,H,W)
        y = y.to(device, non_blocking=True)
        bg_mask = bg_mask.to(device, non_blocking=True)  # bool (B,1,H,W)

        pred = model(x)                              # (B,1,H,W)

        # Optional loss (your criterion should understand bool masks if used)
        if criterion is not None:
            loss = criterion(pred, y, bg_mask)
            loss_list.append(loss.item())

        # Linear-domain PSNR over masked pixels (background mask)
        ps_list.append(psnr(pred, y, bg_mask))

        # Convert to numpy for per-sample SSIM/ASINH + BG stats
        pred_np = pred.detach().cpu().numpy()        # (B,1,H,W)
        y_np    = y.detach().cpu().numpy()
        m_np    = bg_mask.detach().cpu().numpy().astype(bool)

        for b in range(pred_np.shape[0]):
            pr = pred_np[b]      # (1,H,W)
            gt = y_np[b]
            mb = m_np[b]         # (1,H,W) bool

            # ---- Linear domain (clipped) ----
            ps_lin = compute_psnr_linear_np(pr, gt)
            ss_lin = compute_ssim_linear_np(pr, gt)
            ps_list.append(ps_lin)
            ss_list.append(ss_lin)

            # ---- Asinh domain (k=5, clipped then asinh, normalized to [0,1]) ----
            ps_as = compute_psnr_asinh_unit_np(pr, gt)
            ss_as = compute_ssim_asinh_unit_np(pr, gt)
            ps_asinh_list.append(ps_as)
            ss_asinh_list.append(ss_as)

            # background stats on prediction (keep as raw pred; or clip if you prefer)
            bg_stats = background_stats_pred(pr, mb)
            bg_nmad_pred_list.append(bg_stats["nmad"])
            bg_sigma_pred_list.append(bg_stats["std"])

    out = {
        "psnr": float(np.nanmean(ps_list)) if ps_list else float("nan"),
        "ssim": float(np.nanmean(ss_list)) if ss_list else float("nan"),
        "psnr_asinh": float(np.nanmean(ps_asinh_list)) if ps_asinh_list else float("nan"),
        "ssim_asinh": float(np.nanmean(ss_asinh_list)) if ss_asinh_list else float("nan"),
        "loss": float(np.mean(loss_list)) if loss_list else float("nan"),
        "bg_nmad_pred": float(np.nanmean(bg_nmad_pred_list)) if bg_nmad_pred_list else float("nan"),
        "bg_sigma_pred": float(np.nanmean(bg_sigma_pred_list)) if bg_sigma_pred_list else float("nan"),
    }
    return out

# -----------------------------
#   Helper: load checkpoint & evaluate (for CLI and train.py)
# -----------------------------
def evaluate_checkpoint(
    root: Path | str,
    split_json: str,
    checkpoint_dir: Path | str,
    device: torch.device,
    batch_size: int = 1,
    num_workers: int = 4,
    verbose: bool = False,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    Load test split, build model from checkpoint, and evaluate with evaluate_loader().
    Returns metrics dict including bg_nmad_pred and bg_sigma_pred.
    """
    root = Path(root).expanduser().resolve()
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    ckpt_path = checkpoint_dir / "final.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Dataset / Loader
    ds = FitsPairDataset(root=str(root), split_json=split_json, split="test")
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    # Model
    model = UNet(in_nc=1, out_nc=1, nf=32).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    torch.set_grad_enabled(False)

    # Evaluate
    metrics = evaluate_loader(model, loader, device, criterion=criterion)

    if verbose:
        print("-" * 70)
        print(f"[EVAL @ {ckpt_path}] "
              f"PSNR(linear, clipped)={metrics['psnr']:.4f} dB, "
              f"SSIM(linear, clipped)={metrics['ssim']:.6f} | "
              f"ASINH-PSNR(k={K_ASINH})={metrics['psnr_asinh']:.4f} dB, "
              f"ASINH-SSIM(k={K_ASINH})={metrics['ssim_asinh']:.6f} | "
              f"LOSS={metrics['loss']:.6g} | "
              f"BG: NMAD(pred)={metrics['bg_nmad_pred']:.6g}, σ(pred)={metrics['bg_sigma_pred']:.6g}")
    return metrics

# -----------------------------
#   CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate checkpoint on test split (npy dataset).")
    ap.add_argument("-d", "--data", required=True, help="Root dir containing train_test_split.json and subfolders.")
    ap.add_argument("-c", "--checkpoint", required=True, help="Checkpoint dir containing final.pth (REQUIRED).")
    ap.add_argument("--json", default="train_test_split.json", help="Split JSON filename.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu.")
    ap.add_argument("--batch", type=int, default=1, help="Batch size.")
    ap.add_argument("--workers", type=int, default=4, help="Number of DataLoader workers.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.data).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Input root not a directory: {root}")

    device = torch.device(args.device)

    # Evaluate checkpoint on test split and print a summary
    _ = evaluate_checkpoint(
        root=root,
        split_json=args.json,
        checkpoint_dir=args.checkpoint,
        device=device,
        batch_size=args.batch,
        num_workers=args.workers,
        verbose=True or args.verbose,
        criterion=None,
    )

if __name__ == "__main__":
    main()