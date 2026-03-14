#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

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


# =========================
# Mock-PSF flux metrics utils
# =========================

EPS = 1e-12


def np_to_py(x):
    """Convert numpy scalars/arrays recursively for json dump."""
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: np_to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [np_to_py(v) for v in x]
    return x


def strip_input_suffix_to_base_name(os_path: Path, input_suffix: str) -> str:
    """
    Example:
      file: xxx_os.npy with input_suffix='os' -> base_name='xxx'
    """
    suffix = f"_{input_suffix}.npy"
    name = os_path.name
    if not name.endswith(suffix):
        raise ValueError(f"os_path does not end with expected suffix '{suffix}': {os_path}")
    return name[:-len(suffix)]


def sidecar_paths_from_os_path(os_path: Path, input_suffix: str, label_suffix: str = "calib") -> Dict[str, Path]:
    base_name = strip_input_suffix_to_base_name(os_path, input_suffix=input_suffix)
    d = os_path.parent
    return {
        "mock_gt": d / f"{base_name}_mock_psf_gt.npy",
        "mock_mask": d / f"{base_name}_mock_psf_mask.npy",
        "calib": d / f"{base_name}_{label_suffix}.npy",
        "os_mock": d / f"{base_name}_os_mock.npy",  # optional, not required in current metric path
    }


def robust_sigma_from_mad(v: np.ndarray) -> float:
    if v.size == 0:
        return np.nan
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if np.isfinite(mad) and mad > 0:
        return float(1.4826 * mad)
    s = float(np.std(v))
    return s if np.isfinite(s) else np.nan


def sigma_clip_1d(v: np.ndarray, nsig: float = 3.0, max_iter: int = 3) -> np.ndarray:
    """
    Simple sigma clipping (1D).
    """
    x = np.asarray(v, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x
    cur = x
    for _ in range(max_iter):
        med = np.median(cur)
        sig = robust_sigma_from_mad(cur)
        if (not np.isfinite(sig)) or sig <= 0:
            break
        keep = np.abs(cur - med) <= (nsig * sig)
        if keep.all():
            break
        nxt = cur[keep]
        if nxt.size == 0:
            break
        cur = nxt
    return cur


def estimate_local_bg_from_calib_mask_pixels(calib_vals: np.ndarray, clip_sigma: float = 3.0) -> Dict[str, float]:
    """
    Estimate local background from calib_ours values within one PSF mask component.
    User-requested logic: use this PSF mask region and simple outlier removal.
    """
    raw = np.asarray(calib_vals, dtype=np.float64)
    raw = raw[np.isfinite(raw)]
    if raw.size == 0:
        return {
            "bg_est": np.nan,
            "bg_sigma": np.nan,
            "n_bg_pix": 0,
            "n_bg_kept": 0,
        }

    clipped = sigma_clip_1d(raw, nsig=clip_sigma, max_iter=3)
    if clipped.size == 0:
        clipped = raw

    bg_est = float(np.median(clipped))            # robust center
    bg_sigma = float(robust_sigma_from_mad(clipped))  # robust sigma
    if (not np.isfinite(bg_sigma)) or bg_sigma < 0:
        bg_sigma = float(np.std(clipped)) if clipped.size > 1 else 0.0

    return {
        "bg_est": bg_est,
        "bg_sigma": bg_sigma,
        "n_bg_pix": int(raw.size),
        "n_bg_kept": int(clipped.size),
    }


def connected_components_coords(mask: np.ndarray, connectivity: int = 8) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Pure numpy/Python connected components for binary mask.
    Returns a list of (ys, xs) index arrays, one per component.
    """
    m = np.asarray(mask).astype(bool)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2D, got {m.shape}")
    H, W = m.shape
    visited = np.zeros((H, W), dtype=np.uint8)
    comps: List[Tuple[np.ndarray, np.ndarray]] = []

    if connectivity == 8:
        nbrs = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dy == 0 and dx == 0)]
    elif connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    ys0, xs0 = np.where(m)
    for sy, sx in zip(ys0.tolist(), xs0.tolist()):
        if visited[sy, sx]:
            continue

        stack = [(sy, sx)]
        visited[sy, sx] = 1
        cy_list: List[int] = []
        cx_list: List[int] = []

        while stack:
            y, x = stack.pop()
            cy_list.append(y)
            cx_list.append(x)

            for dy, dx in nbrs:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= H or nx < 0 or nx >= W:
                    continue
                if visited[ny, nx]:
                    continue
                if not m[ny, nx]:
                    continue
                visited[ny, nx] = 1
                stack.append((ny, nx))

        comps.append((np.asarray(cy_list, dtype=np.int32), np.asarray(cx_list, dtype=np.int32)))

    return comps


def safe_mag_from_flux(flux: float) -> float:
    if not np.isfinite(flux) or flux <= 0:
        return np.nan
    return float(-2.5 * np.log10(flux + EPS))


def nmad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(x * x)))


def summarize_source_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize per-source flux metrics across a list of records.
    """
    if len(records) == 0:
        return {
            "n_sources_total": 0,
            "n_sources_valid_flux": 0,
            "n_sources_valid_mag": 0,
        }

    gt_flux = np.asarray([r.get("gt_flux", np.nan) for r in records], dtype=np.float64)
    pred_flux = np.asarray([r.get("pred_flux_bgsub", np.nan) for r in records], dtype=np.float64)
    rel_err = np.asarray([r.get("rel_flux_error", np.nan) for r in records], dtype=np.float64)
    dmag = np.asarray([r.get("delta_mag", np.nan) for r in records], dtype=np.float64)
    flux_snr_lin = np.asarray([r.get("flux_snr_linear", np.nan) for r in records], dtype=np.float64)
    flux_snr_db = np.asarray([r.get("flux_snr_db", np.nan) for r in records], dtype=np.float64)
    inj_snr = np.asarray([r.get("inj_flux_snr_linear", np.nan) for r in records], dtype=np.float64)

    valid_flux = np.isfinite(gt_flux) & np.isfinite(pred_flux) & (gt_flux > 0)
    valid_rel = np.isfinite(rel_err)
    valid_mag = np.isfinite(dmag)
    valid_flux_snr_lin = np.isfinite(flux_snr_lin)
    valid_flux_snr_db = np.isfinite(flux_snr_db)
    valid_inj_snr = np.isfinite(inj_snr)

    out: Dict[str, Any] = {
        "n_sources_total": int(len(records)),
        "n_sources_valid_flux": int(valid_flux.sum()),
        "n_sources_valid_rel_flux_error": int(valid_rel.sum()),
        "n_sources_valid_mag": int(valid_mag.sum()),
        "n_sources_valid_flux_snr": int(valid_flux_snr_lin.sum()),
        "n_sources_valid_inj_flux_snr": int(valid_inj_snr.sum()),
    }

    if valid_rel.any():
        re = rel_err[valid_rel]
        out.update({
            "relative_flux_error_mean": float(np.mean(re)),
            "relative_flux_error_median": float(np.median(re)),
            "relative_flux_error_std": float(np.std(re)),
            "relative_flux_error_rms": float(np.sqrt(np.mean(re * re))),
            "relative_flux_error_nmad": nmad(re),
            "bias_rel_flux_error_median": float(np.median(re)),  # alias for clarity
            "scatter_rel_flux_error_rms": float(np.sqrt(np.mean(re * re))),
            "scatter_rel_flux_error_nmad": nmad(re),
        })

    if valid_mag.any():
        dm = dmag[valid_mag]
        out.update({
            "delta_mag_mean": float(np.mean(dm)),
            "delta_mag_median": float(np.median(dm)),
            "delta_mag_std": float(np.std(dm)),
            "delta_mag_rms": float(np.sqrt(np.mean(dm * dm))),
            "delta_mag_nmad": nmad(dm),
            "bias_delta_mag_median": float(np.median(dm)),
            "scatter_delta_mag_rms": float(np.sqrt(np.mean(dm * dm))),
            "scatter_delta_mag_nmad": nmad(dm),
        })

    if valid_flux_snr_lin.any():
        fs = flux_snr_lin[valid_flux_snr_lin]
        out.update({
            "flux_snr_linear_mean": float(np.mean(fs)),
            "flux_snr_linear_median": float(np.median(fs)),
        })

    if valid_flux_snr_db.any():
        fsd = flux_snr_db[valid_flux_snr_db]
        out.update({
            "flux_snr_db_mean": float(np.mean(fsd)),
            "flux_snr_db_median": float(np.median(fsd)),
        })

    # PSNR-style global flux-SNR (over per-source flux values), to match PSNR scale:
    # flux_snr_db_global = 20 log10( max(F_gt) / RMSE(F_pred - F_gt) )
    if valid_flux.any():
        gt = gt_flux[valid_flux]
        pd = pred_flux[valid_flux]
        ferr = pd - gt
        rmse_flux = float(np.sqrt(np.mean(ferr * ferr)))
        peak_flux = float(np.max(gt))
        flux_snr_global_linear = float(peak_flux / (rmse_flux + EPS))
        flux_snr_global_db = float(20.0 * np.log10(flux_snr_global_linear + EPS))
        out.update({
            "flux_rmse": rmse_flux,
            "flux_peak_reference": peak_flux,
            "flux_snr_global_linear_psnr_style": flux_snr_global_linear,
            "flux_snr_global_db_psnr_style": flux_snr_global_db,  # preferred primary (PSNR-like scale)
            "flux_snr_primary": flux_snr_global_db,
            "flux_snr_primary_unit": "dB",
        })

    return out


def build_binned_rms_curve(
    x: np.ndarray,
    y: np.ndarray,
    nbins: int = 12,
    x_name: str = "gt_flux",
    y_name: str = "rms_relative_flux_error",
    log_x: bool = True,
    min_count_per_bin: int = 3,
) -> Dict[str, Any]:
    """
    Build a curve with bin centers and y=RMS(y_in_bin). Also returns counts and edges.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if log_x:
        valid_pos = x > 0
        x = x[valid_pos]
        y = y[valid_pos]

    if x.size == 0:
        return {
            "x_name": x_name,
            "y_name": y_name,
            "binning": {"nbins": int(nbins), "log_x": bool(log_x), "min_count_per_bin": int(min_count_per_bin)},
            "x_values": [],
            "y_values": [],
            "counts": [],
            "bin_edges": [],
            "element_to_x_values_note": "y_values[i] corresponds to x_values[i] (bin center).",
        }

    x_min, x_max = float(np.min(x)), float(np.max(x))
    if x_max <= x_min:
        # Degenerate case: single bin
        yv = rms(y) if x.size >= min_count_per_bin else np.nan
        return {
            "x_name": x_name,
            "y_name": y_name,
            "binning": {"nbins": 1, "log_x": bool(log_x), "min_count_per_bin": int(min_count_per_bin)},
            "x_values": [x_min],
            "y_values": [yv],
            "counts": [int(x.size)],
            "bin_edges": [x_min, x_max],
            "element_to_x_values_note": "y_values[i] corresponds to x_values[i] (bin center).",
        }

    if log_x:
        edges = np.logspace(np.log10(x_min), np.log10(x_max), int(nbins) + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])  # geometric centers
    else:
        edges = np.linspace(x_min, x_max, int(nbins) + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

    y_values: List[float] = []
    counts: List[int] = []
    x_values: List[float] = []

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            m = (x >= lo) & (x <= hi)
        else:
            m = (x >= lo) & (x < hi)

        c = int(np.sum(m))
        counts.append(c)
        x_values.append(float(centers[i]))

        if c >= int(min_count_per_bin):
            y_values.append(float(rms(y[m])))
        else:
            y_values.append(np.nan)

    return {
        "x_name": x_name,
        "y_name": y_name,
        "binning": {
            "nbins": int(nbins),
            "log_x": bool(log_x),
            "min_count_per_bin": int(min_count_per_bin),
        },
        "x_values": x_values,  # <-- each element's x-axis value
        "y_values": y_values,
        "counts": counts,
        "bin_edges": [float(v) for v in edges.tolist()],
        "element_to_x_values_note": "y_values[i] corresponds to x_values[i] (bin center).",
    }


def evaluate_mock_flux_metrics_for_pred(
    pred_img: np.ndarray,
    calib_img: np.ndarray,
    mock_gt_img: np.ndarray,
    mock_mask_img: np.ndarray,
    sample_id: str,
    clip_sigma: float = 3.0,
    connectivity: int = 8,
) -> Dict[str, Any]:
    """
    Evaluate per-PSF flux metrics for one prediction image.

    Assumptions (matched to user's generation/update):
    - mock_psf_gt is source-only (no background)
    - prediction is a restored full image (with background)
    - calib_ours contains only background in the mock PSF mask region
    """
    pred = np.asarray(pred_img, dtype=np.float32)
    calib = np.asarray(calib_img, dtype=np.float32)
    gt = np.asarray(mock_gt_img, dtype=np.float32)
    mm = (np.asarray(mock_mask_img) > 0)

    if pred.ndim != 2 or calib.ndim != 2 or gt.ndim != 2 or mm.ndim != 2:
        raise ValueError(f"All inputs must be 2D. got pred={pred.shape}, calib={calib.shape}, gt={gt.shape}, mask={mm.shape}")
    if not (pred.shape == calib.shape == gt.shape == mm.shape):
        raise ValueError(f"Shape mismatch: pred={pred.shape}, calib={calib.shape}, gt={gt.shape}, mask={mm.shape}")

    comps = connected_components_coords(mm, connectivity=connectivity)

    source_records: List[Dict[str, Any]] = []

    for comp_idx, (ys, xs) in enumerate(comps):
        if ys.size == 0:
            continue

        pred_vals = pred[ys, xs].astype(np.float64)
        calib_vals = calib[ys, xs].astype(np.float64)
        gt_vals = gt[ys, xs].astype(np.float64)

        bg_stats = estimate_local_bg_from_calib_mask_pixels(calib_vals, clip_sigma=clip_sigma)
        bg_est = bg_stats["bg_est"]
        bg_sigma = bg_stats["bg_sigma"]

        n_pix = int(ys.size)

        gt_flux = float(np.sum(gt_vals[np.isfinite(gt_vals)]))
        pred_flux_raw = float(np.sum(pred_vals[np.isfinite(pred_vals)]))
        pred_flux_bgsub = float(np.sum((pred_vals - bg_est)[np.isfinite(pred_vals)])) if np.isfinite(bg_est) else np.nan

        flux_err = float(pred_flux_bgsub - gt_flux) if (np.isfinite(pred_flux_bgsub) and np.isfinite(gt_flux)) else np.nan
        rel_flux_err = float(flux_err / (gt_flux + EPS)) if (np.isfinite(flux_err) and gt_flux > 0) else np.nan

        mag_gt = safe_mag_from_flux(gt_flux)
        mag_pred = safe_mag_from_flux(pred_flux_bgsub)
        delta_mag = float(mag_pred - mag_gt) if (np.isfinite(mag_gt) and np.isfinite(mag_pred)) else np.nan

        # "Injected" flux SNR estimated from GT flux / estimated flux noise from local background
        sigma_flux_est = float(bg_sigma * np.sqrt(n_pix)) if np.isfinite(bg_sigma) else np.nan
        inj_flux_snr_linear = float(gt_flux / (sigma_flux_est + EPS)) if (np.isfinite(sigma_flux_est) and gt_flux > 0) else np.nan
        inj_flux_snr_db = float(20.0 * np.log10(inj_flux_snr_linear + EPS)) if np.isfinite(inj_flux_snr_linear) else np.nan

        # Reconstruction flux-SNR using GT vs prediction flux error (PSNR-like local interpretation)
        # Keep both linear and dB; dB is the primary one to align with PSNR scale.
        flux_snr_linear = float(gt_flux / (abs(flux_err) + EPS)) if (np.isfinite(flux_err) and gt_flux > 0) else np.nan
        flux_snr_db = float(20.0 * np.log10(flux_snr_linear + EPS)) if np.isfinite(flux_snr_linear) else np.nan

        # Brightness proxies (for optional plotting/analysis)
        gt_peak = float(np.nanmax(gt_vals)) if gt_vals.size > 0 else np.nan
        pred_peak = float(np.nanmax(pred_vals - bg_est)) if (pred_vals.size > 0 and np.isfinite(bg_est)) else np.nan

        cy = float(np.mean(ys))
        cx = float(np.mean(xs))

        source_records.append({
            "sample_id": sample_id,
            "component_index": int(comp_idx),
            "n_pix": n_pix,
            "center_yx_est": [cy, cx],

            "bg_est_from_calib_mask": bg_est,
            "bg_sigma_from_calib_mask": bg_sigma,
            "n_bg_pix": int(bg_stats["n_bg_pix"]),
            "n_bg_kept_after_clip": int(bg_stats["n_bg_kept"]),

            "gt_flux": gt_flux,
            "pred_flux_raw_sum": pred_flux_raw,
            "pred_flux_bgsub": pred_flux_bgsub,
            "flux_error": flux_err,
            "rel_flux_error": rel_flux_err,

            "gt_mag_inst": mag_gt,
            "pred_mag_inst": mag_pred,
            "delta_mag": delta_mag,

            "inj_flux_snr_linear": inj_flux_snr_linear,
            "inj_flux_snr_db": inj_flux_snr_db,

            # Primary "FLUX-SNR" reported per-source (PSNR-like scale compatible with dB PSNR)
            "flux_snr_linear": flux_snr_linear,
            "flux_snr_db": flux_snr_db,

            "gt_peak_in_mask": gt_peak,
            "pred_peak_bgsub_in_mask": pred_peak,
        })

    per_image_summary = summarize_source_records(source_records)
    return {
        "sample_id": sample_id,
        "n_components_found": int(len(source_records)),
        "per_image_summary": per_image_summary,
        "source_records": source_records,
    }


def group_records_by_sample(all_source_records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    d: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_source_records:
        sid = str(r.get("sample_id", "unknown"))
        d.setdefault(sid, []).append(r)
    return d


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Run inference on first N samples of train/test, save predictions, and compute mock-PSF flux metrics JSON.")
    ap.add_argument("-c", "--checkpoint", required=True, help="Path to checkpoint dir or final.pth")
    ap.add_argument("-d", "--data", required=True, help="Root dir containing train_test_split.json and subfolders")
    ap.add_argument("--arch", default="unet", choices=["unet", "pmn_unet"], help="model architecture")
    ap.add_argument("-x", "--input-suffix", default="os_mock", help="input npy suffix")
    ap.add_argument("-y", "--label-suffix", default="calib", help="label npy suffix (also used for local bg estimation)")
    split = ap.add_mutually_exclusive_group()
    split.add_argument("--train", action="store_true", help="Use train split")
    split.add_argument("--test", action="store_true", help="Use test split (default)")
    ap.add_argument("--json", default="train_test_split.json", help="Split JSON filename")
    ap.add_argument("-o", "--output", default="", help="Output directory; default: <ckpt_dir>/mock_results/<split>")
    ap.add_argument("--batch", type=int, default=1, help="Batch size for inference")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="cuda or cpu")
    # ap.add_argument("--eval", action="store_true", help="Also run metrics on the same subset")

    # Added: mock-PSF flux metrics controls
    ap.add_argument("--output-name", default="mock_results", help="Output folder name")
    ap.add_argument("--mock-metrics-json-name", default="mock_flux_metrics.json", help="Output JSON filename for mock-PSF flux metrics")
    ap.add_argument("--bg-clip-sigma", type=float, default=3.0, help="Sigma for simple outlier removal on calib_ours values inside each PSF mask")
    ap.add_argument("--cc-connectivity", type=int, default=8, choices=[4, 8], help="Connected component connectivity for mock_psf_mask")
    ap.add_argument("--curve-bins", type=int, default=12, help="Number of bins for brightness-vs-RMS curve")
    ap.add_argument("--curve-log-x", action="store_true", help="Use log x-axis bins for brightness-vs-RMS curve (GT flux)")
    ap.add_argument("--curve-min-count", type=int, default=3, help="Minimum sources per bin to report curve value")
    ap.add_argument("--save-per-source-records", action="store_true", help="Save detailed per-source records in JSON (can be large)")

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
        out_dir = ckpt_dir / args.output_name / split_name
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

    # Mock-PSF metrics accumulators
    all_source_records: List[Dict[str, Any]] = []
    per_image_summaries: List[Dict[str, Any]] = []
    missing_sidecar_errors: List[str] = []

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
            os_path_b = Path(os_paths[b])

            # Match eval_ours.py expectation: <subdir>_pred.fits.fz
            out_name = f"{os_path_b.parent.name}_pred_mock.fits.fz"
            out_fp = out_dir / out_name

            # save_comp_fits_fz_single(out_fp, img2d)
            # print(f"[OK] {out_fp}")
            saved += 1

            # ---------- mock-PSF flux metrics ----------
            side = sidecar_paths_from_os_path(
                os_path_b,
                input_suffix=args.input_suffix
            )

            # Check sidecar files
            for k in ("mock_gt", "mock_mask", "calib"):
                if not side[k].is_file():
                    raise FileNotFoundError(f"Missing sidecar [{k}] for mock metrics: {side[k]}")

            mock_gt = np.load(side["mock_gt"]).astype(np.float32)
            mock_mask = np.load(side["mock_mask"])
            calib_img = np.load(side["calib"]).astype(np.float32)

            sample_id = os_path_b.parent.name

            res = evaluate_mock_flux_metrics_for_pred(
                pred_img=img2d,
                calib_img=calib_img,
                mock_gt_img=mock_gt,
                mock_mask_img=mock_mask,
                sample_id=sample_id,
                clip_sigma=float(args.bg_clip_sigma),
                connectivity=int(args.cc_connectivity),
            )

            # per_image_summaries.append({
            #     "sample_id": sample_id,
            #     "n_components_found": int(res["n_components_found"]),
            #     "per_image_summary": res["per_image_summary"],
            #     "files": {
            #         "os_path": str(os_path_b),
            #         "pred_fits_fz": str(out_fp),
            #         "mock_gt_npy": str(side["mock_gt"]),
            #         "mock_mask_npy": str(side["mock_mask"]),
            #         "calib_npy": str(side["calib"]),
            #     }
            # })

            all_source_records.extend(res["source_records"])


    print(f"[DONE] saved {saved} files to {out_dir}")

    global_summary = summarize_source_records(all_source_records)

    # Build brightness-vs-RMS curve (x = GT flux, y = RMS(relative flux error))
    gt_flux_all = np.asarray([r.get("gt_flux", np.nan) for r in all_source_records], dtype=np.float64)
    rel_err_all = np.asarray([r.get("rel_flux_error", np.nan) for r in all_source_records], dtype=np.float64)

    curve_brightness_rms = build_binned_rms_curve(
        x=gt_flux_all,
        y=rel_err_all,
        nbins=int(args.curve_bins),
        x_name="gt_flux_brightness",
        y_name="rms_relative_flux_error",
        log_x=bool(args.curve_log_x),
        min_count_per_bin=int(args.curve_min_count),
    )

    # Optional helpful curve: x = injected flux SNR, y = RMS(relative flux error)
    inj_snr_all = np.asarray([r.get("inj_flux_snr_linear", np.nan) for r in all_source_records], dtype=np.float64)
    curve_inj_snr_rms = build_binned_rms_curve(
        x=inj_snr_all,
        y=rel_err_all,
        nbins=int(args.curve_bins),
        x_name="injected_flux_snr_linear",
        y_name="rms_relative_flux_error",
        log_x=True,
        min_count_per_bin=int(args.curve_min_count),
    )

    # Per-sample summaries regroup (already have per_image_summaries, but this ensures consistency)
    grouped = group_records_by_sample(all_source_records)
    per_sample_summary_compact = []
    for sid in sorted(grouped.keys()):
        per_sample_summary_compact.append({
            "sample_id": sid,
            "summary": summarize_source_records(grouped[sid]),
        })

    json_out = {
        "meta": {
            "split": split_name,
            "num_saved_predictions": int(saved),
            "checkpoint": str(ckpt_path),
            "root": str(root),
            "arch": str(args.arch),
            "input_suffix": str(args.input_suffix),
            "label_suffix_for_bg_estimation": str(args.label_suffix),
            "mock_metrics_definition": {
                "prediction_type": "restored full image (with background)",
                "mock_gt_type": "source-only (no background)",
                "background_estimation": "sigma-clipped median of calib_ours values inside each connected component of mock_psf_mask",
                "pred_flux": "sum(prediction_in_component - estimated_local_background)",
                "gt_flux": "sum(mock_psf_gt in component)",
                "relative_flux_error": "(pred_flux_bgsub - gt_flux) / gt_flux",
                "delta_mag": "(-2.5*log10(pred_flux_bgsub)) - (-2.5*log10(gt_flux)) for positive flux only",
                "flux_snr_linear": "gt_flux / abs(pred_flux_bgsub - gt_flux)",
                "flux_snr_db": "20*log10(flux_snr_linear), used as primary to be PSNR-like in scale",
                "inj_flux_snr_linear": "gt_flux / (bg_sigma * sqrt(n_pix_in_component)) from calib_ours within same component",
            },
            "curve_element_mapping_note": "For any curve, y_values[i] corresponds to x_values[i] (the bin center).",
        },
        "summary_global": global_summary,
        "curve_brightness_vs_rms_rel_flux_error": curve_brightness_rms,
        "curve_injected_snr_vs_rms_rel_flux_error": curve_inj_snr_rms,
        "per_image_summaries": per_image_summaries,
        "per_sample_summary_compact": per_sample_summary_compact,
        "mock_metrics_errors": missing_sidecar_errors,
    }

    if args.save_per_source_records:
        json_out["per_source_records"] = all_source_records

    json_fp = out_dir.parent / args.mock_metrics_json_name
    with open(json_fp, "w", encoding="utf-8") as f:
        json.dump(np_to_py(json_out), f, ensure_ascii=False, indent=2)

    print(f"[MOCK_METRICS_JSON] {json_fp}")
    print("[MOCK_METRICS] curve_brightness_vs_rms_rel_flux_error:")
    print("  x_values[i] = GT flux bin center")
    print("  y_values[i] = RMS(relative flux error) in that bin")
    print("  counts[i]   = number of PSFs in that bin")


if __name__ == "__main__":
    main()