#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import random
import sys

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.ndimage import gaussian_filter, zoom

# --------------------------- Parameters ---------------------------
SIGMA_CLIP = 4.0
SKY_BIN_PX = 1024                # must divide image size (2048×2048 is divisible)
SKY_SMOOTH_SIGMA = SKY_BIN_PX / 4.0

# --------------------------- Utility functions ---------------------------
def read_first_image(fp: Path) -> np.ndarray:
    """Read the first image array from a FITS/FITS.FZ file (first HDU containing data)."""
    with fits.open(fp) as hdul:
        hdu = next((h for h in hdul if hasattr(h, "data") and isinstance(h.data, np.ndarray)), None)
        if hdu is None:
            raise ValueError(f"No image array in {fp}")
        arr = hdu.data.astype(np.float64, copy=False)
    arr[~np.isfinite(arr)] = np.nan
    return arr

def norm_by_median(img: np.ndarray) -> np.ndarray:
    med = np.nanmedian(img)
    if not np.isfinite(med) or med == 0:
        raise ValueError("Median invalid/zero.")
    out = img / med
    out[~np.isfinite(out)] = 1.0
    return out

def estimate_global_sky_logbin(img_norm: np.ndarray, bin_px: int, smooth_sigma: float) -> np.ndarray:
    """
    Log-domain: coarse block median -> linear interpolation (reflect boundary) -> wide Gaussian.
    Returns a very-low-frequency global sky estimate (multiplicative factor).
    """
    h, w = img_norm.shape
    assert h % bin_px == 0 and w % bin_px == 0, "SKY_BIN_PX must divide image size"
    eps = 1e-6
    log_im = np.log(np.clip(img_norm, eps, None))

    Hc, Wc = h // bin_px, w // bin_px
    B = log_im.reshape(Hc, bin_px, Wc, bin_px)
    coarse = np.nanmedian(np.nanmedian(B, axis=1), axis=2)     # (Hc, Wc)

    coarse_up = zoom(coarse, (bin_px, bin_px), order=1, mode="reflect")
    if smooth_sigma and smooth_sigma > 0:
        coarse_up = gaussian_filter(coarse_up, sigma=smooth_sigma, mode="nearest")

    sky = np.exp(coarse_up)
    sky /= np.nanmedian(sky)
    sky[~np.isfinite(sky)] = 1.0
    return sky

def choose_skyflat_in_subdir(subdir: Path) -> Path | None:
    """Pick one SKYFLAT_*.fits.fz under subdir/calib (random choice if multiple exist)."""
    calib = subdir / "calib"
    if not calib.is_dir():
        return None
    cands = sorted([p for p in calib.glob("SKYFLAT_*.fits.fz") if p.is_file()])
    if not cands:
        return None
    return random.choice(cands)

def write_fits_fz_float32(data: np.ndarray, out_path: Path):
    """
    Write a compressed .fits.fz (RICE_1). float32 is sufficient for a multiplicative PRNU map.
    """
    data32 = data.astype("float32", copy=False)
    hdu = fits.CompImageHDU(data=data32, compression_type="RICE_1")
    hdu.writeto(out_path, overwrite=True)


def main():
    ap = argparse.ArgumentParser(description="Estimate PRNU from randomly sampled skyflats in subdirs.")
    ap.add_argument("-i", "--input", required=True, type=Path, help="Top-level directory (non-recursive). Each subdir must contain calib/ with SKYFLAT_*.fits.fz files.")
    ap.add_argument("-n", "--number", default=50, type=int, help="Number of subdirectories to randomly sample (default: 50)")
    ap.add_argument("-o", "--output", required=True, type=Path, help="Output directory")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = ap.parse_args()

    root: Path = args.input
    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)

    if not root.is_dir():
        print(f"[ERROR] Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    # 1) Enumerate top-level subdirs, collect one SKYFLAT per subdir
    subdirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    picks = []
    for sd in subdirs:
        f = choose_skyflat_in_subdir(sd)
        if f is not None:
            picks.append((sd, f))

    if not picks:
        print("[ERROR] No subdir with calib/SKYFLAT_*.fits.fz found.", file=sys.stderr)
        sys.exit(2)

    # 2) Randomly sample n subdirs
    n = max(1, min(args.number, len(picks)))
    random.shuffle(picks)
    chosen = picks[:n]

    print(f"[INFO] Found {len(picks)} usable subdirs, sampling n={n}.")
    for sd, f in chosen:
        print(f"  - {sd.name}: {f.name}")

    # 3) Per frame: normalise -> estimate global sky -> remove sky -> collect high-pass
    hp_list = []
    img_shape = None
    for sd, f in chosen:
        img = read_first_image(f)
        if img_shape is None:
            img_shape = img.shape
        elif img.shape != img_shape:
            print(f"[WARN] Shape mismatch in {f} ({img.shape} vs {img_shape}), skipping.", file=sys.stderr)
            continue

        img_norm = norm_by_median(img)
        sky = estimate_global_sky_logbin(img_norm, bin_px=SKY_BIN_PX, smooth_sigma=SKY_SMOOTH_SIGMA)
        high = img_norm / sky
        high /= np.nanmedian(high)
        hp_list.append(high)

    if len(hp_list) == 0:
        print("[ERROR] No valid frames to stack.", file=sys.stderr)
        sys.exit(3)

    # 4) Stack frames to derive PRNU (high-frequency component)
    stack = np.dstack(hp_list)                                  # [H, W, N]
    clipped = sigma_clip(stack, sigma=SIGMA_CLIP, axis=2, masked=True)
    prnu = np.nanmedian(clipped, axis=2)
    prnu /= np.nanmedian(prnu)

    # 5) Save result (compressed .fits.fz)
    (out_dir / "prnu").mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / "prnu" /"super_prnu.fits.fz"
    write_fits_fz_float32(prnu, out_path)

    # Also save the list of sampled files
    used_list = out_dir / "prnu" / "used_skyflats.txt"
    with used_list.open("w", encoding="utf-8") as f:
        for sd, fp in chosen:
            f.write(str(fp) + "\n")

    print(f"[OK] PRNU saved: {out_path}")
    print(f"[OK] Used file list: {used_list}")

if __name__ == "__main__":
    main()
