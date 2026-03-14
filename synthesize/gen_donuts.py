#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Randomly sample 50 subdirectories, read SKYFLAT*.fits.fz from each calib/ folder,
apply PRNU from the specified base/prnu/syth_prnu.fits.fz, subtract sky to extract donuts,
and save each donut frame under $base/donuts/ as .fits.fz.
"""

from pathlib import Path
import argparse
import random
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, zoom

# ------------------- Parameters (keep consistent with PRNU synthesis) ------------------
SKY_BIN_PX = 256
SKY_SMOOTH_SIGMA = SKY_BIN_PX / 4.0


def read_first_image(fp: Path) -> np.ndarray:
    with fits.open(fp) as hdul:
        hdu = next((h for h in hdul if hasattr(h, "data") and isinstance(h.data, np.ndarray)), None)
        if hdu is None:
            raise ValueError(f"No image array in {fp}")
        arr = np.asarray(hdu.data, dtype=np.float64)
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
    """Log-domain: coarse block median -> upsampling (reflect boundary) -> wide Gaussian -> global sky (multiplicative)."""
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


def donut_auto_flatten(donut_raw: np.ndarray,
                       frac=1/32,     # background scale = short_side * frac; default ~64px for 2048px images
                       gain=1.8,      # ring amplification factor; tune between 1.5 and 2.5
                       post_sigma=0.8 # light post-smoothing; set to 0 to disable
                       ) -> np.ndarray:
    """
    Input:  donut_raw = (master/Sky)/PRNU  (multiplicative, ring < 1)
    Output: donut_clean (multiplicative, ring < 1, background ≈ 1; no mask/threshold/radius)
    """
    eps = 1e-6
    H, W = donut_raw.shape
    L = np.log(np.clip(donut_raw, eps, None))              # multiplicative -> additive

    # Adaptive large-scale background: sigma = frac * min(H, W)
    sigma = max(3.0, float(min(H, W)) * float(frac))
    bg = gaussian_filter(L, sigma=sigma, mode="nearest")   # smooth to estimate background

    res = L - bg                                           # local high-pass; rings go negative
    res -= np.nanmedian(res)                               # remove bias
    y = np.minimum(res, 0.0) * float(gain)                 # keep only negative component and amplify

    if post_sigma > 0:
        y = gaussian_filter(y, sigma=post_sigma, mode="nearest")

    donut_clean = np.exp(y)
    donut_clean /= np.nanmedian(donut_clean)
    donut_clean[~np.isfinite(donut_clean)] = 1.0
    return donut_clean


def write_fits_fz_float32(data: np.ndarray, out_path: Path, header: dict | None = None):
    """
    Write image as .fits.fz (tile-compressed, RICE_1). Data is stored in hdul[1] (extension HDU);
    hdul[0] is an empty PrimaryHDU.
    """
    data32 = np.asarray(data, dtype=np.float32)

    primary = fits.PrimaryHDU()  # empty primary unit (hdul[0])
    ext = fits.CompImageHDU(data=data32, compression_type="RICE_1")  # compressed extension (hdul[1])
    ext.header["EXTNAME"] = ext.header.get("EXTNAME", "IMAGE")

    if header:
        for k, v in header.items():
            try:
                ext.header[k] = v
            except Exception:
                pass

    hdul = fits.HDUList([primary, ext])
    hdul.writeto(out_path, overwrite=True)


def process_one_skyflat(skyflat_fp: Path, prnu: np.ndarray) -> np.ndarray:
    # Read & normalise
    master = norm_by_median(read_first_image(skyflat_fp))

    # Remove global sky
    sky = estimate_global_sky_logbin(master, SKY_BIN_PX, SKY_SMOOTH_SIGMA)
    master_high = master / sky
    master_high /= np.nanmedian(master_high)

    if master_high.shape != prnu.shape:
        raise ValueError(f"Shape mismatch: master {master_high.shape} vs prnu {prnu.shape}")

    # Multiplicative decomposition: Donut ≈ (Master/Sky)/PRNU
    donut_absorb = master_high / prnu
    donut_absorb /= np.nanmedian(donut_absorb)
    donut_absorb[~np.isfinite(donut_absorb)] = 1.0

    # Flatten background, preserve rings
    donut_clean = donut_auto_flatten(donut_absorb)
    return donut_clean


def main():
    ap = argparse.ArgumentParser(description="Sample 50 subdirs -> extract donut from SKYFLAT using PRNU and save as FITS.FZ (primary HDU).")
    ap.add_argument("-i", "--input", required=True, type=Path, help="Root directory containing subdirectories (each with a calib/ folder)")
    ap.add_argument("-b", "--base",  required=True, type=Path, help="Base path (PRNU expected at $base/prnu/syth_prnu.fits.fz)")
    ap.add_argument("-n", "--num",   default=50, type=int, help="Number of subdirectories to randomly sample (default: 50)")
    ap.add_argument("--seed", default=0, type=int, help="Random seed (default: 0)")
    args = ap.parse_args()

    input_root: Path = args.input
    base_root: Path = args.base
    prnu_path = base_root / "prnu" / "syth_prnu.fits.fz"
    out_dir = base_root / "donuts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load PRNU and normalise to median ≈ 1
    if not prnu_path.exists():
        raise FileNotFoundError(f"PRNU not found: {prnu_path}")
    prnu = norm_by_median(read_first_image(prnu_path))

    # Collect subdirectories and randomly sample
    subdirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if not subdirs:
        raise RuntimeError(f"No subdirectories under {input_root}")
    rng = random.Random(args.seed)
    sel = rng.sample(subdirs, k=min(args.num, len(subdirs)))

    print(f"[INFO] Total subdirs: {len(subdirs)}; Selected: {len(sel)}")
    ok = 0
    skipped = 0

    for sd in sel:
        try:
            calib = sd / "calib"
            if not calib.is_dir():
                print(f"[WARN] No calib/ in {sd}; skip")
                skipped += 1
                continue

            # Find SKYFLAT*.fits.fz
            candidates = sorted(calib.glob("SKYFLAT*.fits.fz"))
            if not candidates:
                print(f"[WARN] No SKYFLAT*.fits.fz in {calib}; skip")
                skipped += 1
                continue

            skyflat_fp = candidates[0]  # take first alphabetically (use [-1] for latest)
            donut = process_one_skyflat(skyflat_fp, prnu)

            # Save (primary HDU)
            out_name = f"{sd.name}__{skyflat_fp.stem}__donut.fits.fz"
            out_fp = out_dir / out_name
            hdr = {
                "CREATOR": "donut_extractor",
                "SOURCE": skyflat_fp.name,
                "PRNUFILE": prnu_path.name,
                "COMMENT": "Donut (multiplicative < 1), primary HDU, float32",
            }
            write_fits_fz_float32(donut, out_fp, header=hdr)
            ok += 1
            print(f"[OK]  {sd.name} -> {out_fp}")

        except Exception as e:
            print(f"[ERR] {sd.name}: {e}")
            skipped += 1

    print(f"[DONE] wrote={ok}, skipped={skipped}, out_dir={out_dir}")


if __name__ == "__main__":
    main()
