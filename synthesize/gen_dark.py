#!/usr/bin/env python3
"""
Fit GMM to super-dark *errors* and save base + model.

Example:
    python fit_superdark_gmm.py \
        --input /data/umiushi0/users/shuhong/muscat34/data/muscat3_g \
        --base  /data/umiushi0/users/shuhong/muscat34/synthesize/muscat3_g
"""

import argparse
from pathlib import Path
import os, pickle
import numpy as np
from astropy.io import fits
from sklearn.mixture import GaussianMixture

# --------------------------- CONFIG ---------------------------
max_pixels_total  = 1_500_000
num_train_nights = 200
num_test_nights  = 20
randomize_files  = True
seed             = 42

use_per_image_clip = True
k_sigma            = 6.0  # median ± k_sigma * MAD

gmm_components  = range(1, 9)
gmm_max_iter    = 1000
reg_covar       = 1e-6

# --------------------------- UTILS ----------------------------
def list_super_darks(root: str | Path, glob_pattern: str, shuffle=True, seed=None):
    files = sorted(Path(root).glob(glob_pattern))
    if len(files) < 2:
        raise RuntimeError(f"Found {len(files)} files; need ≥2. Pattern: {glob_pattern}")
    if shuffle:
        r = np.random.default_rng(seed)
        r.shuffle(files)
    return files

def load_super_dark(fp: Path) -> np.ndarray:
    with fits.open(fp, memmap=True) as hdul:
        for h in hdul:
            if hasattr(h, "data") and isinstance(h.data, np.ndarray) and h.data.ndim == 2:
                return np.asarray(h.data, dtype=np.float64)
    raise ValueError(f"No 2D image in {fp}")

def _robust_sigma_1d(x: np.ndarray) -> float:
    m = np.median(x)
    mad = np.median(np.abs(x - m))
    s = 1.4826 * mad
    return float(s if s > 0 else np.std(x))

def per_image_clip(arr, k: float):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        finite = np.isfinite(arr)
        x = arr[finite]
        if x.size == 0:
            return arr[:0]
        m = np.median(x)
        s = _robust_sigma_1d(x)
        lo, hi = m - k*s, m + k*s
        return arr[(arr >= lo) & (arr <= hi) & finite]
    else:
        finite = np.isfinite(arr)
        x = arr[finite]
        if x.size == 0:
            return np.full_like(arr, np.nan, dtype=np.float64), np.zeros_like(arr, bool)
        m = np.median(x)
        s = _robust_sigma_1d(x)
        lo, hi = m - k*s, m + k*s
        mask = finite & (arr >= lo) & (arr <= hi)
        out = arr.astype(np.float64, copy=True)
        out[~mask] = np.nan
        return out, mask


def main(input_path: str, base_path: str):
    rng = np.random.default_rng(seed)
    input_path = Path(input_path)
    out_dir = Path(base_path) / "dark"
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/calib/DARK_*.fits.fz"
    files_all   = list_super_darks(input_path, pattern, shuffle=randomize_files, seed=seed)
    train_files = files_all[:min(num_train_nights, len(files_all))]
    print(f"Train nights: {len(train_files)}")

    # Base = mean of training super-darks
    base_sum = None
    base_cnt = None
    for fp in train_files:
        im = load_super_dark(fp)
        if use_per_image_clip:
            im_clip, mask = per_image_clip(im, 3.5)
        else:
            im_clip = im
            mask = np.isfinite(im_clip)

        if base_sum is None:
            base_sum = np.zeros_like(im_clip)
            base_cnt = np.zeros_like(im_clip, dtype=np.int32)

        base_sum[mask] += im_clip[mask]
        base_cnt[mask] += 1

    den  = np.where(base_cnt > 0, base_cnt, 1)
    base = base_sum / den
    if np.any(base_cnt == 0):
        fill_val = np.nanmedian(base)
        base[base_cnt == 0] = fill_val

    base_out = out_dir / "super_dark.fits.fz"
    comp_hdu = fits.CompImageHDU(data=base, compression_type='RICE_1')
    comp_hdu.writeto(base_out, overwrite=True)
    print(f"Saved base super-dark to: {base_out}")

    # Collect errors
    emp_err_pool = []
    for fp in train_files:
        with fits.open(fp, memmap=True) as hdul:
            for h in hdul:
                if hasattr(h,"data") and isinstance(h.data,np.ndarray) and h.data.ndim==2:
                    err = np.asarray(h.data, np.float64) - base
                    x = err.ravel()
                    if use_per_image_clip:
                        x = per_image_clip(x, k_sigma)
                    if x.size > 0:
                        emp_err_pool.append(x)
                    break

    emp_err = np.concatenate(emp_err_pool) if emp_err_pool else np.array([], dtype=np.float64)
    if emp_err.size > max_pixels_total:
        idx = rng.choice(emp_err.size, size=max_pixels_total, replace=False)
        emp_err = emp_err[idx]

    print(f"Train error pixels used: {emp_err.size:,}")

    # GMM fit
    X = emp_err.reshape(-1, 1)
    best, best_bic = None, np.inf
    for k in gmm_components:
        gm = GaussianMixture(n_components=k, covariance_type='full',
                             reg_covar=reg_covar, max_iter=gmm_max_iter,
                             random_state=int(rng.integers(0, 2**31-1)))
        gm.fit(X)
        bic = gm.bic(X)
        if bic < best_bic:
            best, best_bic = gm, bic

    print(f"GMM: k={best.n_components}, BIC={best_bic:.1f}")

    # Save model + quantiles
    q_lo, q_hi = np.quantile(emp_err, [0.0005, 0.9995])
    out_gmm = out_dir / "gmm_error.pkl"
    with open(out_gmm, "wb") as f:
        pickle.dump({"gmm": best, "q_lo": float(q_lo), "q_hi": float(q_hi)}, f)
    print(f"Saved GMM model to: {out_gmm}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit GMM on super-dark errors and save base + model.")
    parser.add_argument("-i", "--input", required=True, help="Input path containing super-dark files.")
    parser.add_argument("-b", "--base", required=True, help="Base output path (will create $base/dark/).")
    args = parser.parse_args()

    main(args.input, args.base)
