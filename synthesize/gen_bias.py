#!/usr/bin/env python3
# gen_bias.py
#
# Build a baseline (super bias) from a folder of LCO super-bias frames and
# learn an adaptive pixel i.i.d. GMM + row-banding model (per-row Gaussian + 1D low-pass).
# Outputs are saved under: <base_dir>/bias/
#
# Usage:
#   python gen_bias.py -i /path/to/muscat3_g -b /data/.../synthesize/muscat3_g
#
# Notes:
# - Baseline is a simple per-pixel mean over TRAIN (no sigma-clipping).
# - Row series is computed with per-row MEAN (you can switch to MEDIAN if needed).
# - The learned payload (pkl) contains:
#     * GMM (sklearn GaussianMixture)
#     * row_model: Gaussian 1D kernel (sigma, truncate), and distributions of total variance and row-fraction.

from __future__ import annotations
import argparse
import json
from pathlib import Path
import pickle
import numpy as np
from astropy.io import fits
from sklearn.mixture import GaussianMixture


def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate super bias baseline and learn adaptive GMM + row-banding model."
    )
    ap.add_argument("-i", "--input", required=True, type=str,
                    help="Input root directory to search (e.g., /.../muscat3_g)")
    ap.add_argument("-b", "--base", required=True, type=str,
                    help="Base output directory (script will write to <base>/bias)")
    ap.add_argument("--pattern", default="**/calib/BIAS_*.fits.fz", type=str,
                    help="Glob pattern under --input (default: **/calib/BIAS_*.fits.fz)")
    ap.add_argument("--seed", default=42, type=int, help="Random seed (default: 42)")
    ap.add_argument("--train-n", dest="num_train_nights", default=100, type=int,
                    help="Max number of files for training (default: 100)")
    ap.add_argument("--test-n", dest="num_test_nights", default=20, type=int,
                    help="(Unused here) reserved for future tests (default: 20)")
    ap.add_argument("--max-pix", dest="max_pixels_total", default=1_500_000, type=int,
                    help="Max pixels pooled for GMM fitting (default: 1.5e6)")
    ap.add_argument("--gmm-max-k", dest="gmm_max_k", default=8, type=int,
                    help="Try components in [1..gmm_max_k] with BIC (default: 8)")
    ap.add_argument("--gmm-max-iter", dest="gmm_max_iter", default=1000, type=int,
                    help="GMM max iterations (default: 1000)")
    ap.add_argument("--reg-covar", dest="reg_covar", default=1e-6, type=float,
                    help="GMM reg_covar (default: 1e-6)")
    ap.add_argument("--row-sigma", dest="row_lp_sigma_rows", default=16.0, type=float,
                    help="Row low-pass Gaussian sigma in rows (default: 16.0)")
    ap.add_argument("--row-trunc", dest="row_lp_truncate", default=3.0, type=float,
                    help="Row low-pass truncate radius (default: 3.0)")
    ap.add_argument("--f-clip", dest="row_fraction_clip_pct", default="20,80", type=str,
                    help="Row-fraction percentile clip (lo,hi) e.g. 20,80 (default)")
    ap.add_argument("--v-clip", dest="vres_clip_pct", default="20,80", type=str,
                    help="Total variance percentile clip (lo,hi) e.g. 20,80 (default)")
    ap.add_argument("--qclip", dest="qclip", default="-400,400", type=str,
                    help="GMM sample clipping lo,hi (default: -400,400)")
    return ap.parse_args()

def list_super_bias(root: Path, glob_pattern: str, shuffle=True, seed=None):
    files = sorted(root.glob(glob_pattern))
    if len(files) < 2:
        raise RuntimeError(f"Found {len(files)} files; need ≥2. Pattern: {glob_pattern}")
    if shuffle:
        r = np.random.default_rng(seed)
        r.shuffle(files)
    return files

def load_img_2d(fp: Path) -> np.ndarray:
    with fits.open(fp) as hdul:
        img = hdul[1].data
        bias = img[:, :2048]
    return bias

def gaussian_kernel1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=float)
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k

def fit_gmm_direct(pixels: np.ndarray, components, seed=None, max_iter=1000, reg_covar=1e-6):
    X = pixels.reshape(-1, 1)
    r = np.random.default_rng(seed)
    best, best_bic = None, np.inf
    for k in components:
        gm = GaussianMixture(
            n_components=k, covariance_type='full',
            reg_covar=reg_covar, max_iter=max_iter,
            random_state=int(r.integers(0, 2**31-1))
        )
        gm.fit(X)
        bic = gm.bic(X)
        if bic < best_bic:
            best, best_bic = gm, bic
    return best, best_bic

def sample_error_from_gmm(gmm, n, seed=None, clip=None):
    r = np.random.default_rng(seed)
    pi  = gmm.weights_.astype(float)
    mu  = gmm.means_.ravel().astype(float)
    if gmm.covariance_type == "full":
        sig = np.sqrt(np.array([c[0,0] for c in gmm.covariances_], dtype=float))
    else:
        sig = np.sqrt(gmm.covariances_.reshape(-1).astype(float))
    comp = r.choice(len(pi), size=n, p=pi)
    x = mu[comp] + sig[comp]*r.normal(size=n)
    if clip is not None:
        lo, hi = clip
        x = np.clip(x, lo, hi)
    return x

def percentile_clip(val_arr, p_lo, p_hi):
    lo = float(np.percentile(val_arr, p_lo))
    hi = float(np.percentile(val_arr, p_hi))
    return lo, hi

def sample_truncated_lognormal(mu_log, sd_log, lo, hi, rng):
    for _ in range(10000):
        z = rng.normal()
        s = np.exp(mu_log + sd_log*z)
        if lo <= s <= hi:
            return float(s)
    return float(np.clip(np.exp(mu_log), lo, hi))

def logit(x):
    return np.log(x/(1-x))

def inv_logit(y):
    return 1.0/(1.0+np.exp(-y))

def sample_truncated_logitnormal(mu, sd, lo, hi, rng):
    lo = float(np.clip(lo, 1e-6, 1-1e-6))
    hi = float(np.clip(hi, 1e-6, 1-1e-6))
    for _ in range(10000):
        z = rng.normal()
        f = inv_logit(mu + sd*z)
        if lo <= f <= hi:
            return float(f)
    return float(np.clip(inv_logit(mu), lo, hi))


# --------------------------- MAIN ---------------------------

def main():
    args = parse_args()
    root_in  = Path(args.input).expanduser().resolve()
    base_dir = (Path(args.base).expanduser().resolve() / "bias")
    base_dir.mkdir(parents=True, exist_ok=True)

    pattern = args.pattern
    seed    = int(args.seed)
    rng     = np.random.default_rng(seed)

    # parse comma args
    fclip_lo, fclip_hi = [float(x) for x in args.row_fraction_clip_pct.split(",")]
    vclip_lo, vclip_hi = [float(x) for x in args.vres_clip_pct.split(",")]
    q_lo, q_hi         = [float(x) for x in args.qclip.split(",")]

    max_pixels_total   = int(args.max_pixels_total)
    num_train_nights   = int(args.num_train_nights)
    gmm_components     = range(1, int(args.gmm_max_k)+1)
    gmm_max_iter       = int(args.gmm_max_iter)
    reg_covar          = float(args.reg_covar)
    row_lp_sigma_rows  = float(args.row_lp_sigma_rows)
    row_lp_truncate    = float(args.row_lp_truncate)

    out_base_fits = "super_bias.fits.fz"
    out_gmm_pkl   = "gmm_error.pkl"

    print(f"[INFO] Input root       : {root_in}")
    print(f"[INFO] Saving under     : {base_dir}")
    print(f"[INFO] Pattern          : {pattern}")
    print(f"[INFO] Seed             : {seed}")

    split_path = root_in / "train_test_split.json"
    with open(split_path, "r") as f:
        split = json.load(f)
    merged = list(split.get("train", [])) + list(split.get("test", []))

    seen = set()
    files_all = []

    for x in merged:
        calib_dir = (root_in / x / "calib").resolve()
        bias_files = sorted(calib_dir.glob("BIAS_*.fits.fz"))
        if len(bias_files) != 1:
            raise RuntimeError(
                f"[ERROR] Expect exactly 1 BIAS_*.fits.fz in {calib_dir}, "
                f"but found {len(bias_files)}: {[str(b) for b in bias_files]}"
            )

        bias_path = bias_files[0].resolve()
        key = str(bias_path)
        if key not in seen:
            seen.add(key)
            files_all.append(bias_path)

    # 0) Split files
    n_select = min(num_train_nights, len(files_all))
    train_files = rng.choice(files_all, size=n_select, replace=False).tolist()
    print(f"[INFO] Train files      : {len(train_files)}")

    # 1) Baseline = mean over TRAIN (NO clipping)
    first = load_img_2d(train_files[0])
    H, W  = first.shape
    acc   = np.zeros_like(first, dtype=np.float64)
    cnt   = np.zeros_like(first, dtype=np.int32)
    for fp in train_files:
        im = load_img_2d(fp)
        finite = np.isfinite(im)
        acc[finite] += im[finite]
        cnt[finite] += 1
    base = np.divide(acc, np.where(cnt > 0, cnt, 1), out=np.zeros_like(acc), where=True)
    if np.any(cnt == 0):
        base[cnt == 0] = np.nanmedian(base)

    fits.CompImageHDU(data=base, compression_type='RICE_1').writeto(
        str(base_dir / out_base_fits), overwrite=True
    )
    print(f"[OK] Baseline saved     : {base_dir/out_base_fits}")

    # 2) Gather stats & fit i.i.d. GMM
    pool_pixels = []
    v_row_list  = []
    v_res_list  = []
    per_file_take = max(1, max_pixels_total // max(1, len(train_files)))

    for fp in train_files:
        im  = load_img_2d(fp)
        res = im - base
        g   = float(np.nanmean(res))
        row_series = np.nanmean(res - g, axis=1)          # per-row MEAN; switch to median if desired
        v_row = float(np.nanmean(row_series**2))          # RMS^2
        v_res = float(np.nanvar(res))
        if np.isfinite(v_row) and np.isfinite(v_res) and v_res > 0:
            v_row_list.append(v_row)
            v_res_list.append(v_res)

        x = res.ravel()
        x = x[np.isfinite(x)]
        if x.size:
            take = min(per_file_take, x.size)
            idx  = rng.choice(x.size, size=take, replace=False)
            pool_pixels.append(x[idx])

    pixels = np.concatenate(pool_pixels) if pool_pixels else np.array([], dtype=np.float64)
    if pixels.size == 0:
        raise RuntimeError("No pixels collected for GMM training.")

    gmm, gmm_bic = fit_gmm_direct(pixels, gmm_components, seed=seed,
                                  max_iter=gmm_max_iter, reg_covar=reg_covar)
    print(f"[OK] GMM fitted         : k={gmm.n_components}, BIC={gmm_bic:.1f}, pixels={pixels.size:,}")

    v_row_arr = np.array(v_row_list, dtype=float)
    v_res_arr = np.array(v_res_list, dtype=float)
    f_arr     = np.clip(v_row_arr / np.maximum(v_res_arr, 1e-12), 1e-6, 1-1e-6)

    # 2.1 Learn distributions (adaptive synthesis)
    # v_res ~ log-normal
    logs_v   = np.log(np.clip(v_res_arr, 1e-12, None))
    mu_log_v = float(np.mean(logs_v))
    sd_log_v = float(np.std(logs_v, ddof=1)) if logs_v.size > 1 else 0.0
    vres_p16 = float(np.percentile(v_res_arr, 16)) if v_res_arr.size else 0.0
    vres_p50 = float(np.percentile(v_res_arr, 50)) if v_res_arr.size else 0.0
    vres_p84 = float(np.percentile(v_res_arr, 84)) if v_res_arr.size else 0.0

    # f ~ logit-normal
    y = logit(f_arr)
    mu_logit_f = float(np.mean(y))
    sd_logit_f = float(np.std(y, ddof=1)) if y.size > 1 else 0.0
    f_p16 = float(np.percentile(f_arr, 16)) if f_arr.size else 0.0
    f_p50 = float(np.percentile(f_arr, 50)) if f_arr.size else 0.0
    f_p84 = float(np.percentile(f_arr, 84)) if f_arr.size else 0.0

    # Gaussian kernel for row low-pass
    row_kernel = gaussian_kernel1d(row_lp_sigma_rows, truncate=row_lp_truncate).astype(np.float32)

    # 3) Save unified payload
    payload = {
        "gmm": gmm,
        "q_lo": float(q_lo),
        "q_hi": float(q_hi),
        "H": int(H), "W": int(W),
        "row_model": {
            "type": "gaussian_per_row_lowpass",
            "kernel": {
                "name": "gaussian1d",
                "sigma_rows": float(row_lp_sigma_rows),
                "truncate": float(row_lp_truncate),
                "coeffs": row_kernel,  # convenience
            },
            # total residual variance distribution (adaptive target)
            "v_res": {
                "dist": "lognormal",
                "mu_log": mu_log_v,
                "sd_log": sd_log_v,
                "p16": vres_p16, "p50": vres_p50, "p84": vres_p84,
                "clip_pct": [vclip_lo, vclip_hi],
            },
            # row fraction distribution f = Var(row)/Var(residual)
            "row_fraction": {
                "dist": "logit_normal",
                "mu": mu_logit_f,
                "sd": sd_logit_f,
                "p16": f_p16, "p50": f_p50, "p84": f_p84,
                "clip_pct": [fclip_lo, fclip_hi],
            },
        },
    }

    pkl_path = base_dir / out_gmm_pkl
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[OK] Model saved        : {pkl_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()
