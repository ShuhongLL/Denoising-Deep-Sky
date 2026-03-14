#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From a clean SCI-like image (ADU) synthesize a noisy observation per your spec:
- Photon noise (Poisson) applied after re-introducing PRNU (super_prnu * donut).
- Dark noise: sample synthetic super-dark (e-/s) via GMM payload, multiply by EXPTIME (no Poisson).
- Read noise: synthesize from bias payload (ADU) as additive noise (row banding + iid GMM).
- Quantization noise: U(-0.5, 0.5) ADU.
- No global baseline bias (BIASLVL) is added.

IO layout:
  --input  <root>/
      <subdir1>/data/*_mean.fits.fz  (HDU[1], clean start; exactly one or error)
                           *_calib.fits.fz (HDU[1], read header GAIN, EXPTIME)
      <subdir2>/data/...
  --base   <base>/
      prnu/super_prnu.fits.fz
      donuts/*donut.fits.fz
      dark/super_dark.fits.fz
      dark/gmm_error.pkl
      dark/error_psd_rfft.npy (optional)
      bias/super_bias.fits.fz
      bias/gmm_error.pkl

Output:
  <subdir_name>_synth.fits.fz  (CompImageHDU, RICE_1) in the subdir root
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import random
import pickle
import re

import numpy as np
from astropy.io import fits
from numpy.fft import irfft2

# ----------------------------- Utils -----------------------------

def read_hdu1_2d(fp: Path) -> np.ndarray:
    with fits.open(fp, memmap=True) as hdul:
        if len(hdul) < 2 or getattr(hdul[1], "data", None) is None:
            raise RuntimeError(f"{fp} does not contain HDU[1] 2D image")
        arr = np.asarray(hdul[1].data, dtype=np.float64)
    if arr.ndim != 2:
        raise RuntimeError(f"{fp} HDU[1] is not 2D")
    return arr

def read_header_gain_exptime(fp: Path) -> tuple[float, float]:
    with fits.open(fp, memmap=True) as hdul:
        hdr = hdul[1].header if len(hdul) > 1 else hdul[0].header
        if "GAIN" not in hdr or "EXPTIME" not in hdr:
            raise RuntimeError(f"{fp} missing GAIN/EXPTIME in header")
        G = float(hdr["GAIN"])
        t = float(hdr["EXPTIME"])
    if not np.isfinite(G) or G <= 0:
        raise RuntimeError(f"Invalid GAIN={G} in {fp}")
    if not np.isfinite(t) or t <= 0:
        raise RuntimeError(f"Invalid EXPTIME={t} in {fp}")
    return G, t

def list_subdirs_nonrecursive(root: Path) -> list[Path]:
    return [p for p in root.iterdir() if p.is_dir()]

def find_single_mean(data_dir: Path) -> Path:
    cands = sorted(data_dir.glob("*_mean.fits.fz"))
    if len(cands) != 1:
        raise RuntimeError(f"{data_dir}: expected exactly 1 *_mean.fits.fz, got {len(cands)}")
    return cands[0]

def find_single_calib(data_dir: Path) -> Path:
    cands = sorted(data_dir.glob("*_calib.fits.fz"))
    if len(cands) != 1:
        raise RuntimeError(f"{data_dir}: expected exactly 1 *_calib.fits.fz, got {len(cands)}")
    return cands[0]

def choose_random_donut(donuts_dir: Path, rng: np.random.Generator) -> Path:
    cands = sorted(donuts_dir.glob("*donut*.fits.fz")) + sorted(donuts_dir.glob("*donut*.fits"))
    if not cands:
        raise RuntimeError(f"No *donut*.fits(.fz) found in {donuts_dir}")
    return cands[rng.integers(0, len(cands))]

# ---------------- PRNU ----------------

def load_prnu_map(prnu_dir: Path, donuts_dir: Path, rng: np.random.Generator) -> np.ndarray:
    super_prnu_fp = prnu_dir / "super_prnu.fits.fz"
    if not super_prnu_fp.exists():
        # fallback: allow non-compressed
        alt = prnu_dir / "super_prnu.fits"
        if not alt.exists():
            raise RuntimeError(f"Missing {super_prnu_fp}")
        super_prnu_fp = alt
    F_super = read_hdu1_2d(super_prnu_fp)
    donut_fp = choose_random_donut(donuts_dir, rng)
    F_donut = read_hdu1_2d(donut_fp)

    if F_super.shape != F_donut.shape:
        raise RuntimeError(f"PRNU shape mismatch: super_prnu {F_super.shape} vs donut {F_donut.shape}")
    F = F_super * F_donut   # total PRNU
    return F

# ---------------- Dark synth (GMM + optional PSD) ----------------

def load_dark_payload(dark_dir: Path):
    base_fp = dark_dir / "super_dark.fits.fz"
    if not base_fp.exists():
        alt = dark_dir / "super_dark.fits"
        if not alt.exists():
            raise RuntimeError(f"Missing {base_fp}")
        base_fp = alt
    base = read_hdu1_2d(base_fp)

    pkl_fp = dark_dir / "gmm_error.pkl"
    if not pkl_fp.exists():
        raise RuntimeError(f"Missing {pkl_fp}")
    with open(pkl_fp, "rb") as f:
        payload = pickle.load(f)
    # accept dict {"gmm":..., "q_lo":..., "q_hi":...} or bare GMM
    if isinstance(payload, dict) and "gmm" in payload:
        gmm = payload["gmm"]
        q_lo = payload.get("q_lo", None)
        q_hi = payload.get("q_hi", None)
    else:
        gmm = payload
        q_lo = q_hi = None

    psd_fp = dark_dir / "error_psd_rfft.npy"
    psd = np.load(psd_fp) if psd_fp.exists() else None
    return base, gmm, q_lo, q_hi, psd

def sample_error_from_gmm(gmm, n: int, rng: np.random.Generator,
                          clip: tuple[float,float] | None = None) -> np.ndarray:
    # sklearn.mixture.GaussianMixture-like
    pi  = np.asarray(gmm.weights_, dtype=float)
    mu  = gmm.means_.ravel().astype(float)
    sig = np.sqrt(gmm.covariances_.reshape(-1)).astype(float)
    comp = rng.choice(len(pi), size=n, p=pi)
    e = mu[comp] + sig[comp]*rng.normal(size=n)
    if clip is not None:
        lo, hi = clip
        e = np.clip(e, lo, hi)
    return e

def sample_correlated_from_psd(psd_rfft: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # psd_rfft shape (H, W//2+1)
    H = psd_rfft.shape[0]
    W = (psd_rfft.shape[1]-1)*2
    Z = (rng.normal(size=psd_rfft.shape) + 1j * rng.normal(size=psd_rfft.shape))
    E = Z * np.sqrt(np.maximum(psd_rfft, 0.0))
    e = irfft2(E, s=(H, W)).real
    e -= np.mean(e)
    return e

def synthesize_super_dark(base_eps: np.ndarray,
                          gmm, q_lo, q_hi,
                          psd_rfft: np.ndarray | None,
                          alpha: float,
                          rng: np.random.Generator,
                          clip_min: float | None = None) -> np.ndarray:
    """
    Return synthetic super-dark in e-/s:
      base + (alpha * correlated + (1-alpha) * iid)
    Variance of mix is matched to iid's variance approximately.
    """
    H, W = base_eps.shape
    iid = sample_error_from_gmm(gmm, H*W, rng, clip=(q_lo, q_hi) if (q_lo is not None and q_hi is not None) else None).reshape(H, W)

    if psd_rfft is None or alpha <= 0:
        err = iid
    else:
        e_corr = sample_correlated_from_psd(psd_rfft, rng)
        # variance match
        var_iid = float(np.var(iid)) + 1e-12
        var_mix = (alpha**2)*float(np.var(e_corr)) + ((1-alpha)**2)*var_iid
        scale = np.sqrt(var_iid/var_mix) if var_mix > 0 else 1.0
        err = scale * (alpha*e_corr + (1-alpha)*iid)

    out = base_eps + err
    if clip_min is not None:
        np.clip(out, clip_min, None, out=out)
    return out  # e-/s

# ---------------- Bias/read-noise synth (ADU) ----------------
def inv_logit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def sample_truncated_lognormal(mu_log: float, sd_log: float, lo: float, hi: float, rng: np.random.Generator) -> float:
    # rejection for simplicity (single scalar)
    for _ in range(10000):
        v = np.exp(mu_log + sd_log * rng.normal())
        if lo <= v <= hi:
            return float(v)
    return float(np.clip(np.exp(mu_log), lo, hi))

def load_bias_payload(bias_dir: Path):
    base_fp = bias_dir / "super_bias.fits.fz"
    if not base_fp.exists():
        alt = bias_dir / "super_bias.fits"
        if not alt.exists():
            raise RuntimeError(f"Missing {base_fp}")
        base_fp = alt
    base = read_hdu1_2d(base_fp)

    pkl_fp = bias_dir / "gmm_error.pkl"
    if not pkl_fp.exists():
        raise RuntimeError(f"Missing {pkl_fp}")
    with open(pkl_fp, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError("Bias payload must be a dict with 'row_model' and 'gmm'.")
    return base, payload

def synthesize_bias_readnoise(base_bias_adu: np.ndarray,
                              payload: dict,
                              row_variance_scale: float,
                              rng: np.random.Generator) -> np.ndarray:
    """
    Follow your 'synthesize_once' idea:
      total = baseline + row_banding + iid_gmm   (all in e-)
    """
    H, W = base_bias_adu.shape
    rm = payload["row_model"]
    vinfo = rm["v_res"]
    finfo = rm["row_fraction"]

    # draw total variance v_target with truncation
    # (use percentiles specified in payload if available)
    clip_pct_v = tuple(vinfo.get("clip_pct", (20.0, 80.0)))
    # generate quick MC to estimate percentiles
    mc = np.exp(vinfo["mu_log"] + vinfo["sd_log"] * rng.normal(size=20000))
    v_lo, v_hi = np.percentile(mc, clip_pct_v)
    v_target = sample_truncated_lognormal(vinfo["mu_log"], vinfo["sd_log"], v_lo, v_hi, rng)

    # draw row fraction f_target with truncation
    clip_pct_f = tuple(finfo.get("clip_pct", (40.0, 60.0)))
    mc_f = inv_logit(finfo["mu"] + finfo["sd"] * rng.normal(size=20000))
    f_lo, f_hi = np.percentile(mc_f, clip_pct_f)
    # sample approx by rejection on logit-normal
    for _ in range(10000):
        f_try = float(inv_logit(finfo["mu"] + finfo["sd"] * rng.normal()))
        if f_lo <= f_try <= f_hi:
            f_target = f_try
            break
    else:
        f_target = float(np.clip(inv_logit(finfo["mu"]), f_lo, f_hi))

    v_row_target = float(f_target * v_target) * float(row_variance_scale)
    v_iid_target = float(max(v_target - v_row_target, 0.0))

    # row banding (white -> lowpass kernel)
    k = np.asarray(rm["kernel"]["coeffs"], dtype=float)
    r_white = rng.normal(size=H)
    r_lp = np.convolve(r_white, k, mode="same")
    vr = float(np.nanvar(r_lp)) + 1e-12
    r_lp *= np.sqrt(max(v_row_target, 0.0) / vr)
    row_2d = r_lp[:, None]  # broadcast to (H, W)

    # iid GMM pixels, then scale to target variance
    gmm = payload["gmm"]
    q_lo = payload.get("q_lo", None)
    q_hi = payload.get("q_hi", None)
    iid = sample_error_from_gmm(gmm, H*W, rng, clip=(q_lo, q_hi) if (q_lo is not None and q_hi is not None) else None).reshape(H, W)
    vi = float(np.nanvar(iid)) + 1e-12
    iid *= np.sqrt(max(v_iid_target, 0.0) / vi)

    # final ADU read-noise frame (with baseline)
    return base_bias_adu + row_2d + iid

# ---------------- Core Synth ----------------
def synthesize_one(clean_adu: np.ndarray,
                   G_e_per_ADU: float,
                   EXPTIME_s: float,
                   prnu_map: np.ndarray,
                   dark_base_eps: np.ndarray,
                   dark_gmm, dark_q_lo, dark_q_hi, dark_psd,
                   dark_alpha: float,
                   dark_clip_min: float | None,
                   bias_base: np.ndarray,
                   bias_payload: dict,
                   row_variance_scale: float,
                   rng: np.random.Generator) -> np.ndarray:
    H, W = clean_adu.shape
    # sanity
    for name, arr in [("PRNU", prnu_map), ("dark_base", dark_base_eps), ("bias_base", bias_base_adu)]:
        if arr.shape != (H, W):
            raise RuntimeError(f"Shape mismatch: clean {clean_adu.shape} vs {name} {arr.shape}")

    # 1) SCI ADU -> electrons, then re-introduce PRNU into expectation
    mu_phot_e = G_e_per_ADU * clean_adu * prnu_map
    mu_phot_e = np.clip(mu_phot_e, 0.0, None)

    # 2) Photon Poisson
    rng_poisson = rng  # same RNG is fine
    N_phot = rng_poisson.poisson(mu_phot_e)

    # 3) Dark: synthesize super-dark (e-/s), multiply by EXPTIME (no Poisson as requested)
    syn_dark_eps = synthesize_super_dark(dark_base_eps, dark_gmm, dark_q_lo, dark_q_hi,
                                         dark_psd, dark_alpha, rng, clip_min=dark_clip_min)
    N_dark = syn_dark_eps * EXPTIME_s  # electrons

    # 4) Read noise in electrons: synthesize from bias payload and add
    N_read = synthesize_bias_readnoise(bias_base, bias_payload, row_variance_scale, rng)

    # 5) Combine in electrons
    e_total = N_phot + N_dark + N_read

    # 6) Convert back to ADU
    out_adu = e_total / G_e_per_ADU

    # # 7) Quantization noise in ADU
    # out_adu += rng.uniform(low=-0.5, high=0.5, size=(H, W))

    # 8) Hotpixel + Cosmic Ray
    n_hot = max(1, int(round(H * W * 0.0001)))  # 0.01% of pixels
    hot_rows = rng.integers(0, H, size=n_hot)
    hot_cols = rng.integers(0, W, size=n_hot)
    hot_vals = rng.uniform(5000.0, 10000.0, size=n_hot)
    out_adu[hot_rows, hot_cols] = hot_vals

    return out_adu


def main():
    ap = argparse.ArgumentParser(description="Simulate noisy observation from clean SCI ADU images.")
    ap.add_argument("-i", "--input", type=Path, required=True, help="Root path containing subdirs (nonrecursive).")
    ap.add_argument("-b", "--base",  type=Path, required=True, help="Base path containing bias/dark/donuts/prnu folders.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed.")
    ap.add_argument("--dark-alpha", type=float, default=0.0, help="Mix ratio for correlated dark error via PSD (0 = iid only).")
    ap.add_argument("--dark-clip-min", type=float, default=None, help="Clip synthetic super-dark to be >= this (e-/s).")
    ap.add_argument("--row-variance", type=float, default=0.5, help="Row-banding variance scale for bias read-noise synth.")
    args = ap.parse_args()

    root = args.input.resolve()
    base = args.base.resolve()

    prnu_dir   = base / "prnu"
    donuts_dir = base / "donuts"
    dark_dir   = base / "dark"
    bias_dir   = base / "bias"

    rng = np.random.default_rng(args.seed)

    # preload shared assets
    F = load_prnu_map(prnu_dir, donuts_dir, rng)
    dark_base, dark_gmm, dark_q_lo, dark_q_hi, dark_psd = load_dark_payload(dark_dir)
    bias_base, bias_payload = load_bias_payload(bias_dir)

    subdirs = list_subdirs_nonrecursive(root)
    if not subdirs:
        print(f"[WARN] No subdirs under {root}", file=sys.stderr)
        return

    for sub in sorted(subdirs):
        data_dir = sub / "data"
        if not data_dir.exists():
            print(f"[SKIP] {sub} has no 'data/'", file=sys.stderr)
            continue

        fp_mean  = find_single_mean(data_dir)
        fp_calib = find_single_calib(data_dir)
        clean_adu = read_hdu1_2d(fp_mean)
        G, t = read_header_gain_exptime(fp_calib)

        # shape checks
        if clean_adu.shape != F.shape:
            raise RuntimeError(f"{sub.name}: shape mismatch clean {clean_adu.shape} vs PRNU {F.shape}")
        if clean_adu.shape != dark_base.shape:
            raise RuntimeError(f"{sub.name}: shape mismatch clean {clean_adu.shape} vs dark {dark_base.shape}")
        if clean_adu.shape != bias_base.shape:
            raise RuntimeError(f"{sub.name}: shape mismatch clean {clean_adu.shape} vs bias {bias_base.shape}")

        synth = synthesize_one(clean_adu, G, t, F,
                               dark_base, dark_gmm, dark_q_lo, dark_q_hi, dark_psd,
                               args.dark_alpha, args.dark_clip_min,
                               bias_base, bias_payload, args.row_variance,
                               rng)
        out_name = f"{sub.name}_synth.fits.fz"
        out_fp = sub / "data" / out_name
        # Write as RICE_1 compressed CompImageHDU
        fits.CompImageHDU(data=synth.astype(np.float32), compression_type="RICE_1").writeto(out_fp, overwrite=True)
        print(f"[OK] {sub.name}: wrote {out_fp}")


if __name__ == "__main__":
    main()
