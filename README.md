<h1 align="center">Denoising the Deep Sky: Physics-Based CCD Noise Formation for Astronomical Imaging</h1>
<p align="center">
  <a href="https://shuhongll.github.io/">Shuhong Liu</a>,
  Xining Ge,
  <a href="https://scholar.google.com/citations?user=6xQKMNYAAAAJ">Ziying Gu</a>,
  <a href="https://xuquanfeng.github.io/">Quanfeng Xu</a>,
  <a href="https://cuiziteng.github.io/">Ziteng Cui</a>,
  <a href="https://sites.google.com/view/linguedu/home">Lin Gu</a>,
  <a href="https://xg-chu.site/">Xuangeng Chu</a>,
  Jun Liu,
  <a href="https://doongli.github.io/">Dong Li</a>,
  <a href="https://www.mi.t.u-tokyo.ac.jp/harada/">Tatsuya Harada</a>
</p>
<h3 align="center">
  <a href="https://arxiv.org/abs/2601.23276">Arxiv</a> |
  <a href="">Hugging Face</a>
</h3>



---

<p align="center">
  <img src="assets/noise_formation.png" alt="fig2" width="100%">
</p>

---

## 🚀 Release

- [x] Code release
- [x] Data release
- [ ] Synthesis pipeline release

---

## 🛠️ Installation

```bash
conda create -n ccd_denoise python=3.10 -y
conda activate ccd_denoise
```

Install PyTorch (CUDA 11.8):

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

Install remaining dependencies:

```bash
pip install -r requirements.txt
```

---

## 📂 Data Structure

> Coming soon.

---

## 🔭 Synthesis

> Coming soon.

---

## 🏋️ Training

```bash
python train.py \
  -d /path/to/data \
  -x <input_suffix> \
  -y <label_suffix> \
  --epochs 140 \
  --batch-size 8 \
  --lr 1e-4 \
  --save runs \
  --name my_run
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `-d`, `--data` | required | Root directory containing subdirs and `train_test_split.json` |
| `-x`, `--input-suffix` | `syn` | Input `.npy` filename suffix (e.g. `os`, `syn`) |
| `-y`, `--label-suffix` | `mean` | Label `.npy` filename suffix (e.g. `calib`, `mean`) |
| `--arch` | `unet` | Model architecture: `unet` or `pmn_unet` |
| `--resume` | | Path to a checkpoint `.pth` to resume from |

---

## 🔮 Inference

```bash
python inference.py \
  -c /path/to/checkpoint \
  -d /path/to/data \
  -x <input_suffix> \
  -y <label_suffix> \
  --format fits.fz \
  -o /path/to/output
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `-c`, `--checkpoint` | required | Path to checkpoint directory or `.pth` file |
| `-d`, `--data` | required | Root data directory |
| `-x`, `--input-suffix` | `os` | Input `.npy` suffix |
| `-y`, `--label-suffix` | `calib` | Label `.npy` suffix |
| `--format` | `fits.fz` | Output format: `fits.fz` or `npy` |
| `--train` / `--test` | `--test` | Which split to run on |
| `-o`, `--output` | `<ckpt_dir>/results/<split>` | Output directory |
| `--eval` | | Also compute metrics after saving predictions |

For mock-PSF flux evaluation (injected sources):

```bash
python inference_psf.py \
  -c /path/to/checkpoint \
  -d /path/to/data \
  -x os_mock \
  -y calib \
  -o /path/to/output
```

---

## 📊 Evaluation

Compute PSNR / SSIM / NMAD on saved predictions:

```bash
python eval.py \
  --data_path /path/to/data \
  --pred_path /path/to/predictions \
  --band_name <band> \
  -y <label_suffix>
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--data_path` | required | Root data directory (containing `train_test_split.json`) |
| `--pred_path` | required | Directory containing `<name>_pred.fits.fz` or `<name>_pred.npy` |
| `--band_name` | required | Band identifier used for the output JSON filename |
| `-y`, `--label-suffix` | `calib` | GT `.npy` suffix (e.g. `calib`, `mean`) |

Results are saved to `<pred_path>/<band_name>.json`.

---

## 🧩 Citation

```bibtex
@article{liu2026denoising,
  title={Denoising the Deep Sky: Physics-Based CCD Noise Formation for Astronomical Imaging},
  author={Liu, Shuhong and Ge, Xining and Gu, Ziying and Gu, Lin and Cui, Ziteng and Chu, Xuangeng and Liu, Jun and Li, Dong and Harada, Tatsuya},
  journal={arXiv preprint arXiv:2601.23276},
  year={2026}
}
```
