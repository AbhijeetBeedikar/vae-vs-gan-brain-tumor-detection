# Reproducing the project from scratch

This guide walks through the whole project in the order it was built and run.
For each step it explains **what the step does**, **how it works**, **what
happened when we ran it** (numbers, problems, fixes) and **the exact commands**
to repeat it yourself.

Total time on a laptop CPU is roughly **1 day**: about 1.5 h preprocessing,
about 6 h training (three models in parallel), about 15 min evaluation, plus
the download.

---

## The idea in one paragraph

We train neural networks **only on healthy brain MRI scans** (IXI dataset, T2
contrast). Each network learns to take a brain slice and reproduce it. Because
it has only seen healthy anatomy, when we show it a slice with a tumour (MSD
BrainTumour dataset) it reproduces a *healthy-looking* version instead. The
pixels where the input differs from that reconstruction are the anomaly, which
should be the tumour. We compare a **VAE**, **f-AnoGAN** (a GAN with an
encoder), a **plain autoencoder** (control) and an **intensity baseline** (no
learning: "bright = tumour"). No tumour labels are used for training. Labels
are only used to pick a threshold on a small validation set and to score the
results.

```
download  ->  preprocess (register, skull-strip, normalise, slice)  ->  split by subject
          ->  train AE / VAE / f-AnoGAN on healthy slices
          ->  reconstruct tumour slices, residual -> anomaly map -> threshold
          ->  Dice / AUROC / AUPRC / FPR, figures, README, report
```

---

## Step 0. What you need

| Requirement | Notes |
|---|---|
| Disk space | about 25 GB free (11.5 GB downloads, about 8 GB extracted, about 4 GB preprocessed) |
| RAM | 16 GB minimum, 32 GB comfortable (all slices are held in memory) |
| Python | 3.12 (3.10 or newer should work) |
| Git | any recent version |
| GPU | optional. Everything here was run on a CPU only (Snapdragon X Elite laptop, Windows 11 ARM) |
| LaTeX | only for building the report (MiKTeX or TeX Live) |

---

## Step 1. Get the code

```bash
git clone https://github.com/<you>/<repo>.git
cd <repo>
```

**What's inside** (and the order it was originally written, matching the git
history):

| Commit | What was added |
|---|---|
| 1. Scaffold | package layout `src/brainad/`, `configs/`, `requirements.txt`, `.gitignore` (keeps data and checkpoints out of git) |
| 2. Preprocessing | `template.py` (T2 brain template), `registration.py` (affine registration in PyTorch), `preprocessing.py` (skull-strip, normalise, slice) |
| 3. Models + evaluation | `models/` (AE, VAE, f-AnoGAN), `train.py`, `anomaly.py`, `metrics.py`, `evaluate.py`, `visualize.py`, `tests/` |
| 4. Pipeline script + README draft | `scripts/run_all.sh`, README |
| 5. Robust registration | grid search over head tilt, shift and scale (fixed bad registrations, see Step 5) |
| 6. Results | `results/` metrics, figures, README findings |
| 7. Midway report | `report/` LaTeX report and its figure script |

All settings live in `configs/base.yaml`. The files `configs/ae.yaml`,
`configs/vae.yaml` and `configs/fanogan.yaml` start from it and override only
model-specific values. Any value can be overridden from the command line with
`--set key=value`.

---

## Step 2. Set up the Python environment

```bash
python -m venv .venv
```

Activate it. On Windows (PowerShell): `.venv\Scripts\Activate.ps1`.
On macOS/Linux: `source .venv/bin/activate`.

Install PyTorch **first**, from the PyTorch site, because PyPI's default build
is not right for every machine:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

(With an NVIDIA GPU, pick the CUDA command from <https://pytorch.org> instead.)

Then the rest, plus the project itself in editable mode (so `import brainad`
works from anywhere):

```bash
pip install -r requirements.txt
pip install -e .
```

**What happened here / Windows-on-ARM notes.**

* PyPI has no PyTorch build for Windows ARM64. The PyTorch CPU index does have
  one, so the `--index-url` above is required.
* SimpleITK, ANTs and dipy (the usual medical registration tools) have **no
  Windows ARM64 builds**. That is why we wrote our own registration in PyTorch
  (Step 5). This also means the project needs nothing beyond PyTorch, numpy
  and scipy.
* For **training speed** on a Snapdragon laptop we made a second environment
  with **x64 Python**, which runs under Windows emulation. The x64 PyTorch
  build includes optimised oneDNN kernels and trained the AE at **670 img/s
  vs. 257 img/s** for the native ARM build (2.6× faster). To do the same,
  install an x64 Python (e.g. the python.org "Windows installer (64-bit)") and:

  ```bash
  py -3.13-64 -m venv .venv-x64
  .venv-x64\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  .venv-x64\Scripts\python -m pip install -r requirements.txt
  .venv-x64\Scripts\python -m pip install -e .
  ```

  (If `py -3.13-64` isn't recognised, use the full path to the x64
  `python.exe` instead.) Keep the path short: an install under a long directory failed with a
  Windows "long path" error. On Intel/AMD machines or Macs you don't need any
  of this.

---

## Step 3. Run the tests (no data needed)

```bash
pytest
```

**What it checks (19 tests, about 15 s):**

* `tests/test_units.py`: config merging, Dice/AUROC/threshold maths, anomaly
  map smoothing and masking, network output shapes.
* `tests/test_registration.py`: applies a known affine transform to a volume
  and checks that our registration recovers it to **under 1 mm**.
* `tests/test_pipeline.py`: a tiny synthetic end-to-end run. Fake "brains"
  with bright "tumours" go through training for all three models and the full
  evaluation, so every code path runs.

Expected: `19 passed`.

---

## Step 4. Download and unpack the data

```bash
python scripts/download_data.py
```

**What it does.** Downloads two public archives into `data/raw/` and extracts
only the `.nii.gz` image files into `data/extracted/`:

| Dataset | File | Size | Contents | Licence |
|---|---|---|---|---|
| IXI | `IXI-T2.tar` | about 3.9 GB | 578 T2 scans of healthy adults | CC BY-SA 3.0 |
| MSD Task01 | `Task01_BrainTumour.tar` | about 7.6 GB | 484 glioma patients, 4 MRI contrasts each (T2 is channel 3), expert tumour labels | CC BY-SA 4.0 |

The script skips files that are already fully downloaded and skips macOS
`._*` junk files inside the MSD archive.

**What happened.** The 7.6 GB MSD download broke partway through (`curl:
(56)` connection reset). If that happens to you, resume it with curl and then
run only the extraction:

```bash
curl -L -C - --retry 10 -o data/raw/Task01_BrainTumour.tar https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
python scripts/download_data.py --extract-only
```

The complete MSD file is exactly **7,608,266,240 bytes**.

---

## Step 5. Preprocess: 3D scans to aligned 2D slices

```bash
python scripts/preprocess.py
```

(Quick trial first: `python scripts/preprocess.py --limit 10` processes 10
scans per dataset. Delete `data/processed/` afterwards, before the real run.)

**Why this step matters.** IXI and MSD were scanned at different hospitals and
processed differently. If the two datasets don't look alike, a model can learn
to flag "different hospital" instead of "tumour". So **every scan goes through
exactly the same pipeline** (`src/brainad/preprocessing.py`, function
`process_volume`):

1. **Load and reorient** to a standard orientation (RAS), roughly rescale
   intensities.
2. **Register (align) to a template brain.**
   * Template: the MNI152 2009a brain. nilearn only ships a *T1* version, so
     `template.py` builds a **T2-like template** from its grey- and
     white-matter probability maps. In T2, CSF is bright (1.0), grey matter
     medium (0.55) and white matter dark (0.35).
   * Registration (`registration.py`): a 12-parameter affine transform
     (rotation, scaling, shear, translation) found by gradient descent in
     PyTorch. It resamples the scan with `grid_sample`, maximises normalised
     cross-correlation (NCC) with the template using Adam, and works coarse to
     fine (4 mm, then 2 mm). Each step uses a random subset of 40k voxels for
     speed.
   * **Grid-search start** (added after a failure, see below): before
     optimising, it scores combinations of head tilt (pitch −25°…+25°),
     front–back shift (±20 mm), up–down shift (±40 mm) and size (0.8–1.0) at
     6 mm resolution. It refines the top 3 for 60 steps each and starts from
     the best one.
3. **Skull-strip**: keep only voxels inside the template's brain mask, the
   same mask for everyone. MSD scans were already skull-stripped by BraTS, so
   for them we also intersect with their own non-zero region.
4. **Normalise intensities**: divide by the 99th percentile of brain
   intensities and clip to [0, 1].
5. **Slice**: take axial slices whose brain area is at least 30% of the
   largest (this drops near-empty top and bottom slices), keep every 2nd one
   (neighbours are near-duplicates), crop square around the brain and resize
   to 128×128 by area averaging. 64×64 versions are made from these at
   training time.
6. For MSD, the expert tumour labels go through the same transform, and all
   tumour classes are merged into one **whole-tumour** mask.

It also records a **registration quality score** per scan: the NCC between
the aligned scan and the template inside the brain.

**Output:** `data/processed/ixi.npz` and `data/processed/msd.npz` (images,
brain masks, tumour masks, subject IDs, slice levels, NCC per scan).

**What happened.**

* First full run: about **30% of IXI scans registered badly** (NCC near 0,
  eyes inside the brain mask). The cause was head tilt plus a poor starting
  position (we started from the centre of mass), which left the optimiser
  stuck in a wrong local optimum. Trying several starting shifts was not
  enough. Adding the **grid search over tilt, shift and scale** fixed it: the
  failing scans went to NCC 0.35–0.50.
* Final run: **578/578 IXI and 484/484 MSD scans processed, 0 failures**. It
  took about 39 min for IXI and 45 min for MSD with 6 worker processes
  (about 25 s per scan).
* Result: **33,524 IXI slices** and **28,072 MSD slices**.
* Quality: median NCC **0.47 (IXI)** and **0.25 (MSD)**. The lower MSD scores
  are expected. We viewed the lowest-scoring scans and they are aligned
  correctly. They are the ones with very large tumours, which naturally
  correlate less with a healthy template. The lowest IXI scores came from
  enlarged ventricles or a thin rim of leftover skull, also fine.

**Check it yourself:**

```bash
python -c "import numpy as np; d=np.load('data/processed/ixi.npz'); print(d['image'].shape, np.median(d['ncc']))"
```

Expect `(33524, 128, 128)` and about `0.47`. A scan with NCC below 0.1 is
worth plotting to check its alignment.

---

## Step 6. Train/validation/test split (automatic)

You don't run anything here. The first time a model is trained,
`src/brainad/data.py` (`make_splits`) shuffles subjects with seed 42 and
writes `data/processed/splits.json`. Every later model reuses it, so **all
models see exactly the same subjects**. Splits are by **subject**, never by
slice, so one person's slices never appear in both training and test.

| Split | Subjects | Slices | Used for |
|---|---|---|---|
| `ixi_train` | 462 | 26,796 | training |
| `ixi_val` | 58 | — | choosing the best checkpoint (healthy reconstruction error) |
| `ixi_test` | 58 | 3,364 | healthy false-positive rate |
| `msd_val` | 48 | 2,784 | **only** choosing the detection threshold |
| `msd_test` | 436 | 25,288 | all reported tumour metrics |

---

## Step 7. Train the three models

One command per model. They are independent, so you can run them one after
another or in three terminals at once:

```bash
python scripts/train.py --config configs/ae.yaml
python scripts/train.py --config configs/vae.yaml
python scripts/train.py --config configs/fanogan.yaml
```

When running all three in parallel on a 12-core CPU, give each 4 threads so
they don't fight over cores. Logging to a file is handy (bash shown):

```bash
python scripts/train.py --config configs/ae.yaml --set training.num_threads=4 > runs/train_ae.log 2>&1 &
```

Each run writes to `runs/<model>_64/`: `model.pt` (best checkpoint),
`history.json` (per-epoch losses) and `val_reconstructions.png`. f-AnoGAN also
writes `gan_samples.png`.

**How the models work** (all share the same DCGAN-style networks: four
4×4 stride-2 convolutions, 1→32→64→128→256 channels, 64×64 to 4×4, then a
linear layer to a **128-number latent code**. The decoder mirrors this):

| Model | Training objective | Reconstruction at test time |
|---|---|---|
| **AE** (2.44M params) | squared error between input and reconstruction | decoder(encoder(x)) |
| **VAE** (2.96M) | squared error + β·KL(posterior ‖ N(0, I)), β = 1, reparameterisation trick | decode the posterior mean |
| **f-AnoGAN** (3.13M) | **Stage 1:** WGAN-GP (generator vs. critic, gradient penalty 10, 5 critic steps per generator step, 60 epochs). **Stage 2:** freeze both, train an encoder for 30 epochs so G(E(x)) matches x in pixels *and* in the critic's features | G(E(x)) |

Shared settings: Adam, learning rate 2e-4 (GAN 1e-4 with betas (0, 0.9)),
batch 64, random left–right flips, 40 epochs for AE/VAE. The checkpoint with
the lowest healthy validation error is kept.

**What happened.**

* Run in parallel with the x64 PyTorch build, 4 threads each:
  **AE 2.9 h, VAE 3.0 h, f-AnoGAN 6.3 h** (5.4 h of which was the GAN stage).
* Final healthy validation error (MSE per pixel): **AE 0.0021, VAE 0.0047,
  f-AnoGAN 0.0056**. The AE reconstructs best, which turns out to be a
  *weakness* for anomaly detection (see Step 9).
* The f-AnoGAN critic's Wasserstein estimate fell steadily from 35.5 to 1.5
  (no collapse), and `gan_samples.png` showed realistic brains at every slice
  level.
* Look at `val_reconstructions.png`. VAE reconstructions are smooth and
  blurry, f-AnoGAN's are sharp, and the AE's are close copies.

**Healthy log lines look like:**
`epoch=40  loss=27.1  recon=20.6  kl=6.50  val_mse=0.00467  minutes=177.45`.
A VAE with `kl` near 0 would mean posterior collapse. Ours stays around 6.5.

---

## Step 8. Evaluate

```bash
python scripts/evaluate.py --baseline --runs runs/ae_64 runs/vae_64 runs/fanogan_64
```

**How detection works** (`src/brainad/anomaly.py`, `src/brainad/evaluate.py`):

1. Reconstruct every test slice with the model.
2. **Residual**: `max(x − x̂, 0)`. Only *brighter than expected* counts,
   because gliomas and oedema are bright on T2. The plain `|x − x̂|` is also
   computed as an ablation.
3. **Smooth** with a Gaussian (σ = 1 pixel).
4. **Mask** to the brain, eroded by 2 pixels (the brain border is where
   reconstructions are least reliable).
5. **Threshold**: try thresholds on the 48 `msd_val` subjects and keep the one
   with the best Dice. Apply that single number to the test set.
6. **Slice score**: mean of the top 1% anomaly values in the slice (for
   "does this slice contain a tumour?").

The **intensity baseline** uses an all-zero "reconstruction", so its anomaly
map is just the (masked, smoothed) image: "bright = tumour".

**Metrics:** Dice (per subject and pooled), pixel AUROC and AUPRC
(threshold-free), slice AUROC (tumour vs. tumour-free MSD slices, and MSD
tumour slices vs. healthy IXI slices), and healthy false-positive rate (share
of healthy IXI test brain pixels flagged).

**Outputs** in `results/`: `summary.md` / `summary.csv` (tables),
`comparison.png` (example maps for every model), `metrics_bar.png`,
`training_curves.png`, plus `results/<model>_64/` with `metrics.json`,
`per_subject.csv` (Dice per patient) and `examples.png`.

Evaluation took about 15 minutes for all four methods.

---

## Step 9. The results you should get

Main setting (positive residual), 436 MSD test patients:

| Model | Dice (subject mean ± sd) | Pixel AUROC | Pixel AUPRC | Slice AUROC vs. healthy | Healthy FPR |
|---|---|---|---|---|---|
| Intensity baseline | 0.394 ± 0.243 | 0.854 | 0.351 | 0.772 | 4.52% |
| AE | 0.380 ± 0.161 | 0.796 | 0.402 | 0.865 | 4.01% |
| **VAE** | **0.514 ± 0.227** | **0.894** | **0.575** | 0.929 | **1.21%** |
| f-AnoGAN | 0.474 ± 0.183 | 0.855 | 0.514 | **0.931** | 1.31% |

Your numbers should be close but not necessarily identical. Multithreaded CPU
training is not bit-for-bit deterministic, so expect differences of about
±0.02 in Dice. The *ordering* (VAE > f-AnoGAN > baseline ≈ AE on Dice) is what
matters.

**What the results mean.**

* **Generative models help.** The VAE and f-AnoGAN clearly beat the plain AE
  and the baseline.
* **The AE reconstructs too well.** It partly copies the tumour into its
  reconstruction, so the residual misses it. Low reconstruction error on
  healthy scans is *not* the goal. Reconstructing *only* healthy anatomy is.
* **VAE vs. f-AnoGAN.** The VAE's blur leaves one compact anomaly blob (best
  Dice). f-AnoGAN is sharper but misplaced fine folds create scattered false
  positives. However, it has the **fewest complete misses** (8.7% of patients
  with Dice < 0.2, vs. 11.7% for the VAE).
* **Positive residual beats absolute residual** for every model.
* **Small tumours are the main failure** (VAE Dice 0.34 on the smallest third,
  0.65 on the largest).

---

## Step 10. Figures for the README and report

The README's results section was written from `results/summary.md` and
`results/comparison.png`.

The midway report (`report/`, ICML 2026 template) has its own figures, made
from the committed `results/` folder, so this works without the data:

```bash
python report/make_figures.py
```

This writes `report/figures/{examples,curves,tumour_size}.pdf` and prints the
tumour-size numbers. Then build the PDF:

```bash
cd report
latexmk -pdf midway_report.tex
```

**What happened.** MiKTeX was set to ask before installing missing packages,
and that hidden prompt silently froze the build. Building with
`latexmk -pdf -pdflatex="pdflatex --enable-installer %O %S" midway_report.tex`
installs the missing packages automatically for that run. The final PDF is 4
pages plus 1 page of references.

---

## Step 11. Version control and GitHub

The project was committed step by step (see the table in Step 1).
`.gitignore` keeps out `data/`, `runs/` (checkpoints), virtual environments
and LaTeX build files, so everything pushed is small. To publish it:

1. Create an **empty** repository on GitHub (no README, licence or
   .gitignore).
2. Connect your local repository to it and push:

```bash
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

(If `origin` already exists, use `git remote set-url origin <url>` instead of
`add`.)

---

## One-command version

After Step 2, this runs Steps 4–8 in sequence (bash; on Windows use Git
Bash). The argument is the image size:

```bash
bash scripts/run_all.sh 64
```

It trains the models one after another, so it takes longer than running them
in parallel.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `pip install torch` finds nothing (Windows ARM) | use `--index-url https://download.pytorch.org/whl/cpu` |
| `No module named brainad` | run `pip install -e .` inside the activated environment |
| Download stops with `curl: (56)` or a connection reset | resume with `curl -C -` (Step 4), then `--extract-only` |
| Some scans have NCC near 0 | registration failed. Make sure you have the grid-search version (commit "Make registration robust…"). Plot the slices to confirm. |
| Training is very slow on Windows ARM | use the x64 Python and PyTorch build (Step 2), and give parallel runs 4 threads each |
| Out of memory during training or evaluation | close other programs, or train the models one at a time |
| LaTeX build hangs with no error | MiKTeX is waiting on a package-install prompt. Use `--enable-installer` (Step 10). |
| `latexmk` says "Nothing to do" after a failed build | delete `midway_report.aux`, `.fdb_latexmk` and `.log`, then rebuild |
| Git warns "CRLF will be replaced by LF" | harmless. `.gitattributes` stores text files with LF line endings. |
