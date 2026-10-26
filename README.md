# Learning Healthy Anatomy to Find Tumours

**Comparing VAEs and GANs for unsupervised anomaly detection in brain MRI**
STAT 453 project: Jenny Brant, Abhijeet Beedikar, Diya Kothari, Daksh Kacham

We train generative models **only on healthy brains** (IXI, T2-weighted) and
ask whether they can find and outline brain tumours (MSD Task01 BrainTumour /
BraTS gliomas) that they have never seen. Each model learns to reconstruct
healthy anatomy. Shown a scan with a tumour, it "repairs" it toward a healthy
brain, so the places where the reconstruction differs from the input mark the
anomaly. No tumour labels are used for training.

Models compared (all with DCGAN-style convolutional layers, 128-d latent code):

| Model | What it is | Role |
|---|---|---|
| **VAE** | Variational autoencoder (Kingma & Welling 2014) | generative model #1 |
| **f-AnoGAN** | WGAN-GP + encoder into its latent space (Schlegl et al. 2019) | generative model #2 |
| **AE** | Plain deterministic autoencoder | baseline: does generative modelling help? |
| **Intensity** | "bright on T2 = tumour" (no learning) | sanity-check baseline |

<!-- RESULTS -->

## Repository layout

```
configs/            base.yaml (shared settings) + one file per model
scripts/
  download_data.py  download + unpack IXI and MSD
  preprocess.py     3D scans -> registered, normalised 2D slices (parallel)
  train.py          train one model on healthy IXI slices
  evaluate.py       metrics, tables and figures for trained models
  run_all.sh        the full pipeline end to end
src/brainad/
  template.py       synthetic T2 MNI152 template + brain mask
  registration.py   affine registration in PyTorch
  preprocessing.py  skull-strip, register, normalise, slice
  data.py           subject-level splits, in-memory batches
  models/           networks.py (DCGAN blocks), autoencoders.py (AE, VAE), fanogan.py
  train.py          training loops
  anomaly.py        residual -> smoothed, masked anomaly map; slice scores
  metrics.py        Dice, pixel AUROC/AUPRC, slice AUROC, healthy FPR
  evaluate.py       full evaluation of one model
  visualize.py      figures
tests/              unit tests, registration test, synthetic end-to-end test
results/            committed metrics and figures
```

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -r requirements.txt && pip install -e .
pytest                                  # ~1 min, needs no data
bash scripts/run_all.sh 64              # download (~11.5 GB) -> preprocess -> train -> evaluate
```

Or step by step:

```bash
python scripts/download_data.py
python scripts/preprocess.py
python scripts/train.py --config configs/vae.yaml            # also ae.yaml, fanogan.yaml
python scripts/evaluate.py --baseline --runs runs/ae_64 runs/vae_64 runs/fanogan_64
```

Any config value can be overridden from the command line, e.g.
`--set data.image_size=128 training.epochs=60`.

## Data

| Dataset | Use | Subjects | Licence |
|---|---|---|---|
| [IXI](https://brain-development.org/ixi-dataset/) T2 | training (healthy) | 578 | CC BY-SA 3.0 |
| [MSD Task01 BrainTumour](http://medicaldecathlon.com/) T2 (from BraTS) | threshold selection + testing | 484 | CC BY-SA 4.0 |

Splits are by subject (`data/processed/splits.json`): IXI 80/10/10
(train / validation / healthy test); MSD 10% for choosing the detection
threshold, 90% for testing.

## Method

### Preprocessing (identical for both datasets)

The two datasets come from different scanners, so without careful matching a
model could flag "different hospital" instead of "tumour". Every scan goes
through `src/brainad/preprocessing.py`:

1. **Reorient** to RAS and roughly scale intensities.
2. **Register** to the MNI152 (ICBM 2009a) template with a 12-parameter affine
   transform. SimpleITK/ANTs have no Windows-on-ARM builds, so
   `registration.py` implements this in PyTorch: trilinear resampling via
   `grid_sample`, a normalised cross-correlation loss, Adam, two resolution
   levels (4 mm then 2 mm). Because nilearn only ships a *T1* template, we
   synthesise a T2-like template from its grey/white-matter maps
   (CSF bright, GM medium, WM dark; `template.py`). A unit test checks that a
   known transform is recovered to well under 1 mm.
3. **Skull-strip** with the template brain mask, the same mask for every
   subject. MSD scans are already skull-stripped, so for them we also
   intersect with their own brain outline.
4. **Normalise**: divide by the 99th percentile of brain intensities, clip to [0, 1].
5. **Slice**: keep axial levels whose brain area is at least 30% of the largest
   (skips near-empty top/bottom slices), every 2nd level, cropped square
   around the brain and resized with area averaging to 128×128. The 64×64
   versions are made from these at load time.

### Detection

For each test slice the model produces a healthy-looking reconstruction
(VAE: decoded posterior mean; f-AnoGAN: G(E(x)); AE: D(E(x))). The anomaly map is

1. the **positive residual** `max(x − x̂, 0)` (gliomas and oedema are bright on T2),
2. smoothed with a Gaussian (σ = 1 pixel),
3. restricted to the brain mask eroded by 2 pixels (reconstructions are least
   reliable at the brain border).

A pixel is called tumour if its score exceeds a threshold chosen to maximise
Dice on the MSD validation subset. A slice's score is the mean of its top 1% of
anomaly values. The plain absolute residual `|x − x̂|` is reported as well, as
an ablation of this post-processing choice.

### Metrics

* **Dice** between predicted and expert whole-tumour masks: global, and mean ± sd over test subjects
* **Pixel AUROC / AUPRC** of the raw anomaly scores (threshold-free)
* **Slice AUROC**: tumour vs non-tumour slices within MSD, and MSD tumour slices vs healthy IXI slices
* **Healthy false-positive rate**: fraction of healthy IXI test brain pixels flagged at the chosen threshold

## Limitations

* Skull stripping of IXI uses the affinely registered template mask rather
  than a dedicated tool such as HD-BET or ROBEX (none run on this machine),
  so brain borders are approximate. This is partly offset by eroding the mask
  for scoring and by the positive residual, which ignores dark borders.
* Affine registration aligns overall brain shape but not individual folds.
* Models are 2D and trained on a CPU, at 64×64 (the proposal's
  "fast experiments" setting), so fine detail is limited.
* MSD (BraTS) T2 scans are mixed-quality, multi-centre and pre-processed
  differently from IXI, so some of the residual comes from domain shift
  rather than pathology.

## References

1. D. P. Kingma, M. Welling. Auto-Encoding Variational Bayes. ICLR 2014.
2. I. Goodfellow et al. Generative Adversarial Nets. NeurIPS 2014.
3. A. Radford, L. Metz, S. Chintala. Unsupervised Representation Learning with DCGANs. ICLR 2016.
4. T. Schlegl et al. Unsupervised Anomaly Detection with GANs to Guide Marker Discovery (AnoGAN). IPMI 2017.
5. T. Schlegl et al. f-AnoGAN: Fast Unsupervised Anomaly Detection with GANs. Medical Image Analysis 54, 2019.
6. C. Baur et al. Deep Autoencoding Models for Unsupervised Anomaly Segmentation in Brain MR Images. BrainLes 2018.
7. C. Baur et al. Autoencoders for Unsupervised Anomaly Segmentation in Brain MR Images: A Comparative Study. MedIA 69, 2021.
8. M. Antonelli et al. The Medical Segmentation Decathlon. Nature Communications 13, 2022.
9. IXI Dataset, Imperial College London, brain-development.org.
10. I. Gulrajani et al. Improved Training of Wasserstein GANs (WGAN-GP). NeurIPS 2017.
