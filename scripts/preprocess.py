"""Preprocess IXI (healthy) and MSD (tumour) T2 scans into 2D slice arrays.

Runs :func:`brainad.preprocessing.process_volume` on every scan in parallel
and writes one file per dataset:

    data/processed/ixi.npz   healthy slices (training / validation / healthy test)
    data/processed/msd.npz   tumour slices with expert tumour masks

Each .npz holds
    image   (N, S, S) float16   normalised T2 slice, 0 outside the brain
    brain   (N, S, S) bool      brain mask
    label   (N, S, S) bool      whole-tumour mask (all False for IXI)
    subject (N,)      int32     index into ``subjects``
    z       (N,)      int16     axial level in template space
    subjects (M,)     str       scan IDs
    ncc     (M,)      float32   registration quality per scan

Usage:
    python scripts/preprocess.py [--limit 10]
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brainad.config import load_config  # noqa: E402
from brainad.preprocessing import MSD_CHANNELS, process_volume  # noqa: E402

CFG: dict = {}


def _init_worker(cfg: dict) -> None:
    """Each worker process gets the config and uses 2 CPU threads."""
    import torch

    torch.set_num_threads(2)
    CFG.update(cfg)


def _process_job(job: tuple) -> tuple:
    """Run the pipeline for one scan; errors are returned, not raised."""
    subject, image_path, label_path, channel, skull_stripped = job
    try:
        out = process_volume(image_path, label_path, CFG, channel=channel, already_skull_stripped=skull_stripped)
        return subject, out, None
    except Exception as exc:  # a corrupt file should not kill a 1-hour job
        return subject, None, repr(exc)


def ixi_jobs(extracted: Path) -> list[tuple]:
    files = sorted((extracted / "IXI-T2").glob("IXI*-T2.nii.gz"))
    # Subject ID = "IXI002" etc. (one T2 scan per subject in IXI).
    return [(f.name.split("-")[0], str(f), None, None, False) for f in files]


def msd_jobs(extracted: Path) -> list[tuple]:
    root = extracted / "Task01_BrainTumour"
    jobs = []
    for f in sorted((root / "imagesTr").glob("BRATS_*.nii.gz")):
        label = root / "labelsTr" / f.name
        # MSD scans are already skull-stripped (BraTS); we use the T2 channel.
        jobs.append((f.name.replace(".nii.gz", ""), str(f), str(label), MSD_CHANNELS["T2w"], True))
    return jobs


def run(name: str, jobs: list[tuple], cfg: dict, out_dir: Path) -> None:
    """Process all scans of one dataset and save the stacked slices."""
    results, failures = {}, []
    start = time.time()
    with Pool(cfg["preprocessing"]["num_workers"], initializer=_init_worker, initargs=(cfg,)) as pool:
        for subject, out, err in tqdm(pool.imap_unordered(_process_job, jobs), total=len(jobs), desc=name):
            if err is None:
                results[subject] = out
            else:
                failures.append((subject, err))

    subjects = sorted(results)  # deterministic order regardless of finishing order
    arrays = {k: [] for k in ("image", "brain", "label", "subject", "z")}
    for idx, s in enumerate(subjects):
        r = results[s]
        n = len(r["image"])
        arrays["image"].append(r["image"].astype(np.float16))  # halves disk/RAM use
        arrays["brain"].append(r["brain"])
        arrays["label"].append(r["label"])
        arrays["subject"].append(np.full(n, idx, dtype=np.int32))
        arrays["z"].append(r["z"].astype(np.int16))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"{name}.npz",
        **{k: np.concatenate(v) for k, v in arrays.items()},
        subjects=np.array(subjects),
        ncc=np.array([results[s]["ncc"] for s in subjects], dtype=np.float32),
    )
    print(f"{name}: {len(subjects)} scans OK, {len(failures)} failed, {time.time() - start:.0f}s")
    for subject, err in failures:
        print(f"  FAILED {subject}: {err}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N scans per dataset (for testing)")
    parser.add_argument("--datasets", nargs="+", default=["ixi", "msd"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    extracted = Path(cfg["paths"]["extracted_dir"])
    out_dir = Path(cfg["paths"]["processed_dir"])
    builders = {"ixi": ixi_jobs, "msd": msd_jobs}
    for name in args.datasets:
        jobs = builders[name](extracted)[: args.limit]
        if not jobs:
            sys.exit(f"No {name} scans found under {extracted}; run scripts/download_data.py first.")
        run(name, jobs, cfg, out_dir)


if __name__ == "__main__":
    main()
