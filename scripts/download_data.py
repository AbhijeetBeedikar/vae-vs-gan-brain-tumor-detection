"""Download and unpack the two public datasets used in the project.

* IXI T2 (healthy volunteers, CC BY-SA 3.0)        ~3.9 GB
  https://brain-development.org/ixi-dataset/
* MSD Task01 BrainTumour (glioma, CC BY-SA 4.0)    ~7.6 GB
  http://medicaldecathlon.com/

Usage:
    python scripts/download_data.py            # download (if missing) + extract
    python scripts/download_data.py --extract-only
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brainad.config import load_config  # noqa: E402

DATASETS = {
    "IXI-T2.tar": "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar",
    "Task01_BrainTumour.tar": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
}


def download(url: str, target: Path) -> None:
    """Stream ``url`` to ``target`` with a progress bar (skips finished files)."""
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        if target.exists() and target.stat().st_size == total:
            print(f"{target.name}: already downloaded")
            return
        tmp = target.with_suffix(".part")
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=target.name) as bar:
            while chunk := response.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        tmp.replace(target)


def extract(archive: Path, out_dir: Path) -> None:
    """Unpack the NIfTI files of a tar archive.

    The MSD archive contains macOS metadata files named ``._*``; they are not
    images, so we skip them. ``filter='data'`` blocks unsafe paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        members = [
            m for m in tf.getmembers()
            if m.isfile() and not Path(m.name).name.startswith("._")
            and (m.name.endswith(".nii.gz") or m.name.endswith(".json"))
        ]
        for m in tqdm(members, desc=f"extract {archive.name}"):
            if not (out_dir / m.name).exists():
                tf.extract(m, out_dir, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw = Path(cfg["paths"]["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)

    for name, url in DATASETS.items():
        if not args.extract_only:
            download(url, raw / name)
        # IXI files go to extracted/IXI-T2/, MSD keeps its own top folder.
        sub = "IXI-T2" if name.startswith("IXI") else ""
        extract(raw / name, Path(cfg["paths"]["extracted_dir"]) / sub)


if __name__ == "__main__":
    main()
