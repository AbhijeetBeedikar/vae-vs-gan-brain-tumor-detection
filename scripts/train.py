"""Train one model on healthy IXI slices.

Usage:
    python scripts/train.py --config configs/vae.yaml
    python scripts/train.py --config configs/ae.yaml --set training.epochs=5 data.image_size=128

Outputs go to runs/<model>_<image_size>/ (checkpoint, history.json, figures).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brainad.config import deep_merge, load_config, parse_override  # noqa: E402
from brainad.data import load_split  # noqa: E402
from brainad.train import train_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. training.epochs=5")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    overrides: dict = {}
    for item in args.set:
        overrides = deep_merge(overrides, parse_override(item))
    cfg = load_config(args.config, overrides)

    run_dir = args.run_dir or Path(cfg["paths"]["runs_dir"]) / f"{cfg['model']['name']}_{cfg['data']['image_size']}"
    train_set = load_split(cfg, "ixi_train")
    val_set = load_split(cfg, "ixi_val")
    train_model(cfg, train_set, val_set, run_dir)


if __name__ == "__main__":
    main()
