"""Regenerate synthetic PolyLoop-BO dataset and images.

This script is designed so you can:
- reproduce the synthetic dataset
- adjust bounds or noise models
- generate new images for new experiments

Usage:
    python scripts/generate_data.py --n 200 --seed 42
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from skimage.io import imsave

from polyloop.simulator import DEFAULT_BOUNDS, simulate_outcomes
from polyloop.vision import generate_synthetic_fiber_image

app = typer.Typer(add_completion=False)


@app.command()
def main(n: int = 200, seed: int = 42, n_images: int = 40) -> None:
    out_csv = Path("data/electrospinning_experiments.csv")
    img_dir = Path("data/fiber_images")
    img_dir.mkdir(parents=True, exist_ok=True)

    params = DEFAULT_BOUNDS.sample(n, seed=seed)
    out = simulate_outcomes(params, seed=seed + 100)
    df = pd.DataFrame({**params, **out})
    df.insert(0, "experiment_id", np.arange(1, len(df) + 1))
    df.to_csv(out_csv, index=False)

    chosen = df.sample(n=min(n_images, len(df)), random_state=seed)
    idx_rows = []
    rng = np.random.default_rng(seed + 7)
    for _, row in chosen.iterrows():
        exp_id = int(row["experiment_id"])
        d_nm = float(row["fiber_diameter_nm"])
        d_img = float(np.clip(d_nm + rng.normal(0, 40), 120, 2200))
        nm_per_px = 30.0
        img, meta = generate_synthetic_fiber_image(d_img, nm_per_px=nm_per_px, seed=1000 + exp_id, n_fibers=55)
        fname = f"exp_{exp_id:04d}.png"
        imsave(str(img_dir / fname), img)
        idx_rows.append(
            {"experiment_id": exp_id, "image_file": fname, "nm_per_px": nm_per_px, "true_diameter_nm": d_img}
        )

    pd.DataFrame(idx_rows).to_csv(img_dir / "index.csv", index=False)
    print(f"Wrote: {out_csv} and {img_dir/'index.csv'}")


if __name__ == "__main__":
    app()
