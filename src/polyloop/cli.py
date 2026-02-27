from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from . import dataset as ds
from .bo import BoxBounds, propose_next
from .features import featurize
from .models import GPSurrogate
from .simulator import DEFAULT_BOUNDS, simulate_outcomes
from .utils import scalarise_objectives
from .vision import generate_synthetic_fiber_image, estimate_diameter_nm


app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def vision_demo(n_images: int = 6, seed: int = 0, outdir: Optional[Path] = None) -> None:
    """Generate synthetic fibre images and estimate diameters."""
    outdir = outdir or Path("outputs/vision_demo")
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    diameters = rng.uniform(250, 1200, size=n_images)

    rows = []
    for i, d in enumerate(diameters):
        img, meta = generate_synthetic_fiber_image(float(d), seed=seed + i)
        est = estimate_diameter_nm(img, nm_per_px=meta.nm_per_px)
        fname = outdir / f"fiber_{i:02d}.png"
        import cv2

        cv2.imwrite(str(fname), img)
        rows.append(
            {
                "image": fname.name,
                "true_diameter_nm": meta.diameter_nm,
                **est,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "vision_results.csv", index=False)

    table = Table(title="Vision demo results")
    for c in df.columns:
        table.add_column(c)
    for _, r in df.iterrows():
        table.add_row(*[str(r[c]) for c in df.columns])
    console.print(table)
    console.print(f"Saved results to: {outdir}")


@app.command()
def simulate(rounds: int = 5, seed: int = 0, n0: int = 20, n_candidates: int = 4096) -> None:
    """Run a closed-loop BO simulation with a synthetic experiment function."""
    df = ds.load_dataset()
    obs, _pool = ds.split_initial(df, n0=n0, seed=seed)

    bounds = BoxBounds.from_dict(DEFAULT_BOUNDS.bounds)

    # constraint: keep total polymer within a plausible window (avoid too dilute / too viscous)
    def total_polymer_ok(X: np.ndarray) -> np.ndarray:
        gelatin = X[:, bounds.names.index("gelatin_wt_pct")]
        wpu = X[:, bounds.names.index("wpu_wt_pct")]
        pei = X[:, bounds.names.index("pei_wt_pct")]
        peox = X[:, bounds.names.index("peox_wt_pct")]
        total = gelatin + wpu + pei + peox
        return (total >= 8.0) & (total <= 26.0)

    console.print(f"[bold]Initial observations:[/bold] {len(obs)} rows")
    history = []

    for t in range(rounds):
        Xdf = featurize(obs)[list(bounds.names)]
        X = Xdf.to_numpy(dtype=float)

        y = scalarise_objectives(
            diameter_nm=obs["fiber_diameter_nm"].to_numpy(),
            bead_index=obs["bead_index"].to_numpy(),
            zone_mm=obs["zone_inhibition_mm"].to_numpy(),
            viability_pct=obs["cell_viability_pct"].to_numpy(),
        )

        model = GPSurrogate(random_state=seed + t).fit(X, y)
        best_y = float(np.max(y))

        Xnext = propose_next(
            surrogate=model,
            bounds=bounds,
            best_y=best_y,
            n_candidates=n_candidates,
            n_suggestions=1,
            seed=seed + 100 + t,
            constraints=[total_polymer_ok],
        )[0]

        xdict = {k: np.array([Xnext[i]]) for i, k in enumerate(bounds.names)}
        ydict = simulate_outcomes(xdict, seed=seed + 200 + t)

        # append to obs dataframe
        new_row = {k: float(Xnext[i]) for i, k in enumerate(bounds.names)}
        for k, v in ydict.items():
            new_row[k] = float(v[0])
        obs = pd.concat([obs, pd.DataFrame([new_row])], ignore_index=True)

        score = float(
            scalarise_objectives(
                np.array([new_row["fiber_diameter_nm"]]),
                np.array([new_row["bead_index"]]),
                np.array([new_row["zone_inhibition_mm"]]),
                np.array([new_row["cell_viability_pct"]]),
            )[0]
        )

        history.append(
            {
                "round": t + 1,
                "suggested_voltage_kV": new_row["voltage_kV"],
                "suggested_flow_mL_h": new_row["flow_rate_mL_h"],
                "suggested_total_polymer_wt_pct": new_row["gelatin_wt_pct"] + new_row["wpu_wt_pct"] + new_row["pei_wt_pct"] + new_row["peox_wt_pct"],
                "score": score,
                "zone_mm": new_row["zone_inhibition_mm"],
                "viability_pct": new_row["cell_viability_pct"],
                "diameter_nm": new_row["fiber_diameter_nm"],
                "bead_index": new_row["bead_index"],
            }
        )

        console.print(f"Round {t+1}: score={score:.3f}, zone={new_row['zone_inhibition_mm']:.2f} mm, viability={new_row['cell_viability_pct']:.1f} %, diameter={new_row['fiber_diameter_nm']:.0f} nm")

    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    pd.DataFrame(history).to_csv(outdir / "closed_loop_history.csv", index=False)
    console.print(f"[green]Saved optimisation history to {outdir/'closed_loop_history.csv'}[/green]")


@app.command()
def suggest(n: int = 5, seed: int = 0) -> None:
    """Suggest next experiments from the current dataset (no simulator)."""
    df = ds.load_dataset()
    bounds = BoxBounds.from_dict(DEFAULT_BOUNDS.bounds)

    Xdf = featurize(df)[list(bounds.names)]
    X = Xdf.to_numpy(dtype=float)
    y = scalarise_objectives(
        diameter_nm=df["fiber_diameter_nm"].to_numpy(),
        bead_index=df["bead_index"].to_numpy(),
        zone_mm=df["zone_inhibition_mm"].to_numpy(),
        viability_pct=df["cell_viability_pct"].to_numpy(),
    )

    model = GPSurrogate(random_state=seed).fit(X, y)
    best_y = float(np.max(y))

    Xnext = propose_next(model, bounds=bounds, best_y=best_y, n_candidates=8192, n_suggestions=n, seed=seed)

    table = Table(title="Suggested experiments (EI maximisation)")
    for k in bounds.names:
        table.add_column(k)
    for row in Xnext:
        table.add_row(*[f"{v:.3g}" for v in row])
    console.print(table)


if __name__ == "__main__":
    app()
