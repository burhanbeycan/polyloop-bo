import numpy as np
import pandas as pd

from polyloop.dataset import load_dataset
from polyloop.simulator import DEFAULT_BOUNDS, simulate_outcomes
from polyloop.models import GPSurrogate
from polyloop.bo import BoxBounds, propose_next
from polyloop.utils import scalarise_objectives
from polyloop.vision import generate_synthetic_fiber_image, estimate_diameter_nm


def test_dataset_loads():
    df = load_dataset()
    assert len(df) > 50
    assert "fiber_diameter_nm" in df.columns


def test_simulator_shapes():
    params = DEFAULT_BOUNDS.sample(10, seed=0)
    out = simulate_outcomes(params, seed=1)
    assert out["fiber_diameter_nm"].shape == (10,)


def test_bo_propose_next():
    # Train a tiny surrogate on synthetic data
    params = DEFAULT_BOUNDS.sample(30, seed=3)
    out = simulate_outcomes(params, seed=4)

    bounds = BoxBounds.from_dict(DEFAULT_BOUNDS.bounds)
    X = np.column_stack([params[k] for k in bounds.names])
    y = scalarise_objectives(
        out["fiber_diameter_nm"],
        out["bead_index"],
        out["zone_inhibition_mm"],
        out["cell_viability_pct"],
    )

    model = GPSurrogate(random_state=0).fit(X, y)
    Xnext = propose_next(model, bounds=bounds, best_y=float(np.max(y)), n_candidates=512, n_suggestions=3, seed=0)
    assert Xnext.shape == (3, len(bounds.names))


def test_vision_estimator_returns_values():
    img, meta = generate_synthetic_fiber_image(diameter_nm=600, seed=0)
    est = estimate_diameter_nm(img, nm_per_px=meta.nm_per_px)
    assert "diameter_mean_nm" in est
    assert est["n_samples"] > 0
