"""Quickstart script for PolyLoop-BO.

Run:
    python examples/quickstart.py

This will:
- load the dataset
- train a surrogate
- print 5 suggested experiments
"""

import numpy as np

from polyloop.dataset import load_dataset
from polyloop.features import featurize
from polyloop.models import GPSurrogate
from polyloop.bo import BoxBounds, propose_next
from polyloop.simulator import DEFAULT_BOUNDS
from polyloop.utils import scalarise_objectives


def main() -> None:
    df = load_dataset()
    bounds = BoxBounds.from_dict(DEFAULT_BOUNDS.bounds)

    Xdf = featurize(df)[list(bounds.names)]
    X = Xdf.to_numpy(float)

    y = scalarise_objectives(
        df["fiber_diameter_nm"].to_numpy(),
        df["bead_index"].to_numpy(),
        df["zone_inhibition_mm"].to_numpy(),
        df["cell_viability_pct"].to_numpy(),
    )

    model = GPSurrogate(random_state=0).fit(X, y)
    Xnext = propose_next(model, bounds, best_y=float(np.max(y)), n_candidates=4096, n_suggestions=5, seed=0)

    print("Suggested experiments (first two rows shown):")
    print(Xnext[:2])


if __name__ == "__main__":
    main()
