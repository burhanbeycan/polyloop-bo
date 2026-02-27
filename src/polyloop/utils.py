from __future__ import annotations

import numpy as np


def scalarise_objectives(
    diameter_nm: np.ndarray,
    bead_index: np.ndarray,
    zone_mm: np.ndarray,
    viability_pct: np.ndarray,
    target_diameter_nm: float = 600.0,
) -> np.ndarray:
    """Convert multi-objective outputs into a single scalar score.

    This provides a simple baseline for BO. In publications you would typically:
    - use Pareto-front methods (qEHVI, qNEHVI)
    - or preference learning

    Score encourages:
    - high antimicrobial zone
    - high viability
    - low bead index
    - diameter close to target
    """
    zone_n = (zone_mm - 0.0) / 25.0
    viab_n = viability_pct / 100.0
    bead_pen = bead_index
    diam_pen = np.abs(diameter_nm - target_diameter_nm) / target_diameter_nm

    score = 1.2 * zone_n + 1.0 * viab_n - 0.7 * bead_pen - 0.6 * diam_pen
    return score
