from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class ParamBounds:
    bounds: Dict[str, Tuple[float, float]]

    def sample(self, n: int, seed: int = 0) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        out: Dict[str, np.ndarray] = {}
        for k, (lo, hi) in self.bounds.items():
            out[k] = lo + rng.random(n) * (hi - lo)
        return out


DEFAULT_BOUNDS = ParamBounds(
    bounds={
        "gelatin_wt_pct": (0.0, 18.0),
        "wpu_wt_pct": (0.0, 18.0),
        "pei_wt_pct": (0.0, 4.0),
        "peox_wt_pct": (0.0, 10.0),
        "allantoin_wt_pct": (0.0, 2.5),
        "hyaluronic_acid_wt_pct": (0.0, 1.5),
        "voltage_kV": (10.0, 25.0),
        "flow_rate_mL_h": (0.2, 1.5),
        "tip_to_collector_cm": (10.0, 20.0),
        "humidity_pct": (20.0, 65.0),
    }
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_outcomes(params: Dict[str, np.ndarray], seed: int = 0) -> Dict[str, np.ndarray]:
    """Synthetic ground-truth mapping from electrospinning settings to outcomes.

    The functional form is constructed to be:
    - nonlinear
    - noisy
    - multi-objective with trade-offs (e.g., antimicrobial vs cytocompatibility)

    Replace this with your real lab measurements in a production system.
    """
    rng = np.random.default_rng(seed)

    gelatin = params["gelatin_wt_pct"]
    wpu = params["wpu_wt_pct"]
    pei = params["pei_wt_pct"]
    peox = params["peox_wt_pct"]
    allantoin = params["allantoin_wt_pct"]
    ha = params["hyaluronic_acid_wt_pct"]
    voltage = params["voltage_kV"]
    flow = params["flow_rate_mL_h"]
    distance = params["tip_to_collector_cm"]
    humidity = params["humidity_pct"]

    total_poly = gelatin + wpu + pei + peox

    # Fibre diameter (nm): increases with polymer loading and flow; decreases with voltage.
    diameter = (
        50 * total_poly
        + 180 * (flow - 0.2)
        - 28 * (voltage - 15)
        + 10 * (distance - 15)
        + 1.2 * (humidity - 40)
        + rng.normal(0, 50, size=total_poly.shape)
    )
    diameter = np.clip(diameter, 120, 2200)

    # Bead index (0-1): worse at high humidity, high flow, low polymer loading.
    bead_raw = 0.10 * (humidity - 40) - 0.22 * (total_poly - 14) + 0.9 * (flow - 0.8) + rng.normal(0, 0.25, size=total_poly.shape)
    bead = _sigmoid(bead_raw)
    bead = np.clip(bead, 0, 1)

    # Contact angle (deg): WPU increases hydrophobicity; gelatin/PEI/HA decreases.
    contact = (
        78
        + 2.2 * (wpu - 8)
        - 1.6 * gelatin
        - 2.5 * pei
        - 8.0 * ha
        + 0.9 * allantoin
        + rng.normal(0, 4.5, size=total_poly.shape)
    )
    contact = np.clip(contact, 0, 130)

    # Antimicrobial zone (mm): PEI drives activity; morphology (beads) can reduce effective contact.
    zone = (
        2.0
        + 4.2 * pei
        + 0.6 * gelatin
        + 0.4 * allantoin
        - 1.2 * bead
        + 0.2 * np.sin(0.25 * voltage)
        + rng.normal(0, 1.2, size=total_poly.shape)
    )
    zone = np.clip(zone, 0, 25)

    # Cytocompatibility (%): PEI lowers viability; HA/allantoin can help; beads penalise.
    viability = (
        92
        - 8.0 * pei
        + 7.0 * ha
        + 3.5 * allantoin
        - 12.0 * bead
        - 12.0 * ((diameter - 600) / 600) ** 2  # penalty if far from 600 nm
        + rng.normal(0, 6, size=total_poly.shape)
    )
    viability = np.clip(viability, 0, 100)

    return {
        "fiber_diameter_nm": diameter,
        "bead_index": bead,
        "contact_angle_deg": contact,
        "zone_inhibition_mm": zone,
        "cell_viability_pct": viability,
    }
