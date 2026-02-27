from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "electrospinning_experiments.csv"


@dataclass(frozen=True)
class DatasetSpec:
    feature_cols: Tuple[str, ...]
    target_cols: Tuple[str, ...]


DEFAULT_SPEC = DatasetSpec(
    feature_cols=(
        "gelatin_wt_pct",
        "wpu_wt_pct",
        "pei_wt_pct",
        "peox_wt_pct",
        "allantoin_wt_pct",
        "hyaluronic_acid_wt_pct",
        "voltage_kV",
        "flow_rate_mL_h",
        "tip_to_collector_cm",
        "humidity_pct",
    ),
    target_cols=(
        "fiber_diameter_nm",
        "bead_index",
        "contact_angle_deg",
        "zone_inhibition_mm",
        "cell_viability_pct",
    ),
)


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the electrospinning dataset.

    The default dataset is synthetic but constructed to be chemically plausible and
    useful for demonstrating closed-loop optimisation patterns.
    """
    p = path or DEFAULT_DATA_PATH
    return pd.read_csv(p)


def split_initial(df: pd.DataFrame, n0: int = 20, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataset into initial observations and a holdout pool."""
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df.iloc[:n0].copy(), df.iloc[n0:].copy()
