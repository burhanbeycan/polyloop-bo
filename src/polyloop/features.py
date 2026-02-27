from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    add_total_polymer: bool = True
    add_pei_fraction: bool = True
    add_additive_total: bool = True


def featurize(df: pd.DataFrame, cfg: FeatureEngineeringConfig = FeatureEngineeringConfig()) -> pd.DataFrame:
    """Return an engineered feature dataframe.

    The feature set is deliberately simple and interpretable:
    - total polymer loading (gelatin + WPU + PEI + PEOx)
    - PEI fraction within polymer phase
    - total 'bio-additive' loading (allantoin + hyaluronic acid)
    """
    X = df.copy()

    polymer_cols = ["gelatin_wt_pct", "wpu_wt_pct", "pei_wt_pct", "peox_wt_pct"]
    if cfg.add_total_polymer:
        X["total_polymer_wt_pct"] = X[polymer_cols].sum(axis=1)

    if cfg.add_pei_fraction:
        denom = X[polymer_cols].sum(axis=1).replace(0, np.nan)
        X["pei_frac_polymer"] = (X["pei_wt_pct"] / denom).fillna(0.0)

    if cfg.add_additive_total:
        X["bio_additives_wt_pct"] = X[["allantoin_wt_pct", "hyaluronic_acid_wt_pct"]].sum(axis=1)

    return X
