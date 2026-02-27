from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

from .models import GPSurrogate


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best_y: float, xi: float = 0.01) -> np.ndarray:
    """Expected improvement for *maximisation*.

    EI = (mu - best - xi) * Phi(z) + sigma * phi(z)
    z = (mu - best - xi) / sigma
    """
    imp = mu - best_y - xi
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


@dataclass(frozen=True)
class BoxBounds:
    lower: np.ndarray
    upper: np.ndarray
    names: Tuple[str, ...]

    @staticmethod
    def from_dict(bounds: Dict[str, Tuple[float, float]]) -> "BoxBounds":
        names = tuple(bounds.keys())
        lower = np.array([bounds[k][0] for k in names], dtype=float)
        upper = np.array([bounds[k][1] for k in names], dtype=float)
        return BoxBounds(lower=lower, upper=upper, names=names)


def random_candidates(bounds: BoxBounds, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.random((n, len(bounds.names)))
    return bounds.lower + u * (bounds.upper - bounds.lower)


def apply_constraints(X: np.ndarray, bounds: BoxBounds, constraints: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None) -> np.ndarray:
    """Filter candidates by user-defined constraints.

    Each constraint takes X and returns a boolean mask of length n.
    """
    if not constraints:
        return X
    mask = np.ones((X.shape[0],), dtype=bool)
    for c in constraints:
        mask &= c(X)
    return X[mask]


def propose_next(
    surrogate: GPSurrogate,
    bounds: BoxBounds,
    best_y: float,
    n_candidates: int = 4096,
    n_suggestions: int = 5,
    seed: int = 0,
    constraints: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,
) -> np.ndarray:
    """Propose next experiments by maximising EI over random candidates."""
    Xcand = random_candidates(bounds, n_candidates, seed=seed)
    Xcand = apply_constraints(Xcand, bounds, constraints=constraints)
    if Xcand.shape[0] == 0:
        raise ValueError("No candidates left after applying constraints.")

    mu, sigma = surrogate.predict(Xcand)
    ei = expected_improvement(mu, sigma, best_y=best_y)
    top_idx = np.argsort(-ei)[:n_suggestions]
    return Xcand[top_idx]
