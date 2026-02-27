from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel


@dataclass
class GPSurrogate:
    """A light-weight Gaussian process surrogate using scikit-learn.

    This is intentionally dependency-light, but the interface mirrors what you'd
    typically do with BoTorch/GPyTorch.
    """

    alpha: float = 1e-6
    normalize_y: bool = True
    random_state: int = 0

    def __post_init__(self) -> None:
        kernel = C(1.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=3,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPSurrogate":
        self.gp.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean, std = self.gp.predict(X, return_std=True)
        std = np.maximum(std, 1e-9)
        return mean, std
