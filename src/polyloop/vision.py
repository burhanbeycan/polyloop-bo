from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class FiberImageMeta:
    nm_per_px: float
    diameter_nm: float
    seed: int
    n_fibers: int
    img_size: int


def generate_synthetic_fiber_image(
    diameter_nm: float,
    nm_per_px: float = 30.0,
    img_size: int = 512,
    n_fibers: int = 45,
    seed: int = 0,
) -> Tuple[np.ndarray, FiberImageMeta]:
    """Generate a synthetic SEM-like grayscale image of fibres.

    The goal is not photo-realism; it's to provide a dataset for developing and
    testing automated diameter-measurement pipelines.
    """
    rng = np.random.default_rng(seed)
    img = rng.normal(loc=25, scale=8, size=(img_size, img_size)).astype(np.float32)

    thickness_px = max(1, int(round(diameter_nm / nm_per_px)))
    for _ in range(n_fibers):
        x0, y0 = rng.integers(0, img_size, size=2)
        angle = rng.uniform(0, np.pi)
        length = rng.integers(int(0.6 * img_size), int(1.2 * img_size))
        x1 = int(x0 + length * np.cos(angle))
        y1 = int(y0 + length * np.sin(angle))
        col = float(rng.uniform(150, 230))
        cv2.line(img, (x0, y0), (x1, y1), col, thickness=thickness_px, lineType=cv2.LINE_AA)

    # blur + noise to mimic imaging
    img = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=1.2, sigmaY=1.2)
    img += rng.normal(loc=0, scale=10, size=img.shape).astype(np.float32)

    img = np.clip(img, 0, 255).astype(np.uint8)
    meta = FiberImageMeta(nm_per_px=nm_per_px, diameter_nm=diameter_nm, seed=seed, n_fibers=n_fibers, img_size=img_size)
    return img, meta


def estimate_diameter_nm(img: np.ndarray, nm_per_px: float = 30.0) -> Dict[str, float]:
    """Estimate fibre diameter using skeleton + distance transform.

    Method:
    1) Otsu threshold to get a fibre mask
    2) skeletonise mask
    3) distance transform gives radius at each pixel
    4) sample radii along skeleton → diameter distribution

    Returns mean/std (nm) and a few quality diagnostics.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # light denoise
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold; fibres are brighter
    _, mask = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_bool = mask.astype(bool)

    # skeletonize expects boolean
    skel = skeletonize(mask_bool)

    # distance transform on fibre mask -> radius in px
    dist = distance_transform_edt(mask_bool)

    radii_px = dist[skel]
    radii_px = radii_px[(radii_px > 0.5) & (radii_px < 200)]  # mild sanity filter

    if radii_px.size < 50:
        return {
            "diameter_mean_nm": float("nan"),
            "diameter_std_nm": float("nan"),
            "n_samples": int(radii_px.size),
            "mask_coverage": float(mask_bool.mean()),
        }

    diam_nm = 2.0 * radii_px * nm_per_px
    return {
        "diameter_mean_nm": float(np.mean(diam_nm)),
        "diameter_std_nm": float(np.std(diam_nm)),
        "n_samples": int(diam_nm.size),
        "mask_coverage": float(mask_bool.mean()),
    }
