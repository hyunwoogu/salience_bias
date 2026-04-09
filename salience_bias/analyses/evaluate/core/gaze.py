from __future__ import annotations

from typing import Any, Mapping, MutableMapping

import numpy as np
from scipy.ndimage import gaussian_filter


def _extract_xy(eye_pos: Mapping[str, Any] | Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(eye_pos, Mapping):
        xv = np.asarray(eye_pos["x"], dtype=float)
        yv = np.asarray(eye_pos["y"], dtype=float)
    else:
        xv = np.asarray(eye_pos.x, dtype=float)
        yv = np.asarray(eye_pos.y, dtype=float)
    if xv.shape != yv.shape:
        raise ValueError(f"eye_pos x/y must have same shape, got {xv.shape} and {yv.shape}")
    return xv, yv


def make_gaze_map(
    eye_pos: Mapping[str, Any] | Any,
    *,
    image_shape: tuple[int, int],
    sigma: float | None = None,
    gauss_params: MutableMapping[str, Any] | None = None,
) -> np.ndarray:
    """
    Blur raw gaze points into a density map [H, W].

    Public subset implementation that avoids internal dependencies.

    `eye_pos` may be:
      - dict with keys 'x' and 'y' (1d arrays or lists of equal length)
      - any object with .x and .y array attributes

    Supported `gauss_params` keys (all optional):
      - h_nbin, w_nbin: output grid size (defaults to image_shape)
      - h_range: (max, min) in the y-coordinate system (matches original code convention)
      - w_range: (min, max) in the x-coordinate system
      - sigma: Gaussian blur sigma in *bins*
      - normalize: if True, normalize map to sum to 1 (default False)
    """
    h_img, w_img = image_shape
    params = dict(gauss_params) if gauss_params is not None else {}

    h = int(params.get("h_nbin", h_img))
    w = int(params.get("w_nbin", w_img))

    # Original code uses h_range=(h/2, -h/2) and w_range=(-w/2, w/2)
    h_range = params.get("h_range", (h / 2, -h / 2))
    w_range = params.get("w_range", (-w / 2, w / 2))

    # Convert to numpy.histogram2d expected (min, max) ranges.
    y_min = float(min(h_range[0], h_range[1]))
    y_max = float(max(h_range[0], h_range[1]))
    x_min = float(min(w_range[0], w_range[1]))
    x_max = float(max(w_range[0], w_range[1]))

    xv, yv = _extract_xy(eye_pos)

    # histogram2d returns shape [y_bins, x_bins]
    hist, _, _ = np.histogram2d(
        yv.reshape(-1),
        xv.reshape(-1),
        bins=(h, w),
        range=((y_min, y_max), (x_min, x_max)),
    )

    sig = sigma
    if sig is None:
        sig = params.get("sigma", None)
    if sig is None:
        # Light default blur; callers should pass explicit sigma/gauss_params for real experiments.
        sig = 1.0

    out = gaussian_filter(hist.astype(np.float64), sigma=float(sig), mode="constant")

    if bool(params.get("normalize", False)):
        s = float(out.sum())
        if s > 0:
            out = out / s

    return out
