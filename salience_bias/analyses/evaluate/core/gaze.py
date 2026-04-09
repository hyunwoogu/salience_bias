from __future__ import annotations

from typing import Any, Mapping, MutableMapping

import numpy as np


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


def _pair_param_np(x: Any, *, name: str = "param") -> tuple[float, float]:
    """Return a length-2 pair from a scalar or length-2 input (NumPy-only)."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    if arr.size == 2:
        return float(arr[0]), float(arr[1])
    raise ValueError(f"Expected {name} to be scalar or length-2, got length {arr.size}")


def _compute_mesh_centers_np(
    h_range: tuple[float, float],
    w_range: tuple[float, float],
    h_nbin: int,
    w_nbin: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute (ww, hh) mesh of bin centers, matching the legacy `gauss_gaze` convention.

    Note: `h_range` may be decreasing (e.g. (17.5, -17.5)); we preserve that ordering.
    """
    h_edges = np.linspace(float(h_range[0]), float(h_range[1]), int(h_nbin) + 1, dtype=float)
    w_edges = np.linspace(float(w_range[0]), float(w_range[1]), int(w_nbin) + 1, dtype=float)
    h_centers = 0.5 * (h_edges[:-1] + h_edges[1:])
    w_centers = 0.5 * (w_edges[:-1] + w_edges[1:])
    ww, hh = np.meshgrid(w_centers, h_centers)  # shapes [H, W]
    return ww, hh


def _gauss_gaze_np(
    g_hw: np.ndarray,
    *,
    h_range: tuple[float, float],
    w_range: tuple[float, float],
    h_nbin: int,
    w_nbin: int,
    sigma_hw: Any = (1.0, 1.0),
    normalize: bool = True,
) -> np.ndarray:
    """
    NumPy implementation of the legacy `gauss_gaze` kernel encoding.

    - `g_hw`: gaze points with shape [..., 2], coordinates in the same units as ranges.
      Convention: g_hw[..., 0] is h (y), g_hw[..., 1] is w (x).
    - Returns: array with shape [..., H, W]
    """
    g_hw = np.asarray(g_hw, dtype=float)
    if g_hw.shape == (0,):
        g_hw = g_hw.reshape(0, 2)
    if g_hw.ndim < 1 or g_hw.shape[-1] != 2:
        raise ValueError(f"gaze points must have shape [..., 2], got {g_hw.shape}")

    sigma_h, sigma_w = _pair_param_np(sigma_hw, name="sigma_hw")
    ww, hh = _compute_mesh_centers_np(h_range, w_range, int(h_nbin), int(w_nbin))

    # Broadcast to [..., H, W]
    dh = hh[None, :, :] - g_hw[..., 0][..., None, None]
    dw = ww[None, :, :] - g_hw[..., 1][..., None, None]
    gauss = np.exp(-0.5 * ((dh / sigma_h) ** 2 + (dw / sigma_w) ** 2))

    if bool(normalize):
        bin_h = (float(h_range[1]) - float(h_range[0])) / float(h_nbin)
        bin_w = (float(w_range[1]) - float(w_range[0])) / float(w_nbin)
        area_bin = abs(bin_h * bin_w)
        gauss = gauss * (area_bin / (2.0 * np.pi * sigma_h * sigma_w))

    return np.asarray(gauss, dtype=np.float64)


def make_gaze_map(
    eye_pos: Mapping[str, Any] | Any,
    *,
    image_shape: tuple[int, int],
    sigma: float | None = None,
    gauss_params: MutableMapping[str, Any] | None = None,
) -> np.ndarray:
    """
    Encode gaze points into a [H, W] Gaussian field.

    This is a public subset implementation intended to match the legacy `gazebo.utils.gauss_gaze`
    behavior used in the original (private) codebase: evaluate a Gaussian at **bin centers**
    for each fixation, then average across fixations.

    `eye_pos` may be:
      - dict with keys 'x' and 'y' (1d arrays or lists of equal length)
      - any object with .x and .y array attributes

    Supported `gauss_params` keys (all optional):
      - h_nbin, w_nbin: output grid size (defaults to image_shape)
      - h_range: (max, min) or (min, max) range for y/height coordinates
      - w_range: (min, max) or (max, min) range for x/width coordinates
      - sigma_hw: (sigma_h, sigma_w) in the same units as `h_range`/`w_range`
      - normalize: if True, apply continuous-Gaussian normalization so the Riemann sum over the
        grid approximates 1 when mass lies within the domain (default False here to preserve
        earlier evaluation usage patterns in this repo).

    Compatibility notes:
      - If `sigma` is passed, it is treated as a scalar `sigma_hw=(sigma, sigma)` in coordinate units.
    """
    h_img, w_img = image_shape
    params = dict(gauss_params) if gauss_params is not None else {}

    h = int(params.get("h_nbin", h_img))
    w = int(params.get("w_nbin", w_img))

    # Original code uses h_range=(h/2, -h/2) and w_range=(-w/2, w/2)
    h_range = params.get("h_range", (h / 2, -h / 2))
    w_range = params.get("w_range", (-w / 2, w / 2))

    xv, yv = _extract_xy(eye_pos)
    x_flat = xv.reshape(-1)
    y_flat = yv.reshape(-1)
    if x_flat.size == 0:
        return np.zeros((h, w), dtype=np.float64)

    pts_hw = np.stack([y_flat, x_flat], axis=-1)  # (h, w) = (y, x)

    # sigma compatibility: treat `sigma` argument as sigma_hw scalar in coordinate units
    sigma_hw = params.get("sigma_hw", None)
    if sigma_hw is None:
        if sigma is not None:
            sigma_hw = (float(sigma), float(sigma))
        else:
            sigma_hw = (1.0, 1.0)

    gaze_stack = _gauss_gaze_np(
        pts_hw,
        h_range=(float(h_range[0]), float(h_range[1])),
        w_range=(float(w_range[0]), float(w_range[1])),
        h_nbin=h,
        w_nbin=w,
        sigma_hw=sigma_hw,
        normalize=bool(params.get("normalize", False)),
    )

    # gaze_stack is [T, H, W]; match evaluation API contract: return [H, W] average map
    return np.mean(gaze_stack, axis=0, dtype=np.float64)
