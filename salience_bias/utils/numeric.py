from __future__ import annotations

from typing import Any


def _get_xp(x: Any, xp: Any | None, pytorch: bool):
    if xp is not None:
        return xp
    if pytorch:
        import torch  # local import (optional dependency)

        return torch
    return __import__("numpy")


def zscore(
    x,
    *,
    axis=None,
    eps: float = 1e-8,
    ddof: int = 0,
    keepdims: bool = True,
    xp=None,
    pytorch: bool = False,
):
    """
    SciPy-free z-scoring for NumPy or Torch.
    """
    xp = _get_xp(x, xp=xp, pytorch=pytorch)

    if getattr(xp, "__name__", "") == "torch":
        mean = xp.mean(x, dim=axis, keepdim=keepdims)
        var = xp.var(x, dim=axis, unbiased=(ddof != 0), keepdim=keepdims)
        return (x - mean) / xp.sqrt(var + eps)

    import numpy as np

    x = np.asarray(x)
    mean = np.mean(x, axis=axis, keepdims=keepdims)
    std = np.std(x, axis=axis, ddof=ddof, keepdims=keepdims)
    return (x - mean) / (std + eps)
