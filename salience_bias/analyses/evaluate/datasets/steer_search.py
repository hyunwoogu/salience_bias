from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.ndimage import zoom


def zoom_to_hw(map_2d: np.ndarray, *, target_hw: Sequence[int] = (224, 224)) -> np.ndarray:
    """
    Zoom map to a target size using the same approach as existing workflows:
    scale with a 1-pixel border and then crop.
    """
    current_h, current_w = map_2d.shape
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    scale_h = (target_h + 2) / current_h
    scale_w = (target_w + 2) / current_w
    return zoom(map_2d, (scale_h, scale_w), order=1)[1:-1, 1:-1]
