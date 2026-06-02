"""Minimal dataset helpers for preprocess workflows."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent


def load_array(lst):
    arr = np.array(lst, dtype=object)
    return np.where(arr == None, np.nan, arr).astype(float)


def extract_info(text, pattern, transform=str):
    match = re.search(pattern, text)
    return transform(match.group(1)) if match else None
