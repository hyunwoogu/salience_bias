from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from salience_bias.analyses.evaluate.core.field_metrics import (
    logits_map_from_pred,
    metric_curve_for_trial,
    pearson_cc_maps,
    sim_maps,
    validate_metric,
    zscore_map_spatial,
)
from salience_bias.analyses.evaluate.core.gaze import make_gaze_map

MetricName = Literal["sauc", "uauc", "ll", "cc", "sim", "nss"]


def evaluate_single(
    pred_map: np.ndarray,
    eye_pos: Mapping[str, Any],
    *,
    metric: str = "sauc",
    neg_fixation_maps: Sequence[np.ndarray] | None = None,
    gauss_params: dict[str, Any] | None = None,
) -> float:
    """
    Scalar score for one predictive map and one scanpath.

    `eye_pos`: dict with 'x' and 'y' (see `make_gaze_map`).
    For `metric='sauc'`, pass `neg_fixation_maps`: fixation density maps [H,W] from other images
    (same shape as `pred_map`), used as Judd-style negative samples. If omitted, raises ValueError.

    `uauc`, `ll`, `cc`, `sim`, and `nss` do not require `neg_fixation_maps`.
    """
    m = validate_metric(metric)
    pred_map = np.asarray(pred_map, dtype=np.float64)
    h, w = pred_map.shape
    gaze = make_gaze_map(eye_pos, image_shape=(h, w), gauss_params=gauss_params)
    gaze = np.asarray(gaze, dtype=np.float64)
    if gaze.ndim == 3:
        gaze = gaze.mean(axis=0)
    if gaze.shape != pred_map.shape:
        raise ValueError(f"gaze map shape {gaze.shape} != pred_map shape {pred_map.shape}")

    if m == "sauc":
        if not neg_fixation_maps:
            raise ValueError("evaluate_single(metric='sauc') requires neg_fixation_maps= list of [H,W] arrays.")
        neg_stack = np.stack([np.asarray(a, dtype=np.float64) for a in neg_fixation_maps], axis=0)
        curve = metric_curve_for_trial(
            metric="sauc",
            mixture=pred_map[None, :, :],
            fixation_avg=gaze,
            neg_avgs=neg_stack,
        )
        if curve is None:
            raise ValueError("sAUC could not be computed (empty negatives?).")
        return float(curve[0])

    if m == "uauc":
        curve = metric_curve_for_trial(
            metric="uauc",
            mixture=pred_map[None, :, :],
            fixation_avg=gaze,
            neg_avgs=np.empty((0,)),
        )
        assert curve is not None
        return float(curve[0])

    if m == "ll":
        logits = logits_map_from_pred(pred_map)
        return float(np.mean(logits * gaze))

    if m == "cc":
        return pearson_cc_maps(pred_map, gaze)

    if m == "sim":
        return sim_maps(pred_map, gaze)

    if m == "nss":
        zs = zscore_map_spatial(pred_map)
        return float(np.mean(zs * gaze))

    raise AssertionError("unreachable")


@dataclass
class ConvexMixtureResult:
    lambdas: np.ndarray
    scores: np.ndarray
    best_lambda: float
    best_score: float


def evaluate_convex_mixture(
    pred_map0: np.ndarray,
    pred_map1: np.ndarray,
    eye_pos: Mapping[str, Any],
    *,
    metric: str = "sauc",
    lambdas: np.ndarray | None = None,
    neg_fixation_maps: Sequence[np.ndarray] | None = None,
    gauss_params: dict[str, Any] | None = None,
) -> ConvexMixtureResult:
    """
    Sweep mixture weights between `pred_map0` and `pred_map1`.

    At each λ, the mixed map is ``(1 - λ) * pred_map0 + λ * pred_map1`` (λ=0 is `pred_map0`, λ=1 is `pred_map1`).
    """
    if lambdas is None:
        lambdas = np.linspace(0, 1, num=40)
    lambdas = np.asarray(lambdas, dtype=np.float64)
    pred_map0 = np.asarray(pred_map0, dtype=np.float64)
    pred_map1 = np.asarray(pred_map1, dtype=np.float64)
    if pred_map0.shape != pred_map1.shape:
        raise ValueError("pred_map0 and pred_map1 must have the same shape")

    lam = lambdas[:, None, None]
    mixture = (1.0 - lam) * pred_map0 + lam * pred_map1
    h, w = pred_map0.shape
    gaze = make_gaze_map(eye_pos, image_shape=(h, w), gauss_params=gauss_params)
    gaze = np.asarray(gaze, dtype=np.float64)
    if gaze.ndim == 3:
        gaze = gaze.mean(axis=0)

    m = validate_metric(metric)
    if m == "sauc":
        if not neg_fixation_maps:
            raise ValueError("evaluate_convex_mixture(metric='sauc') requires neg_fixation_maps.")
        neg_stack = np.stack([np.asarray(a, dtype=np.float64) for a in neg_fixation_maps], axis=0)
        neg_avgs = neg_stack
    else:
        neg_avgs = np.empty((0,))

    scores_list = metric_curve_for_trial(metric=m, mixture=mixture, fixation_avg=gaze, neg_avgs=neg_avgs)
    if scores_list is None:
        raise ValueError("Metric returned no scores.")
    scores = np.asarray(scores_list, dtype=np.float64)
    i_best = int(np.nanargmax(scores))
    return ConvexMixtureResult(
        lambdas=lambdas,
        scores=scores,
        best_lambda=float(lambdas[i_best]),
        best_score=float(scores[i_best]),
    )


def evaluate_multi(
    samples: Sequence[Mapping[str, Any]],
    *,
    metric: str = "ll",
    aggregate: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], float]:
    """
    Per-sample scores. Each record may contain:
      - pred_map [H,W]
      - eye_pos dict
      - optional neg_fixation_maps for sauc
      - optional id
    """
    rows: list[dict[str, Any]] = []
    for rec in samples:
        pid = rec.get("id")
        score = evaluate_single(
            rec["pred_map"],
            rec["eye_pos"],
            metric=metric,
            neg_fixation_maps=rec.get("neg_fixation_maps"),
            gauss_params=rec.get("gauss_params"),
        )
        rows.append({"id": pid, "metric": metric, "score": score})
    if aggregate:
        mean_score = float(np.mean([r["score"] for r in rows]))
        return rows, mean_score
    return rows
