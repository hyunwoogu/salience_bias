from __future__ import annotations

from typing import Callable, Mapping, TypeVar

import numpy as np

from salience_bias.utils.numeric import zscore

K = TypeVar("K")

METRICS = frozenset({"sauc", "uauc", "ll", "cc", "sim", "nss"})


def _auc_from_pos_neg(pos: np.ndarray, neg: np.ndarray) -> float:
    """
    AUC for a single positive score vs many negative scores.

    This matches the common "shuffled AUC" setting where there is exactly one positive
    sample (the fixation readout) and many negatives (reads at fixations from other images).

    Tie handling: counts ties as 0.5.
    """
    pos_v = float(np.asarray(pos).reshape(-1)[0])
    neg_v = np.asarray(neg, dtype=np.float64).reshape(-1)
    if neg_v.size == 0:
        return float("nan")
    return float((neg_v < pos_v).mean() + 0.5 * (neg_v == pos_v).mean())


def mixture_stack(*, clip_map: np.ndarray, deepgaze_map: np.ndarray, mix_weights: np.ndarray) -> np.ndarray:
    """Returns mixture stack with shape [n_weights, H, W]."""
    return mix_weights[:, None, None] * clip_map + (1 - mix_weights)[:, None, None] * deepgaze_map


def fixation_average(fixation_stack: np.ndarray, *, start_order: int = 1) -> np.ndarray | None:
    """Mean fixation map over orders >= start_order."""
    if fixation_stack.shape[0] <= start_order:
        return None
    return np.mean(fixation_stack[start_order:], axis=0)


def negative_fixation_avgs(
    fixations_by_trial: Mapping[K, np.ndarray],
    *,
    exclude: Callable[[K], bool],
    start_order: int = 1,
) -> np.ndarray:
    """
    Collect negative maps for trials with exclude(key) True and len(fix) > start_order.
    Returns [Nneg, H, W], or shape (0,) if none (callers handle empty).
    """
    neg_list = [
        np.mean(fix[start_order:], axis=0)
        for key, fix in fixations_by_trial.items()
        if exclude(key) and (len(fix) > start_order)
    ]
    if len(neg_list) == 0:
        return np.empty((0,))
    return np.asarray(neg_list)


def sauc_curve_from_reads(*, reads_pos: np.ndarray, reads_neg: np.ndarray) -> list[float]:
    """
    sAUC per mixture weight from scalar reads.

    reads_pos: shape [n_weights]
    reads_neg: shape [n_neg, n_weights]
    """
    reads_pos = np.asarray(reads_pos, dtype=np.float64).reshape(-1)
    reads_neg = np.asarray(reads_neg, dtype=np.float64)
    if reads_neg.ndim != 2:
        raise ValueError("reads_neg must be [n_neg, n_weights]")
    if reads_neg.shape[1] != reads_pos.shape[0]:
        raise ValueError("reads_neg second dimension must match reads_pos length")
    return [_auc_from_pos_neg(reads_pos[i], reads_neg[:, i]) for i in range(reads_pos.shape[0])]


def sauc_from_scalar_reads(*, reads_pos_scalar: float, reads_neg_scalars: np.ndarray) -> float | None:
    reads_neg_scalars = np.asarray(reads_neg_scalars, dtype=np.float64).reshape(-1)
    if reads_neg_scalars.size == 0:
        return None
    return float(_auc_from_pos_neg(np.asarray([reads_pos_scalar], dtype=np.float64), reads_neg_scalars))


def sauc_curve_for_trial(
    *,
    mixture: np.ndarray,
    fixation_avg: np.ndarray,
    neg_avgs: np.ndarray,
) -> list[float] | None:
    if np.asarray(neg_avgs).size == 0:
        return None
    neg_avgs = np.asarray(neg_avgs, dtype=np.float64)
    if neg_avgs.ndim == 4 and neg_avgs.shape[1] == 1:
        neg_avgs = neg_avgs[:, 0, :, :]
    reads_pos = np.mean(mixture * fixation_avg, axis=(-2, -1))
    reads_neg = np.mean(mixture * neg_avgs[:, None, :, :], axis=(-2, -1))
    return sauc_curve_from_reads(reads_pos=reads_pos, reads_neg=reads_neg)


def validate_metric(metric: str) -> str:
    m = metric.lower()
    if m not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)}, got {metric!r}")
    return m


def zscore_map_spatial(z: np.ndarray) -> np.ndarray:
    """Z-score the predictive map treating all pixels as one sample (same first step as `ll`)."""
    flat = np.asarray(z, dtype=np.float64).ravel()
    z0 = zscore(flat, axis=-1).reshape(np.asarray(z).shape)
    return np.asarray(z0, dtype=np.float64)


def _logsumexp_flat(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return float("-inf")
    m = float(np.max(x))
    if not np.isfinite(m):
        m = 0.0
    return float(m + np.log(np.sum(np.exp(x - m))))


def logits_map_from_pred(z: np.ndarray) -> np.ndarray:
    """Per-map spatial z-score then subtract log-sum-exp over pixels (log-softmax partition)."""
    z0 = zscore_map_spatial(z)
    lse = _logsumexp_flat(z0)
    return z0 - lse


def pearson_cc_maps(p: np.ndarray, q: np.ndarray) -> float:
    """Pearson r between two same-shaped maps (flattened). Returns NaN if undefined."""
    a = np.asarray(p, dtype=np.float64).ravel()
    b = np.asarray(q, dtype=np.float64).ravel()
    if a.size == 0 or b.size != a.size:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def sim_maps(p: np.ndarray, q: np.ndarray) -> float:
    """SIM: normalize p and q to distributions over pixels, then sum_i min(p_i, q_i)."""
    a = np.asarray(p, dtype=np.float64).ravel()
    b = np.asarray(q, dtype=np.float64).ravel()
    sa = float(np.sum(a))
    sb = float(np.sum(b))
    if sa <= 0.0 or sb <= 0.0:
        return float("nan")
    pa = a / sa
    qb = b / sb
    return float(np.minimum(pa, qb).sum())


def metric_curve_for_trial(
    *,
    metric: str,
    mixture: np.ndarray,
    fixation_avg: np.ndarray,
    neg_avgs: np.ndarray,
) -> list[float] | None:
    m = validate_metric(metric)
    if m == "sauc":
        return sauc_curve_for_trial(mixture=mixture, fixation_avg=fixation_avg, neg_avgs=neg_avgs)
    if m == "uauc":
        return uauc_curve_for_trial(mixture=mixture, fixation_avg=fixation_avg)
    if m == "ll":
        return ll_curve_for_trial(mixture=mixture, fixation_avg=fixation_avg)
    if m == "cc":
        return cc_curve_for_trial(mixture=mixture, fixation_avg=fixation_avg)
    if m == "sim":
        return sim_curve_for_trial(mixture=mixture, fixation_avg=fixation_avg)
    if m == "nss":
        return nss_curve_for_trial(mixture=mixture, fixation_avg=fixation_avg)
    raise AssertionError("unreachable")


def uauc_curve_for_trial(*, mixture: np.ndarray, fixation_avg: np.ndarray) -> list[float]:
    reads_pos = np.mean(mixture * fixation_avg, axis=(-2, -1))
    out: list[float] = []
    for i_m in range(reads_pos.shape[0]):
        neg = np.asarray(mixture[i_m], dtype=np.float64).ravel()
        out.append(_auc_from_pos_neg(reads_pos[i_m], neg))
    return out


def ll_curve_for_trial(*, mixture: np.ndarray, fixation_avg: np.ndarray) -> list[float]:
    n_w = mixture.shape[0]
    scores: list[float] = []
    for i_m in range(n_w):
        logits = logits_map_from_pred(mixture[i_m])
        scores.append(float(np.mean(logits * fixation_avg)))
    return scores


def cc_curve_for_trial(*, mixture: np.ndarray, fixation_avg: np.ndarray) -> list[float]:
    n_w = mixture.shape[0]
    return [pearson_cc_maps(mixture[i_m], fixation_avg) for i_m in range(n_w)]


def sim_curve_for_trial(*, mixture: np.ndarray, fixation_avg: np.ndarray) -> list[float]:
    n_w = mixture.shape[0]
    return [sim_maps(mixture[i_m], fixation_avg) for i_m in range(n_w)]


def nss_curve_for_trial(*, mixture: np.ndarray, fixation_avg: np.ndarray) -> list[float]:
    n_w = mixture.shape[0]
    scores: list[float] = []
    for i_m in range(n_w):
        zs = zscore_map_spatial(mixture[i_m])
        scores.append(float(np.mean(zs * fixation_avg)))
    return scores
