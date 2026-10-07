"""Evaluation metrics used throughout the report.

Forecast accuracy   RMSE, MAE, MARD per horizon; Clarke Error Grid zones (clinical acceptability)
Uncertainty         empirical coverage and mean width of the P10-P90 interval
Event prediction    AUROC, AUPRC, sensitivity / precision at an operating point, false alarms per
                    day, and lead time (how long before an excursion the alarm first fired)
Uncertainty of the metrics themselves: bootstrap over patients (not windows), because windows
from one person are strongly correlated.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.mean(np.abs(y[m] - p[m])))


def mard(y: np.ndarray, p: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(p) & (y > 0)
    return float(np.mean(np.abs(y[m] - p[m]) / y[m]) * 100)


def clarke_zones(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Clarke Error Grid zone (A-E) for each reference/prediction pair (mg/dL)."""
    r, p = np.asarray(ref, float), np.asarray(pred, float)
    z = np.full(r.shape, "B", dtype="<U1")
    e = ((r >= 180) & (p <= 70)) | ((r <= 70) & (p >= 180))
    c = (((r >= 70) & (r <= 290)) & (p >= r + 110)) | (((r >= 130) & (r <= 180)) & (p <= (7 / 5) * r - 182))
    d = (((r >= 240) & (p >= 70) & (p <= 180)) | ((r <= 175 / 3) & (p <= 180) & (p >= 70))
         | (((r >= 175 / 3) & (r <= 70)) & (p >= (6 / 5) * r)))
    a = ((r <= 70) & (p <= 70)) | ((p <= 1.2 * r) & (p >= 0.8 * r))
    z[d] = "D"
    z[c] = "C"
    z[e] = "E"
    z[a] = "A"
    return z


def clarke_summary(ref: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    m = np.isfinite(ref) & np.isfinite(pred)
    z = clarke_zones(ref[m], pred[m])
    out = {k: float(np.mean(z == k) * 100) for k in "ABCDE"}
    out["A+B"] = out["A"] + out["B"]
    return out


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])) * 100), float(np.mean(hi[m] - lo[m]))


def event_scores(label: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    m = np.isfinite(prob)
    y, p = label[m].astype(int), prob[m]
    if y.min() == y.max():
        return {"auroc": float("nan"), "auprc": float("nan"), "prevalence": float(y.mean() * 100)}
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "prevalence": float(y.mean() * 100)}


def operating_point(label: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    """Window-level sensitivity and precision at a probability threshold."""
    alarm = prob >= threshold
    tp = float(np.sum(alarm & label))
    fp = float(np.sum(alarm & ~label))
    return {"threshold": threshold, "sensitivity": 100 * tp / max(float(np.sum(label)), 1),
            "precision": 100 * tp / max(tp + fp, 1)}


def false_alarm_episodes(anchor_bins: np.ndarray, prob: np.ndarray, label: np.ndarray, threshold: float,
                         gap: int = 6) -> int:
    """Alarm episodes (alarming anchors merged when < `gap` bins apart) containing no true positive.

    This is what a clinician experiences as a false alert, rather than every alarming window.
    """
    idx = np.flatnonzero(prob >= threshold)
    if len(idx) == 0:
        return 0
    false_eps, start = 0, 0
    for k in range(1, len(idx) + 1):
        if k == len(idx) or anchor_bins[idx[k]] - anchor_bins[idx[k - 1]] > gap:
            if not label[idx[start:k]].any():
                false_eps += 1
            start = k
    return false_eps


def lead_times(anchor_bins: np.ndarray, prob: np.ndarray, onset_bins: np.ndarray, threshold: float,
               window: int = 24) -> np.ndarray:
    """Minutes between the first alarm in the 2 h before each excursion onset and the onset.

    Returns NaN for excursions with no alarm in the window (missed).
    """
    out = np.full(len(onset_bins), np.nan)
    for i, onset in enumerate(onset_bins):
        sel = (anchor_bins < onset) & (anchor_bins >= onset - window) & (prob >= threshold)
        if sel.any():
            out[i] = (onset - anchor_bins[sel].min()) * 5.0
    return out


def excursion_onsets(cgm_in: np.ndarray, high: bool = True, threshold: float | None = None, min_len: int = 3) -> np.ndarray:
    """Bin indices where a sustained (>= min_len bins) excursion starts."""
    thr = threshold if threshold is not None else (180.0 if high else 70.0)
    x = np.nan_to_num(cgm_in, nan=0.0 if high else 999.0)
    flag = x > thr if high else x < thr
    onsets, i, n = [], 0, len(flag)
    while i < n:
        if flag[i]:
            j = i
            while j < n and flag[j]:
                j += 1
            if j - i >= min_len:
                onsets.append(i)
            i = j
        else:
            i += 1
    return np.array(onsets, dtype=int)


def bootstrap_ci(groups: np.ndarray, fn: Callable[[np.ndarray], float], n_boot: int = 300, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of fn(mask) resampling *patients* (groups) with replacement."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[g] for g in pick])
        v = fn(idx)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))
