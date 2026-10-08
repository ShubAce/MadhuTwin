"""Clinical-utility analyses: what a diabetologist on the jury will ask beyond AUROC.

sensitivity_at_false_alerts   best excursion detection while keeping false alert episodes
                              per patient-day at or below a budget (alert fatigue is the
                              main reason CGM alarms get switched off)
calibration_curve             do predicted probabilities match observed frequencies?
decision_curve                net benefit across risk thresholds vs alerting everyone/no one
platt_fit / platt_apply       site-level recalibration of event probabilities
crossfit_recalibrate          the same, fitted on other folds' patients only (honest evaluation)
subgroup_table                error by sex, age, BMI, glycaemic status and therapy
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from twin.data.windows import PatientArrays

from .metrics import excursion_onsets, false_alarm_episodes, lead_times


def sensitivity_at_false_alerts(arrays: list[PatientArrays], idx: dict, score: np.ndarray, event: str,
                                max_fa_per_day: float = 1.0, after_calibration: bool = True) -> dict:
    """Sweep alert thresholds; return the operating point with the highest share of excursions
    caught in the 2 h before onset while false-alert episodes stay within the daily budget."""
    ok = idx[f"{event}_ok"] > 0
    pats = np.unique(idx["patient"])
    per = []
    days = 0.0
    for i in pats:
        a = arrays[i]
        m = (idx["patient"] == i) & ok
        start = a.calib_bins if after_calibration else 0
        onsets = excursion_onsets(a.cgm_in, high=(event == "spike"))
        onsets = onsets[onsets >= start + 6]
        per.append((idx["anchor"][m], score[m], idx[event][m] > 0, onsets))
        days += (len(a.cgm) - start) * 5 / 1440
    finite = score[ok][np.isfinite(score[ok])]
    if len(finite) == 0:
        return {}
    best = {"max_false_alerts_per_day": max_fa_per_day, "detected_pct": 0.0}
    for thr in np.unique(np.quantile(finite, np.linspace(0.5, 0.999, 80))):
        fa = sum(false_alarm_episodes(an, sc, lab, thr) for an, sc, lab, _ in per) / max(days, 1e-9)
        if fa > max_fa_per_day:
            continue
        leads = np.concatenate([lead_times(an, sc, on, thr) for an, sc, _, on in per]) if per else np.array([])
        if not len(leads):
            continue
        det = float(np.mean(np.isfinite(leads)) * 100)
        if det > best["detected_pct"]:
            best = {"max_false_alerts_per_day": max_fa_per_day, "threshold": float(thr), "detected_pct": det,
                    "false_alerts_per_day": float(fa), "median_lead_min": float(np.nanmedian(leads)) if np.isfinite(leads).any() else None,
                    "excursions": int(len(leads))}
    return best


def calibration_curve(label: np.ndarray, prob: np.ndarray, bins: int = 10) -> list[dict]:
    m = np.isfinite(prob)
    y, p = label[m].astype(float), prob[m]
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        s = (p >= lo) & (p < hi)
        if s.sum() >= 20:
            out.append({"predicted": float(p[s].mean()), "observed": float(y[s].mean()), "n": int(s.sum())})
    return out


def decision_curve(label: np.ndarray, prob: np.ndarray, thresholds: np.ndarray | None = None) -> list[dict]:
    """Net benefit = TP/N - FP/N * pt/(1-pt) (Vickers & Elkin 2006)."""
    thresholds = np.arange(0.05, 0.81, 0.05) if thresholds is None else thresholds
    m = np.isfinite(prob)
    y, p = label[m].astype(bool), prob[m]
    n, prev = len(y), y.mean()
    out = []
    for pt in thresholds:
        a = p >= pt
        w = pt / (1 - pt)
        out.append({"threshold": float(pt), "model": float((a & y).sum() / n - (a & ~y).sum() / n * w),
                    "alert_all": float(prev - (1 - prev) * w), "alert_none": 0.0})
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def platt_fit(label: np.ndarray, prob: np.ndarray, l2: float = 1e-3, iters: int = 50) -> list[float]:
    """Logistic recalibration p' = sigmoid(a * logit(p) + b), shrunk towards the identity map.
    Two parameters, so it is stable even for a small site with few events."""
    m = np.isfinite(prob)
    y, z = label[m].astype(float), _logit(prob[m])
    if len(y) < 50 or y.sum() < 5 or y.sum() > len(y) - 5:
        return [1.0, 0.0]
    X = np.column_stack([z, np.ones_like(z)])
    w0 = np.array([1.0, 0.0])
    w = w0.copy()
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (p - y) / len(y) + l2 * (w - w0)
        hess = (X * (p * (1 - p))[:, None]).T @ X / len(y) + l2 * np.eye(2)
        step = np.linalg.solve(hess, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return [float(w[0]), float(w[1])]


def platt_apply(prob: np.ndarray, ab: list[float]) -> np.ndarray:
    out = 1 / (1 + np.exp(-(ab[0] * _logit(prob) + ab[1])))
    return np.where(np.isfinite(prob), out, np.nan)


def crossfit_recalibrate(prob: np.ndarray, label: np.ndarray, ok: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Recalibrate out-of-fold scores: each fold's map is fitted on the *other* folds' patients only,
    as a site would recalibrate on its own past patients before going live."""
    out = np.full(len(prob), np.nan)
    for f in np.unique(fold):
        tr = (fold != f) & ok & np.isfinite(prob)
        out[fold == f] = platt_apply(prob[fold == f], platt_fit(label[tr], prob[tr]))
    return out


def subgroup_table(per_patient: pd.DataFrame, by: list[str]) -> list[dict]:
    """per_patient has one row per patient with squared-error sums/counts and attributes."""
    rows = []
    for col in by:
        for val, g in per_patient.groupby(col, observed=True):
            rows.append({"attribute": col, "group": str(val), "patients": int(len(g)),
                         "rmse_60": float(np.sqrt(g.se60.sum() / max(g.n60.sum(), 1))),
                         "rmse_120": float(np.sqrt(g.se120.sum() / max(g.n120.sum(), 1)))})
    return rows
