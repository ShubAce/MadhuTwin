"""Shared experiment helpers: splits, baselines, and metric tables on a common evaluation set."""

from __future__ import annotations

import numpy as np

from twin.data.windows import G_SCALE, HORIZONS, H, PatientArrays

from .metrics import (
    bootstrap_ci,
    clarke_summary,
    coverage,
    event_scores,
    excursion_onsets,
    false_alarm_episodes,
    lead_times,
    mae,
    mard,
    operating_point,
    rmse,
)


def split_patients(arrays: list[PatientArrays], fractions=(0.7, 0.1, 0.2), seed: int = 0) -> tuple[list, list, list]:
    """Stratified (by glycaemic status) patient-level split."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for status in sorted({a.status for a in arrays}):
        group = [a for a in arrays if a.status == status]
        idx = rng.permutation(len(group))
        n_tr, n_va = int(round(fractions[0] * len(group))), int(round(fractions[1] * len(group)))
        tr += [group[i] for i in idx[:n_tr]]
        va += [group[i] for i in idx[n_tr : n_tr + n_va]]
        te += [group[i] for i in idx[n_tr + n_va :]]
    return tr, va, te


def group_folds(arrays: list[PatientArrays], k: int = 5, seed: int = 0) -> list[tuple[list, list]]:
    """k folds over patient *groups* (several recordings of one person stay together)."""
    groups = sorted({a.group for a in arrays})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    fold_of = {groups[g]: i % k for i, g in enumerate(order)}
    return [([a for a in arrays if fold_of[a.group] != f], [a for a in arrays if fold_of[a.group] == f]) for f in range(k)]


def baselines(now: np.ndarray, cgm_slope_mg_per_min: np.ndarray, physics: np.ndarray, physics_prior: np.ndarray) -> dict:
    """Absolute 5..120-min forecasts (n, H) for the reference methods."""
    t = np.arange(1, H + 1)[None, :] * 5.0
    slope = np.nan_to_num(cgm_slope_mg_per_min)[:, None]
    lin = np.clip(now[:, None] + slope * np.minimum(t, 45.0), 40, 400)
    return {
        "Persistence": np.repeat(now[:, None], H, axis=1),
        "Linear trend": lin,
        "Twin (EHR prior)": now[:, None] + physics_prior * G_SCALE,
        "Twin (personalised)": now[:, None] + physics * G_SCALE,
    }


def slope_now(arrays: list[PatientArrays], after_calibration: bool) -> np.ndarray:
    out = []
    for a in arrays:
        sel = a.anchors >= a.calib_bins if after_calibration else np.ones(len(a.anchors), bool)
        k = a.anchors[sel]
        prev = a.cgm_in[np.clip(k - 6, 0, None)]
        out.append((a.cgm[k] - prev) / 30.0)
    return np.concatenate(out) if out else np.array([])


def forecast_rows(y: np.ndarray, preds: dict[str, np.ndarray], groups: np.ndarray, ci: bool = True,
                  quantiles: dict[str, tuple[np.ndarray, np.ndarray]] | None = None) -> list[dict]:
    """y and preds are absolute glucose (n, H). One row per method."""
    rows = []
    for name, p in preds.items():
        r = {"method": name}
        for h in HORIZONS:
            yy, pp = y[:, h - 1], p[:, h - 1]
            r[f"rmse_{h * 5}"] = rmse(yy, pp)
            r[f"mae_{h * 5}"] = mae(yy, pp)
            r[f"mard_{h * 5}"] = mard(yy, pp)
        for h in (6, 12, 24):
            cz = clarke_summary(y[:, h - 1], p[:, h - 1])
            r[f"clarkeA_{h * 5}"] = cz["A"]
            r[f"clarkeAB_{h * 5}"] = cz["A+B"]
        if ci:
            for h in (12, 24):
                yy, pp = y[:, h - 1], p[:, h - 1]
                lo, hi = bootstrap_ci(groups, lambda idx, yy=yy, pp=pp: rmse(yy[idx], pp[idx]), n_boot=200)
                r[f"rmse_{h * 5}_ci"] = [lo, hi]
        if quantiles and name in quantiles:
            lo_q, hi_q = quantiles[name]
            for h in (6, 12, 24):
                cov, width = coverage(y[:, h - 1], lo_q[:, h - 1], hi_q[:, h - 1])
                r[f"coverage80_{h * 5}"] = cov
                r[f"width80_{h * 5}"] = width
        rows.append(r)
    return rows


def best_threshold(label: np.ndarray, prob: np.ndarray) -> float:
    """Threshold maximising F1 on a validation set."""
    grid = np.unique(np.quantile(prob, np.linspace(0.5, 0.995, 120)))
    best, thr = -1.0, 0.5
    for t in grid:
        a = prob >= t
        tp = np.sum(a & label)
        f1 = 2 * tp / max(np.sum(a) + np.sum(label), 1)
        if f1 > best:
            best, thr = f1, float(t)
    return thr


def event_rows(arrays: list[PatientArrays], data_index: dict, scores: dict[str, np.ndarray], event: str,
               thresholds: dict[str, float] | None = None, after_calibration: bool = True,
               alert_scores: dict[str, np.ndarray] | None = None) -> list[dict]:
    """AUROC/AUPRC on eligible anchors + alert behaviour (lead time, false alarms/day) per method.

    `data_index` holds aligned arrays for the evaluation anchors: patient, anchor, label, eligible.
    `alert_scores` optionally replaces the scores used for the operating point (e.g. probabilities
    rescaled per cross-validation fold so that 0.5 is each fold's own training-chosen threshold).
    """
    pat, anc, lab, ok = data_index["patient"], data_index["anchor"], data_index[event], data_index[f"{event}_ok"] > 0
    rows = []
    for name, sc in scores.items():
        r = {"method": name, "event": event, **event_scores(lab[ok].astype(bool), sc[ok])}
        grp = data_index["group"][ok]
        lab_ok, sc_ok = lab[ok].astype(bool), sc[ok]
        lo, hi = bootstrap_ci(grp, lambda idx, y=lab_ok, s=sc_ok: event_scores(y[idx], s[idx])["auroc"], n_boot=200)
        r["auroc_ci"] = [lo, hi]
        if thresholds and name in thresholds:
            thr = thresholds[name]
            if alert_scores and name in alert_scores:
                sc = alert_scores[name]
            r.update(operating_point(lab[ok].astype(bool), sc[ok], thr))
            leads, fa, days = [], 0, 0.0
            for i in np.unique(pat):
                a = arrays[i]
                m = (pat == i) & ok
                onsets = excursion_onsets(a.cgm_in, high=(event == "spike"))
                start = a.calib_bins if after_calibration else 0
                onsets = onsets[onsets >= start + 6]
                leads.append(lead_times(anc[m], sc[m], onsets, thr))
                fa += false_alarm_episodes(anc[m], sc[m], lab[m].astype(bool), thr)
                days += (len(a.cgm) - start) * 5 / 1440
            leads = np.concatenate(leads) if leads else np.array([])
            r["excursions"] = int(len(leads))
            r["detected_pct"] = float(np.mean(np.isfinite(leads)) * 100) if len(leads) else float("nan")
            r["detected_30min_ahead_pct"] = float(np.mean(leads >= 30) * 100) if len(leads) else float("nan")
            r["median_lead_min"] = float(np.nanmedian(leads)) if np.isfinite(leads).any() else float("nan")
            r["false_alerts_per_day"] = fa / max(days, 1e-9)
        rows.append(r)
    return rows


def physics_event_score(now: np.ndarray, physics: np.ndarray, event: str) -> np.ndarray:
    traj = now[:, None] + physics * G_SCALE
    return traj.max(axis=1) - 180.0 if event == "spike" else 70.0 - traj.min(axis=1)


def linear_event_score(now: np.ndarray, slope: np.ndarray, event: str) -> np.ndarray:
    proj = now + np.nan_to_num(slope) * 45.0
    return np.maximum(now, proj) - 180.0 if event == "spike" else 70.0 - np.minimum(now, proj)
