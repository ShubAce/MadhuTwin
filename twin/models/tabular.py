"""Tabular features per anchor, grouped by data stream (for gradient boosting and ablations).

Groups
    cgm       recent glucose level, trend, variability
    wearable  heart rate (vs personal 24 h baseline), activity, HRV, sleep
    events    logged meals (carbs/fat/protein on board), insulin on board, oral drugs
    ehr       the static EHR vector (demographics, labs, therapy, comorbidities, genetics)
    physics   the personalised mechanistic twin forecast + synchronised hidden state
Time of day is always included.
"""

from __future__ import annotations

import numpy as np

from twin.data.physics import UKF_FEATURES
from twin.data.windows import CH, G_SCALE, HORIZONS, STATIC_NAMES, PatientArrays

GROUPS = ("cgm", "wearable", "events", "ehr", "physics")


def _win(x: np.ndarray, anchors: np.ndarray, L: int) -> np.ndarray:
    idx = anchors[:, None] + np.arange(-L + 1, 1)[None, :]
    return x[np.clip(idx, 0, len(x) - 1)]


def _nanstat(f, a: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return f(a, axis=1)


def _masked(arr: PatientArrays, ch: str, avail: str) -> np.ndarray:
    v = arr.dyn[:, CH[ch]].astype(float)
    return np.where(arr.dyn[:, CH[avail]] > 0, v, np.nan)


def features(arr: PatientArrays, groups: tuple[str, ...] = GROUPS, physics: str = "physics") -> tuple[np.ndarray, list[str]]:
    a = arr.anchors
    cols: dict[str, np.ndarray] = {}
    g = arr.cgm_in
    now = arr.cgm[a]
    cols["cgm_now"] = now
    if "cgm" in groups:
        for lag in (1, 2, 3, 6, 12, 24):
            cols[f"cgm_d{lag * 5}"] = now - g[np.clip(a - lag, 0, None)]
        for L in (12, 24, 72):
            w = _win(g, a, L)
            cols[f"cgm_mean{L * 5}"] = _nanstat(np.nanmean, w)
            cols[f"cgm_std{L * 5}"] = _nanstat(np.nanstd, w)
        w = _win(g, a, 24)
        cols["cgm_min120"] = _nanstat(np.nanmin, w)
        cols["cgm_max120"] = _nanstat(np.nanmax, w)
        w6 = _win(g, a, 6)
        t = np.arange(6) - 2.5
        cols["cgm_slope30"] = _nanstat(np.nanmean, (w6 - _nanstat(np.nanmean, w6)[:, None]) * t) / np.mean(t**2) / 5.0
        cols["cgm_accel"] = cols["cgm_d15"] - (cols["cgm_d30"] - cols["cgm_d15"])
    if "wearable" in groups:
        hr = _masked(arr, "hr", "hr_avail") * 15 + 75
        cols["hr_now"] = _nanstat(np.nanmean, _win(hr, a, 3))
        cols["hr_dev"] = _nanstat(np.nanmean, _win(hr, a, 12)) - _nanstat(np.nanmean, _win(hr, a, 288))
        mets = _masked(arr, "mets", "act_avail") * 1.5 + 1.3
        cols["mets_15"] = _nanstat(np.nanmean, _win(mets, a, 3))
        cols["mets_60"] = _nanstat(np.nanmean, _win(mets, a, 12))
        steps = _masked(arr, "steps", "act_avail") * 100
        cols["steps_60"] = _nanstat(np.nansum, _win(steps, a, 12))
        cols["steps_6h"] = _nanstat(np.nansum, _win(steps, a, 72))
        hrv = np.exp(_masked(arr, "hrv", "hrv_avail") * 0.5 + np.log(30))
        w = _win(hrv, a, 72)
        cols["hrv_6h"] = _nanstat(np.nanmean, w)
        last = np.where(np.isfinite(w), np.arange(72)[None, :], -1).max(axis=1)
        cols["hrv_last"] = np.where(last >= 0, w[np.arange(len(a)), np.clip(last, 0, None)], np.nan)
        sl = np.where(arr.dyn[:, CH["sleep_avail"]] > 0, arr.dyn[:, CH["asleep"]], np.nan)
        cols["asleep_now"] = sl[a]
        cols["sleep_frac_8h"] = _nanstat(np.nanmean, _win(sl, a, 96))
        deep = np.where(arr.dyn[:, CH["sleep_avail"]] > 0, arr.dyn[:, CH["deep_sleep"]], np.nan)
        cols["deep_frac_8h"] = _nanstat(np.nanmean, _win(deep, a, 96))
    if "events" in groups:
        d = arr.dyn
        cols["cob"] = d[a, CH["cob"]] * 50
        cols["fpob"] = d[a, CH["fpob"]] * 50
        cols["iob"] = d[a, CH["iob"]] * 10
        carbs = d[:, CH["carbs"]] * 50
        for L in (6, 12, 24):
            cols[f"carbs_{L * 5}"] = _win(carbs, a, L).sum(axis=1)
        w = _win(carbs, a, 72) > 0
        last = np.where(w, np.arange(72)[None, :], -1).max(axis=1)
        cols["min_since_meal"] = np.where(last >= 0, (71 - last) * 5.0, 400.0)
        cols["insulin_2h"] = _win(d[:, CH["insulin"]] * 10, a, 24).sum(axis=1)
        cols["oad_6h"] = _win(d[:, CH["oad"]], a, 72).sum(axis=1)
    cols["tod_sin"] = arr.dyn[a, CH["tod_sin"]]
    cols["tod_cos"] = arr.dyn[a, CH["tod_cos"]]
    if "ehr" in groups:
        for i, name in enumerate(STATIC_NAMES):
            cols[f"ehr_{name}"] = np.full(len(a), arr.static[i])
    if "physics" in groups:
        ph = getattr(arr, physics) * G_SCALE
        for h in HORIZONS:
            cols[f"twin_d{h * 5}"] = ph[:, h - 1]
        cols["twin_max"] = ph.max(axis=1)
        cols["twin_min"] = ph.min(axis=1)
        for i, name in enumerate(UKF_FEATURES):
            cols[f"sync_{name}"] = arr.ukf[:, i]
    names = list(cols)
    return np.column_stack([cols[k] for k in names]).astype(np.float32), names


def stack(arrays: list[PatientArrays], groups: tuple[str, ...] = GROUPS, after_calibration: bool = False,
          physics: str = "physics") -> dict:
    """Concatenate features/targets/labels over patients. `after_calibration` keeps only anchors
    after each patient's personalisation period (honest evaluation of the personalised twin)."""
    X, Y, P, sp, sp_ok, hy, hy_ok, grp, pid_idx, anchor = [], [], [], [], [], [], [], [], [], []
    names = None
    for i, arr in enumerate(arrays):
        keep = arr.anchors >= arr.calib_bins if after_calibration else np.ones(len(arr.anchors), bool)
        if not keep.any():
            continue
        x, names = features(arr, groups, physics)
        X.append(x[keep])
        Y.append(arr.y[keep] * G_SCALE)
        P.append(arr.cgm[arr.anchors][keep])
        sp.append(arr.spike[keep]); sp_ok.append(arr.spike_ok[keep])  # noqa: E702
        hy.append(arr.hypo[keep]); hy_ok.append(arr.hypo_ok[keep])  # noqa: E702
        grp.append(np.full(keep.sum(), arr.group, dtype=object))
        pid_idx.append(np.full(keep.sum(), i))
        anchor.append(arr.anchors[keep])
    return {"X": np.concatenate(X), "names": names, "Y": np.concatenate(Y), "now": np.concatenate(P),
            "spike": np.concatenate(sp), "spike_ok": np.concatenate(sp_ok), "hypo": np.concatenate(hy),
            "hypo_ok": np.concatenate(hy_ok), "group": np.concatenate(grp), "patient": np.concatenate(pid_idx),
            "anchor": np.concatenate(anchor)}
