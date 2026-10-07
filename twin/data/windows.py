"""Turn common-schema tables into model-ready arrays (the fusion of both streams).

For each patient:
    dyn      (n_bins, C)   normalised wearable/event channels on the 5-min grid (NaN -> 0 + masks)
    static   (S,)          normalised EHR vector with missingness indicators
    anchors  (n_anchor,)   bin indices from which a forecast is issued
    physics  (n_anchor, H) mechanistic-twin forecast deltas (personalised and EHR-prior variants)
    ukf      (n_anchor, 5) synchronised hidden-state features
    y        (n_anchor, H) future CGM minus current CGM, every 5 min up to 2 h (NaN when missing)
    spike / hypo           event labels (>=15 min above 180 / below 70 within 2 h) + eligibility

Everything a model sees at an anchor is information available at that moment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

GRID = 5
HISTORY = 72          # bins (6 h) of history fed to sequence models
H = 24                # forecast bins (2 h)
HORIZONS = (6, 12, 18, 24)  # 30, 60, 90, 120 min
G_SCALE, G_CENTER = 50.0, 140.0

DYN_CHANNELS = (
    "cgm", "cgm_avail", "cgm_slope", "hr", "hr_avail", "mets", "act_avail", "steps", "hrv", "hrv_avail",
    "asleep", "deep_sleep", "sleep_avail", "carbs", "cob", "fpob", "insulin", "iob", "oad", "tod_sin", "tod_cos",
)
CH = {c: i for i, c in enumerate(DYN_CHANNELS)}
MODALITIES = {
    "cgm": ("cgm", "cgm_avail", "cgm_slope"),
    "wearable": ("hr", "hr_avail", "mets", "act_avail", "steps", "hrv", "hrv_avail", "asleep", "deep_sleep", "sleep_avail"),
    "events": ("carbs", "cob", "fpob", "insulin", "iob", "oad"),
}

STATIC_NUMERIC = {  # feature: (center, scale)
    "age": (50, 15), "bmi": (26, 5), "hba1c_pct": (7, 1.5), "fpg_mgdl": (130, 40), "fasting_insulin_uU": (12, 8),
    "homa_ir": (3.5, 3), "tg_mgdl": (150, 70), "hdl_mgdl": (45, 12), "ldl_mgdl": (105, 35), "egfr": (90, 25),
    "diabetes_years": (5, 6), "tcf7l2_risk_alleles": (0.7, 0.7), "prs_t2d": (0, 1),
}
STATIC_BINARY = ("on_metformin", "on_sulfonylurea", "on_dpp4", "on_sglt2", "on_insulin", "hypertension", "dyslipidemia")
STATIC_NAMES = (
    [f"{k}" for k in STATIC_NUMERIC] + ["sex_m"] + list(STATIC_BINARY)
    + ["status_normal", "status_prediabetes", "status_t2d"] + [f"{k}_missing" for k in STATIC_NUMERIC]
)


def static_vector(row: dict) -> np.ndarray:
    vals, miss = [], []
    for k, (c, s) in STATIC_NUMERIC.items():
        v = row.get(k)
        ok = v is not None and pd.notna(v)
        vals.append(np.clip((float(v) - c) / s, -4, 4) if ok else 0.0)
        miss.append(0.0 if ok else 1.0)
    vals.append(1.0 if row.get("sex") == "M" else 0.0)
    vals += [float(row.get(k) or 0.0) if pd.notna(row.get(k)) else 0.0 for k in STATIC_BINARY]
    st = row.get("status")
    vals += [float(st == "normal"), float(st == "prediabetes"), float(st == "t2d")]
    return np.array(vals + miss, dtype=np.float32)


def _decay_sum(x: np.ndarray, tau_bins: float) -> np.ndarray:
    a = np.exp(-1.0 / tau_bins)
    out = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc = acc * a + v
        out[i] = acc
    return out


def _runs(mask: np.ndarray, min_len: int = 3) -> np.ndarray:
    """Boolean array marking bins that belong to a run of >= min_len consecutive True values."""
    out = np.zeros_like(mask)
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


@dataclass
class PatientArrays:
    pid: str
    source: str
    group: str
    status: str
    start: pd.Timestamp
    dyn: np.ndarray
    cgm: np.ndarray            # raw CGM (targets), NaN when missing
    cgm_in: np.ndarray         # gap-filled CGM used as model input (<= 20 min gaps)
    static: np.ndarray
    anchors: np.ndarray
    y: np.ndarray
    spike: np.ndarray
    spike_ok: np.ndarray
    hypo: np.ndarray
    hypo_ok: np.ndarray
    physics: np.ndarray | None = None        # personalised twin forecast deltas (normalised)
    physics_prior: np.ndarray | None = None  # EHR-prior twin forecast deltas (normalised)
    ukf: np.ndarray | None = None
    calib_bins: int = 0
    extra: dict = field(default_factory=dict)


def build_patient(static_row: dict, series: pd.DataFrame, stride: int = 3, min_history: int = 36) -> PatientArrays:
    s = series.sort_values("ts").drop_duplicates("ts").set_index("ts")
    idx = pd.date_range(s.index.min(), s.index.max(), freq=f"{GRID}min")
    s = s.reindex(idx)
    n = len(s)
    cgm = s["cgm"].to_numpy(dtype=float)
    cgm_in = s["cgm"].interpolate(limit=4, limit_area="inside").to_numpy(dtype=float)

    dyn = np.zeros((n, len(DYN_CHANNELS)), dtype=np.float32)
    ok = np.isfinite(cgm_in)
    dyn[:, CH["cgm"]] = np.where(ok, (cgm_in - G_CENTER) / G_SCALE, 0)
    dyn[:, CH["cgm_avail"]] = ok
    slope = np.full(n, np.nan)
    slope[3:] = (cgm_in[3:] - cgm_in[:-3]) / 15.0
    dyn[:, CH["cgm_slope"]] = np.nan_to_num(slope / 2.0)

    hr = s["hr"].to_numpy(dtype=float)
    dyn[:, CH["hr"]] = np.nan_to_num((hr - 75) / 15)
    dyn[:, CH["hr_avail"]] = np.isfinite(hr)
    mets = s["mets"].to_numpy(dtype=float)
    steps = s["steps"].to_numpy(dtype=float)
    dyn[:, CH["mets"]] = np.nan_to_num((mets - 1.3) / 1.5)
    dyn[:, CH["act_avail"]] = np.isfinite(mets) | np.isfinite(steps)
    dyn[:, CH["steps"]] = np.nan_to_num(steps / 100.0)
    hrv = s["hrv_rmssd"].to_numpy(dtype=float)
    dyn[:, CH["hrv"]] = np.nan_to_num((np.log(np.clip(hrv, 3, None)) - np.log(30)) / 0.5)
    dyn[:, CH["hrv_avail"]] = np.isfinite(hrv)
    stage = s["sleep_stage"].to_numpy(dtype=float)
    dyn[:, CH["asleep"]] = np.nan_to_num(stage) > 0
    dyn[:, CH["deep_sleep"]] = np.nan_to_num(stage) == 2
    dyn[:, CH["sleep_avail"]] = np.isfinite(stage)

    carbs = np.nan_to_num(s["carbs"].to_numpy(dtype=float))
    fp = np.nan_to_num(s["fat"].to_numpy(dtype=float)) + np.nan_to_num(s["protein"].to_numpy(dtype=float))
    ins = np.nan_to_num(s["insulin_units"].to_numpy(dtype=float))
    dyn[:, CH["carbs"]] = carbs / 50.0
    dyn[:, CH["cob"]] = _decay_sum(carbs, 90 / GRID) / 50.0
    dyn[:, CH["fpob"]] = _decay_sum(fp, 180 / GRID) / 50.0
    dyn[:, CH["insulin"]] = ins / 10.0
    dyn[:, CH["iob"]] = _decay_sum(ins, 150 / GRID) / 10.0
    dyn[:, CH["oad"]] = np.nan_to_num(s["oad_dose"].to_numpy(dtype=float)) > 0
    tod = (idx.hour * 60 + idx.minute).to_numpy() / 1440.0
    dyn[:, CH["tod_sin"]] = np.sin(2 * np.pi * tod)
    dyn[:, CH["tod_cos"]] = np.cos(2 * np.pi * tod)

    # anchors: a real CGM reading now, enough history, and some observed future
    obs = np.isfinite(cgm)
    hist_cov = pd.Series(ok.astype(float)).rolling(HISTORY, min_periods=1).mean().to_numpy()
    fut_cnt = np.array([obs[k + 1 : k + 1 + H].sum() for k in range(n)])
    cand = np.flatnonzero(obs & (np.arange(n) >= min_history) & (hist_cov >= 0.5) & (fut_cnt >= 4) & (np.arange(n) + H < n))
    if stride > 1 and obs.mean() > 0.6:
        cand = cand[cand % stride == 0]
    anchors = cand

    fut = anchors[:, None] + np.arange(1, H + 1)[None, :]
    y = (cgm[fut] - cgm[anchors][:, None]) / G_SCALE

    high = _runs(np.nan_to_num(cgm_in, nan=0) > 180)
    low = _runs(np.nan_to_num(cgm_in, nan=999) < 70)
    spike = high[fut].any(axis=1)
    hypo = low[fut].any(axis=1)
    spike_ok = ~high[anchors] & (cgm[anchors] <= 180)
    hypo_ok = ~low[anchors] & (cgm[anchors] >= 70)

    return PatientArrays(
        pid=str(static_row["patient_id"]), source=str(static_row.get("source")), group=str(static_row.get("group") or static_row["patient_id"]),
        status=str(static_row.get("status")), start=idx[0], dyn=dyn, cgm=cgm, cgm_in=cgm_in,
        static=static_vector(static_row), anchors=anchors, y=y.astype(np.float32),
        spike=spike, spike_ok=spike_ok, hypo=hypo, hypo_ok=hypo_ok,
    )
