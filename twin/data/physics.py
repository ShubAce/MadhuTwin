"""Attach mechanistic-twin features to a patient's arrays (the hybrid in 'hybrid twin').

Two twins are run per patient:
    prior        initialised from the EHR only (what a doctor's twin knows on day 1)
    personalised fitted on the calibration period (first days of wearable data), then synced

Both are synchronised with the UKF over the full record and asked for a strictly causal 2-hour
forecast at every anchor. The personalised forecast, plus the synced hidden state, become
inputs to the ML models; the prior forecast is kept to quantify the value of personalisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from twin.record import PatientRecord
from twin.twin import DigitalTwin

from .windows import G_SCALE, H, PatientArrays

UKF_FEATURES = ("si_log_mult", "insulin_action", "insulin_excess", "plasma_minus_isf", "glucose_sd")


def _features(tw: DigitalTwin, anchors_min: np.ndarray, cgm_now: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fc = tw.forecast(anchors_min)
    cols = np.arange(1, H + 1) * 5 - 1
    phys = (fc.gi[:, cols] - cgm_now[:, None]) / G_SCALE
    tr = tw.trace
    k = np.clip(np.searchsorted(tr.t, anchors_min, side="right") - 1, 0, len(tr.t) - 1)
    m, sd = tr.mean[k], tr.sd[k]
    ib = float(tw.params.Ib[0])
    ukf = np.stack([m[:, 6], m[:, 1] / 10.0, (m[:, 2] - ib) / 20.0, (m[:, 0] - m[:, 3]) / 20.0, sd[:, 0] / 20.0], axis=1)
    return phys.astype(np.float32), ukf.astype(np.float32)


def twin_features(arr: PatientArrays, static_row: dict, series: pd.DataFrame, events: pd.DataFrame,
                  calib_minutes: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Physics and UKF features from a twin personalised on the first `calib_minutes` of the record
    (0 = the EHR-prior twin), without modifying `arr`. Returns (physics, ukf, calibration bins)."""
    rec = PatientRecord.from_frames(static_row, series, events)
    offset = int((arr.start - rec.start).total_seconds() // 60)
    tw = DigitalTwin.from_record(rec)
    if calib_minutes > 0:
        try:
            tw.personalize(calib_minutes)
        except ValueError:  # too little calibration data: stay on the EHR prior
            pass
    tw.sync()
    phys, ukf = _features(tw, arr.anchors * 5 + 2 + offset, arr.cgm[arr.anchors])
    return phys, ukf, max((calib_minutes - offset) // 5, 0)


def attach_physics(arr: PatientArrays, static_row: dict, series: pd.DataFrame, events: pd.DataFrame,
                   calib_minutes: int) -> PatientArrays:
    rec = PatientRecord.from_frames(static_row, series, events)
    offset = int((arr.start - rec.start).total_seconds() // 60)
    anchors_min = arr.anchors * 5 + 2 + offset
    cgm_now = arr.cgm[arr.anchors]

    prior = DigitalTwin.from_record(rec)
    prior.sync()
    arr.physics_prior, _ = _features(prior, anchors_min, cgm_now)

    tw = DigitalTwin.from_record(rec)
    try:
        fit = tw.personalize(calib_minutes)
        arr.extra["fit"] = {"params": fit.params, "tau_scale": fit.tau_scale, "rmse_prior": fit.rmse_prior, "rmse_fit": fit.rmse_fit}
    except ValueError as e:  # too little calibration data: stay on the EHR prior
        arr.extra["fit_error"] = str(e)
    tw.sync()
    arr.physics, arr.ukf = _features(tw, anchors_min, cgm_now)
    arr.extra["si_multiplier_daily"] = (
        pd.Series(tw.trace.si_multiplier, index=rec.start + pd.to_timedelta(tw.trace.t, unit="min")).resample("1D").mean().round(3).tolist()
    )
    arr.calib_bins = max((calib_minutes - offset) // 5, 0)
    return arr
