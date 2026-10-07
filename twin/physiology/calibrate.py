"""Fit a personal mechanistic twin to one person's EHR + CGM + meal + activity record.

Maximum-a-posteriori estimation of five physiological parameters:

    Gb    fasting glucose set-point         prior: lab fasting plasma glucose
    SI    insulin sensitivity               prior: 8.4e-4 / HOMA-IR (from fasting glucose & insulin)
    beta  glucose-stimulated secretion      prior: by glycaemic status
    dawn  dawn-phenomenon amplitude         prior: 0.08
    tau_s meal absorption speed multiplier  prior: 1.0

The EHR therefore *initialises* the twin and the wearable stream *personalises* it. Fitting uses
multiple shooting: the record is cut into overlapping windows (60 min burn-in + 180 min scored),
each initialised from the observed CGM, and all windows are simulated as one vectorised batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from twin.record import PatientRecord

from .fast import integrate, pack_inputs, pack_params
from .model import G_IDX, GI_IDX, I_IDX, U_KEYS, Inputs, Physiology, default_physiology, initial_state

STATUS_BETA_PRIOR = {"normal": 0.13, "prediabetes": 0.09, "t2d": 0.035}
SIGMA_CGM = 15.0  # mg/dL, combined sensor + model noise used to weight residuals
BURN_IN, SCORED, STRIDE = 60, 180, 120  # minutes
PARAM_NAMES = ("Gb", "SI", "beta", "dawn", "tau_scale")


def k_inc_from_beta(beta: np.ndarray | float) -> np.ndarray | float:
    """Incretin gain co-varies with beta-cell function (calibrated against OGTT responses)."""
    return 0.35 + 4.0 * np.asarray(beta)


def prior_physiology(static: dict) -> Physiology:
    """Population-informed starting twin from the EHR alone (no wearable data yet)."""
    def num(key, default):
        v = static.get(key)
        return default if v is None or not np.isfinite(float(v)) else float(v)

    fpg = float(np.clip(num("fpg_mgdl", 110.0), 70, 350))
    ins = float(np.clip(num("fasting_insulin_uU", 10.0), 2.0, 60.0))
    homa = fpg * ins / 405.0
    status = static.get("status") or "t2d"
    beta = STATUS_BETA_PRIOR.get(status, 0.06)
    return default_physiology(
        1, weight=num("weight_kg", 70.0), Gb=fpg, Ib=ins, SI=float(np.clip(8.4e-4 / max(homa, 0.3), 5e-5, 2.5e-3)),
        beta=beta, k_inc=k_inc_from_beta(beta), dawn=0.08,
        SG=0.018 if status == "normal" else (0.015 if status == "prediabetes" else 0.012),
        renal_thr_base=90.0 if float(static.get("on_sglt2") or 0) > 0 else 180.0,
    )


def _windows(rec: PatientRecord, min_coverage: float = 2 / 3) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Window starts (minutes), initial CGM, and per-window scored observation indices/values.

    A window is kept when it holds at least `min_coverage` of the readings the sensor would
    produce over the scored period (5-min Dexcom, 15-min Libre/Shanghai alike).
    """
    interval = float(np.median(np.diff(rec.cgm_t))) if len(rec.cgm_t) > 1 else 5.0
    min_obs = max(int(round(min_coverage * SCORED / max(interval, 1.0))), 4)
    starts, g0, obs_t, obs_y = [], [], [], []
    cgm_at = dict(zip(rec.cgm_t.tolist(), rec.cgm.tolist(), strict=True))
    sparse = interval > 5.0
    for s in range(0, rec.T - BURN_IN - SCORED, STRIDE):
        init = cgm_at.get(s + 2)
        if init is None and sparse:
            # sparse sensors drift in phase after restarts: start from the next reading within 15 min
            k = int(np.searchsorted(rec.cgm_t, s + 2))
            if k < len(rec.cgm_t) and rec.cgm_t[k] - (s + 2) < 15:
                s = int(rec.cgm_t[k]) - 2
                init = float(rec.cgm[k])
        if init is None or s + BURN_IN + SCORED > rec.T:
            continue
        sel = (rec.cgm_t >= s + BURN_IN) & (rec.cgm_t < s + BURN_IN + SCORED)
        if sel.sum() < min_obs:
            continue
        starts.append(s)
        g0.append(init)
        obs_t.append(rec.cgm_t[sel] - s)
        obs_y.append(rec.cgm[sel])
    return np.array(starts, int), np.array(g0, float), obs_t, obs_y


def window_inputs(full: Inputs, starts: np.ndarray, length: int) -> Inputs:
    """Slice a single-patient Inputs into a batch of windows starting at `starts`."""
    idx = starts[:, None] + np.arange(length)[None, :]
    return Inputs(**{k: getattr(full, k)[0][idx] for k in U_KEYS})


def simulate_windows(p1: Physiology, rec: PatientRecord, starts: np.ndarray, g0: np.ndarray, length: int) -> np.ndarray:
    """Simulate every window from its observed starting glucose; returns Gi of shape (n_win, length)."""
    u = window_inputs(rec.inputs(p1, extra=length), starts, length)
    pw = p1.repeat(len(starts))
    x = initial_state(pw)
    x[:, G_IDX] = g0
    x[:, GI_IDX] = g0
    x[:, I_IDX] = pw.Ib
    _, gi, _ = integrate(x, pack_params(pw), pack_inputs(u))
    return gi


@dataclass
class FitResult:
    patient_id: str
    params: dict
    tau_scale: float
    rmse_fit: float
    rmse_prior: float
    n_windows: int
    success: bool

    def physiology(self) -> Physiology:
        return Physiology.from_records([self.params])


def fit_patient(rec: PatientRecord, max_iter: int = 60) -> FitResult:
    prior = prior_physiology(rec.static)
    starts, g0, obs_t, obs_y = _windows(rec)
    if len(starts) < 5:
        raise ValueError(f"{rec.patient_id}: not enough CGM windows to fit ({len(starts)})")
    length = BURN_IN + SCORED
    rows = np.concatenate([np.full(len(t), i) for i, t in enumerate(obs_t)])
    cols = np.concatenate(obs_t)
    y = np.concatenate(obs_y)

    mu = np.array([np.log(prior.Gb[0]), np.log(prior.SI[0]), np.log(prior.beta[0]), 0.08, 0.0])
    sd = np.array([0.15, 0.4, 0.6, 0.08, 0.3])

    def unpack(theta: np.ndarray) -> tuple[Physiology, float]:
        beta = float(np.exp(theta[2]))
        p = prior.with_(Gb=np.exp(theta[0]), SI=np.exp(theta[1]), beta=beta, k_inc=k_inc_from_beta(beta),
                        dawn=float(np.clip(theta[3], 0.0, 0.4)))
        return p, float(np.exp(theta[4]))

    def residuals(theta: np.ndarray) -> np.ndarray:
        p, tau_s = unpack(theta)
        sim = simulate_windows(p, rec.with_tau_scale(tau_s), starts, g0, length)
        return sim[rows, cols] - y

    def loss(theta: np.ndarray) -> float:
        r = residuals(theta) / SIGMA_CGM
        return float(np.mean(r**2) + np.sum(((theta - mu) / sd) ** 2) / len(starts))

    res = minimize(loss, mu.copy(), method="Powell",
                   bounds=[(np.log(60), np.log(350)), (np.log(3e-5), np.log(4e-3)), (np.log(0.008), np.log(0.6)),
                           (0.0, 0.4), (np.log(0.5), np.log(2.0))],
                   options={"maxiter": max_iter, "xtol": 1e-3, "ftol": 1e-4})
    p, tau_s = unpack(res.x)
    rmse = lambda th: float(np.sqrt(np.mean(residuals(th) ** 2)))  # noqa: E731
    return FitResult(rec.patient_id, p.to_records()[0], tau_s, rmse(res.x), rmse(mu), len(starts), bool(res.success))
