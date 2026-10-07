"""Consumer-device sensor models that turn true physiology into realistic wearable streams.

The goal is not perfect data but *realistic imperfection*, so that models trained on the
synthetic cohort learn to cope with what real devices produce:

CGM            interstitial lag (from the physiology model), per-session calibration gain/offset,
               autocorrelated noise, 1-h warm-up after each sensor change, dropouts, and
               nocturnal compression lows (lying on the sensor), clipped to 40-400 mg/dL.
Heart rate     resting rate + activity + circadian/sleep dip + stress + illness + adrenergic
               response to hypoglycaemia; off-wrist gaps while charging.
HRV (RMSSD)    sleep-stage dependent; suppressed by stress, illness, sleep debt, activity and
               marked hyperglycaemia; only reported when still (as smartwatches do).
"""

from __future__ import annotations

import warnings

import numpy as np

from .lifestyle import AWAKE, DEEP, LIGHT, REM, Lifestyle

GRID = 5


def cgm_readings(gi: np.ndarray, asleep: np.ndarray, rng: np.random.Generator, session_days: int = 14) -> np.ndarray:
    """5-min CGM readings from per-minute interstitial glucose."""
    T = len(gi)
    n = T // GRID
    idx = np.arange(n) * GRID + 2
    true = gi[idx].astype(float)
    session = (idx // (session_days * 1440)).astype(int)
    n_sessions = int(session.max()) + 1
    gain = rng.normal(1.0, 0.035, n_sessions)[session]
    offset = rng.normal(0.0, 3.0, n_sessions)[session]
    ar = np.zeros(n)
    eps = rng.normal(0, 6.5, n)
    for k in range(1, n):
        ar[k] = 0.75 * ar[k - 1] + eps[k]
    y = gain * true + offset + ar + rng.normal(0, 3.0, n)

    # compression lows during sleep (~15% of nights)
    sleep_bins = asleep[idx].astype(int)
    # true sleep onsets only: asleep now and awake for the whole preceding hour
    prior = np.convolve(np.r_[np.zeros(12), sleep_bins], np.ones(12), "valid")[:n]
    for start in np.flatnonzero((sleep_bins == 1) & (prior == 0)):
        if rng.random() < 0.15:
            s = start + int(rng.integers(12, 60))
            L = int(rng.integers(4, 10))
            depth = float(rng.uniform(25, 45))
            prof = depth * np.sin(np.linspace(0, np.pi, L))
            y[s : s + L] -= prof[: max(0, min(L, n - s))]

    y = np.clip(np.round(y), 40, 400)
    # warm-up after each sensor change and random signal loss
    for s in range(n_sessions):
        first = np.flatnonzero(session == s)[0]
        y[first : first + 12] = np.nan
    for _ in range(rng.poisson(0.35 * T / 1440)):
        s = int(rng.integers(0, n))
        y[s : s + int(rng.integers(3, 24))] = np.nan
    return y


def heart_rate(life: Lifestyle, glucose: np.ndarray, rhr: float, rng: np.random.Generator) -> np.ndarray:
    """Per-minute heart rate (bpm)."""
    T = life.T
    tod = (np.arange(T) % 1440) / 1440.0
    asleep = life.sleep_stage > 0
    hr = (rhr + 2.5 * np.sin(2 * np.pi * (tod - 0.35))
          + 11.0 * np.maximum(life.mets - 1.3, 0)
          - np.where(asleep, 7.0 + 2.0 * (life.sleep_stage == DEEP), 0.0)
          + 7.0 * life.stress + 12.0 * life.illness
          + 0.45 * np.maximum(70.0 - glucose, 0.0)
          + 0.02 * np.maximum(glucose - 200.0, 0.0))
    noise = np.zeros(T)
    e = rng.normal(0, 1.2, T)
    for t in range(1, T):
        noise[t] = 0.85 * noise[t - 1] + e[t]
    return np.clip(hr + noise, 38, 195)


def hrv_rmssd(life: Lifestyle, glucose: np.ndarray, base: float, rng: np.random.Generator) -> np.ndarray:
    """5-min RMSSD (ms); NaN when the wearer is moving or the watch does not sample."""
    n = life.T // GRID
    idx = np.arange(n) * GRID + 2
    st = life.sleep_stage[idx]
    stage_f = np.select([st == DEEP, st == LIGHT, st == REM], [1.35, 1.12, 0.92], default=0.85)
    mets = life.mets[idx]
    debt = np.zeros(n)
    for night in life.nights:
        if night["deficit"] > 1.5:
            lo, hi = max(night["wake"], 0) // GRID, min(night["wake"] + 900, life.T) // GRID
            debt[lo:hi] = 1.0
    r = (base * stage_f * (1 - 0.25 * life.stress[idx]) * (1 - 0.40 * life.illness[idx]) * (1 - 0.08 * debt)
         * np.exp(-0.15 * np.maximum(mets - 1.3, 0)) * np.where(glucose[idx] > 250, 0.9, 1.0)
         * np.exp(rng.normal(0, 0.18, n)))
    still = mets < 1.6
    sampled = (st != AWAKE) | (rng.random(n) < 0.35)
    return np.where(still & sampled, np.round(r, 1), np.nan)


def wear_gaps(T: int, rng: np.random.Generator) -> np.ndarray:
    """Boolean per-minute mask: True when the watch is off-wrist (daily charging, ~45 min)."""
    off = np.zeros(T, dtype=bool)
    for d in range(T // 1440):
        s = d * 1440 + int(rng.normal(19 * 60, 90))
        off[max(s, 0) : min(s + int(rng.normal(45, 12)), T)] = True
    return off


def five_minute(x: np.ndarray, how: str = "mean") -> np.ndarray:
    n = len(x) // GRID
    v = x[: n * GRID].reshape(n, GRID)
    if how == "sum":
        return v.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(v, axis=1)
