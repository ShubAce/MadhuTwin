"""Unscented Kalman Filter that keeps the digital twin synchronised with live CGM data.

Augmented state (7):  G, X, I, Gi, E1, E2 (physiology, see model.py)  +  s = log insulin-
sensitivity multiplier, modelled as a slow random walk. Every 5 minutes the filter

    1. propagates all sigma points through the mechanistic model (as one vectorised batch),
    2. corrects them with the new CGM reading (the interstitial state Gi is observed).

The filtered `s` is a live estimate of how insulin sensitivity is drifting away from the
patient's personal baseline: it falls during illness, stress or after poor sleep, which the
dashboard surfaces as an early-warning insight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from twin.physiology.fast import PARAM_ORDER, integrate_shared, pack_inputs, pack_params
from twin.physiology.model import G_IDX, GI_IDX, I_IDX, Inputs, Physiology, initial_state

N_PHYS = 6
S_IDX = 6
DIM = 7
STEP = 5  # minutes between filter updates

# process noise standard deviations per 5-minute step
Q_SD = np.array([3.0, 0.6, 1.2, 0.8, 0.02, 0.002, 0.008])
R_SD = 9.0           # CGM measurement noise (mg/dL)
S_BOUNDS = (-1.0, 0.7)


@dataclass
class SyncTrace:
    t: np.ndarray            # minute index of each filter step
    mean: np.ndarray         # (n_steps, 7) filtered state
    sd: np.ndarray           # (n_steps, 7) filtered standard deviations
    innovation: np.ndarray   # (n_steps,) CGM minus predicted (NaN when no reading)

    @property
    def si_multiplier(self) -> np.ndarray:
        return np.exp(self.mean[:, S_IDX])

    def state_at(self, minute: int) -> np.ndarray:
        k = int(np.clip(np.searchsorted(self.t, minute, side="right") - 1, 0, len(self.t) - 1))
        return self.mean[k]


def _sigma_points(m: np.ndarray, P: np.ndarray, c: float) -> np.ndarray:
    try:
        L = np.linalg.cholesky(c * P)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh((P + P.T) / 2)
        L = V @ np.diag(np.sqrt(np.clip(w, 1e-9, None) * c))
    return np.vstack([m, m + L.T, m - L.T])


def run_ukf(p: Physiology, inputs: Inputs, cgm_t: np.ndarray, cgm: np.ndarray, t_start: int | None = None,
            q_sd: np.ndarray = Q_SD, r_sd: float = R_SD) -> SyncTrace:
    """Filter a single patient's record. `p` is a one-patient Physiology; `inputs` covers the record."""
    T = inputs.shape[1]
    obs = dict(zip(cgm_t.tolist(), cgm.tolist(), strict=True))
    t0 = int(cgm_t[0]) if t_start is None else t_start

    n = DIM
    alpha, beta_, kappa = 1.0, 2.0, 0.0
    lam = alpha**2 * (n + kappa) - n
    wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = wm[0] + (1 - alpha**2 + beta_)

    m = np.zeros(n)
    m[:N_PHYS] = initial_state(p)[0]
    g0 = obs.get(t0, float(p.Gb[0]))
    m[G_IDX] = m[GI_IDX] = g0
    P = np.diag([20.0, 5.0, 8.0, 10.0, 0.3, 0.05, 0.3]) ** 2
    Q = np.diag(q_sd**2)

    pw = p.repeat(2 * n + 1)
    P_sig = pack_params(pw)
    si_col = PARAM_ORDER.index("SI")
    si_base = P_sig[:, si_col].copy()
    U = pack_inputs(inputs)[0]
    steps = list(range(t0, T - STEP, STEP))
    means = np.empty((len(steps), n))
    sds = np.empty((len(steps), n))
    innov = np.full(len(steps), np.nan)

    for k, t in enumerate(steps):
        # --- predict: propagate sigma points STEP minutes through the model ---
        X = _sigma_points(m, P, n + lam)
        X[:, S_IDX] = np.clip(X[:, S_IDX], *S_BOUNDS)
        P_sig[:, si_col] = si_base * np.exp(X[:, S_IDX])
        X[:, :N_PHYS] = integrate_shared(np.ascontiguousarray(X[:, :N_PHYS]), P_sig, U, t, STEP)
        m = wm @ X
        d = X - m
        P = (wc[:, None] * d).T @ d + Q

        # --- update with the CGM reading at the new time point, if any ---
        z = obs.get(t + STEP)
        if z is not None:
            Zs = X[:, GI_IDX]
            zhat = wm @ Zs
            dz = Zs - zhat
            S = wc @ (dz * dz) + r_sd**2
            Pxz = (wc * dz) @ d
            K = Pxz / S
            innov[k] = z - zhat
            m = m + K * (z - zhat)
            P = P - np.outer(K, K) * S
        P = (P + P.T) / 2 + np.eye(n) * 1e-9
        m[S_IDX] = np.clip(m[S_IDX], *S_BOUNDS)
        m[I_IDX] = max(m[I_IDX], 0.0)
        means[k] = m
        sds[k] = np.sqrt(np.clip(np.diag(P), 0, None))

    return SyncTrace(np.array(steps) + STEP, means, sds, innov)
