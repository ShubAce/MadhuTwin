"""Numba-compiled integrator: the same equations as `model.derivatives`, compiled to machine code.

`model.py` remains the readable reference implementation; `tests/test_physiology.py` checks
that both produce identical trajectories. Parameters and inputs are packed into dense arrays:

    P  (n, NP)      per-patient parameters in PARAM_ORDER
    U  (n, T, NU)   per-minute inputs in model.U_KEYS order (or (T, NU) shared by all rows)
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .model import HYPO_COUNTERREG_THRESHOLD, U_KEYS, Inputs, Physiology

PARAM_ORDER = ("Gb", "Ib", "SI", "SG", "p2", "n", "beta", "k_inc", "dawn", "tau_isf", "k_ex", "k_ex_si",
               "k_renal", "k_cr", "renal_thr_base")
NP = len(PARAM_ORDER)
NU = len(U_KEYS)
_CR = HYPO_COUNTERREG_THRESHOLD


def pack_params(p: Physiology) -> np.ndarray:
    return np.ascontiguousarray(np.stack([getattr(p, k) for k in PARAM_ORDER], axis=1), dtype=np.float64)


def pack_inputs(u: Inputs) -> np.ndarray:
    return np.ascontiguousarray(np.stack([getattr(u, k) for k in U_KEYS], axis=-1), dtype=np.float64)


@njit(cache=True, fastmath=False)
def _step(x, P, u):
    """In-place one-minute Euler step for a single row. x: (6,), P: (NP,), u: (NU,)."""
    G, X, I, Gi, E1, E2 = x[0], x[1], x[2], x[3], x[4], x[5]
    Gb, Ib, SI, SG, p2, n, beta, k_inc, dawn, tau_isf, k_ex, k_ex_si, k_renal, k_cr, thr_base = (
        P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], P[12], P[13], P[14])
    ra, mets, tod, iex, su, m_si, egp, m_beta, m_inc, renal_thr = (
        u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9])
    d = (tod - 390.0 + 720.0) % 1440.0 - 720.0
    gb_eff = Gb * egp * (1.0 + dawn * np.exp(-0.5 * (d / 80.0) ** 2))
    activity = max(mets - 1.5, 0.0)
    si_eff = SI * m_si * (1.0 + k_ex_si * E2)
    renal = k_renal * (max(G - renal_thr, 0.0) - max(gb_eff - thr_base, 0.0))
    counter = k_cr * max(_CR - G, 0.0)
    restore = 1.0
    if G < gb_eff:
        restore = 1.0 / (1.0 + (iex + max(su, 0.0)) / n / 8.0)
    dG = -SG * restore * (G - gb_eff) - si_eff * X * G + ra - k_ex * E1 * G - renal + counter
    dX = -p2 * (X - (I - Ib))
    secretion = beta * m_beta * max(G - gb_eff, 0.0) + k_inc * m_inc * ra
    dI = -n * (I - Ib) + secretion + iex + su
    dGi = (G - Gi) / tau_isf
    dE1 = (activity - E1) / 10.0
    dE2 = (activity - E2) / 360.0
    x[0] = min(max(G + dG, 20.0), 700.0)
    x[1] = X + dX
    x[2] = max(I + dI, 0.0)
    x[3] = min(max(Gi + dGi, 20.0), 700.0)
    x[4] = E1 + dE1
    x[5] = E2 + dE2


@njit(cache=True)
def integrate(x0, P, U):
    """Simulate n rows over T minutes. Returns (G, Gi, final_state) with G, Gi of shape (n, T)."""
    n, T = U.shape[0], U.shape[1]
    G = np.empty((n, T))
    Gi = np.empty((n, T))
    xf = np.empty((n, 6))
    for i in range(n):
        x = x0[i].copy()
        for t in range(T):
            _step(x, P[i], U[i, t])
            G[i, t] = x[0]
            Gi[i, t] = x[3]
        xf[i] = x
    return G, Gi, xf


@njit(cache=True)
def integrate_shared(x0, P, U, t0, minutes):
    """Advance n rows that share one input stream U (T, NU) from minute t0 for `minutes` steps."""
    n = x0.shape[0]
    x = x0.copy()
    for i in range(n):
        for t in range(t0, t0 + minutes):
            _step(x[i], P[i], U[t])
    return x
