"""Mechanistic glucose-insulin model: the physiological core of the digital twin.

An extended Bergman minimal model with endogenous insulin secretion, suitable for Type 2
Diabetes, prediabetes and normoglycaemia:

    dG/dt  = -SG (G - Gb_eff) - SI m_si X G + Ra(t) - k_ex E1 G - renal(G) + counterreg(G)
    dX/dt  = -p2 (X - (I - Ib))
    dI/dt  = -n (I - Ib) + beta m_beta [G - Gb_eff]+ + k_inc m_inc Ra(t) + Iex(t) + SU(t)
    dGi/dt = (G - Gi) / tau_isf                       (interstitial glucose seen by a CGM)
    dE1/dt = (A(t) - E1) / 10 ;  dE2/dt = (A(t) - E2) / 360   (acute / lingering exercise)

The fasting state (Gb, Ib) is the patient's equilibrium *on their usual therapy*, so the
EHR's fasting glucose and insulin anchor the model directly. Daily-life effects enter as
time-varying multipliers built by `twin.physiology.inputs`:

* m_si   insulin-sensitivity multiplier (sleep loss, illness, stress, post-exercise boost)
* egp    endogenous-glucose multiplier on Gb (dawn phenomenon, illness, stress, metformin what-ifs)
* m_beta sulfonylurea enhancement of glucose-stimulated secretion; SU(t) glucose-independent part
* m_inc  incretin multiplier (DPP-4 inhibitors)
* renal  glycosuria above a renal threshold that SGLT2 inhibitors lower from ~180 to ~90 mg/dL

All functions are vectorised over a leading batch axis, so the same code simulates a cohort,
a single patient, or the sigma points of the Unscented Kalman Filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

import numpy as np

STATE_NAMES = ("G", "X", "I", "Gi", "E1", "E2")
G_IDX, X_IDX, I_IDX, GI_IDX, E1_IDX, E2_IDX = range(6)

HYPO_COUNTERREG_THRESHOLD = 70.0  # mg/dL, glucagon response begins
U_KEYS = ("ra", "mets", "tod", "iex", "su", "m_si", "egp", "m_beta", "m_inc", "renal_thr")


@dataclass
class Physiology:
    """Per-patient parameters. Every field is an array of shape (n,) (batch of patients)."""

    weight: np.ndarray        # kg
    Gb: np.ndarray            # fasting plasma glucose on usual therapy, mg/dL
    Ib: np.ndarray            # fasting plasma insulin, uU/mL
    SI: np.ndarray            # insulin sensitivity, 1/min per uU/mL
    SG: np.ndarray            # glucose effectiveness, 1/min
    p2: np.ndarray            # remote insulin action rate, 1/min
    n: np.ndarray             # insulin clearance, 1/min
    beta: np.ndarray          # glucose-stimulated secretion, uU/mL/min per mg/dL
    k_inc: np.ndarray         # incretin gain, uU/mL per (mg/dL) of appearing glucose
    dawn: np.ndarray          # dawn phenomenon amplitude (fraction of Gb)
    tau_isf: np.ndarray       # plasma -> interstitial lag, min
    VG: np.ndarray            # glucose distribution volume, dL/kg
    VI: np.ndarray            # insulin distribution volume, mL/kg
    k_ex: np.ndarray          # acute exercise glucose uptake, 1/min per MET above light
    k_ex_si: np.ndarray       # post-exercise insulin-sensitivity gain per lingering MET
    k_renal: np.ndarray       # renal glucose excretion rate above threshold, 1/min
    k_cr: np.ndarray          # counter-regulatory gain below 70 mg/dL, 1/min
    f_bio: np.ndarray = field(default=None)            # meal carbohydrate bioavailability
    renal_thr_base: np.ndarray = field(default=None)   # renal threshold on usual therapy, mg/dL

    _DEFAULTS = {"f_bio": 0.9, "renal_thr_base": 180.0}

    def __post_init__(self) -> None:
        n = len(np.atleast_1d(self.Gb))
        for f in fields(self):
            v = getattr(self, f.name)
            if v is None:
                v = self._DEFAULTS[f.name]
            setattr(self, f.name, np.broadcast_to(np.asarray(v, dtype=float), (n,)).copy())

    def __len__(self) -> int:
        return len(self.Gb)

    def take(self, idx) -> Physiology:
        return Physiology(**{f.name: np.atleast_1d(getattr(self, f.name)[idx]) for f in fields(self)})

    def repeat(self, k: int) -> Physiology:
        """Repeat a single patient's parameters k times (e.g. for UKF sigma points)."""
        return Physiology(**{f.name: np.repeat(getattr(self, f.name), k) for f in fields(self)})

    def with_(self, **kw) -> Physiology:
        return replace(self, **{k: np.broadcast_to(np.asarray(v, float), (len(self),)).copy() for k, v in kw.items()})

    def to_records(self) -> list[dict]:
        return [{f.name: float(getattr(self, f.name)[i]) for f in fields(self)} for i in range(len(self))]

    @classmethod
    def from_records(cls, recs: list[dict]) -> Physiology:
        return cls(**{f.name: np.array([r.get(f.name, cls._DEFAULTS.get(f.name)) for r in recs], float)
                      for f in fields(cls)})

    @property
    def homa_ir(self) -> np.ndarray:
        return self.Gb * self.Ib / 405.0

    @property
    def homa_b(self) -> np.ndarray:
        return 360.0 * self.Ib / np.maximum(self.Gb - 63.0, 5.0)


def default_physiology(n: int = 1, **overrides) -> Physiology:
    """Typical adult with T2D on oral therapy; overrides accept scalars or (n,) arrays."""
    base = dict(
        weight=68.0, Gb=150.0, Ib=12.0, SI=1.8e-4, SG=0.012, p2=0.025, n=0.16, beta=0.09,
        k_inc=1.2, dawn=0.12, tau_isf=8.0, VG=1.7, VI=120.0, k_ex=0.0028, k_ex_si=0.45,
        k_renal=0.006, k_cr=0.06, f_bio=0.9, renal_thr_base=180.0,
    )
    base.update(overrides)
    return Physiology(**{k: np.broadcast_to(np.asarray(v, float), (n,)).copy() for k, v in base.items()})


@dataclass
class Inputs:
    """Exogenous drivers on a 1-minute grid. Arrays have shape (n, T) unless noted."""

    ra: np.ndarray                  # meal glucose appearance, mg/dL/min
    mets: np.ndarray                # metabolic equivalents (1 = rest)
    tod: np.ndarray                 # minute of day, shape (T,) or (n, T)
    iex: np.ndarray | None = None   # exogenous insulin appearance, uU/mL/min
    su: np.ndarray | None = None    # sulfonylurea glucose-independent secretion, uU/mL/min
    m_si: np.ndarray | None = None
    egp: np.ndarray | None = None
    m_beta: np.ndarray | None = None
    m_inc: np.ndarray | None = None
    renal_thr: np.ndarray | None = None  # mg/dL

    def __post_init__(self) -> None:
        self.ra = np.atleast_2d(np.asarray(self.ra, float))
        n, T = self.ra.shape
        self.mets = np.broadcast_to(np.atleast_2d(np.asarray(self.mets, float)), (n, T))
        self.tod = np.broadcast_to(np.asarray(self.tod, float), (n, T))
        defaults = {"iex": 0.0, "su": 0.0, "m_si": 1.0, "egp": 1.0, "m_beta": 1.0, "m_inc": 1.0, "renal_thr": 180.0}
        for name, d in defaults.items():
            v = getattr(self, name)
            setattr(self, name, np.broadcast_to(np.asarray(d if v is None else v, float), (n, T)))

    @property
    def shape(self) -> tuple[int, int]:
        return self.ra.shape

    def window(self, t0: int, t1: int) -> Inputs:
        return Inputs(**{k: getattr(self, k)[:, t0:t1] for k in U_KEYS})


def dawn_profile(tod: np.ndarray) -> np.ndarray:
    """Early-morning rise in hepatic glucose output, peaking ~06:30."""
    d = (tod - 390.0 + 720.0) % 1440.0 - 720.0
    return np.exp(-0.5 * (d / 80.0) ** 2)


def initial_state(p: Physiology) -> np.ndarray:
    n = len(p)
    x = np.zeros((n, 6))
    x[:, G_IDX] = p.Gb
    x[:, I_IDX] = p.Ib
    x[:, GI_IDX] = p.Gb
    return x


def derivatives(x: np.ndarray, p: Physiology, u: dict[str, np.ndarray]) -> np.ndarray:
    """State derivatives for a batch: x is (n, 6); each u[...] is (n,)."""
    G, X, I, Gi, E1, E2 = (x[:, k] for k in range(6))
    gb_eff = p.Gb * u["egp"] * (1.0 + p.dawn * dawn_profile(u["tod"]))
    activity = np.maximum(u["mets"] - 1.5, 0.0)
    si_eff = p.SI * u["m_si"] * (1.0 + p.k_ex_si * E2)

    # Glycosuria relative to the fasting equilibrium on usual therapy, so a change of renal
    # threshold (starting / stopping an SGLT2 inhibitor) shifts glucose while Gb stays anchored.
    renal = p.k_renal * (np.maximum(G - u["renal_thr"], 0.0) - np.maximum(gb_eff - p.renal_thr_base, 0.0))
    counter = p.k_cr * np.maximum(HYPO_COUNTERREG_THRESHOLD - G, 0.0)
    # Below the set-point the liver raises output to restore glucose. Drug-driven insulin
    # (injections, sulfonylureas) is not switched off as glucose falls and suppresses that
    # response: the mechanism behind hypos after a dose without food. Endogenous secretion
    # self-limits, so people not on these drugs keep a symmetric restoring force.
    drug_insulin = (u["iex"] + np.maximum(u["su"], 0.0)) / p.n
    restore = np.where(G < gb_eff, 1.0 / (1.0 + drug_insulin / 8.0), 1.0)
    dG = (-p.SG * restore * (G - gb_eff) - si_eff * X * G + u["ra"] - p.k_ex * E1 * G - renal + counter)

    dX = -p.p2 * (X - (I - p.Ib))
    secretion = p.beta * u["m_beta"] * np.maximum(G - gb_eff, 0.0) + p.k_inc * u["m_inc"] * u["ra"]
    dI = -p.n * (I - p.Ib) + secretion + u["iex"] + u["su"]
    dGi = (G - Gi) / p.tau_isf
    dE1 = (activity - E1) / 10.0
    dE2 = (activity - E2) / 360.0
    return np.stack([dG, dX, dI, dGi, dE1, dE2], axis=1)


def step(x: np.ndarray, p: Physiology, u: dict[str, np.ndarray], dt: float = 1.0) -> np.ndarray:
    """One explicit-Euler step (stable for dt <= 2 min with physiological rates)."""
    x_new = x + dt * derivatives(x, p, u)
    x_new[:, G_IDX] = np.clip(x_new[:, G_IDX], 20.0, 700.0)
    x_new[:, I_IDX] = np.maximum(x_new[:, I_IDX], 0.0)
    x_new[:, GI_IDX] = np.clip(x_new[:, GI_IDX], 20.0, 700.0)
    return x_new


def _u_at(inputs: Inputs, t: int) -> dict[str, np.ndarray]:
    return {k: getattr(inputs, k)[:, t] for k in U_KEYS}


def simulate(p: Physiology, inputs: Inputs, x0: np.ndarray | None = None, full: bool = False) -> dict[str, np.ndarray]:
    """Integrate the model over the input horizon on a 1-minute grid.

    Uses the compiled integrator (twin.physiology.fast) and returns G, Gi and the final state;
    `full=True` runs the readable reference implementation and returns every state.
    """
    n, T = inputs.shape
    if len(p) != n:
        raise ValueError(f"physiology batch {len(p)} != inputs batch {n}")
    x = initial_state(p) if x0 is None else np.array(x0, dtype=float, copy=True)
    if full:
        return simulate_reference(p, inputs, x)
    from .fast import integrate, pack_inputs, pack_params
    G, Gi, xf = integrate(np.ascontiguousarray(x), pack_params(p), pack_inputs(inputs))
    return {"G": G, "Gi": Gi, "final_state": xf}


def simulate_reference(p: Physiology, inputs: Inputs, x: np.ndarray) -> dict[str, np.ndarray]:
    n, T = inputs.shape
    out = np.empty((n, T, 6))
    for t in range(T):
        x = step(x, p, _u_at(inputs, t))
        out[:, t] = x
    res = {name: out[:, :, k] for k, name in enumerate(STATE_NAMES)}
    res["final_state"] = x
    return res


def propagate(x: np.ndarray, p: Physiology, inputs: Inputs, t0: int, minutes: int) -> np.ndarray:
    """Advance state x from minute t0 for `minutes` steps (used by the UKF and forecaster)."""
    for t in range(t0, t0 + minutes):
        x = step(x, p, _u_at(inputs, t))
    return x


def warm_start(p: Physiology, inputs: Inputs, minutes: int = 1440) -> np.ndarray:
    """Run the first `minutes` of inputs to obtain a realistic, non-equilibrium initial state."""
    return simulate(p, inputs.window(0, minutes))["final_state"]
