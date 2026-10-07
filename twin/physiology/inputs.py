"""Turn discrete life events (meals, doses, activity, sleep) into model input arrays."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .foods import absorption_time_constant
from .model import Inputs, Physiology

# Subcutaneous insulin absorption time constants (min); peak of the gamma-2 kernel = tau.
INSULIN_TAU = {"rapid": 35.0, "regular": 55.0, "nph": 240.0}
INSULIN_BIOAVAILABILITY = 0.7
PREMIX_REGULAR_FRACTION = 0.3  # 30/70 biphasic human insulin, the most common premix in India


@dataclass(frozen=True)
class Meal:
    t: int                 # minute index on the simulation grid
    carbs: float           # g
    protein: float = 0.0
    fat: float = 0.0
    fiber: float = 0.0
    gi: float = 60.0
    name: str = "meal"
    tau_scale: float = 1.0  # personal gastric-emptying speed (fitted per patient)

    @property
    def tau(self) -> float:
        return self.tau_scale * absorption_time_constant(self.carbs, self.fat, self.fiber, self.protein, self.gi)


@dataclass(frozen=True)
class Dose:
    t: int
    drug: str              # see DRUG_CLASSES
    amount: float          # IU for insulin, mg for oral agents


DRUG_CLASSES = {
    # insulin products -> absorption profile
    "insulin_rapid": "rapid", "insulin_regular": "regular", "insulin_premix_30_70": "premix",
    "insulin_nph": "nph",
    # oral agents
    "glimepiride": "sulfonylurea", "gliclazide": "sulfonylurea",
}
SU_REFERENCE_DOSE = {"glimepiride": 2.0, "gliclazide": 80.0}  # mg giving the reference effect


def gamma2_kernel(T: int, tau: float) -> np.ndarray:
    """Unit-area two-compartment absorption kernel k(t) = t/tau^2 * exp(-t/tau)."""
    t = np.arange(T, dtype=float)
    return t / tau**2 * np.exp(-t / tau)


def _add_kernel(out: np.ndarray, t0: int, amount: float, tau: float) -> None:
    T = len(out)
    if t0 >= T or amount <= 0:
        return
    span = min(T - max(t0, 0), int(8 * tau) + 1)
    if t0 < 0:  # event before the window: add its tail
        k = gamma2_kernel(span - t0, tau)[-t0:]
        out[: len(k)] += amount * k[: T]
        return
    out[t0 : t0 + span] += amount * gamma2_kernel(span, tau)


def meal_appearance(meals: Iterable[Meal], T: int, weight: float, VG: float, f_bio: float = 0.9) -> np.ndarray:
    """Rate of meal glucose appearance in plasma (mg/dL/min)."""
    ra = np.zeros(T)
    for m in meals:
        if m.carbs > 0:
            _add_kernel(ra, m.t, f_bio * m.carbs * 1000.0 / (VG * weight), m.tau)
    return ra


def insulin_appearance(doses: Iterable[Dose], T: int, weight: float, VI: float) -> np.ndarray:
    """Plasma appearance of injected insulin (uU/mL/min)."""
    iex = np.zeros(T)
    scale = INSULIN_BIOAVAILABILITY * 1e6 / (VI * weight)  # IU -> uU/mL
    for d in doses:
        kind = DRUG_CLASSES.get(d.drug)
        if kind in ("rapid", "regular", "nph"):
            _add_kernel(iex, d.t, d.amount * scale, INSULIN_TAU[kind])
        elif kind == "premix":
            _add_kernel(iex, d.t, PREMIX_REGULAR_FRACTION * d.amount * scale, INSULIN_TAU["regular"])
            _add_kernel(iex, d.t, (1 - PREMIX_REGULAR_FRACTION) * d.amount * scale, INSULIN_TAU["nph"])
    return iex


def sulfonylurea_drive(doses: Iterable[Dose], T: int, clearance_factor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Sulfonylurea effect: (glucose-independent secretion uU/mL/min, beta-cell multiplier).

    The glucose-independent component is expressed relative to its daily mean, because the
    patient's fasting equilibrium (Gb, Ib) already reflects average drug exposure. Reduced
    renal clearance (CKD, elderly) prolongs and amplifies exposure: a classic hypo risk.
    """
    level = np.zeros(T)
    for d in doses:
        if DRUG_CLASSES.get(d.drug) == "sulfonylurea":
            rel = d.amount / SU_REFERENCE_DOSE.get(d.drug, 1.0)
            _add_kernel(level, d.t, rel * 600.0 * clearance_factor, 150.0 * clearance_factor)
    if not level.any():
        return np.zeros(T), np.ones(T)
    su = 1.1 * (level - level.mean())
    m_beta = 1.0 + 0.35 * level / max(level.max(), 1e-9)
    return su, m_beta


def build_inputs(
    params: Physiology,
    T: int,
    start_minute_of_day: int | Sequence[int],
    meals: Sequence[Sequence[Meal]],
    mets: np.ndarray | None = None,
    doses: Sequence[Sequence[Dose]] | None = None,
    m_si: np.ndarray | None = None,
    egp: np.ndarray | None = None,
    therapy: Sequence[set[str]] | None = None,
    su_clearance: Sequence[float] | None = None,
) -> Inputs:
    """Assemble a batch of model inputs on a 1-minute grid of length T.

    `therapy` lists continuous drug classes per patient: "dpp4" amplifies incretin secretion,
    "sglt2" lowers the renal glucose threshold. Discrete doses (insulin, sulfonylureas) come
    from `doses`.
    """
    n = len(params)
    starts = np.broadcast_to(np.asarray(start_minute_of_day), (n,))
    tod = (starts[:, None] + np.arange(T)[None, :]) % 1440
    ra = np.stack([meal_appearance(meals[i], T, params.weight[i], params.VG[i], params.f_bio[i]) for i in range(n)])
    doses = doses or [[] for _ in range(n)]
    iex = np.stack([insulin_appearance(doses[i], T, params.weight[i], params.VI[i]) for i in range(n)])
    su_pairs = [sulfonylurea_drive(doses[i], T, 1.0 if su_clearance is None else su_clearance[i]) for i in range(n)]
    su = np.stack([s for s, _ in su_pairs])
    m_beta = np.stack([b for _, b in su_pairs])
    therapy = therapy or [set() for _ in range(n)]
    m_inc = np.array([1.7 if "dpp4" in th else 1.0 for th in therapy])[:, None] * np.ones((1, T))
    renal_thr = np.array([90.0 if "sglt2" in th else 180.0 for th in therapy])[:, None] * np.ones((1, T))
    return Inputs(
        ra=ra, mets=np.ones((n, T)) if mets is None else mets, tod=tod, iex=iex, su=su,
        m_si=m_si, egp=egp, m_beta=m_beta, m_inc=m_inc, renal_thr=renal_thr,
    )
