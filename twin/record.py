"""Minute-resolution view of one person's data, as consumed by the mechanistic twin.

A `PatientRecord` is built from the common-schema tables (static row, 5-min series, events)
and only contains what a deployed system would actually know: *logged* meals and doses,
measured CGM and activity. It never sees simulator ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from twin.physiology.inputs import DRUG_CLASSES, Dose, Meal, build_inputs
from twin.physiology.model import Inputs, Physiology

MAX_MEAL_CARBS = 250.0
DEFAULT_GI = 55.0
REST_METS = 1.2


@dataclass
class PatientRecord:
    patient_id: str
    start: pd.Timestamp
    T: int                       # minutes covered
    tod0: int                    # minute of day at start
    cgm_t: np.ndarray            # minute index of each CGM observation
    cgm: np.ndarray              # mg/dL
    mets: np.ndarray             # (T,) metabolic equivalents
    meals: list[Meal]
    doses: list[Dose] = field(default_factory=list)
    therapy: set[str] = field(default_factory=set)   # continuous drug classes: dpp4, sglt2
    static: dict = field(default_factory=dict)

    @classmethod
    def from_frames(cls, static_row: pd.Series | dict, series: pd.DataFrame, events: pd.DataFrame,
                    tau_scale: float = 1.0) -> PatientRecord:
        static = dict(static_row)
        series = series.sort_values("ts")
        start = series["ts"].iloc[0]
        minutes = ((series["ts"] - start).dt.total_seconds() // 60).astype(int).to_numpy()
        T = int(minutes[-1]) + 5
        mets = np.full(T, REST_METS)
        m = series["mets"].to_numpy(dtype=float)
        for k, t in enumerate(minutes):
            if np.isfinite(m[k]):
                mets[t : t + 5] = m[k]
        ok = series["cgm"].notna().to_numpy()

        meals, doses = [], []
        ev = events.sort_values("ts")
        rel = ((ev["ts"] - start).dt.total_seconds() // 60).astype(int).to_numpy()
        for (_, e), t in zip(ev.iterrows(), rel, strict=True):
            if not 0 <= t < T:
                continue
            kind = e["kind"]
            if kind == "meal":
                carbs = float(np.nan_to_num(e["carbs"]))
                if carbs > 0:
                    gi = float(e["gi"]) if pd.notna(e.get("gi")) else DEFAULT_GI
                    meals.append(Meal(int(t), min(carbs, MAX_MEAL_CARBS), float(np.nan_to_num(e["protein"])),
                                      float(np.nan_to_num(e["fat"])), float(np.nan_to_num(e["fiber"])), gi,
                                      str(e["label"]), tau_scale=tau_scale))
            elif kind in ("insulin", "oad") and str(e["label"]) in DRUG_CLASSES:
                if str(e["label"]) == "insulin_glargine":
                    continue  # basal insulin is part of the fasting equilibrium
                doses.append(Dose(int(t), str(e["label"]), float(np.nan_to_num(e["amount"]))))
        therapy = {k for k, col in (("dpp4", "on_dpp4"), ("sglt2", "on_sglt2")) if float(static.get(col) or 0) > 0}
        return cls(patient_id=str(static["patient_id"]), start=start, T=T, tod0=int(start.hour * 60 + start.minute),
                   cgm_t=minutes[ok] + 2, cgm=series["cgm"].to_numpy(dtype=float)[ok], mets=mets,
                   meals=meals, doses=doses, therapy=therapy, static=static)

    @property
    def renal_thr_base(self) -> float:
        return 90.0 if "sglt2" in self.therapy else 180.0

    @property
    def su_clearance(self) -> float:
        egfr = self.static.get("egfr")
        age = self.static.get("age") or 0
        return 1.4 if ((egfr is not None and pd.notna(egfr) and egfr < 60) or age >= 70) else 1.0

    def with_tau_scale(self, tau_scale: float) -> PatientRecord:
        return replace(self, meals=[replace(m, tau_scale=tau_scale) for m in self.meals])

    def inputs(self, p: Physiology, extra: int = 0, meals: list[Meal] | None = None,
               doses: list[Dose] | None = None, mets: np.ndarray | None = None,
               therapy: set[str] | None = None) -> Inputs:
        """Model inputs over [0, T + extra) minutes for a single-patient Physiology."""
        T = self.T + extra
        mt = np.full(T, REST_METS)
        src = self.mets if mets is None else mets
        mt[: min(len(src), T)] = src[:T]
        return build_inputs(p, T, self.tod0, [self.meals if meals is None else meals], mets=mt[None, :],
                            doses=[self.doses if doses is None else doses],
                            therapy=[self.therapy if therapy is None else therapy], su_clearance=[self.su_clearance])

    def truncate(self, minutes: int) -> PatientRecord:
        """The record as known at `minutes` after start (for calibration on early days only)."""
        keep = self.cgm_t < minutes
        return replace(self, T=min(minutes, self.T), cgm_t=self.cgm_t[keep], cgm=self.cgm[keep],
                       mets=self.mets[:minutes], meals=[m for m in self.meals if m.t < minutes],
                       doses=[d for d in self.doses if d.t < minutes])
