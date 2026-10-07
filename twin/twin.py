"""The DigitalTwin: one patient's virtual replica.

    twin = DigitalTwin.from_record(record)        # initialised from the EHR (population priors)
    twin.personalize(calibration_minutes=7*1440)  # MODEL:    learn this patient's physiology
    twin.sync()                                   # SYNC:     track hidden state from live CGM
    twin.forecast(anchors)                        # SIMULATE: physics forecast from any moment
    twin.what_if(minute, meals=[...], walk=...)   # SIMULATE: counterfactual scenarios

Forecasts are strictly causal: from an anchor time they use only meals and doses logged up to
that moment, the filtered physiological state, and resting activity thereafter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from twin.physiology.calibrate import FitResult, fit_patient, prior_physiology
from twin.physiology.fast import integrate, pack_inputs, pack_params
from twin.physiology.inputs import (
    DRUG_CLASSES,
    INSULIN_BIOAVAILABILITY,
    INSULIN_TAU,
    PREMIX_REGULAR_FRACTION,
    SU_REFERENCE_DOSE,
    Dose,
    Meal,
    gamma2_kernel,
)
from twin.physiology.model import Inputs, Physiology
from twin.record import REST_METS, PatientRecord
from twin.sync.ukf import N_PHYS, S_IDX, SyncTrace, run_ukf

HORIZON = 120


@dataclass
class Forecast:
    anchors: np.ndarray      # minute index of each anchor
    gi: np.ndarray           # (n_anchor, horizon) interstitial glucose (what a CGM would read)
    g: np.ndarray            # (n_anchor, horizon) plasma glucose


@dataclass
class DigitalTwin:
    record: PatientRecord
    params: Physiology
    tau_scale: float = 1.0
    fit: FitResult | None = None
    trace: SyncTrace | None = None
    personalized_until: int | None = None
    notes: dict = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: PatientRecord) -> DigitalTwin:
        return cls(record=record, params=prior_physiology(record.static))

    # ------------------------------------------------------------------ MODEL
    def personalize(self, calibration_minutes: int | None = None) -> FitResult:
        rec = self.record if calibration_minutes is None else self.record.truncate(calibration_minutes)
        self.fit = fit_patient(rec)
        p = self.fit.physiology()
        p.renal_thr_base[:] = self.record.renal_thr_base
        self.params = p
        self.tau_scale = self.fit.tau_scale
        self.personalized_until = calibration_minutes or self.record.T
        return self.fit

    # ------------------------------------------------------------------- SYNC
    @property
    def _record(self) -> PatientRecord:
        return self.record.with_tau_scale(self.tau_scale)

    def sync(self) -> SyncTrace:
        rec = self._record
        self.trace = run_ukf(self.params, rec.inputs(self.params), rec.cgm_t, rec.cgm)
        return self.trace

    # --------------------------------------------------------------- SIMULATE
    def _past_only_inputs(self, anchors: np.ndarray, horizon: int, mets_now: np.ndarray) -> Inputs:
        """Inputs over [a, a+horizon) using only events logged at or before each anchor a."""
        rec, p = self._record, self.params
        n = len(anchors)
        tau_ax = np.arange(horizon)
        ra = np.zeros((n, horizon))
        iex = np.zeros((n, horizon))
        su_level = np.zeros((n, horizon))
        w, VG, VI = float(p.weight[0]), float(p.VG[0]), float(p.VI[0])

        def add(target: np.ndarray, t_event: int, amount: float, tau: float) -> None:
            L = int(8 * tau) + horizon
            sel = (anchors >= t_event) & (anchors - t_event < L - horizon)
            if not sel.any():
                return
            k = gamma2_kernel(L, tau)
            idx = (anchors[sel] - t_event)[:, None] + tau_ax[None, :]
            target[sel] += amount * k[idx]

        for m in rec.meals:
            add(ra, m.t, float(p.f_bio[0]) * m.carbs * 1000.0 / (VG * w), m.tau)
        scale = INSULIN_BIOAVAILABILITY * 1e6 / (VI * w)
        su_mean_doses = []
        for d in rec.doses:
            kind = DRUG_CLASSES.get(d.drug)
            if kind in ("rapid", "regular", "nph"):
                add(iex, d.t, d.amount * scale, INSULIN_TAU[kind])
            elif kind == "premix":
                add(iex, d.t, PREMIX_REGULAR_FRACTION * d.amount * scale, INSULIN_TAU["regular"])
                add(iex, d.t, (1 - PREMIX_REGULAR_FRACTION) * d.amount * scale, INSULIN_TAU["nph"])
            elif kind == "sulfonylurea":
                rel = d.amount / SU_REFERENCE_DOSE.get(d.drug, 1.0)
                add(su_level, d.t, rel * 600.0 * rec.su_clearance, 150.0 * rec.su_clearance)
                su_mean_doses.append(rel * 600.0 * rec.su_clearance)
        su_mean = sum(su_mean_doses) / max(rec.T, 1)
        su = 1.1 * (su_level - su_mean) if su_mean_doses else np.zeros_like(su_level)
        m_beta = 1.0 + 0.35 * su_level / max(su_level.max(), 1e-9) if su_mean_doses else np.ones_like(su_level)

        decay = np.exp(-tau_ax / 15.0)[None, :]
        mets = REST_METS + (np.maximum(mets_now, REST_METS) - REST_METS)[:, None] * decay
        tod = (rec.tod0 + anchors[:, None] + tau_ax[None, :]) % 1440
        therapy = rec.therapy
        return Inputs(ra=ra, mets=mets, tod=tod, iex=iex, su=su, m_beta=m_beta,
                      m_inc=np.full((n, horizon), 1.7 if "dpp4" in therapy else 1.0),
                      renal_thr=np.full((n, horizon), 90.0 if "sglt2" in therapy else 180.0))

    def _states_at(self, anchors: np.ndarray) -> np.ndarray:
        if self.trace is None:
            self.sync()
        tr = self.trace
        k = np.clip(np.searchsorted(tr.t, anchors, side="right") - 1, 0, len(tr.t) - 1)
        return tr.mean[k]

    def _rollout(self, x0: np.ndarray, u: Inputs) -> tuple[np.ndarray, np.ndarray]:
        n, _ = u.shape
        pw = self.params.repeat(n)
        pw.SI = pw.SI * np.exp(x0[:, S_IDX])
        g, gi, _ = integrate(np.ascontiguousarray(x0[:, :N_PHYS]), pack_params(pw), pack_inputs(u))
        return gi, g

    def forecast(self, anchors: np.ndarray, horizon: int = HORIZON) -> Forecast:
        anchors = np.asarray(anchors, dtype=int)
        x0 = self._states_at(anchors)
        rec = self.record
        mets_now = rec.mets[np.clip(anchors, 0, len(rec.mets) - 1)]
        gi, g = self._rollout(x0, self._past_only_inputs(anchors, horizon, mets_now))
        return Forecast(anchors, gi, g)

    def what_if(self, minute: int, meals: list[Meal] | None = None, doses: list[Dose] | None = None,
                walk_minutes: int = 0, walk_mets: float = 3.5, walk_delay: int = 30,
                si_multiplier: float | None = None, horizon: int = 240) -> dict:
        """Counterfactual from the synced state at `minute`.

        `meals` and `doses` use minute offsets relative to `minute`. Returns baseline and
        scenario trajectories (CGM-equivalent) for side-by-side display.
        """
        anchors = np.array([minute, minute])
        x0 = self._states_at(anchors)
        if si_multiplier is not None:
            x0[1, S_IDX] = np.log(si_multiplier)
        u = self._past_only_inputs(anchors, horizon, self.record.mets[[min(minute, len(self.record.mets) - 1)] * 2])
        u.ra, u.iex, u.mets = u.ra.copy(), u.iex.copy(), u.mets.copy()  # inputs hold read-only broadcast views
        rec, p = self._record, self.params
        w, VG = float(p.weight[0]), float(p.VG[0])
        for m in meals or []:
            mm = Meal(m.t, m.carbs, m.protein, m.fat, m.fiber, m.gi, m.name, tau_scale=self.tau_scale)
            if 0 <= mm.t < horizon:
                k = gamma2_kernel(horizon - mm.t, mm.tau)
                u.ra[1, mm.t :] += float(p.f_bio[0]) * mm.carbs * 1000.0 / (VG * w) * k
        scale = INSULIN_BIOAVAILABILITY * 1e6 / (float(p.VI[0]) * w)
        for d in doses or []:
            kind = DRUG_CLASSES.get(d.drug)
            if kind in INSULIN_TAU and 0 <= d.t < horizon:
                u.iex[1, d.t :] += d.amount * scale * gamma2_kernel(horizon - d.t, INSULIN_TAU[kind])
        if walk_minutes > 0:
            a, b = walk_delay, min(walk_delay + walk_minutes, horizon)
            u.mets[1, a:b] = np.maximum(u.mets[1, a:b], walk_mets)
        gi, _ = self._rollout(x0, u)
        return {"minutes": np.arange(1, horizon + 1), "baseline": gi[0], "scenario": gi[1],
                "baseline_peak": float(gi[0].max()), "scenario_peak": float(gi[1].max()),
                "baseline_time_above_180": float((gi[0] > 180).mean()), "scenario_time_above_180": float((gi[1] > 180).mean()),
                "rec": rec.patient_id}
