"""In-memory store of virtual patients: demo clock, live state, AGP, insights and what-if."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from twin.ehr.codes import DRUGS
from twin.eval.metrics import excursion_onsets
from twin.physiology.foods import FOODS
from twin.physiology.inputs import DRUG_CLASSES, Dose, Meal, build_inputs
from twin.physiology.model import Physiology, default_physiology, initial_state, simulate
from twin.record import PatientRecord
from twin.twin import DigitalTwin

STEP = 5


def fidelity_rating(f: dict) -> str:
    """How far to trust a 24-h simulation, from the twin's replay of the last 24 h."""
    d_mean, d_tir = abs(f["mean_twin"] - f["mean_cgm"]), abs(f["tir_twin"] - f["tir_cgm"])
    if f["mae"] <= 20 and d_mean <= 15 and d_tir <= 15:
        return "good"
    return "fair" if f["mae"] <= 35 and d_mean <= 30 else "poor"
DAY_BINS = 1440 // STEP
DEMO_DAYS = 4


def _f(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else x


@dataclass
class Patient:
    bundle: dict
    _twin: DigitalTwin | None = field(default=None, repr=False)

    # -------------------------------------------------------------- basics
    @property
    def id(self) -> str:
        return self.bundle["id"]

    @cached_property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.bundle["start"])

    @cached_property
    def cgm(self) -> np.ndarray:
        return np.array([np.nan if v is None else v for v in self.bundle["series"]["cgm"]], dtype=float)

    @cached_property
    def anchors(self) -> np.ndarray:
        return np.array(self.bundle["predictions"]["anchors"], dtype=int)

    @cached_property
    def origin(self) -> int:
        """First local midnight after the twin's calibration period: demo day 1, 00:00."""
        b = int(self.bundle["calib_bins"])
        tod = ((self.start + pd.Timedelta(minutes=STEP * b)).hour * 60 + (self.start + pd.Timedelta(minutes=STEP * b)).minute) // STEP
        return b + (DAY_BINS - tod) % DAY_BINS

    def bin_at(self, clock_min: float) -> int:
        span = max(self.bundle["n_bins"] - self.origin - 1, 1)
        return int(self.origin + min(int(clock_min // STEP), span))

    def time_of(self, b: int) -> pd.Timestamp:
        return self.start + pd.Timedelta(minutes=STEP * int(b))

    def anchor_index(self, b: int) -> int | None:
        k = int(np.searchsorted(self.anchors, b, side="right") - 1)
        return k if k >= 0 else None

    # -------------------------------------------------------------- views
    def summary(self, clock: float) -> dict:
        b = self.bin_at(clock)
        k = self.anchor_index(b)
        pr = self.bundle["predictions"]
        lo = max(b - DAY_BINS, 0)
        c = self.cgm[lo : b + 1]
        c = c[np.isfinite(c)]
        alerts = [dict(a, time=self.time_of(a["bin"]).isoformat()) for a in self.bundle["alerts"] if self.origin - DAY_BINS <= a["bin"] <= b]
        last_val = self.cgm[: b + 1][np.isfinite(self.cgm[: b + 1])]
        return {
            "id": self.id, "story": self.bundle["story"], "source": self.bundle["source"], "display": self.bundle["display"],
            "status": self.bundle["ehr"]["status"], "now": self.time_of(b).isoformat(),
            "glucose": _f(float(last_val[-1])) if len(last_val) else None,
            "trend": self._trend(b),
            "risk_spike": _f(pr["spike"][k]) if k is not None else None,
            "risk_hypo": _f(pr["hypo"][k]) if k is not None else None,
            "tir_24h": float(np.mean((c >= 70) & (c <= 180)) * 100) if len(c) else None,
            "sparkline": [round(float(v)) for v in self.cgm[max(b - 72, 0) : b + 1] if np.isfinite(v)][::-1][::2][::-1],
            "last_alert": alerts[-1] if alerts else None,
        }

    def _trend(self, b: int) -> float | None:
        c = self.cgm
        prev = c[max(b - 3, 0)]
        return _f(float((c[b] - prev) / 15.0)) if np.isfinite(c[b]) and np.isfinite(prev) else None

    def state(self, clock: float, history_h: float = 24) -> dict:
        b = self.bin_at(clock)
        lo = max(b - int(history_h * 60 // STEP), 0)
        s = self.bundle["series"]
        k = self.anchor_index(b)
        pr = self.bundle["predictions"]
        fc = None
        if k is not None:
            a = int(self.anchors[k])
            fc = {"anchor_bin": a, "anchor_time": self.time_of(a).isoformat(),
                  "times": [self.time_of(a + h).isoformat() for h in range(1, 25)],
                  "q10": pr["q10"][k], "q50": pr["q50"][k], "q90": pr["q90"][k], "twin": pr["twin"][k],
                  "spike": pr["spike"][k], "hypo": pr["hypo"][k],
                  "why_spike": (pr.get("why_spike") or [None] * (k + 1))[k], "why_hypo": (pr.get("why_hypo") or [None] * (k + 1))[k]}
        events = [dict(e, time=self.time_of(e["bin"]).isoformat()) for e in self.bundle["events"] if lo <= e["bin"] <= b]
        alerts = [dict(a, time=self.time_of(a["bin"]).isoformat()) for a in self.bundle["alerts"] if self.origin - DAY_BINS <= a["bin"] <= b]
        return {
            "id": self.id, "now": self.time_of(b).isoformat(), "bin": b,
            "times": [self.time_of(i).isoformat() for i in range(lo, b + 1)],
            "series": {k2: v[lo : b + 1] for k2, v in s.items()},
            "events": events, "forecast": fc, "alerts": alerts[-20:], "si_today": self.si_today(b),
            "gate": pr.get("gate"), "track_record": self.track_record(clock),
        }

    def track_record(self, clock: float, days: int = 7) -> dict:
        """How this patient's past alerts turned out (only alerts whose 2-h outcome is already known).

        An alert is confirmed when a sustained excursion (>= 15 min beyond the threshold, as in the
        evaluation) is under way within the next 2 h; an excursion is caught when an alert of
        that kind was raised in the 2 h before it began."""
        b = self.bin_at(clock)
        # only out-of-sample alerts: after the calibration days the twin and TwinNet were personalised on
        lo = max(b - days * DAY_BINS, int(self.bundle["calib_bins"]))
        c = pd.Series(self.cgm).interpolate(limit=2, limit_area="inside").to_numpy()  # bridge 15-min sensor gaps
        out = {"days": days, "since": self.time_of(lo).isoformat(), "hours": round((b - lo) * STEP / 60, 1)}
        for kind, high in (("spike", True), ("hypo", False)):
            onsets = excursion_onsets(c, high=high)
            flag = np.nan_to_num(c, nan=0.0 if high else 999.0)
            flag = flag > 180 if high else flag < 70
            sustained = np.zeros(len(c), bool)
            for o in onsets:
                j = o
                while j < len(c) and flag[j]:
                    j += 1
                sustained[o:j] = True
            bins = [a["bin"] for a in self.bundle["alerts"] if a["kind"] == kind]
            known = [a for a in bins if lo <= a and a + 24 <= b]
            confirmed = [a for a in known if sustained[a + 1 : a + 25].any()]
            leads = [int(o - a) * STEP for a in confirmed for o in onsets[(onsets > a) & (onsets <= a + 24)][:1]]
            exc = onsets[(onsets >= lo) & (onsets + 3 <= b)]
            caught = [o for o in exc if any(o - 24 <= a < o for a in bins)]
            out[kind] = {"alerts": len(known), "confirmed": len(confirmed), "median_lead_min": int(np.median(leads)) if leads else None,
                         "excursions": int(len(exc)), "caught": len(caught)}
        return out

    def si_today(self, b: int) -> float | None:
        si = self.bundle["twin"].get("si_daily") or []
        d = int((self.time_of(b).normalize() - self.start.normalize()).days)
        return _f(si[d]) if 0 <= d < len(si) else None

    # -------------------------------------------------------------- AGP / clinical metrics
    def agp(self, clock: float, days: int = 14) -> dict:
        b = self.bin_at(clock)
        lo = max(b - days * DAY_BINS, 0)
        c = self.cgm[lo : b + 1]
        t = pd.date_range(self.time_of(lo), periods=len(c), freq="5min")
        ok = np.isfinite(c)
        v = c[ok]
        tod = (t.hour * 60 + t.minute)[ok] // 30
        prof = []
        for slot in range(48):
            x = v[tod == slot]
            prof.append({"slot": slot, "time": f"{slot // 2:02d}:{30 * (slot % 2):02d}",
                         **({f"p{p}": float(np.percentile(x, p)) for p in (5, 25, 50, 75, 95)} if len(x) >= 3 else {})})
        mean = float(v.mean()) if len(v) else float("nan")
        return {
            "days": round(len(c) / DAY_BINS, 1), "active_pct": float(ok.mean() * 100), "mean": mean,
            "gmi": 3.31 + 0.02392 * mean, "cv": float(v.std() / mean * 100) if len(v) else None,
            "tir": float(np.mean((v >= 70) & (v <= 180)) * 100), "tar1": float(np.mean((v > 180) & (v <= 250)) * 100),
            "tar2": float(np.mean(v > 250) * 100), "tbr1": float(np.mean((v >= 54) & (v < 70)) * 100),
            "tbr2": float(np.mean(v < 54) * 100), "profile": prof,
            "targets": {"tir": ">70%", "tar1": "<25%", "tar2": "<5%", "tbr1": "<4%", "tbr2": "<1%", "cv": "<=36%"},
        }

    # -------------------------------------------------------------- twin
    @property
    def params(self) -> Physiology:
        p = {k: v for k, v in self.bundle["twin"]["params"].items() if v is not None}
        if "Gb" not in p:
            from twin.physiology.calibrate import prior_physiology
            return prior_physiology(self.bundle["static"])
        # the bundle stores the fitted parameters; fixed physiological constants come from defaults
        return default_physiology(1, **p)

    @property
    def twin(self) -> DigitalTwin:
        if self._twin is None:
            st = self.bundle["static"]
            s = self.bundle["series"]
            series = pd.DataFrame({"ts": pd.date_range(self.start, periods=len(s["cgm"]), freq="5min"), "cgm": self.cgm,
                                   "mets": np.array([np.nan if x is None else x for x in s["mets"]], float)})
            ev = pd.DataFrame([{"ts": self.time_of(e["bin"]), "kind": e["kind"], "label": e["food"] or e["label"],
                                "carbs": e.get("carbs"), "protein": e.get("protein"), "fat": e.get("fat"),
                                "fiber": e.get("fiber"), "gi": e.get("gi"), "amount": e.get("amount")}
                               for e in self.bundle["events"]])
            if ev.empty:
                ev = pd.DataFrame(columns=["ts", "kind", "label", "carbs", "protein", "fat", "fiber", "gi", "amount"])
            for c in ("carbs", "protein", "fat", "fiber", "gi", "amount"):
                ev[c] = pd.to_numeric(ev[c], errors="coerce")
            rec = PatientRecord.from_frames(st, series, ev)
            tw = DigitalTwin(record=rec, params=self.params, tau_scale=float(self.bundle["twin"].get("tau_scale") or 1.0))
            tw.params.renal_thr_base[:] = rec.renal_thr_base
            tw.sync()
            self._twin = tw
        return self._twin

    def what_if(self, clock: float, meal: dict | None = None, walk: dict | None = None, insulin: dict | None = None,
                sleep_hours: float | None = None, illness: bool = False, horizon: int = 240) -> dict:
        b = self.bin_at(clock)
        minute = b * STEP + 2
        meals = []
        if meal and meal.get("food") in FOODS:
            f = FOODS[meal["food"]]
            k = float(meal.get("portion", 1.0))
            meals.append(Meal(int(meal.get("in_min", 0)), f.carbs * k, f.protein * k, f.fat * k, f.fiber * k, f.gi, f.key))
        doses = [Dose(int(insulin.get("in_min", 0)), insulin.get("drug", "insulin_rapid"), float(insulin["units"]))] if insulin else []
        si_mult = None
        if sleep_hours is not None or illness:
            si_mult = 1.0
            if sleep_hours is not None:
                si_mult *= 1.0 - 0.22 * min(max(7.0 - float(sleep_hours), 0) / 3.0, 1.2)
            if illness:
                si_mult *= 0.7
            si_mult *= float(np.exp(self.twin._states_at(np.array([minute]))[0, 6]))
        res = self.twin.what_if(minute, meals=meals, doses=doses, walk_minutes=int((walk or {}).get("minutes", 0)),
                                walk_delay=int((walk or {}).get("delay_min", 30)), si_multiplier=si_mult, horizon=horizon)
        base, scen = res["baseline"], res["scenario"]
        times = [(self.time_of(b) + pd.Timedelta(minutes=int(m))).isoformat() for m in res["minutes"][4::5]]
        return {
            "times": times, "baseline": np.round(base[4::5]).tolist(), "scenario": np.round(scen[4::5]).tolist(),
            "baseline_peak": round(float(base.max())), "scenario_peak": round(float(scen.max())),
            "baseline_min": round(float(base.min())), "scenario_min": round(float(scen.min())),
            "baseline_above_180_min": int((base > 180).sum()), "scenario_above_180_min": int((scen > 180).sum()),
            "baseline_below_70_min": int((base < 70).sum()), "scenario_below_70_min": int((scen < 70).sum()),
        }

    # ---------------------------------------------------------- therapy simulator
    def therapy_plan(self, clock: float) -> dict:
        """The starting plan: medication doses logged in the last 24 h, re-timed to the next 24 h."""
        b = self.bin_at(clock)
        now = self.time_of(b)
        _, doses, _ = self.twin.template_day(b * STEP + 2)
        rec = self.twin.record
        names = {"insulin_rapid": "Rapid-acting insulin", "insulin_regular": "Regular insulin", "insulin_nph": "NPH insulin"}
        rows = [{"drug": d.drug, "display": DRUGS[d.drug][1] if d.drug in DRUGS else names.get(d.drug, d.drug), "kind": DRUG_CLASSES[d.drug],
                 "offset_min": int(d.t), "time": (now + pd.Timedelta(minutes=int(d.t))).strftime("%H:%M"),
                 "amount": float(d.amount), "unit": "IU" if DRUG_CLASSES[d.drug] in ("rapid", "regular", "nph", "premix") else "mg"}
                for d in sorted(doses, key=lambda d: d.t)]
        egfr = self.bundle["static"].get("egfr")
        return {"start": now.isoformat(), "doses": rows, "dpp4": "dpp4" in rec.therapy, "sglt2": "sglt2" in rec.therapy,
                "egfr": egfr, "reduced_su_clearance": rec.su_clearance > 1.0,
                "meals_replayed": len(self.twin.template_day(b * STEP + 2)[0])}

    def therapy_whatif(self, clock: float, doses: list[dict] | None = None, dpp4: bool | None = None,
                       sglt2: bool | None = None, pattern: str = "yesterday") -> dict:
        """Compare the usual regimen with a modified one over the next 24 h on this patient's twin,
        plus a dose-response sweep of the usual insulin (or sulfonylurea) doses.

        `pattern` stress-tests the day: "yesterday" (replayed as logged), "skip_lunch",
        "late_dinner" (2 h later) or "big_dinner" (+50% carbohydrate, e.g. a festival).
        """
        b = self.bin_at(clock)
        minute = b * STEP + 2
        tw = self.twin
        meals, usual, mets = tw.template_day(minute)
        tod0 = self.time_of(b).hour * 60 + self.time_of(b).minute
        at = lambda m: (tod0 + m.t) % 1440  # noqa: E731
        if pattern == "skip_lunch":
            meals = [m for m in meals if not 11 * 60 <= at(m) < 15 * 60 + 30]
        elif pattern == "late_dinner":
            meals = [replace(m, t=min(m.t + 120, 1439)) if 18 * 60 <= at(m) < 22 * 60 + 30 else m for m in meals]
        elif pattern == "big_dinner":
            meals = [replace(m, carbs=m.carbs * 1.5) if 18 * 60 <= at(m) < 22 * 60 + 30 else m for m in meals]
        therapy = set(tw.record.therapy)
        new_doses = usual if doses is None else [Dose(int(d["offset_min"]), str(d["drug"]), float(d["amount"]))
                                                 for d in doses if str(d.get("drug")) in DRUG_CLASSES]
        new_therapy = {c for c, on in (("dpp4", "dpp4" in therapy if dpp4 is None else dpp4),
                                       ("sglt2", "sglt2" in therapy if sglt2 is None else sglt2)) if on}
        insulin = [d for d in usual if DRUG_CLASSES[d.drug] in ("rapid", "regular", "nph", "premix")]
        group, sweep_name = (insulin, "usual insulin doses") if insulin else (
            [d for d in usual if DRUG_CLASSES[d.drug] == "sulfonylurea"], "usual sulfonylurea dose")
        scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2] if group else []
        sweep = [[Dose(d.t, d.drug, d.amount * s) if d in group else d for d in usual] for s in scales]
        plans = [{"doses": usual, "therapy": therapy}, {"doses": new_doses, "therapy": new_therapy}]
        plans += [{"doses": s, "therapy": therapy} for s in sweep]
        gi = tw.therapy_day(minute, plans, meals=meals, mets=mets)
        tod = (self.time_of(b).hour * 60 + self.time_of(b).minute + np.arange(gi.shape[1])) % 1440

        def stats(g: np.ndarray) -> dict:
            night = tod < 360
            return {"mean": round(float(g.mean())), "tir_pct": round(float(np.mean((g >= 70) & (g <= 180)) * 100), 1),
                    "below_70_min": int((g < 70).sum()), "below_54_min": int((g < 54).sum()),
                    "night_below_70_min": int(((g < 70) & night).sum()), "above_180_pct": round(float(np.mean(g > 180) * 100), 1),
                    "above_250_min": int((g > 250).sum()), "min": round(float(g.min())), "max": round(float(g.max())),
                    "min_time": (self.time_of(b) + pd.Timedelta(minutes=int(np.argmin(g)))).strftime("%H:%M")}

        times = [(self.time_of(b) + pd.Timedelta(minutes=int(m))).isoformat() for m in range(4, gi.shape[1], 5)]
        return {"times": times, "usual": np.round(gi[0, 4::5]).tolist(), "scenario": np.round(gi[1, 4::5]).tolist(),
                "usual_stats": stats(gi[0]), "scenario_stats": stats(gi[1]),
                "dose_response": {"what": sweep_name, "rows": [{"scale": s, **stats(gi[2 + i])} for i, s in enumerate(scales)]},
                "pattern": pattern, "fidelity": self.replay_fidelity(clock),
                "assumptions": "Meals, activity and dose times of the last 24 h are replayed; insulin sensitivity stays at "
                               "today's synced value. Educational decision support, not a dosing calculator."}

    def replay_fidelity(self, clock: float) -> dict | None:
        """How well the twin reproduces the last 24 h when it replays them open-loop from the state it
        had 24 h ago with the logged meals, doses and activity: the trust check for the simulator."""
        b = self.bin_at(clock)
        minute = b * STEP + 2
        if minute < 2 * 1440:
            return None
        tw = self.twin
        meals, doses, mets = tw.template_day(minute)
        sim = tw.therapy_day(minute - 1440, [{"doses": doses, "therapy": set(tw.record.therapy)}], meals=meals, mets=mets)[0][::STEP]
        obs = self.cgm[b - DAY_BINS : b]
        ok = np.isfinite(obs)
        if ok.sum() < 48:  # at least 12 h of readings, also for 15-minute sensors
            return None
        s, o = sim[: len(obs)][ok], obs[ok]
        tir = lambda x: round(float(np.mean((x >= 70) & (x <= 180)) * 100), 1)  # noqa: E731
        out = {"mean_twin": round(float(s.mean())), "mean_cgm": round(float(o.mean())), "tir_twin": tir(s), "tir_cgm": tir(o),
               "below_70_twin_min": int((s < 70).sum() * STEP), "below_70_cgm_min": int((o < 70).sum() * STEP),
               "mae": round(float(np.abs(s - o).mean()), 1)}
        return out | {"rating": fidelity_rating(out)}

    def meal_ranking(self, region_first: bool = True) -> list[dict]:
        """Personal glycaemic response of this twin to common Indian meals (from fasting set-point)."""
        p = self.params
        tau = float(self.bundle["twin"].get("tau_scale") or 1.0)
        keys = [k for k, f in FOODS.items() if ("lunch" in f.tags or "dinner" in f.tags or "breakfast" in f.tags)]
        n = len(keys)
        pw = p.repeat(n)
        T = 240
        meals = [[Meal(0, FOODS[k].carbs, FOODS[k].protein, FOODS[k].fat, FOODS[k].fiber, FOODS[k].gi, k, tau_scale=tau)] for k in keys]
        u = build_inputs(pw, T, 13 * 60, meals)
        g = simulate(pw, u, x0=initial_state(pw))["Gi"]
        base = float(p.Gb[0])
        out = [{"food": k, "name": FOODS[k].name, "carbs": FOODS[k].carbs, "gi": FOODS[k].gi, "tags": list(FOODS[k].tags),
                "rise": round(float(g[i].max() - base)), "peak_min": int(np.argmax(g[i]) + 1),
                "above_180_min": int((g[i] > 180).sum())} for i, k in enumerate(keys)]
        return sorted(out, key=lambda r: r["rise"])

    def walk_benefit(self) -> dict:
        p = self.params.repeat(2)
        tau = float(self.bundle["twin"].get("tau_scale") or 1.0)
        f = FOODS["rice_dal"]
        T = 240
        mets = np.ones((2, T))
        mets[1, 20:35] = 3.3
        u = build_inputs(p, T, 21 * 60, [[Meal(0, f.carbs, f.protein, f.fat, f.fiber, f.gi, f.key, tau_scale=tau)]] * 2, mets=mets)
        g = simulate(p, u)["Gi"]
        return {"meal": f.name, "peak_without": round(float(g[0].max())), "peak_with_walk": round(float(g[1].max())),
                "reduction": round(float(g[0].max() - g[1].max()))}

    def insights(self, clock: float) -> list[dict]:
        b = self.bin_at(clock)
        out = []
        si = [x for x in (self.bundle["twin"].get("si_daily") or [])]
        today = self.si_today(b)
        if today is not None and today < 0.8:
            out.append({"kind": "warning", "title": "Insulin resistance spike",
                        "text": f"The synced twin estimates insulin sensitivity {100 * (1 - today):.0f}% below this patient's baseline today. "
                                "Typical causes: infection, stress, poor sleep, steroid use. Consider checking symptoms and ketones if very high glucose."})
        elif si and today is not None:
            out.append({"kind": "info", "title": "Insulin sensitivity stable", "text": f"Twin estimate today: {today:.2f}x personal baseline."})
        params = self.bundle["twin"]["params"]
        if params.get("dawn") and params["dawn"] > 0.12:
            out.append({"kind": "info", "title": "Dawn phenomenon",
                        "text": f"The twin's fitted dawn effect raises the fasting set-point by about {params['dawn'] * 100:.0f}% between 4 and 8 am. "
                                "Pre-breakfast readings will overstate overnight control."})
        rank = self.meal_ranking()
        if rank:
            hi, lo = rank[-1], rank[0]
            out.append({"kind": "food", "title": "Personal meal response",
                        "text": f"For this twin, {hi['name']} raises glucose most (+{hi['rise']} mg/dL); {lo['name']} least (+{lo['rise']} mg/dL)."})
        wb = self.walk_benefit()
        if wb["reduction"] >= 5:
            out.append({"kind": "activity", "title": "Post-dinner walk",
                        "text": f"A 15-minute walk 20 min after {wb['meal'].lower()} lowers this twin's peak by ~{wb['reduction']} mg/dL "
                                f"({wb['peak_without']} -> {wb['peak_with_walk']})."})
        ag = self.agp(clock, days=7)
        if ag["tbr1"] + ag["tbr2"] >= 4:
            out.append({"kind": "warning", "title": "Time below range above target",
                        "text": f"{ag['tbr1'] + ag['tbr2']:.1f}% of readings <70 mg/dL in the last 7 days (target <4%). Review sulfonylurea/insulin dosing and meal timing."})
        if ag["tar2"] >= 5:
            out.append({"kind": "warning", "title": "Very high glucose",
                        "text": f"{ag['tar2']:.1f}% of readings >250 mg/dL in the last 7 days (target <5%)."})
        meds = [e for e in self.bundle["events"] if e["kind"] in ("oad", "insulin") and e["bin"] <= b]
        presc = self.bundle["ehr"].get("medications") or []
        if presc and meds and self.bundle["source"] == "synthetic":
            days = max((b - 0) / DAY_BINS, 1)
            expected = sum(len(m.get("times", [])) for m in presc if m["class"] in ("biguanide", "sulfonylurea", "dpp4", "sglt2", "insulin", "agi")) * days
            taken = len(meds)
            if expected > 0:
                pct = min(100.0, 100 * taken / expected)
                out.append({"kind": "info" if pct >= 90 else "warning", "title": "Medication adherence",
                            "text": f"{pct:.0f}% of scheduled antidiabetic doses logged as taken."})
        return out


class Store:
    def __init__(self, demo_dir: Path, results_dir: Path):
        self.demo_dir = demo_dir
        self.results_dir = results_dir
        index = json.load(open(demo_dir / "index.json"))
        self.patients: dict[str, Patient] = {}
        for item in index:
            self.patients[item["id"]] = Patient(json.load(open(demo_dir / f"{item['id']}.json")))
        self.order = [i["id"] for i in index]
        self.default_clock = 6 * 60 + 30  # demo day 1, 06:30

    def get(self, pid: str) -> Patient:
        return self.patients[pid]

    @property
    def max_clock(self) -> int:
        return DEMO_DAYS * 1440

    def evidence(self) -> dict:
        files = {"synthetic": "synthetic/results.json", "real": "real/results.json", "cgm_light": "cgm_light.json", "extra": "synthetic/extra.json",
                 "fidelity": "fidelity.json", "robustness": "robustness.json", "bench": "bench.json"}
        # results files may hold NaN (e.g. no lead time when nothing was caught); JSON has no NaN, so send null
        return {k: json.load(open(self.results_dir / f), parse_constant=lambda c: None)
                for k, f in files.items() if (self.results_dir / f).exists()}
