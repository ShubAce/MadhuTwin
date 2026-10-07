"""Generate the synthetic Indian T2D cohort: EHR + wearable streams that are causally coupled.

    profile (EHR latent) -> lifestyle (meals, meds, sleep, activity) -> physiology simulation
        -> device models (CGM, HR, HRV, steps, sleep) -> labs derived from simulated glucose

Because labs are *measured from the simulated body* (HbA1c from mean glucose via the ADAG
relation, fasting glucose from pre-breakfast values), the static and dynamic streams are
consistent by construction, which is what lets a fusion model learn real cross-stream signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from twin.ehr.population import Profile, sample_population
from twin.ingest.schema import EVENT_COLUMNS, SERIES_COLUMNS, STATIC_COLUMNS, conform
from twin.physiology.inputs import build_inputs
from twin.physiology.model import Physiology, default_physiology, simulate
from twin.sensors import devices
from twin.sensors.lifestyle import Lifestyle, simulate_lifestyle

PHYS_KEYS = ("weight", "Gb", "Ib", "SI", "beta", "k_inc", "SG", "dawn")


@dataclass
class Cohort:
    profiles: list[Profile]
    static: pd.DataFrame
    series: pd.DataFrame
    events: pd.DataFrame
    truth: pd.DataFrame          # true glucose & hidden drivers on the 5-min grid (evaluation only)
    lab_history: pd.DataFrame    # longitudinal EHR labs (HbA1c, FPG, weight, BP)
    nights: pd.DataFrame


def physiology_for(profiles: list[Profile]) -> Physiology:
    kw = {k: np.array([p.physiology[k] for p in profiles]) for k in PHYS_KEYS}
    kw["renal_thr_base"] = np.array([90.0 if p.therapy_flags.get("sglt2") else 180.0 for p in profiles])
    return default_physiology(len(profiles), **kw)


def _simulate_chunk(profiles: list[Profile], lives: list[Lifestyle]) -> tuple[np.ndarray, np.ndarray]:
    p = physiology_for(profiles)
    T = lives[0].T
    therapy = [{k for k in ("dpp4", "sglt2") if pr.therapy_flags.get(k)} for pr in profiles]
    su_clear = [1.4 if (pr.labs_latent["egfr"] < 60 or pr.age >= 70) else 1.0 for pr in profiles]
    u = build_inputs(p, T, 0, [lf.meals for lf in lives], mets=np.stack([lf.mets for lf in lives]),
                     doses=[lf.doses for lf in lives], m_si=np.stack([lf.m_si for lf in lives]),
                     egp=np.stack([lf.egp for lf in lives]), therapy=therapy, su_clearance=su_clear)
    res = simulate(p, u)
    return res["G"], res["Gi"]


def _lab_history(rng: np.random.Generator, pr: Profile, a1c_now: float, fpg_now: float, ref: pd.Timestamp) -> list[dict]:
    """Quarterly labs over the previous two years, trending towards today's values."""
    rows = []
    trend = rng.normal(0.08 if pr.status == "t2d" else 0.03, 0.12)   # HbA1c change per quarter
    for q in range(8, 0, -1):
        date = ref - pd.DateOffset(months=3 * q) + pd.Timedelta(days=int(rng.integers(-10, 10)))
        a1c = float(np.clip(a1c_now - trend * q + rng.normal(0, 0.25), 4.3, 14))
        rows.append(dict(patient_id=pr.patient_id, date=date.normalize(), hba1c_pct=round(a1c, 1),
                         fpg_mgdl=round(float(np.clip(fpg_now - 28.7 * trend * q * 0.7 + rng.normal(0, 10), 65, 400))),
                         weight_kg=round(pr.weight_kg + rng.normal(0.25 * q * np.sign(trend), 1.0), 1),
                         sbp=round(pr.cardio["sbp"] + rng.normal(0, 7)), dbp=round(pr.cardio["dbp"] + rng.normal(0, 5))))
    return rows


def generate_cohort(n: int = 200, days: int = 14, seed: int = 7, start: str = "2026-09-01",
                    chunk: int = 100, festival: bool = True, verbose: bool = True) -> Cohort:
    rng = np.random.default_rng(seed)
    profiles = sample_population(n, seed)
    warm = 1  # one burn-in day so the first recorded day starts from a lived-in state
    t0 = pd.Timestamp(start) - pd.Timedelta(days=warm)
    total_days = days + warm
    static_rows, series_parts, truth_parts, event_rows, hist_rows, night_rows = [], [], [], [], [], []

    for c0 in range(0, n, chunk):
        prs = profiles[c0 : c0 + chunk]
        lives = [simulate_lifestyle(pr, total_days, t0, rng, festival_day=(int(rng.integers(3, total_days)) if festival and rng.random() < 0.35 else None))
                 for pr in prs]
        G, Gi = _simulate_chunk(prs, lives)
        keep = slice(warm * 1440, total_days * 1440)
        ts = pd.date_range(start, periods=days * 1440 // devices.GRID, freq=f"{devices.GRID}min")
        for i, (pr, lf) in enumerate(zip(prs, lives, strict=True)):
            g, gi = G[i, keep], Gi[i, keep]
            stage = lf.sleep_stage[keep]
            asleep = stage > 0
            sub = _slice_life(lf, keep)
            hr = devices.heart_rate(sub, g, pr.cardio["rhr"], rng)
            off = devices.wear_gaps(len(g), rng)
            hr[off] = np.nan
            steps = sub.steps.copy()
            steps[off] = np.nan
            mets = sub.mets.copy()
            mets[off] = np.nan
            hrv = devices.hrv_rmssd(sub, g, pr.cardio["hrv_rmssd"], rng)
            hrv[devices.five_minute(off.astype(float)) > 0] = np.nan
            cgm = devices.cgm_readings(gi, asleep, rng)

            logged = np.zeros((len(ts), 4))
            for m in lf.meal_log:
                k = (m["t"] - warm * 1440) // devices.GRID
                if 0 <= k < len(ts):
                    logged[k] += [m["carbs"], m["protein"], m["fat"], m["fiber"]]
                    event_rows.append(dict(patient_id=pr.patient_id, ts=ts[k], kind="meal", label=m["label"],
                                           carbs=m["carbs"], protein=m["protein"], fat=m["fat"], fiber=m["fiber"],
                                           kcal=m["kcal"], gi=m["gi"], amount=np.nan, food=m["food"]))
            insulin = np.zeros(len(ts))
            oad = np.zeros(len(ts))
            for e in lf.med_log:
                k = (e["t"] - warm * 1440) // devices.GRID
                if not (0 <= k < len(ts)) or not e["taken"]:
                    continue
                if e["drug"].startswith("insulin"):
                    insulin[k] += e["dose"]
                elif e["drug"] in {"metformin", "glimepiride", "gliclazide", "teneligliptin", "sitagliptin",
                                   "vildagliptin", "dapagliflozin", "empagliflozin", "voglibose"}:
                    oad[k] += 1
                event_rows.append(dict(patient_id=pr.patient_id, ts=ts[k], kind="insulin" if e["drug"].startswith("insulin") else "oad",
                                       label=e["drug"], amount=e["dose"]))
            true_carbs = np.zeros(len(ts))
            for m in lf.meals:
                k = (m.t - warm * 1440) // devices.GRID
                if 0 <= k < len(ts):
                    true_carbs[k] += m.carbs

            series_parts.append(pd.DataFrame({
                "patient_id": pr.patient_id, "ts": ts, "cgm": cgm, "hr": devices.five_minute(hr),
                "mets": devices.five_minute(mets), "steps": devices.five_minute(np.nan_to_num(steps), "sum"),
                "hrv_rmssd": hrv, "sleep_stage": stage[2 :: devices.GRID][: len(ts)].astype(float),
                "carbs": logged[:, 0], "protein": logged[:, 1], "fat": logged[:, 2], "fiber": logged[:, 3],
                "insulin_units": insulin, "oad_dose": oad,
            }))
            truth_parts.append(pd.DataFrame({
                "patient_id": pr.patient_id, "ts": ts, "glucose_true": g[2 :: devices.GRID][: len(ts)],
                "carbs_true": true_carbs, "stress": devices.five_minute(sub.stress), "illness": devices.five_minute(sub.illness),
                "m_si": devices.five_minute(sub.m_si),
            }))

            # labs measured from the simulated body
            mean_g = float(np.mean(g))
            a1c = float(np.clip((mean_g + 46.7) / 28.7 + rng.normal(0, 0.15), 4.3, 14.0))
            prebreakfast = [g[max(n_["wake"] - warm * 1440 + 30, 0)] for n_ in lf.nights[warm:] if 0 <= n_["wake"] - warm * 1440 + 30 < len(g)]
            fpg = float(np.mean(prebreakfast) + rng.normal(0, 4)) if prebreakfast else pr.physiology["Gb"]
            ins = float(pr.physiology["Ib"] * np.exp(rng.normal(0, 0.1)))
            ll = pr.labs_latent
            static_rows.append(dict(
                patient_id=pr.patient_id, source="synthetic", status=pr.status, age=pr.age, sex=pr.sex,
                height_cm=pr.height_cm, weight_kg=pr.weight_kg, bmi=pr.bmi, hba1c_pct=round(a1c, 1), fpg_mgdl=round(fpg),
                fasting_insulin_uU=round(ins, 1), homa_ir=round(fpg * ins / 405.0, 2), tg_mgdl=round(ll["tg_mgdl"]),
                chol_mgdl=round(ll["chol_mgdl"]), hdl_mgdl=round(ll["hdl_mgdl"]), ldl_mgdl=round(ll["ldl_mgdl"]),
                egfr=ll["egfr"], diabetes_years=pr.diabetes_years,
                on_metformin=float(pr.therapy_flags["metformin"]), on_sulfonylurea=float(pr.therapy_flags["sulfonylurea"]),
                on_dpp4=float(pr.therapy_flags["dpp4"]), on_sglt2=float(pr.therapy_flags["sglt2"]),
                on_insulin=float(pr.therapy_flags["insulin"]), hypertension=float("hypertension" in pr.conditions),
                dyslipidemia=float("dyslipidemia" in pr.conditions),
                tcf7l2_risk_alleles=float(pr.genetics["TCF7L2_rs7903146_T_alleles"]), prs_t2d=pr.genetics["prs_t2d_z"],
            ))
            hist_rows += _lab_history(rng, pr, a1c, fpg, pd.Timestamp(start))
            for nt in lf.nights[warm:]:
                night_rows.append(dict(patient_id=pr.patient_id, **nt))
        if verbose:
            print(f"  simulated {min(c0 + chunk, n)}/{n} patients")

    static = conform(pd.DataFrame(static_rows), STATIC_COLUMNS)
    series = conform(pd.concat(series_parts, ignore_index=True), SERIES_COLUMNS)
    ev = pd.DataFrame(event_rows)
    events = conform(ev, EVENT_COLUMNS).assign(food=ev.get("food"))
    return Cohort(profiles, static, series, events, pd.concat(truth_parts, ignore_index=True),
                  pd.DataFrame(hist_rows), pd.DataFrame(night_rows))


def _slice_life(lf: Lifestyle, keep: slice) -> Lifestyle:
    off = keep.start
    nights = [dict(n, onset=n["onset"] - off, wake=n["wake"] - off) for n in lf.nights]
    return Lifestyle(start=lf.start, T=keep.stop - keep.start, meals=[], meal_log=[], doses=[], med_log=[],
                     mets=lf.mets[keep], steps=lf.steps[keep], sleep_stage=lf.sleep_stage[keep],
                     stress=lf.stress[keep], illness=lf.illness[keep], m_si=lf.m_si[keep], egp=lf.egp[keep],
                     nights=nights)
