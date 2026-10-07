"""Day-by-day life simulation for a synthetic patient.

Produces the *true* exogenous drivers of physiology (what was eaten, injected, walked, slept)
and what the patient *logged* (imperfectly: some meals are never logged and carbohydrate
estimates are noisy). The twin only ever sees the logged view, as in real life.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from twin.ehr.population import Profile
from twin.physiology.foods import FOODS
from twin.physiology.inputs import Dose, Meal

DAY = 1440
AWAKE, LIGHT, DEEP, REM = 0, 1, 2, 3

BREAKFAST = {
    "south": ["idli_sambar", "plain_dosa", "masala_dosa", "upma", "pongal", "ragi_dosa", "oats"],
    "north": ["aloo_paratha", "puri_bhaji", "poha", "bread_omelette", "moong_chilla", "oats"],
    "west": ["poha", "upma", "bread_omelette", "moong_chilla", "oats", "aloo_paratha"],
    "east": ["puri_bhaji", "bread_omelette", "poha", "oats", "moong_chilla"],
}
LUNCH = {
    "south": ["rice_sambar", "rice_sambar", "curd_rice", "ragi_mudde", "jowar_roti_meal", "brown_rice_dal", "chicken_biryani"],
    "north": ["roti_sabzi", "roti_sabzi", "rajma_chawal", "rice_dal", "chole_bhature", "brown_rice_dal", "chicken_biryani"],
    "west": ["roti_sabzi", "rice_dal", "jowar_roti_meal", "pav_bhaji", "brown_rice_dal", "veg_biryani"],
    "east": ["fish_curry_rice", "fish_curry_rice", "rice_dal", "veg_biryani", "roti_sabzi", "brown_rice_dal"],
}
DINNER = {
    "south": ["rice_sambar", "curd_rice", "plain_dosa", "roti_sabzi", "ragi_mudde", "khichdi", "chicken_biryani"],
    "north": ["roti_sabzi", "paneer_roti", "chicken_curry_roti", "khichdi", "rice_dal", "rajma_chawal"],
    "west": ["roti_sabzi", "khichdi", "pav_bhaji", "rice_dal", "jowar_roti_meal", "paneer_roti"],
    "east": ["fish_curry_rice", "rice_dal", "roti_sabzi", "chicken_curry_roti", "khichdi"],
}
SNACKS = ["biscuits", "samosa", "medu_vada", "banana", "apple", "sprouts", "roasted_chana"]
SWEETS = ["gulab_jamun", "jalebi", "laddoo"]
NONVEG = {"chicken_biryani", "fish_curry_rice", "chicken_curry_roti", "bread_omelette"}
HEALTHY = {"low-gi", "millet"}


@dataclass
class Lifestyle:
    start: pd.Timestamp
    T: int
    meals: list[Meal]                       # what was actually eaten
    meal_log: list[dict]                    # what the patient logged
    doses: list[Dose]                       # insulin + sulfonylurea doses that drive physiology
    med_log: list[dict]                     # every medication intake (for the EHR/MAR view)
    mets: np.ndarray
    steps: np.ndarray
    sleep_stage: np.ndarray                 # per minute, 0 = awake
    stress: np.ndarray
    illness: np.ndarray
    m_si: np.ndarray
    egp: np.ndarray
    nights: list[dict] = field(default_factory=list)


def _pick(rng: np.random.Generator, options: list[str], profile: Profile) -> str:
    ls = profile.lifestyle
    opts = [o for o in options if not (ls["vegetarian"] and o in NONVEG)] or options
    healthy = [o for o in opts if HEALTHY & set(FOODS[o].tags)]
    if healthy and rng.random() < ls["diet_quality"] * (0.9 if profile.status == "t2d" else 0.6):
        return healthy[rng.integers(len(healthy))]
    other = [o for o in opts if o not in healthy] or opts
    return other[rng.integers(len(other))]


def _hypnogram(rng: np.random.Generator, minutes: int, osa: bool) -> np.ndarray:
    """Per-minute sleep stages with ~90 min cycles; deep sleep early, REM late."""
    stages = np.full(minutes, LIGHT, dtype=np.int8)
    t, k = 0, 0
    while t < minutes:
        cycle = int(rng.normal(92, 10))
        deep = int(cycle * max(0.32 * 0.62**k, 0.02))
        rem = int(cycle * min(0.12 + 0.07 * k, 0.4))
        light1 = int((cycle - deep - rem) * 0.55)
        seg = [LIGHT] * light1 + [DEEP] * deep + [LIGHT] * (cycle - deep - rem - light1) + [REM] * rem
        stages[t : t + len(seg)] = np.array(seg[: minutes - t], dtype=np.int8)
        t += cycle
        k += 1
    wake_p = 0.0035 if not osa else 0.012
    for s in np.flatnonzero(rng.random(minutes) < wake_p):
        stages[s : s + int(rng.integers(2, 9))] = AWAKE
    stages[: int(rng.integers(5, 20))] = LIGHT  # sleep latency handled by onset time; start light
    return stages


def simulate_lifestyle(profile: Profile, days: int, start: pd.Timestamp, rng: np.random.Generator,
                       festival_day: int | None = None, log_rate: float = 0.85) -> Lifestyle:
    ls = profile.lifestyle
    T = days * DAY
    offset = 600 if ls["shift_worker"] else 0
    mets = np.full(T, {"sedentary": 1.3, "moderate": 1.45, "active": 1.55}[ls["activity"]])
    steps = np.zeros(T)
    stage = np.zeros(T, dtype=np.int8)
    stress = np.zeros(T)
    illness = np.zeros(T)
    m_si = np.ones(T)
    egp = np.ones(T)
    meals: list[Meal] = []
    meal_log: list[dict] = []
    doses: list[Dose] = []
    med_log: list[dict] = []
    nights: list[dict] = []

    ill_start = int(rng.integers(1, max(days - 2, 2))) if rng.random() < 0.12 * days / 14 else None
    ill_days = set(range(ill_start, ill_start + int(rng.integers(2, 4)))) if ill_start is not None else set()

    def add_meal(t: int, key: str, portion: float, label: str) -> None:
        if not 0 <= t < T:
            return
        f = FOODS[key]
        m = Meal(t, f.carbs * portion, f.protein * portion, f.fat * portion, f.fiber * portion, f.gi, key,
                 tau_scale=profile.physiology["tau_scale"] * (1.25 if profile.therapy_flags.get("agi") else 1.0))
        meals.append(m)
        if rng.random() < log_rate:
            err = float(np.exp(rng.normal(0, 0.2)))  # people misjudge portions by ~20%
            meal_log.append(dict(t=t, food=key, label=label, carbs=m.carbs * err, protein=m.protein * err,
                                 fat=m.fat * err, fiber=m.fiber * err, gi=f.gi, kcal=f.kcal * portion * err))

    def bout(t0: int, minutes: int, met: float, cadence: float) -> None:
        t0, t1 = max(t0, 0), min(t0 + minutes, T)
        if t1 > t0:
            mets[t0:t1] = np.maximum(mets[t0:t1], met + rng.normal(0, 0.15, t1 - t0))
            steps[t0:t1] += np.maximum(rng.normal(cadence, 8, t1 - t0), 0)

    for d in range(days):
        base = d * DAY + offset
        sick = d in ill_days
        weekend = (start + pd.Timedelta(days=d)).dayofweek == 6
        wake = base + int(ls["wake_min"] + rng.normal(0, 18) + (50 if weekend else 0) + (40 if sick else 0))
        sleep_h = float(np.clip(rng.normal(ls["sleep_mean_h"], ls["sleep_var_h"]), 3.5, 9.5))
        if rng.random() < 0.08:
            sleep_h = max(3.5, sleep_h - 2.0)   # occasional bad night
        if sick:
            sleep_h = min(9.5, sleep_h + 0.8)
        onset = wake - int(sleep_h * 60)
        lo, hi = max(onset, 0), min(wake, T)
        if hi > lo:
            stage[lo:hi] = _hypnogram(rng, wake - onset, "osa" in profile.conditions)[lo - onset : hi - onset]
            mets[lo:hi] = 0.95
        deficit = max(ls["sleep_need_h"] - sleep_h, 0.0)
        nights.append(dict(date=(start + pd.Timedelta(days=d)).date().isoformat(), onset=onset, wake=wake,
                           hours=round(sleep_h, 2), deficit=round(deficit, 2)))
        day_end = min(base + DAY + int(ls["wake_min"]) - int(ls["sleep_mean_h"] * 60), T)
        si_day = 1.0 - profile.physiology["k_sleep"] * min(deficit / 3.0, 1.2)
        m_si[max(wake, 0) : max(day_end, 0)] *= si_day

        if sick:
            illness[max(base, 0) : min(base + DAY, T)] = 1.0
        if rng.random() < ls["stress_days_per_week"] / 7.0 and not sick:
            s0 = base + int(rng.normal(11 * 60, 60))
            stress[max(s0, 0) : min(s0 + int(rng.normal(300, 60)), T)] = 1.0

        # meals
        portion = lambda sick=sick: ls["portion"] * float(np.exp(rng.normal(0, 0.15))) * (0.7 if sick else 1.0)  # noqa: E731
        fest = festival_day is not None and d == festival_day
        if ls["chai_per_day"] >= 1:
            add_meal(wake + int(rng.normal(15, 5)), "chai_sugar" if ls["chai_sugar"] else "chai_nosugar", 1.0, "chai")
        skip_breakfast = rng.random() < 0.06
        t_bf = wake + int(rng.normal(95, 20))
        if not skip_breakfast:
            add_meal(t_bf, _pick(rng, BREAKFAST[profile.region], profile), portion(), "breakfast")
        t_lunch = base + int(rng.normal(13 * 60 + 20, 30))
        skip_lunch = rng.random() < 0.04
        if not skip_lunch:
            add_meal(t_lunch, _pick(rng, LUNCH[profile.region], profile), portion() * (1.25 if fest else 1.0), "lunch")
        if ls["chai_per_day"] >= 2:
            t_tea = base + int(rng.normal(17 * 60, 30))
            add_meal(t_tea, "chai_sugar" if ls["chai_sugar"] else "chai_nosugar", 1.0, "chai")
            if rng.random() < 0.55:
                add_meal(t_tea + 3, SNACKS[rng.integers(len(SNACKS))], 1.0, "snack")
        if ls["chai_per_day"] >= 3:
            add_meal(base + int(rng.normal(11 * 60, 30)), "chai_sugar" if ls["chai_sugar"] else "chai_nosugar", 1.0, "chai")
        t_dinner = base + int(ls["dinner_min"] + rng.normal(0, 25))
        add_meal(t_dinner, _pick(rng, DINNER[profile.region], profile), portion() * (1.3 if fest else 1.0), "dinner")
        n_sweets = rng.poisson(ls["sweets_per_week"] / 7.0) + (2 if fest else 0)
        for _ in range(n_sweets):
            add_meal((t_lunch if rng.random() < 0.4 else t_dinner) + int(rng.integers(20, 40)),
                     SWEETS[rng.integers(len(SWEETS))], 1.0, "sweet")

        # medications (dose times anchored to the day's actual meal times where clinically paired)
        for med in profile.meds:
            for tmin in med.times:
                t = base + tmin
                if med.drug == "insulin_premix_30_70":
                    t = (t_bf if tmin < 12 * 60 else t_dinner) - int(rng.integers(10, 30))
                taken = rng.random() < (min(profile.adherence + 0.08, 1.0) if med.drug.startswith("insulin") else profile.adherence)
                if not (0 <= t < T):
                    continue
                med_log.append(dict(t=t, drug=med.drug, dose=med.dose, taken=bool(taken)))
                if taken and (med.drug.startswith("insulin") or med.drug in ("glimepiride", "gliclazide")):
                    drug = "insulin_premix_30_70" if med.drug == "insulin_premix_30_70" else med.drug
                    if med.drug == "insulin_glargine":
                        continue  # basal insulin is part of the fasting equilibrium (Ib)
                    doses.append(Dose(t, drug, med.dose))

        # activity
        act_scale = 0.4 if sick else 1.0
        n_bouts = rng.poisson({"sedentary": 7, "moderate": 12, "active": 16}[ls["activity"]] * act_scale)
        for _ in range(n_bouts):
            bout(int(rng.integers(wake + 30, max(wake + 31, base + 21 * 60))), int(rng.integers(2, 10)), 2.8, 95)
        awake_lo, awake_hi = max(wake, 0), min(base + 22 * 60, T)
        if awake_hi > awake_lo:
            steps[awake_lo:awake_hi] += rng.poisson({"sedentary": 2.0, "moderate": 3.0, "active": 4.0}[ls["activity"]] * act_scale,
                                                    awake_hi - awake_lo)
        if ls["morning_walk"] and not sick and rng.random() < 0.8:
            bout(wake + int(rng.normal(25, 8)), int(rng.normal(32, 8)), 3.6, 108)
        if ls["post_dinner_walk"] and not sick and rng.random() < 0.7:
            bout(t_dinner + int(rng.normal(25, 5)), int(rng.normal(15, 4)), 3.1, 100)
        if ls["activity"] == "active" and not sick and rng.random() < 0.4:
            bout(base + int(rng.normal(18 * 60 + 30, 40)), int(rng.normal(40, 10)), float(rng.uniform(5, 7)), 140)

    # stress and illness: counter-regulatory hormones raise hepatic output and cut sensitivity
    egp *= 1.0 + 0.06 * stress + 0.15 * illness
    m_si *= (1.0 - 0.10 * stress) * (1.0 - 0.30 * illness)
    return Lifestyle(start=start, T=T, meals=sorted(meals, key=lambda m: m.t), meal_log=meal_log, doses=doses,
                     med_log=med_log, mets=mets, steps=steps, sleep_stage=stage, stress=stress, illness=illness,
                     m_si=m_si, egp=egp, nights=nights)
