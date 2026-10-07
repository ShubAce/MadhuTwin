"""Sample a synthetic, India-calibrated population of patient profiles.

Each profile carries everything the twin needs: demographics, diagnoses, India-typical
prescriptions, genetics, lifestyle habits and latent physiology. Latent physiology is drawn
from relationships *measured on real people*: the regressions of personal-twin parameters on
HOMA-IR and HbA1c obtained by fitting the mechanistic model to the 45 CGMacros participants
(see scripts/fit_twins.py). Epidemiological rates follow ICMR-INDIAB and Indian clinical
practice; all identities are fictitious.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .names import CITIES, FIRST_NAMES, LANGUAGES, SURNAMES

REGIONS = {"south": 0.30, "north": 0.35, "west": 0.20, "east": 0.15}
STATUS_MIX = {"t2d": 0.65, "prediabetes": 0.25, "normal": 0.10}

# Regressions estimated on fitted CGMacros twins (log-scale where noted).
LOGSI_INTERCEPT, LOGSI_HOMA = -7.198, -0.619          # log SI ~ log HOMA-IR, resid sd 0.51
LOGBETA_INTERCEPT, LOGBETA_A1C = 1.007, -0.573        # log beta ~ HbA1c, resid sd 0.77
DAWN_INTERCEPT, DAWN_A1C = -0.208, 0.0495              # dawn ~ HbA1c, resid sd 0.087
LOGTAU_INTERCEPT, LOGTAU_A1C = 0.990, -0.163           # log tau_scale ~ HbA1c, resid sd 0.32


@dataclass
class Medication:
    drug: str
    dose: float
    times: list[int]            # minute of day for each daily dose
    started_years_ago: float


@dataclass
class Profile:
    patient_id: str
    name: str
    sex: str
    age: int
    region: str
    city: str
    language: str
    status: str
    diabetes_years: float
    height_cm: float
    weight_kg: float
    bmi: float
    waist_cm: float
    genetics: dict
    family_history: list[str]
    conditions: dict[str, float]          # condition key -> years since onset
    meds: list[Medication]
    adherence: float
    lifestyle: dict
    physiology: dict
    cardio: dict
    labs_latent: dict
    therapy_flags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Profile:
        return cls(**{**d, "meds": [Medication(**m) for m in d["meds"]]})


def egfr_ckd_epi_2021(creatinine_mgdl: float, age: float, sex: str) -> float:
    k, a = (0.7, -0.241) if sex == "F" else (0.9, -0.302)
    r = creatinine_mgdl / k
    return 142.0 * min(r, 1.0) ** a * max(r, 1.0) ** -1.200 * 0.9938**age * (1.012 if sex == "F" else 1.0)


def _choice(rng: np.random.Generator, d: dict):
    keys = list(d)
    return keys[rng.choice(len(keys), p=np.array(list(d.values())) / sum(d.values()))]


def _prescribe(rng: np.random.Generator, status: str, years: float, a1c: float, weight: float,
               egfr: float, conditions: dict) -> list[Medication]:
    meds: list[Medication] = []
    started = lambda: float(np.round(rng.uniform(0.2, max(years, 0.3)), 1))  # noqa: E731
    if status == "t2d":
        if egfr >= 30 and rng.random() < 0.88:
            meds.append(Medication("metformin", float(rng.choice([500, 1000])), [8 * 60 + 30, 21 * 60], started()))
        if rng.random() < (0.30 + 0.06 * min(years, 8)):
            if rng.random() < 0.55:
                meds.append(Medication("glimepiride", float(rng.choice([1, 2, 3, 4])), [8 * 60], started()))
            else:
                meds.append(Medication("gliclazide", float(rng.choice([40, 80, 120])), [8 * 60], started()))
        if rng.random() < 0.42:
            drug = _choice(rng, {"teneligliptin": 0.55, "sitagliptin": 0.25, "vildagliptin": 0.20})
            dose, times = {"teneligliptin": (20, [8 * 60]), "sitagliptin": (100, [8 * 60]),
                           "vildagliptin": (50, [8 * 60, 20 * 60])}[drug]
            meds.append(Medication(drug, dose, times, started()))
        if egfr >= 25 and rng.random() < 0.25 + 0.15 * ("cad" in conditions or "ckd" in conditions):
            drug = _choice(rng, {"dapagliflozin": 0.6, "empagliflozin": 0.4})
            meds.append(Medication(drug, 10.0, [8 * 60], started()))
        if rng.random() < 0.08:
            meds.append(Medication("voglibose", 0.3, [8 * 60, 13 * 60, 20 * 60 + 30], started()))
        if (years > 8 and a1c > 8.0 and rng.random() < 0.55) or rng.random() < 0.06:
            total = float(np.round(weight * rng.uniform(0.35, 0.6)))
            if rng.random() < 0.7:  # premixed 30/70 twice daily: most common insulin regimen in India
                am = float(np.round(total * 0.6))
                meds.append(Medication("insulin_premix_30_70", am, [8 * 60 - 15], started()))
                meds.append(Medication("insulin_premix_30_70", total - am, [20 * 60 + 15], started()))
            else:
                meds.append(Medication("insulin_glargine", float(np.round(total * 0.5)), [22 * 60], started()))
    if "dyslipidemia" in conditions and rng.random() < 0.75:
        meds.append(Medication(_choice(rng, {"atorvastatin": 0.65, "rosuvastatin": 0.35}),
                               float(rng.choice([10, 20, 40])), [21 * 60 + 30], started()))
    if "hypertension" in conditions:
        meds.append(Medication("telmisartan", float(rng.choice([20, 40, 80])), [8 * 60], started()))
        if rng.random() < 0.4:
            meds.append(Medication("amlodipine", float(rng.choice([2.5, 5, 10])), [8 * 60], started()))
    if "cad" in conditions:
        meds.append(Medication("aspirin", 75.0, [13 * 60 + 30], started()))
    if "hypothyroidism" in conditions:
        meds.append(Medication("levothyroxine", float(rng.choice([25, 50, 75, 100])), [6 * 60 + 30], started()))
    return meds


def sample_profile(rng: np.random.Generator, idx: int, status: str | None = None) -> Profile:
    region = _choice(rng, REGIONS)
    sex = "F" if rng.random() < 0.5 else "M"
    status = status or _choice(rng, STATUS_MIX)
    age = int(np.clip(rng.normal({"t2d": 54, "prediabetes": 46, "normal": 38}[status], 10), 25, 80))

    # --- genetics (South Asian allele frequencies; risk alleles enriched in T2D) ---
    p_t = {"t2d": 0.36, "prediabetes": 0.32, "normal": 0.28}[status]
    tcf7l2 = int(rng.binomial(2, p_t))
    prs = float(np.round(rng.normal({"t2d": 0.45, "prediabetes": 0.2, "normal": -0.1}[status], 1.0), 2))
    fam = []
    if rng.random() < {"t2d": 0.62, "prediabetes": 0.48, "normal": 0.3}[status] + 0.05 * prs:
        fam.append(_choice(rng, {"father": 0.4, "mother": 0.4, "both parents": 0.2}) + ": type 2 diabetes")
    if rng.random() < 0.25:
        fam.append(_choice(rng, {"father": 0.6, "mother": 0.4}) + ": coronary artery disease")
    if rng.random() < 0.3:
        fam.append(_choice(rng, {"father": 0.5, "mother": 0.5}) + ": hypertension")

    # --- anthropometry: South Asian phenotype, overweight at BMI >= 23 ---
    height = float(np.round(rng.normal(168 if sex == "M" else 155, 6.5), 1))
    bmi = float(np.clip(rng.normal({"t2d": 26.8, "prediabetes": 25.6, "normal": 23.2}[status] + 0.25 * prs, 3.6), 17.5, 44))
    weight = float(np.round(bmi * (height / 100) ** 2, 1))
    waist = float(np.round(bmi * 3.3 + (8 if sex == "M" else 4) + rng.normal(0, 4), 1))

    years = float(np.round(np.clip(rng.gamma(2.0, 4.0), 0.3, age - 22), 1)) if status == "t2d" else 0.0

    # --- glycaemic control and latent physiology ---
    if status == "t2d":
        a1c_target = float(np.clip(rng.normal(7.8 + 0.04 * min(years, 15), 1.2), 6.2, 12.0))
        fpg = float(np.clip(rng.normal(28.7 * a1c_target - 46.7 - 35, 18), 100, 300))
    elif status == "prediabetes":
        a1c_target = float(np.clip(rng.normal(6.0, 0.22), 5.7, 6.4))
        fpg = float(np.clip(rng.normal(104, 6), 95, 122))
    else:
        a1c_target = float(np.clip(rng.normal(5.2, 0.25), 4.5, 5.6))
        fpg = float(np.clip(rng.normal(89, 6), 72, 99))
    log_homa = np.log({"t2d": 3.6, "prediabetes": 2.8, "normal": 1.5}[status]) + 0.05 * (bmi - 25) + rng.normal(0, 0.35)
    homa = float(np.exp(log_homa))
    ins = float(np.clip(homa * 405.0 / fpg, 2.0, 60.0))
    SI = float(np.exp(LOGSI_INTERCEPT + LOGSI_HOMA * np.log(homa) + rng.normal(0, 0.35)))
    beta = float(np.clip(np.exp(LOGBETA_INTERCEPT + LOGBETA_A1C * a1c_target + rng.normal(0, 0.45)) * 0.9**tcf7l2, 0.008, 0.5))
    dawn = float(np.clip(DAWN_INTERCEPT + DAWN_A1C * a1c_target + rng.normal(0, 0.06), 0.0, 0.35))
    tau_scale = float(np.clip(np.exp(LOGTAU_INTERCEPT + LOGTAU_A1C * a1c_target + rng.normal(0, 0.2)), 0.55, 1.8))
    physiology = dict(
        weight=weight, Gb=fpg, Ib=ins, SI=SI, beta=beta, k_inc=0.35 + 4.0 * beta,
        SG={"t2d": 0.012, "prediabetes": 0.015, "normal": 0.018}[status] * float(np.exp(rng.normal(0, 0.15))),
        dawn=dawn, tau_scale=tau_scale, k_sleep=float(np.clip(rng.normal(0.22, 0.06), 0.05, 0.4)),
    )

    # --- comorbidities (prevalence among Indian T2D clinic populations) ---
    cond: dict[str, float] = {}
    if status in ("t2d", "prediabetes"):
        cond[status] = years if status == "t2d" else float(np.round(rng.uniform(0.2, 3), 1))
    p = lambda base: rng.random() < base  # noqa: E731
    if p({"t2d": 0.55, "prediabetes": 0.35, "normal": 0.15}[status] + 0.01 * max(age - 50, 0)):
        cond["hypertension"] = float(np.round(rng.uniform(0.5, 12), 1))
    if p({"t2d": 0.70, "prediabetes": 0.5, "normal": 0.25}[status]):
        cond["dyslipidemia"] = float(np.round(rng.uniform(0.5, 10), 1))
    if bmi >= 27.5:
        cond["obesity"] = float(np.round(rng.uniform(1, 10), 1))
    if p({"t2d": 0.40, "prediabetes": 0.3, "normal": 0.12}[status] + 0.02 * max(bmi - 25, 0)):
        cond["nafld"] = float(np.round(rng.uniform(0.5, 6), 1))
    creat = float(np.clip(rng.normal(0.85 if sex == "F" else 1.0, 0.15) + (0.02 * years if status == "t2d" else 0), 0.5, 3.5))
    egfr = egfr_ckd_epi_2021(creat, age, sex)
    if egfr < 60:
        cond["ckd"] = float(np.round(rng.uniform(0.5, 5), 1))
        if status == "t2d":
            cond["diabetic_nephropathy"] = cond["ckd"]
    if status == "t2d" and p(0.08 + 0.02 * years):
        cond["retinopathy"] = float(np.round(rng.uniform(0.3, max(years - 3, 0.5)), 1))
    if status == "t2d" and p(0.10 + 0.015 * years):
        cond["neuropathy"] = float(np.round(rng.uniform(0.3, max(years - 2, 0.5)), 1))
    if p({"t2d": 0.09, "prediabetes": 0.04, "normal": 0.02}[status] + 0.004 * max(age - 50, 0)):
        cond["cad"] = float(np.round(rng.uniform(0.5, 8), 1))
    if p(0.11 if sex == "F" else 0.03):
        cond["hypothyroidism"] = float(np.round(rng.uniform(1, 12), 1))
    if bmi >= 29 and p(0.25):
        cond["osa"] = float(np.round(rng.uniform(0.5, 5), 1))

    meds = _prescribe(rng, status, years, a1c_target, weight, egfr, cond)
    classes = {m.drug for m in meds}
    therapy = {
        "metformin": "metformin" in classes,
        "sulfonylurea": bool(classes & {"glimepiride", "gliclazide"}),
        "dpp4": bool(classes & {"teneligliptin", "sitagliptin", "vildagliptin"}),
        "sglt2": bool(classes & {"dapagliflozin", "empagliflozin"}),
        "agi": "voglibose" in classes,
        "insulin": bool(classes & {"insulin_premix_30_70", "insulin_glargine"}),
    }

    # --- lifestyle ---
    late_diner = rng.random() < {"south": 0.45, "north": 0.6, "west": 0.6, "east": 0.55}[region]
    lifestyle = dict(
        wake_min=int(np.clip(rng.normal(6 * 60 + 20, 35), 4 * 60 + 45, 8 * 60 + 30)),
        sleep_need_h=float(np.clip(rng.normal(7.0, 0.6), 5.5, 8.5)),
        sleep_mean_h=float(np.clip(rng.normal(6.6 - 0.3 * ("osa" in cond), 0.7), 4.8, 8.3)),
        sleep_var_h=float(np.clip(rng.normal(0.75, 0.25), 0.3, 1.6)),
        shift_worker=bool(rng.random() < 0.06),
        activity="sedentary" if rng.random() < 0.45 else ("moderate" if rng.random() < 0.7 else "active"),
        morning_walk=bool(rng.random() < (0.45 if status == "t2d" else 0.25)),
        post_dinner_walk=bool(rng.random() < (0.25 if region == "south" else 0.15)),
        vegetarian=bool(rng.random() < {"south": 0.35, "north": 0.45, "west": 0.55, "east": 0.12}[region]),
        dinner_min=int(np.clip(rng.normal(21 * 60 + (25 if late_diner else -20), 25), 19 * 60, 22 * 60 + 45)),
        chai_per_day=int(rng.choice([0, 1, 2, 3], p=[0.12, 0.33, 0.40, 0.15])),
        chai_sugar=bool(rng.random() < (0.55 if status == "t2d" else 0.8)),
        sweets_per_week=float(np.clip(rng.gamma(1.5, 1.2), 0, 8)),
        portion=float(np.clip(rng.normal(1.0 + 0.012 * (bmi - 24), 0.12), 0.7, 1.5)),
        stress_days_per_week=float(np.clip(rng.gamma(1.4, 0.7), 0, 4)),
        diet_quality=float(np.clip(rng.normal(0.4 if status == "t2d" else 0.3, 0.2), 0, 1)),  # 1 = follows low-GI advice
    )

    fitness = {"sedentary": 0.0, "moderate": 0.5, "active": 1.0}[lifestyle["activity"]]
    cardio = dict(
        rhr=float(np.clip(rng.normal(74 - 6 * fitness + (3 if status == "t2d" else 0) + 0.2 * (bmi - 25), 6), 52, 98)),
        hrv_rmssd=float(np.clip(np.exp(rng.normal(np.log(40) - 0.018 * (age - 30) + 0.15 * fitness
                                                  - (0.012 * years if status == "t2d" else 0)
                                                  - (0.15 if "neuropathy" in cond else 0), 0.3)), 8, 90)),
        sbp=float(np.round(np.clip(rng.normal(124 + 0.4 * (age - 40) + 10 * ("hypertension" in cond), 11), 100, 185))),
        dbp=float(np.round(np.clip(rng.normal(78 + 0.15 * (age - 40) + 6 * ("hypertension" in cond), 8), 60, 115))),
    )

    tg = float(np.clip(np.exp(rng.normal(np.log({"t2d": 165, "prediabetes": 145, "normal": 115}[status]), 0.35)), 50, 900))
    hdl = float(np.clip(rng.normal(39 if sex == "M" else 45, 7), 22, 90))
    ldl = float(np.clip(rng.normal(108 - (25 if any(m.drug.endswith("statin") for m in meds) else 0), 28), 40, 230))
    labs_latent = dict(
        tg_mgdl=tg, hdl_mgdl=hdl, ldl_mgdl=ldl, chol_mgdl=ldl + hdl + tg / 5.0,
        creatinine_mgdl=creat, egfr=float(np.round(egfr)),
        alt=float(np.clip(rng.normal(30 + 15 * ("nafld" in cond), 10), 8, 140)),
        uacr=float(np.clip(np.exp(rng.normal(np.log(12 + (40 if "diabetic_nephropathy" in cond else 0) + 1.5 * years), 0.7)), 2, 1500)),
        b12=float(np.clip(rng.normal(420 - (130 if therapy["metformin"] else 0), 110), 120, 1100)),
        tsh=float(np.clip(np.exp(rng.normal(np.log(2.2 + (2.5 if "hypothyroidism" in cond else 0)), 0.35)), 0.3, 15)),
        a1c_target=a1c_target,
    )

    first = FIRST_NAMES[region][sex]
    surname = SURNAMES[region]
    return Profile(
        patient_id=f"MT-{idx:04d}", name=f"{first[rng.integers(len(first))]} {surname[rng.integers(len(surname))]}",
        sex=sex, age=age, region=region, city=CITIES[region][rng.integers(len(CITIES[region]))],
        language=LANGUAGES[region][rng.integers(len(LANGUAGES[region]))], status=status, diabetes_years=years,
        height_cm=height, weight_kg=weight, bmi=float(np.round(bmi, 1)), waist_cm=waist,
        genetics={"TCF7L2_rs7903146_T_alleles": tcf7l2, "prs_t2d_z": prs}, family_history=fam,
        conditions=cond, meds=meds, adherence=float(np.clip(rng.beta(8, 1.6), 0.5, 1.0)),
        lifestyle=lifestyle, physiology=physiology, cardio=cardio, labs_latent=labs_latent, therapy_flags=therapy,
    )


def sample_population(n: int, seed: int = 7, status_mix: dict | None = None) -> list[Profile]:
    rng = np.random.default_rng(seed)
    mix = status_mix or STATUS_MIX
    statuses = rng.choice(list(mix), size=n, p=np.array(list(mix.values())) / sum(mix.values()))
    return [sample_profile(rng, i + 1, str(s)) for i, s in enumerate(statuses)]
