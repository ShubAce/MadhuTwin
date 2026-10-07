"""Loader for ShanghaiT2DM (Zhao et al., Sci Data 2023; Figshare, CC BY 4.0) into the common schema.

Used as an *external validation* cohort: different country, different CGM (15-min), a
rice-based diet, many insulin users, and no wearable heart-rate/activity data, so it also
tests how the twin copes with a missing modality.

Free-text fields are parsed:
* Dietary intake "Rice 150 g\\nEgg 50 g" -> carbohydrate / protein / fat / GI estimates from a
  per-100 g lookup of common Chinese foods (approximate, documented below).
* Insulin "insulin aspart 70/30, 12 IU" -> premix / rapid / regular / basal doses.
* Oral agents -> therapy flags and sulfonylurea dose events.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from twin.paths import SHANGHAI_RAW

from .schema import EVENT_COLUMNS, GRID_MINUTES, SERIES_COLUMNS, STATIC_COLUMNS, conform, diabetes_status

# food keyword -> (carbs, protein, fat per 100 g, glycaemic index); first match wins, so order matters
FOOD_TABLE: list[tuple[str, tuple[float, float, float, float]]] = [
    ("porridge", (10, 1.5, 0.3, 78)), ("congee", (10, 1.5, 0.3, 78)), ("rice in soup", (12, 1.5, 0.5, 75)),
    ("rice cake", (45, 3, 0.5, 82)), ("rice noodle", (24, 1.5, 0.3, 61)), ("black rice", (30, 3, 0.8, 50)),
    ("rice", (28, 2.6, 0.3, 73)),
    ("steamed bread", (47, 7, 1, 85)), ("steamed bun", (47, 7, 1, 85)), ("pork bun", (35, 9, 7, 70)),
    ("vegetable bun", (38, 6, 4, 70)), ("bean curd roll", (8, 15, 10, 30)), ("corn bread", (45, 7, 2, 65)),
    ("buckwheat bread", (40, 8, 2, 50)), ("whole wheat bread", (43, 10, 3, 55)), ("bread", (50, 8, 3, 75)),
    ("coarse grain", (25, 3, 1, 55)), ("buckwheat", (25, 4, 1, 50)), ("oat", (12, 2.5, 1.5, 55)),
    ("noodle", (25, 4.5, 0.5, 50)), ("dumpling", (25, 8, 6, 60)), ("wonton", (22, 8, 5, 60)),
    ("pancake", (40, 6, 8, 70)), ("cake", (50, 5, 15, 65)), ("biscuit", (70, 7, 15, 70)),
    ("cereal", (65, 10, 6, 65)), ("meal replacement", (50, 20, 5, 40)), ("millet", (12, 2, 0.7, 70)),
    ("sweet potato leaves", (4, 3, 0.3, 15)), ("sweet potato", (20, 1.6, 0.1, 63)), ("potato", (17, 2, 0.1, 78)),
    ("taro", (18, 1.5, 0.2, 55)), ("yam", (12, 1.5, 0.1, 51)), ("corn", (20, 3.3, 1.2, 55)), ("pumpkin", (6, 1, 0.1, 65)),
    ("soybean milk", (3, 3, 1.8, 30)), ("yogurt", (12, 3.5, 3, 35)), ("milk", (5, 3.2, 3.5, 30)),
    ("coffee and milk", (4, 1.5, 1.5, 35)), ("coffee", (1, 0.2, 0, 30)), ("beer", (4, 0.4, 0, 66)),
    ("rice wine", (5, 1.5, 0, 60)),
    ("red date", (60, 2, 0.5, 55)), ("watermelon", (7, 0.6, 0.2, 72)), ("banana", (22, 1.1, 0.3, 51)),
    ("apple", (13, 0.3, 0.2, 36)), ("orange", (11, 0.9, 0.1, 43)), ("peach", (10, 0.9, 0.3, 42)),
    ("pear", (12, 0.4, 0.1, 38)), ("grapefruit", (8, 0.7, 0.1, 25)), ("kiwi", (14, 1.1, 0.5, 52)),
    ("grape", (16, 0.7, 0.2, 46)), ("fruit", (12, 0.7, 0.2, 45)),
    ("peanut", (15, 25, 45, 14)), ("walnut", (14, 15, 65, 15)),
    ("tofu", (3, 8, 4, 15)), ("egg", (1, 13, 10, 0)), ("fish", (0, 18, 5, 0)), ("shrimp", (0, 20, 1, 0)),
    ("crab", (0, 18, 1, 0)), ("sea cucumber", (1, 16, 0.2, 0)), ("hairtail", (0, 18, 5, 0)), ("croaker", (0, 18, 3, 0)),
    ("beef", (0, 26, 10, 0)), ("mutton", (0, 25, 15, 0)), ("pork", (2, 20, 20, 0)), ("rib", (0, 18, 20, 0)),
    ("sausage", (10, 12, 25, 28)), ("ham", (5, 14, 15, 28)), ("meatball", (8, 14, 15, 30)), ("chicken", (1, 22, 8, 0)),
    ("duck", (0, 18, 15, 0)), ("pig feet", (0, 23, 19, 0)), ("pigeon", (1, 10, 5, 0)), ("nugget", (15, 15, 18, 46)),
    ("soup", (2, 1.5, 1, 30)), ("gluten", (6, 25, 1, 30)), ("pickle", (3, 1, 0.2, 15)),
]
VEGETABLE_DEFAULT = (3.5, 1.5, 0.3, 15)
UNKNOWN_DEFAULT = (8, 4, 3, 50)
VEGETABLE_WORDS = ("vegetable", "cabbage", "cucumber", "lettuce", "spinach", "tomato", "celery", "radish", "eggplant",
                   "broccoli", "cauliflower", "gourd", "bean sprout", "pepper", "mushroom", "fungus", "amaranth",
                   "daisy", "bamboo", "carrot", "leek", "onion", "kelp", "seaweed", "greens", "lotus")

INSULIN_MAP = [
    (r"aspart\s*(70/30|50/50)|30r|70/30|40r|50r|m30|mix", "insulin_premix_30_70"),
    (r"degludec|detemir|glargine|glarigine|lantus|levemir|toujeo", "insulin_glargine"),
    (r"aspart|glulisine|lispro|novorapid|humalog", "insulin_rapid"),
    (r"novolin r|gansulin r|humulin r|regular|\br\b", "insulin_regular"),
]
OAD_CLASS = {
    "metformin": "metformin", "acarbose": "agi", "voglibose": "agi", "sitagliptin": "dpp4", "linagliptin": "dpp4",
    "saxagliptin": "dpp4", "alogliptin": "dpp4", "vildagliptin": "dpp4", "liraglutide": "dpp4", "dulaglutide": "dpp4",
    "gliclazide": "sulfonylurea", "glimepiride": "sulfonylurea", "glipizide": "sulfonylurea", "gliquidone": "sulfonylurea",
    "repaglinide": "sulfonylurea", "pioglitazone": "tzd", "dapagliflozin": "sglt2", "empagliflozin": "sglt2",
    "canagliflozin": "sglt2",
}
MMOL_CHOL, MMOL_TG = 38.67, 88.57


def parse_meal(text: str) -> dict | None:
    carbs = protein = fat = 0.0
    gi_weight = 0.0
    for line in str(text).split("\n"):
        m = re.match(r"\s*(.+?)\s+([\d.]+)\s*(g|ml)\s*$", line.strip(), flags=re.I)
        if not m:
            continue
        name, amount = m.group(1).lower(), float(m.group(2))
        row = next((v for k, v in FOOD_TABLE if k in name), None)
        if row is None:
            row = VEGETABLE_DEFAULT if any(w in name for w in VEGETABLE_WORDS) else UNKNOWN_DEFAULT
        c, p, f, gi = row
        carbs += c * amount / 100
        protein += p * amount / 100
        fat += f * amount / 100
        gi_weight += gi * c * amount / 100
    if carbs + protein + fat <= 0:
        return None
    return dict(carbs=carbs, protein=protein, fat=fat, fiber=0.0, gi=gi_weight / carbs if carbs > 0 else 40.0,
                kcal=4 * carbs + 4 * protein + 9 * fat)


def parse_insulin(text: str) -> list[tuple[str, float]]:
    out = []
    for m in re.finditer(r"([A-Za-z][A-Za-z0-9 /\-\xa0]*?)[, ]+\s*([\d.]+)\s*IU", str(text)):
        name = m.group(1).replace("\xa0", " ").strip().lower()
        drug = next((d for pat, d in INSULIN_MAP if re.search(pat, name)), "insulin_regular")
        out.append((drug, float(m.group(2))))
    return out


def parse_oads(text: str) -> list[tuple[str, float]]:
    out = []
    for name in OAD_CLASS:
        m = re.search(name + r"[^\d,]*([\d.]+)\s*(mg|g)?", str(text).lower())
        if m:
            dose = float(m.group(1)) * (1000 if (m.group(2) == "g" and float(m.group(1)) < 5) else 1)
            out.append((name, dose))
    return out


def _flags(agents: str) -> dict:
    a = str(agents).lower()
    cls = {c for n, c in OAD_CLASS.items() if n in a}
    insulin = any(re.search(p, a) for p, _ in INSULIN_MAP) or "insulin" in a or "novolin" in a or "humulin" in a
    return dict(on_metformin=float("metformin" in cls), on_sulfonylurea=float("sulfonylurea" in cls),
                on_dpp4=float("dpp4" in cls), on_sglt2=float("sglt2" in cls), on_insulin=float(insulin))


def load_static(root: Path = SHANGHAI_RAW) -> pd.DataFrame:
    s = pd.read_excel(root / "Shanghai_T2DM_Summary.xlsx")
    s.columns = [re.sub(r"\s+", " ", c).strip() for c in s.columns]
    col = lambda key: next(c for c in s.columns if c.lower().startswith(key.lower()))  # noqa: E731
    num = lambda key: pd.to_numeric(s[col(key)], errors="coerce")  # noqa: E731
    rid = s[col("Patient Number")].astype(str).str.strip()
    hba1c_pct = num("HbA1c") / 10.929 + 2.15
    fpg = num("Fasting Plasma Glucose")
    ins = num("Fasting Insulin") / 6.0
    flags = pd.DataFrame([_flags(a) for a in s[col("Hypoglycemic Agents")]])
    comorb = s[col("Comorbidities")].astype(str).str.lower()
    df = pd.DataFrame({
        "record": rid,
        "source": "shanghai",
        "status": [diabetes_status(a, f, known_t2d=True) for a, f in zip(hba1c_pct, fpg, strict=True)],
        "age": num("Age"), "sex": s[col("Gender")].map({1: "F", 2: "M"}),
        "height_cm": num("Height") * 100, "weight_kg": num("Weight"), "bmi": num("BMI"),
        "hba1c_pct": hba1c_pct, "fpg_mgdl": fpg, "fasting_insulin_uU": ins, "homa_ir": fpg * ins / 405.0,
        "tg_mgdl": num("Triglyceride") * MMOL_TG, "chol_mgdl": num("Total Cholesterol") * MMOL_CHOL,
        "hdl_mgdl": num("High-Density") * MMOL_CHOL, "ldl_mgdl": num("Low-Density") * MMOL_CHOL,
        "egfr": num("Estimated Glomerular"),
        "diabetes_years": num("Duration of diabetes"),
        "hypertension": comorb.str.contains("hypertension").astype(float),
        "dyslipidemia": comorb.str.contains("lipid").astype(float),
    })
    df = pd.concat([df, flags], axis=1)
    return df


def load_record(path: Path, patient_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_excel(path)
    d.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in d.columns]
    find = lambda key: next((c for c in d.columns if c.lower().startswith(key.lower())), None)  # noqa: E731
    d["ts"] = pd.to_datetime(d[find("Date")])
    d = d.sort_values("ts")
    cgm = pd.to_numeric(d[find("CGM")], errors="coerce")

    events = []
    for _, r in d.iterrows():
        diet = r.get(find("Dietary intake"))
        if isinstance(diet, str) and "not available" not in diet.lower():
            m = parse_meal(diet)
            if m:
                events.append(dict(patient_id=patient_id, ts=r["ts"], kind="meal", label="meal", amount=np.nan, **m))
        for key in ("Insulin dose - s.c.", "CSII - bolus insulin"):
            c = find(key)
            if c and pd.notna(r.get(c)):
                if key.startswith("CSII"):
                    v = pd.to_numeric(r[c], errors="coerce")
                    if pd.notna(v) and v > 0:
                        events.append(dict(patient_id=patient_id, ts=r["ts"], kind="insulin", label="insulin_regular", amount=float(v)))
                else:
                    for drug, units in parse_insulin(r[c]):
                        events.append(dict(patient_id=patient_id, ts=r["ts"], kind="insulin", label=drug, amount=units))
        c = find("Non-insulin")
        if c and pd.notna(r.get(c)):
            for drug, dose in parse_oads(r[c]):
                label = drug if drug in ("gliclazide", "glimepiride") else drug
                events.append(dict(patient_id=patient_id, ts=r["ts"], kind="oad", label=label, amount=dose))
    ev = pd.DataFrame(events)

    rule = f"{GRID_MINUTES}min"
    grid_idx = pd.date_range(d["ts"].min().floor(rule), d["ts"].max().ceil(rule), freq=rule)
    raw = pd.Series(cgm.to_numpy(), index=d["ts"].dt.round(rule)).groupby(level=0).mean()
    grid = pd.DataFrame(index=grid_idx)
    grid["cgm"] = raw.reindex(grid_idx)
    grid.index.name = "ts"
    grid = grid.reset_index()
    grid["patient_id"] = patient_id
    for c in ("carbs", "protein", "fat", "fiber", "insulin_units", "oad_dose"):
        grid[c] = 0.0
    if len(ev):
        b = ev.assign(ts=ev["ts"].dt.floor(rule))
        meals = b[b.kind == "meal"].groupby("ts")[["carbs", "protein", "fat", "fiber"]].sum()
        ins = b[b.kind == "insulin"].groupby("ts")["amount"].sum()
        oad = b[b.kind == "oad"].groupby("ts").size()
        g = grid.set_index("ts")
        g.loc[g.index.intersection(meals.index), ["carbs", "protein", "fat", "fiber"]] = meals.loc[g.index.intersection(meals.index)].to_numpy()
        g.loc[g.index.intersection(ins.index), "insulin_units"] = ins.loc[g.index.intersection(ins.index)].to_numpy()
        g.loc[g.index.intersection(oad.index), "oad_dose"] = oad.loc[g.index.intersection(oad.index)].to_numpy()
        grid = g.reset_index()
    return conform(grid, SERIES_COLUMNS), conform(ev, EVENT_COLUMNS) if len(ev) else conform(pd.DataFrame(columns=list(EVENT_COLUMNS)), EVENT_COLUMNS)


def load(root: Path = SHANGHAI_RAW) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(static, series, events), one patient_id per recording (SH-<patient>-<record>)."""
    static = load_static(root)
    series, events = [], []
    for _, row in static.iterrows():
        path = next(iter(sorted((root / "Shanghai_T2DM").glob(f"{row['record']}.xls*"))), None)
        if path is None:
            continue
        pid = "SH-" + "-".join(row["record"].split("_")[:2])
        s, e = load_record(path, pid)
        series.append(s)
        events.append(e)
    static = static.assign(patient_id="SH-" + static["record"].str.split("_").str[:2].str.join("-"))
    static["group"] = "SH-" + static["record"].str.split("_").str[0]
    st = conform(static, STATIC_COLUMNS).assign(group=static["group"].to_numpy())
    return st, pd.concat(series, ignore_index=True), pd.concat(events, ignore_index=True)
