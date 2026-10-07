"""Loader for CGMacros (PhysioNet, CC BY-NC-SA 4.0) into the common schema.

Source quirks handled here:
* Minute-level rows; Dexcom G6 (5-min) and Libre Pro (15-min) are linearly interpolated to
  1 min by the publishers. We keep Dexcom as the primary CGM and Libre as `cgm_alt`.
* METs are stored x10. Weight is in lb, height in inches. HbA1c is in % despite the dictionary.
* Meal-type labels mix case ("Lunch"/"lunch", "Snacks"/"snack").
* Macros describe the whole plate; `Amount Consumed` (%) scales them to what was eaten.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from twin.paths import CGMACROS_RAW

from .schema import EVENT_COLUMNS, GRID_MINUTES, SERIES_COLUMNS, STATIC_COLUMNS, conform, diabetes_status

LB_TO_KG = 0.45359237
IN_TO_CM = 2.54


def _pid(subject: int) -> str:
    return f"CGM-{int(subject):03d}"


def load_static(root: Path = CGMACROS_RAW) -> pd.DataFrame:
    bio = pd.read_csv(root / "bio.csv")
    bio.columns = [c.strip() for c in bio.columns]
    df = pd.DataFrame({
        "patient_id": bio["subject"].map(_pid),
        "source": "cgmacros",
        "age": bio["Age"],
        "sex": bio["Gender"].str.strip().str.upper(),
        "height_cm": bio["Height"] * IN_TO_CM,
        "weight_kg": bio["Body weight"] * LB_TO_KG,
        "bmi": bio["BMI"],
        "hba1c_pct": bio["A1c PDL (Lab)"],
        "fpg_mgdl": bio["Fasting GLU - PDL (Lab)"],
        "fasting_insulin_uU": bio["Insulin"],
        "tg_mgdl": bio["Triglycerides"],
        "chol_mgdl": bio["Cholesterol"],
        "hdl_mgdl": bio["HDL"],
        "ldl_mgdl": bio["LDL (Cal)"].where(bio["LDL (Cal)"] < 700),     # 800 marks a calculation error
        "ethnicity": bio["Self-identify"].str.strip(),
    })
    df["homa_ir"] = df["fpg_mgdl"] * df["fasting_insulin_uU"] / 405.0
    df["status"] = [diabetes_status(a, f) for a, f in zip(df["hba1c_pct"], df["fpg_mgdl"], strict=True)]
    # CGMacros participants were not on glucose-lowering drugs during the study.
    for col in ("on_metformin", "on_sulfonylurea", "on_dpp4", "on_sglt2", "on_insulin"):
        df[col] = 0.0
    return conform(df, STATIC_COLUMNS)


def _read_subject(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    d["ts"] = pd.to_datetime(d["Timestamp"])
    return d.sort_values("ts").drop_duplicates("ts")


def _mets(d: pd.DataFrame, weight_kg: float | None) -> pd.Series:
    """METs per minute. Ten exports lack the METs column; for those we invert the standard
    energy equation kcal/min = METs * 3.5 * kg / 200 using Fitbit's activity calories."""
    if "METs" in d.columns:
        return d["METs"] / 10.0
    if weight_kg and "Calories (Activity)" in d.columns:
        return (d["Calories (Activity)"] * 200.0 / (3.5 * weight_kg)).clip(0.5, 20.0)
    return pd.Series(np.nan, index=d.index)


def load_subject(subject: int, root: Path = CGMACROS_RAW, weight_kg: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (series on the 5-min grid, meal events) for one participant."""
    pid = _pid(subject)
    d = _read_subject(root / f"CGMacros-{int(subject):03d}.csv")
    d["mets_1min"] = _mets(d, weight_kg)

    meal_rows = d[d["Meal Type"].notna()].copy()
    frac = (meal_rows.get("Amount Consumed", pd.Series(100.0, index=meal_rows.index)).fillna(100.0) / 100.0).clip(0, 1)
    events = pd.DataFrame({
        "patient_id": pid,
        "ts": meal_rows["ts"],
        "kind": "meal",
        "label": meal_rows["Meal Type"].str.strip().str.lower().str.replace(r"^snack.*", "snack", regex=True),
        "carbs": meal_rows["Carbs"] * frac,
        "protein": meal_rows["Protein"] * frac,
        "fat": meal_rows["Fat"] * frac,
        "fiber": meal_rows["Fiber"] * frac,
        "kcal": meal_rows["Calories"] * frac,
    })
    events = events[(events["carbs"].fillna(0) + events["protein"].fillna(0) + events["fat"].fillna(0)) > 0]

    minute = d.set_index("ts")
    full = pd.date_range(minute.index.min().floor(f"{GRID_MINUTES}min"), minute.index.max(), freq="1min")
    minute = minute.reindex(full)
    rule = f"{GRID_MINUTES}min"
    grid = pd.DataFrame({
        "cgm": minute["Dexcom GL"].resample(rule).mean(),
        "cgm_alt": minute["Libre GL"].resample(rule).mean(),
        "hr": minute["HR"].resample(rule).mean(),
        "mets": minute["mets_1min"].resample(rule).mean(),
    })
    grid.index.name = "ts"
    grid = grid.reset_index()
    grid["patient_id"] = pid

    bins = events.assign(ts=events["ts"].dt.floor(rule)).groupby("ts")[["carbs", "protein", "fat", "fiber"]].sum()
    grid = grid.merge(bins, left_on="ts", right_index=True, how="left")
    for c in ("carbs", "protein", "fat", "fiber"):
        grid[c] = grid[c].fillna(0.0)
    grid["insulin_units"] = 0.0
    grid["oad_dose"] = 0.0
    return conform(grid, SERIES_COLUMNS), conform(events, EVENT_COLUMNS)


def subjects(root: Path = CGMACROS_RAW) -> list[int]:
    return sorted(int(p.stem.split("-")[1]) for p in root.glob("CGMacros-0*.csv"))


def load(root: Path = CGMACROS_RAW) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the full dataset as (static, series, events)."""
    static = load_static(root)
    weights = dict(zip(static["patient_id"], static["weight_kg"], strict=True))
    parts = [load_subject(s, root, weights.get(_pid(s))) for s in subjects(root)]
    series = pd.concat([p[0] for p in parts], ignore_index=True)
    events = pd.concat([p[1] for p in parts], ignore_index=True)
    static = static[static["patient_id"].isin(series["patient_id"].unique())].reset_index(drop=True)
    return static, series, events


def summary(static: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    g = series.groupby("patient_id")
    s = pd.DataFrame({
        "days": g["ts"].agg(lambda x: (x.max() - x.min()).total_seconds() / 86400),
        "cgm_coverage": g["cgm"].apply(lambda x: x.notna().mean()),
        "cgm_mean": g["cgm"].mean(),
        "tir_70_180": g["cgm"].apply(lambda x: ((x >= 70) & (x <= 180)).sum() / max(x.notna().sum(), 1)),
        "tar_180": g["cgm"].apply(lambda x: (x > 180).sum() / max(x.notna().sum(), 1)),
        "hr_coverage": g["hr"].apply(lambda x: x.notna().mean()),
    })
    return static.set_index("patient_id")[["status", "age", "bmi", "hba1c_pct", "fpg_mgdl", "homa_ir"]].join(s).round(
        {"days": 1, "cgm_coverage": 2, "cgm_mean": 1, "tir_70_180": 2, "tar_180": 2, "hr_coverage": 2, "homa_ir": 2}
    ).assign(gmi=lambda x: np.round(3.31 + 0.02392 * x["cgm_mean"], 2))
