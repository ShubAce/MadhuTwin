"""Loader for the BIG IDEAs Lab Glycemic Variability and Wearable Device Data (PhysioNet, ODC-By 1.0).

16 adults (HbA1c 5.2-6.4%) wore a Dexcom G6 and an Empatica E4 wristband for ~10 days.
It is the real-world check of the *wearable* pathway: heart rate (1 Hz) and beat-to-beat
intervals, from which we compute HRV as RMSSD in 5-minute windows, the same definition the
synthetic device model uses.

The EHR side is sparse (sex and HbA1c only); everything else is left missing so models
exercise their missing-data handling, exactly as in deployment with a thin record.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from twin.paths import RAW

from .schema import EVENT_COLUMNS, GRID_MINUTES, SERIES_COLUMNS, STATIC_COLUMNS, conform, diabetes_status

ROOT = RAW / "bigideas"
RULE = f"{GRID_MINUTES}min"


def _two_col(path: Path) -> pd.DataFrame:
    """Empatica exports: first column timestamp, second column value (header names vary)."""
    d = pd.read_csv(path, encoding="utf-8-sig")
    ts, val = d.columns[0], d.columns[1]
    stamp = d[ts].astype(str)
    fmt = "%m/%d/%y %H:%M" if "/" in stamp.iloc[0] else "ISO8601"  # HR exports use m/d/yy hh:mm
    out = pd.DataFrame({"ts": pd.to_datetime(stamp, format=fmt, errors="coerce"), "v": pd.to_numeric(d[val], errors="coerce")})
    return out.dropna().sort_values("ts")


def rmssd_5min(ibi: pd.DataFrame, min_pairs: int = 20) -> pd.Series:
    """RMSSD (ms) per 5-min bin from successive, artefact-free beat-to-beat intervals."""
    t = ibi["ts"].to_numpy()
    x = ibi["v"].to_numpy()
    ok = (x > 0.3) & (x < 2.0)
    t, x = t[ok], x[ok]
    gap = np.diff(t).astype("timedelta64[ms]").astype(float) / 1000.0
    consecutive = np.abs(gap - x[1:]) < 0.25 * x[1:]          # the next beat follows directly
    d = np.diff(x)
    keep = consecutive & (np.abs(d) < 0.2)                     # reject ectopic / artefact jumps
    df = pd.DataFrame({"ts": t[1:][keep], "d2": d[keep] ** 2})
    g = df.set_index("ts")["d2"].resample(RULE)
    out = np.sqrt(g.mean()) * 1000.0
    return out.where(g.count() >= min_pairs)


HEADERLESS_FOOD = ["date", "time", "time_begin", "logged_food", "amount", "unit", "searched_food", "calorie",
                   "total_carb", "sugar", "protein"]  # participant 003's export has no header and fewer columns


def _read_food(path: Path) -> pd.DataFrame:
    first = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()[0]
    if first[:4].isdigit():
        return pd.read_csv(path, header=None, names=HEADERLESS_FOOD, usecols=range(len(HEADERLESS_FOOD)),
                           encoding="utf-8-sig", on_bad_lines="skip")
    return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")


def _meals(food: pd.DataFrame, pid: str) -> pd.DataFrame:
    food = food.copy()
    food.columns = [c.strip().lstrip("﻿").lower() for c in food.columns]
    if "time" not in food.columns and "time_of_day" in food.columns:
        food = food.rename(columns={"time_of_day": "time"})
    when = pd.to_datetime(food["time_begin"], errors="coerce") if "time_begin" in food else pd.Series(pd.NaT, index=food.index)
    if when.isna().all():
        when = pd.to_datetime(food["date"].astype(str) + " " + food["time"].astype(str), errors="coerce")
    food["ts"] = when
    def num(c: str) -> pd.Series:  # columns missing from an export count as 0
        return pd.to_numeric(food[c], errors="coerce").fillna(0.0) if c in food else pd.Series(0.0, index=food.index)

    food = food.assign(carbs=num("total_carb"), protein=num("protein"), fat=num("total_fat"), fiber=num("dietary_fiber"),
                       kcal=num("calorie")).dropna(subset=["ts"])
    meals = food.groupby(food["ts"].dt.floor("15min"))[["carbs", "protein", "fat", "fiber", "kcal"]].sum().reset_index()
    meals = meals[(meals.carbs + meals.protein + meals.fat) > 0]
    return meals.assign(patient_id=pid, kind="meal", label="meal")


def load(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    demo = pd.read_csv(root / "Demographics.csv")
    demo.columns = [c.strip().lstrip("﻿") for c in demo.columns]
    static_rows, series, events = [], [], []
    for _, r in demo.iterrows():
        sid = f"{int(r['ID']):03d}"
        folder = root / sid
        if not (folder / f"Dexcom_{sid}.csv").exists():
            continue
        pid = f"BIG-{sid}"
        dx = pd.read_csv(folder / f"Dexcom_{sid}.csv")
        dx = dx[dx["Event Type"] == "EGV"]
        cgm = pd.DataFrame({"ts": pd.to_datetime(dx.iloc[:, 1], errors="coerce"),
                            "v": pd.to_numeric(dx["Glucose Value (mg/dL)"], errors="coerce")}).dropna()
        grid = pd.date_range(cgm.ts.min().floor(RULE), cgm.ts.max().ceil(RULE), freq=RULE)
        s = pd.DataFrame(index=grid)
        s["cgm"] = cgm.set_index("ts")["v"].resample(RULE).mean().reindex(grid)
        if (folder / f"HR_{sid}.csv").exists():
            hr = _two_col(folder / f"HR_{sid}.csv")
            s["hr"] = hr.set_index("ts")["v"].resample(RULE).mean().reindex(grid)
        if (folder / f"IBI_{sid}.csv").exists():
            s["hrv_rmssd"] = rmssd_5min(_two_col(folder / f"IBI_{sid}.csv")).reindex(grid)
        meals = _meals(_read_food(folder / f"Food_Log_{sid}.csv"), pid) \
            if (folder / f"Food_Log_{sid}.csv").exists() else pd.DataFrame()
        for c in ("carbs", "protein", "fat", "fiber", "insulin_units", "oad_dose"):
            s[c] = 0.0
        if len(meals):
            b = meals.assign(ts=meals.ts.dt.floor(RULE)).groupby("ts")[["carbs", "protein", "fat", "fiber"]].sum()
            idx = s.index.intersection(b.index)
            s.loc[idx, ["carbs", "protein", "fat", "fiber"]] = b.loc[idx].to_numpy()
            events.append(meals)
        s.index.name = "ts"
        series.append(conform(s.reset_index().assign(patient_id=pid), SERIES_COLUMNS))
        a1c = float(r["HbA1c"])
        static_rows.append({"patient_id": pid, "source": "bigideas", "status": diabetes_status(a1c), "hba1c_pct": a1c,
                            "sex": "F" if str(r["Gender"]).upper().startswith("F") else "M",
                            "on_metformin": 0.0, "on_sulfonylurea": 0.0, "on_dpp4": 0.0, "on_sglt2": 0.0, "on_insulin": 0.0})
    static = conform(pd.DataFrame(static_rows), STATIC_COLUMNS)
    ev = conform(pd.concat(events, ignore_index=True), EVENT_COLUMNS) if events else conform(pd.DataFrame(columns=list(EVENT_COLUMNS)), EVENT_COLUMNS)
    return static, pd.concat(series, ignore_index=True), ev
