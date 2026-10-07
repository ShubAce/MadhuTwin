"""Common data schema shared by every source (CGMacros, ShanghaiT2DM, synthetic cohort).

A *patient dataset* is three tables:

static      one row per patient: demographics, labs, diagnoses, therapy, genetics (EHR stream)
series      one row per patient per 5-minute bin (wearable / IoT stream)
events      one row per discrete event: meals, medication doses, fingerstick checks

Missing modalities are NaN, never imputed at ingestion time, so models can learn to cope
with absent sensors (e.g. ShanghaiT2DM has no heart-rate or activity data).
"""

from __future__ import annotations

import pandas as pd

GRID_MINUTES = 5

STATIC_COLUMNS = {
    "patient_id": "str",
    "source": "str",               # cgmacros | shanghai | synthetic
    "status": "str",               # normal | prediabetes | t2d
    "age": "float",
    "sex": "str",                  # F | M
    "height_cm": "float",
    "weight_kg": "float",
    "bmi": "float",
    "hba1c_pct": "float",
    "fpg_mgdl": "float",
    "fasting_insulin_uU": "float",
    "homa_ir": "float",
    "tg_mgdl": "float",
    "chol_mgdl": "float",
    "hdl_mgdl": "float",
    "ldl_mgdl": "float",
    "egfr": "float",
    "diabetes_years": "float",
    "on_metformin": "float",
    "on_sulfonylurea": "float",
    "on_dpp4": "float",
    "on_sglt2": "float",
    "on_insulin": "float",
    "hypertension": "float",
    "dyslipidemia": "float",
    "tcf7l2_risk_alleles": "float",  # rs7903146 T allele count (0/1/2), synthetic cohort only
    "prs_t2d": "float",              # polygenic risk score (z), synthetic cohort only
}

SERIES_COLUMNS = {
    "patient_id": "str",
    "ts": "datetime64[ns]",
    "cgm": "float",          # mg/dL, primary CGM
    "cgm_alt": "float",      # mg/dL, second CGM if worn
    "hr": "float",           # bpm, mean over bin
    "mets": "float",         # metabolic equivalents, mean over bin
    "steps": "float",        # steps in bin
    "hrv_rmssd": "float",    # ms
    "sleep_stage": "float",  # 0 awake, 1 light, 2 deep, 3 REM; NaN unknown
    "carbs": "float",        # g eaten in bin
    "protein": "float",
    "fat": "float",
    "fiber": "float",
    "insulin_units": "float",  # IU injected in bin
    "oad_dose": "float",       # oral antidiabetic taken in bin (1 = dose taken)
}

EVENT_COLUMNS = {
    "patient_id": "str",
    "ts": "datetime64[ns]",
    "kind": "str",           # meal | insulin | oad | fingerstick
    "label": "str",
    "carbs": "float",
    "protein": "float",
    "fat": "float",
    "fiber": "float",
    "kcal": "float",
    "gi": "float",
    "amount": "float",
}


def conform(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Add missing columns as NaN, order them canonically and coerce dtypes."""
    out = df.copy()
    for col, dtype in columns.items():
        if col not in out.columns:
            out[col] = pd.NaT if dtype.startswith("datetime") else (pd.NA if dtype == "str" else float("nan"))
    out = out[list(columns)]
    for col, dtype in columns.items():
        if dtype == "float":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        elif dtype == "str":
            out[col] = out[col].astype("string")
        elif dtype.startswith("datetime"):
            out[col] = pd.to_datetime(out[col])
    return out


def diabetes_status(hba1c_pct: float, fpg: float | None = None, known_t2d: bool = False) -> str:
    """ADA criteria: HbA1c >= 6.5% or FPG >= 126 -> diabetes; 5.7-6.4% or FPG 100-125 -> prediabetes."""
    if known_t2d or (hba1c_pct is not None and hba1c_pct >= 6.5) or (fpg is not None and fpg >= 126 and hba1c_pct is None):
        return "t2d"
    if (hba1c_pct is not None and hba1c_pct >= 5.7) or (fpg is not None and fpg >= 100):
        return "prediabetes"
    return "normal"
