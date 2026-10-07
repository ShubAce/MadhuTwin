"""Turn SHAP contributions into plain-language reasons a clinician can scan in a second."""

from __future__ import annotations

import numpy as np


def _hhmm(minutes_ago: float) -> str:
    return f"{int(minutes_ago)} min ago" if minutes_ago < 120 else f"{minutes_ago / 60:.1f} h ago"


def phrase(name: str, value: float, contribution: float, event: str) -> str | None:
    """A short human sentence for one feature, or None if it should not be shown."""
    v = value
    if not np.isfinite(v):
        return None
    if name == "cgm_now":
        return f"Glucose now {v:.0f} mg/dL"
    if name in ("cgm_slope30", "cgm_d30", "cgm_d15"):
        mins = 30 if name != "cgm_d15" else 15
        rate = v * 30 if name == "cgm_slope30" else v
        return f"{'Rising' if rate > 0 else 'Falling'} {abs(rate):.0f} mg/dL over {mins} min"
    if name.startswith("cgm_d"):
        return f"Change over last {name[5:]} min: {v:+.0f} mg/dL"
    if name in ("cgm_mean120", "cgm_mean60", "cgm_mean360"):
        return f"Average glucose (last {int(name[8:]) // 60 or 1} h) {v:.0f} mg/dL"
    if name == "cob":
        return f"{v:.0f} g carbohydrate still being absorbed" if v >= 3 else "No carbohydrate on board"
    if name.startswith("carbs_"):
        return f"{v:.0f} g carbs eaten in the last {name[6:]} min" if v >= 3 else None
    if name == "min_since_meal":
        return f"Last logged meal {_hhmm(v)}" if v < 400 else "No meal logged in 6 h"
    if name == "fpob":
        return f"High-fat/protein meal slowing absorption ({v:.0f} g)" if v > 15 else None
    if name == "iob":
        return f"{v:.1f} U insulin still active" if v >= 0.5 else None
    if name == "insulin_2h":
        return f"{v:.0f} U insulin injected in the last 2 h" if v > 0 else None
    if name == "oad_6h":
        return "Oral medication taken recently" if v > 0 else "No oral medication logged recently"
    if name == "hr_dev":
        # raised resting heart rate suggests stress/illness (spikes) or an adrenergic response (lows)
        return f"Heart rate {v:+.0f} bpm above personal 24 h baseline" if v >= 5 else None
    if name == "hr_now":
        return None
    if name in ("hrv_6h", "hrv_last"):
        return f"HRV {v:.0f} ms ({'suppressed' if v < 20 else 'normal'})"
    if name in ("mets_60", "mets_15"):
        if event == "hypo":
            return "Physically active right now (raises insulin action)" if v >= 2.5 else None
        return "Sedentary for the last hour" if v < 1.5 else None
    if name in ("steps_60", "steps_6h"):
        if event == "hypo":
            return f"Active: {v:.0f} steps in the last {'hour' if name == 'steps_60' else '6 h'}" if v >= 1500 else None
        return f"Inactive: {v:.0f} steps in the last hour" if name == "steps_60" and v < 300 else None
    if name == "sleep_frac_8h":
        return f"Slept {v * 8:.1f} of the last 8 h"
    if name == "asleep_now":
        return "Asleep (nocturnal risk window)" if v > 0.5 else None
    if name.startswith("twin_d") or name in ("twin_max", "twin_min"):
        if name == "twin_max":
            return f"Twin simulation projects a rise of {v:+.0f} mg/dL"
        if name == "twin_min":
            return f"Twin simulation projects a fall of {v:+.0f} mg/dL"
        return f"Twin simulation: {v:+.0f} mg/dL in {name[6:]} min"
    if name == "sync_si_log_mult":
        pct = (np.exp(v) - 1) * 100
        return f"Insulin sensitivity {abs(pct):.0f}% {'below' if pct < 0 else 'above'} personal baseline" if abs(pct) >= 10 else None
    if name == "sync_insulin_excess":
        return "Circulating insulin above basal" if v > 0.2 else None
    if name == "ehr_hba1c_pct":
        return f"HbA1c {v * 1.5 + 7:.1f}% (EHR)"
    if name == "ehr_on_sulfonylurea" and v > 0.5:
        return "On a sulfonylurea (EHR)"
    if name == "ehr_on_insulin" and v > 0.5:
        return "On insulin therapy (EHR)"
    if name == "ehr_homa_ir":
        return f"Insulin resistance HOMA-IR {v * 3 + 3.5:.1f} (EHR)"
    if name == "ehr_diabetes_years":
        return f"Diabetes for {max(v * 6 + 5, 0):.0f} years (EHR)"
    if name in ("tod_sin", "tod_cos"):
        return None
    return None


def reasons(contribs: list[tuple[str, float]], values: dict[str, float], event: str, k: int = 3) -> list[str]:
    """Top-k risk-increasing reasons, de-duplicated."""
    out: list[str] = []
    for name, c in contribs:
        if c <= 0:
            continue
        p = phrase(name, values.get(name, np.nan), c, event)
        if p and p not in out:
            out.append(p)
        if len(out) >= k:
            break
    return out
