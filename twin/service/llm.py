"""Optional LLM layer for 'Ask the twin': Claude with tool use over the twin's own functions.

Disabled by default. Enable with MADHUTWIN_LLM=on plus Anthropic credentials (ANTHROPIC_API_KEY,
or a profile from `ant auth login`). Every number in an answer comes from a tool call into the
patient's twin (forecast, AGP, record, physiology, meal simulation, what-if); the system prompt
forbids dose recommendations. If the model declines or any error occurs, the API falls back to
the deterministic offline engine (assistant.py).
"""

from __future__ import annotations

import json
import os

import anthropic

from twin.physiology.foods import FOODS

from .store import Patient

MODEL = os.environ.get("MADHUTWIN_LLM_MODEL", "claude-opus-5-5")
MAX_TURNS = 6

SYSTEM = """You are the clinical decision-support assistant inside MadhuTwin, a digital twin of a person with (or at risk of) type 2 diabetes. Your reader is the treating doctor.

Answer from the patient's twin, not from general knowledge: call the tools to get the forecast, glucose summary, record, learned physiology, personal meal responses, or to simulate a scenario, and quote the numbers they return. Combine tools when a question needs it (for example, compare two meals by simulating both).

Write 2 to 5 sentences of plain clinical language, leading with the answer. Say when data is missing or a result is uncertain; synthetic patients are simulated, real-world recordings are de-identified research data.

You support clinical judgement; you do not make treatment decisions. Never recommend specific drug doses or changes to a dose. If asked, you can show a simulated effect and state that the decision rests with the treating clinician."""

TOOLS = [
    {"name": "get_current_forecast", "description": "The twin's 2-hour forecast at the current time: glucose now, median and 80% interval every 5 minutes, probability of glucose >180 or <70 for 15+ minutes, and the main contributing factors.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "get_glucose_summary", "description": "Ambulatory glucose profile metrics over the last N days: mean, GMI, coefficient of variation, time in range 70-180, time above 180/250, time below 70/54.",
     "input_schema": {"type": "object", "properties": {"days": {"type": "integer", "minimum": 1, "maximum": 14, "description": "Look-back window in days."}},
                      "required": ["days"], "additionalProperties": False}},
    {"name": "get_patient_record", "description": "EHR summary: demographics, problem list, medications with timing, latest labs, family history, genetic markers.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "get_twin_physiology", "description": "Physiology learned by the twin: insulin sensitivity and beta-cell response relative to healthy, fasting set-point, dawn effect, absorption speed, how much personalisation reduced twin error, and the daily insulin-sensitivity trend from the synced filter (drops suggest illness, stress or poor sleep).",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "rank_meals", "description": "Predicted glucose rise for common Indian meals, simulated on this patient's personalised twin from a fasting start, sorted from gentlest to largest.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "simulate_scenario", "description": "Run a what-if on the twin from the current state over the next 4 hours and compare with no change. Any combination of: a meal from the Indian food list with a portion multiplier, a walk after it, extra rapid-acting insulin (educational only), last night's sleep, acute illness.",
     "input_schema": {"type": "object", "properties": {
         "food": {"type": "string", "enum": sorted(FOODS), "description": "Food key from the Indian food table."},
         "portion": {"type": "number", "minimum": 0.25, "maximum": 3, "description": "Portion multiplier, 1 = standard serving."},
         "walk_minutes": {"type": "integer", "minimum": 0, "maximum": 90, "description": "Walk starting 20 minutes after the meal."},
         "extra_rapid_insulin_units": {"type": "number", "minimum": 0, "maximum": 20},
         "sleep_hours": {"type": "number", "minimum": 2, "maximum": 12},
         "illness": {"type": "boolean"}}, "additionalProperties": False}},
    {"name": "get_insights", "description": "Twin-generated insights for this patient (insulin-resistance spikes, dawn phenomenon, meal response, walking benefit, time below range, adherence).",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
]


def available() -> bool:
    return os.environ.get("MADHUTWIN_LLM", "off").lower() == "on"


def _run_tool(p: Patient, clock: float, name: str, args: dict) -> dict:
    if name == "get_current_forecast":
        st = p.state(clock, history_h=1)
        fc = st["forecast"] or {}
        return {"time": st["now"], "glucose_now": next((v for v in reversed(st["series"]["cgm"]) if v is not None), None),
                "p_above_180": fc.get("spike"), "p_below_70": fc.get("hypo"), "median_next_2h_every_15min": (fc.get("q50") or [])[2::3],
                "p10_next_2h_every_15min": (fc.get("q10") or [])[2::3], "p90_next_2h_every_15min": (fc.get("q90") or [])[2::3],
                "drivers_high": fc.get("why_spike"), "drivers_low": fc.get("why_hypo")}
    if name == "get_glucose_summary":
        a = p.agp(clock, days=int(min(max(args.get("days", 7), 1), 14)))
        return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in a.items() if k != "profile"}
    if name == "get_patient_record":
        e = p.bundle["ehr"]
        return {"display": p.bundle["display"], "story": p.bundle["story"], "status": e.get("status"),
                "diabetes_years": e.get("diabetes_years"), "bmi": e.get("bmi"), "conditions": [c["display"] for c in e.get("conditions", [])],
                "medications": e.get("medications"), "labs": {lab["display"]: f"{lab['value']} {lab['unit']}" for lab in e.get("labs", [])},
                "family_history": e.get("family_history"), "genetics": e.get("genetics")}
    if name == "get_twin_physiology":
        tw = p.bundle["twin"]
        prm = tw["params"]
        return {"insulin_sensitivity_pct_of_healthy": round(prm["SI"] / 6.5e-4 * 100) if prm.get("SI") else None,
                "beta_cell_response_pct_of_healthy": round(prm["beta"] / 0.13 * 100) if prm.get("beta") else None,
                "fasting_setpoint_mgdl": prm.get("Gb"), "dawn_effect_pct": round(prm["dawn"] * 100) if prm.get("dawn") is not None else None,
                "absorption_speed_vs_average": round(1 / tw["tau_scale"], 2) if tw.get("tau_scale") else None,
                "twin_error_mgdl_before_after_personalisation": [tw.get("fit_rmse_prior"), tw.get("fit_rmse_personalised")],
                "insulin_sensitivity_today_vs_baseline": p.si_today(p.bin_at(clock)), "daily_insulin_sensitivity_vs_baseline": tw.get("si_daily")}
    if name == "rank_meals":
        return {"meals": [{"meal": m["name"], "rise_mgdl": m["rise"], "carbs_g": m["carbs"], "gi": m["gi"]} for m in p.meal_ranking()]}
    if name == "simulate_scenario":
        food = args.get("food")
        meal = {"food": food, "portion": float(args.get("portion", 1.0)), "in_min": 0} if food in FOODS else None
        walk = {"minutes": int(args["walk_minutes"]), "delay_min": 20} if args.get("walk_minutes") else None
        ins = {"drug": "insulin_rapid", "units": float(args["extra_rapid_insulin_units"]), "in_min": 0} if args.get("extra_rapid_insulin_units") else None
        r = p.what_if(clock, meal=meal, walk=walk, insulin=ins, sleep_hours=args.get("sleep_hours"), illness=bool(args.get("illness", False)))
        return {k: v for k, v in r.items() if k not in ("times", "baseline", "scenario")}
    if name == "get_insights":
        return {"insights": p.insights(clock)}
    raise ValueError(f"unknown tool {name}")


def answer(p: Patient, clock: float, question: str) -> dict:
    client = anthropic.Anthropic()
    d = p.bundle["display"]
    intro = (f"Patient: {d.get('name')} ({d.get('label')}). Clinical story: {p.bundle['story']}. "
             f"Current time in the record: {p.time_of(p.bin_at(clock)).strftime('%Y-%m-%d %H:%M')}.\n\nQuestion: {question}")
    messages: list[dict] = [{"role": "user", "content": intro}]
    used: list[str] = []
    for _ in range(MAX_TURNS):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("model declined")
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_uses:
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            return {"intent": "llm", "answer": text, "facts": used, "engine": response.model}
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for tu in tool_uses:
            used.append(tu.name)
            try:
                out = _run_tool(p, clock, tu.name, dict(tu.input or {}))
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": json.dumps(out, default=float)})
            except Exception as e:  # report the failure to the model rather than dropping the call
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": f"Error: {e}", "is_error": True})
        messages.append({"role": "user", "content": results})
    raise RuntimeError("tool loop did not finish")
