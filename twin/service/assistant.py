"""'Ask the twin': grounded answers to a clinician's questions about a virtual patient.

The offline engine routes a question to one of the twin's own tools (forecast, AGP, personal
meal response, walk simulation, insulin-sensitivity trace, medications) and composes an answer
from the numbers it returns, citing them. It never recommends specific doses: decision support
for a clinician, not a prescriber. An optional LLM layer (see llm.py) can rephrase and combine
the same tool outputs when an API key is configured.
"""

from __future__ import annotations

import re

from .store import Patient

INTENTS = [
    ("hypo", r"\b(hypo|low|lows|below 70|drop|crash)"),
    ("why", r"\b(why|cause|reason|spike|high|rising|alert)"),
    ("food", r"\b(food|meal|eat|diet|rice|roti|idli|dosa|biryani|millet|ragi|breakfast|lunch|dinner)"),
    ("walk", r"\b(walk|exercise|activity|steps|active)"),
    ("sensitivity", r"\b(sensitiv|resistan|ill|sick|infection|stress|fever)"),
    ("meds", r"\b(medic|drug|dose|insulin|metformin|adheren|pill|tablet)"),
    ("sleep", r"\b(sleep|night|tired)"),
    ("summary", r"\b(summary|summari|overview|how is|how's|week|control|report)"),
]


def route(question: str) -> str:
    q = question.lower()
    for intent, pat in INTENTS:
        if re.search(pat, q):
            return intent
    return "summary"


def answer(p: Patient, clock: float, question: str) -> dict:
    intent = route(question)
    st = p.state(clock, history_h=6)
    fc = st["forecast"] or {}
    name = (p.bundle["display"].get("name") or p.id).split(" ")[0]
    facts: list[str] = []

    if intent == "why":
        prob = fc.get("spike")
        peak = max((v for v in fc.get("q50") or [] if v is not None), default=None)
        why = fc.get("why_spike") or []
        text = (f"{name}'s twin gives a {prob:.0%} chance of glucose above 180 mg/dL (for 15+ min) in the next 2 hours; "
                f"the median forecast peaks at {peak:.0f} mg/dL." if prob is not None and peak is not None else
                "No forecast is available at this moment.")
        if why:
            text += " Main drivers: " + "; ".join(why) + "."
        facts = why
    elif intent == "hypo":
        prob = fc.get("hypo")
        low = min((v for v in fc.get("q10") or [] if v is not None), default=None)
        ag = p.agp(clock, days=7)
        text = (f"Hypoglycaemia risk in the next 2 h: {prob:.0%}. The pessimistic (10th percentile) forecast reaches "
                f"{low:.0f} mg/dL. Over the last 7 days, {ag['tbr1'] + ag['tbr2']:.1f}% of readings were below 70 mg/dL (target <4%).")
        why = fc.get("why_hypo") or []
        if why:
            text += " Contributing: " + "; ".join(why) + "."
        meds = [m["display"] for m in p.bundle["ehr"].get("medications", []) if m.get("class") in ("sulfonylurea", "insulin")]
        if meds:
            text += f" Hypoglycaemia-prone therapy on record: {', '.join(meds)}."
        facts = why
    elif intent == "food":
        rank = p.meal_ranking()
        best = ", ".join(f"{r['name']} (+{r['rise']})" for r in rank[:3])
        worst = ", ".join(f"{r['name']} (+{r['rise']})" for r in rank[-3:][::-1])
        text = (f"Simulated on {name}'s personalised twin from a fasting start, the gentlest meals are {best} mg/dL; "
                f"the largest rises come from {worst} mg/dL. Use the What-if tab to test a specific plate and portion.")
        facts = [best, worst]
    elif intent == "walk":
        wb = p.walk_benefit()
        text = (f"In {name}'s twin, a 15-minute walk starting 20 minutes after {wb['meal'].lower()} lowers the post-meal peak "
                f"from {wb['peak_without']} to {wb['peak_with_walk']} mg/dL (-{wb['reduction']} mg/dL).")
        facts = [f"peak {wb['peak_without']} -> {wb['peak_with_walk']}"]
    elif intent == "sensitivity":
        si = [x for x in (p.bundle["twin"].get("si_daily") or []) if x is not None]
        today = st.get("si_today")
        if si and today is not None:
            low_day = min(range(len(si)), key=lambda i: si[i])
            text = (f"The synchronised twin tracks insulin sensitivity relative to {name}'s own baseline. Today: {today:.2f}x. "
                    f"Lowest in the record: {si[low_day]:.2f}x on day {low_day + 1}. Drops below ~0.8x usually reflect illness, "
                    "stress or poor sleep, and precede higher glucose for the same meals.")
        else:
            text = "Insulin-sensitivity tracking is not available for this record."
    elif intent == "meds":
        meds = p.bundle["ehr"].get("medications") or []
        lines = [f"{m['display']} {m.get('dose', '')}{m.get('unit', '') or ''} at {', '.join(m.get('times') or [])}".strip() for m in meds]
        text = "Current medications: " + ("; ".join(lines) if lines else "none recorded") + "."
        adh = [i for i in p.insights(clock) if i["title"] == "Medication adherence"]
        if adh:
            text += " " + adh[0]["text"]
        text += " Dose decisions remain with the treating clinician."
    elif intent == "sleep":
        s = p.bundle["series"]
        b = p.bin_at(clock)
        sl = [x for x in s["sleep"][max(b - 288, 0) : b + 1] if x is not None]
        hrs = sum(1 for x in sl if x > 0) * 5 / 60 if sl else None
        text = (f"{name} slept about {hrs:.1f} h in the last 24 h (wearable). In the twin, each hour of sleep deficit lowers "
                "next-day insulin sensitivity by ~7%, raising post-meal peaks." if hrs is not None else "No sleep data for this record.")
    else:
        ag = p.agp(clock, days=7)
        ins = p.insights(clock)
        text = (f"Last {ag['days']:.0f} days: mean glucose {ag['mean']:.0f} mg/dL, GMI {ag['gmi']:.1f}%, time in range "
                f"{ag['tir']:.0f}% (target >70%), above 250: {ag['tar2']:.1f}%, below 70: {ag['tbr1'] + ag['tbr2']:.1f}%, "
                f"CV {ag['cv']:.0f}%.")
        if ins:
            text += " Twin insights: " + " ".join(i["text"] for i in ins[:2])
    return {"intent": intent, "answer": text, "facts": facts, "engine": "offline"}
