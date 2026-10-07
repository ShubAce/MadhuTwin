"""MadhuTwin API: the doctor-facing interface to the virtual patients.

    uvicorn twin.service.app:app --port 8000

Every request is written to an audit trail (who, what, when, purpose), as the DPDP Act's
accountability principle expects; patient data is synthetic or de-identified research data.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from twin.ehr.fhir import risk_assessment
from twin.paths import ARTIFACTS, ROOT
from twin.physiology.foods import FOODS

from . import assistant
from .store import Store

DEMO_DIR = Path(os.environ.get("MADHUTWIN_DEMO", ARTIFACTS / "demo"))
RESULTS_DIR = Path(os.environ.get("MADHUTWIN_RESULTS", ARTIFACTS / "results"))
DIST = ROOT / "dashboard" / "dist"

app = FastAPI(title="MadhuTwin API", version="0.1.0",
              description="Hybrid physiological + ML digital twin for Type 2 Diabetes (proof of concept, synthetic / de-identified data).")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
store = Store(DEMO_DIR, RESULTS_DIR)
AUDIT: deque = deque(maxlen=2000)


@app.middleware("http")
async def audit(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    if request.url.path.startswith("/api/patients"):
        parts = request.url.path.split("/")
        AUDIT.appendleft({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "user": request.headers.get("x-user", "demo-clinician"),
                          "purpose": request.headers.get("x-purpose", "treatment"), "method": request.method,
                          "resource": request.url.path, "patient": parts[3] if len(parts) > 3 else None,
                          "status": response.status_code, "ms": round((time.time() - t0) * 1000)})
    return response


def _patient(pid: str):
    try:
        return store.get(pid)
    except KeyError:
        raise HTTPException(404, f"unknown patient {pid}") from None


@app.get("/api/meta")
def meta():
    return {"name": "MadhuTwin", "version": app.version, "default_clock": store.default_clock, "max_clock": store.max_clock,
            "patients": len(store.order), "disclaimer": "Research prototype. Synthetic and de-identified data only. Not a medical device."}


@app.get("/api/patients")
def patients(clock: float | None = None):
    c = store.default_clock if clock is None else clock
    rows = [store.get(pid).summary(c) for pid in store.order]
    return sorted(rows, key=lambda r: -_priority(r))


def _priority(r: dict) -> float:
    """Current lows first, then predicted lows, current very-high, predicted new spikes, current high."""
    g = r["glucose"]
    spike = (r["risk_spike"] or 0) if (g is None or g <= 180) else 0
    score = max(spike, 2.5 * (r["risk_hypo"] or 0))
    if g is not None and g < 70:
        score = max(score, 3.0)
    elif g is not None and g > 250:
        score = max(score, 1.2)
    elif g is not None and g > 180:
        score = max(score, 0.8)
    return score


@app.get("/api/patients/{pid}")
def patient(pid: str):
    p = _patient(pid)
    b = p.bundle
    return {"id": p.id, "story": b["story"], "source": b["source"], "display": b["display"], "ehr": b["ehr"], "twin": b["twin"],
            "gate": b["predictions"].get("gate"), "consent": {"status": "granted", "purpose": ["treatment", "care-coordination"],
                                                                "artefact": f"consent-{p.id}", "expires": "2027-09-01"}}


@app.get("/api/patients/{pid}/state")
def state(pid: str, clock: float | None = None, history_h: float = 24):
    return _patient(pid).state(store.default_clock if clock is None else clock, history_h)


@app.get("/api/patients/{pid}/agp")
def agp(pid: str, clock: float | None = None, days: int = 14):
    return _patient(pid).agp(store.default_clock if clock is None else clock, days)


@app.get("/api/patients/{pid}/insights")
def insights(pid: str, clock: float | None = None):
    p = _patient(pid)
    c = store.default_clock if clock is None else clock
    return {"insights": p.insights(c), "meal_ranking": p.meal_ranking(), "walk": p.walk_benefit()}


class WhatIf(BaseModel):
    clock: float | None = None
    meal: dict | None = Field(default=None, examples=[{"food": "rice_sambar", "portion": 1.0, "in_min": 0}])
    walk: dict | None = Field(default=None, examples=[{"minutes": 15, "delay_min": 20}])
    insulin: dict | None = Field(default=None, examples=[{"drug": "insulin_rapid", "units": 4, "in_min": 0}])
    sleep_hours: float | None = None
    illness: bool = False
    horizon: int = 240


@app.post("/api/patients/{pid}/whatif")
def whatif(pid: str, body: WhatIf):
    p = _patient(pid)
    c = store.default_clock if body.clock is None else body.clock
    return p.what_if(c, meal=body.meal, walk=body.walk, insulin=body.insulin, sleep_hours=body.sleep_hours,
                     illness=body.illness, horizon=min(max(body.horizon, 60), 480))


class Ask(BaseModel):
    question: str
    clock: float | None = None


@app.post("/api/patients/{pid}/ask")
def ask(pid: str, body: Ask):
    p = _patient(pid)
    c = store.default_clock if body.clock is None else body.clock
    try:
        from . import llm
        if llm.available():
            return llm.answer(p, c, body.question)
    except Exception as e:  # never let the optional LLM path break the offline answer
        res = assistant.answer(p, c, body.question)
        res["llm_error"] = type(e).__name__
        return res
    return assistant.answer(p, c, body.question)


@app.get("/api/patients/{pid}/fhir")
def fhir(pid: str, clock: float | None = None):
    p = _patient(pid)
    f = DEMO_DIR / f"{pid}.fhir.json"
    bundle = json.load(open(f)) if f.exists() else {"resourceType": "Bundle", "type": "collection", "entry": []}
    st = p.state(store.default_clock if clock is None else clock, history_h=1)
    fc = st["forecast"]
    if fc:
        ra = risk_assessment(pid, fc["anchor_time"], fc["spike"] or 0, fc["hypo"] or 0, fc["q50"], fc.get("why_spike"),
                             tz="+05:30" if p.bundle["source"] == "synthetic" else "+00:00")
        bundle = dict(bundle, entry=bundle["entry"] + [{"fullUrl": f"urn:uuid:{ra['id']}", "resource": ra}])
    return JSONResponse(bundle, media_type="application/fhir+json")


@app.get("/api/foods")
def foods():
    return [f.as_dict() for f in FOODS.values()]


@app.get("/api/evidence")
def evidence():
    return store.evidence()


@app.get("/api/audit")
def audit_log(limit: int = 100):
    return list(AUDIT)[:limit]


@app.websocket("/ws/live")
async def live(ws: WebSocket, clock: float | None = None, speed: float = 60.0, interval: float = 1.0):
    """Stream the demo clock: each tick advances `speed` patient-minutes per `interval` seconds."""
    await ws.accept()
    c = store.default_clock if clock is None else clock
    try:
        while True:
            await ws.send_json({"clock": c, "patients": [store.get(pid).summary(c) for pid in store.order]})
            await asyncio.sleep(interval)
            c = (c + speed) % store.max_clock
    except WebSocketDisconnect:
        return


if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")
