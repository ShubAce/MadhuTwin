"""Build the demo cohort shown in the doctor dashboard.

Each virtual patient becomes one JSON bundle with: EHR, personalised twin parameters, wearable
streams, out-of-sample TwinNet forecasts (with conformal intervals) and spike/hypo
probabilities at every anchor, plus plain-language reasons from LightGBM SHAP values.

Synthetic patients are drawn from the held-out *test* split and selected for clinically
distinct stories; real-world recordings are excluded from production-model training.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import torch

from twin.data.windows import G_SCALE, H, PatientArrays
from twin.ehr.codes import CONDITIONS, DRUGS, LABS
from twin.ehr.fhir import minimal_bundle, patient_bundle
from twin.ehr.population import Profile
from twin.models.gbm import GBMForecaster
from twin.models.tabular import features
from twin.models.twinnet import TwinNet, TwinNetConfig, Windows, predict
from twin.paths import ARTIFACTS, PROCESSED, SYNTHETIC

from . import DEMO_REAL_HOLDOUT
from .explain import reasons

DEMO = ARTIFACTS / "demo"
ALERT_THRESHOLDS = {"spike": 0.5, "hypo": 0.3}


def _r(x, nd: int = 1):
    x = np.asarray(x, dtype=float)
    return np.where(np.isfinite(x), np.round(x, nd), np.nan).tolist()


def _num(row: pd.Series, key: str) -> float | None:
    return None if key not in row or pd.isna(row.get(key)) else round(float(row[key]), 1)


def _clean(o):
    """Replace NaN/inf with None for strict JSON."""
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return _clean(o.item())
    return o


# ---------------------------------------------------------------------------- selection
def select_synthetic(test: list[PatientArrays], profiles: dict, events: pd.DataFrame, truth: pd.DataFrame) -> dict[str, str]:
    """story -> patient id, chosen by clinical criteria among held-out test patients."""
    def after(a, mask):
        return mask & (a.anchors >= a.calib_bins)

    rows = []
    # only illness that falls in the replay window (after each twin's 7-day calibration) is visible in the demo
    window = truth[truth.ts >= truth.ts.min().normalize() + pd.Timedelta(days=7)]
    ill_days = window[window.illness > 0.5].groupby("patient_id").ts.agg(lambda s: s.dt.normalize().nunique())
    sweets = events[(events.kind == "meal") & (events.label == "sweet")].assign(day=lambda d: d.ts.dt.normalize())
    fest = sweets.groupby(["patient_id", "day"]).size().groupby("patient_id").max()
    for a in test:
        pr = profiles[a.pid]
        th = pr["therapy_flags"]
        c = a.cgm[np.isfinite(a.cgm)]
        rows.append(dict(
            pid=a.pid, status=a.status, a1c=pr["labs_latent"]["a1c_target"], age=pr["age"], region=pr["region"],
            insulin=th["insulin"] and any(m["drug"] == "insulin_premix_30_70" for m in pr["meds"]), su=th["sulfonylurea"],
            sglt2=th["sglt2"], dpp4=th["dpp4"], egfr=pr["labs_latent"]["egfr"], tir=float(np.mean((c >= 70) & (c <= 180))),
            hypos=int(np.sum(after(a, a.hypo & a.hypo_ok))), spikes=int(np.sum(after(a, a.spike & a.spike_ok))),
            ill=int(ill_days.get(a.pid, 0)), fest=int(fest.get(a.pid, 0)), dinner=pr["lifestyle"]["dinner_min"],
            sleep=pr["lifestyle"]["sleep_mean_h"],
        ))
    df = pd.DataFrame(rows)
    chosen: dict[str, str] = {}

    def pick(story: str, frame: pd.DataFrame, by: str, ascending: bool = False) -> None:
        frame = frame[~frame.pid.isin(chosen.values())]
        if len(frame):
            chosen[story] = frame.sort_values(by, ascending=ascending).iloc[0].pid

    t2d = df[df.status == "t2d"]
    pick("Premixed insulin with hypoglycaemia", t2d[t2d.insulin], "hypos")
    pick("Sulfonylurea-related low glucose", t2d[t2d.su & ~t2d.insulin], "hypos")
    pick("Viral illness: insulin resistance detected by the twin", t2d[t2d.ill > 0], "ill")
    pick("Festival feast: sweets and post-meal spikes", t2d[t2d.fest >= 3], "spikes")
    pick("Poorly controlled, frequent spikes", t2d, "a1c")
    pick("Well controlled on metformin + DPP-4 inhibitor", t2d[t2d.dpp4 & (t2d.a1c < 7.2)], "tir")
    pick("SGLT2 inhibitor with reduced kidney function", t2d[t2d.sglt2], "egfr", ascending=True)
    pre = df[(df.status == "prediabetes") & (df.age < 45)]
    pick("Prediabetes: late dinners and short sleep", pre, "dinner")
    return chosen


# ----------------------------------------------------------------------------- bundles
def _ehr_synthetic(pr: dict, static: pd.Series, labhist: pd.DataFrame) -> dict:
    labs = []
    for key, (loinc, display, unit) in LABS.items():
        v = static.get(key) if key in static else pr["labs_latent"].get(key)
        if v is None or not np.isfinite(float(v)):
            continue
        labs.append({"key": key, "loinc": loinc, "display": display, "value": round(float(v), 1), "unit": unit})
    return {
        "status": pr["status"], "diabetes_years": pr["diabetes_years"], "bmi": pr["bmi"], "weight_kg": pr["weight_kg"],
        "height_cm": pr["height_cm"], "waist_cm": pr["waist_cm"],
        "conditions": [{"key": k, "snomed": CONDITIONS[k][0], "display": CONDITIONS[k][1], "years": y}
                       for k, y in sorted(pr["conditions"].items(), key=lambda kv: -kv[1])],
        "medications": [{"drug": m["drug"], "display": DRUGS[m["drug"]][1], "atc": DRUGS[m["drug"]][0],
                         "class": DRUGS[m["drug"]][2], "dose": m["dose"], "unit": DRUGS[m["drug"]][3],
                         "times": [f"{t // 60:02d}:{t % 60:02d}" for t in m["times"]]} for m in pr["meds"]],
        "labs": labs,
        "lab_history": labhist.assign(date=labhist.date.dt.strftime("%Y-%m-%d"))[["date", "hba1c_pct", "fpg_mgdl", "weight_kg", "sbp", "dbp"]].to_dict("records"),
        "family_history": pr["family_history"],
        "genetics": {"TCF7L2_rs7903146": ["CC (no risk allele)", "CT (1 risk allele)", "TT (2 risk alleles)"][pr["genetics"]["TCF7L2_rs7903146_T_alleles"]],
                     "prs_z": pr["genetics"]["prs_t2d_z"]},
        "vitals": {"sbp": pr["cardio"]["sbp"], "dbp": pr["cardio"]["dbp"], "resting_hr": round(pr["cardio"]["rhr"])},
        "lifestyle": {k: pr["lifestyle"][k] for k in ("vegetarian", "activity", "morning_walk", "post_dinner_walk", "chai_per_day", "chai_sugar")},
        "adherence": round(pr["adherence"], 2),
    }


def _ehr_real(static: pd.Series) -> dict:
    labs = []
    for key, (loinc, display, unit) in LABS.items():
        v = static.get(key)
        if v is not None and pd.notna(v):
            labs.append({"key": key, "loinc": loinc, "display": display, "value": round(float(v), 1), "unit": unit})
    meds = [{"drug": k, "display": k.replace("on_", "").replace("_", " ").title(), "class": k[3:]} for k in
            ("on_metformin", "on_sulfonylurea", "on_dpp4", "on_sglt2", "on_insulin") if float(static.get(k) or 0) > 0]
    conds = [{"key": static.get("status"), "display": {"t2d": "Type 2 diabetes mellitus", "prediabetes": "Prediabetes",
                                                       "normal": "No diabetes"}.get(static.get("status"), "")}]
    if float(static.get("hypertension") or 0) > 0:
        conds.append({"key": "hypertension", "display": "Hypertensive disorder"})
    return {"status": static.get("status"), "diabetes_years": static.get("diabetes_years"), "bmi": static.get("bmi"),
            "weight_kg": static.get("weight_kg"), "height_cm": static.get("height_cm"), "conditions": conds,
            "medications": meds, "labs": labs, "lab_history": [], "family_history": [], "genetics": None, "vitals": {}}


RECORD_STATIC = ("patient_id", "status", "age", "sex", "weight_kg", "fpg_mgdl", "fasting_insulin_uU", "hba1c_pct", "egfr",
                 "on_metformin", "on_sulfonylurea", "on_dpp4", "on_sglt2", "on_insulin")


def build_bundle(a: PatientArrays, story: str, display: dict, ehr: dict, series: pd.DataFrame, events: pd.DataFrame,
                 net: TwinNet, q_conf: np.ndarray, gbm: GBMForecaster, static_row: dict) -> dict:
    w = Windows([a])
    pr = predict(net, w)
    q = w.now[:, None, None] + pr["q"] * G_SCALE
    lo, med, hi = q[..., 0] - q_conf[None, :], q[..., 1], q[..., 2] + q_conf[None, :]
    X, names = features(a)
    vals = [dict(zip(names, row, strict=True)) for row in X]
    why = {}
    for ev in ("spike", "hypo"):
        if ev in gbm.event_models:
            why[ev] = [reasons(c, v, ev) for c, v in zip(gbm.explain(X, ev, top=8), vals, strict=True)]

    # alerts with hysteresis: raise at the threshold, clear below 70% of it, and stay quiet for
    # 60 min after an alert clears, so a flickering probability does not page a clinician repeatedly
    alerts = []
    for ev, thr in ALERT_THRESHOLDS.items():
        p = pr[ev]
        start, active, quiet_until = [], False, -1
        for i, b in enumerate(a.anchors):
            if active and p[i] < 0.7 * thr:
                active, quiet_until = False, b + 12
            elif not active and p[i] >= thr and b >= quiet_until:
                active = True
                start.append(i)
        for i in start:
            traj = med[i]
            peak_k = int(np.argmax(traj) if ev == "spike" else np.argmin(traj))
            alerts.append({"bin": int(a.anchors[i]), "kind": ev, "prob": round(float(p[i]), 2),
                           "predicted_value": round(float(traj[peak_k])), "in_min": (peak_k + 1) * 5,
                           "why": why.get(ev, [[]] * len(p))[i]})
    alerts.sort(key=lambda r: r["bin"])

    s = series.sort_values("ts").set_index("ts").reindex(pd.date_range(a.start, periods=len(a.cgm), freq="5min"))
    ev_rows = []
    for _, e in events.sort_values("ts").iterrows():
        b = int((e["ts"] - a.start).total_seconds() // 300)
        if 0 <= b < len(a.cgm):
            ev_rows.append({"bin": b, "kind": e["kind"], "label": str(e["label"]), "food": e.get("food") if isinstance(e.get("food"), str) else None,
                            **{k: _num(e, k) for k in ("carbs", "protein", "fat", "fiber", "gi", "amount")}})
    fit = a.extra.get("fit", {})
    params = fit.get("params") or {}
    return _clean({
        "id": a.pid, "source": a.source, "story": story, "display": display, "ehr": ehr,
        "static": {k: (static_row.get(k) if not isinstance(static_row.get(k), float) or np.isfinite(static_row.get(k)) else None)
                   for k in RECORD_STATIC},
        "start": a.start.isoformat(), "step_min": 5, "n_bins": len(a.cgm), "calib_bins": int(a.calib_bins),
        "series": {"cgm": _r(a.cgm, 0), "hr": _r(s["hr"], 0), "steps": _r(s["steps"], 0), "hrv": _r(s["hrv_rmssd"], 0),
                   "sleep": _r(s["sleep_stage"], 0), "mets": _r(s["mets"], 1)},
        "events": ev_rows,
        "twin": {"params": {k: params.get(k) for k in ("Gb", "Ib", "SI", "beta", "dawn", "SG", "weight", "k_inc", "renal_thr_base")},
                 "tau_scale": fit.get("tau_scale"), "fit_rmse_prior": fit.get("rmse_prior"), "fit_rmse_personalised": fit.get("rmse_fit"),
                 "si_daily": a.extra.get("si_multiplier_daily", [])},
        "predictions": {"anchors": a.anchors.tolist(), "q10": [_r(r, 0) for r in lo], "q50": [_r(r, 0) for r in med],
                        "q90": [_r(r, 0) for r in hi], "twin": [_r(r, 0) for r in (w.now[:, None] + w.physics * G_SCALE)],
                        "spike": _r(pr["spike"], 3), "hypo": _r(pr["hypo"], 3), "gate": _r(pr["gate"].mean(axis=0), 2),
                        "why_spike": why.get("spike"), "why_hypo": why.get("hypo")},
        "alerts": alerts,
    })


def build_demo() -> list[dict]:
    DEMO.mkdir(parents=True, exist_ok=True)
    models = ARTIFACTS / "models"

    def load(net_file: str, gbm_file: str):
        ck = torch.load(models / net_file, weights_only=False)
        net = TwinNet(TwinNetConfig(**ck["cfg"]))
        net.load_state_dict(ck["state"])
        q = np.asarray(ck["conformal_q"], dtype=float)
        assert len(q) == H
        return net, q, pickle.load(open(models / gbm_file, "rb"))

    # Each patient is served by the model trained for its domain: synthetic patients by the
    # synthetic-trained TwinNet (they are held-out test patients), real recordings by the
    # sim-to-real production model (which excludes the demo hold-outs from its training).
    net_syn, q_syn, gbm_syn = load("twinnet_synthetic.pt", "gbm_synthetic.pkl")
    has_final = (models / "twinnet_final.pt").exists() and (models / "gbm_final.pkl").exists()
    net_real, q_real, gbm_real = load("twinnet_final.pt", "gbm_final.pkl") if has_final else (net_syn, q_syn, gbm_syn)
    print(f"demo models: synthetic patients -> twinnet_synthetic; real recordings -> {'twinnet_final' if has_final else 'twinnet_synthetic'}")

    index = []
    syn = pickle.load(open(ARTIFACTS / "datasets" / "synthetic.pkl", "rb"))
    split = json.load(open(ARTIFACTS / "results" / "synthetic" / "split.json"))
    test = [a for a in syn if a.pid in set(split["test"])]
    profiles = {p["patient_id"]: p for p in json.load(open(SYNTHETIC / "full" / "profiles.json", encoding="utf-8"))}
    full = SYNTHETIC / "full"
    events_all = pd.read_parquet(full / "events.parquet")
    truth = pd.read_parquet(full / "truth.parquet", columns=["patient_id", "ts", "illness"])
    chosen = select_synthetic(test, profiles, events_all, truth)
    ids = list(chosen.values())
    static = pd.read_parquet(full / "static.parquet").set_index("patient_id")
    series = pd.read_parquet(full / "series.parquet", filters=[("patient_id", "in", ids)])
    labhist = pd.read_parquet(full / "lab_history.parquet")
    by_id = {a.pid: a for a in test}
    for story, pid in chosen.items():
        pr = profiles[pid]
        disp = {"name": pr["name"], "age": pr["age"], "sex": pr["sex"], "city": pr["city"], "language": pr["language"],
                "region": pr["region"], "label": "Synthetic patient (India-calibrated)"}
        b = build_bundle(by_id[pid], story, disp, _ehr_synthetic(pr, static.loc[pid], labhist[labhist.patient_id == pid]),
                         series[series.patient_id == pid], events_all[events_all.patient_id == pid], net_syn, q_syn, gbm_syn,
                         static.loc[pid].to_dict() | {"patient_id": pid})
        json.dump(b, open(DEMO / f"{pid}.json", "w"), separators=(",", ":"))
        fb = patient_bundle(Profile.from_dict(pr), static.loc[pid], labhist, series[series.patient_id == pid])
        json.dump(fb, open(DEMO / f"{pid}.fhir.json", "w"), separators=(",", ":"))
        index.append({"id": pid, "story": story, "source": "synthetic"})
        print(f"  {pid}: {story} ({len(b['alerts'])} alerts)")

    for name, label in (("cgmacros", "Real-world data: CGMacros (PhysioNet, USA)"), ("shanghai", "External cohort: ShanghaiT2DM (China)")):
        arrs = {a.pid: a for a in pickle.load(open(ARTIFACTS / "datasets" / f"{name}.pkl", "rb"))}
        st = pd.read_parquet(PROCESSED / f"{name}_static.parquet").set_index("patient_id")
        se = pd.read_parquet(PROCESSED / f"{name}_series.parquet")
        ev = pd.read_parquet(PROCESSED / f"{name}_events.parquet")
        for pid in DEMO_REAL_HOLDOUT:
            if pid not in arrs:
                continue
            row = st.loc[pid]
            disp = {"name": f"Participant {pid}", "age": row.get("age"), "sex": row.get("sex"), "city": None,
                    "language": None, "label": label}
            story = "Real-world recording (held out from training)"
            b = build_bundle(arrs[pid], story, disp, _ehr_real(row), se[se.patient_id == pid], ev[ev.patient_id == pid], net_real, q_real, gbm_real,
                             row.to_dict() | {"patient_id": pid})
            json.dump(b, open(DEMO / f"{pid}.json", "w"), separators=(",", ":"))
            fb = minimal_bundle(pid, row.to_dict() | {"patient_id": pid}, se[se.patient_id == pid])
            json.dump(fb, open(DEMO / f"{pid}.fhir.json", "w"), separators=(",", ":"))
            index.append({"id": pid, "story": story, "source": name})
            print(f"  {pid}: {label} ({len(b['alerts'])} alerts)")
    json.dump(index, open(DEMO / "index.json", "w"), indent=1)
    return index
