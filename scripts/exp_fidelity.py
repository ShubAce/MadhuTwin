"""Experiment 7: 24-hour fidelity of the personalised twin (the trust check behind the therapy simulator).

For every day after each patient's calibration period, the synced twin is set to its filtered
state at the start of the day and replays the day open-loop with the meals, doses and activity
that were logged. The simulated day is compared with the CGM: mean glucose, time in range,
time below 70 and mean absolute error, and each day is rated good / fair / poor exactly as in
the dashboard.

    python scripts/exp_fidelity.py [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from twin.eval.experiments import strict_json
from twin.paths import ARTIFACTS, PROCESSED, SYNTHETIC
from twin.physiology.model import Physiology
from twin.record import PatientRecord
from twin.service.store import fidelity_rating
from twin.twin import DigitalTwin

DAY = 1440


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _patient_days(args) -> list[dict]:
    arr, row, series, events = args
    fit = arr.extra.get("fit")
    if not fit:
        return []
    rec = PatientRecord.from_frames(row, series, events)
    tw = DigitalTwin(record=rec, params=Physiology.from_records([fit["params"]]), tau_scale=float(fit["tau_scale"]))
    tw.params.renal_thr_base[:] = rec.renal_thr_base
    tw.sync()
    offset = int((arr.start - rec.start).total_seconds() // 60)
    calib = arr.calib_bins * 5 + offset
    days = []
    for m in range(calib + DAY, rec.T + 1, DAY):  # each full day after calibration
        meals, doses, mets = tw.template_day(m)
        sim = tw.therapy_day(m - DAY, [{"doses": doses, "therapy": rec.therapy}], meals=meals, mets=mets)[0]
        sel = (rec.cgm_t >= m - DAY) & (rec.cgm_t < m)
        if sel.sum() < 48:  # at least 12 h of readings
            continue
        s, o = sim[rec.cgm_t[sel] - (m - DAY)], rec.cgm[sel]
        tir = lambda x: float(np.mean((x >= 70) & (x <= 180)) * 100)  # noqa: E731
        step = float(np.median(np.diff(rec.cgm_t[sel]))) if sel.sum() > 1 else 5.0
        d = {"patient": arr.pid, "mean_twin": float(s.mean()), "mean_cgm": float(o.mean()), "tir_twin": tir(s), "tir_cgm": tir(o),
             "below_70_twin_min": float((s < 70).sum() * step), "below_70_cgm_min": float((o < 70).sum() * step),
             "mae": float(np.abs(s - o).mean())}
        d["rating"] = fidelity_rating({k: round(v, 1) if isinstance(v, float) else v for k, v in d.items()})
        days.append(d)
    return days


def _load(name: str, sample: int | None = None) -> list:
    arrays = pickle.load(open(ARTIFACTS / "datasets" / f"{name}.pkl", "rb"))
    if name == "synthetic":
        split = json.load(open(ARTIFACTS / "results" / "synthetic" / "split.json"))
        test = [a for a in arrays if a.pid in set(split["test"])]
        rng = np.random.default_rng(0)
        t2d = [a for a in test if a.status == "t2d"]
        arrays = [t2d[i] for i in rng.choice(len(t2d), min(sample or 50, len(t2d)), replace=False)]
        d = SYNTHETIC / "full"
        ids = [a.pid for a in arrays]
        st = pd.read_parquet(d / "static.parquet").set_index("patient_id")
        se = pd.read_parquet(d / "series.parquet", filters=[("patient_id", "in", ids)])
        ev = pd.read_parquet(d / "events.parquet")
    else:
        st = pd.read_parquet(PROCESSED / f"{name}_static.parquet").set_index("patient_id")
        se = pd.read_parquet(PROCESSED / f"{name}_series.parquet")
        ev = pd.read_parquet(PROCESSED / f"{name}_events.parquet")
    se_g, ev_g = dict(tuple(se.groupby("patient_id"))), dict(tuple(ev.groupby("patient_id")))
    empty_ev = ev.iloc[:0]
    return [(a, st.loc[a.pid].to_dict() | {"patient_id": a.pid}, se_g[a.pid], ev_g.get(a.pid, empty_ev)) for a in arrays if a.pid in se_g]


def summarise(days: list[dict]) -> dict:
    df = pd.DataFrame(days)
    d_mean = (df.mean_twin - df.mean_cgm).abs()
    d_tir = (df.tir_twin - df.tir_cgm).abs()
    low = df[df.below_70_cgm_min >= 15]
    return {"patients": int(df.patient.nunique()), "days": int(len(df)),
            "mean_abs_error_median": float(df.mae.median()), "mean_glucose_error_median": float(d_mean.median()),
            "tir_error_median_pp": float(d_tir.median()),
            "rating_pct": {k: float((df.rating == k).mean() * 100) for k in ("good", "fair", "poor")},
            "days_with_lows": int(len(low)),
            "lows_reproduced_pct": float((low.below_70_twin_min > 0).mean() * 100) if len(low) else None}


def main(workers: int = 6) -> None:
    out = {}
    for name in ("synthetic", "cgmacros", "bigideas", "shanghai"):
        jobs = _load(name)
        log(f"{name}: replaying every post-calibration day of {len(jobs)} patients")
        with ProcessPoolExecutor(workers) as ex:
            days = [d for ds in ex.map(_patient_days, jobs, chunksize=2) for d in ds]
        out[name] = summarise(days)
        s = out[name]
        log(f"  {s['days']} days: median MAE {s['mean_abs_error_median']:.1f} mg/dL, mean-glucose error {s['mean_glucose_error_median']:.1f}, "
            f"TIR error {s['tir_error_median_pp']:.1f} pp; good/fair/poor {s['rating_pct']['good']:.0f}/{s['rating_pct']['fair']:.0f}/"
            f"{s['rating_pct']['poor']:.0f}%; lows reproduced on {s['lows_reproduced_pct']}% of {s['days_with_lows']} days with lows")
    json.dump(strict_json(out), open(ARTIFACTS / "results" / "fidelity.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    main(ap.parse_args().workers)
