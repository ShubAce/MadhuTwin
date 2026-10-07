"""Experiment 4: CGM-light mode, the affordability scenario for India.

The patient wears a CGM for one week; the twin is personalised on it. In week two the sensor
is removed: only 4 fingerstick checks a day (pre-breakfast, pre-lunch, pre-dinner, bedtime;
glucometer error ~5%) and the smartwatch remain. The synced twin keeps estimating continuous
glucose. We compare its estimate with the true glucose of held-out synthetic patients, against
the naive alternative of carrying the last fingerstick forward, and check how well it recovers
time-in-range (the metric clinicians manage to).

    python scripts/exp_cgm_light.py
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from twin.eval.metrics import clarke_summary, mard
from twin.paths import ARTIFACTS, SYNTHETIC
from twin.record import PatientRecord
from twin.twin import DigitalTwin

CHECK_TIMES = (7 * 60 + 30, 13 * 60, 19 * 60 + 30, 22 * 60 + 30)
CALIB_DAYS = 7


def _one(args):
    pid, row, series, events, truth, seed = args
    rng = np.random.default_rng(seed)
    rec = PatientRecord.from_frames(row, series, events)
    tw = DigitalTwin.from_record(rec)
    calib = CALIB_DAYS * 1440
    tw.personalize(calib)

    # week 2: replace CGM with fingersticks drawn from the true plasma glucose
    t_true = ((truth["ts"] - rec.start).dt.total_seconds() // 60).to_numpy().astype(int)
    g_true = truth["glucose_true"].to_numpy()
    keep = rec.cgm_t < calib
    fs_t, fs_y = [], []
    for day in range(CALIB_DAYS, int(rec.T // 1440)):
        for m in CHECK_TIMES:
            t = day * 1440 + m + int(rng.integers(-20, 20))
            k = np.searchsorted(t_true, t)
            if 0 <= k < len(g_true):
                fs_t.append(int(t_true[k]) + 2)
                fs_y.append(float(g_true[k] * (1 + rng.normal(0, 0.05))))
    light = PatientRecord(**{**rec.__dict__, "cgm_t": np.r_[rec.cgm_t[keep], fs_t].astype(int), "cgm": np.r_[rec.cgm[keep], fs_y]})
    tw.record = light
    trace = tw.sync()

    # evaluate every 30 min in week two against the true glucose
    sel = (t_true >= calib) & (np.arange(len(t_true)) % 6 == 0)
    t_eval, y_eval = t_true[sel] + 2, g_true[sel]
    k = np.clip(np.searchsorted(trace.t, t_eval, side="right") - 1, 0, len(trace.t) - 1)
    twin_est = trace.mean[k, 0]
    fs_t, fs_y = np.array(fs_t), np.array(fs_y)
    j = np.searchsorted(fs_t, t_eval, side="right") - 1
    carry = np.where(j >= 0, fs_y[np.clip(j, 0, None)], np.nan)
    tir = lambda x: float(np.mean((x >= 70) & (x <= 180)) * 100)  # noqa: E731
    return {"pid": pid, "y": y_eval, "twin": twin_est, "carry": carry,
            "tir_true": tir(y_eval), "tir_twin": tir(twin_est), "tir_carry": tir(carry[np.isfinite(carry)]),
            "tir_fingersticks": tir(fs_y)}


def main(workers: int = 4) -> None:
    # each worker holds the scientific stack (~0.5 GB); stay well inside 16 GB machines
    t0 = time.time()
    split = json.load(open(ARTIFACTS / "results" / "synthetic" / "split.json"))
    ids = split["test"]
    full = SYNTHETIC / "full"
    st = pd.read_parquet(full / "static.parquet").set_index("patient_id")
    se = pd.read_parquet(full / "series.parquet", filters=[("patient_id", "in", ids)])
    ev = pd.read_parquet(full / "events.parquet", filters=[("patient_id", "in", ids)])
    tr = pd.read_parquet(full / "truth.parquet", columns=["patient_id", "ts", "glucose_true"], filters=[("patient_id", "in", ids)])
    se_g, ev_g, tr_g = dict(tuple(se.groupby("patient_id"))), dict(tuple(ev.groupby("patient_id"))), dict(tuple(tr.groupby("patient_id")))
    jobs = [(pid, st.loc[pid].to_dict() | {"patient_id": pid}, se_g[pid], ev_g.get(pid, ev.iloc[:0]), tr_g[pid], i) for i, pid in enumerate(ids)]
    res = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(_one, j) for j in jobs]):
            res.append(f.result())
    y = np.concatenate([r["y"] for r in res])
    tw = np.concatenate([r["twin"] for r in res])
    ca = np.concatenate([r["carry"] for r in res])
    ok = np.isfinite(ca)
    within20 = lambda p: float(np.mean(np.abs(p[ok] - y[ok]) / y[ok] <= 0.2) * 100)  # noqa: E731
    tir_err = lambda k: float(np.mean([abs(r[k] - r["tir_true"]) for r in res]))  # noqa: E731
    out = {
        "n_patients": len(res), "n_points": int(ok.sum()), "fingersticks_per_day": len(CHECK_TIMES),
        "twin": {"mard": mard(y[ok], tw[ok]), "within_20pct": within20(tw), "clarke": clarke_summary(y[ok], tw[ok]),
                 "tir_abs_error_pp": tir_err("tir_twin")},
        "carry_forward": {"mard": mard(y[ok], ca[ok]), "within_20pct": within20(ca), "clarke": clarke_summary(y[ok], ca[ok]),
                          "tir_abs_error_pp": tir_err("tir_carry")},
        "fingersticks_only_tir_abs_error_pp": tir_err("tir_fingersticks"),
    }
    (ARTIFACTS / "results" / "cgm_light.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
