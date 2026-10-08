"""Latency and throughput on a CPU (no GPU): what it costs to run MadhuTwin for a clinic's panel.

    python scripts/bench.py
"""

from __future__ import annotations

import json
import os
import pickle
import platform
import time

import numpy as np
import pandas as pd
import torch

from twin.eval.experiments import strict_json
from twin.models.tabular import stack
from twin.models.twinnet import TwinNet, TwinNetConfig, Windows, personalise, predict
from twin.paths import ARTIFACTS, SYNTHETIC
from twin.physiology.calibrate import fit_patient
from twin.record import PatientRecord
from twin.twin import DigitalTwin


def timed(f, repeat: int = 5) -> float:
    f()  # warm-up (Numba compilation, allocator)
    t0 = time.perf_counter()
    for _ in range(repeat):
        f()
    return (time.perf_counter() - t0) / repeat


def main() -> None:
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    arrays = pickle.load(open(ARTIFACTS / "datasets" / "synthetic.pkl", "rb"))
    split = json.load(open(ARTIFACTS / "results" / "synthetic" / "split.json"))
    a = next(x for x in arrays if x.pid in set(split["test"]) and x.status == "t2d")
    test = [x for x in arrays if x.pid in set(split["test"])][:20]
    del arrays
    ck = torch.load(ARTIFACTS / "models" / "twinnet_synthetic.pt", weights_only=False)
    net = TwinNet(TwinNetConfig(**ck["cfg"]))
    net.load_state_dict(ck["state"])
    gbm = pickle.load(open(ARTIFACTS / "models" / "gbm_synthetic.pkl", "rb"))
    w1 = Windows([a], after_calibration=True)
    wn = Windows(test, after_calibration=True)
    d = stack(test, after_calibration=True)

    full = SYNTHETIC / "full"
    st = pd.read_parquet(full / "static.parquet").set_index("patient_id")
    se = pd.read_parquet(full / "series.parquet", filters=[("patient_id", "==", a.pid)])
    ev = pd.read_parquet(full / "events.parquet")
    rec = PatientRecord.from_frames(st.loc[a.pid].to_dict() | {"patient_id": a.pid}, se, ev[ev.patient_id == a.pid])
    days = rec.T / 1440

    one = timed(lambda: predict(net, w1, idx=np.array([0])), 20)
    batch_t = timed(lambda: predict(net, wn), 2)
    gbm_t = timed(lambda: gbm.predict(d["X"], d["now"]), 2)
    pers_t = timed(lambda: personalise(net, a), 2)
    fit_t = timed(lambda: fit_patient(rec.truncate(7 * 1440)), 1)
    tw = DigitalTwin.from_record(rec)
    sync_t = timed(lambda: tw.sync(), 2)
    m = 9 * 1440
    _, doses, _ = tw.template_day(m)
    plans = [{"doses": doses}] * 10
    ther_t = timed(lambda: tw.therapy_day(m, plans), 5)

    out = {
        "cpu": platform.processor() or platform.machine(), "threads": torch.get_num_threads(),
        "twinnet_single_forecast_ms": one * 1000, "twinnet_throughput_per_s": len(wn) / batch_t,
        "lightgbm_throughput_per_s": len(d["X"]) / gbm_t,
        "personalise_twinnet_s": pers_t, "mechanistic_fit_7_days_s": fit_t,
        "ukf_sync_ms_per_day": sync_t / days * 1000, "therapy_sim_10_plans_24h_ms": ther_t * 1000,
        # compute-bound: every patient needs one TwinNet and one LightGBM forecast per 5-minute CGM reading
        "patients_refreshed_per_5_min": int(300 / (1 / (len(wn) / batch_t) + 1 / (len(d["X"]) / gbm_t))),
    }
    json.dump(strict_json(out), open(ARTIFACTS / "results" / "bench.json", "w"), indent=1)
    for k, v in out.items():
        print(f"  {k:32s} {v:.1f}" if isinstance(v, float) else f"  {k:32s} {v}")


if __name__ == "__main__":
    main()
