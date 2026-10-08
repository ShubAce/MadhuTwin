"""Experiment 6: "the twin learns you", with the whole twin personalised on k days.

For each of 50 unseen synthetic people with T2D (the same sample as exp_extra.py) and each
k in {0, 1, 2, 3, 5, 7} days:

* the mechanistic twin is fitted on the first k days (k = 0: the EHR-prior twin) and synced,
* TwinNet is fine-tuned on the same k days. Day 0 is the population TwinNet with the physics
  input switched off (its modality dropout supports this): a twin that has not been fitted yet is
  not fed to a network trained on fitted twins. The day-0 score with the EHR-prior twin switched
  on is kept as `day0_with_prior_twin` for transparency,

and both are scored on days 8-14. Updates the learning curve in artifacts/results/synthetic/extra.json.

    python scripts/exp_learning.py [--workers 6]
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from twin.data.physics import twin_features
from twin.data.windows import G_SCALE
from twin.eval.experiments import strict_json
from twin.models.twinnet import TwinNet, TwinNetConfig, Windows, personalise, predict
from twin.paths import ARTIFACTS, SYNTHETIC

OUT = ARTIFACTS / "results" / "synthetic"
DAYS = (0, 1, 2, 3, 5, 7)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _twin_job(args):
    arr, row, series, events = args
    return {k: twin_features(arr, row, series, events, k * 1440) for k in DAYS}


def main(curve_patients: int = 50, workers: int = 6) -> None:
    torch.set_num_threads(8)
    arrays = pickle.load(open(ARTIFACTS / "datasets" / "synthetic.pkl", "rb"))
    split = json.load(open(OUT / "split.json"))
    test = [a for a in arrays if a.pid in set(split["test"])]
    del arrays
    rng = np.random.default_rng(0)  # identical sample to exp_extra.py
    t2d = [a for a in test if a.status == "t2d"]
    sample = [t2d[i] for i in rng.choice(len(t2d), min(curve_patients, len(t2d)), replace=False)]
    ids = [a.pid for a in sample]
    full = SYNTHETIC / "full"
    static = pd.read_parquet(full / "static.parquet").set_index("patient_id")
    series = pd.read_parquet(full / "series.parquet", filters=[("patient_id", "in", ids)])
    events = pd.read_parquet(full / "events.parquet")
    events = events[events.patient_id.isin(ids)]

    log(f"refitting and syncing the mechanistic twin with {DAYS} days for {len(sample)} patients ({workers} workers)")
    jobs = [(a, static.loc[a.pid].to_dict() | {"patient_id": a.pid}, series[series.patient_id == a.pid], events[events.patient_id == a.pid])
            for a in sample]
    with ProcessPoolExecutor(workers) as ex:
        feats = list(ex.map(_twin_job, jobs))

    ck = torch.load(ARTIFACTS / "models" / "twinnet_synthetic.pt", weights_only=False)
    net = TwinNet(TwinNetConfig(**ck["cfg"]))
    net.load_state_dict(ck["state"])
    log("fine-tuning TwinNet on the same days and scoring days 8-14")
    curve, day0_prior = [], [0.0, 0]
    no_physics = TwinNetConfig(**{**ck["cfg"], "use_physics": False})
    for k in DAYS:
        acc = {key: [0.0, 0] for key in ("rmse_60", "rmse_120", "physics_rmse_60", "physics_rmse_120")}
        for a, f in zip(sample, feats, strict=True):
            phys, ukf, k_bins = f[k]
            b = copy.copy(a)  # same CGM, wearables and labels; only the twin's knowledge differs
            b.physics, b.ukf, b.extra = phys, ukf, {}
            wa = Windows([b], after_calibration=True)  # always days 8-14
            if k == 0:
                p = predict(net, wa, modality_override=no_physics)
                e = predict(net, wa)["q"][:, 23, 1] * G_SCALE - wa.y[:, 23] * G_SCALE  # same day 0, EHR-prior twin switched on
                day0_prior[0] += float(np.nansum(e**2))
                day0_prior[1] += int(np.isfinite(e).sum())
            else:
                p = predict(personalise(net, b, max_bin=k_bins, min_windows=20), wa)
            ya = wa.now[:, None] + wa.y * G_SCALE
            for h, col in ((60, 11), (120, 23)):
                for key, pred in ((f"rmse_{h}", wa.now + p["q"][:, col, 1] * G_SCALE), (f"physics_rmse_{h}", wa.now + wa.physics[:, col] * G_SCALE)):
                    e = pred - ya[:, col]
                    acc[key][0] += float(np.nansum(e**2))
                    acc[key][1] += int(np.isfinite(e).sum())
        curve.append({"days": k, **{key: float(np.sqrt(s / max(n, 1))) for key, (s, n) in acc.items()}})
        log(f"  {k} days: MadhuTwin RMSE60 {curve[-1]['rmse_60']:.2f} RMSE120 {curve[-1]['rmse_120']:.2f} · "
            f"mechanistic twin alone RMSE120 {curve[-1]['physics_rmse_120']:.2f}")

    path = OUT / "extra.json"
    extra = json.load(open(path))
    extra["learning_curve"] = curve
    extra["learning_curve_definition"] = ("whole twin personalised on k days: mechanistic fit + UKF sync + TwinNet fine-tuning; "
                                          "day 0 = population TwinNet with the physics input off")
    extra["day0_with_prior_twin"] = {"rmse_120": float(np.sqrt(day0_prior[0] / max(day0_prior[1], 1)))}
    extra["curve_patients"] = len(sample)
    json.dump(strict_json(extra), open(path, "w"), indent=1, default=float)
    log("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--patients", type=int, default=50)
    a = ap.parse_args()
    main(a.patients, a.workers)
