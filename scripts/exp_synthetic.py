"""Experiment 1 (synthetic cohort, all modalities): benchmark, stream-fusion ablation, illness detection.

    python scripts/exp_synthetic.py [--twinnet-ablations] [--epochs 8]

Evaluation is on held-out *patients* (20%), and only on days 8-14, i.e. after each patient's
twin was personalised on days 1-7. Writes artifacts/results/synthetic/*.json and model files.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd
import torch

from twin.data.windows import G_SCALE, HORIZONS, H
from twin.eval.experiments import (
    baselines,
    best_threshold,
    event_rows,
    forecast_rows,
    linear_event_score,
    physics_event_score,
    slope_now,
    split_patients,
    strict_json,
)
from twin.eval.metrics import event_scores, rmse
from twin.models.gbm import GBMForecaster
from twin.models.tabular import GROUPS, stack
from twin.models.twinnet import TwinNet, TwinNetConfig, Windows, predict, train
from twin.paths import ARTIFACTS, SYNTHETIC

OUT = ARTIFACTS / "results" / "synthetic"
MODELS = ARTIFACTS / "models"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def conformal_q(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Conformalised quantile regression: per-horizon widening that guarantees ~(1-alpha) coverage."""
    e = np.maximum(lo - y, y - hi)
    q = np.empty(e.shape[1])
    for h in range(e.shape[1]):
        v = e[:, h][np.isfinite(e[:, h])]
        q[h] = np.quantile(v, min(1.0, (1 - alpha) * (1 + 1 / max(len(v), 1))))
    return q


def twinnet_abs(pred: dict, now: np.ndarray, q_conf: np.ndarray | None = None):
    q = now[:, None, None] + pred["q"] * G_SCALE
    lo, med, hi = q[..., 0], q[..., 1], q[..., 2]
    if q_conf is not None:
        lo, hi = lo - q_conf[None, :], hi + q_conf[None, :]
    return lo, med, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--samples", type=int, default=250_000)
    ap.add_argument("--twinnet-ablations", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smoke test on 80 patients")
    args = ap.parse_args()
    torch.set_num_threads(16)
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    log("loading synthetic dataset")
    with open(ARTIFACTS / "datasets" / "synthetic.pkl", "rb") as fh:
        arrays = pickle.load(fh)
    if args.quick:
        arrays = arrays[:80]
    tr, va, te = split_patients(arrays, seed=0)
    log(f"split: {len(tr)} train / {len(va)} val / {len(te)} test patients")
    json.dump({"train": [a.pid for a in tr], "val": [a.pid for a in va], "test": [a.pid for a in te]},
              open(OUT / "split.json", "w"), indent=1)

    w_tr, w_va, w_te = Windows(tr), Windows(va), Windows(te, after_calibration=True)
    y_te = w_te.now[:, None] + w_te.y * G_SCALE
    idx = {"patient": w_te.patient, "anchor": w_te.anchor, "group": w_te.group, "spike": w_te.spike,
           "spike_ok": w_te.spike_ok, "hypo": w_te.hypo, "hypo_ok": w_te.hypo_ok}
    log(f"anchors: train {len(w_tr)}, val {len(w_va)}, test (days 8-14) {len(w_te)}")

    # ---------------- baselines
    slope = slope_now(te, after_calibration=True)
    preds = baselines(w_te.now, slope, w_te.physics, np.concatenate([a.physics_prior[a.anchors >= a.calib_bins] for a in te]))

    # ---------------- gradient boosting (all streams)
    log("tabular features")
    d_tr, d_va, d_te = stack(tr), stack(va), stack(te, after_calibration=True)
    assert len(d_te["now"]) == len(w_te.now)
    log("fitting LightGBM (3 quantiles x 4 horizons + spike/hypo classifiers)")
    gbm = GBMForecaster.fit(d_tr)
    pickle.dump(gbm, open(MODELS / "gbm_synthetic.pkl", "wb"))
    g_te, g_va = gbm.predict(d_te["X"], d_te["now"]), gbm.predict(d_va["X"], d_va["now"])

    def expand(qarr: np.ndarray) -> np.ndarray:  # (n, 4 horizons) -> (n, H) with NaN between horizons
        full = np.full((len(qarr), H), np.nan)
        for i, h in enumerate(HORIZONS):
            full[:, h - 1] = qarr[:, i]
        return full

    preds["LightGBM fusion"] = expand(g_te["quantiles"][:, :, 1])
    quant = {"LightGBM fusion": (expand(g_te["quantiles"][:, :, 0]), expand(g_te["quantiles"][:, :, 2]))}

    # ---------------- TwinNet (full hybrid)
    log("training TwinNet (full hybrid)")
    net = TwinNet(TwinNetConfig())
    hist = train(net, w_tr, w_va, epochs=args.epochs, samples_per_epoch=args.samples, log=log)
    p_va = predict(net, w_va)
    lo_va, _, hi_va = twinnet_abs(p_va, w_va.now)
    y_va = w_va.now[:, None] + w_va.y * G_SCALE
    q_conf = conformal_q(y_va, lo_va, hi_va)
    torch.save({"state": net.state_dict(), "cfg": net.cfg.__dict__, "conformal_q": q_conf.tolist(), "history": hist},
               MODELS / "twinnet_synthetic.pt")
    p_te = predict(net, w_te)
    lo, med, hi = twinnet_abs(p_te, w_te.now, q_conf)
    preds["TwinNet hybrid"] = med
    quant["TwinNet hybrid"] = (lo, hi)

    log("forecast metrics")
    f_rows = forecast_rows(y_te, preds, w_te.group, quantiles=quant)

    # ---------------- events
    log("event metrics")
    results_events = []
    for ev in ("spike", "hypo"):
        ok_va = getattr(w_va, f"{ev}_ok") > 0
        scores_te = {
            "Linear trend": linear_event_score(w_te.now, slope, ev),
            "Twin (personalised)": physics_event_score(w_te.now, w_te.physics, ev),
            "LightGBM fusion": g_te.get(ev, np.full(len(w_te), np.nan)),
            "TwinNet hybrid": p_te[ev],
        }
        thresholds = {}
        if ev in g_va:
            thresholds["LightGBM fusion"] = best_threshold(getattr(w_va, ev)[ok_va] > 0, g_va[ev][ok_va])
        thresholds["TwinNet hybrid"] = best_threshold(getattr(w_va, ev)[ok_va] > 0, p_va[ev][ok_va])
        results_events += event_rows(te, idx, scores_te, ev, thresholds)

    # ---------------- stream-fusion ablation (LightGBM, retrained per configuration)
    log("ablation: LightGBM per stream combination")
    configs = {
        "CGM only": ("cgm",),
        "CGM + EHR": ("cgm", "ehr"),
        "CGM + wearables": ("cgm", "wearable"),
        "CGM + meals/meds": ("cgm", "events"),
        "CGM + wearables + meals/meds": ("cgm", "wearable", "events"),
        "Both streams (dynamic + EHR)": ("cgm", "wearable", "events", "ehr"),
        "Both streams + physics twin": GROUPS,
    }
    abl = []
    for name, groups in configs.items():
        dtr, dte = stack(tr, groups), stack(te, groups, after_calibration=True)
        m = GBMForecaster.fit(dtr, quantiles=(0.5,), max_rows=400_000)
        pr = m.predict(dte["X"], dte["now"])
        row = {"config": name, "groups": list(groups), "n_features": len(dtr["names"])}
        for i, h in enumerate(HORIZONS):
            row[f"rmse_{h * 5}"] = rmse(dte["now"] + dte["Y"][:, h - 1], pr["quantiles"][:, i, 0])
        for ev in ("spike", "hypo"):
            if ev in pr:
                ok = dte[f"{ev}_ok"]
                s = event_scores(dte[ev][ok], pr[ev][ok])
                row[f"{ev}_auroc"], row[f"{ev}_auprc"] = s["auroc"], s["auprc"]
        abl.append(row)
        log(f"  {name}: RMSE60={row['rmse_60']:.2f} RMSE120={row['rmse_120']:.2f} spike AUROC={row.get('spike_auroc', float('nan')):.3f}")

    tn_abl = []
    if args.twinnet_ablations:
        log("ablation: TwinNet variants")
        variants = {"TwinNet without physics": dict(use_physics=False), "TwinNet without EHR": dict(use_static=False),
                    "TwinNet without wearables": dict(use_wearable=False), "TwinNet without meals/meds": dict(use_events=False)}
        for name, kw in variants.items():
            m = TwinNet(TwinNetConfig(**kw))
            train(m, w_tr, w_va, epochs=args.epochs, samples_per_epoch=args.samples, log=log)
            pr = predict(m, w_te)
            _, med_v, _ = twinnet_abs(pr, w_te.now)
            row = {"config": name, **{f"rmse_{h * 5}": rmse(y_te[:, h - 1], med_v[:, h - 1]) for h in HORIZONS}}
            for ev in ("spike", "hypo"):
                ok = getattr(w_te, f"{ev}_ok") > 0
                s = event_scores(getattr(w_te, ev)[ok] > 0, pr[ev][ok])
                row[f"{ev}_auroc"], row[f"{ev}_auprc"] = s["auroc"], s["auprc"]
            tn_abl.append(row)
            log(f"  {name}: RMSE60={row['rmse_60']:.2f} RMSE120={row['rmse_120']:.2f}")

    # ---------------- illness detection from the synced twin (UKF insulin-sensitivity drift)
    log("illness detection")
    truth = pd.read_parquet(SYNTHETIC / "full" / "truth.parquet", columns=["patient_id", "ts", "illness"])
    truth["day"] = truth["ts"].dt.normalize()
    ill = truth.groupby(["patient_id", "day"]).illness.mean().gt(0.5)
    rows_ill = []
    for a in te:
        si = a.extra.get("si_multiplier_daily", [])
        days = pd.date_range(a.start.normalize(), periods=len(si), freq="1D")
        for d, v in zip(days, si, strict=True):
            if (a.pid, d) in ill.index and np.isfinite(v):
                rows_ill.append({"patient": a.pid, "day": d, "si_mult": v, "ill": bool(ill[(a.pid, d)]), "after_calib": a.calib_bins * 5 <= (d - a.start).total_seconds() / 60})
    ill_df = pd.DataFrame(rows_ill)
    ill_res = event_scores(ill_df["ill"].to_numpy(), -np.log(ill_df["si_mult"].to_numpy()))
    ill_res["n_ill_days"] = int(ill_df["ill"].sum())
    ill_res["n_days"] = int(len(ill_df))
    ill_res["mean_si_mult_ill"] = float(ill_df.loc[ill_df.ill, "si_mult"].mean())
    ill_res["mean_si_mult_well"] = float(ill_df.loc[~ill_df.ill, "si_mult"].mean())
    ill_df.to_csv(OUT / "illness_days.csv", index=False)
    log(f"  illness-day AUROC {ill_res['auroc']:.3f} on {ill_res['n_ill_days']} ill / {ill_res['n_days']} days")

    # ---------------- gate behaviour + sample trajectories for figures
    gate = p_te["gate"].mean(axis=0).tolist()
    sample = {}
    for a_i in np.unique(w_te.patient)[:6]:
        m = np.flatnonzero(w_te.patient == a_i)[::4][:200]
        sample[te[a_i].pid] = {"anchor": w_te.anchor[m].tolist(), "now": w_te.now[m].tolist(),
                               "y": np.nan_to_num(y_te[m], nan=-1).tolist(), "med": med[m].tolist(),
                               "lo": lo[m].tolist(), "hi": hi[m].tolist(), "twin": preds["Twin (personalised)"][m].tolist()}
    clarke_pairs = {"ref": y_te[:, 11].tolist(), "TwinNet hybrid": med[:, 11].tolist(), "Persistence": preds["Persistence"][:, 11].tolist()}

    json.dump(strict_json({"forecast": f_rows, "events": results_events, "ablation_gbm": abl, "ablation_twinnet": tn_abl,
               "illness_detection": ill_res, "gate_by_horizon": gate, "twinnet_history": hist,
               "n_test_patients": len(te), "n_test_anchors": int(len(w_te))}), open(OUT / "results.json", "w"), indent=1, default=float)
    json.dump(sample, open(OUT / "sample_trajectories.json", "w"))
    np.savez_compressed(OUT / "clarke_60.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in clarke_pairs.items()})
    log("done")
    for r in f_rows:
        print(f"  {r['method']:24s} RMSE 30/60/120: {r['rmse_30']:.1f} / {r['rmse_60']:.1f} / {r['rmse_120']:.1f}   Clarke A+B@60: {r['clarkeAB_60']:.1f}%")
    for r in results_events:
        print(f"  {r['event']:5s} {r['method']:22s} AUROC {r['auroc']:.3f} AUPRC {r['auprc']:.3f}"
              + (f"  detected {r['detected_pct']:.0f}% lead {r['median_lead_min']:.0f} min FA/day {r['false_alerts_per_day']:.2f}" if 'median_lead_min' in r else ""))


if __name__ == "__main__":
    main()
