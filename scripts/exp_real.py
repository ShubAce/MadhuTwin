"""Experiments 2-3: real-world validation (CGMacros, 5-fold patient CV) and external validation
(ShanghaiT2DM: different country, CGM device, diet, therapy mix, and no wearables).

Training regimes compared for TwinNet:
    zero-shot      trained on the synthetic cohort only
    real-only      trained from scratch on real training folds
    sim-to-real    pretrained on synthetic, fine-tuned on real training folds

Also trains the production models used by the API: TwinNet (synthetic -> CGMacros + Shanghai),
LightGBM (synthetic + CGMacros), with conformal intervals calibrated on out-of-fold real data.

    python scripts/exp_real.py
"""

from __future__ import annotations

import copy
import json
import pickle
import time

import numpy as np
import torch

from twin.data.windows import G_SCALE, HORIZONS, H
from twin.eval.experiments import (
    baselines,
    best_threshold,
    event_rows,
    forecast_rows,
    group_folds,
    linear_event_score,
    physics_event_score,
    slope_now,
)
from twin.models.gbm import GBMForecaster
from twin.models.tabular import stack
from twin.models.twinnet import TwinNet, TwinNetConfig, Windows, predict, train
from twin.paths import ARTIFACTS
from twin.service import DEMO_REAL_HOLDOUT

OUT = ARTIFACTS / "results"
MODELS = ARTIFACTS / "models"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load(name: str):
    with open(ARTIFACTS / "datasets" / f"{name}.pkl", "rb") as fh:
        return pickle.load(fh)


def load_net(path) -> tuple[TwinNet, dict]:
    ck = torch.load(path, weights_only=False)
    net = TwinNet(TwinNetConfig(**ck["cfg"]))
    net.load_state_dict(ck["state"])
    return net, ck


def expand(qarr: np.ndarray) -> np.ndarray:
    full = np.full((len(qarr), H), np.nan)
    for i, h in enumerate(HORIZONS):
        full[:, h - 1] = qarr[:, i]
    return full


def finetune(base: TwinNet, w_tr: Windows, epochs: int = 4, lr: float = 5e-4) -> TwinNet:
    net = copy.deepcopy(base)
    train(net, w_tr, None, epochs=epochs, samples_per_epoch=min(len(w_tr), 80_000), lr=lr, log=lambda m: None)
    return net


def scratch(w_tr: Windows, epochs: int = 8) -> TwinNet:
    net = TwinNet(TwinNetConfig())
    train(net, w_tr, None, epochs=epochs, samples_per_epoch=min(len(w_tr), 80_000), lr=2e-3, log=lambda m: None)
    return net


def cross_validate(name: str, arrays: list, pre: TwinNet, gbm_syn: GBMForecaster, gbm_extra_train: list | None = None,
                   k: int = 5, regimes=("zero-shot", "real-only", "sim-to-real")) -> dict:
    """Out-of-fold predictions for every method; evaluation on days after personalisation."""
    folds = group_folds(arrays, k=k, seed=0)
    pool: dict[str, list] = {}
    events_pool: dict[str, dict[str, list]] = {"spike": {}, "hypo": {}}
    alert_pool: dict[str, dict[str, list]] = {"spike": {}, "hypo": {}}
    y_all, groups, test_arrays, index_parts = [], [], [], []
    offset = 0
    for f, (tr, te) in enumerate(folds):
        log(f"  {name} fold {f + 1}/{k}: {len(tr)} train / {len(te)} test recordings")
        w_tr, w_te = Windows(tr), Windows(te, after_calibration=True)
        if len(w_te) == 0:
            continue
        y = w_te.now[:, None] + w_te.y * G_SCALE
        slope = slope_now(te, after_calibration=True)
        phys_prior = np.concatenate([a.physics_prior[a.anchors >= a.calib_bins] for a in te])
        preds = baselines(w_te.now, slope, w_te.physics, phys_prior)
        ev_scores = {e: {"Linear trend": linear_event_score(w_te.now, slope, e),
                         "Twin (personalised)": physics_event_score(w_te.now, w_te.physics, e)} for e in ("spike", "hypo")}

        d_tr, d_te = stack(tr + (gbm_extra_train or [])), stack(te, after_calibration=True)
        g = GBMForecaster.fit(d_tr, max_rows=300_000)
        gp = g.predict(d_te["X"], d_te["now"])
        preds["LightGBM (real-trained)"] = expand(gp["quantiles"][:, :, 1])
        gz = gbm_syn.predict(d_te["X"], d_te["now"])
        preds["LightGBM (synthetic-only)"] = expand(gz["quantiles"][:, :, 1])
        for e in ("spike", "hypo"):
            # a fold can lack enough events to train a classifier: keep arrays aligned with NaN
            ev_scores[e]["LightGBM (real-trained)"] = gp.get(e, np.full(len(w_te), np.nan))
            if e in gp:
                gtr = g.predict(d_tr["X"], d_tr["now"])[e]
                ok_tr = d_tr[f"{e}_ok"]
                alert_pool[e].setdefault("LightGBM (real-trained)", []).append(_rescale(gp[e], best_threshold(d_tr[e][ok_tr], gtr[ok_tr])))

        nets = {}
        if "zero-shot" in regimes:
            nets["TwinNet zero-shot (synthetic)"] = pre
        if "real-only" in regimes:
            nets["TwinNet real-only"] = scratch(w_tr)
        if "sim-to-real" in regimes:
            nets["TwinNet sim-to-real"] = finetune(pre, w_tr)
        for nm, net in nets.items():
            p = predict(net, w_te)
            if nm == "TwinNet sim-to-real":
                p_tr = predict(net, w_tr)
                for e in ("spike", "hypo"):
                    ok_tr = getattr(w_tr, f"{e}_ok") > 0
                    thr = best_threshold(getattr(w_tr, e)[ok_tr] > 0, p_tr[e][ok_tr])
                    alert_pool[e].setdefault(nm, []).append(_rescale(p[e], thr))
            q = w_te.now[:, None, None] + p["q"] * G_SCALE
            preds[nm] = q[..., 1]
            pool.setdefault(nm + "|lo", []).append(q[..., 0])
            pool.setdefault(nm + "|hi", []).append(q[..., 2])
            for e in ("spike", "hypo"):
                ev_scores[e][nm] = p[e]
        for nm, v in preds.items():
            pool.setdefault(nm, []).append(v)
        for e in ("spike", "hypo"):
            for nm, v in ev_scores[e].items():
                events_pool[e].setdefault(nm, []).append(v)
        y_all.append(y)
        groups.append(w_te.group)
        index_parts.append({"patient": w_te.patient + offset, "anchor": w_te.anchor, "group": w_te.group,
                            "spike": w_te.spike, "spike_ok": w_te.spike_ok, "hypo": w_te.hypo, "hypo_ok": w_te.hypo_ok})
        test_arrays += te
        offset += len(te)

    y = np.concatenate(y_all)
    grp = np.concatenate(groups)
    preds = {nm: np.concatenate(v) for nm, v in pool.items() if "|" not in nm}
    quant = {nm: (np.concatenate(pool[nm + "|lo"]), np.concatenate(pool[nm + "|hi"])) for nm in preds if nm + "|lo" in pool}
    idx = {k2: np.concatenate([p[k2] for p in index_parts]) for k2 in index_parts[0]}
    ev_rows = []
    for e in ("spike", "hypo"):
        scores = {nm: np.concatenate(v) for nm, v in events_pool[e].items()}
        # operating points use thresholds chosen on each fold's *training* patients (rescaled to 0.5)
        alerts = {nm: np.concatenate(v) for nm, v in alert_pool[e].items() if len(v) == len(index_parts)}
        ev_rows += event_rows(test_arrays, idx, scores, e, {nm: 0.5 for nm in alerts}, alert_scores=alerts)
    return {"forecast": forecast_rows(y, preds, grp, quantiles=quant), "events": ev_rows, "y": y, "preds": preds,
            "quant": quant, "n_recordings": len(test_arrays), "n_anchors": int(len(y))}


def _rescale(prob: np.ndarray, thr: float) -> np.ndarray:
    """Monotone map sending a fold's threshold to 0.5, so pooled folds share one operating point."""
    thr = float(np.clip(thr, 1e-6, 1 - 1e-6))
    return np.where(prob < thr, 0.5 * prob / thr, 0.5 + 0.5 * (prob - thr) / (1 - thr))


def conformal_from(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = 0.2) -> list[float]:
    e = np.maximum(lo - y, y - hi)
    return [float(np.quantile(e[:, h][np.isfinite(e[:, h])], 1 - alpha)) for h in range(e.shape[1])]


def main() -> None:
    torch.set_num_threads(16)
    (OUT / "real").mkdir(parents=True, exist_ok=True)
    syn = load("synthetic")
    split = json.load(open(OUT / "synthetic" / "split.json"))
    syn_train = [a for a in syn if a.pid in set(split["train"])]
    pre, ck = load_net(MODELS / "twinnet_synthetic.pt")
    gbm_syn = pickle.load(open(MODELS / "gbm_synthetic.pkl", "rb"))
    cgm, sh = load("cgmacros"), load("shanghai")

    log("CGMacros 5-fold cross-validation")
    res_c = cross_validate("CGMacros", cgm, pre, gbm_syn)
    log("ShanghaiT2DM 5-fold cross-validation (external population, no wearables)")
    res_s = cross_validate("Shanghai", sh, pre, gbm_syn, regimes=("zero-shot", "sim-to-real"))

    # production models: synthetic-pretrained TwinNet fine-tuned on all real data
    log("training production models")
    real_train = [a for a in cgm + sh if a.pid not in DEMO_REAL_HOLDOUT]
    final = finetune(pre, Windows(real_train), epochs=5)
    lo, hi = res_c["quant"]["TwinNet sim-to-real"]
    q_conf = conformal_from(res_c["y"], lo, hi)
    torch.save({"state": final.state_dict(), "cfg": final.cfg.__dict__, "conformal_q": q_conf,
                "trained_on": ["synthetic (700 patients)", "CGMacros + ShanghaiT2DM (minus demo hold-outs)"],
                "holdout": list(DEMO_REAL_HOLDOUT)},
               MODELS / "twinnet_final.pt")
    gbm_final = GBMForecaster.fit(stack(syn_train[:300] + [a for a in cgm if a.pid not in DEMO_REAL_HOLDOUT]), max_rows=500_000)
    pickle.dump(gbm_final, open(MODELS / "gbm_final.pkl", "wb"))

    def strip(r: dict) -> dict:
        return {k: v for k, v in r.items() if k in ("forecast", "events", "n_recordings", "n_anchors")}

    json.dump({"cgmacros": strip(res_c), "shanghai": strip(res_s), "conformal_q_real": q_conf},
              open(OUT / "real" / "results.json", "w"), indent=1, default=float)
    np.savez_compressed(OUT / "real" / "clarke_cgmacros_60.npz", ref=res_c["y"][:, 11].astype(np.float32),
                        twinnet=res_c["preds"]["TwinNet sim-to-real"][:, 11].astype(np.float32))
    for nm, res in (("CGMacros", res_c), ("Shanghai", res_s)):
        print(f"\n{nm}: {res['n_recordings']} recordings, {res['n_anchors']} evaluation anchors")
        for r in res["forecast"]:
            print(f"  {r['method']:30s} RMSE 30/60/120: {r['rmse_30']:.1f} / {r['rmse_60']:.1f} / {r['rmse_120']:.1f}   A+B@60 {r['clarkeAB_60']:.1f}%")
        for r in res["events"]:
            extra = f"  detected {r['detected_pct']:.0f}% lead {r['median_lead_min']:.0f} min FA/day {r['false_alerts_per_day']:.2f}" if "median_lead_min" in r else ""
            print(f"  {r['event']:5s} {r['method']:30s} AUROC {r['auroc']:.3f} AUPRC {r['auprc']:.3f}{extra}")
    log("done")


if __name__ == "__main__":
    main()
