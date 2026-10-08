"""Experiment 5 (synthetic test patients): personalisation, ensemble, clinical utility, fairness.

* TwinNet personalised: the population model fine-tuned on each patient's own days 1-7
* MadhuTwin ensemble: personalised TwinNet averaged with LightGBM; its event probabilities are
  recalibrated (Platt) on the validation patients, never on test patients
* TwinNet-only learning curve (the full-twin curve is scripts/exp_learning.py)
* Subgroup error (sex, age, BMI by Asian cut-offs, status, therapy, region)
* Calibration and decision curves for spike alerts; detection at <= 1 false alert per day

    python scripts/exp_extra.py
"""

from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd
import torch

from twin.data.windows import G_SCALE, HORIZONS, H
from twin.eval.clinical import (
    calibration_curve,
    decision_curve,
    platt_apply,
    platt_fit,
    sensitivity_at_false_alerts,
    subgroup_table,
)
from twin.eval.experiments import forecast_rows, strict_json
from twin.eval.metrics import event_scores
from twin.models.gbm import GBMForecaster
from twin.models.tabular import stack
from twin.models.twinnet import TwinNet, TwinNetConfig, Windows, personalise, predict
from twin.paths import ARTIFACTS, SYNTHETIC

OUT = ARTIFACTS / "results" / "synthetic"
BINS_PER_DAY = 288


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(curve_patients: int = 50) -> None:
    torch.set_num_threads(16)
    arrays = pickle.load(open(ARTIFACTS / "datasets" / "synthetic.pkl", "rb"))
    split = json.load(open(OUT / "split.json"))
    test = [a for a in arrays if a.pid in set(split["test"])]
    val = [a for a in arrays if a.pid in set(split["val"])]
    del arrays
    ck = torch.load(ARTIFACTS / "models" / "twinnet_synthetic.pt", weights_only=False)
    net = TwinNet(TwinNetConfig(**ck["cfg"]))
    net.load_state_dict(ck["state"])
    q_conf = np.asarray(ck["conformal_q"])
    gbm: GBMForecaster = pickle.load(open(ARTIFACTS / "models" / "gbm_synthetic.pkl", "rb"))

    w = Windows(test, after_calibration=True)
    y = w.now[:, None] + w.y * G_SCALE
    log(f"{len(test)} test patients, {len(w)} forecasts (days 8-14)")
    base = predict(net, w)
    d = stack(test, after_calibration=True)
    gp = gbm.predict(d["X"], d["now"])
    gb = np.full((len(w), H), np.nan)
    for i, h in enumerate(HORIZONS):
        gb[:, h - 1] = gp["quantiles"][:, i, 1]

    log("personalising TwinNet on each patient's days 1-7")
    parts = [predict(personalise(net, a), Windows([a], after_calibration=True)) for a in test]
    pers = {k: np.concatenate([p[k] for p in parts]) for k in ("q", "spike", "hypo")}

    log("recalibration map for the ensemble's risks, fitted on the validation patients")
    wv = Windows(val, after_calibration=True)
    pv = [predict(personalise(net, a), Windows([a], after_calibration=True)) for a in val]
    dv = stack(val, after_calibration=True)
    gv = gbm.predict(dv["X"], dv["now"])
    recal = {}
    for e in ("spike", "hypo"):
        ens_v = (np.concatenate([p[e] for p in pv]) + gv[e]) / 2
        okv = getattr(wv, f"{e}_ok") > 0
        recal[e] = platt_fit(getattr(wv, e)[okv] > 0, ens_v[okv])
    log(f"  maps (a, b): {recal}")
    qp = w.now[:, None, None] + pers["q"] * G_SCALE
    ens = np.where(np.isfinite(gb), (qp[..., 1] + gb) / 2, qp[..., 1])
    preds = {"Persistence": np.repeat(w.now[:, None], H, axis=1), "TwinNet hybrid": w.now[:, None] + base["q"][..., 1] * G_SCALE,
             "LightGBM fusion": gb, "TwinNet personalised": qp[..., 1], "MadhuTwin ensemble": ens}
    quant = {"TwinNet personalised": (qp[..., 0] - q_conf, qp[..., 2] + q_conf), "MadhuTwin ensemble": (qp[..., 0] - q_conf, qp[..., 2] + q_conf)}
    f_rows = forecast_rows(y, preds, w.group, quantiles=quant)

    idx = {"patient": w.patient, "anchor": w.anchor, "group": w.group, "spike": w.spike, "spike_ok": w.spike_ok,
           "hypo": w.hypo, "hypo_ok": w.hypo_ok}
    ev = []
    for e in ("spike", "hypo"):
        scores = {"TwinNet hybrid": base[e], "LightGBM fusion": gp.get(e), "TwinNet personalised": pers[e],
                  "MadhuTwin ensemble": platt_apply((pers[e] + gp[e]) / 2, recal[e])}
        ok = idx[f"{e}_ok"] > 0
        for nm, sc in scores.items():
            if sc is None:
                continue
            r = {"event": e, "method": nm, **event_scores(idx[e][ok] > 0, sc[ok])}
            fa = sensitivity_at_false_alerts(test, idx, sc, e, 1.0)
            r["caught_at_1fa_pct"], r["lead_at_1fa_min"] = fa.get("detected_pct"), fa.get("median_lead_min")
            ev.append(r)
    raw = (pers["spike"] + gp["spike"]) / 2
    sc_ens = platt_apply(raw, recal["spike"])
    ok = idx["spike_ok"] > 0
    clinical = {"spike_calibration": calibration_curve(idx["spike"][ok] > 0, sc_ens[ok]),
                "spike_decision_curve": decision_curve(idx["spike"][ok] > 0, sc_ens[ok]),
                "spike_calibration_uncalibrated": calibration_curve(idx["spike"][ok] > 0, raw[ok]),
                "spike_decision_curve_uncalibrated": decision_curve(idx["spike"][ok] > 0, raw[ok]),
                "recalibration": recal}
    rc = ARTIFACTS / "models" / "recalibration.json"
    maps = json.load(open(rc)) if rc.exists() else {}
    json.dump(maps | {"synthetic": recal}, open(rc, "w"), indent=1)

    log("subgroups")
    profiles = {p["patient_id"]: p for p in json.load(open(SYNTHETIC / "full" / "profiles.json", encoding="utf-8"))}
    rows = []
    for i, a in enumerate(test):
        m = w.patient == i
        pr = profiles[a.pid]
        th = pr["therapy_flags"]
        err60, err120 = (ens[m, 11] - y[m, 11]), (ens[m, 23] - y[m, 23])
        rows.append({"sex": pr["sex"], "age": pd.cut([pr["age"]], [0, 45, 60, 120], labels=["<45", "45-60", ">60"])[0],
                     "bmi": pd.cut([pr["bmi"]], [0, 23, 27.5, 100], labels=["<23", "23-27.5", ">=27.5"])[0],
                     "status": pr["status"], "region": pr["region"],
                     "therapy": "insulin" if th["insulin"] else ("sulfonylurea" if th["sulfonylurea"] else ("other drugs" if pr["meds"] else "none")),
                     "se60": float(np.nansum(err60**2)), "n60": int(np.isfinite(err60).sum()),
                     "se120": float(np.nansum(err120**2)), "n120": int(np.isfinite(err120).sum())})
    subgroups = subgroup_table(pd.DataFrame(rows), ["sex", "age", "bmi", "status", "therapy", "region"])

    log(f"learning curve: personal fine-tuning with 0-7 days, {curve_patients} T2D patients")
    rng = np.random.default_rng(0)
    t2d = [a for a in test if a.status == "t2d"]
    sample = [t2d[i] for i in rng.choice(len(t2d), min(curve_patients, len(t2d)), replace=False)]
    curve = []
    for k in (0, 1, 2, 3, 5, 7):
        se60 = se120 = n60 = n120 = 0.0
        for a in sample:
            wa = Windows([a], after_calibration=True)
            model = net if k == 0 else personalise(net, a, max_bin=k * BINS_PER_DAY, min_windows=20)
            p = predict(model, wa)
            ya = wa.now[:, None] + wa.y * G_SCALE
            pa = wa.now[:, None] + p["q"][..., 1] * G_SCALE
            e60, e120 = pa[:, 11] - ya[:, 11], pa[:, 23] - ya[:, 23]
            se60 += np.nansum(e60**2); n60 += np.isfinite(e60).sum()  # noqa: E702
            se120 += np.nansum(e120**2); n120 += np.isfinite(e120).sum()  # noqa: E702
        curve.append({"days": k, "rmse_60": float(np.sqrt(se60 / n60)), "rmse_120": float(np.sqrt(se120 / n120))})
        log(f"  {k} days: RMSE60 {curve[-1]['rmse_60']:.2f}  RMSE120 {curve[-1]['rmse_120']:.2f}")

    old = json.load(open(OUT / "extra.json")) if (OUT / "extra.json").exists() else {}
    keep = {k: v for k, v in old.items() if k in ("learning_curve", "learning_curve_definition", "day0_with_prior_twin")}
    json.dump(strict_json({"forecast": f_rows, "events": ev, "clinical": clinical, "subgroups": subgroups, "learning_curve": curve,
               "curve_patients": len(sample), **keep, "learning_curve_twinnet_only": curve}), open(OUT / "extra.json", "w"), indent=1, default=float)
    for r in f_rows:
        print(f"  {r['method']:22s} RMSE 30/60/120: {r['rmse_30']:.1f} / {r['rmse_60']:.1f} / {r['rmse_120']:.1f}")
    for r in ev:
        print(f"  {r['event']:5s} {r['method']:22s} AUROC {r['auroc']:.3f} AUPRC {r['auprc']:.3f} caught@1FA {r['caught_at_1fa_pct']}")
    log("done")


if __name__ == "__main__":
    main()
