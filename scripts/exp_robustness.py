"""Experiment 8: robustness of the deployed TwinNet to missing data at inference time.

Real deployments in India will often lack a smartwatch, a complete EHR or meal logs, and CGM
readings drop out. One trained model (synthetic, with modality dropout) is scored on the
synthetic test patients (days 8-14) with:

* a whole modality removed: wearables, EHR, meals/doses, or the physics twin
* random loss of 10 / 25 / 50% of past CGM readings (the current reading is kept; the synced
  twin's state is not recomputed, so this isolates the network's sensitivity)

    python scripts/exp_robustness.py
"""

from __future__ import annotations

import json
import pickle
import time

import numpy as np
import torch

from twin.data.windows import CH, G_SCALE, HISTORY
from twin.eval.experiments import strict_json
from twin.eval.metrics import event_scores
from twin.models.twinnet import TwinNet, TwinNetConfig, Windows, _to_t, apply_modalities
from twin.paths import ARTIFACTS

OUT = ARTIFACTS / "results"
CGM_IDX = [CH["cgm"], CH["cgm_avail"], CH["cgm_slope"]]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(net: TwinNet, w: Windows, cfg: TwinNetConfig, cgm_loss: float = 0.0, seed: int = 0, batch: int = 4096) -> dict:
    rng = np.random.default_rng(seed)
    net.eval()
    qs, ev = [], []
    with torch.no_grad():
        for s in range(0, len(w), batch):
            b = apply_modalities(w.batch(np.arange(s, min(s + batch, len(w)))), cfg)
            if cgm_loss > 0:
                lost = rng.random((len(b["dyn"]), HISTORY - 1)) < cgm_loss  # never the current reading
                d = b["dyn"]
                for c in CGM_IDX:
                    d[:, : HISTORY - 1, c] = np.where(lost, 0.0, d[:, : HISTORY - 1, c])
            t = _to_t(b, "cpu")
            q, logits, _ = net(t["dyn"], t["static"], t["physics"], t["ukf"], t["phys_avail"], t["static_avail"])
            qs.append(q.numpy())
            ev.append(torch.sigmoid(logits).numpy())
    q, e = np.concatenate(qs), np.concatenate(ev)
    y = w.y * G_SCALE
    err = lambda h: float(np.sqrt(np.nanmean((q[:, h, 1] * G_SCALE - y[:, h]) ** 2)))  # noqa: E731
    ok = w.spike_ok > 0
    return {"rmse_30": err(5), "rmse_60": err(11), "rmse_120": err(23),
            "spike_auroc": event_scores(w.spike[ok] > 0, e[ok, 0])["auroc"],
            "hypo_auroc": event_scores(w.hypo[w.hypo_ok > 0] > 0, e[w.hypo_ok > 0, 1])["auroc"]}


def main() -> None:
    torch.set_num_threads(16)
    arrays = pickle.load(open(ARTIFACTS / "datasets" / "synthetic.pkl", "rb"))
    split = json.load(open(OUT / "synthetic" / "split.json"))
    test = [a for a in arrays if a.pid in set(split["test"])]
    del arrays
    ck = torch.load(ARTIFACTS / "models" / "twinnet_synthetic.pt", weights_only=False)
    base = TwinNetConfig(**ck["cfg"])
    net = TwinNet(base)
    net.load_state_dict(ck["state"])
    w = Windows(test, after_calibration=True)
    log(f"{len(test)} test patients, {len(w)} forecasts")
    rows = []
    for name, cfg in (("All data", base), ("No smartwatch (wearables missing)", TwinNetConfig(**{**ck["cfg"], "use_wearable": False})),
                      ("No EHR", TwinNetConfig(**{**ck["cfg"], "use_static": False})),
                      ("No meal or dose logs", TwinNetConfig(**{**ck["cfg"], "use_events": False})),
                      ("No physics twin", TwinNetConfig(**{**ck["cfg"], "use_physics": False})),
                      ("Only CGM (no smartwatch, EHR, logs or twin)", TwinNetConfig(**{**ck["cfg"], "use_wearable": False, "use_static": False,
                                                                                     "use_events": False, "use_physics": False}))):
        rows.append({"condition": name, **run(net, w, cfg)})
        log(f"  {name:45s} RMSE60 {rows[-1]['rmse_60']:.2f} RMSE120 {rows[-1]['rmse_120']:.2f} spike AUROC {rows[-1]['spike_auroc']:.3f}")
    gaps = []
    for loss in (0.1, 0.25, 0.5):
        gaps.append({"cgm_readings_lost_pct": loss * 100, **run(net, w, base, cgm_loss=loss)})
        log(f"  CGM readings lost {loss:.0%}: RMSE60 {gaps[-1]['rmse_60']:.2f} RMSE120 {gaps[-1]['rmse_120']:.2f}")
    json.dump(strict_json({"modalities": rows, "cgm_dropout": gaps, "n_forecasts": len(w), "model": "TwinNet hybrid (population, synthetic)"}),
              open(OUT / "robustness.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
