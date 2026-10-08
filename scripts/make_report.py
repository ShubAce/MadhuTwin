"""Generate report figures (docs/figures) and docs/evaluation_report.md from experiment outputs.

    python scripts/make_report.py

Chart conventions: one validated categorical palette (blue / orange / aqua, de-emphasis grey),
2 px lines, hairline solid grids, no dual axes, legends for >= 2 series, direct labels used sparingly.
"""

from __future__ import annotations

import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from twin.eval.metrics import clarke_summary  # noqa: E402
from twin.paths import ARTIFACTS, PROCESSED, ROOT, SYNTHETIC  # noqa: E402

FIG = ROOT / "docs" / "figures"
RES = ARTIFACTS / "results"
S1, S2, S3, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a9a8a1"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e7e6e0"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
METHOD_COLOR = {"TwinNet hybrid": S1, "TwinNet sim-to-real": S1, "LightGBM fusion": S2, "LightGBM (real-trained)": S2,
                "Twin (personalised)": S3, "Persistence": GREY, "MadhuTwin ensemble": S1, "TwinNet personalised": S1}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "semibold", "axes.titlesize": 11,
    "axes.titlecolor": INK, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 2,
})


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  docs/figures/{name}.png")


def fig_horizon(rows: list[dict], name: str, title: str, methods: list[str]) -> None:
    hz = [30, 60, 90, 120]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ends = {}
    for m in methods:
        r = next((x for x in rows if x["method"] == m), None)
        if r is None:
            continue
        y = [r[f"rmse_{h}"] for h in hz]
        ax.plot(hz, y, marker="o", ms=6, color=METHOD_COLOR.get(m, GREY), mec="white", mew=1.5, label=m)
        ends[m] = y[-1]
    # direct end labels only where they cannot collide; the legend carries the rest
    for m, v in ends.items():
        if all(abs(v - w) > 2.5 for k, w in ends.items() if k != m):
            ax.annotate(m, (hz[-1], v), xytext=(8, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.set_xticks(hz, [f"{h} min" for h in hz])
    ax.set_ylabel("RMSE (mg/dL)")
    ax.set_title(title, loc="left")
    ax.set_xlim(25, 150)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=9)
    save(fig, name)


def fig_ablation(abl: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    names = [r["config"] for r in abl]
    vals = [r["rmse_60"] for r in abl]
    colors = [GREY] * (len(abl) - 1) + [S1]
    ax.barh(names, vals, color=colors, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 0.2, i, f"{v:.1f}", va="center", color=INK2, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE at 60 min (mg/dL), lower is better")
    ax.set_title("Fusing both streams + the physics twin reduces error", loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.15)
    save(fig, "ablation_streams")


def fig_clarke(ref: np.ndarray, pred: np.ndarray, name: str, title: str) -> dict:
    m = np.isfinite(ref) & np.isfinite(pred)
    ref, pred = ref[m], pred[m]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ref), min(len(ref), 20000), replace=False)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.scatter(ref[idx], pred[idx], s=2, alpha=0.18, color=S1, linewidths=0)
    k = dict(color=INK2, lw=0.9)
    ax.plot([0, 400], [0, 400], color=MUTED, lw=0.8)
    ax.plot([0, 175 / 3], [70, 70], **k); ax.plot([175 / 3, 400 / 1.2], [70, 400], **k)  # noqa: E702
    ax.plot([70, 70], [84, 400], **k); ax.plot([70, 70], [0, 56], **k); ax.plot([70, 400], [56, 320], **k)  # noqa: E702
    ax.plot([180, 180], [0, 70], **k); ax.plot([180, 400], [70, 70], **k); ax.plot([240, 240], [70, 180], **k)  # noqa: E702
    ax.plot([240, 400], [180, 180], **k); ax.plot([130, 180], [0, 70], **k); ax.plot([70, 290], [180, 400], **k)  # noqa: E702
    ax.plot([0, 70], [180, 180], **k)
    for lab, (x, y) in {"A": (30, 15), "B": (370, 260), "B ": (280, 370), "C": (160, 370), "C ": (160, 15), "D": (30, 140), "D ": (370, 120), "E": (30, 370), "E ": (370, 15)}.items():
        ax.text(x, y, lab.strip(), color=INK2, fontsize=11, ha="center", va="center", fontweight="semibold")
    ax.set_xlim(0, 400); ax.set_ylim(0, 400)  # noqa: E702
    ax.set_xlabel("Measured glucose (mg/dL)")
    ax.set_ylabel("Predicted glucose (mg/dL)")
    z = clarke_summary(ref, pred)
    ax.set_title(f"{title}\nZone A {z['A']:.1f}% · A+B {z['A+B']:.1f}%", loc="left", fontsize=10)
    ax.grid(False)
    save(fig, name)
    return z


def fig_illness(path) -> None:
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    groups = [df.loc[~df.ill, "si_mult"], df.loc[df.ill, "si_mult"]]
    rng = np.random.default_rng(1)
    for i, (g, c) in enumerate(zip(groups, [GREY, S1], strict=True)):
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(g)), g, s=10, alpha=0.5, color=c, linewidths=0)
        ax.plot([i - 0.25, i + 0.25], [g.median()] * 2, color=INK, lw=2)
        ax.text(i + 0.3, g.median(), f"median {g.median():.2f}", va="center", fontsize=9, color=INK2)
    ax.set_xticks([0, 1], [f"Well days (n={len(groups[0])})", f"Ill days (n={len(groups[1])})"])
    ax.axhline(1.0, color=MUTED, lw=0.8)
    ax.set_ylabel("Twin's insulin sensitivity vs baseline")
    ax.set_title("The synced twin detects illness from CGM alone", loc="left")
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.6, 1.9)
    save(fig, "illness_detection")


def fig_twin_validity() -> None:
    p = pd.read_parquet(ARTIFACTS / "twins" / "cgmacros_twins.parquet")
    s = pd.read_parquet(PROCESSED / "cgmacros_static.parquet")
    d = p.merge(s[["patient_id", "status", "hba1c_pct", "homa_ir"]], on="patient_id")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    colors = {"normal": GREY, "prediabetes": S2, "t2d": S1}
    LABEL = {"normal": "Normal", "prediabetes": "Prediabetes", "t2d": "T2D"}
    for st, g in d.groupby("status"):
        axes[0].scatter(g.homa_ir, g.SI * 1e4, s=30, color=colors[st], edgecolors="white", linewidths=1, label=LABEL[st])
        axes[1].scatter(g.hba1c_pct, g.beta, s=30, color=colors[st], edgecolors="white", linewidths=1, label=LABEL[st])
    r1 = np.corrcoef(np.log(d.homa_ir), np.log(d.SI))[0, 1]
    r2 = np.corrcoef(d.hba1c_pct, np.log(d.beta))[0, 1]
    axes[0].set_xscale("log"); axes[0].set_yscale("log")  # noqa: E702
    axes[0].set_xlabel("HOMA-IR (lab)"); axes[0].set_ylabel("Fitted insulin sensitivity (x1e-4)")  # noqa: E702
    axes[0].set_title(f"Insulin resistance (r = {r1:.2f}, log-log)", loc="left", fontsize=10)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("HbA1c % (lab)"); axes[1].set_ylabel("Fitted beta-cell response")  # noqa: E702
    axes[1].set_title(f"Beta-cell failure (r = {r2:.2f})", loc="left", fontsize=10)
    axes[1].legend(fontsize=9, loc="upper right")
    fig.suptitle("Twins fitted to 45 real people recover known physiology from CGM", x=0.02, ha="left", fontweight="semibold", fontsize=11)
    fig.tight_layout()
    save(fig, "twin_validity")


def _per_patient(series: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    g = series.groupby("patient_id")["cgm"]
    out = pd.DataFrame({"mean": g.mean(), "cv": g.std() / g.mean() * 100,
                        "tir": g.apply(lambda x: ((x >= 70) & (x <= 180)).sum() / max(x.notna().sum(), 1) * 100)})
    return out.join(static.set_index("patient_id")["status"])


def fig_realism() -> None:
    real = _per_patient(pd.read_parquet(PROCESSED / "cgmacros_series.parquet"), pd.read_parquet(PROCESSED / "cgmacros_static.parquet"))
    syn = _per_patient(pd.read_parquet(SYNTHETIC / "full" / "series.parquet", columns=["patient_id", "cgm"]),
                       pd.read_parquet(SYNTHETIC / "full" / "static.parquet"))
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    order = ["normal", "prediabetes", "t2d"]
    for ax, col, lab in zip(axes, ["mean", "tir", "cv"], ["Mean glucose (mg/dL)", "Time in range (%)", "Glucose CV (%)"], strict=True):
        for i, st in enumerate(order):
            for j, (df, c) in enumerate(((real, GREY), (syn, S1))):
                v = df.loc[df.status == st, col].dropna()
                if len(v) == 0:
                    continue
                pos = i + (j - 0.5) * 0.34
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                ax.plot([pos, pos], [np.percentile(v, 5), np.percentile(v, 95)], color=c, lw=1)
                ax.add_patch(plt.Rectangle((pos - 0.12, q1), 0.24, q3 - q1, color=c, alpha=0.35, lw=0))
                ax.plot([pos - 0.12, pos + 0.12], [med, med], color=c, lw=2)
        ax.set_xticks(range(3), ["Normal", "Prediabetes", "T2D"])
        ax.set_title(lab, loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].plot([], [], color=GREY, lw=6, alpha=0.5, label="CGMacros (real, n=45)")
    axes[0].plot([], [], color=S1, lw=6, alpha=0.5, label="Synthetic India cohort (n=1000)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Synthetic CGM statistics vs real people (5th-95th, IQR, median); the Indian T2D cohort is deliberately sicker (median HbA1c 8.0% vs 7.1%)",
                 x=0.02, ha="left", fontweight="semibold", fontsize=10)
    fig.tight_layout()
    save(fig, "synthetic_realism")


def fig_example() -> None:
    idx = json.load(open(ARTIFACTS / "demo" / "index.json"))
    pid = next(i["id"] for i in idx if "Festival" in i["story"] or "Premixed" in i["story"])
    b = json.load(open(ARTIFACTS / "demo" / f"{pid}.json"))
    pr = b["predictions"]
    anchors = np.array(pr["anchors"])
    spike = np.array([np.nan if v is None else v for v in pr["spike"]])
    k0 = int(np.flatnonzero(anchors >= b["calib_bins"])[0])
    cand = [k for k in range(k0, len(anchors)) if np.nan_to_num(spike[k]) > 0.6]
    k = cand[len(cand) // 2] if cand else k0 + 200
    a = anchors[k]
    cgm = np.array([np.nan if v is None else v for v in b["series"]["cgm"]], float)
    t = (np.arange(len(cgm)) - a) * 5 / 60
    lo, hi = a - 12 * 12, a + 24
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.axhspan(70, 180, color=GOOD, alpha=0.07, lw=0)
    ax.plot(t[lo:a + 1], cgm[lo:a + 1], color=S1, label="Observed CGM")
    ax.plot(t[a:hi + 1], cgm[a:hi + 1], color=S1, alpha=0.35, lw=1.5, label="What actually happened")
    ft = np.arange(0, 25) * 5 / 60
    q10 = [cgm[a]] + pr["q10"][k]; q50 = [cgm[a]] + pr["q50"][k]; q90 = [cgm[a]] + pr["q90"][k]; tw = [cgm[a]] + pr["twin"][k]  # noqa: E702
    ax.fill_between(ft, q10, q90, color=S2, alpha=0.18, lw=0, label="80% interval")
    ax.plot(ft, q50, color=S2, label="MadhuTwin forecast")
    ax.plot(ft, tw, color=S3, label="Physics twin")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.text(0.05, ax.get_ylim()[1] * 0.97, f"forecast issued · P(>180) = {spike[k]:.0%}", fontsize=9, color=INK2, va="top")
    ax.set_xlabel("Hours relative to forecast")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title(f"{b['display']['name']} ({b['story']})", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", ncol=3)
    ax.set_xlim(-12, 2.2)
    save(fig, "example_forecast")


def _fmt(x, d: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def _by(rows: list[dict], m: str) -> dict:
    return next((r for r in rows if r["method"] == m), {})


def _ev(rows: list[dict], e: str, m: str) -> dict:
    return next((r for r in rows if r["event"] == e and r["method"] == m), {})


def _best_real(r: dict) -> str:
    return "MadhuTwin ensemble" if _by(r["forecast"], "MadhuTwin ensemble") else "TwinNet sim-to-real"


def update_readme(syn: dict, real: dict | None, extra: dict | None) -> None:
    """Write headline numbers and a results table into README.md between marker comments."""
    f = _fmt
    pers = _by(syn["forecast"], "Persistence")
    src, best_name = (extra, "MadhuTwin ensemble") if extra else (syn, "TwinNet hybrid")
    best, sp, hy = _by(src["forecast"], best_name), _ev(src["events"], "spike", best_name), _ev(src["events"], "hypo", best_name)
    abl = syn["ablation_gbm"]
    light = json.load(open(RES / "cgm_light.json")) if (RES / "cgm_light.json").exists() else None
    if sp.get("caught_at_1fa_pct") is not None:
        alert_txt = (f"{f(sp['caught_at_1fa_pct'], 0)}% of glucose spikes flagged a median {f(sp.get('lead_at_1fa_min'), 0)} min ahead "
                     f"with at most 1 false alert per patient-day")
    else:
        alert_txt = (f"{f(sp.get('detected_pct'), 0)}% of glucose spikes flagged a median {f(sp.get('median_lead_min'), 0)} min ahead "
                     f"at {f(sp.get('false_alerts_per_day'), 2)} false alerts per patient-day")
    head = (f"**Headline results (unseen patients):** 60-min forecast error {f(best['rmse_60'])} mg/dL vs {f(pers['rmse_60'])} for persistence; "
            f"{f(best['clarkeAB_60'])}% clinically acceptable (Clarke A+B); {alert_txt}; fusing both streams + the physics twin cuts "
            f"60-min error from {f(abl[0]['rmse_60'])} (CGM only) to {f(abl[-1]['rmse_60'])} mg/dL.")
    if extra:
        lc = extra["learning_curve"]
        head += f" The twin learns each person: 2-hour error falls from {f(lc[0]['rmse_120'])} to {f(lc[-1]['rmse_120'])} mg/dL after a week of their data."
    if real:
        cg, cgp = _by(real["cgmacros"]["forecast"], _best_real(real["cgmacros"])), _by(real["cgmacros"]["forecast"], "Persistence")
        head += f" On {real['cgmacros']['n_recordings']} real people (CGMacros): {f(cg['rmse_60'])} vs {f(cgp['rmse_60'])} mg/dL."
    rows = ["| Evaluation | Metric | MadhuTwin | Persistence baseline |", "|---|---|---:|---:|"]
    rows.append(f"| Synthetic India cohort, {syn['n_test_patients']} unseen patients | RMSE 30 / 60 / 120 min (mg/dL) | "
                f"{f(best['rmse_30'])} / {f(best['rmse_60'])} / {f(best['rmse_120'])} | {f(pers['rmse_30'])} / {f(pers['rmse_60'])} / {f(pers['rmse_120'])} |")
    rows.append(f"| | Clarke A+B at 60 min | {f(best['clarkeAB_60'])}% | {f(pers['clarkeAB_60'])}% |")
    if sp.get("caught_at_1fa_pct") is not None:
        rows.append(f"| | Spike >180 within 2 h: AUROC · caught at <=1 false alert/day · median lead | {f(sp.get('auroc'), 3)} · "
                    f"{f(sp['caught_at_1fa_pct'], 0)}% · {f(sp.get('lead_at_1fa_min'), 0)} min | – |")
    else:
        rows.append(f"| | Spike >180 within 2 h: AUROC · caught · median lead · false alerts/day | {f(sp.get('auroc'), 3)} · "
                    f"{f(sp.get('detected_pct'), 0)}% · {f(sp.get('median_lead_min'), 0)} min · {f(sp.get('false_alerts_per_day'), 2)} | – |")
    rows.append(f"| | Hypo <70 within 2 h: AUROC | {f(hy.get('auroc'), 3)} | – |")
    rows.append(f"| | Illness detected from CGM by the synced twin (AUROC) | {f(syn['illness_detection']['auroc'], 2)} | – |")
    if extra:
        lc = extra["learning_curve"]
        rows.append(f"| The twin learns you ({extra['curve_patients']} T2D patients) | 60 / 120-min RMSE with 0 vs 7 days of personal data | "
                    f"{f(lc[0]['rmse_60'])} / {f(lc[0]['rmse_120'])} → {f(lc[-1]['rmse_60'])} / {f(lc[-1]['rmse_120'])} | – |")
    if real:
        for key, label in (("cgmacros", "CGMacros, real people, 5-fold patient CV"), ("bigideas", "BIG IDEAs, real wristband HR/HRV, 5-fold CV"),
                           ("shanghai", "ShanghaiT2DM, external, no wearables")):
            if key not in real:
                continue
            m = _best_real(real[key])
            t, b = _by(real[key]["forecast"], m), _by(real[key]["forecast"], "Persistence")
            spr = _ev(real[key]["events"], "spike", m)
            rows.append(f"| {label} ({real[key]['n_recordings']}) | RMSE 30 / 60 / 120 · Clarke A+B 60 · spike AUROC · spikes caught at <=1 false alert/day | "
                        f"{f(t['rmse_30'])} / {f(t['rmse_60'])} / {f(t['rmse_120'])} · {f(t['clarkeAB_60'])}% · {f(spr.get('auroc'), 3)} · "
                        f"{f(spr.get('caught_at_1fa_pct'), 0)}% | {f(b['rmse_30'])} / {f(b['rmse_60'])} / {f(b['rmse_120'])} · {f(b['clarkeAB_60'])}% · – · – |")
    if light:
        rows.append(f"| CGM-light, synthetic patients (1 week CGM, then 4 fingersticks/day; upper bound) | MARD of continuous estimate · time-in-range error | "
                    f"{f(light['twin']['mard'])}% · ±{f(light['twin']['tir_abs_error_pp'])} pp | carry-forward {f(light['carry_forward']['mard'])}% · "
                    f"±{f(light['carry_forward']['tir_abs_error_pp'])} pp |")
    fid = json.load(open(RES / "fidelity.json")) if (RES / "fidelity.json").exists() else None
    rob = json.load(open(RES / "robustness.json")) if (RES / "robustness.json").exists() else None
    bench = json.load(open(RES / "bench.json")) if (RES / "bench.json").exists() else None
    if fid:
        parts = " · ".join(f"{lab} {f(fid[k]['mean_glucose_error_median'])} / {f(fid[k]['tir_error_median_pp'])} pp"
                           for k, lab in (("synthetic", "synthetic"), ("cgmacros", "CGMacros"), ("bigideas", "BIG IDEAs"), ("shanghai", "Shanghai")) if k in fid)
        rows.append(f"| 24-hour twin replay (every post-calibration day, open loop) | median error in daily mean glucose (mg/dL) / time in range | {parts} | – |")
    if bench:
        rows.append(f"| Speed on one laptop CPU ({bench['threads']} threads) | forecast · personalise twin + TwinNet · 24-h therapy simulation | "
                    f"{f(bench['twinnet_single_forecast_ms'])} ms · {f(bench['mechanistic_fit_7_days_s'] + bench['personalise_twinnet_s'])} s · "
                    f"{f(bench['therapy_sim_10_plans_24h_ms'] / 10)} ms per plan | – |")
    abl_rows = ["", "Does fusing the two streams help? (LightGBM retrained per combination, synthetic test patients)", "",
                "| Streams | RMSE 60 min | Spike AUROC |", "|---|---:|---:|"]
    abl_rows += [f"| {r['config']} | {r['rmse_60']:.2f} | {r.get('spike_auroc', float('nan')):.3f} |" for r in abl]
    if rob:
        abl_rows += ["", "Robust to missing data (one deployed model, data removed at inference, synthetic test patients)", "",
                     "| Missing at inference | RMSE 60 min | RMSE 120 min | Spike AUROC |", "|---|---:|---:|---:|"]
        abl_rows += [f"| {r['condition']} | {r['rmse_60']:.2f} | {r['rmse_120']:.2f} | {r['spike_auroc']:.3f} |" for r in rob["modalities"]]
        abl_rows += [f"| {r['cgm_readings_lost_pct']:.0f}% of past CGM readings lost | {r['rmse_60']:.2f} | {r['rmse_120']:.2f} | {r['spike_auroc']:.3f} |"
                     for r in rob["cgm_dropout"]]
    note = ("MadhuTwin = the mechanistic twin fitted to each person's own earlier data, TwinNet fine-tuned on the same data and averaged "
            "with LightGBM; risks are recalibrated on held-out patients (validation patients, or other folds' patients for real data). "
            "Real-data models are pretrained on the synthetic cohort and fine-tuned on the training folds only (sim-to-real)."
            if extra else "")
    table = "\n".join(["### Results", "", *([note, ""] if note else []), *rows, *abl_rows, "",
                       "Full tables, confidence intervals and figures: [`docs/evaluation_report.md`](docs/evaluation_report.md).", ""])
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for marker, content in (("RESULTS_HEADLINE", head), ("RESULTS_TABLE", table)):
        start, end = f"<!-- {marker} -->", f"<!-- /{marker} -->"
        if end in readme:
            pre, rest = readme.split(start, 1)
            readme = pre + start + "\n" + content + "\n" + end + rest.split(end, 1)[1]
        else:
            readme = readme.replace(start, start + "\n" + content + "\n" + end)
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    print("updated README.md results")


def fig_learning_curve(extra: dict) -> None:
    lc = extra["learning_curve"]
    d = [r["days"] for r in lc]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for key, color, lab in (("rmse_120", S2, "2-hour forecast"), ("rmse_60", S1, "1-hour forecast")):
        y = [r[key] for r in lc]
        ax.plot(d, y, marker="o", ms=6, color=color, mec="white", mew=1.5, label=lab)
        ax.annotate(f"{y[0]:.1f} → {y[-1]:.1f}", (d[-1], y[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.set_xticks(d)
    ax.set_xlabel("Days of the person's own data (twin fitted + TwinNet fine-tuned)")
    ax.set_ylabel("RMSE on days 8-14 (mg/dL)")
    ax.set_title(f"The twin learns you ({extra['curve_patients']} unseen T2D patients)", loc="left")
    ax.set_xlim(-0.3, max(d) + 2.2)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(0.8, 1.0))
    save(fig, "twin_learns_you")


def fig_calibration(extra: dict | None, real: dict | None) -> None:
    series = []
    if extra:
        series.append(("Synthetic test", extra["clinical"]["spike_calibration"], S1))
    if real and "clinical" in real:
        for key, lab, c in (("cgmacros", "CGMacros (real)", S2), ("bigideas", "BIG IDEAs (real)", S3), ("shanghai", "ShanghaiT2DM (real)", GREY)):
            if key in real["clinical"]:
                series.append((lab, real["clinical"][key]["spike_calibration"], c))
    if not series:
        return
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, label="Perfect calibration")
    for lab, pts, c in series:
        ax.plot([q["predicted"] for q in pts], [q["observed"] for q in pts], marker="o", ms=6, color=c, mec="white", mew=1.5, label=lab)
    ax.set_xlabel("Predicted probability of a spike within 2 h")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Spike-risk calibration (MadhuTwin)", loc="left")
    if real and "clinical" in real:
        ax.text(0.98, 0.02, "real cohorts: site-recalibrated,\ncross-fitted by patient fold", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color=MUTED)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)  # noqa: E702
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "calibration_spike")


def fig_decision(extra: dict | None, real: dict | None) -> None:
    panels = []
    if extra:
        panels.append(("Synthetic test", extra["clinical"]["spike_decision_curve"]))
    if real and "clinical" in real and "cgmacros" in real["clinical"]:
        panels.append(("CGMacros (real)", real["clinical"]["cgmacros"]["spike_decision_curve"]))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.4), squeeze=False)
    for ax, (title, rows) in zip(axes[0], panels, strict=True):
        t = [r["threshold"] for r in rows]
        model, allp = [r["model"] for r in rows], [r["alert_all"] for r in rows]
        ax.plot(t, model, color=S1, label="Alert on MadhuTwin risk")
        ax.plot(t, allp, color=GREY, label="Alert everyone")
        ax.axhline(0, color=INK2, lw=1, label="Alert no one")
        ax.set_ylim(-0.02, max(max(model), max(allp)) * 1.15 + 0.01)
        ax.set_xlabel("Risk threshold for an alert")
        ax.set_title(title, loc="left", fontsize=10)
    axes[0][0].set_ylabel("Net benefit")
    axes[0][0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Decision curves: spike alerts give net benefit across thresholds", x=0.02, ha="left", fontweight="semibold", fontsize=11)
    fig.tight_layout()
    save(fig, "decision_curve_spike")


def fig_subgroups(extra: dict) -> None:
    names = {"sex": "Sex", "age": "Age", "bmi": "BMI", "status": "Status", "therapy": "Therapy", "region": "Region"}
    rows = [r for r in extra["subgroups"] if r["patients"] >= 5]
    labels = [f"{names.get(r['attribute'], r['attribute'])}: {r['group']} (n={r['patients']})" for r in rows]
    vals = [r["rmse_60"] for r in rows]
    overall = _by(extra["forecast"], "MadhuTwin ensemble").get("rmse_60")
    fig, ax = plt.subplots(figsize=(6.2, 0.26 * len(rows) + 1.1))
    ax.barh(range(len(rows)), vals, color=S1, height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=8, color=INK2)
    if overall:
        ax.axvline(overall, color=INK2, lw=1)
        ax.text(overall + 0.2, len(rows) - 0.35, f"all patients {overall:.1f}", fontsize=8, color=INK2, ha="left", va="center")
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("1-hour RMSE (mg/dL)")
    ax.set_title("MadhuTwin error by subgroup (synthetic test patients)", loc="left")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    save(fig, "subgroups")


def render_video_script(syn: dict, real: dict | None, extra: dict | None = None) -> None:
    """Fill docs/video_script.template.md with numbers from the results and the demo cohort."""
    from twin.service.store import Store

    light = json.load(open(RES / "cgm_light.json")) if (RES / "cgm_light.json").exists() else None
    fits = pd.read_csv(ARTIFACTS / "twins" / "cgmacros_twins_summary.csv")
    by = lambda rows, m: next((r for r in rows if r["method"] == m), {})  # noqa: E731
    ev = lambda rows, e, m: next((r for r in rows if r["event"] == e and r["method"] == m), {})  # noqa: E731
    src, name = (extra, "MadhuTwin ensemble") if extra else (syn, "TwinNet hybrid")
    tn, pers = by(src["forecast"], name), by(syn["forecast"], "Persistence")
    sp, hy = ev(src["events"], "spike", name), ev(src["events"], "hypo", name)
    abl = syn["ablation_gbm"]
    f0 = lambda x: "–" if x is None else f"{x:.0f}"  # noqa: E731
    f1 = lambda x: "–" if x is None else f"{x:.1f}"  # noqa: E731
    f2 = lambda x: "–" if x is None else f"{x:.2f}"  # noqa: E731
    tok = {
        "india_dm": "101 million", "india_pre": "136 million", "n_test": str(syn["n_test_patients"]),
        "fit_prior": f1(fits.rmse_prior.mean()), "fit_pers": f1(fits.rmse_fit.mean()),
        "fit_gain": f"{(1 - fits.rmse_fit.mean() / fits.rmse_prior.mean()) * 100:.0f}",
        "ill_auroc": f2(syn["illness_detection"]["auroc"]), "ill_si_ill": f2(syn["illness_detection"]["mean_si_mult_ill"]),
        "ill_si_well": f2(syn["illness_detection"]["mean_si_mult_well"]),
        "gate30": f2(syn["gate_by_horizon"][5]), "gate120": f2(syn["gate_by_horizon"][23]),
        "tn_rmse60": f1(tn.get("rmse_60")), "pers_rmse60": f1(pers.get("rmse_60")),
        "gain60": f"{(1 - tn['rmse_60'] / pers['rmse_60']) * 100:.0f}", "tn_clarkeAB60": f1(tn.get("clarkeAB_60")),
        "tn_cov60": f"{tn.get('coverage80_60', float('nan')):.0f}", "spike_detected": f"{sp.get('detected_pct', float('nan')):.0f}",
        "spike_lead": f"{sp.get('median_lead_min', float('nan')):.0f}", "spike_fa": f2(sp.get("false_alerts_per_day")),
        "spike_caught1": f0(sp.get("caught_at_1fa_pct")), "spike_lead1": f0(sp.get("lead_at_1fa_min")),
        "hypo_auroc": f2(hy.get("auroc")), "abl_cgm_rmse60": f1(abl[0]["rmse_60"]), "abl_both_rmse60": f1(abl[-2]["rmse_60"]),
        "abl_full_rmse60": f1(abl[-1]["rmse_60"]), "abl_cgm_auroc": f2(abl[0].get("spike_auroc")), "abl_full_auroc": f2(abl[-1].get("spike_auroc")),
    }
    if extra:
        lc = {r["days"]: r for r in extra["learning_curve"]}
        tok |= {"lc0_120": f1(lc[0]["rmse_120"]), "lc1_120": f1(lc[1]["rmse_120"]), "lc7_120": f1(lc[7]["rmse_120"]),
                "lc_gain": f"{(1 - lc[7]['rmse_120'] / lc[0]['rmse_120']) * 100:.0f}", "lc_n": str(extra["curve_patients"])}
    if real:
        cg, cgp = by(real["cgmacros"]["forecast"], _best_real(real["cgmacros"])), by(real["cgmacros"]["forecast"], "Persistence")
        sh, shp = by(real["shanghai"]["forecast"], _best_real(real["shanghai"])), by(real["shanghai"]["forecast"], "Persistence")
        if "bigideas" in real:
            bi, bip = by(real["bigideas"]["forecast"], _best_real(real["bigideas"])), by(real["bigideas"]["forecast"], "Persistence")
            tok |= {"bi_rmse60": f1(bi.get("rmse_60")), "bi_pers60": f1(bip.get("rmse_60")), "bi_n": str(real["bigideas"]["n_recordings"])}
        tok |= {"cg_rmse60": f1(cg.get("rmse_60")), "cg_pers60": f1(cgp.get("rmse_60")), "cg_ab60": f1(cg.get("clarkeAB_60")),
                "sh_rmse60": f1(sh.get("rmse_60")), "sh_pers60": f1(shp.get("rmse_60")),
                "cg_spike_real_only": f2(ev(real["cgmacros"]["events"], "spike", "TwinNet real-only").get("auroc")),
                "cg_spike_s2r": f2(ev(real["cgmacros"]["events"], "spike", "TwinNet sim-to-real").get("auroc"))}
    if light:
        tok |= {"light_mard": f1(light["twin"]["mard"]), "light_carry_mard": f1(light["carry_forward"]["mard"]),
                "light_tir": f1(light["twin"]["tir_abs_error_pp"])}

    store = Store(ARTIFACTS / "demo", RES)
    idx = json.load(open(ARTIFACTS / "demo" / "index.json"))
    story = lambda word: next(i["id"] for i in idx if word in i["story"])  # noqa: E731
    clock_label = lambda c: f"Day {int(c // 1440) + 1} · {int(c % 1440) // 60:02d}:{int(c % 60):02d}"  # noqa: E731
    hp = store.get(story("Premixed"))
    d = hp.bundle["display"]
    alert = next(a for a in hp.bundle["alerts"] if a["kind"] == "hypo" and a["bin"] >= hp.origin)
    hclock = (alert["bin"] - hp.origin) * 5
    tok |= {"hypo_id": hp.id, "hypo_name": d["name"], "hypo_first": d["name"].split()[0], "hypo_age": str(d["age"]),
            "hypo_city": d["city"], "hypo_pronoun": "she" if d["sex"] == "F" else "he",
            "hypo_pronoun_cap": "She" if d["sex"] == "F" else "He", "hypo_clock": str(hclock), "hypo_time": clock_label(hclock),
            "hypo_prob": f"{alert['prob'] * 100:.0f}%"}
    wi = hp.what_if(780, meal={"food": "rice_dal", "portion": 1.0, "in_min": 0}, walk={"minutes": 15, "delay_min": 20})
    tok |= {"whatif_base_below70": str(wi["baseline_below_70_min"]), "whatif_scen_below70": str(wi["scenario_below_70_min"]),
            "whatif_scen_peak": str(wi["scenario_peak"])}
    fp = store.get(story("Festival"))
    tok |= {"fest_id": fp.id, "fest_name": fp.bundle["display"]["name"]}
    ip = store.get(story("illness"))
    si = ip.bundle["twin"]["si_daily"]
    day0 = int((ip.time_of(ip.origin).normalize() - ip.start.normalize()).days)
    low_day = min(range(day0, len(si)), key=lambda k: si[k] if si[k] is not None else 9)
    iclock = (low_day - day0) * 1440 + 10 * 60
    tok |= {"ill_id": ip.id, "ill_name": ip.bundle["display"]["name"], "ill_clock": str(iclock), "ill_time": clock_label(iclock),
            "ill_si_today": f2(ip.si_today(ip.bin_at(iclock)))}

    rob = json.load(open(RES / "robustness.json")) if (RES / "robustness.json").exists() else None
    fid = json.load(open(RES / "fidelity.json")) if (RES / "fidelity.json").exists() else None
    bench = json.load(open(RES / "bench.json")) if (RES / "bench.json").exists() else None
    if rob:
        m = {r["condition"]: r for r in rob["modalities"]}
        tok |= {"rob_full60": f1(m["All data"]["rmse_60"]), "rob_nowatch60": f1(m["No smartwatch (wearables missing)"]["rmse_60"]),
                "rob_noehr60": f1(m["No EHR"]["rmse_60"]), "rob_cgm50": f1(rob["cgm_dropout"][-1]["rmse_60"])}
    if fid:
        tok |= {"fid_cg_mean": f1(fid["cgmacros"]["mean_glucose_error_median"]), "fid_cg_tir": f1(fid["cgmacros"]["tir_error_median_pp"]),
                "fid_syn_mean": f1(fid["synthetic"]["mean_glucose_error_median"])}
    if bench:
        tok |= {"bench_ms": f1(bench["twinnet_single_forecast_ms"]), "bench_pers_s": f1(bench["mechanistic_fit_7_days_s"] + bench["personalise_twinnet_s"]),
                "bench_millions": f"{bench['patients_refreshed_per_5_min'] / 1e6:.0f}"}
    tp = store.get(story("SGLT2"))  # therapy simulator story: should the SGLT2 inhibitor be stopped at eGFR ~50?
    stop = tp.therapy_whatif(390, sglt2=False)
    plan = tp.therapy_plan(390)
    add_dpp4 = tp.therapy_whatif(390, dpp4=True)
    su = [d for d in plan["doses"] if d["kind"] == "sulfonylurea"]
    double_su = tp.therapy_whatif(390, doses=[d | {"amount": d["amount"] * 2} if d["kind"] == "sulfonylurea" else d for d in plan["doses"]])
    u, sc = stop["usual_stats"], stop["scenario_stats"]
    tok |= {"ther_id": tp.id, "ther_name": tp.bundle["display"]["name"], "ther_first": tp.bundle["display"]["name"].split()[0],
            "ther_egfr": f"{plan['egfr']:.0f}", "ther_tir_usual": f1(u["tir_pct"]), "ther_tir_stop": f1(sc["tir_pct"]),
            "ther_tar_usual": f1(u["above_180_pct"]), "ther_tar_stop": f1(sc["above_180_pct"]),
            "ther_mean_usual": str(u["mean"]), "ther_mean_stop": str(sc["mean"]),
            "ther_fid_rating": (stop["fidelity"] or {}).get("rating", "–"), "ther_fid_mae": f1((stop["fidelity"] or {}).get("mae")),
            "ther_tir_dpp4": f1(add_dpp4["scenario_stats"]["tir_pct"]), "ther_tir_2su": f1(double_su["scenario_stats"]["tir_pct"]),
            "ther_su_name": su[0]["display"] if su else "the sulfonylurea",
            "ther_poss": {"F": "her", "M": "his"}.get(tp.bundle["display"].get("sex"), "their")}
    (RES / "demo_tokens.json").write_text(json.dumps(tok, indent=1), encoding="utf-8")
    text = (ROOT / "docs" / "video_script.template.md").read_text(encoding="utf-8")
    for k, v in tok.items():
        text = text.replace("{{" + k + "}}", v)
    text = text.replace("(this file is generated by `scripts/make_report.py` from `docs/video_script.template.md`)", "(generated from `docs/video_script.template.md`)")
    missing = sorted(set(re.findall(r"{{(\w+)}}", text)))
    if missing:
        raise ValueError(f"unfilled video-script tokens: {missing}")
    (ROOT / "docs" / "video_script.md").write_text(text, encoding="utf-8")
    print("wrote docs/video_script.md")


def md_table(rows: list[dict], cols: list[tuple[str, str, str]]) -> str:
    head = "| " + " | ".join(c[1] for c in cols) + " |\n|" + "|".join("---" if i == 0 else "---:" for i in range(len(cols))) + "|\n"
    body = ""
    for r in rows:
        cells = []
        for key, _, fmt in cols:
            v = r.get(key)
            cells.append("–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (fmt.format(v) if fmt else str(v)))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


FCOLS = [("method", "Method", ""), ("rmse_30", "RMSE 30", "{:.1f}"), ("rmse_60", "RMSE 60", "{:.1f}"), ("rmse_90", "RMSE 90", "{:.1f}"),
         ("rmse_120", "RMSE 120", "{:.1f}"), ("mard_60", "MARD 60 %", "{:.1f}"), ("clarkeA_60", "Clarke A 60 %", "{:.1f}"),
         ("clarkeAB_60", "Clarke A+B 60 %", "{:.1f}"), ("coverage80_60", "80% PI coverage 60", "{:.1f}")]
ECOLS = [("event", "Event", ""), ("method", "Method", ""), ("auroc", "AUROC", "{:.3f}"), ("auprc", "AUPRC", "{:.3f}"),
         ("prevalence", "Prevalence %", "{:.1f}"), ("detected_pct", "Caught %", "{:.0f}"), ("detected_30min_ahead_pct", ">=30 min ahead %", "{:.0f}"),
         ("median_lead_min", "Median lead (min)", "{:.0f}"), ("false_alerts_per_day", "False alerts/day", "{:.2f}"),
         ("caught_at_1fa_pct", "Caught at <=1 FA/day %", "{:.0f}"), ("lead_at_1fa_min", "Lead at <=1 FA/day (min)", "{:.0f}")]


def main() -> None:
    syn = json.load(open(RES / "synthetic" / "results.json"))
    real = json.load(open(RES / "real" / "results.json")) if (RES / "real" / "results.json").exists() else None
    extra = json.load(open(RES / "synthetic" / "extra.json")) if (RES / "synthetic" / "extra.json").exists() else None
    print("figures:")
    syn_rows = syn["forecast"] + [r for r in (extra or {}).get("forecast", []) if r["method"] == "MadhuTwin ensemble"]
    fig_horizon(syn_rows, "rmse_by_horizon_synthetic", f"Forecast error by horizon · {syn['n_test_patients']} unseen synthetic patients",
                ["Persistence", "Twin (personalised)", "LightGBM fusion", "MadhuTwin ensemble" if extra else "TwinNet hybrid"])
    if real:
        fig_horizon(real["cgmacros"]["forecast"], "rmse_by_horizon_cgmacros",
                    f"Forecast error by horizon · {real['cgmacros']['n_recordings']} real people (CGMacros, 5-fold CV)",
                    ["Persistence", "Twin (personalised)", "LightGBM (real-trained)", _best_real(real["cgmacros"])])
    fig_ablation(syn["ablation_gbm"])
    c = np.load(RES / "synthetic" / "clarke_60.npz")
    fig_clarke(c["ref"], c["TwinNet hybrid"], "clarke_twinnet_synthetic_60", "TwinNet, 60-min forecasts (synthetic test)")
    if (RES / "real" / "clarke_cgmacros_60.npz").exists():
        cr = np.load(RES / "real" / "clarke_cgmacros_60.npz")
        if "ensemble" in cr:
            fig_clarke(cr["ref"], cr["ensemble"], "clarke_twinnet_cgmacros_60", "MadhuTwin, 60-min forecasts (CGMacros, real)")
        else:
            fig_clarke(cr["ref"], cr["twinnet"], "clarke_twinnet_cgmacros_60", "TwinNet sim-to-real, 60-min (CGMacros)")
    fig_illness(RES / "synthetic" / "illness_days.csv")
    fig_twin_validity()
    fig_realism()
    fig_example()
    if extra:
        fig_learning_curve(extra)
        fig_subgroups(extra)
    fig_calibration(extra, real)
    fig_decision(extra, real)

    out = ["# Evaluation report", "",
           "Generated by `scripts/make_report.py` from `artifacts/results/`. All evaluation is on held-out patients; "
           "synthetic results use days 8-14 after each twin was personalised on days 1-7.", "",
           f"## 1. Synthetic India-calibrated cohort ({syn['n_test_patients']} unseen patients, {syn['n_test_anchors']:,} forecasts)", "",
           "### Glucose forecasting", "", md_table(syn["forecast"], FCOLS), "",
           "### Adverse-event prediction (2-hour horizon, >=15 min beyond threshold)", "", md_table(syn["events"], ECOLS), "",
           "### Stream-fusion ablation (LightGBM retrained per combination)", "",
           md_table(syn["ablation_gbm"], [("config", "Streams", ""), ("rmse_60", "RMSE 60", "{:.2f}"), ("rmse_120", "RMSE 120", "{:.2f}"),
                                          ("spike_auroc", "Spike AUROC", "{:.3f}"), ("hypo_auroc", "Hypo AUROC", "{:.3f}")]), ""]
    if syn.get("ablation_twinnet"):
        out += ["### TwinNet ablations (retrained without one stream)", "",
                md_table(syn["ablation_twinnet"], [("config", "Variant", ""), ("rmse_30", "RMSE 30", "{:.2f}"), ("rmse_60", "RMSE 60", "{:.2f}"),
                                                   ("rmse_120", "RMSE 120", "{:.2f}"), ("spike_auroc", "Spike AUROC", "{:.3f}"), ("hypo_auroc", "Hypo AUROC", "{:.3f}")]), ""]
    ill = syn["illness_detection"]
    out += ["### Illness detection by the synchronised twin", "",
            f"Daily mean of the UKF insulin-sensitivity multiplier as a score for 'ill day': AUROC **{ill['auroc']:.3f}** "
            f"({ill['n_ill_days']} ill days of {ill['n_days']}); mean multiplier {ill['mean_si_mult_ill']:.2f} on ill days vs "
            f"{ill['mean_si_mult_well']:.2f} on well days.", "",
            "TwinNet's learned trust in the physics twin by horizon (gate, 5..120 min): "
            + ", ".join(f"{g:.2f}" for g in syn["gate_by_horizon"][5::6]), ""]
    if extra:
        out += ["### Personalisation and the MadhuTwin ensemble", "",
                "TwinNet personalised = the population model fine-tuned on each patient's own days 1-7. MadhuTwin ensemble = personalised "
                "TwinNet averaged with LightGBM; its event probabilities are recalibrated (Platt scaling) on the validation patients. "
                "'Caught at <=1 FA/day' = share of excursions alerted in the 2 h before onset at the "
                "operating point that keeps false-alert episodes at or below one per patient-day. It is a point on the alert-burden curve, "
                "like sensitivity at a fixed specificity; the 'Caught %' and 'False alerts/day' columns instead use a threshold fixed in "
                "advance (validation data, or training folds for real data).", "",
                md_table(extra["forecast"], FCOLS), "", md_table(extra["events"], ECOLS), "",
                f"### The twin learns you ({extra['curve_patients']} unseen T2D patients; error on days 8-14)", "",
                "On day k the mechanistic twin is fitted to the person's first k days and synced, and TwinNet is fine-tuned on the same "
                "days. Day 0 is the population TwinNet with the physics input off (no personal data). "
                + (f"Feeding the unfitted EHR-prior twin into TwinNet on day 0 would give a 2-h RMSE of {extra['day0_with_prior_twin']['rmse_120']:.1f}."
                   if "day0_with_prior_twin" in extra else ""), "",
                md_table(extra["learning_curve"], [("days", "Days of personal data", ""), ("rmse_60", "RMSE 60", "{:.2f}"),
                                                   ("rmse_120", "RMSE 120", "{:.2f}")]), "",
                "### Subgroup error (MadhuTwin ensemble)", "",
                md_table(extra["subgroups"], [("attribute", "Attribute", ""), ("group", "Group", ""), ("patients", "Patients", ""),
                                              ("rmse_60", "RMSE 60", "{:.1f}"), ("rmse_120", "RMSE 120", "{:.1f}")]), ""]
    if real:
        for key, title in (("cgmacros", "Real-world validation: CGMacros (PhysioNet)"),
                           ("bigideas", "Real wearables: BIG IDEAs (PhysioNet; wristband heart rate and HRV)"),
                           ("shanghai", "External validation: ShanghaiT2DM")):
            if key not in real:
                continue
            r = real[key]
            out += [f"## {title} ({r['n_recordings']} recordings, {r['n_anchors']:,} forecasts, 5-fold patient CV)", "",
                    "MadhuTwin ensemble event probabilities are recalibrated for the site (Platt scaling fitted on the other folds' "
                    "patients only).", "",
                    md_table(r["forecast"], FCOLS), "", md_table(r["events"], ECOLS), ""]
    fid = json.load(open(RES / "fidelity.json")) if (RES / "fidelity.json").exists() else None
    rob = json.load(open(RES / "robustness.json")) if (RES / "robustness.json").exists() else None
    bench = json.load(open(RES / "bench.json")) if (RES / "bench.json").exists() else None
    if fid:
        out += ["## 24-hour twin fidelity (the trust check behind the therapy simulator)", "",
                "Each day after calibration, the synced twin starts from its filtered state and replays the day open-loop with the logged "
                "meals, doses and activity; the simulated day is compared with the CGM and rated as in the dashboard.", "",
                md_table([{"cohort": k, **v, "good": v["rating_pct"]["good"], "fair": v["rating_pct"]["fair"], "poor": v["rating_pct"]["poor"]}
                          for k, v in fid.items()],
                         [("cohort", "Cohort", ""), ("patients", "Patients", ""), ("days", "Days", ""), ("mean_abs_error_median", "Median MAE (mg/dL)", "{:.1f}"),
                          ("mean_glucose_error_median", "Median daily-mean error", "{:.1f}"), ("tir_error_median_pp", "Median TIR error (pp)", "{:.1f}"),
                          ("good", "Good %", "{:.0f}"), ("fair", "Fair %", "{:.0f}"), ("poor", "Poor %", "{:.0f}"),
                          ("lows_reproduced_pct", "Days with lows reproduced %", "{:.0f}")]),
                "The open-loop replay tracks daily mean glucose and time in range but rarely reproduces short lows; hypoglycaemia warnings "
                "therefore come from the 2-hour forecast, and the therapy simulator reports mean glucose, time in range and time above range.", ""]
    if rob:
        out += ["## Robustness to missing data at inference", "",
                f"{rob['model']}, {rob['n_forecasts']:,} forecasts. CGM dropout removes past readings from the network's input; the synced twin's "
                "state is not recomputed, which isolates the network's sensitivity.", "",
                md_table(rob["modalities"], [("condition", "Missing at inference", ""), ("rmse_30", "RMSE 30", "{:.2f}"), ("rmse_60", "RMSE 60", "{:.2f}"),
                                             ("rmse_120", "RMSE 120", "{:.2f}"), ("spike_auroc", "Spike AUROC", "{:.3f}"), ("hypo_auroc", "Hypo AUROC", "{:.3f}")]), "",
                md_table(rob["cgm_dropout"], [("cgm_readings_lost_pct", "Past CGM readings lost %", "{:.0f}"), ("rmse_60", "RMSE 60", "{:.2f}"),
                                              ("rmse_120", "RMSE 120", "{:.2f}"), ("spike_auroc", "Spike AUROC", "{:.3f}")]), ""]
    if bench:
        out += ["## Speed on a CPU", "",
                md_table([{"what": k.replace("_", " "), "value": v} for k, v in bench.items() if k not in ("cpu", "threads")],
                         [("what", f"Measure ({bench['cpu']}, {bench['threads']} threads)", ""), ("value", "Value", "{:,.1f}")]), ""]
    out += ["## Figures", ""] + [f"![{f.stem}](figures/{f.name})" for f in sorted(FIG.glob("*.png"))] + [""]
    (ROOT / "docs" / "evaluation_report.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote docs/evaluation_report.md")
    render_video_script(syn, real, extra)
    update_readme(syn, real, extra)


if __name__ == "__main__":
    main()
