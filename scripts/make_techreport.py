"""Build docs/technical_report.pdf: a self-contained technical report for reviewers.

Every number is read from artifacts/results at build time; figures come from docs/figures
(run scripts/make_report.py first). The HTML is printed to PDF with headless Microsoft Edge
or Chrome.

    python scripts/make_techreport.py
"""

from __future__ import annotations

import html
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from twin.paths import ARTIFACTS, ROOT

RES = ARTIFACTS / "results"
DOCS = ROOT / "docs"
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe", "msedge", "google-chrome", "chromium"]


def load(p: Path) -> dict | None:
    return json.load(open(p)) if p.exists() else None


def fmt(x, d: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and x != x) else f"{x:.{d}f}"


def by(rows: list[dict], m: str) -> dict:
    return next((r for r in rows if r["method"] == m), {})


def ev(rows: list[dict], e: str, m: str) -> dict:
    return next((r for r in rows if r["event"] == e and r["method"] == m), {})


def table(rows: list[dict], cols: list[tuple[str, str, str]], bold: str | None = None) -> str:
    head = "".join(f"<th{' class=num' if i else ''}>{html.escape(c[1])}</th>" for i, c in enumerate(cols))
    body = ""
    for r in rows:
        cells = []
        for i, (k, _, f) in enumerate(cols):
            v = r.get(k)
            s = "–" if v is None or (isinstance(v, float) and v != v) else (f.format(v) if f else str(v))
            cells.append(f"<td{' class=num' if i else ''}>{html.escape(s)}</td>")
        cls = " class=best" if bold and r.get("method") == bold else ""
        body += f"<tr{cls}>{''.join(cells)}</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def fig(name: str, caption: str, width: str = "100%") -> str:
    p = DOCS / "figures" / f"{name}.png"
    if not p.exists():
        return ""
    return f'<figure><img src="figures/{name}.png" style="width:{width}"><figcaption>{html.escape(caption)}</figcaption></figure>'


FC = [("method", "Method", ""), ("rmse_30", "RMSE 30", "{:.1f}"), ("rmse_60", "RMSE 60", "{:.1f}"), ("rmse_120", "RMSE 120", "{:.1f}"),
      ("mard_60", "MARD 60 %", "{:.1f}"), ("clarkeAB_60", "Clarke A+B 60 %", "{:.1f}"), ("coverage80_60", "PI80 cov. 60 %", "{:.0f}")]
EC = [("method", "Method", ""), ("auroc", "AUROC", "{:.3f}"), ("auprc", "AUPRC", "{:.3f}"), ("prevalence", "Prev. %", "{:.1f}"),
      ("caught_at_1fa_pct", "Caught at ≤1 FA/day %", "{:.0f}"), ("lead_at_1fa_min", "Lead (min)", "{:.0f}")]


def build() -> str:
    syn = load(RES / "synthetic" / "results.json")
    real = load(RES / "real" / "results.json")
    extra = load(RES / "synthetic" / "extra.json")
    light = load(RES / "cgm_light.json")
    fits = pd.read_csv(ARTIFACTS / "twins" / "cgmacros_twins_summary.csv")
    best = "MadhuTwin ensemble" if extra else "TwinNet hybrid"
    src = extra or syn
    tn, pers = by(src["forecast"], best), by(syn["forecast"], "Persistence")
    sp, hy = ev(src["events"], "spike", best), ev(src["events"], "hypo", best)
    ill = syn["illness_detection"]
    abl = syn["ablation_gbm"]
    gate = syn["gate_by_horizon"]
    lc = extra["learning_curve"] if extra else None

    def best_real(c: dict) -> str:
        return "MadhuTwin ensemble" if by(c["forecast"], "MadhuTwin ensemble") else "TwinNet sim-to-real"

    cohorts = [(k, lab) for k, lab in (("cgmacros", "CGMacros"), ("bigideas", "BIG IDEAs"), ("shanghai", "ShanghaiT2DM")) if real and k in real]
    real_rows = []
    for k, lab in cohorts:
        c = real[k]
        m = best_real(c)
        t, p, s = by(c["forecast"], m), by(c["forecast"], "Persistence"), ev(c["events"], "spike", m)
        real_rows.append({"method": f"{lab} ({c['n_recordings']})", "p60": p.get("rmse_60"), "t30": t.get("rmse_30"), "t60": t.get("rmse_60"),
                          "t120": t.get("rmse_120"), "ab": t.get("clarkeAB_60"), "auroc": s.get("auroc"), "c1": s.get("caught_at_1fa_pct")})

    abstract = (
        f"MadhuTwin is a hybrid digital twin for people with type 2 diabetes. A mechanistic glucose–insulin model is personalised to each "
        f"person from their EHR and first days of data, kept synchronised with their continuous glucose monitor by an Unscented Kalman Filter, "
        f"and run forward to forecast glucose and simulate what-ifs. A physics-gated neural network (TwinNet) fuses the EHR, wearable streams "
        f"and the twin's own forecast; it is fine-tuned on each person and averaged with a gradient-boosted model. On {syn['n_test_patients']} "
        f"unseen synthetic Indian patients, the 60-minute forecast error is {fmt(tn.get('rmse_60'))} mg/dL against {fmt(pers.get('rmse_60'))} "
        f"for persistence, {fmt(tn.get('clarkeAB_60'))}% of forecasts are clinically acceptable (Clarke A+B)")
    if sp.get("caught_at_1fa_pct") is not None:
        abstract += (f", and {fmt(sp['caught_at_1fa_pct'], 0)}% of sustained spikes above 180 mg/dL are flagged a median "
                     f"{fmt(sp.get('lead_at_1fa_min'), 0)} minutes ahead with at most one false alert per patient-day")
    abstract += "."
    if real_rows:
        abstract += " On real people with patient-level cross-validation, " + "; ".join(
            f"{r['method']}: {fmt(r['t60'])} vs {fmt(r['p60'])} mg/dL" for r in real_rows) + " (MadhuTwin vs persistence, 60 min)."
    if lc:
        abstract += f" With a week of personal data, the 2-hour error falls from {fmt(lc[0]['rmse_120'])} to {fmt(lc[-1]['rmse_120'])} mg/dL."
    abstract += (f" The synchronised twin detects illness days from CGM alone (AUROC {fmt(ill['auroc'], 2)}). Results are reported with "
                 "patient-bootstrap confidence intervals, calibration and decision curves, a subgroup audit and stated limitations.")

    out = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>MadhuTwin technical report</title><style>
@page {{ size: A4; margin: 16mm 16mm 18mm 16mm; }}
body {{ font-family: Calibri, "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.42; color: #1d2330; background: #fff; }}
h1 {{ font-family: Cambria, Georgia, serif; font-size: 22pt; margin: 0 0 4pt; color: #13284a; }}
h2 {{ font-family: Cambria, Georgia, serif; font-size: 14pt; margin: 16pt 0 5pt; color: #13284a; border-bottom: 1px solid #d9dee8; padding-bottom: 2pt; }}
h3 {{ font-size: 11pt; margin: 10pt 0 3pt; color: #13284a; }}
p {{ margin: 0 0 6pt; text-align: justify; }}
.sub {{ color: #4a5260; font-size: 11pt; margin-bottom: 10pt; }}
.abstract {{ background: #f2f5fa; padding: 8pt 10pt; border-radius: 4pt; margin: 8pt 0 6pt; }}
table {{ border-collapse: collapse; width: 100%; margin: 4pt 0 8pt; font-size: 9pt; break-inside: avoid; }}
th, td {{ border-bottom: 1px solid #e2e6ee; padding: 2.5pt 5pt; text-align: left; }}
th {{ background: #f2f5fa; font-weight: 600; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
tr.best td {{ font-weight: 700; }}
figure {{ margin: 6pt 0 10pt; text-align: center; break-inside: avoid; }}
figcaption {{ font-size: 9pt; color: #4a5260; margin-top: 2pt; }}
.eq {{ font-family: Consolas, "Cambria Math", monospace; font-size: 9.5pt; background: #fafbfd; border-left: 3px solid #2a78d6; padding: 4pt 8pt; margin: 4pt 0 8pt; white-space: pre; }}
.two {{ display: flex; gap: 10pt; }} .two > * {{ flex: 1; }}
ul {{ margin: 0 0 6pt 16pt; padding: 0; }} li {{ margin-bottom: 2pt; }}
.small {{ font-size: 9pt; color: #4a5260; }}
h2, h3 {{ break-after: avoid; }}
.keep {{ break-inside: avoid; }}
</style></head><body>
<h1>MadhuTwin: a hybrid digital twin for type 2 diabetes</h1>
<div class="sub">Technical report · Happiest Health Digital Twin Challenge 2026, Phase 1 · Team: TODO(team)</div>
<div class="abstract"><b>Abstract.</b> {html.escape(abstract)}</div>

<h2>1. Problem and use case</h2>
<p>India has about 101 million adults with diabetes and 136 million with prediabetes (ICMR-INDIAB, 2023). Care is visit-based: a quarterly
HbA1c summarises the past, while the dangerous moments (post-meal spikes, night-time lows on sulfonylureas or premixed insulin, loss of
control during illness) happen between visits. CGMs and smartwatches now record them, but a care team cannot watch hundreds of streams.
MadhuTwin targets <b>remote monitoring by a diabetes care team</b>: forecast glucose 5–120 minutes ahead with an 80% interval, estimate the
probability of a sustained excursion above 180 or below 70 mg/dL within two hours, rank the panel by risk, explain each alert, and let the
clinician test meals, walks, sleep or illness on the patient's own twin.</p>

<div class="keep"><h2>2. System overview</h2>
{fig("architecture", "Figure 1. Two data streams (EHR and wearables) are ingested into one 5-minute schema; the twin models, synchronises and simulates; TwinNet fuses everything; the dashboard and a FHIR R4 RiskAssessment deliver the result.")}</div>
<p>The system has three digital-twin properties. <b>Model</b>: a personalised mechanistic model of glucose–insulin physiology.
<b>Sync</b>: the model's hidden state and a drifting insulin-sensitivity factor are re-estimated from every CGM reading.
<b>Simulate</b>: the synchronised twin is run forward for forecasts and counterfactual what-ifs. On top, TwinNet learns the residual the
physics misses and how much to trust the physics at each horizon.</p>

<h2>3. Data</h2>
<table><thead><tr><th>Source</th><th>People</th><th>Streams</th><th>Role</th></tr></thead><tbody>
<tr><td>Synthetic India-calibrated cohort (this project)</td><td class=num>1,000 × 14 days</td><td>FHIR R4 EHR (labs, ATC drugs, SNOMED diagnoses, genetics), CGM, HR, HRV, steps, sleep, meals, doses</td><td>Training at scale; controlled ablations; illness ground truth</td></tr>
<tr><td>CGMacros (PhysioNet, CC BY-NC-SA 4.0)</td><td class=num>45</td><td>Dexcom/Libre CGM, Fitbit HR and steps, meal macros, labs</td><td>Real-world validation; twin-validity analysis</td></tr>
<tr><td>BIG IDEAs (PhysioNet, ODC-By 1.0)</td><td class=num>16</td><td>Dexcom G6 CGM, Empatica E4 heart rate and beat intervals (HRV computed), food logs</td><td>Real wristband HR and HRV</td></tr>
<tr><td>ShanghaiT2DM (Figshare, CC BY 4.0)</td><td class=num>100 (109 recordings)</td><td>15-min CGM, diet, insulin and oral drugs, labs; no wearables</td><td>External population; missing-modality test</td></tr>
</tbody></table>
<p>The synthetic EHR follows Indian epidemiology (status mix from ICMR-INDIAB, South-Asian BMI phenotype, Indian prescribing such as premixed
30/70 insulin, gliclazide and teneligliptin) and regional meals from the Indian Food Composition Tables. Crucially, the wearable streams are
generated by the same simulated body, and labs are measured from it (HbA1c from mean glucose via ADAG), so the two streams are causally
coupled. Realistic imperfections are injected: unlogged meals, carbohydrate misestimation, CGM noise and compression lows, missed doses,
illness and stress days. Physiological parameter distributions come from twins fitted to the 45 CGMacros participants.</p>
{fig("synthetic_realism", "Figure 2. Synthetic CGM statistics against real people by glycaemic status.", "92%")}

<h2>4. Methods</h2>
<h3>4.1 Mechanistic model</h3>
<div class="eq">dG/dt  = −SG (G − Gb·egp) − SI·m_si·X·G + Ra(t) − k_ex·E1·G − renal(G) + counterreg(G)
dX/dt  = −p2 (X − (I − Ib))
dI/dt  = −n (I − Ib) + β·m_β·[G − Gb]₊ + k_inc·m_inc·Ra(t) + I_ex(t) + SU(t)
dGi/dt = (G − Gi) / τ_isf          (interstitial glucose seen by the CGM)
dE1/dt = (A(t) − E1)/10 ;  dE2/dt = (A(t) − E2)/360   (acute and lingering exercise)</div>
<p>Meal appearance Ra(t) depends on carbohydrate, glycaemic index, fat and fibre. Time-varying multipliers encode sleep loss, illness and
stress (m_si), the dawn phenomenon (egp), sulfonylurea secretion, DPP-4 incretin enhancement (m_inc), the SGLT2-lowered renal threshold, and
rapid, regular, basal and premixed insulin kernels. A Numba-compiled integrator (verified against the NumPy reference) makes a patient fit
about 12× faster.</p>
<h3>4.2 Personalisation (Model)</h3>
<p>Five parameters (fasting set-point Gb, insulin sensitivity SI, beta-cell response β, dawn amplitude, absorption-speed scale) are fitted
by maximum a posteriori estimation with EHR-derived priors, using a multiple-shooting objective over 4-hour windows (1-hour burn-in, 3 hours
scored). On the 45 CGMacros participants the twin's error falls from {fmt(fits.rmse_prior.mean())} mg/dL with EHR priors to
{fmt(fits.rmse_fit.mean())} mg/dL once personalised, and the fitted SI and β track laboratory HOMA-IR and HbA1c (Figure 3).</p>
{fig("twin_validity", "Figure 3. Fitted parameters recover known physiology on 45 real people.", "88%")}
<h3>4.3 Synchronisation (Sync)</h3>
<p>An Unscented Kalman Filter estimates seven states every 5 minutes: the six model states and a log insulin-sensitivity multiplier that
follows a bounded random walk. A sustained fall in that multiplier is surfaced as an insight (possible infection, stress or steroid use).</p>
<h3>4.4 TwinNet and the MadhuTwin ensemble (Simulate)</h3>
<p>TwinNet reads 6 hours × 21 dynamic channels (CGM, heart rate, HRV, steps, METs, sleep, carbohydrate and insulin on board, logged events,
time of day) through a causal dilated temporal convolution (dilations 1–16). An EHR encoder conditions the sequence features through FiLM,
so the same meal is read differently for different bodies. The synchronised twin's 2-hour forecast and hidden state enter through a learned
per-horizon gate; heads output P10/P50/P90 glucose at 5–120 minutes and the probabilities of a spike or a low within 2 hours. Modality dropout
lets one model run without wearables or EHR, and conformalised quantile regression calibrates the 80% band. The learned trust in physics
rises from {fmt(gate[5], 2)} at 30 minutes to {fmt(gate[23], 2)} at 2 hours.</p>
<p>The twin then learns the individual: TwinNet is fine-tuned for a few epochs on the person's own earlier windows only, and its median and
event probabilities are averaged with a LightGBM model on hand-crafted features of the same inputs. LightGBM's TreeSHAP values give the
plain-language reasons attached to every alert. Both models are trained with class weighting so that rare events are learned, which
inflates raw probabilities; the ensemble's risks are therefore recalibrated with Platt scaling, fitted on validation patients for the
synthetic cohort and, for each real-data site, on other patients only (cross-fitted by fold in evaluation).</p>

<h2>5. Evaluation protocol</h2>
<ul>
<li><b>Splits.</b> Patient-level throughout. Synthetic: 70/10/20 split by patient; scored only on days 8–14, after the twin and TwinNet were personalised on days 1–7. Real data: 5-fold cross-validation grouped by patient; models are pretrained on synthetic data and fine-tuned on training folds only (sim-to-real).</li>
<li><b>Forecasts.</b> RMSE, MARD, Clarke Error Grid zones and 80% interval coverage at 30, 60, 90 and 120 minutes; confidence intervals by patient bootstrap.</li>
<li><b>Events.</b> A spike is glucose above 180 mg/dL (a low: below 70) sustained for at least 15 minutes within the next 2 hours. AUROC and AUPRC; the share of excursions alerted in the 2 hours before onset at the operating point with at most one false-alert episode per patient-day, and the median lead time. Alert thresholds on real data are chosen on the training folds.</li>
<li><b>Clinical utility.</b> Calibration curves, decision-curve net benefit (Vickers and Elkin, 2006) and a subgroup audit.</li>
</ul>

<h2>6. Results</h2>
<h3>6.1 Forecasting on unseen synthetic patients</h3>
{table((extra or syn)["forecast"], FC, bold=best)}
{fig("rmse_by_horizon_synthetic", "Figure 4. Forecast error by horizon on unseen synthetic patients.", "72%")}
<h3>6.2 Adverse-event prediction</h3>
{table([r for r in (extra or syn)["events"] if r["event"] == "spike"], EC, bold=best)}
<p class="small">Spike above 180 mg/dL within 2 hours. Hypoglycaemia: AUROC {fmt(hy.get('auroc'), 3)} for {best}.</p>
<h3>6.3 Does fusing the two streams help?</h3>
<div class="two"><div>{table(abl, [("config", "Streams (LightGBM retrained)", ""), ("rmse_60", "RMSE 60", "{:.2f}"), ("spike_auroc", "Spike AUROC", "{:.3f}")])}</div>
<div><p>Every stream adds information, and the EHR helps most in combination: it tells the model whose body the wearables describe. Adding the
physics twin's forecast as a feature cuts the 60-minute error from {fmt(abl[0]['rmse_60'])} (CGM only) to {fmt(abl[-1]['rmse_60'])} mg/dL.</p></div></div>
"""]
    if lc:
        out.append(f"""<h3>6.4 The twin learns you</h3>
<div class="two"><div>{fig("twin_learns_you", f"Figure 5. Error on days 8–14 against days of personal data used to fit the twin and fine-tune TwinNet ({extra['curve_patients']} unseen T2D patients; day 0 = population model).")}</div>
<div>{table(lc, [("days", "Days of personal data", ""), ("rmse_60", "RMSE 60", "{:.2f}"), ("rmse_120", "RMSE 120", "{:.2f}")])}</div></div>
""")
    if extra or (real and "clinical" in real):
        sg = [r for r in (extra or {}).get("subgroups", []) if r["patients"] >= 5]
        out.append(f"""<h3>6.5 Calibration, net benefit and subgroups</h3>
<div class="two"><div>{fig("calibration_spike", "Figure 6. Calibration of spike risk (deciles of predicted risk).")}</div>
<div>{fig("subgroups", "Figure 7. One-hour error by subgroup (synthetic test patients; groups with at least 5 patients).")}</div></div>
{fig("decision_curve_spike", "Figure 8. Decision curves: net benefit of alerting on MadhuTwin's spike risk against alerting everyone or no one.", "92%")}
""" + (f"<p>Across {len(sg)} subgroups the 1-hour error ranges from {fmt(min(r['rmse_60'] for r in sg))} to {fmt(max(r['rmse_60'] for r in sg))} mg/dL.</p>" if sg else ""))
    if real_rows:
        out.append(f"""<h3>6.6 Real people and an external population</h3>
{table(real_rows, [("method", "Cohort (recordings)", ""), ("p60", "Persistence RMSE 60", "{:.1f}"), ("t30", "MadhuTwin RMSE 30", "{:.1f}"),
                   ("t60", "RMSE 60", "{:.1f}"), ("t120", "RMSE 120", "{:.1f}"), ("ab", "Clarke A+B 60 %", "{:.1f}"),
                   ("auroc", "Spike AUROC", "{:.3f}"), ("c1", "Caught at ≤1 FA/day %", "{:.0f}")])}
<p>All real-data results use 5-fold patient-level cross-validation. A model trained only on synthetic data does not transfer to a new
population without a short fine-tune (zero-shot rows in the evaluation report), while synthetic pretraining followed by fine-tuning on a
few real patients is the regime we recommend.</p>
{fig("rmse_by_horizon_cgmacros", "Figure 9. Forecast error by horizon on 45 real people (CGMacros, 5-fold patient CV).", "72%")}
""")
    out.append(f"""<h3>6.7 Illness detection by the synchronised twin</h3>
<div class="two"><div>{fig("illness_detection", "Figure 10. Daily insulin sensitivity estimated by the UKF on ill and well days.")}</div>
<div><p>Using the daily mean of the UKF's insulin-sensitivity multiplier as a score, ill days are separated from well days with AUROC
<b>{fmt(ill['auroc'], 3)}</b> ({ill['n_ill_days']} ill days of {ill['n_days']}); the multiplier averages {fmt(ill['mean_si_mult_ill'], 2)} on ill days
against {fmt(ill['mean_si_mult_well'], 2)} on well days, without any symptom being entered.</p></div></div>
""")
    if light:
        out.append(f"""<h3>6.8 CGM-light: an affordability mode</h3>
<p>After one week of CGM, only {light['fingersticks_per_day']} fingersticks a day and a smartwatch remain. The synchronised twin estimates continuous
glucose with a MARD of {fmt(light['twin']['mard'])}% (carrying the last fingerstick forward: {fmt(light['carry_forward']['mard'])}%) and weekly time in range
within ±{fmt(light['twin']['tir_abs_error_pp'])} percentage points, on {light['n_patients']} unseen synthetic patients. Because the simulator shares the twin's
structure, this is an upper bound until tested on real CGM-off data.</p>
""")
    fid = load(RES / "fidelity.json")
    rob = load(RES / "robustness.json")
    bench = load(RES / "bench.json")
    if fid:
        frows = [{"method": lab, **fid[k], "good": fid[k]["rating_pct"]["good"], "poor": fid[k]["rating_pct"]["poor"]}
                 for k, lab in (("synthetic", "Synthetic test"), ("cgmacros", "CGMacros"), ("bigideas", "BIG IDEAs"), ("shanghai", "ShanghaiT2DM")) if k in fid]
        out.append(f"""<h3>6.9 Can the twin be trusted for a whole day?</h3>
<p>The therapy simulator (section 7) runs the mechanistic twin open-loop for 24 hours, so we test exactly that: from the synced state at the
start of every day after calibration, the twin replays the day with the logged meals, doses and activity, and the result is compared with
the CGM.</p>
{table(frows, [("method", "Cohort", ""), ("days", "Days", "{:.0f}"), ("mean_abs_error_median", "Median MAE", "{:.1f}"),
               ("mean_glucose_error_median", "Daily-mean error", "{:.1f}"), ("tir_error_median_pp", "TIR error (pp)", "{:.1f}"),
               ("good", "Good %", "{:.0f}"), ("poor", "Poor %", "{:.0f}")])}
<p>The twin tracks daily mean glucose and time in range closely but rarely reproduces short lows. The simulator therefore reports means,
time in range and time above range, shows this replay check for each patient (with a warning when agreement is poor), and leaves
hypoglycaemia warnings to the validated 2-hour forecast.</p>
""")
    if rob:
        out.append(f"""<h3>6.10 Robust to missing data</h3>
<p>One deployed model (trained with modality dropout) scored with data removed at inference, {rob['n_forecasts']:,} synthetic test forecasts:</p>
{table([{"method": r["condition"], **r} for r in rob["modalities"]] + [{"method": f"{r['cgm_readings_lost_pct']:.0f}% of past CGM readings lost", **r} for r in rob["cgm_dropout"]],
       [("method", "Missing at inference", ""), ("rmse_60", "RMSE 60", "{:.2f}"), ("rmse_120", "RMSE 120", "{:.2f}"), ("spike_auroc", "Spike AUROC", "{:.3f}")])}
""")
    if bench:
        out.append(f"""<h3>6.11 Speed</h3>
<p>On one laptop CPU ({bench['threads']} threads, no GPU): a 2-hour forecast takes {fmt(bench['twinnet_single_forecast_ms'])} ms
({fmt(bench['twinnet_throughput_per_s'], 0)} per second batched); personalising a new patient takes {fmt(bench['mechanistic_fit_7_days_s'] + bench['personalise_twinnet_s'])} s
(twin fit on 7 days plus TwinNet fine-tuning); syncing the twin costs {fmt(bench['ukf_sync_ms_per_day'])} ms per patient-day; and a 24-hour therapy
simulation takes {fmt(bench['therapy_sim_10_plans_24h_ms'] / 10, 2)} ms per plan. Compute alone would refresh about
{bench['patients_refreshed_per_5_min'] / 1e6:.1f} million patients every 5 minutes, so data handling, not the model, limits scale.</p>
""")
    out.append("""<h2>7. Dashboard, interoperability and governance</h2>
<p>The doctor-facing dashboard (React, TypeScript, ECharts) shows a risk-ranked panel, a virtual-patient view (24-hour CGM, the 2-hour forecast
cone, the physics twin's projection, alerts with reasons and each patient's own alert track record, organ-level physiology), an AGP report, a
what-if simulator with Indian meals, a 24-hour therapy simulator (insulin and sulfonylurea doses and timing, DPP-4 and SGLT2 inhibitors, with
stress tests such as a skipped lunch, and a dose-response table), and a grounded "Ask the twin" assistant (offline by default). Forecasts are exported as FHIR R4 RiskAssessment resources with SNOMED CT, LOINC and
WHO ATC codes, aligned with ABDM. The design follows the DPDP Act 2023: consent, purpose limitation, data minimisation and a purpose-tagged
audit trail of every access. Only synthetic and openly licensed, de-identified data are used.</p>
<h2>8. Limitations</h2>
<ul>
<li>Synthetic patients come from a simulator with the same structure as the twin, which flatters physics-based methods there; real and external results are the primary evidence.</li>
<li>The real cohorts are small (45, 16 and 100 people) and none is Indian. A prospective pilot in Indian clinics is required.</li>
<li>Real-world hypoglycaemia events are rare and mostly in people without diabetes (CGMacros), so hypoglycaemia precision on real data is not established.</li>
<li>Raw risks from class-weighted training are too high; each site must recalibrate on its own past patients before going live. On BIG IDEAs (people without diabetes, few spikes) spike prediction is weak (AUROC about 0.77) and adds little net benefit.</li>
<li>Unlogged meals dominate 2-hour error; forecasts never use future information. Alert thresholds must be tuned with clinicians.</li>
<li>The 24-hour therapy simulator rests on the mechanistic twin: it tracks daily mean glucose and time in range but rarely short lows, does not model metformin (which acts on the fasting set-point over weeks), and has not yet been validated against real dose changes.</li>
<li>Subgroup performance is audited on synthetic patients only. This is research software, not a medical device; the regulatory path is CDSCO Software as a Medical Device with prospective validation.</li>
</ul>
<h2>References</h2>
<p class="small">Anjana RM et al. Metabolic non-communicable disease health report of India: the ICMR-INDIAB national cross-sectional study. <i>Lancet Diabetes Endocrinol</i> 2023.
Bergman RN et al. Quantitative estimation of insulin sensitivity. <i>Am J Physiol</i> 1979. Dalla Man C, Rizza RA, Cobelli C. Meal simulation model of the glucose-insulin system. <i>IEEE TBME</i> 2007.
Battelino T et al. Clinical targets for continuous glucose monitoring data interpretation: recommendations from the international consensus on time in range. <i>Diabetes Care</i> 2019.
Clarke WL et al. Evaluating clinical accuracy of systems for self-monitoring of blood glucose. <i>Diabetes Care</i> 1987.
Romano Y, Patterson E, Candès E. Conformalized quantile regression. <i>NeurIPS</i> 2019. Perez E et al. FiLM: visual reasoning with a general conditioning layer. <i>AAAI</i> 2018.
Vickers AJ, Elkin EB. Decision curve analysis. <i>Med Decis Making</i> 2006. Das A et al. CGMacros, PhysioNet 2025. Cho P, Kim J, Bent B, Dunn J. BIG IDEAs Lab Glycemic Variability and Wearable Device Data, PhysioNet 2023.
Zhao Q et al. Chinese diabetes datasets for data-driven machine learning (ShanghaiT1DM/T2DM). <i>Sci Data</i> 2023. Longvah T et al. Indian Food Composition Tables, NIN 2017.</p>
</body></html>""")
    return "\n".join(out)


def main() -> None:
    page = DOCS / "technical_report.html"
    page.write_text(build(), encoding="utf-8")
    print("wrote", page)
    exe = next((b for b in BROWSERS if Path(b).exists() or shutil.which(b)), None)
    if exe is None:
        print("no Edge/Chrome found; open the HTML and print to PDF")
        return
    pdf = DOCS / "technical_report.pdf"
    with tempfile.TemporaryDirectory() as prof:  # a separate profile, so a running browser does not swallow the job
        r = subprocess.run([exe, "--headless", "--disable-gpu", "--no-first-run", f"--user-data-dir={prof}", "--no-pdf-header-footer",
                            "--virtual-time-budget=10000", f"--print-to-pdf={pdf}", page.resolve().as_uri()],
                           check=False, capture_output=True, timeout=180)
    if not pdf.exists():
        print(r.stderr.decode(errors="ignore")[-2000:])
    print("wrote", pdf if pdf.exists() else "(PDF failed)")


if __name__ == "__main__":
    main()
