# MadhuTwin — Project Plan

> Happiest Health Digital Twin Challenge 2026 · Working name *MadhuTwin* ("Madhumeha" = diabetes); rename freely.
> Status (2026-10-08): M0–M3 done; M4 done (what-if, explanations, CGM-light, Ask-the-twin with an optional LLM, Hindi/Kannada companion, consent and audit, external validation); M5 in progress (final models, report, deck, video script). Pending from the team: team details, video recording, GitHub repo.

---

## 1. The brief, decoded

| # | Requirement (from the brief)                                                                                   | How we satisfy it                                                                          |
| - | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| 1 | One chronic condition prevalent in India                                                                       | **Type 2 Diabetes**                                                                  |
| 2 | Static/historical stream: demographics, diagnoses, labs, genetic markers                                       | FHIR R4 synthetic EHR (India-calibrated) + real labs from CGMacros / ShanghaiT2DM          |
| 3 | Dynamic stream: HRV, CGM, sleep, steps                                                                         | Real CGM + HR + steps (CGMacros) + physiology-driven synthetic wearables (adds HRV, sleep) |
| 4 | Model ingests both and predicts an adverse event in advance                                                    | Hybrid twin: forecast glucose 30–120 min ahead + spike/hypo event probabilities           |
| 5 | Conceptual UI dashboard for a doctor                                                                           | Working React dashboard: triage → virtual patient → what-if simulator                    |
| 6 | Synthetic / open data only                                                                                     | CGMacros (open), ShanghaiT2DM (open), own generators; no raw data committed                |
| 7 | Public GitHub repo, folder`TeamName_CollegeName`                                                             | Repo + top-level folder named accordingly                                                  |
| 8 | README: team, title, problem, stack/model, ≥20-min video, license, architecture PDF/PPT, presentation PDF/PPT | See §14 checklist                                                                         |
| 9 | Phase 2 live virtual demo                                                                                      | Deployed URL + one-command local run + recorded backup                                     |

**How we win.** Most teams will submit a classifier and a Streamlit page. A *digital twin* has three properties a classifier lacks, and we build all three:

1. **Model.** A mechanistic, personalised model of the patient's glucose–insulin physiology.
2. **Sync.** The model is continuously re-calibrated from live sensor data (state estimation).
3. **Simulate.** It can be run forward under hypothetical inputs ("what if she eats biryani at 9 pm and walks 15 min after?").

---

## 2. Why Type 2 Diabetes

- India has ~101 M people with diabetes and ~136 M with prediabetes (ICMR-INDIAB, Lancet D&E 2023). This is the strongest India story.
- The brief's own example is a 2-hour glucose-spike prediction, a crisp, measurable target.
- Open real data contains **both** streams in the same people (see §4).
- The physiology is well modelled (Bergman minimal model, Dalla Man meal model), which enables a real mechanistic twin.
- India-specific angles: high-carb diets, the "thin-fat" South-Asian phenotype (insulin resistance at lower BMI), wide sulfonylurea use (hypoglycaemia risk), and CGM cost barriers.

**Prediction targets**

| Target               | Definition                                               | Horizon             |
| -------------------- | -------------------------------------------------------- | ------------------- |
| Glucose trajectory   | Quantiles P10/P50/P90                                    | 30, 60, 90, 120 min |
| Hyperglycaemic spike | CGM > 180 mg/dL (level-2: > 250)                         | within 2 h          |
| Hypoglycaemia        | CGM < 70 mg/dL (level-2: < 54)                           | within 2 h          |
| Weekly deterioration | Time-in-range drop + nocturnal HR↑ / HRV↓ + sleep debt | 7-day window        |

Honest caveat for the jury: accuracy at 2 hours depends on meal information. The twin uses logged meals and learns each patient's meal routine, and we report metrics per horizon.

---

## 3. Architecture

```
 ┌────────────── DATA LAYER ───────────────┐   ┌──────────────── TWIN CORE ────────────────┐   ┌──────── EXPERIENCE ────────┐
 │ EHR generator ──► FHIR R4 bundles       │   │  Mechanistic model (minimal model +       │   │ Doctor dashboard (React)   │
 │  (Patient, Condition, Observation,      │──►│  meal absorption + exercise + sleep)      │──►│  • Triage panel            │
 │   MedicationStatement, FamilyHistory,   │   │        ▲ params from EHR                  │   │  • Virtual patient view    │
 │   genetic markers)                      │   │        │                                  │   │  • AGP report              │
 │                                         │   │  UKF / particle filter  ◄── live CGM      │   │  • What-if simulator       │
 │ Wearable simulator ─┐                   │   │  (sync: re-estimates insulin sensitivity) │   │  • Talk-to-twin assistant  │
 │ CGMacros / Shanghai ├─► common schema ──┼──►│                                           │   │ Patient companion (PWA,    │
 │                     │   (Parquet)       │   │  Fusion forecaster (static enc + temporal │   │  EN/HI/KN alerts)          │
 │ Replay engine ──────┴─► WebSocket stream│   │  enc + physics forecast) → quantiles +    │   │ FHIR RiskAssessment export │
 └─────────────────────────────────────────┘   │  event heads → SHAP → plain-language why  │   └────────────────────────────┘
                                               │  What-if engine (counterfactual rollouts) │
                                               └───────────────────────────────────────────┘
          Governance across all layers: consent registry · pseudonymisation · audit log · model card (DPDP Act 2023)
```

---

## 4. Data strategy

| Source                                                                                                        | What it gives                                                                                                                                                     | Use                                                         | License / note                                                   |
| ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------- |
| [CGMacros](https://physionet.org/content/cgmacros/1.0.0/) (PhysioNet, 2025)                                    | 45 adults (15 healthy, 16 prediabetic, 14 T2D), ~10 days; Dexcom G6 + Libre Pro CGM, Fitbit HR/steps/METs, meal macros; labs: HbA1c, FPG, fasting insulin, lipids | Primary real training/validation                            | CC BY-NC-SA 4.0, so we do not redistribute; download script only |
| [ShanghaiT2DM](https://figshare.com/articles/dataset/Diabetes_Datasets-ShanghaiT1DM_and_ShanghaiT2DM/20444397) | 100 T2D patients, CGM (15-min), labs, meds, diet                                                                                                                  | **External validation** (different population/device) | Verify license before use                                        |
| Synthetic EHR generator (ours)                                                                                | 1,000+ India-calibrated patients as FHIR R4; SNOMED/LOINC codes; meds; family history; TCF7L2 rs7903146 + polygenic risk score                                    | Pretraining, demo cohort, genetic markers                   | Ours (Apache-2.0)                                                |
| Synthetic wearable simulator (ours)                                                                           | 5-min CGM, minute HR, HRV (RMSSD), sleep stages, steps in Apple Health / Google Fit / Libre-style formats, with noise, gaps and compression lows                  | Pretraining, adds HRV/sleep, live demo                      | Ours                                                             |

**Coupling rule (key technical point).** The simulated wearable streams are generated *from* the twin parameters implied by the EHR, so the two streams are causally consistent:

- HbA1c ↔ mean CGM (ADAG: eAG = 28.7 × A1c − 46.7)
- HOMA-IR ↔ post-meal spike size
- short sleep → next-day insulin resistance
- post-meal walk → blunted spike
- sulfonylurea → hypo risk

We validate realism by comparing synthetic and CGMacros distributions side by side.

**Sim-to-real.** Pretrain on the synthetic cohort, fine-tune on CGMacros, test externally on ShanghaiT2DM.

---

## 5. Modelling

1. **Baselines:** persistence, linear extrapolation, ARIMA, LightGBM on engineered features.
2. **Mechanistic twin:** Bergman minimal model + two-compartment meal absorption + exercise and sleep modifiers. Parameters are initialised from EHR (HOMA-IR, HbA1c, BMI, meds).
3. **Sync:** an Unscented Kalman Filter (or particle filter) re-estimates glucose state and insulin sensitivity from each CGM reading. This is the "constantly updated" property of a twin.
4. **Fusion forecaster (PyTorch):**
   - temporal encoder (TCN/Transformer) over the last 6–12 h of multichannel signals: CGM, HR, HRV, steps, sleep stage, carbs on board, meds, time-of-day
   - static encoder (MLP over the EHR vector)
   - FiLM conditioning + cross-attention
   - the mechanistic forecast as an extra input (physics-informed residual)
   - heads: quantiles at 4 horizons + spike/hypo probabilities
   - loss: pinball + BCE; conformal calibration for guaranteed interval coverage
5. **Personalisation:** per-patient latent parameters from the UKF + light fine-tuning, so the twin learns *this* patient (e.g., spikes on white rice, not on millets).
6. **Explainability:** SHAP / attention attributions turned into a sentence, e.g. *"High-carb dinner (92 g) + 4.1 h sleep + HbA1c 8.9% → 78% spike risk in ~95 min."*
7. **Optional benchmark:** a time-series foundation model (e.g., Chronos) zero-shot vs. ours.

---

## 6. Evaluation protocol

- **Splits:** patient-level (no leakage); 5-fold grouped CV on CGMacros; ShanghaiT2DM as external test.
- **Forecast metrics:** RMSE, MAE, MARD per horizon; **Clarke Error Grid** (% in zones A+B).
- **Event metrics:** AUROC, AUPRC, sensitivity at ≤ 1 false alarm/day, median lead time.
- **Ablation:** EHR-only · wearables-only · fused · fused + physics. This proves the stream fusion matters, which is the core ask.
- **Trust checks:** calibration (reliability plots, interval coverage); subgroup performance by sex/age/BMI; 95% CIs via bootstrap.
- **Synthetic realism:** distribution comparisons vs. CGMacros.

---

## 7. Doctor dashboard (screens)

1. **Panel triage.** All patients ranked by current risk, with sparkline, TIR and last alert.
2. **Virtual patient.**
   - EHR card (diagnoses, labs, meds, genetic risk)
   - stylised body view (pancreas/heart/liver status)
   - live CGM with a 2-hour forecast cone
   - HR/HRV, sleep hypnogram, steps
   - alerts with lead time and the "why" sentence
3. **AGP report.** The international standard CGM report: TIR/TAR/TBR, GMI, CV%, modal-day percentile plot.
4. **What-if simulator.** Indian meal presets (IFCT 2017: idli-sambar, rice-dal, roti-sabzi, biryani, ragi mudde…), post-meal walk, medication timing/dose, sleep. Shows baseline vs. scenario curves.
5. **Talk-to-twin.** An LLM assistant that answers doctor questions using tool calls into the twin API (forecast, what-if, labs) and cites data. It does not prescribe. Has an offline fallback.
6. **Actions.** Acknowledge alert, add a note, message the patient, export FHIR `RiskAssessment` + `Observation`.

**Patient companion (PWA).** Alerts and nudges in English / Hindi / Kannada (e.g., "15-min walk after lunch").

---

## 8. Differentiators (beyond the brief)

- A true twin (model + sync + simulate), not just a classifier
- EHR and wearables causally coupled in simulation
- Interoperability: FHIR R4 / ABDM-ready, with SNOMED CT + LOINC
- Indian food database in the what-if engine
- **CGM-light mode:** a 14-day CGM calibration period, after which the twin runs on smartwatch data + occasional fingersticks. This addresses CGM cost in India, and we evaluate it with CGM masked out.
- External validation on a second population
- DPDP Act compliance section, model card, SaMD/CDSCO regulatory-path note
- Clinician feedback: get 1–2 diabetologists to review the dashboard and quote them in the deck

---

## 9. Tech stack

| Area             | Choice                                                                                        |
| ---------------- | --------------------------------------------------------------------------------------------- |
| Core             | Python 3.12, NumPy/Pandas/Polars, SciPy, filterpy (UKF)                                       |
| ML               | PyTorch, LightGBM, scikit-learn, SHAP, MAPIE (conformal), MLflow (tracking)                   |
| Health standards | `fhir.resources` (FHIR R4), SNOMED CT / LOINC codes                                         |
| Storage          | Parquet + DuckDB (analytics), SQLite (app state, audit log)                                   |
| API              | FastAPI, Pydantic, WebSockets (live stream replay)                                            |
| Dashboard        | React + TypeScript + Vite, Tailwind + shadcn/ui, Apache ECharts, TanStack Query               |
| Assistant        | LLM with tool use (Claude API), with offline templated fallback                               |
| Ops              | Docker Compose, GitHub Actions CI (pytest, ruff, type-check, frontend build), live deployment |
| Docs             | Architecture diagram (PDF + PPTX), deck (PPTX + PDF), model card, video script                |

---

## 10. Repository layout

```
TeamName_CollegeName/
├── README.md                 # all mandatory submission sections
├── LICENSE                   # Apache-2.0
├── docs/                     # architecture.pdf/.pptx, presentation.pptx/.pdf, model_card.md,
│                             # dpdp_compliance.md, evaluation_report.md, video_script.md
├── data/                     # raw/ (gitignored), synthetic/ samples, download scripts
├── twin/                     # Python package
│   ├── ehr/                  # synthetic EHR generator + FHIR serialisation
│   ├── physiology/           # minimal model, meal, exercise, sleep modules
│   ├── sensors/              # wearable simulator + device-format exporters
│   ├── ingest/               # CGMacros / Shanghai loaders → common schema
│   ├── sync/                 # UKF / particle filter
│   ├── models/               # baselines, fusion forecaster, event heads
│   ├── explain/              # SHAP → natural-language reasons
│   ├── whatif/               # counterfactual rollouts, Indian food table
│   └── eval/                 # metrics, Clarke grid, ablations, reports
├── api/                      # FastAPI app, WebSocket stream, FHIR export
├── dashboard/                # React app (doctor)
├── companion/                # PWA (patient)
├── notebooks/                # EDA, results figures
├── tests/
├── docker-compose.yml
└── .github/workflows/ci.yml
```

---

## 11. Milestones (≈5 weeks; a submittable build exists from M3 onward)

| Milestone                | Scope                                                                                                                              | Exit criterion                                             |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **M0** (days 1–2) | Repo, license, CI, download scripts, EDA notebook on CGMacros                                                                      | Data loads; EDA figures                                    |
| **M1** (week 1)    | EHR generator (FHIR), physiology simulator, wearable simulator, common schema, loaders                                             | 1,000-patient synthetic cohort; realism plots vs. CGMacros |
| **M2** (week 2)    | Baselines, mechanistic twin + UKF sync, fusion forecaster, event heads, evaluation harness                                         | Ablation table + Clarke grid; fusion beats single-stream   |
| **M3** (week 3)    | FastAPI + WebSocket replay, dashboard core (triage, patient view, forecast cone, alerts, AGP)                                      | **Submittable MVP**, end-to-end demo                 |
| **M4** (week 4)    | What-if engine, explanations, CGM-light mode, talk-to-twin, patient companion (EN/HI/KN), consent + audit log, external validation | All differentiators demo-able                              |
| **M5** (week 5)    | Hardening, deployment, README, architecture PDF/PPTX, deck, model card, video script, rehearsal, code freeze                       | Submission checklist 100%                                  |

---

## 12. Suggested roles

- **Twin / ML lead:** physiology model, UKF, fusion model, evaluation
- **Data & standards lead:** generators, FHIR/ABDM, loaders, data validation
- **Frontend / UX lead:** dashboard, companion app, demo flow
- **Clinical & story lead:** clinical realism checks, clinician feedback, deck, video recording

---

## 13. Risks and mitigations

| Risk                           | Mitigation                                                                      |
| ------------------------------ | ------------------------------------------------------------------------------- |
| Small real dataset (45 people) | Synthetic pretraining, grouped CV, bootstrap CIs, external test set             |
| 2 h horizon is inherently hard | Meal-aware model, per-horizon reporting, honest framing                         |
| Synthetic data looks fake      | Physiology-driven generator, calibrated and compared against real distributions |
| Live demo fails                | Deployed URL + Docker one-command run + recorded backup video                   |
| Dataset licensing              | Never commit raw data; scripted download; attribution in README                 |
| Over-claiming clinical value   | Proof-of-concept framing, model card, limitations section, regulatory note      |

---

## 14. Submission checklist (README must contain)

- [ ] Team details + college/incubator
- [ ] Project title
- [ ] Problem statement + healthcare use case
- [ ] Tech stack, AI/ML model/framework details
- [ ] Video link (≥ 20 min, unlisted YouTube)
- [ ] Open-source license details (code Apache-2.0 + dataset licenses)
- [ ] Architecture diagram (PDF/PPT) in `docs/`
- [ ] Presentation (PDF/PPT) in `docs/`
- [ ] Repo public; folder named `TeamName_CollegeName`; all links accessible
- [ ] Submission form: team leader name, phone, email, repo link

**Video outline (~23 min):**

- Problem & India context (2)
- What a digital twin is (3)
- Data & fusion (4)
- Models & results (5)
- Live demo (6)
- Privacy, DPDP & regulatory (2)
- Impact & roadmap (1)

---

## 15. Open items

- [ ] Exact Phase 1 deadline date
- [ ] Team name, college/incubator, member names
- [ ] GitHub account/org for the public repo
- [ ] LLM API key for talk-to-twin (optional; offline fallback planned)
- [ ] 1–2 clinicians willing to review the dashboard
