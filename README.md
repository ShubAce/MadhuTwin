# MadhuTwin: a hybrid digital twin for Type 2 Diabetes

**Happiest Health Digital Twin Challenge 2026 · Phase 1 submission**

MadhuTwin builds a living virtual replica of a person with type 2 diabetes. It fuses their electronic health record with wearable streams, simulates their glucose–insulin physiology, stays synchronised with their CGM every 5 minutes, and **warns a doctor up to 2 hours before a glucose spike or a hypoglycaemic episode**, with the reasons and "what-if" simulations to act on it.

<!-- RESULTS_HEADLINE -->

| | |
|---|---|
| Video (≥ 20 min) | **TODO(team): unlisted YouTube link** |
| Architecture diagram | [`docs/architecture.pdf`](docs/architecture.pdf) · [`docs/architecture.pptx`](docs/architecture.pptx) |
| Presentation | [`docs/presentation.pdf`](docs/presentation.pdf) · [`docs/presentation.pptx`](docs/presentation.pptx) |
| Evaluation report | [`docs/evaluation_report.md`](docs/evaluation_report.md) |
| Model card · Privacy (DPDP) | [`docs/model_card.md`](docs/model_card.md) · [`docs/dpdp_compliance.md`](docs/dpdp_compliance.md) |

---

## 1. Team details

| | |
|---|---|
| Team name | **TODO(team)** |
| College / incubator | **TODO(team)** |
| Team leader | **TODO(team)**: name, email, phone (submission form only) |
| Members | **TODO(team)**: name, programme, role |

## 2. Project title

**MadhuTwin: a hybrid physiological + AI digital twin that fuses EHR and wearable data to predict glycaemic events in people with Type 2 Diabetes in India.**

*Madhumeha* is the classical Indian name for diabetes.

## 3. Problem statement and healthcare use case

India has about **101 million people with diabetes and 136 million with prediabetes** (ICMR-INDIAB, *Lancet Diabetes & Endocrinology* 2023). Care is reactive: a quarterly HbA1c shows what already happened, while day-to-day risk sits unseen between visits. That risk includes post-meal spikes from high-carbohydrate meals, night-time lows from sulfonylureas or premixed insulin, and loss of control during illness. CGMs and smartwatches now produce the data to see it, but a clinician cannot watch hundreds of streams.

**Use case: remote monitoring by a diabetes care team.** For every patient, MadhuTwin:

1. **Predicts** glucose for the next 5–120 minutes with an 80% interval, and the probability of a sustained excursion above 180 mg/dL or below 70 mg/dL within 2 hours.
2. **Prioritises** the panel by risk, raising alerts with plain-language reasons. Example: *"62 g carbohydrate still being absorbed · Rising 18 mg/dL over 30 min · Insulin sensitivity 34% below personal baseline"*.
3. **Explains and simulates.** The doctor tests "what if she walks 15 minutes after dinner?", "swap rice-sambar for jowar roti?" or "what if he is unwell?" on the patient's own twin.
4. **Detects hidden change.** The synced twin tracks insulin sensitivity day to day, flagging illness or stress from CGM alone.
5. **Interoperates.** It exports FHIR R4, including a `RiskAssessment` resource carrying the forecast, aligned with the ABDM terminology stack.

## 4. What makes it a digital twin

A predictive model alone is not a twin. MadhuTwin implements the three defining properties:

| Property | Implementation |
|---|---|
| **Model**: a virtual replica of the patient's physiology | Extended Bergman minimal model: endogenous insulin secretion, incretin effect, meal absorption by GI/fat/fibre, exercise, sleep-driven insulin sensitivity, dawn phenomenon, counter-regulation, and Indian prescribing (metformin, sulfonylureas, DPP-4i, SGLT2i, premixed 30/70 and basal insulin). Personalised by MAP fitting of 5 parameters, with priors from the EHR. |
| **Sync**: continuously updated from real-time data | An Unscented Kalman Filter updates hidden state (plasma glucose, insulin, insulin action) and a drifting insulin-sensitivity factor from every CGM reading. |
| **Simulate**: run forward to predict and explore | Strictly causal 2-hour forecasts and counterfactual what-ifs (meals from an Indian food table, walks, insulin, sleep, illness). |

**TwinNet** sits on top: a physics-gated fusion network that combines both data streams with the twin's own forecast and learns *how much to trust physics at each horizon*.

## 5. Two data streams, fused

| Stream | Content | Sources |
|---|---|---|
| **Static / historical (EHR)** | Demographics, diagnoses (SNOMED CT), labs (LOINC: HbA1c, FPG, fasting insulin, lipids, eGFR, uACR, B12, TSH), medications with timing (WHO ATC), 2-year lab history, family history, **genetic markers** (TCF7L2 rs7903146, T2D polygenic risk score) | Synthetic India cohort (FHIR R4), CGMacros labs, ShanghaiT2DM clinical records |
| **Dynamic (wearables / IoT)** | CGM (5- or 15-min), heart rate, HRV (RMSSD), steps, METs, sleep stages, logged meals and doses | Synthetic device models (Apple Health / Libre-style), CGMacros (Dexcom, Libre, Fitbit), ShanghaiT2DM (CGM only) |

**Data used (sandbox rules respected: synthetic and open data only)**

- **Synthetic India-calibrated cohort, 1,000 patients × 14 days.** EHR drawn from Indian epidemiology (ICMR-INDIAB, South-Asian BMI phenotype, regional diets from IFCT 2017, Indian prescribing). Wearable streams are *generated by the same physiology*, so the two streams are causally coupled; labs are measured from the simulated body (HbA1c from mean glucose via ADAG). Physiological parameter distributions come from twins fitted to 45 real people.
- **[CGMacros](https://physionet.org/content/cgmacros/1.0.0/)** (PhysioNet, CC BY-NC-SA 4.0): 45 adults (healthy, prediabetic, T2D) with two CGMs, Fitbit, meal macros and labs. This is the real-world validation set.
- **[ShanghaiT2DM](https://figshare.com/articles/dataset/20444397)** (Figshare, CC BY 4.0): 100 T2D patients (109 recordings) on a rice-based diet with heavy insulin use and *no wearables*. This is the external, missing-modality test.

## 6. Technical stack, AI/ML models and frameworks

| Layer | Stack |
|---|---|
| Physiology and sync | Python 3.12, NumPy, SciPy (Powell MAP fit), Numba-compiled ODE integrator (verified equal to the NumPy reference; 12× faster end-to-end per patient), custom vectorised UKF |
| ML | PyTorch (TwinNet: causal dilated TCN + FiLM EHR conditioning + physics-gated residual + quantile and event heads + modality dropout), conformalised quantile intervals; LightGBM (quantile regressors, event classifiers) + TreeSHAP for explanations |
| Health standards | HL7 FHIR R4 (validated with `fhir.resources`), SNOMED CT, LOINC, UCUM, WHO ATC, dbSNP |
| Serving | FastAPI, WebSocket live replay, purpose-tagged audit trail; optional grounded LLM assistant (Claude via the Anthropic SDK, tool use over the twin; off by default) |
| Dashboard | React 18 + TypeScript + Vite, Tailwind CSS 4, Apache ECharts (palette validated for colour-vision deficiency; light and dark themes) |
| Ops | Docker (multi-stage), docker-compose, GitHub Actions CI (pytest + ruff + dashboard build) |

**Evaluation protocol.** Patient-level splits throughout. Synthetic: 70/10/20 by patient, scored only on days 8–14 after personalisation. Real data: 5-fold CV grouped by patient. Metrics: RMSE / MAE / MARD per horizon, Clarke Error Grid, 80% interval coverage, AUROC / AUPRC with patient-bootstrap CIs, excursions caught, lead time and false alerts per day. Stream-by-stream ablations included.

<!-- RESULTS_TABLE -->

## 7. Run it

**Option A: Docker** (dashboard + API with the prebuilt demo cohort)

```bash
docker compose up --build
# open http://localhost:8000
```

**Option B: local development**

```bash
python -m venv .venv && .venv/Scripts/activate          # Windows (use source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt && pip install -e . --no-deps
uvicorn twin.service.app:app --port 8000                 # API, serving artifacts/demo
cd dashboard && npm install && npm run dev                # http://localhost:5173
```

**Reproduce everything from scratch** (about 2–3 hours on a 16-core CPU): `python scripts/run_all.py`, or step by step:

```bash
python scripts/download_data.py        # CGMacros (CSV members only, via HTTP range requests) + ShanghaiT2DM
python scripts/fit_twins.py            # personal twins for the 45 CGMacros participants
python scripts/generate_cohort.py      # synthetic India cohort: 1,000 patients x 14 days + FHIR bundles
python scripts/build_dataset.py        # fused datasets + twin features for all three sources
python scripts/exp_synthetic.py --twinnet-ablations   # benchmark, ablations, illness detection
python scripts/exp_real.py             # CGMacros CV, Shanghai external validation, production models
python scripts/exp_cgm_light.py        # CGM-light affordability mode
python scripts/build_demo.py           # dashboard demo cohort
python scripts/make_report.py          # figures + docs/evaluation_report.md
pytest -q
```

Optional "Ask the twin" LLM layer: set `MADHUTWIN_LLM=on` and `ANTHROPIC_API_KEY`. Without them, a grounded offline engine answers.

## 8. Repository layout

```
twin/physiology/   mechanistic model (reference + Numba), meal/drug inputs, Indian food table, personal fitting
twin/sync/         Unscented Kalman Filter
twin/twin.py       DigitalTwin: personalise · sync · forecast · what-if
twin/ehr/          India-calibrated population sampler, clinical codes, FHIR R4 export
twin/sensors/      lifestyle simulator (meals, meds, sleep, activity, illness) and device models (CGM, HR, HRV)
twin/ingest/       CGMacros and ShanghaiT2DM loaders -> common schema
twin/data/         fused model inputs (21 dynamic channels + EHR vector) and physics features
twin/models/       TwinNet (PyTorch), LightGBM forecaster, tabular features
twin/eval/         metrics (Clarke grid, lead time, false alerts), experiment helpers
twin/service/      FastAPI app, demo store, explanations, assistant (offline + optional LLM)
dashboard/         React doctor dashboard
scripts/           end-to-end pipeline
docs/              architecture, presentation, evaluation report, model card, DPDP note, video script
tests/             physiology, pipeline, API tests
```

## 9. Limitations

- Synthetic patients come from a simulator with the same structure as the twin, which flatters physics-based methods there. Real and external validation are the primary evidence.
- Real cohorts are small (45 + 100 people) and none is Indian. The next step is a prospective pilot with Indian clinics using CGM and wearables.
- Unlogged meals dominate 2-hour error. Forecasts never use future information.
- This is research software, not a medical device. See the model card for intended use and the regulatory path (CDSCO SaMD).

## 10. Licence

- **Code:** Apache License 2.0, see [`LICENSE`](LICENSE).
- **Data:** CGMacros (CC BY-NC-SA 4.0) and ShanghaiT2DM (CC BY 4.0) are downloaded by script and never committed raw. Demo bundles derived from them carry their licences. See [`NOTICE.md`](NOTICE.md). Synthetic patients are generated by this project; all names are fictitious.

## 11. Acknowledgements

CGMacros (Das et al., PhysioNet 2025); ShanghaiT1DM/T2DM (Zhao et al., *Scientific Data* 2023); ICMR-INDIAB study group; Indian Food Composition Tables (NIN, 2017); Bergman, Dalla Man and colleagues for the minimal-model lineage; Battelino et al. 2019 international consensus on time in range.
