# Model card: MadhuTwin hybrid digital twin (v0.1, proof of concept)

## What it does

For one person with type 2 diabetes or prediabetes, MadhuTwin maintains a virtual replica that

1. **models** their glucose–insulin physiology (a mechanistic model personalised to them),
2. **stays in sync** with their CGM every 5 minutes (Unscented Kalman Filter), and
3. **simulates** the next 2–4 hours, both as a forecast and under "what-if" scenarios.

On top of the twin, TwinNet (a neural network) forecasts glucose 5–120 minutes ahead with an 80% interval and gives the probability of a hyperglycaemic excursion (>180 mg/dL for at least 15 min) or hypoglycaemia (<70 mg/dL for at least 15 min) in the next 2 hours. A LightGBM model gives the plain-language reasons (SHAP).

## Intended use

- **Users:** clinicians (diabetologists, physicians, diabetes educators) reviewing patients remotely.
- **Use:** decision support. It helps prioritise patients, anticipate excursions, explain them, and explore lifestyle scenarios with the patient.
- **Out of scope:** autonomous treatment decisions, insulin dose calculation, type 1 diabetes, pregnancy, children, inpatient or ICU glucose management. The what-if insulin control is educational only.

## Components

| Component | Method | Trained or fitted on |
|---|---|---|
| Physiology model | Extended Bergman minimal model: endogenous secretion, incretin effect, meal absorption by GI/fat/fibre, exercise, sleep-driven insulin sensitivity, dawn phenomenon, counter-regulation, drug effects (metformin, sulfonylureas, DPP-4i, SGLT2i, premixed and basal insulin) | Calibrated against OGTT physiology; population relationships estimated from 45 real people (CGMacros) |
| Personalisation | MAP fit of 5 parameters (fasting set-point, insulin sensitivity, beta-cell response, dawn effect, absorption speed), with EHR-derived priors; multiple-shooting objective | The patient's first days of data (7 days synthetic, first half of the record for real data) |
| Synchronisation | Unscented Kalman Filter on 7 states, including a drifting log insulin-sensitivity multiplier | Runs online |
| TwinNet | Causal dilated TCN over 6 h × 21 channels; EHR encoder with FiLM conditioning; physics-gated residual; quantile and event heads; modality dropout; conformal intervals | 700 synthetic patients, then fine-tuned on real data (sim-to-real) |
| Per-person fine-tuning | A few epochs of TwinNet fine-tuning (learning rate 3e-4) on the person's own earlier windows only | Each patient's own first days |
| Therapy simulator | 24-h open-loop simulation from the synced state; yesterday's meals and activity replayed (or a stress pattern: skipped lunch, late or festival dinner); insulin and sulfonylurea doses and timing, DPP-4 and SGLT2 inhibitors; sulfonylurea effect relative to usual exposure; per-patient trust check (replay of the last 24 h against the CGM, rated good / fair / poor) | – |
| MadhuTwin ensemble | Mean of the personalised TwinNet median and the LightGBM median; mean of the two event probabilities, recalibrated (Platt scaling) because class-weighted training inflates raw risks | Recalibration: validation patients (synthetic); each real-data site's own past patients (cross-fitted by fold in evaluation) |
| LightGBM fusion and explanations | LightGBM quantile regressors and event classifiers; TreeSHAP reasons | Synthetic + CGMacros + BIG IDEAs |

## Data

- **Synthetic India-calibrated cohort:** 1,000 patients × 14 days. EHR comes from Indian epidemiology (ICMR-INDIAB prevalence, South-Asian BMI phenotype, Indian prescribing patterns, regional diets from IFCT 2017). Wearable streams are simulated from the same physiology, so the two streams are causally consistent. Labs are measured from the simulated body; for example, HbA1c follows from mean glucose via the ADAG relation.
- **CGMacros** (PhysioNet, CC BY-NC-SA 4.0): 45 US adults (healthy, prediabetic, T2D), Dexcom/Libre CGM, Fitbit, meal macros, labs.
- **ShanghaiT2DM** (Figshare, CC BY 4.0): 100 Chinese T2D patients (109 recordings), 15-minute CGM, diet, insulin and oral drugs, labs. It has no wearables, so it doubles as the missing-modality external test.
- **BIG IDEAs Glycemic Variability and Wearable Device Data** (PhysioNet, ODC-By 1.0): 16 US adults with normal glucose or prediabetes, about 10 days of Dexcom G6 CGM with an Empatica E4 wristband (heart rate, inter-beat intervals from which we compute 5-minute RMSSD HRV, accelerometry) and food logs. It is the only cohort with real HRV.

## Evaluation (see `docs/evaluation_report.md`)

- **Splits:** patient-level throughout. Synthetic: 70/10/20 split by patient, evaluated only on days 8–14 after personalisation. Real data: 5-fold cross-validation grouped by patient.
- **Forecast metrics:** RMSE, MAE and MARD at 30/60/90/120 min; Clarke Error Grid; 80% interval coverage.
- **Event metrics:** AUROC and AUPRC with patient-bootstrap CIs; share of excursions caught, ≥30 min ahead and median lead time; false-alert episodes per day at a threshold chosen on validation data (training folds for real data); the share of excursions caught at the best operating point with at most one false alert per patient-day (a point on the alert-burden curve, like sensitivity at a fixed specificity).
- **Clinical utility:** calibration curves and decision-curve net benefit for spike alerts; error by subgroup (sex, age, BMI by Asian cut-offs, glycaemic status, therapy, region); a learning curve of error against days of personal data.
- **Ablations:** stream-by-stream (LightGBM) and leave-one-stream-out (TwinNet).
- **Twin validity:** fitted parameters against labs on real people, and illness detection from the synced insulin-sensitivity trace.

## Known limitations and risks

- The synthetic data generator and the mechanistic twin share a model structure, which flatters physics-based methods on synthetic data. Real and external results are the primary evidence.
- The real cohorts are small (45, 16 and 100 people) and none is Indian. Prospective validation in Indian clinics is required.
- Synthetic-only models do not transfer to a new population without fine-tuning (ShanghaiT2DM zero-shot is worse than persistence). Pretraining on synthetic data plus fine-tuning on a few real patients is the supported regime; it also improves spike discrimination over real-only training.
- Hypoglycaemia on CGMacros is rare and occurs mostly in people without diabetes (often likely sensor artefacts), so real-world hypoglycaemia precision is low and not yet established.
- Unlogged meals are the main source of 2-hour error. Forecasts only use meals logged up to the moment of prediction.
- CGM sensor bias differs between devices (CGMacros Dexcom vs Libre disagree by up to 40 mg/dL); the twin's set-point absorbs a constant bias.
- Subgroup error is audited on synthetic patients only; the real cohorts are too small for a subgroup audit.
- Per-person fine-tuning needs at least a few days of CGM; for a brand-new patient the population model is used.
- Alert thresholds trade sensitivity against alert fatigue and must be tuned with clinicians.
- The 24-hour therapy simulator rests on the mechanistic twin: it tracks daily mean glucose and time in range but rarely short lows, does not model metformin (which acts on the fasting set-point over weeks), and has not yet been validated against real dose changes. Its output is decision support for discussion with the care team, not a dosing calculator.

## Regulatory note

In India, clinical deployment would be Software as a Medical Device under the CDSCO Medical Device Rules 2017. That means a risk classification (likely Class B/C for decision support on chronic disease), a quality management system (ISO 13485), clinical performance evaluation, and post-market surveillance. This proof of concept makes no clinical claims.
