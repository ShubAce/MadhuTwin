// Builds docs/presentation.pptx from experiment results, report figures and dashboard screenshots.
// Every number on a slide is read from artifacts/results at build time.
//   NODE_PATH=docs/deck/node_modules node docs/deck/build_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { THEME } = require("./theme");
const { applyTheme } = require(process.env.PPTX_SKILL_THEME || "./apply_theme_shim.js");

const ROOT = path.join(__dirname, "..", "..");
const FIG = path.join(ROOT, "docs", "figures");
const SHOTS = path.join(ROOT, "docs", "screenshots");
const OUT = path.join(ROOT, "docs", "presentation.pptx");
const C = THEME.colors;
const INK2 = "4A5260";
const MUTED = "7A8291";

const readJSON = (p) => (fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : null);
const syn = readJSON(path.join(ROOT, "artifacts", "results", "synthetic", "results.json"));
const real = readJSON(path.join(ROOT, "artifacts", "results", "real", "results.json"));
const light = readJSON(path.join(ROOT, "artifacts", "results", "cgm_light.json"));
const fitCsv = fs.readFileSync(path.join(ROOT, "artifacts", "twins", "cgmacros_twins_summary.csv"), "utf8").trim().split("\n");
const fitHead = fitCsv[0].split(",");
const fitRows = fitCsv.slice(1).map((l) => Object.fromEntries(l.split(",").map((v, i) => [fitHead[i], v])));
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const fitPrior = mean(fitRows.map((r) => Number(r.rmse_prior)));
const fitPers = mean(fitRows.map((r) => Number(r.rmse_fit)));

const f1 = (x) => (x == null || !isFinite(x) ? "–" : Number(x).toFixed(1));
const f0 = (x) => (x == null || !isFinite(x) ? "–" : Number(x).toFixed(0));
const f2 = (x) => (x == null || !isFinite(x) ? "–" : Number(x).toFixed(2));
const byMethod = (rows, m) => rows.find((r) => r.method === m) || {};
const ev = (rows, e, m) => rows.find((r) => r.event === e && r.method === m) || {};

const TN = byMethod(syn.forecast, "TwinNet hybrid");
const PERS = byMethod(syn.forecast, "Persistence");
const GBM = byMethod(syn.forecast, "LightGBM fusion");
const SPIKE = ev(syn.events, "spike", "TwinNet hybrid");
const HYPO = ev(syn.events, "hypo", "TwinNet hybrid");
const ILL = syn.illness_detection;
const gate = syn.gate_by_horizon;

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "MadhuTwin: a hybrid digital twin for Type 2 Diabetes";
  pres.author = "MadhuTwin team";
  pres.subject = "Happiest Health Digital Twin Challenge 2026";

  const footer = { text: { text: "MadhuTwin · Happiest Health Digital Twin Challenge 2026", options: { x: 0.5, y: 7.05, w: 8, h: 0.3, fontSize: 10, color: MUTED, fontFace: "Calibri" } } };
  pres.defineSlideMaster({
    title: "DARK",
    background: { color: C.dk2 },
    objects: [],
    placeholders: [],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: "FFFFFF" },
    objects: [footer],
    slideNumber: { x: 12.3, y: 7.05, w: 0.6, h: 0.3, fontSize: 10, color: MUTED, align: "right" },
    margin: [0.5, 0.5, 0.6, 0.5],
  });

  const addTitle = (s, text, sub) => {
    s.addText(text, { x: 0.5, y: 0.32, w: 10.6, h: 0.75, fontFace: "Cambria", fontSize: 30, bold: true, color: C.dk2, margin: 0, valign: "top", isTextBox: true, fit: "shrink" });
    if (sub) s.addText(sub, { x: 0.5, y: 1.08, w: 10.6, h: 0.4, fontSize: 15, color: INK2, margin: 0, isTextBox: true });
  };
  // Motif: the MODEL / SYNC / SIMULATE triad, with the current pillar highlighted
  const triad = (s, active) => {
    ["MODEL", "SYNC", "SIMULATE"].forEach((t, i) => {
      const on = t === active;
      s.addShape("roundRect", { x: 11.3 + i * 0.68, y: 0.38, w: 0.62, h: 0.42, rectRadius: 0.08, fill: { color: on ? C.dk2 : "E7EBF2" }, line: { color: on ? C.dk2 : "E7EBF2" }, objectName: `triad-${t}` });
      s.addText(t === "SIMULATE" ? "SIM" : t, { x: 11.3 + i * 0.68, y: 0.38, w: 0.62, h: 0.42, fontSize: 9, bold: true, align: "center", valign: "middle", color: on ? "FFFFFF" : "8A93A3", margin: 0, isTextBox: true });
    });
  };
  const stat = (s, x, y, w, value, label, color = C.dk2) => {
    s.addText(value, { x, y, w, h: 0.75, fontFace: "Calibri", fontSize: 40, bold: true, color, margin: 0, isTextBox: true, valign: "bottom" });
    s.addText(label, { x, y: y + 0.78, w, h: 0.6, fontSize: 13, color: INK2, margin: 0, isTextBox: true, valign: "top" });
  };
  const bullets = (s, items, opts) => {
    s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
      { fontSize: 15, color: "222833", paraSpaceAfter: 8, margin: 0, valign: "top", isTextBox: true, ...opts });
  };
  const img = (s, file, x, y, w, h) => {
    const p = fs.existsSync(path.join(FIG, file)) ? path.join(FIG, file) : path.join(SHOTS, file);
    if (fs.existsSync(p)) s.addImage({ path: p, x, y, w, h, sizing: { type: "contain", w, h }, objectName: file });
    else s.addText(`[missing ${file}]`, { x, y, w, h, color: C.accent6, isTextBox: true });
  };
  const card = (s, x, y, w, h, title, body, fill = "F2F5FA") => {
    s.addShape("roundRect", { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill } });
    s.addText([{ text: title, options: { bold: true, fontSize: 15, color: C.dk2, breakLine: true } }, { text: body, options: { fontSize: 13, color: INK2 } }],
      { x: x + 0.18, y: y + 0.12, w: w - 0.36, h: h - 0.24, valign: "top", margin: 0, paraSpaceAfter: 4, isTextBox: true });
  };
  const sec = (title) => pres.addSection({ title });

  // 1 · Title
  sec("Opening");
  let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Opening" });
  s.addText("MadhuTwin", { x: 0.8, y: 1.7, w: 11, h: 1.1, fontFace: "Cambria", fontSize: 60, bold: true, color: "FFFFFF", margin: 0, isTextBox: true });
  s.addText("A hybrid digital twin for Type 2 Diabetes that fuses EHR and wearable data to warn doctors up to 2 hours before glucose spikes and lows",
    { x: 0.8, y: 2.9, w: 10.5, h: 1.0, fontSize: 22, color: "DCE4F2", margin: 0, isTextBox: true });
  ["MODEL", "SYNC", "SIMULATE"].forEach((t, i) => {
    s.addShape("roundRect", { x: 0.8 + i * 1.75, y: 4.3, w: 1.6, h: 0.55, rectRadius: 0.1, fill: { color: i === 0 ? C.accent4 : "2A3F66" }, line: { color: i === 0 ? C.accent4 : "2A3F66" } });
    s.addText(t, { x: 0.8 + i * 1.75, y: 4.3, w: 1.6, h: 0.55, fontSize: 14, bold: true, align: "center", valign: "middle", color: i === 0 ? C.dk2 : "FFFFFF", margin: 0, isTextBox: true });
  });
  s.addText("Team: TODO(team) · College / incubator: TODO(team)", { x: 0.8, y: 5.6, w: 11, h: 0.4, fontSize: 16, color: "FFFFFF", margin: 0, isTextBox: true });
  s.addText("Happiest Health Digital Twin Challenge 2026 · Phase 1", { x: 0.8, y: 6.05, w: 11, h: 0.4, fontSize: 14, color: "AEBBD3", margin: 0, isTextBox: true });
  s.addNotes("Introduce the team. MadhuTwin, from Madhumeha, the classical Indian name for diabetes, is a living virtual replica of each patient: it models their physiology, stays synchronised with their CGM, and simulates the next hours to warn the care team early.");

  // 2 · Problem
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Opening" });
  addTitle(s, "Diabetes care in India is reactive; the risk lives between visits");
  stat(s, 0.5, 1.7, 3.6, "101 million", "adults in India live with diabetes");
  stat(s, 0.5, 3.4, 3.6, "136 million", "more have prediabetes");
  s.addText("Source: ICMR-INDIAB, Lancet Diabetes & Endocrinology 2023", { x: 0.5, y: 5.0, w: 3.8, h: 0.4, fontSize: 11, color: MUTED, margin: 0, isTextBox: true });
  card(s, 4.8, 1.7, 8.0, 1.25, "A quarterly HbA1c describes the past", "It cannot warn about tonight's low on a sulfonylurea or premixed insulin, or tomorrow's illness-driven rise.");
  card(s, 4.8, 3.1, 8.0, 1.25, "Post-meal spikes go unseen", "Rice- and roti-heavy meals, late dinners and festival sweets drive excursions that clinic visits never capture.");
  card(s, 4.8, 4.5, 8.0, 1.25, "Data exists, attention does not", "CGMs and smartwatches stream thousands of readings a day; no care team can watch hundreds of patients' streams.");
  s.addNotes("Set the scene: India's diabetes burden is enormous, care is visit-based and reactive. The dangerous moments, post-meal spikes, nocturnal lows, illness, happen between visits. Wearables now capture them, but clinicians need the signal, not the stream.");

  // 3 · Solution
  sec("The digital twin");
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "The digital twin" });
  addTitle(s, "MadhuTwin: a living virtual replica of each patient", "Not just a predictor: the three defining properties of a digital twin");
  const pillars = [
    ["MODEL", "Personalised glucose–insulin physiology", "An extended minimal model with meals, exercise, sleep, dawn effect and Indian therapies, fitted to each patient with EHR-based priors."],
    ["SYNC", "Continuously updated from live data", "An Unscented Kalman Filter re-estimates hidden insulin state and a drifting insulin sensitivity every 5 minutes."],
    ["SIMULATE", "Run forward to predict and explore", "Causal 2-hour forecasts and what-ifs: a different meal, a walk after dinner, a short night, an illness."],
  ];
  pillars.forEach(([t, h, d], i) => {
    const x = 0.5 + i * 4.15;
    s.addShape("roundRect", { x, y: 1.85, w: 3.95, h: 3.4, rectRadius: 0.1, fill: { color: i === 0 ? C.dk2 : "F2F5FA" }, line: { color: i === 0 ? C.dk2 : "F2F5FA" } });
    s.addText([
      { text: t, options: { fontSize: 22, bold: true, color: i === 0 ? C.accent4 : C.dk2, breakLine: true } },
      { text: h, options: { fontSize: 16, bold: true, color: i === 0 ? "FFFFFF" : "222833", breakLine: true } },
      { text: d, options: { fontSize: 14, color: i === 0 ? "DCE4F2" : INK2 } },
    ], { x: x + 0.25, y: 2.05, w: 3.45, h: 3.0, valign: "top", margin: 0, paraSpaceAfter: 8, isTextBox: true });
  });
  s.addText("On top: TwinNet, a physics-gated network that fuses both data streams with the twin's own forecast, and explains every alert in plain language.",
    { x: 0.5, y: 5.6, w: 12.3, h: 0.8, fontSize: 16, color: C.dk2, margin: 0, isTextBox: true });
  s.addNotes("Explain why this is a digital twin and not just a classifier: model, sync, simulate. TwinNet adds machine learning on top of the twin, and learns how much to trust the physics.");

  // 4 · Architecture
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "The digital twin" });
  addTitle(s, "Architecture: two data streams fused into the twin");
  img(s, "architecture.png", 0.5, 1.25, 12.3, 5.65);
  s.addNotes("Walk left to right: EHR and wearable streams are ingested into one schema; the twin core models, syncs and simulates; TwinNet produces forecasts and alerts; the clinician sees them in the dashboard and any EHR can receive them as FHIR. Governance spans everything.");

  // 5 · Data
  sec("Data");
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data" });
  addTitle(s, "Two data streams, three open or synthetic sources", "Sandbox rules respected: no real Indian patient data is used");
  card(s, 0.5, 1.75, 6.0, 2.25, "Static / historical: EHR", "Demographics · diagnoses (SNOMED CT) · labs incl. HbA1c, fasting glucose & insulin, lipids, eGFR (LOINC) · medications with timing (WHO ATC) · 2-year lab history · family history · TCF7L2 rs7903146 genotype · polygenic risk score", "EAF2FC");
  card(s, 0.5, 4.15, 6.0, 2.25, "Dynamic / real-time: wearables & IoT", "CGM (5- or 15-min) · heart rate · HRV (RMSSD) · steps & METs · sleep stages · logged meals from an Indian food table · insulin and tablet doses", "FDF0EA");
  const src = [
    ["1,000 × 14 days", "Synthetic India-calibrated patients", "ICMR-INDIAB mix, regional diets, Indian prescribing; streams generated by the same physiology"],
    ["45 people", "CGMacros (PhysioNet)", "Real CGM, Fitbit, meals, labs; healthy to T2D (USA)"],
    ["100 patients", "ShanghaiT2DM (Figshare)", "External: rice-based diet, insulin users, no wearables"],
  ];
  src.forEach(([v, t, d], i) => {
    const y = 1.75 + i * 1.6;
    s.addText(v, { x: 6.9, y, w: 2.6, h: 0.7, fontSize: 28, bold: true, color: C.dk2, margin: 0, valign: "middle", isTextBox: true });
    s.addText([{ text: t, options: { bold: true, fontSize: 15, color: "222833", breakLine: true } }, { text: d, options: { fontSize: 13, color: INK2 } }],
      { x: 9.5, y, w: 3.3, h: 1.45, margin: 0, valign: "top", isTextBox: true });
  });
  s.addNotes("The EHR stream follows India's ABDM stack and exports as FHIR R4. Wearable streams come from CGM, smartwatch and logs. We use three sources: a synthetic India-calibrated cohort for scale, CGMacros as real-world validation, and ShanghaiT2DM as an external population with no wearables.");

  // 6 · Synthetic cohort
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data" });
  addTitle(s, "A synthetic Indian cohort whose two streams are causally linked");
  img(s, "synthetic_realism.png", 0.5, 1.35, 7.6, 2.6);
  bullets(s, [
    "Status mix, South-Asian BMI phenotype, comorbidities and prescribing calibrated to Indian clinics: premixed 30/70 insulin, gliclazide, teneligliptin",
    "Regional meals from IFCT 2017: idli-sambar, roti-sabzi, ragi mudde, biryani, chai with sugar, festival sweets",
    "Wearables are simulated by the same physiology, and labs are measured from the simulated body (HbA1c from mean glucose)",
    "Realistic imperfection: unlogged meals, carb misestimation, CGM noise and compression lows, missed doses, illness and stress days",
  ], { x: 8.4, y: 1.4, w: 4.4, h: 5.3, fontSize: 14 });
  s.addText("Physiology parameters are sampled from relationships measured by fitting twins to 45 real people, so synthetic glucose behaves like real glucose.",
    { x: 0.5, y: 4.2, w: 7.6, h: 0.9, fontSize: 14, color: C.dk2, margin: 0, isTextBox: true });
  s.addNotes("The key design choice: the synthetic EHR and wearables are not two independent random tables. Both are generated from one simulated body, so a fusion model can learn genuine cross-stream relationships. Statistics match real CGM data by glycaemic status.");

  // 7 · MODEL
  sec("Model · Sync · Simulate");
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Model · Sync · Simulate" });
  addTitle(s, "MODEL: personal twins recover known physiology from glucose data");
  triad(s, "MODEL");
  img(s, "twin_validity.png", 0.5, 1.35, 8.4, 3.6);
  stat(s, 9.3, 1.5, 3.6, `${f1(fitPrior)} → ${f1(fitPers)}`, "mg/dL twin error, EHR prior → personalised (45 real people, 1–4 h windows)");
  stat(s, 9.3, 3.3, 3.6, `${f0((1 - fitPers / fitPrior) * 100)}% lower`, "error once the twin is fitted to the individual");
  s.addText("Fitted insulin sensitivity tracks lab HOMA-IR and beta-cell response tracks HbA1c, without the model ever seeing those relationships during fitting.",
    { x: 0.5, y: 5.2, w: 12.3, h: 0.8, fontSize: 15, color: C.dk2, margin: 0, isTextBox: true });
  s.addNotes("The twin's five personal parameters are fitted from a week of CGM, meals and activity, with priors from the EHR. On 45 real people, the fitted parameters line up with laboratory measures: evidence the twin is physiologically meaningful, not a curve fit.");

  // 8 · SYNC
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Model · Sync · Simulate" });
  addTitle(s, "SYNC: the twin notices illness from CGM alone");
  triad(s, "SYNC");
  img(s, "illness_detection.png", 0.5, 1.35, 6.4, 4.7);
  stat(s, 7.4, 1.5, 5.3, f2(ILL.auroc), `AUROC for identifying ill days from the synced insulin-sensitivity trace (${ILL.n_ill_days} ill days, ${ILL.n_days} patient-days)`);
  stat(s, 7.4, 3.3, 5.3, `${f2(ILL.mean_si_mult_ill)}× vs ${f2(ILL.mean_si_mult_well)}×`, "insulin sensitivity relative to baseline on ill vs well days");
  s.addText("The Unscented Kalman Filter tracks a drifting insulin-sensitivity factor every 5 minutes. A sustained drop prompts the dashboard insight: check for infection, stress or steroid use.",
    { x: 7.4, y: 5.1, w: 5.4, h: 1.3, fontSize: 14, color: INK2, margin: 0, isTextBox: true });
  s.addNotes("Synchronisation is what keeps a twin alive. Besides glucose and insulin, the filter estimates how insulin sensitivity is drifting. On held-out synthetic patients it separates ill days from well days without any symptom input.");

  // 9 · TwinNet
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Model · Sync · Simulate" });
  addTitle(s, "SIMULATE + learn: TwinNet fuses both streams with the twin");
  triad(s, "SIMULATE");
  const box = (x, y, w, h, t, fill, color = "222833") => {
    s.addShape("roundRect", { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill } });
    s.addText(t, { x: x + 0.1, y, w: w - 0.2, h, fontSize: 13, color, align: "center", valign: "middle", margin: 0, isTextBox: true });
  };
  const arr = (x1, y1, x2, y2) => s.addShape("line", { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipV: y2 < y1, line: { color: "8A93A3", width: 1.75, endArrowType: "triangle" } });
  box(0.5, 1.7, 3.0, 0.95, "6 h × 21 channels: CGM, HR, HRV, steps, sleep, carbs/insulin on board", "FDF0EA");
  box(0.5, 2.95, 3.0, 0.95, "EHR vector: labs, therapy, comorbidities, genetics", "EAF2FC");
  box(0.5, 4.2, 3.0, 0.95, "Physics twin forecast + synced hidden state", C.dk2, "FFFFFF");
  box(4.1, 1.7, 2.3, 0.95, "Causal dilated TCN", "F2F5FA");
  box(4.1, 2.95, 2.3, 0.95, "MLP → FiLM conditions the TCN", "F2F5FA");
  box(4.1, 4.2, 2.3, 0.95, "MLP", "F2F5FA");
  box(7.0, 2.95, 2.0, 0.95, "Fusion", "F2F5FA");
  box(9.6, 1.9, 3.2, 1.0, "Glucose 5–120 min: median + 80% band (conformal)", "EAF7F1");
  box(9.6, 3.2, 3.2, 1.0, "P(spike > 180) · P(hypo < 70) in 2 h", "EAF7F1");
  box(9.6, 4.5, 3.2, 1.0, "Gate: trust in physics per horizon", "FFF5DC");
  arr(3.5, 2.17, 4.1, 2.17); arr(3.5, 3.42, 4.1, 3.42); arr(3.5, 4.67, 4.1, 4.67);
  arr(6.4, 2.17, 7.0, 3.2); arr(6.4, 3.42, 7.0, 3.42); arr(6.4, 4.67, 7.0, 3.65);
  arr(9.0, 3.42, 9.6, 2.4); arr(9.0, 3.42, 9.6, 3.7); arr(9.0, 3.42, 9.6, 5.0);
  s.addText(`Learned trust in the physics twin rises with horizon: ${f2(gate[5])} at 30 min, ${f2(gate[11])} at 60 min, ${f2(gate[23])} at 2 h. Modality dropout during training lets the same model run without wearables or EHR. Pretrained on synthetic patients, fine-tuned on real ones.`,
    { x: 0.5, y: 5.6, w: 12.3, h: 0.9, fontSize: 14, color: C.dk2, margin: 0, isTextBox: true });
  s.addNotes("TwinNet: the dynamic stream goes through a causal temporal convolution, the EHR conditions it through FiLM, and the twin's forecast enters through a gate. The gate is interpretable: the network leans on physics more at longer horizons, where data-driven extrapolation is weakest.");

  // 10 · Forecast results
  sec("Results");
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  addTitle(s, "Accurate 2-hour glucose forecasts on unseen patients", `${syn.n_test_patients} held-out patients · days 8–14 after personalisation · ${syn.n_test_anchors.toLocaleString("en-IN")} forecasts`);
  img(s, "rmse_by_horizon_synthetic.png", 0.5, 1.6, 7.4, 4.3);
  stat(s, 8.3, 1.7, 4.5, `${f1(TN.rmse_60)} mg/dL`, `60-min error (RMSE); persistence ${f1(PERS.rmse_60)} · LightGBM ${f1(GBM.rmse_60)}`);
  stat(s, 8.3, 3.35, 4.5, `${f1(TN.clarkeAB_60)}%`, "of 60-min forecasts in Clarke zones A+B (clinically acceptable)");
  stat(s, 8.3, 5.0, 4.5, `${f0(TN.coverage80_60)}%`, "of outcomes inside the 80% interval (target 80%)");
  s.addNotes("Error grows with horizon for every method, but the hybrid stays well below simply assuming glucose stays flat. Clarke Error Grid zones A and B are clinically acceptable. Conformal calibration keeps the 80% band honest.");

  // 11 · Alerts
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  addTitle(s, "Early warnings with lead time a clinician can act on");
  stat(s, 0.5, 1.5, 3.0, `${f0(SPIKE.detected_pct)}%`, "of sustained spikes (>180) flagged in advance");
  stat(s, 3.7, 1.5, 3.0, `${f0(SPIKE.median_lead_min)} min`, "median warning before the spike begins");
  stat(s, 6.9, 1.5, 2.8, f2(SPIKE.false_alerts_per_day), "false alert episodes per patient-day");
  stat(s, 9.9, 1.5, 2.9, f2(HYPO.auroc), `hypoglycaemia AUROC (spike AUROC ${f2(SPIKE.auroc)})`);
  img(s, "example_forecast.png", 0.5, 3.15, 8.2, 3.75);
  s.addText("Every alert carries its reasons from the event model, e.g. carbohydrate still absorbing, a rising trend, insulin sensitivity below baseline, or a sedentary hour after a meal.",
    { x: 9.0, y: 3.4, w: 3.8, h: 2.4, fontSize: 14, color: INK2, margin: 0, isTextBox: true });
  s.addNotes("Alert metrics are at a threshold chosen on validation data, with hysteresis so a flickering probability doesn't page anyone repeatedly. The lead time is measured from the first alert to the start of the sustained excursion.");

  // 12 · Ablation
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  addTitle(s, "Fusing both streams matters, and the twin adds the most");
  img(s, "ablation_streams.png", 0.5, 1.35, 7.8, 4.6);
  const abl = syn.ablation_gbm;
  const first = abl[0], last = abl[abl.length - 1], noPhys = abl[abl.length - 2];
  bullets(s, [
    `CGM alone: ${f1(first.rmse_60)} mg/dL at 60 min; spike AUROC ${f2(first.spike_auroc)}`,
    `Both streams (wearables + meals/meds + EHR): ${f1(noPhys.rmse_60)}; AUROC ${f2(noPhys.spike_auroc)}`,
    `+ physics twin: ${f1(last.rmse_60)}; AUROC ${f2(last.spike_auroc)}`,
    "Each stream helps; the EHR helps most in combination, telling the model whose body the wearables describe",
  ], { x: 8.6, y: 1.5, w: 4.2, h: 4.8 });
  s.addNotes("This is the core ask of the challenge: does fusing the static and dynamic streams help? The same model was retrained on each combination of streams; the full hybrid wins on both forecast error and event discrimination.");

  // 13 · Real world + external
  if (real) {
    const cg = byMethod(real.cgmacros.forecast, "TwinNet sim-to-real");
    const cgP = byMethod(real.cgmacros.forecast, "Persistence");
    const cgZ = byMethod(real.cgmacros.forecast, "TwinNet zero-shot (synthetic)");
    const sh = byMethod(real.shanghai.forecast, "TwinNet sim-to-real");
    const shZ = byMethod(real.shanghai.forecast, "TwinNet zero-shot (synthetic)");
    const shP = byMethod(real.shanghai.forecast, "Persistence");
    s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
    addTitle(s, "It transfers to real people and to an external population");
    img(s, "rmse_by_horizon_cgmacros.png", 0.5, 1.35, 7.4, 4.3);
    s.addTable([
      [{ text: "60-min RMSE (mg/dL)", options: { bold: true } }, { text: "CGMacros", options: { bold: true } }, { text: "Shanghai", options: { bold: true } }],
      ["Persistence", f1(cgP.rmse_60), f1(shP.rmse_60)],
      ["TwinNet, synthetic only", f1(cgZ.rmse_60), f1(shZ.rmse_60)],
      ["TwinNet, sim-to-real", f1(cg.rmse_60), f1(sh.rmse_60)],
      ["Clarke A+B, sim-to-real", `${f1(cg.clarkeAB_60)}%`, `${f1(sh.clarkeAB_60)}%`],
    ], { x: 8.2, y: 1.5, w: 4.6, colW: [2.2, 1.2, 1.2], fontSize: 13, color: "222833", border: { type: "solid", color: "DDE2EA", pt: 0.75 }, fill: { color: "FFFFFF" } });
    s.addText(`${real.cgmacros.n_recordings} CGMacros participants (5-fold patient cross-validation) and ${real.shanghai.n_recordings} ShanghaiT2DM recordings from a different country, sensor and diet, with no wearables at all.`,
      { x: 8.2, y: 4.3, w: 4.6, h: 1.6, fontSize: 13, color: INK2, margin: 0, isTextBox: true });
    s.addNotes("Real-world evidence: pretraining on the synthetic Indian cohort and fine-tuning on a few real patients (sim-to-real) is the regime we recommend for deployment. Shanghai shows the model still works when a whole modality is missing.");
  }

  // 14 · CGM-light
  if (light) {
    s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
    addTitle(s, "CGM-light: one week of CGM, then fingersticks and a smartwatch", "An affordability mode for India: the twin keeps estimating continuous glucose after the sensor comes off");
    stat(s, 0.5, 1.9, 3.9, `${f1(light.twin.mard)}%`, `twin MARD vs true glucose (carry-forward fingersticks: ${f1(light.carry_forward.mard)}%)`);
    stat(s, 4.7, 1.9, 3.9, `${f0(light.twin.within_20pct)}%`, `of twin estimates within 20% of true glucose (carry-forward: ${f0(light.carry_forward.within_20pct)}%)`);
    stat(s, 8.9, 1.9, 3.9, `±${f1(light.twin.tir_abs_error_pp)} pp`, `error in weekly time-in-range (fingersticks alone: ±${f1(light.fingersticks_only_tir_abs_error_pp)} pp)`);
    card(s, 0.5, 4.2, 12.3, 1.9, "How it works", `Week 1 personalises the twin on CGM. In week 2 only ${light.fingersticks_per_day} fingerstick checks a day remain; the Unscented Kalman Filter fuses them with logged meals, doses and smartwatch activity to estimate glucose in between. Evaluated on ${light.n_patients} held-out synthetic patients against their true glucose.`);
    s.addNotes("CGM sensors are expensive for many Indian families. A short CGM period teaches the twin the person's physiology; afterwards the twin uses cheap fingersticks and smartwatch data to keep an estimate of continuous glucose and time in range.");
  }

  // 15–17 · Dashboard
  sec("Dashboard");
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Dashboard" });
  addTitle(s, "The doctor's view: a risk-ranked remote-monitoring panel");
  img(s, "panel_light.png", 0.5, 1.3, 12.3, 5.6);
  s.addNotes("Every patient on the panel is a live twin. Current lows first, then predicted lows, patients already high, and predicted new spikes. Status always uses an icon and a label, never colour alone.");
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Dashboard" });
  addTitle(s, "The virtual patient: forecast, reasons and learned physiology");
  img(s, "patient_light.png", 0.5, 1.3, 12.3, 5.6);
  s.addNotes("The glucose chart shows 24 hours of CGM, the 2-hour forecast cone and the physics twin's projection, with meals, medication and sleep lanes below. The risk card explains the alert. The virtual-patient panel turns fitted parameters into organ-level physiology a clinician recognises.");
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Dashboard" });
  addTitle(s, "What-if on the patient's own twin, and a companion in their language");
  img(s, "whatif_light.png", 0.5, 1.3, 8.9, 5.6);
  img(s, "companion_hi.png", 9.7, 1.3, 3.1, 5.6);
  s.addNotes("The doctor tests a meal swap or a walk on this patient's twin and sees the predicted peak change. Personal meal ranking shows which Indian meals suit this body. The patient companion delivers simple nudges in Hindi, Kannada or English.");

  // 18 · Responsible AI
  sec("Responsible by design");
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Responsible by design" });
  addTitle(s, "Responsible by design: privacy, interoperability, explainability");
  const rows = [
    ["DPDP Act 2023", "Consent artefacts, purpose limitation, data minimisation, purpose-tagged audit trail; synthetic and de-identified data only"],
    ["ABDM-ready interoperability", "FHIR R4 with SNOMED CT, LOINC, WHO ATC; CGM as SampledData; the forecast exported as a RiskAssessment"],
    ["Explainable", "Reasons on every alert (SHAP), a visible trust-in-physics gate, organ-level physiology, AGP report"],
    ["Honest evaluation", "Patient-level splits, external validation, bootstrap CIs, model card with limitations"],
    ["Regulatory path", "Decision support, not autonomous treatment: CDSCO Software as a Medical Device route with prospective validation"],
  ];
  rows.forEach(([t, d], i) => {
    const y = 1.5 + i * 1.05;
    s.addShape("ellipse", { x: 0.55, y: y + 0.1, w: 0.5, h: 0.5, fill: { color: i === 0 ? C.accent4 : C.dk2 }, line: { color: i === 0 ? C.accent4 : C.dk2 } });
    s.addText(String(i + 1), { x: 0.55, y: y + 0.1, w: 0.5, h: 0.5, fontSize: 14, bold: true, align: "center", valign: "middle", color: i === 0 ? C.dk2 : "FFFFFF", margin: 0, isTextBox: true });
    s.addText([{ text: t, options: { bold: true, fontSize: 16, color: C.dk2, breakLine: true } }, { text: d, options: { fontSize: 14, color: INK2 } }],
      { x: 1.3, y, w: 11.5, h: 0.95, margin: 0, valign: "top", isTextBox: true });
  });
  s.addNotes("Health data is the most sensitive personal data. The design follows the DPDP Act and ABDM, every answer is explainable, and the evaluation states its limits plainly.");

  // 19 · Roadmap
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Responsible by design" });
  addTitle(s, "From proof of concept to Indian clinics");
  const steps = [
    ["Now", "Proof of concept", "Twin, TwinNet, dashboard, FHIR; validated on synthetic, US and Chinese data"],
    ["3 months", "Co-design", "Diabetologist review of alerts and thresholds; native-speaker review of patient text"],
    ["6 months", "Pilot", "50–100 people with T2D in Bengaluru clinics: CGM + smartwatch + ABDM-linked EHR"],
    ["12 months", "Validation", "Prospective accuracy and outcome study; CDSCO SaMD submission"],
  ];
  steps.forEach(([w, t, d], i) => {
    const x = 0.5 + i * 3.1;
    s.addShape("ellipse", { x: x + 0.05, y: 1.75, w: 0.4, h: 0.4, fill: { color: i === 0 ? C.accent4 : C.dk2 }, line: { color: i === 0 ? C.accent4 : C.dk2 } });
    if (i < steps.length - 1) s.addShape("line", { x: x + 0.5, y: 1.95, w: 2.55, h: 0, line: { color: "C9D1DE", width: 2 } });
    s.addText([{ text: w, options: { bold: true, fontSize: 14, color: MUTED, breakLine: true } }, { text: t, options: { bold: true, fontSize: 18, color: C.dk2, breakLine: true } }, { text: d, options: { fontSize: 13, color: INK2 } }],
      { x, y: 2.35, w: 2.85, h: 2.3, valign: "top", margin: 0, paraSpaceAfter: 4, isTextBox: true });
  });
  card(s, 0.5, 5.0, 12.3, 1.5, "Impact we are aiming for", "Fewer nocturnal hypoglycaemia episodes in people on sulfonylureas and premixed insulin, more time in range through personalised meal and activity advice, and a care team that sees risk before it becomes an emergency.");
  s.addNotes("Close with the path forward: clinical co-design, a pilot in Bengaluru, then prospective validation. The goal is measurable: fewer lows, more time in range, earlier action.");

  // 20 · Closing
  s = pres.addSlide({ masterName: "DARK", sectionTitle: "Responsible by design" });
  s.addText("Thank you", { x: 0.8, y: 2.0, w: 11, h: 1.0, fontFace: "Cambria", fontSize: 54, bold: true, color: "FFFFFF", margin: 0, isTextBox: true });
  s.addText("MadhuTwin: model · sync · simulate, for every person living with diabetes", { x: 0.8, y: 3.1, w: 11, h: 0.6, fontSize: 20, color: "DCE4F2", margin: 0, isTextBox: true });
  s.addText("Code, data pipeline, evaluation and dashboard: GitHub repository TODO(team) · Apache-2.0", { x: 0.8, y: 4.4, w: 11, h: 0.5, fontSize: 16, color: "FFFFFF", margin: 0, isTextBox: true });
  s.addText("Team: TODO(team)", { x: 0.8, y: 5.0, w: 11, h: 0.5, fontSize: 16, color: "AEBBD3", margin: 0, isTextBox: true });
  s.addNotes("Thank the jury and invite questions.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
