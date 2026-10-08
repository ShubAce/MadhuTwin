// Builds docs/architecture.pptx: a single, editable architecture diagram (native shapes).
// Export to PDF with docs/deck/export_pdf.ps1 (PowerPoint) for docs/architecture.pdf.
const path = require("path");
const pptxgen = require("pptxgenjs");
const { THEME } = require("./theme");
const { applyTheme } = require(process.env.PPTX_SKILL_THEME || "./apply_theme_shim.js");

const OUT = path.join(__dirname, "..", "architecture.pptx");
const H = THEME.colors;

function card(slide, { x, y, w, h, fill, line, title, lines, titleColor = H.dk2, size = 12, name }) {
  slide.addShape("roundRect", { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: line || fill, width: 0.75 }, objectName: name });
  slide.addText(
    [
      { text: title, options: { bold: true, color: titleColor, fontSize: size + 1, breakLine: true } },
      ...lines.map((t, i) => ({ text: t, options: { color: "333A45", fontSize: size, breakLine: i < lines.length - 1 } })),
    ],
    { x: x + 0.12, y: y + 0.08, w: w - 0.24, h: h - 0.16, valign: "top", margin: 0, paraSpaceAfter: 2, isTextBox: true, fontFace: "Calibri" },
  );
}

function arrow(slide, x1, y1, x2, y2, color = "7C8798") {
  // OOXML extents must be non-negative: draw from the top-left corner and flip as needed
  slide.addShape("line", {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1, line: { color, width: 2, endArrowType: "triangle" },
  });
}

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "MadhuTwin architecture";
  pres.author = "MadhuTwin team";
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };

  s.addText("MadhuTwin architecture: two data streams fused into a living digital twin", {
    x: 0.5, y: 0.3, w: 12.3, h: 0.6, fontFace: "Cambria", fontSize: 24, bold: true, color: H.dk2, margin: 0, isTextBox: true,
  });
  s.addText("Type 2 Diabetes · predicts glucose spikes and hypoglycaemia up to 2 hours ahead · doctor-facing dashboard", {
    x: 0.5, y: 0.9, w: 12.3, h: 0.35, fontSize: 13, color: "5A6474", margin: 0, isTextBox: true,
  });

  // column headers
  const cols = [
    { x: 0.5, w: 3.5, t: "1 · Data streams" },
    { x: 4.45, w: 4.75, t: "2 · Digital twin core" },
    { x: 9.65, w: 3.2, t: "3 · Clinician experience" },
  ];
  cols.forEach((c) => s.addText(c.t, { x: c.x, y: 1.4, w: c.w, h: 0.3, fontSize: 13, bold: true, color: H.dk2, margin: 0, isTextBox: true }));

  // 1. data streams
  card(s, { x: 0.5, y: 1.8, w: 3.5, h: 2.05, fill: "EAF2FC", line: "BFD5F2", title: "Static / historical: EHR",
    lines: ["Demographics, diagnoses (SNOMED CT)", "Labs: HbA1c, FPG, insulin, lipids, eGFR (LOINC)", "Medications with timing (WHO ATC)", "Family history · TCF7L2 rs7903146 · polygenic risk", "FHIR R4 bundles, ABDM-aligned"], name: "ehr" });
  card(s, { x: 0.5, y: 3.95, w: 3.5, h: 1.75, fill: "FDF0EA", line: "F4CDB9", title: "Dynamic / real-time: wearables & IoT",
    lines: ["CGM every 5–15 min", "Heart rate, HRV (RMSSD), steps, METs", "Sleep stages (hypnogram)", "Logged meals (Indian foods) and doses"], name: "wearables" });
  card(s, { x: 0.5, y: 5.8, w: 3.5, h: 0.95, fill: "F3F4F6", line: "DADDE2", title: "Sources",
    lines: ["Synthetic India cohort 1,000 × 14 d · CGMacros (45, real) · BIG IDEAs (16, real HRV) · ShanghaiT2DM (100, external)"], size: 10, name: "sources" });

  // 2. twin core
  card(s, { x: 4.45, y: 1.8, w: 4.75, h: 0.8, fill: "F3F4F6", line: "DADDE2", title: "Ingest & fuse",
    lines: ["Common 5-min schema · missing-aware · EHR vector + 21 dynamic channels"], size: 10, name: "ingest" });
  const triad = [
    { x: 4.45, t: "MODEL", d: "Personalised glucose–insulin physiology (MAP fit, EHR priors)" },
    { x: 6.1, t: "SYNC", d: "Unscented Kalman filter every 5 min: state + insulin-sensitivity drift" },
    { x: 7.75, t: "SIMULATE", d: "2-h forecasts and what-if: meals, walks, sleep, illness" },
  ];
  triad.forEach((p) => {
    s.addShape("roundRect", { x: p.x, y: 2.8, w: 1.45, h: 1.45, rectRadius: 0.1, fill: { color: H.dk2 }, line: { color: H.dk2 }, objectName: p.t });
    s.addText([{ text: p.t, options: { bold: true, fontSize: 14, color: "FFFFFF", breakLine: true } }, { text: p.d, options: { fontSize: 11, color: "DCE4F2" } }],
      { x: p.x + 0.1, y: 2.88, w: 1.27, h: 1.3, valign: "top", margin: 0, isTextBox: true, paraSpaceAfter: 3 });
  });
  arrow(s, 5.92, 3.52, 6.08, 3.52, H.accent4);
  arrow(s, 7.57, 3.52, 7.73, 3.52, H.accent4);
  card(s, { x: 4.45, y: 4.45, w: 4.75, h: 2.3, fill: "EAF2FC", line: "BFD5F2", title: "TwinNet: physics-gated fusion network",
    lines: ["Causal dilated TCN over 6 h × 21 channels", "EHR encoder conditions the dynamics (FiLM)", "Gate learns how much to trust the physics twin per horizon", "Outputs: 5–120 min quantiles (conformal 80% band) + spike / hypo probability", "Fine-tuned per person, averaged with LightGBM; SHAP reasons"], name: "twinnet" });

  // 3. experience
  card(s, { x: 9.65, y: 1.8, w: 3.2, h: 0.8, fill: "F3F4F6", line: "DADDE2", title: "FastAPI + WebSocket",
    lines: ["Live replay · audit trail · FHIR export"], size: 10, name: "api" });
  card(s, { x: 9.65, y: 2.8, w: 3.2, h: 2.75, fill: "EAF7F1", line: "BCE6D3", title: "Doctor dashboard",
    lines: ["Risk-ranked remote-monitoring panel", "Virtual patient: organ-level physiology", "24 h CGM + 2 h forecast cone + alerts with reasons", "AGP report (international consensus)", "What-if simulator with Indian meals", "Ask the twin (grounded assistant)"], name: "dashboard" });
  card(s, { x: 9.65, y: 5.65, w: 3.2, h: 1.1, fill: "F3F4F6", line: "DADDE2", title: "Interoperability",
    lines: ["FHIR R4 RiskAssessment carries the twin's forecast to any EHR"], size: 10, name: "fhir" });

  // flows
  arrow(s, 4.0, 2.9, 4.45, 2.2);
  arrow(s, 4.0, 4.8, 4.45, 2.45);
  arrow(s, 6.8, 2.6, 6.8, 2.8);
  arrow(s, 6.8, 4.25, 6.8, 4.45);
  arrow(s, 9.2, 5.5, 9.65, 4.3);
  arrow(s, 9.2, 3.52, 9.65, 3.52);

  // governance band
  s.addShape("roundRect", { x: 0.5, y: 6.9, w: 12.35, h: 0.42, rectRadius: 0.06, fill: { color: "FFF5DC" }, line: { color: "F2D58A" }, objectName: "governance" });
  s.addText([
    { text: "Governance  ", options: { bold: true, color: "7A5300" } },
    { text: "consent artefacts · purpose-tagged audit log · de-identified & synthetic data · model card · DPDP Act 2023 · CDSCO SaMD pathway", options: { color: "4A3A10" } },
  ], { x: 0.65, y: 6.92, w: 12.1, h: 0.38, fontSize: 11, valign: "middle", margin: 0, isTextBox: true });

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
