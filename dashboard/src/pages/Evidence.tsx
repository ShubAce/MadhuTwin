import type { EChartsOption } from "echarts";
import Chart, { Tokens, baseOption, valueAxis } from "../components/Chart";
import { Legend, Section, Stat } from "../components/ui";
import { useData } from "../hooks";

type Row = Record<string, number | string | number[] | string[] | null>;
type Cohort = { forecast: Row[]; events: Row[]; n_recordings: number; n_anchors: number };
type CalPt = { predicted: number; observed: number; n: number };
type DcaPt = { threshold: number; model: number; alert_all: number; alert_none: number };
type Clinical = { method?: string; spike_calibration: CalPt[]; spike_decision_curve: DcaPt[] };
interface Evidence {
  synthetic?: {
    forecast: Row[];
    events: Row[];
    ablation_gbm: Row[];
    ablation_twinnet: Row[];
    illness_detection: Row;
    gate_by_horizon: number[];
    n_test_patients: number;
    n_test_anchors: number;
  };
  real?: { cgmacros: Cohort; shanghai: Cohort; bigideas?: Cohort; clinical?: Record<string, Clinical> };
  extra?: {
    forecast: Row[];
    events: Row[];
    clinical: Clinical;
    subgroups: { attribute: string; group: string; patients: number; rmse_60: number; rmse_120: number }[];
    learning_curve: { days: number; rmse_60: number; rmse_120: number }[];
    curve_patients: number;
  };
  fidelity?: Record<string, { patients: number; days: number; mean_abs_error_median: number; mean_glucose_error_median: number; tir_error_median_pp: number;
    rating_pct: { good: number; fair: number; poor: number }; days_with_lows: number; lows_reproduced_pct: number | null }>;
  robustness?: { modalities: Row[]; cgm_dropout: Row[]; n_forecasts: number; model: string };
  bench?: Record<string, number | string>;
  cgm_light?: {
    n_patients: number;
    fingersticks_per_day: number;
    twin: { mard: number; within_20pct: number; tir_abs_error_pp: number; clarke: Record<string, number> };
    carry_forward: { mard: number; within_20pct: number; tir_abs_error_pp: number; clarke: Record<string, number> };
    fingersticks_only_tir_abs_error_pp: number;
  };
}

const n = (v: unknown, d = 1) => (typeof v === "number" && isFinite(v) ? v.toFixed(d) : "–");
const HZ = [30, 60, 90, 120];
const ENS = "MadhuTwin ensemble";
const LINE_METHODS: [string, string][] = [[ENS, "--series-1"], ["TwinNet hybrid", "--series-1"], ["LightGBM fusion", "--series-2"], ["Twin (personalised)", "--series-3"], ["Persistence", "--deemph"]];
const ATTR: Record<string, string> = { sex: "Sex", age: "Age", bmi: "BMI", status: "Status", therapy: "Therapy", region: "Region" };
const cssVar = (v: string) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

/** Synthetic rows from the main experiment, plus the personalised and ensemble rows from the extra experiment. */
function mergeRows(base: Row[], extra: Row[] | undefined, key: (r: Row) => string): Row[] {
  if (!extra) return base;
  const ex = new Map(extra.map((r) => [key(r), r]));
  const merged = base.map((r) => ({ ...r, ...(ex.get(key(r)) ?? {}) }));
  return [...merged, ...extra.filter((r) => !base.some((b) => key(b) === key(r)))];
}

/** Headline method for a real cohort: the ensemble when the upgraded experiment produced it. */
const bestOf = (c: Cohort) => (c.forecast.some((r) => r.method === ENS) ? ENS : "TwinNet sim-to-real");

function ForecastTable({ rows, cov = true }: { rows: Row[]; cov?: boolean }) {
  return (
    <div className="overflow-x-auto">
      <table className="data min-w-[760px]">
        <thead>
          <tr>
            <th>Method</th>
            {HZ.map((h) => <th key={h} className="num">RMSE {h} min</th>)}
            <th className="num">MARD 60</th>
            <th className="num">Clarke A 60</th>
            <th className="num">Clarke A+B 60</th>
            {cov && <th className="num">80% PI coverage 60</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={String(r.method)} className={r.method === ENS ? "font-semibold" : ""}>
              <td className="font-medium">{String(r.method)}</td>
              {HZ.map((h) => (
                <td key={h} className="num">
                  {n(r[`rmse_${h}`])}
                  {Array.isArray(r[`rmse_${h}_ci`]) && <span className="muted font-normal"> ({n((r[`rmse_${h}_ci`] as number[])[0])}–{n((r[`rmse_${h}_ci`] as number[])[1])})</span>}
                </td>
              ))}
              <td className="num">{n(r.mard_60)}%</td>
              <td className="num">{n(r.clarkeA_60)}%</td>
              <td className="num">{n(r.clarkeAB_60)}%</td>
              {cov && <td className="num">{r.coverage80_60 != null ? `${n(r.coverage80_60)}%` : "–"}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EventTable({ rows }: { rows: Row[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="data min-w-[860px]">
        <thead>
          <tr><th>Event</th><th>Method</th><th className="num">AUROC (95% CI)</th><th className="num">AUPRC</th><th className="num">Prevalence</th>
            <th className="num">Excursions caught</th><th className="num">Median lead</th><th className="num">False alerts / day</th>
            <th className="num" title="Point on the alert-burden curve: best detection with at most one false alert per patient-day">Caught at ≤1 false alert/day</th></tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className={r.method === ENS ? "font-semibold" : ""}>
              <td>{r.event === "spike" ? "Spike >180" : "Hypo <70"}</td>
              <td className="font-medium">{String(r.method)}</td>
              <td className="num">{n(r.auroc, 3)}{Array.isArray(r.auroc_ci) && <span className="muted font-normal"> ({n((r.auroc_ci as number[])[0], 2)}–{n((r.auroc_ci as number[])[1], 2)})</span>}</td>
              <td className="num">{n(r.auprc, 3)}</td>
              <td className="num">{n(r.prevalence)}%</td>
              <td className="num">{r.detected_pct != null ? `${n(r.detected_pct, 0)}% of ${r.excursions}` : "–"}</td>
              <td className="num">{r.median_lead_min != null ? `${n(r.median_lead_min, 0)} min` : "–"}</td>
              <td className="num">{r.false_alerts_per_day != null ? n(r.false_alerts_per_day, 2) : "–"}</td>
              <td className="num">{r.caught_at_1fa_pct != null ? `${n(r.caught_at_1fa_pct, 0)}%${r.lead_at_1fa_min != null ? ` · ${n(r.lead_at_1fa_min, 0)} min` : ""}` : "–"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RealSection({ title, c }: { title: string; c: Cohort }) {
  return (
    <Section title={title}>
      <ForecastTable rows={c.forecast} />
      <div className="mt-4"><EventTable rows={c.events} /></div>
    </Section>
  );
}

export default function Evidence() {
  const { data } = useData<Evidence>("/api/evidence");
  const syn = data?.synthetic;
  const real = data?.real;
  const extra = data?.extra;
  if (!syn) return <p className="muted">Evaluation results not found. Run scripts/exp_synthetic.py and scripts/exp_real.py.</p>;
  const forecast = mergeRows(syn.forecast, extra?.forecast, (r) => String(r.method));
  const events = mergeRows(syn.events, extra?.events, (r) => `${r.event}|${r.method}`);
  const best = extra ? ENS : "TwinNet hybrid";
  const by = (m: string) => forecast.find((r) => r.method === m);
  const tn = by(best), pers = by("Persistence");
  const spike = events.find((r) => r.event === "spike" && r.method === best);
  const gain60 = tn && pers ? (1 - Number(tn.rmse_60) / Number(pers.rmse_60)) * 100 : null;
  const lc = extra?.learning_curve;
  const realCohorts: [string, string, Cohort | undefined][] = real
    ? [["CGMacros", "45 people, wristband steps/HR", real.cgmacros], ["BIG IDEAs", "real wristband HR + HRV", real.bigideas], ["ShanghaiT2DM", "external, no wearables", real.shanghai]]
    : [];
  const lineMethods = LINE_METHODS.filter(([m]) => by(m) && !(extra && m === "TwinNet hybrid"));

  // direct end labels only where they cannot collide; the legend carries the rest
  const ends = lineMethods.map(([m]) => Number(by(m)!.rmse_120));
  const clear = (i: number) => ends.every((v, j) => j === i || Math.abs(v - ends[i]) > 2.5);
  const horizonChart = (t: Tokens): EChartsOption => ({
    ...baseOption(t),
    grid: { left: 44, right: 150, top: 14, bottom: 28 },
    xAxis: { type: "category", data: HZ.map((h) => `${h} min`), axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false }, axisLabel: { color: t.muted, fontSize: 11 } },
    yAxis: valueAxis(t, { name: "RMSE mg/dL", nameTextStyle: { color: t.muted, fontSize: 11 } }),
    series: lineMethods.map(([m, v], i) => {
      const c = cssVar(v);
      return {
        name: m, type: "line" as const, data: HZ.map((h) => Number(by(m)![`rmse_${h}`])), symbol: "circle", symbolSize: 8,
        lineStyle: { width: 2, color: c }, itemStyle: { color: c, borderColor: t.surface, borderWidth: 2 },
        endLabel: { show: clear(i), color: t.ink2, fontSize: 11, formatter: m },
      };
    }),
  });
  const abl = syn.ablation_gbm;
  const ablChart = (t: Tokens): EChartsOption => ({
    ...baseOption(t),
    tooltip: { ...(baseOption(t).tooltip as object), trigger: "item" },
    grid: { left: 210, right: 56, top: 4, bottom: 4 },
    xAxis: { type: "value", show: false, min: 0 },
    yAxis: { type: "category", inverse: true, data: abl.map((r) => String(r.config)), axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: t.ink2, fontSize: 11 } },
    series: [{
      type: "bar", barMaxWidth: 14,
      data: abl.map((r, i) => ({ value: Number(Number(r.rmse_60).toFixed(1)), itemStyle: { color: i === abl.length - 1 ? t.s1 : t.deemph, borderRadius: [0, 4, 4, 0] } })),
      label: { show: true, position: "right", color: t.ink2, fontSize: 11, formatter: "{c}" },
    }],
  });

  const learnChart = (t: Tokens): EChartsOption => ({
    ...baseOption(t),
    grid: { left: 44, right: 110, top: 14, bottom: 40 },
    xAxis: { type: "category", data: lc!.map((r) => String(r.days)), name: "days of the person's own data", nameLocation: "middle", nameGap: 26,
      nameTextStyle: { color: t.muted, fontSize: 11 }, axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false }, axisLabel: { color: t.muted, fontSize: 11 } },
    yAxis: valueAxis(t, { name: "RMSE mg/dL", nameTextStyle: { color: t.muted, fontSize: 11 }, scale: true }),
    series: ([["2-hour forecast", "rmse_120", t.s2], ["1-hour forecast", "rmse_60", t.s1]] as const).map(([name, k, c]) => ({
      name, type: "line" as const, data: lc!.map((r) => Number(r[k].toFixed(1))), symbol: "circle", symbolSize: 8,
      lineStyle: { width: 2, color: c }, itemStyle: { color: c, borderColor: t.surface, borderWidth: 2 },
      endLabel: { show: true, color: t.ink2, fontSize: 11, formatter: name },
    })),
  });

  const calSeries: [string, CalPt[], string][] = [
    ...(extra ? [["Synthetic test", extra.clinical.spike_calibration, "--series-1"] as [string, CalPt[], string]] : []),
    ...(([["CGMacros", "cgmacros", "--series-2"], ["BIG IDEAs", "bigideas", "--series-3"], ["ShanghaiT2DM", "shanghai", "--deemph"]] as const)
      .filter(([, k]) => real?.clinical?.[k])
      .map(([lab, k, v]) => [lab, real!.clinical![k].spike_calibration, v] as [string, CalPt[], string])),
  ];
  const calChart = (t: Tokens): EChartsOption => ({
    ...baseOption(t),
    tooltip: { ...(baseOption(t).tooltip as object), trigger: "item", formatter: (p: unknown) => {
      const q = p as { seriesName: string; value: number[] };
      return `${q.seriesName}<br/>predicted ${(q.value[0] * 100).toFixed(0)}% · observed ${(q.value[1] * 100).toFixed(0)}%`;
    } },
    grid: { left: 44, right: 16, top: 14, bottom: 40 },
    xAxis: valueAxis(t, { min: 0, max: 1, name: "predicted risk of a spike in 2 h", nameLocation: "middle", nameGap: 26, nameTextStyle: { color: t.muted, fontSize: 11 } }),
    yAxis: valueAxis(t, { min: 0, max: 1, name: "observed", nameTextStyle: { color: t.muted, fontSize: 11 } }),
    series: [
      { type: "line", data: [[0, 0], [1, 1]], symbol: "none", silent: true, lineStyle: { color: t.axis, width: 1 }, tooltip: { show: false } },
      ...calSeries.map(([name, pts, v]) => {
        const c = cssVar(v);
        return { name, type: "line" as const, data: pts.map((p) => [p.predicted, p.observed]), symbol: "circle", symbolSize: 8,
          lineStyle: { width: 2, color: c }, itemStyle: { color: c, borderColor: t.surface, borderWidth: 2 } };
      }),
    ],
  });

  const dca = real?.clinical?.cgmacros ?? extra?.clinical;
  const dcaName = real?.clinical?.cgmacros ? "CGMacros, real people" : "synthetic test patients";
  const dcaChart = (t: Tokens): EChartsOption => ({
    ...baseOption(t),
    grid: { left: 52, right: 24, top: 14, bottom: 40 },
    xAxis: { type: "category", data: dca!.spike_decision_curve.map((r) => r.threshold.toFixed(2)), name: "risk threshold for an alert", nameLocation: "middle", nameGap: 26,
      nameTextStyle: { color: t.muted, fontSize: 11 }, axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false }, axisLabel: { color: t.muted, fontSize: 11, interval: 2 } },
    yAxis: valueAxis(t, { name: "net benefit", nameTextStyle: { color: t.muted, fontSize: 11 }, min: -0.02 }),
    series: ([["MadhuTwin alerts", "model", t.s1], ["Alert everyone", "alert_all", t.deemph], ["Alert no one", "alert_none", t.ink2]] as const).map(([name, k, c]) => ({
      name, type: "line" as const, symbol: "none", data: dca!.spike_decision_curve.map((r) => Number(Math.max(r[k], -0.02).toFixed(4))),
      lineStyle: { width: k === "alert_none" ? 1 : 2, color: c }, itemStyle: { color: c },
    })),
  });

  const sub = (extra?.subgroups ?? []).filter((r) => r.patients >= 5);
  const subChart = (t: Tokens): EChartsOption => ({
    ...baseOption(t),
    tooltip: { ...(baseOption(t).tooltip as object), trigger: "item" },
    grid: { left: 190, right: 44, top: 4, bottom: 4 },
    xAxis: { type: "value", show: false, min: 0 },
    yAxis: { type: "category", inverse: true, data: sub.map((r) => `${ATTR[r.attribute] ?? r.attribute}: ${r.group} (n=${r.patients})`), axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: t.ink2, fontSize: 11 } },
    series: [{
      type: "bar", barMaxWidth: 12,
      data: sub.map((r) => ({ value: Number(r.rmse_60.toFixed(1)), itemStyle: { color: t.s1, borderRadius: [0, 4, 4, 0] } })),
      label: { show: true, position: "right", color: t.ink2, fontSize: 11, formatter: "{c}" },
      markLine: tn ? { symbol: "none", silent: true, lineStyle: { color: t.ink2, type: "solid", width: 1 }, label: { formatter: "all", color: t.muted, fontSize: 10 }, data: [{ xAxis: Number(tn.rmse_60) }] } : undefined,
    }],
  });

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-[20px] font-semibold">Model evidence</h1>
        <p className="text-[13px] ink-2">
          Held-out evaluation: {syn.n_test_patients} unseen synthetic patients ({syn.n_test_anchors.toLocaleString()} forecasts, days 8–14 after personalisation)
          {real ? `, ${real.cgmacros.n_recordings} real CGMacros participants${real.bigideas ? `, ${real.bigideas.n_recordings} real BIG IDEAs participants with wristband HR and HRV` : ""} and ${real.shanghai.n_recordings} ShanghaiT2DM recordings (external population), all with 5-fold patient cross-validation.` : "."}
          {extra && " MadhuTwin = the mechanistic twin fitted to each person's own earlier data, TwinNet fine-tuned on the same data and averaged with LightGBM, with risks recalibrated on held-out patients."}
        </p>
      </div>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-5">
        <Stat label={`60-min forecast error (${extra ? "MadhuTwin" : "TwinNet"})`} value={tn ? `${n(tn.rmse_60)} mg/dL` : "–"} sub={gain60 != null ? `${gain60.toFixed(0)}% lower than persistence` : ""} />
        <Stat label="Clinically acceptable (Clarke A+B, 60 min)" value={tn ? `${n(tn.clarkeAB_60)}%` : "–"} sub="Zones A+B" />
        {spike?.caught_at_1fa_pct != null
          ? <Stat label="Spike alerts at ≤1 false alert/day" value={`${n(spike.caught_at_1fa_pct, 0)}% caught`} sub={`median ${n(spike.lead_at_1fa_min, 0)} min ahead · AUROC ${n(spike.auroc, 3)}`} />
          : <Stat label="Spike alerts" value={spike ? `${n(spike.detected_pct, 0)}% caught` : "–"} sub={spike ? `median ${n(spike.median_lead_min, 0)} min ahead · ${n(spike.false_alerts_per_day, 2)} false/day` : ""} />}
        {lc && <Stat label="The twin learns you (2-hour error)" value={`${n(lc[0].rmse_120)} → ${n(lc[lc.length - 1].rmse_120)}`} sub={`mg/dL, 0 → ${lc[lc.length - 1].days} days of personal data`} />}
        <Stat label="Illness detected from CGM (AUROC)" value={n(syn.illness_detection.auroc, 2)} sub="Synced twin, insulin-sensitivity drift" />
      </div>
      {realCohorts.length > 0 && (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          {realCohorts.filter(([, , c]) => c).map(([name, note, c]) => {
            const m = bestOf(c!), r = c!.forecast.find((x) => x.method === m), p = c!.forecast.find((x) => x.method === "Persistence");
            return <Stat key={name} label={`Real data · ${name} (${c!.n_recordings}) · 60-min error`} value={`${n(r?.rmse_60)} mg/dL`}
              sub={`Persistence ${n(p?.rmse_60)} · Clarke A+B ${n(r?.clarkeAB_60)}% · ${note}`} />;
          })}
        </div>
      )}
      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="Forecast error by horizon (synthetic test set)">
          <Legend items={lineMethods.map(([m, v]) => ({ label: m, color: `var(${v})` }))} />
          <Chart option={horizonChart} height={260} ariaLabel="RMSE by horizon per method" deps={[data]} />
        </Section>
        <Section title="Does fusing both streams matter? (ablation, RMSE at 60 min)">
          <Chart option={ablChart} height={Math.max(200, abl.length * 30)} ariaLabel="Ablation of data streams" deps={[data]} />
          <p className="mt-1 text-[12px] muted">LightGBM retrained for each stream combination on the same patients. Lower is better; the full hybrid is highlighted.</p>
        </Section>
      </div>
      {(lc || calSeries.length > 0) && (
        <div className="grid gap-4 lg:grid-cols-2">
          {lc && (
            <Section title={`The twin learns you (${extra!.curve_patients} unseen people with T2D, error on days 8–14)`}>
              <Legend items={[{ label: "1-hour forecast", color: "var(--series-1)" }, { label: "2-hour forecast", color: "var(--series-2)" }]} />
              <Chart option={learnChart} height={240} ariaLabel="Forecast error versus days of personal data" deps={[data]} />
              <p className="mt-1 text-[12px] muted">On day k the mechanistic twin is fitted to the person's first k days and TwinNet is fine-tuned on the same days, in seconds on a CPU. Day 0 is the population model with no personal data.</p>
            </Section>
          )}
          {calSeries.length > 0 && (
            <Section title="Are the spike risks honest? (calibration)">
              <Legend items={calSeries.map(([label, , v]) => ({ label, color: `var(${v})` }))} />
              <Chart option={calChart} height={240} ariaLabel="Calibration of spike risk" deps={[data]} />
              <p className="mt-1 text-[12px] muted">Points near the diagonal mean a "30% risk" comes true about 30% of the time (deciles of predicted risk). Both models are trained with class weighting, which inflates raw risks, so MadhuTwin's risks are recalibrated (Platt scaling): on validation patients for the synthetic cohort, and per real-data site on other patients only.</p>
            </Section>
          )}
        </div>
      )}
      {(dca || sub.length > 0) && (
        <div className="grid gap-4 lg:grid-cols-2">
          {dca && (
            <Section title={`Is alerting worth it? (decision curve, spike alerts, ${dcaName})`}>
              <Legend items={[{ label: "MadhuTwin alerts", color: "var(--series-1)" }, { label: "Alert everyone", color: "var(--deemph)" }, { label: "Alert no one", color: "var(--ink-2)" }]} />
              <Chart option={dcaChart} height={240} ariaLabel="Decision curve for spike alerts" deps={[data]} />
              <p className="mt-1 text-[12px] muted">Net benefit above both "alert everyone" and "alert no one" means acting on MadhuTwin's risk beats either default at that threshold (Vickers & Elkin 2006).</p>
            </Section>
          )}
          {sub.length > 0 && (
            <Section title="Does it work for everyone? (1-hour error by subgroup, synthetic test patients)">
              <Chart option={subChart} height={Math.max(200, sub.length * 22)} ariaLabel="Forecast error by subgroup" deps={[data]} />
              <p className="mt-1 text-[12px] muted">BMI groups use Asian cut-offs (23 and 27.5). Groups with fewer than 5 patients are hidden.</p>
            </Section>
          )}
        </div>
      )}
      <Section title="Glucose forecasting · synthetic test patients (95% CI by patient bootstrap)"><ForecastTable rows={forecast} /></Section>
      <Section title="Adverse-event prediction · synthetic test patients"><EventTable rows={events} /></Section>
      {syn.ablation_twinnet.length > 0 && (
        <Section title="TwinNet ablations (retrained without one stream)">
          <table className="data">
            <thead><tr><th>Variant</th>{HZ.map((h) => <th key={h} className="num">RMSE {h}</th>)}<th className="num">Spike AUROC</th><th className="num">Hypo AUROC</th></tr></thead>
            <tbody>
              {(() => {
                const hy = syn.forecast.find((r) => r.method === "TwinNet hybrid");
                const ev = (e: string) => syn.events.find((r) => r.event === e && r.method === "TwinNet hybrid")?.auroc;
                return [...(hy ? [{ config: "TwinNet hybrid (all streams)", ...hy, spike_auroc: ev("spike"), hypo_auroc: ev("hypo") }] : []), ...syn.ablation_twinnet].map((r) => (
                  <tr key={String(r.config)}><td className="font-medium">{String(r.config)}</td>{HZ.map((h) => <td key={h} className="num">{n((r as Row)[`rmse_${h}`])}</td>)}<td className="num">{n((r as Row).spike_auroc, 3)}</td><td className="num">{n((r as Row).hypo_auroc, 3)}</td></tr>
                ));
              })()}
            </tbody>
          </table>
        </Section>
      )}
      {real && (
        <>
          <RealSection title={`Real-world validation · CGMacros (${real.cgmacros.n_recordings} people, 5-fold patient CV)`} c={real.cgmacros} />
          {real.bigideas && <RealSection title={`Real wearables · BIG IDEAs (${real.bigideas.n_recordings} people; Dexcom CGM + Empatica wristband heart rate and HRV; 5-fold patient CV)`} c={real.bigideas} />}
          <RealSection title={`External validation · ShanghaiT2DM (${real.shanghai.n_recordings} recordings; different country, sensor, diet; no wearables)`} c={real.shanghai} />
        </>
      )}
      {data?.cgm_light && (
        <Section title={`CGM-light mode · 1 week of CGM, then ${data.cgm_light.fingersticks_per_day} fingersticks a day + smartwatch (${data.cgm_light.n_patients} unseen patients)`}>
          <table className="data">
            <thead><tr><th>Estimate of continuous glucose in week 2</th><th className="num">MARD</th><th className="num">Within 20%</th><th className="num">Clarke A+B</th><th className="num">Weekly TIR error</th></tr></thead>
            <tbody>
              <tr><td className="font-medium">Synced twin (UKF + physiology + wearables)</td><td className="num">{n(data.cgm_light.twin.mard)}%</td><td className="num">{n(data.cgm_light.twin.within_20pct, 0)}%</td><td className="num">{n(data.cgm_light.twin.clarke["A+B"])}%</td><td className="num">±{n(data.cgm_light.twin.tir_abs_error_pp)} pp</td></tr>
              <tr><td>Last fingerstick carried forward</td><td className="num">{n(data.cgm_light.carry_forward.mard)}%</td><td className="num">{n(data.cgm_light.carry_forward.within_20pct, 0)}%</td><td className="num">{n(data.cgm_light.carry_forward.clarke["A+B"])}%</td><td className="num">±{n(data.cgm_light.carry_forward.tir_abs_error_pp)} pp</td></tr>
            </tbody>
          </table>
          <p className="mt-2 text-[12px] muted">An affordability mode for India: a short CGM period personalises the twin, which then keeps estimating glucose and time in range from cheap fingersticks and smartwatch data. Measured on synthetic patients, whose simulator shares the twin's structure: an upper bound until validated on real data.</p>
        </Section>
      )}
      {data?.fidelity && (
        <Section title="Can the twin be trusted for a whole day? (24-hour open-loop replay of every day after calibration)">
          <div className="overflow-x-auto">
            <table className="data min-w-[760px]">
              <thead><tr><th>Cohort</th><th className="num">Days</th><th className="num">Median error</th><th className="num">Daily-mean error</th><th className="num">Time-in-range error</th><th className="num">Good · fair · poor</th></tr></thead>
              <tbody>
                {([["synthetic", "Synthetic test"], ["cgmacros", "CGMacros (real)"], ["bigideas", "BIG IDEAs (real)"], ["shanghai", "ShanghaiT2DM (real)"]] as const)
                  .filter(([k]) => data.fidelity![k]).map(([k, lab]) => {
                    const f = data.fidelity![k];
                    return <tr key={k}><td className="font-medium">{lab}</td><td className="num">{f.days}</td><td className="num">{n(f.mean_abs_error_median)} mg/dL</td>
                      <td className="num">{n(f.mean_glucose_error_median)} mg/dL</td><td className="num">{n(f.tir_error_median_pp)} pp</td>
                      <td className="num">{n(f.rating_pct.good, 0)}% · {n(f.rating_pct.fair, 0)}% · {n(f.rating_pct.poor, 0)}%</td></tr>;
                  })}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-[12px] muted">From the synced state at the start of each day, the twin replays the day with the logged meals, doses and activity. It tracks daily mean glucose and time in range, but rarely reproduces short lows, so the therapy simulator reports averages and time in range, shows this check for each patient, and leaves hypoglycaemia warnings to the 2-hour forecast.</p>
        </Section>
      )}
      {data?.robustness && (
        <Section title="Robust to missing data (one deployed model; data removed at inference)">
          <table className="data">
            <thead><tr><th>Missing at inference</th><th className="num">RMSE 60</th><th className="num">RMSE 120</th><th className="num">Spike AUROC</th></tr></thead>
            <tbody>
              {data.robustness.modalities.map((r) => <tr key={String(r.condition)}><td className="font-medium">{String(r.condition)}</td><td className="num">{n(r.rmse_60)}</td><td className="num">{n(r.rmse_120)}</td><td className="num">{n(r.spike_auroc, 3)}</td></tr>)}
              {data.robustness.cgm_dropout.map((r) => <tr key={String(r.cgm_readings_lost_pct)}><td>{n(r.cgm_readings_lost_pct, 0)}% of past CGM readings lost</td><td className="num">{n(r.rmse_60)}</td><td className="num">{n(r.rmse_120)}</td><td className="num">{n(r.spike_auroc, 3)}</td></tr>)}
            </tbody>
          </table>
          <p className="mt-2 text-[12px] muted">Trained with modality dropout, the same model keeps working without a smartwatch, an EHR or meal logs; the physics twin is the input it misses most.</p>
        </Section>
      )}
      {data?.bench && (
        <Section title={`Fast enough for a whole clinic on one CPU (${String(data.bench.threads)} threads, no GPU)`}>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Stat label="One 2-hour forecast" value={`${n(data.bench.twinnet_single_forecast_ms)} ms`} sub={`${Math.round(Number(data.bench.twinnet_throughput_per_s)).toLocaleString()} forecasts / s batched`} />
            <Stat label="Personalise a new patient" value={`${n(Number(data.bench.mechanistic_fit_7_days_s) + Number(data.bench.personalise_twinnet_s))} s`} sub="twin fit on 7 days + TwinNet fine-tune" />
            <Stat label="24-hour therapy simulation" value={`${n(Number(data.bench.therapy_sim_10_plans_24h_ms) / 10, 2)} ms`} sub="per plan; 10 plans at once" />
            <Stat label="Patients refreshed every 5 min" value={`${(Number(data.bench.patients_refreshed_per_5_min) / 1e6).toFixed(1)} M`} sub="compute-bound estimate" />
          </div>
        </Section>
      )}
      <Section title="Honest limitations">
        <ul className="list-disc space-y-1 pl-5 text-[13px] ink-2">
          <li>Synthetic patients are generated by a model with the same structure as the mechanistic twin, which flatters physics-based methods there; real-world and external results are the primary evidence.</li>
          <li>CGMacros (45 people, ~10 days), BIG IDEAs (16 people with normal glucose or prediabetes, ~10 days) and ShanghaiT2DM (100 people) are small and none is Indian; confidence intervals are reported by patient bootstrap.</li>
          <li>Personalisation only ever uses a person's earlier data, never the period being scored. Forecasts use only meals and doses logged up to the moment of prediction; unlogged meals are the main source of 2-hour error.</li>
          <li>Synthetic-only models do not transfer to a new population without fine-tuning (see ShanghaiT2DM zero-shot); sim-to-real fine-tuning is required. Alert thresholds are chosen on training folds; real deployments need clinician-tuned thresholds.</li>
          <li>Real-world hypoglycaemia events are rare and mostly in people without diabetes (CGMacros), so hypoglycaemia precision on real data is not yet established.</li>
          <li>The 24-hour therapy simulator rests on the mechanistic twin: it tracks daily mean glucose and time in range but rarely short lows, does not model metformin (which acts on the fasting set-point over weeks), and has not yet been validated against real dose changes.</li>
          <li>No prospective or Indian clinical validation yet; this is a proof of concept, not a medical device.</li>
        </ul>
      </Section>
    </div>
  );
}
