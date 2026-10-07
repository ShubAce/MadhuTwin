import type { EChartsOption } from "echarts";
import Chart, { Tokens, baseOption, valueAxis } from "../components/Chart";
import { Legend, Section, Stat } from "../components/ui";
import { useData } from "../hooks";

type Row = Record<string, number | string | number[] | string[] | null>;
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
  real?: { cgmacros: { forecast: Row[]; events: Row[]; n_recordings: number; n_anchors: number }; shanghai: { forecast: Row[]; events: Row[]; n_recordings: number; n_anchors: number } };
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
const LINE_METHODS: [string, string][] = [["TwinNet hybrid", "--series-1"], ["LightGBM fusion", "--series-2"], ["Twin (personalised)", "--series-3"], ["Persistence", "--deemph"]];

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
            <tr key={String(r.method)}>
              <td className="font-medium">{String(r.method)}</td>
              {HZ.map((h) => (
                <td key={h} className="num">
                  {n(r[`rmse_${h}`])}
                  {Array.isArray(r[`rmse_${h}_ci`]) && <span className="muted"> ({n((r[`rmse_${h}_ci`] as number[])[0])}–{n((r[`rmse_${h}_ci`] as number[])[1])})</span>}
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
      <table className="data min-w-[760px]">
        <thead>
          <tr><th>Event</th><th>Method</th><th className="num">AUROC (95% CI)</th><th className="num">AUPRC</th><th className="num">Prevalence</th>
            <th className="num">Excursions caught</th><th className="num">≥30 min ahead</th><th className="num">Median lead</th><th className="num">False alerts / day</th></tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td>{r.event === "spike" ? "Spike >180" : "Hypo <70"}</td>
              <td className="font-medium">{String(r.method)}</td>
              <td className="num">{n(r.auroc, 3)}{Array.isArray(r.auroc_ci) && <span className="muted"> ({n((r.auroc_ci as number[])[0], 2)}–{n((r.auroc_ci as number[])[1], 2)})</span>}</td>
              <td className="num">{n(r.auprc, 3)}</td>
              <td className="num">{n(r.prevalence)}%</td>
              <td className="num">{r.detected_pct != null ? `${n(r.detected_pct, 0)}% of ${r.excursions}` : "–"}</td>
              <td className="num">{r.detected_30min_ahead_pct != null ? `${n(r.detected_30min_ahead_pct, 0)}%` : "–"}</td>
              <td className="num">{r.median_lead_min != null ? `${n(r.median_lead_min, 0)} min` : "–"}</td>
              <td className="num">{r.false_alerts_per_day != null ? n(r.false_alerts_per_day, 2) : "–"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Evidence() {
  const { data } = useData<Evidence>("/api/evidence");
  const syn = data?.synthetic;
  const real = data?.real;
  if (!syn) return <p className="muted">Evaluation results not found. Run scripts/exp_synthetic.py and scripts/exp_real.py.</p>;
  const by = (m: string) => syn.forecast.find((r) => r.method === m);
  const tn = by("TwinNet hybrid"), pers = by("Persistence");
  const spikeTN = syn.events.find((r) => r.event === "spike" && r.method === "TwinNet hybrid");
  const gain60 = tn && pers ? (1 - Number(tn.rmse_60) / Number(pers.rmse_60)) * 100 : null;
  const cg = real?.cgmacros.forecast.find((r) => r.method === "TwinNet sim-to-real");
  const cgP = real?.cgmacros.forecast.find((r) => r.method === "Persistence");

  const horizonChart = (t: Tokens): EChartsOption => ({
    ...baseOption(t),
    grid: { left: 44, right: 130, top: 14, bottom: 28 },
    xAxis: { type: "category", data: HZ.map((h) => `${h} min`), axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false }, axisLabel: { color: t.muted, fontSize: 11 } },
    yAxis: valueAxis(t, { name: "RMSE mg/dL", nameTextStyle: { color: t.muted, fontSize: 11 } }),
    series: LINE_METHODS.filter(([m]) => by(m)).map(([m, v]) => {
      const c = getComputedStyle(document.documentElement).getPropertyValue(v).trim();
      return {
        name: m, type: "line" as const, data: HZ.map((h) => Number(by(m)![`rmse_${h}`])), symbol: "circle", symbolSize: 8,
        lineStyle: { width: 2, color: c }, itemStyle: { color: c, borderColor: t.surface, borderWidth: 2 },
        endLabel: { show: true, color: t.ink2, fontSize: 11, formatter: m },
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

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-[20px] font-semibold">Model evidence</h1>
        <p className="text-[13px] ink-2">
          Held-out evaluation: {syn.n_test_patients} unseen synthetic patients ({syn.n_test_anchors.toLocaleString()} forecasts, days 8–14 after personalisation),
          {real ? ` ${real.cgmacros.n_recordings} real CGMacros participants (5-fold patient cross-validation) and ${real.shanghai.n_recordings} ShanghaiT2DM recordings (external population).` : ""}
        </p>
      </div>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="60-min forecast error (TwinNet)" value={tn ? `${n(tn.rmse_60)} mg/dL` : "–"} sub={gain60 != null ? `${gain60.toFixed(0)}% lower than persistence` : ""} />
        <Stat label="Clinically acceptable (Clarke A+B, 60 min)" value={tn ? `${n(tn.clarkeAB_60)}%` : "–"} sub="Zones A+B" />
        <Stat label="Spike alerts" value={spikeTN ? `${n(spikeTN.detected_pct, 0)}% caught` : "–"} sub={spikeTN ? `median ${n(spikeTN.median_lead_min, 0)} min ahead · ${n(spikeTN.false_alerts_per_day, 2)} false/day` : ""} />
        <Stat label="Illness detected from CGM (AUROC)" value={n(syn.illness_detection.auroc, 2)} sub="Synced twin, insulin-sensitivity drift" />
      </div>
      {cg && cgP && (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <Stat label="Real data (CGMacros) · 60-min error" value={`${n(cg.rmse_60)} mg/dL`} sub={`Persistence ${n(cgP.rmse_60)} · sim-to-real TwinNet`} />
          <Stat label="Real data (CGMacros) · Clarke A+B at 60 min" value={`${n(cg.clarkeAB_60)}%`} sub="Pretrained on synthetic, fine-tuned on real folds" />
        </div>
      )}
      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="Forecast error by horizon (synthetic test set)">
          <Legend items={LINE_METHODS.filter(([m]) => by(m)).map(([m, v]) => ({ label: m, color: `var(${v})` }))} />
          <Chart option={horizonChart} height={260} ariaLabel="RMSE by horizon per method" deps={[data]} />
        </Section>
        <Section title="Does fusing both streams matter? (ablation, RMSE at 60 min)">
          <Chart option={ablChart} height={Math.max(200, abl.length * 30)} ariaLabel="Ablation of data streams" deps={[data]} />
          <p className="mt-1 text-[12px] muted">LightGBM retrained for each stream combination on the same patients. Lower is better; the full hybrid is highlighted.</p>
        </Section>
      </div>
      <Section title="Glucose forecasting · synthetic test patients (95% CI by patient bootstrap)"><ForecastTable rows={syn.forecast} /></Section>
      <Section title="Adverse-event prediction · synthetic test patients"><EventTable rows={syn.events} /></Section>
      {syn.ablation_twinnet.length > 0 && (
        <Section title="TwinNet ablations (retrained without one stream)">
          <table className="data">
            <thead><tr><th>Variant</th>{HZ.map((h) => <th key={h} className="num">RMSE {h}</th>)}<th className="num">Spike AUROC</th><th className="num">Hypo AUROC</th></tr></thead>
            <tbody>
              {[...(tn ? [{ config: "TwinNet hybrid (all streams)", ...tn, spike_auroc: spikeTN?.auroc, hypo_auroc: syn.events.find((r) => r.event === "hypo" && r.method === "TwinNet hybrid")?.auroc }] : []), ...syn.ablation_twinnet].map((r) => (
                <tr key={String(r.config)}><td className="font-medium">{String(r.config)}</td>{HZ.map((h) => <td key={h} className="num">{n((r as Row)[`rmse_${h}`])}</td>)}<td className="num">{n((r as Row).spike_auroc, 3)}</td><td className="num">{n((r as Row).hypo_auroc, 3)}</td></tr>
              ))}
            </tbody>
          </table>
        </Section>
      )}
      {real && (
        <>
          <Section title={`Real-world validation · CGMacros (${real.cgmacros.n_recordings} people, 5-fold patient CV)`}>
            <ForecastTable rows={real.cgmacros.forecast} />
            <div className="mt-4"><EventTable rows={real.cgmacros.events} /></div>
          </Section>
          <Section title={`External validation · ShanghaiT2DM (${real.shanghai.n_recordings} recordings; different country, sensor, diet; no wearables)`}>
            <ForecastTable rows={real.shanghai.forecast} />
            <div className="mt-4"><EventTable rows={real.shanghai.events} /></div>
          </Section>
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
          <p className="mt-2 text-[12px] muted">An affordability mode for India: a short CGM period personalises the twin, which then keeps estimating glucose and time in range from cheap fingersticks and smartwatch data.</p>
        </Section>
      )}
      <Section title="Honest limitations">
        <ul className="list-disc space-y-1 pl-5 text-[13px] ink-2">
          <li>Synthetic patients are generated by a model with the same structure as the mechanistic twin, which flatters physics-based methods there; real-world and external results are the primary evidence.</li>
          <li>CGMacros (45 people, ~10 days) and ShanghaiT2DM (100 people) are small; confidence intervals are reported by patient bootstrap.</li>
          <li>Forecasts use only meals and doses logged up to the moment of prediction. Unlogged meals are the main source of 2-hour error.</li>
          <li>No prospective or Indian clinical validation yet; this is a proof of concept, not a medical device.</li>
        </ul>
      </Section>
    </div>
  );
}
