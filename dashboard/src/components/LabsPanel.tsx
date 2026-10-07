import type { EChartsOption } from "echarts";
import { PatientDetail } from "../api";
import Chart, { Tokens, baseOption, valueAxis } from "./Chart";

export default function LabsPanel({ p }: { p: PatientDetail }) {
  const e = p.ehr;
  const hist = e.lab_history ?? [];
  const a1cNow = e.labs.find((l) => l.key === "hba1c_pct");
  const option = (t: Tokens): EChartsOption => {
    const pts = [...hist.map((h) => [h.date, h.hba1c_pct]), ...(a1cNow ? [["Today", a1cNow.value]] : [])];
    return {
      ...baseOption(t),
      grid: { left: 40, right: 56, top: 14, bottom: 26 },
      xAxis: { type: "category", data: pts.map((x) => String(x[0]).slice(0, 7)), axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false }, axisLabel: { color: t.muted, fontSize: 11 } },
      yAxis: valueAxis(t, { min: 4, max: Math.max(10, ...pts.map((x) => Number(x[1]))) + 0.5 }),
      series: [{
        name: "HbA1c", type: "line", data: pts.map((x) => x[1]), symbol: "circle", symbolSize: 8,
        lineStyle: { width: 2, color: t.s1 }, itemStyle: { color: t.s1, borderColor: t.surface, borderWidth: 2 },
        endLabel: { show: true, color: t.ink, fontSize: 11, formatter: (q: { value?: unknown }) => `${Number(q.value).toFixed(1)}%` },
        markLine: { silent: true, symbol: "none", lineStyle: { color: t.axis, type: "solid" }, label: { color: t.muted, fontSize: 11, formatter: "7% target" }, data: [{ yAxis: 7 }] },
      }],
    };
  };
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="space-y-4">
        {hist.length > 0 && (
          <div>
            <h3 className="mb-1 text-[13px] font-medium">HbA1c over 2 years (EHR, quarterly)</h3>
            <Chart option={option} height={200} ariaLabel="HbA1c history" deps={[p]} />
          </div>
        )}
        <div>
          <h3 className="mb-1 text-[13px] font-medium">Latest laboratory results</h3>
          <table className="data">
            <thead><tr><th>Test</th><th className="num">Value</th><th>LOINC</th></tr></thead>
            <tbody>
              {e.labs.map((l) => (
                <tr key={l.key}><td>{l.display.split("[")[0].replace(/\/Hemoglobin.total in Blood/, "")}</td><td className="num">{l.value} {l.unit}</td><td className="muted tabular">{l.loinc}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="space-y-4">
        <div>
          <h3 className="mb-1 text-[13px] font-medium">Medications</h3>
          <table className="data">
            <thead><tr><th>Drug</th><th>Dose & timing</th><th>ATC</th></tr></thead>
            <tbody>
              {e.medications.length ? e.medications.map((m, i) => (
                <tr key={i}><td>{m.display}</td><td>{m.dose != null ? `${m.dose} ${m.unit ?? ""}` : ""} {m.times?.length ? `at ${m.times.join(", ")}` : ""}</td><td className="muted tabular">{m.atc ?? ""}</td></tr>
              )) : <tr><td colSpan={3} className="muted">No glucose-lowering medication recorded</td></tr>}
            </tbody>
          </table>
          {e.adherence != null && <p className="mt-1 text-[12px] muted">Typical adherence (synthetic profile): {(e.adherence * 100).toFixed(0)}%</p>}
        </div>
        <div>
          <h3 className="mb-1 text-[13px] font-medium">Problem list</h3>
          <div className="flex flex-wrap gap-1.5">
            {e.conditions.map((c) => (
              <span key={c.key} className="chip" title={c.snomed ? `SNOMED CT ${c.snomed}` : undefined}>
                {c.display}{c.years != null ? ` · ${c.years < 1 ? "<1" : c.years.toFixed(0)} y` : ""}
              </span>
            ))}
          </div>
        </div>
        {(e.genetics || e.family_history.length > 0) && (
          <div>
            <h3 className="mb-1 text-[13px] font-medium">Family history & genetic markers</h3>
            <ul className="space-y-1 text-[13px] ink-2">
              {e.family_history.map((f) => <li key={f}>{f.charAt(0).toUpperCase() + f.slice(1)}</li>)}
              {e.genetics && <li>TCF7L2 rs7903146: {e.genetics.TCF7L2_rs7903146}</li>}
              {e.genetics && <li>Type 2 diabetes polygenic risk score: {e.genetics.prs_z > 0 ? "+" : ""}{e.genetics.prs_z.toFixed(2)} SD</li>}
            </ul>
          </div>
        )}
        {e.vitals?.sbp && <p className="text-[13px] ink-2">Blood pressure {e.vitals.sbp}/{e.vitals.dbp} mmHg · BMI {e.bmi?.toFixed(1)} kg/m² · waist {e.waist_cm?.toFixed(0)} cm</p>}
      </div>
    </div>
  );
}
