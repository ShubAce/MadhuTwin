import type { EChartsOption } from "echarts";
import { Agp } from "../api";
import Chart, { Tokens, alpha, baseOption, valueAxis } from "./Chart";
import { Legend, Status } from "./ui";

/** Ambulatory Glucose Profile per the 2019 international consensus on time in range. */
export default function AgpPanel({ agp }: { agp: Agp }) {
  const segs = [
    { key: "tar2", label: "Very high >250", v: agp.tar2, color: "var(--serious)", target: "<5%", ok: agp.tar2 < 5 },
    { key: "tar1", label: "High 181–250", v: agp.tar1, color: "var(--warning)", target: "<25%", ok: agp.tar1 < 25 },
    { key: "tir", label: "In range 70–180", v: agp.tir, color: "var(--good)", target: ">70%", ok: agp.tir > 70 },
    { key: "tbr1", label: "Low 54–69", v: agp.tbr1, color: "var(--critical)", target: "<4%", ok: agp.tbr1 < 4 },
    { key: "tbr2", label: "Very low <54", v: agp.tbr2, color: "var(--critical-2)", target: "<1%", ok: agp.tbr2 < 1 },
  ];
  const prof = agp.profile;
  const option = (t: Tokens): EChartsOption => {
    const x = prof.map((p) => p.time);
    const val = (k: "p5" | "p25" | "p50" | "p75" | "p95") => prof.map((p) => p[k] ?? null);
    const diff = (a: (number | null)[], b: (number | null)[]) => a.map((v, i) => (v != null && b[i] != null ? v - (b[i] as number) : null));
    const p5 = val("p5"), p25 = val("p25"), p50 = val("p50"), p75 = val("p75"), p95 = val("p95");
    return {
      ...baseOption(t),
      grid: { left: 44, right: 16, top: 12, bottom: 28 },
      xAxis: { type: "category", data: x, boundaryGap: false, axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false },
               axisLabel: { color: t.muted, fontSize: 11, interval: 5 } },
      yAxis: valueAxis(t, { min: 40, max: 350 }),
      tooltip: {
        ...(baseOption(t).tooltip as object),
        formatter: (ps: unknown) => {
          const i = (ps as { dataIndex: number }[])[0]?.dataIndex ?? 0;
          const f = (v: number | null) => (v == null ? "–" : Math.round(v));
          return `<div style="font-size:12px"><div style="color:${t.muted}">${x[i]}</div><b>${f(p50[i])}</b> median · 25–75%: ${f(p25[i])}–${f(p75[i])} · 5–95%: ${f(p5[i])}–${f(p95[i])} mg/dL</div>`;
        },
      },
      series: [
        { name: "p5", type: "line", data: p5, stack: "outer", showSymbol: false, lineStyle: { opacity: 0 }, silent: true,
          markArea: { silent: true, itemStyle: { color: t.rangeWash }, data: [[{ yAxis: 70 }, { yAxis: 180 }]] } },
        { name: "5-95", type: "line", data: diff(p95, p5), stack: "outer", showSymbol: false, lineStyle: { opacity: 0 }, areaStyle: { color: alpha(t.s1, 0.1) }, silent: true },
        { name: "p25", type: "line", data: p25, stack: "inner", showSymbol: false, lineStyle: { opacity: 0 }, silent: true },
        { name: "25-75", type: "line", data: diff(p75, p25), stack: "inner", showSymbol: false, lineStyle: { opacity: 0 }, areaStyle: { color: alpha(t.s1, 0.28) }, silent: true },
        { name: "Median", type: "line", data: p50, showSymbol: false, lineStyle: { width: 2, color: t.s1 }, itemStyle: { color: t.s1 } },
      ],
    };
  };
  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
      <div>
        <div className="mb-2 flex flex-wrap items-center gap-3">
          <h3 className="text-[13px] font-medium">Ambulatory glucose profile · last {agp.days.toFixed(0)} days</h3>
          <Legend items={[
            { label: "Median", color: "var(--series-1)" },
            { label: "25–75th percentile", color: "rgba(42,120,214,0.28)", kind: "area" },
            { label: "5–95th percentile", color: "rgba(42,120,214,0.1)", kind: "area" },
            { label: "Target 70–180", color: "var(--range-wash)", kind: "area" },
          ]} />
        </div>
        <Chart option={option} height={280} ariaLabel="Ambulatory glucose profile" deps={[agp]} />
      </div>
      <div className="space-y-3">
        <div>
          <h3 className="mb-2 text-[13px] font-medium">Time in ranges</h3>
          <div className="flex h-4 w-full gap-[2px] overflow-hidden rounded" role="img" aria-label="Time in ranges stacked bar">
            {segs.map((s) => (
              <span key={s.key} style={{ width: `${Math.max(s.v, 0)}%`, background: s.color }} title={`${s.label}: ${s.v.toFixed(1)}%`} />
            ))}
          </div>
          <table className="data mt-2">
            <tbody>
              {segs.map((s) => (
                <tr key={s.key}>
                  <td><span className="mr-2 inline-block h-2.5 w-2.5 rounded-sm align-middle" style={{ background: s.color }} />{s.label}</td>
                  <td className="num font-medium">{s.v.toFixed(1)}%</td>
                  <td className="num"><Status level={s.ok ? "good" : "warning"}>{s.target}</Status></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <table className="data">
          <tbody>
            <tr><td>Mean glucose</td><td className="num font-medium">{agp.mean.toFixed(0)} mg/dL</td></tr>
            <tr><td>GMI (estimated A1c)</td><td className="num font-medium">{agp.gmi.toFixed(1)}%</td></tr>
            <tr><td>Glycaemic variability (CV)</td><td className="num font-medium">{agp.cv.toFixed(0)}% <span className="muted">(target ≤36%)</span></td></tr>
            <tr><td>CGM active</td><td className="num font-medium">{agp.active_pct.toFixed(0)}%</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
