import type { EChartsOption } from "echarts";
import { State } from "../api";
import Chart, { Tokens, alpha, baseOption, timeAxis, valueAxis } from "./Chart";

/** Small multiples sharing one time axis and one crosshair: never two scales on one plot. */
export default function WearablesPanel({ state }: { state: State }) {
  const ts = state.times;
  const s = state.series;
  const hasWear = s.hr.some((v) => v != null) || s.steps.some((v) => v != null);
  if (!hasWear) {
    return (
      <p className="text-[13px] ink-2">
        No wearable heart-rate, HRV, activity or sleep data in this record (ShanghaiT2DM has CGM + EHR only). The twin and TwinNet fall back to
        CGM + EHR + logged meals and medication: modality dropout during training makes this a supported mode, evaluated in the external validation.
      </p>
    );
  }
  // hourly step totals (columns), from 5-min bins
  const hourly: [string, number][] = [];
  for (let i = 0; i < ts.length; i += 12) {
    const chunk = s.steps.slice(i, i + 12).filter((v): v is number => v != null);
    hourly.push([ts[i], chunk.reduce((a, b) => a + b, 0)]);
  }
  const common = (t: Tokens, title: string, extra: EChartsOption): EChartsOption => ({
    ...baseOption(t),
    title: { text: title, left: 0, top: 0, textStyle: { fontSize: 12, fontWeight: 500, color: t.ink2 } },
    grid: { left: 44, right: 16, top: 26, bottom: 22 },
    xAxis: timeAxis(t, { min: ts[0], max: ts[ts.length - 1] }),
    ...extra,
  });
  const charts: { key: string; h: number; option: (t: Tokens) => EChartsOption; label: string }[] = [
    {
      key: "hr", h: 150, label: "Heart rate",
      option: (t) => common(t, "Heart rate (bpm)", {
        yAxis: valueAxis(t, { scale: true }),
        series: [{ name: "Heart rate", type: "line", data: ts.map((x, i) => [x, s.hr[i]]), showSymbol: false, lineStyle: { width: 2, color: t.s1 }, itemStyle: { color: t.s1 } }],
      }),
    },
    {
      key: "hrv", h: 150, label: "HRV",
      option: (t) => common(t, "HRV, RMSSD (ms) · measured at rest", {
        yAxis: valueAxis(t, { scale: true }),
        series: [{ name: "HRV", type: "scatter", data: ts.map((x, i) => [x, s.hrv[i]]).filter((d) => d[1] != null), symbolSize: 5, itemStyle: { color: alpha(t.s1, 0.7) } }],
      }),
    },
    {
      key: "steps", h: 150, label: "Steps per hour",
      option: (t) => common(t, "Steps per hour", {
        yAxis: valueAxis(t),
        series: [{ name: "Steps", type: "bar", data: hourly, barMaxWidth: 14, itemStyle: { color: t.s1, borderRadius: [4, 4, 0, 0] } }],
      }),
    },
    {
      key: "sleep", h: 150, label: "Sleep stages",
      option: (t) => common(t, "Sleep stage (hypnogram)", {
        yAxis: { type: "category", data: ["Awake", "Light", "Deep", "REM"].reverse(), axisLine: { show: false }, axisTick: { show: false },
                 axisLabel: { color: t.muted, fontSize: 11 }, splitLine: { show: true, lineStyle: { color: t.grid } } },
        series: [{
          name: "Sleep stage", type: "line", step: "end", showSymbol: false, lineStyle: { width: 2, color: t.s1 },
          data: ts.map((x, i) => [x, s.sleep[i] == null ? null : ["Awake", "Light", "Deep", "REM"][s.sleep[i] as number]]),
        }],
      }),
    },
  ];
  return (
    <div className="space-y-2">
      {charts.map((c) => (
        <Chart key={c.key} option={c.option} height={c.h} group={`wear-${state.id}`} ariaLabel={c.label} deps={[state]} />
      ))}
      <p className="text-[12px] muted">
        Wearable signals feed the twin: activity raises glucose uptake, poor sleep lowers next-day insulin sensitivity, and illness or stress shows as
        higher resting heart rate with suppressed HRV.
      </p>
    </div>
  );
}
