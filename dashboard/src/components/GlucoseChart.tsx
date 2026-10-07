import type { EChartsOption } from "echarts";
import { State } from "../api";
import Chart, { Tokens, alpha, baseOption, timeAxis, valueAxis } from "./Chart";
import { Legend } from "./ui";

const LANES = ["Alerts", "Medication", "Meals", "Sleep"];

function eventLabel(e: State["events"][number]) {
  if (e.kind === "meal") return `${e.food ? e.food.replace(/_/g, " ") : e.label}${e.carbs != null ? ` · ${Math.round(e.carbs)} g carbs` : ""}`;
  if (e.kind === "insulin") return `${e.label.replace(/_/g, " ")} · ${e.amount ?? ""} IU`;
  return `${e.label.replace(/_/g, " ")}${e.amount ? ` · ${e.amount} mg` : ""}`;
}

export default function GlucoseChart({ state, showTwin = true }: { state: State; showTwin?: boolean }) {
  const fc = state.forecast;
  const option = (t: Tokens): EChartsOption => {
    const obs = state.times.map((ts, i) => [ts, state.series.cgm[i]]);
    const nowTs = state.now;
    const lastObs = [...state.series.cgm].reverse().find((v) => v != null) ?? null;
    const fTimes = fc ? [fc.anchor_time, ...fc.times] : [];
    const q = (arr: (number | null)[] | undefined) => (fc && arr ? [lastObs, ...arr] : []);
    const q10 = q(fc?.q10), q50 = q(fc?.q50), q90 = q(fc?.q90), tw = q(fc?.twin);
    const band = fTimes.map((ts, i) => [ts, q90[i] != null && q10[i] != null ? (q90[i] as number) - (q10[i] as number) : null]);
    const maxY = Math.max(300, ...state.series.cgm.filter((v): v is number => v != null), ...(q90.filter((v) => v != null) as number[])) + 10;

    const sleepPts = state.times.filter((_, i) => (state.series.sleep[i] ?? 0) > 0).map((ts) => [ts, "Sleep"]);
    const meals = state.events.filter((e) => e.kind === "meal").map((e) => ({ value: [e.time, "Meals"], name: eventLabel(e) }));
    const meds = state.events.filter((e) => e.kind === "insulin" || e.kind === "oad").map((e) => ({ value: [e.time, "Medication"], name: eventLabel(e), symbol: e.kind === "insulin" ? "diamond" : "rect" }));
    const startTs = state.times[0];
    const alerts = state.alerts.filter((a) => a.time && a.time >= startTs).map((a) => ({
      value: [a.time!, "Alerts"],
      name: `${a.kind === "hypo" ? "Low" : "High"} glucose predicted (${Math.round(a.prob * 100)}%): ${a.predicted_value} mg/dL in ${a.in_min} min`,
      itemStyle: { color: a.kind === "hypo" ? t.critical : t.warning },
    }));
    const xMax = fTimes.length ? fTimes[fTimes.length - 1] : nowTs;

    return {
      ...baseOption(t),
      grid: [
        { left: 84, right: 96, top: 16, height: 330 },
        { left: 84, right: 96, top: 366, height: 88 },
      ],
      xAxis: [
        timeAxis(t, { gridIndex: 0, min: startTs, max: xMax, axisLabel: { show: false } }),
        timeAxis(t, { gridIndex: 1, min: startTs, max: xMax }),
      ],
      yAxis: [
        valueAxis(t, { gridIndex: 0, min: 40, max: Math.ceil(maxY / 100) * 100, name: "mg/dL", nameTextStyle: { color: t.muted, fontSize: 11 } }),
        { type: "category", gridIndex: 1, data: LANES, axisLine: { show: false }, axisTick: { show: false },
          axisLabel: { color: t.muted, fontSize: 11 }, splitLine: { show: true, lineStyle: { color: t.grid } } },
      ],
      axisPointer: { link: [{ xAxisIndex: "all" }] },
      tooltip: {
        ...(baseOption(t).tooltip as object),
        formatter: (ps: unknown) => {
          const arr = (Array.isArray(ps) ? ps : [ps]) as { seriesName: string; value: [string, number | string | null]; dataIndex: number }[];
          if (!arr.length) return "";
          const ts = new Date(arr[0].value[0]).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
          const rows: string[] = [];
          const line = (c: string, v: string, label: string) =>
            `<div style="display:flex;align-items:center;gap:8px"><span style="width:12px;height:2px;background:${c};display:inline-block"></span><b>${v}</b><span style="color:${t.ink2}">${label}</span></div>`;
          for (const p of arr) {
            const v = p.value[1];
            if (v == null || typeof v === "string") continue;
            if (p.seriesName === "Observed CGM") rows.push(line(t.s1, `${Math.round(v)} mg/dL`, "observed"));
            if (p.seriesName === "Forecast") {
              const i = p.dataIndex;
              rows.push(line(t.s2, `${Math.round(v)} mg/dL`, `forecast (80%: ${q10[i] == null ? "–" : Math.round(q10[i] as number)}–${q90[i] == null ? "–" : Math.round(q90[i] as number)})`));
            }
            if (p.seriesName === "Physics twin") rows.push(line(t.s3, `${Math.round(v)} mg/dL`, "physics twin"));
          }
          return `<div style="font-size:12px"><div style="color:${t.muted};margin-bottom:4px">${ts}</div>${rows.join("")}</div>`;
        },
      },
      series: [
        {
          name: "Observed CGM", type: "line", xAxisIndex: 0, yAxisIndex: 0, data: obs, showSymbol: false, connectNulls: false,
          lineStyle: { width: 2, color: t.s1, cap: "round", join: "round" }, itemStyle: { color: t.s1 }, z: 5,
          endLabel: { show: false },
          markArea: { silent: true, itemStyle: { color: t.rangeWash }, data: [[{ yAxis: 70 }, { yAxis: 180 }]] },
          markLine: {
            silent: true, symbol: "none",
            label: { color: t.muted, fontSize: 11, position: "end", formatter: (p: { value?: unknown; name?: string }) => p.name ?? "" },
            lineStyle: { color: t.axis, width: 1, type: "solid" },
            data: [
              { yAxis: 180, name: "180" },
              { yAxis: 70, name: "70" },
              { xAxis: nowTs, name: "Now", lineStyle: { color: t.ink2, width: 1 }, label: { position: "start", color: t.ink2, formatter: "Now" } },
            ],
          },
        },
        { name: "lo", type: "line", xAxisIndex: 0, yAxisIndex: 0, data: fTimes.map((ts, i) => [ts, q10[i]]), stack: "cone", showSymbol: false, lineStyle: { opacity: 0 }, silent: true, tooltip: { show: false } },
        { name: "band", type: "line", xAxisIndex: 0, yAxisIndex: 0, data: band, stack: "cone", showSymbol: false, lineStyle: { opacity: 0 }, areaStyle: { color: alpha(t.s2, 0.16) }, silent: true, tooltip: { show: false } },
        {
          name: "Forecast", type: "line", xAxisIndex: 0, yAxisIndex: 0, data: fTimes.map((ts, i) => [ts, q50[i]]), showSymbol: false,
          lineStyle: { width: 2, color: t.s2 }, itemStyle: { color: t.s2 }, z: 6,
          endLabel: { show: !!fc, color: t.ink, fontSize: 11, formatter: (p: { value?: unknown }) => `${Math.round((p.value as [string, number])[1])} in 2 h` },
        },
        ...(showTwin ? [{
          name: "Physics twin", type: "line" as const, xAxisIndex: 0, yAxisIndex: 0, data: fTimes.map((ts, i) => [ts, tw[i]]), showSymbol: false,
          lineStyle: { width: 2, color: t.s3 }, itemStyle: { color: t.s3 }, z: 4,
          endLabel: { show: !!fc, color: t.ink2, fontSize: 11, formatter: "Twin", offset: [0, 12] as [number, number] },
        }] : []),
        { name: "sleep", type: "scatter", xAxisIndex: 1, yAxisIndex: 1, data: sleepPts, symbol: "rect", symbolSize: [4, 10], itemStyle: { color: alpha(t.deemph, 0.6) }, silent: true, tooltip: { show: false } },
        { name: "meals", type: "scatter", xAxisIndex: 1, yAxisIndex: 1, data: meals, symbol: "circle", symbolSize: 10,
          itemStyle: { color: t.ink2, borderColor: t.surface, borderWidth: 2 }, tooltip: { trigger: "item", formatter: (p: { name: string }) => p.name } },
        { name: "meds", type: "scatter", xAxisIndex: 1, yAxisIndex: 1, data: meds, symbolSize: 10,
          itemStyle: { color: t.ink2, borderColor: t.surface, borderWidth: 2 }, tooltip: { trigger: "item", formatter: (p: { name: string }) => p.name } },
        { name: "alerts", type: "scatter", xAxisIndex: 1, yAxisIndex: 1, data: alerts, symbol: "triangle", symbolSize: 12,
          itemStyle: { borderColor: t.surface, borderWidth: 2 }, tooltip: { trigger: "item", formatter: (p: { name: string }) => p.name } },
      ],
    };
  };
  return (
    <div>
      <div className="mb-2">
        <Legend items={[
          { label: "Observed CGM", color: "var(--series-1)" },
          { label: "TwinNet forecast (median)", color: "var(--series-2)" },
          { label: "80% interval (conformal)", color: "rgba(235,104,52,0.25)", kind: "area" },
          ...(showTwin ? [{ label: "Physics twin projection", color: "var(--series-3)" }] : []),
          { label: "Target 70–180 mg/dL", color: "var(--range-wash)", kind: "area" as const },
        ]} />
      </div>
      <Chart option={option} height={480} ariaLabel="Glucose: last 24 hours and 2-hour forecast" deps={[state, showTwin]} />
    </div>
  );
}
