import type { EChartsOption } from "echarts";
import { FlaskConical, Play } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Food, InsightsResponse, WhatIfResult, post } from "../api";
import { useData } from "../hooks";
import Chart, { Tokens, baseOption, timeAxis, valueAxis } from "./Chart";
import { Legend } from "./ui";

const GROUPS: [string, string][] = [["breakfast", "Breakfast"], ["lunch", "Lunch"], ["dinner", "Dinner"], ["snack", "Snacks & sweets"]];

export default function WhatIfPanel({ id, clock, insights, preset }: { id: string; clock: number; insights: InsightsResponse | null; preset?: URLSearchParams }) {
  const { data: foods } = useData<Food[]>("/api/foods");
  const num = (k: string, d: number) => (preset?.get(k) != null && !Number.isNaN(Number(preset.get(k))) ? Number(preset.get(k)) : d);
  const [food, setFood] = useState(preset?.get("food") ?? "rice_sambar");
  const [portion, setPortion] = useState(num("portion", 1));
  const [inMin, setInMin] = useState(num("in", 0));
  const [walk, setWalk] = useState(num("walk", 0));
  const [insulin, setInsulin] = useState(0);
  const [sleep, setSleep] = useState<number | null>(preset?.get("sleep") ? num("sleep", 7) : null);
  const [ill, setIll] = useState(preset?.get("ill") === "1");
  const [res, setRes] = useState<WhatIfResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);

  const grouped = useMemo(() => {
    const seen = new Set<string>();
    return GROUPS.map(([tag, label]) => {
      const items = (foods ?? []).filter((f) => f.tags.includes(tag) && !seen.has(f.key));
      items.forEach((f) => seen.add(f.key));
      return { label, items };
    });
  }, [foods]);
  const sel = foods?.find((f) => f.key === food);

  const run = async () => {
    setBusy(true);
    setErr(null);
    try {
      const r = await post<WhatIfResult>(`/api/patients/${encodeURIComponent(id)}/whatif`, {
        clock,
        meal: food ? { food, portion, in_min: inMin } : null,
        walk: walk ? { minutes: walk, delay_min: inMin + 20 } : null,
        insulin: insulin ? { drug: "insulin_rapid", units: insulin, in_min: inMin } : null,
        sleep_hours: sleep,
        illness: ill,
        horizon: 240,
      });
      setRes(r);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  // deep link: #/patient/<id>?tab=whatif&food=...&walk=15&run=1[&clock=780] runs the scenario on open,
  // once the app clock has reached the linked time (the clock is set after metadata loads)
  const ran = useRef(false);
  useEffect(() => {
    if (ran.current || preset?.get("run") !== "1") return;
    const want = preset.get("clock");
    if (want != null && Number(want) !== clock) return;
    ran.current = true;
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clock]);

  const chart = (t: Tokens): EChartsOption => ({
    ...baseOption(t),
    grid: { left: 44, right: 70, top: 12, bottom: 28 },
    xAxis: timeAxis(t),
    yAxis: valueAxis(t, { min: 40, max: Math.max(300, ...(res ? [...res.baseline, ...res.scenario] : [])) + 10 }),
    series: res ? [
      { name: "Without change", type: "line", data: res.times.map((x, i) => [x, res.baseline[i]]), showSymbol: false,
        lineStyle: { width: 2, color: t.deemph }, itemStyle: { color: t.deemph },
        markArea: { silent: true, itemStyle: { color: t.rangeWash }, data: [[{ yAxis: 70 }, { yAxis: 180 }]] },
        endLabel: { show: true, color: t.ink2, fontSize: 11, formatter: "Baseline" } },
      { name: "Scenario", type: "line", data: res.times.map((x, i) => [x, res.scenario[i]]), showSymbol: false,
        lineStyle: { width: 2, color: t.s1 }, itemStyle: { color: t.s1 },
        endLabel: { show: true, color: t.ink, fontSize: 11, formatter: "Scenario" },
        markPoint: { symbol: "circle", symbolSize: 8, itemStyle: { color: t.s1, borderColor: t.surface, borderWidth: 2 },
                     label: { show: true, position: "top", color: t.ink, fontSize: 11, formatter: (p: { value?: unknown }) => `peak ${p.value}` },
                     data: [{ type: "max", name: "peak" }] } },
    ] : [],
  });

  const ranked = useMemo(() => {
    const all = insights?.meal_ranking ?? [];
    if (showAll || all.length <= 13) return all;
    const pick = new Set([...all.slice(0, 6), ...all.slice(-6)].map((m) => m.food));
    if (food) pick.add(food);
    return all.filter((m) => pick.has(m.food));
  }, [insights, showAll, food]);

  const rankOption = (t: Tokens): EChartsOption => {
    const r = ranked;
    return {
      ...baseOption(t),
      tooltip: { ...(baseOption(t).tooltip as object), trigger: "item",
        formatter: (p: unknown) => {
          const m = r[(p as { dataIndex: number }).dataIndex];
          return `<div style="font-size:12px"><b>+${m.rise} mg/dL</b> ${m.name}<br/><span style="color:${t.muted}">${m.carbs} g carbs · GI ${m.gi} · peak at ${m.peak_min} min</span></div>`;
        } },
      grid: { left: 190, right: 48, top: 4, bottom: 4 },
      xAxis: { type: "value", show: false },
      yAxis: { type: "category", data: r.map((m) => m.name), axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: t.ink2, fontSize: 11 } },
      series: [{
        type: "bar", barMaxWidth: 12, barCategoryGap: "35%",
        data: r.map((m) => ({ value: m.rise, itemStyle: { color: m.food === food ? t.s1 : t.deemph, borderRadius: [0, 4, 4, 0] } })),
        label: { show: true, position: "right", color: t.ink2, fontSize: 11, formatter: "+{c}" },
      }],
    };
  };

  return (
    <div className="grid gap-4 xl:grid-cols-[320px_1fr]">
      <div className="space-y-3 text-[13px]">
        <label className="block">
          <span className="mb-1 block font-medium">Meal</span>
          <select className="field w-full" value={food} onChange={(e) => setFood(e.target.value)}>
            <option value="">No meal</option>
            {grouped.map((g) => (
              <optgroup key={g.label} label={g.label}>
                {g.items.map((f) => <option key={f.key} value={f.key}>{f.name}</option>)}
              </optgroup>
            ))}
          </select>
          {sel && <span className="mt-1 block text-[12px] muted">{sel.serving} · {Math.round(sel.carbs * portion)} g carbs · GI {sel.gi} · {Math.round(sel.kcal * portion)} kcal</span>}
        </label>
        <label className="block">
          <span className="mb-1 flex justify-between font-medium"><span>Portion</span><span className="tabular">{portion.toFixed(2)}x</span></span>
          <input type="range" min={0.5} max={2} step={0.25} value={portion} onChange={(e) => setPortion(Number(e.target.value))} className="w-full" />
        </label>
        <label className="block">
          <span className="mb-1 block font-medium">Eaten</span>
          <select className="field w-full" value={inMin} onChange={(e) => setInMin(Number(e.target.value))}>
            <option value={0}>Now</option><option value={30}>In 30 min</option><option value={60}>In 1 hour</option>
          </select>
        </label>
        <label className="block">
          <span className="mb-1 block font-medium">Walk after the meal</span>
          <select className="field w-full" value={walk} onChange={(e) => setWalk(Number(e.target.value))}>
            <option value={0}>No walk</option><option value={10}>10 min</option><option value={15}>15 min (shatapavali)</option><option value={30}>30 min brisk</option>
          </select>
        </label>
        <label className="block">
          <span className="mb-1 flex justify-between font-medium"><span>Extra rapid insulin</span><span className="tabular">{insulin} IU</span></span>
          <input type="range" min={0} max={10} step={1} value={insulin} onChange={(e) => setInsulin(Number(e.target.value))} className="w-full" />
          <span className="block text-[12px] muted">Educational simulation only, not a dosing calculator.</span>
        </label>
        <label className="block">
          <span className="mb-1 flex justify-between font-medium"><span>Sleep last night</span><span className="tabular">{sleep == null ? "as recorded" : `${sleep} h`}</span></span>
          <input type="range" min={3} max={9} step={0.5} value={sleep ?? 7} onChange={(e) => setSleep(Number(e.target.value))} className="w-full" />
        </label>
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={ill} onChange={(e) => setIll(e.target.checked)} />
          <span>Acute illness (insulin sensitivity -30%)</span>
        </label>
        <button className="btn btn-primary w-full justify-center" onClick={run} disabled={busy}>
          {busy ? <FlaskConical size={14} /> : <Play size={14} />} {busy ? "Simulating…" : "Run on this patient's twin"}
        </button>
        {err && <p style={{ color: "var(--critical)" }}>{err}</p>}
      </div>
      <div className="space-y-4">
        <div>
          <div className="mb-2 flex flex-wrap items-center gap-3">
            <h3 className="text-[13px] font-medium">Next 4 hours from now</h3>
            <Legend items={[{ label: "Scenario", color: "var(--series-1)" }, { label: "Without change", color: "var(--deemph)" }, { label: "Target 70–180", color: "var(--range-wash)", kind: "area" }]} />
          </div>
          {res ? (
            <>
              <Chart option={chart} height={260} ariaLabel="What-if simulation" deps={[res]} />
              <div className="mt-2 grid grid-cols-1 gap-2 text-[13px] sm:grid-cols-3">
                <div className="card px-3 py-2"><div className="muted text-[12px]">Peak glucose</div><div className="font-semibold">{res.baseline_peak} → {res.scenario_peak} mg/dL</div></div>
                <div className="card px-3 py-2"><div className="muted text-[12px]">Minutes above 180</div><div className="font-semibold">{res.baseline_above_180_min} → {res.scenario_above_180_min}</div></div>
                <div className="card px-3 py-2"><div className="muted text-[12px]">Minutes below 70</div><div className="font-semibold">{res.baseline_below_70_min} → {res.scenario_below_70_min}</div></div>
              </div>
            </>
          ) : (
            <div className="grid h-[260px] place-items-center rounded-lg border text-[13px] ink-2" style={{ borderColor: "var(--border)" }}>
              Choose a scenario and run it on the personalised twin.
            </div>
          )}
        </div>
        {insights && (
          <div>
            <h3 className="mb-1 text-[13px] font-medium">This patient's predicted rise for common Indian meals</h3>
            <p className="mb-2 text-[12px] muted">Simulated on the personalised twin from fasting; selected meal highlighted. {insights.walk.reduction > 0 && `A 15-min walk after ${insights.walk.meal.toLowerCase()} lowers the peak by ~${insights.walk.reduction} mg/dL.`}</p>
            <Chart option={rankOption} height={Math.max(200, ranked.length * 20)} ariaLabel="Personal meal response ranking" deps={[ranked, food]} />
            {insights.meal_ranking.length > 13 && (
              <button className="btn mt-2" onClick={() => setShowAll((v) => !v)}>
                {showAll ? "Show gentlest and largest only" : `Show all ${insights.meal_ranking.length} meals`}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
