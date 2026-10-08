import type { EChartsOption } from "echarts";
import { FlaskConical, Play, ShieldCheck, TriangleAlert } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { TherapyDose, TherapyPlan, TherapyResult, post } from "../api";
import { useData } from "../hooks";
import Chart, { Tokens, baseOption, timeAxis, valueAxis } from "./Chart";
import { Legend, Status } from "./ui";

const PATTERNS: [string, string][] = [
  ["yesterday", "As yesterday (meals and activity replayed)"],
  ["skip_lunch", "Skips lunch"],
  ["late_dinner", "Dinner 2 hours later"],
  ["big_dinner", "Festival dinner (+50% carbohydrate)"],
];
const SHIFTS = [-60, -30, 0, 30, 60];

/** "Tomorrow looks like yesterday": compare the usual regimen with an adjusted one over 24 h on this patient's twin. */
export default function TherapyPanel({ id, clock, preset }: { id: string; clock: number; preset?: URLSearchParams }) {
  const enc = encodeURIComponent(id);
  const { data: plan } = useData<TherapyPlan>(`/api/patients/${enc}/therapy?clock=${clock}`, [clock]);
  const [doses, setDoses] = useState<(TherapyDose & { shift: number })[]>([]);
  const [dpp4, setDpp4] = useState(false);
  const [sglt2, setSglt2] = useState(false);
  const [pattern, setPattern] = useState(preset?.get("pattern") ?? "yesterday");
  const [res, setRes] = useState<TherapyResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  type Row = TherapyDose & { shift: number };
  const seq = useRef(0);
  const simulate = async (ds: Row[], d4: boolean, s2: boolean, pat: string) => {
    const mine = ++seq.current;
    setBusy(true);
    setErr(null);
    try {
      const r = await post<TherapyResult>(`/api/patients/${enc}/therapy`, {
        clock, dpp4: d4, sglt2: s2, pattern: pat,
        doses: ds.map((d) => ({ drug: d.drug, amount: d.amount, offset_min: Math.max(0, Math.min(1439, d.offset_min + d.shift)) })),
      });
      if (mine === seq.current) setRes(r);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      if (mine === seq.current) setBusy(false);
    }
  };
  const run = (pat = pattern) => simulate(doses, dpp4, sglt2, pat);

  // start from the usual plan and simulate it at once, so the usual day and the dose-response table
  // appear immediately; ?tab=therapy&scale=0.8 pre-scales every dose (for deep links)
  useEffect(() => {
    if (!plan) return;
    const k = Number(preset?.get("scale") ?? 1) || 1;
    const init = plan.doses.map((d) => ({ ...d, amount: Math.round(d.amount * k * 2) / 2, shift: 0 }));
    // ?dpp4=1 / ?sglt2=0 pre-set the drug-class toggles (deep links for demos)
    const flag = (k: string, d: boolean) => (preset?.get(k) === "1" ? true : preset?.get(k) === "0" ? false : d);
    const d4 = flag("dpp4", plan.dpp4), s2 = flag("sglt2", plan.sglt2);
    setDoses(init);
    setDpp4(d4);
    setSglt2(s2);
    setRes(null);
    simulate(init, d4, s2, pattern);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plan]);

  const set = (i: number, patch: Partial<TherapyDose & { shift: number }>) => setDoses((ds) => ds.map((d, j) => (j === i ? { ...d, ...patch } : d)));
  const changed = plan != null && (dpp4 !== plan.dpp4 || sglt2 !== plan.sglt2 || doses.some((d, i) => d.shift !== 0 || d.amount !== plan.doses[i]?.amount));

  const chart = (t: Tokens): EChartsOption => ({
    ...baseOption(t),
    grid: { left: 44, right: 16, top: 12, bottom: 28 },
    xAxis: timeAxis(t),
    yAxis: valueAxis(t, { min: 40, max: Math.max(300, ...(res ? [...res.usual, ...res.scenario] : [])) + 10 }),
    series: res ? [
      { name: "Usual regimen", type: "line", data: res.times.map((x, i) => [x, res.usual[i]]), showSymbol: false,
        lineStyle: { width: 2, color: t.deemph }, itemStyle: { color: t.deemph },
        markArea: { silent: true, itemStyle: { color: t.rangeWash }, data: [[{ yAxis: 70 }, { yAxis: 180 }]] },
        endLabel: { show: false } },
      ...(changed ? [{ name: "Adjusted", type: "line" as const, data: res.times.map((x, i) => [x, res.scenario[i]]), showSymbol: false,
        lineStyle: { width: 2, color: t.s1 }, itemStyle: { color: t.s1 } }] : []),
    ] : [],
  });

  const u = res?.usual_stats, s = res?.scenario_stats;
  const cmp = (a: number | undefined, b: number | undefined, unit = "") => (a == null ? "–" : changed && b != null ? `${a}${unit} → ${b}${unit}` : `${a}${unit}`);

  return (
    <div className="grid gap-4 xl:grid-cols-[340px_1fr]">
      <div className="space-y-3 text-[13px]">
        <label className="block">
          <span className="mb-1 block font-medium">Day to simulate</span>
          <select className="field w-full" value={pattern} onChange={(e) => { setPattern(e.target.value); run(e.target.value); }}>
            {PATTERNS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
          </select>
          {plan && <span className="mt-1 block text-[12px] muted">{plan.meals_replayed} meals and the last 24 h of activity, from now onwards.</span>}
        </label>
        <div>
          <span className="mb-1 block font-medium">Glucose-lowering doses (last 24 h, same times)</span>
          {plan && doses.length === 0 && <p className="text-[12px] muted">No insulin or sulfonylurea doses were logged in the last 24 h.</p>}
          <div className="space-y-2">
            {doses.map((d, i) => (
              <div key={`${d.drug}-${d.offset_min}`} className="rounded-md border px-2 py-2" style={{ borderColor: "var(--border)" }}>
                <div className="mb-1 flex justify-between gap-2"><span className="font-medium">{d.display}</span><span className="muted tabular">{d.time}</span></div>
                <div className="flex items-center gap-2">
                  <input type="number" min={0} step={d.unit === "IU" ? 1 : 10} value={d.amount} aria-label={`${d.display} dose`}
                    onChange={(e) => set(i, { amount: Math.max(0, Number(e.target.value)) })} className="field w-20 tabular" />
                  <span className="muted">{d.unit}</span>
                  <select className="field ml-auto" value={d.shift} aria-label={`${d.display} timing`} onChange={(e) => set(i, { shift: Number(e.target.value) })}>
                    {SHIFTS.map((m) => <option key={m} value={m}>{m === 0 ? "same time" : `${m > 0 ? "+" : ""}${m} min`}</option>)}
                  </select>
                </div>
                {plan && d.amount !== plan.doses[i]?.amount && <span className="text-[12px] muted">usual {plan.doses[i]?.amount} {d.unit}</span>}
              </div>
            ))}
          </div>
          {plan?.reduced_su_clearance && doses.some((d) => d.kind === "sulfonylurea") && (
            <p className="mt-1 text-[12px]" style={{ color: "var(--serious)" }}>Reduced kidney function or age ≥ 70: sulfonylurea exposure is prolonged in this twin.</p>
          )}
        </div>
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={dpp4} onChange={(e) => setDpp4(e.target.checked)} />
          <span>DPP-4 inhibitor (incretin enhancement)</span>
        </label>
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={sglt2} onChange={(e) => setSglt2(e.target.checked)} disabled={plan?.egfr != null && plan.egfr < 25 && !plan.sglt2} />
          <span>SGLT2 inhibitor (renal threshold lowered){plan?.egfr != null && plan.egfr < 25 && !plan.sglt2 ? " · not with eGFR < 25" : ""}</span>
        </label>
        <button className="btn btn-primary w-full justify-center" onClick={() => run()} disabled={busy || !plan}>
          {busy ? <FlaskConical size={14} /> : <Play size={14} />} {busy ? "Simulating…" : "Simulate 24 h on this patient's twin"}
        </button>
        <p className="text-[12px] muted">Educational decision support for discussion with the care team, not a dosing calculator. Metformin acts over weeks on the fasting set-point and is not simulated here.</p>
        {err && <p style={{ color: "var(--critical)" }}>{err}</p>}
      </div>
      <div className="space-y-4">
        {res?.fidelity?.rating === "poor" && (
          <div className="flex gap-2 rounded-md border px-3 py-2 text-[13px]" style={{ borderColor: "var(--serious)" }} role="alert">
            <TriangleAlert size={16} className="mt-0.5 shrink-0" style={{ color: "var(--serious)" }} aria-hidden />
            <span>This twin does not yet reproduce the patient's days well (replay error {res.fidelity.mae} mg/dL; see the trust check below). Treat this 24-hour simulation as unreliable until the twin is refitted on more data.</span>
          </div>
        )}
        <div>
          <div className="mb-2 flex flex-wrap items-center gap-3">
            <h3 className="text-[13px] font-medium">Next 24 hours on the twin</h3>
            <Legend items={[...(changed ? [{ label: "Adjusted", color: "var(--series-1)" }] : []), { label: "Usual regimen", color: "var(--deemph)" }, { label: "Target 70–180", color: "var(--range-wash)", kind: "area" as const }]} />
          </div>
          {res ? <Chart option={chart} height={260} ariaLabel="24-hour therapy simulation" deps={[res, changed]} />
            : <div className="grid h-[260px] place-items-center rounded-lg border text-[13px] ink-2" style={{ borderColor: "var(--border)" }}>Loading the usual regimen…</div>}
          {u && (
            <div className="mt-2 grid grid-cols-2 gap-2 text-[13px] lg:grid-cols-4">
              <div className="card px-3 py-2"><div className="muted text-[12px]">Mean glucose</div><div className="font-semibold tabular">{cmp(u.mean, s?.mean)} mg/dL</div></div>
              <div className="card px-3 py-2"><div className="muted text-[12px]">Time in range 70–180</div><div className="font-semibold tabular">{cmp(u.tir_pct, s?.tir_pct, "%")}</div></div>
              <div className="card px-3 py-2"><div className="muted text-[12px]">Time above 180</div><div className="font-semibold tabular">{cmp(u.above_180_pct, s?.above_180_pct, "%")}</div></div>
              <div className="card px-3 py-2"><div className="muted text-[12px]">Lowest glucose</div><div className="font-semibold tabular">{cmp(u.min, s?.min)} mg/dL</div><div className="text-[12px] muted">at {changed && s ? s.min_time : u.min_time}</div></div>
            </div>
          )}
        </div>
        {res && res.dose_response.rows.length > 0 && (
          <div>
            <h3 className="mb-1 text-[13px] font-medium">Dose–response on this twin: scaling the {res.dose_response.what}</h3>
            <div className="overflow-x-auto">
              <table className="data min-w-[520px]">
                <thead><tr><th>Dose</th><th className="num">Mean</th><th className="num">Time in range</th><th className="num">Above 180</th><th className="num">Lowest</th></tr></thead>
                <tbody>
                  {res.dose_response.rows.map((r) => (
                    <tr key={r.scale} className={r.scale === 1 ? "font-semibold" : ""}>
                      <td>{Math.round(r.scale * 100)}%{r.scale === 1 ? " (usual)" : ""}</td>
                      <td className="num">{r.mean}</td><td className="num">{r.tir_pct}%</td><td className="num">{r.above_180_pct}%</td><td className="num">{r.min}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        {res?.fidelity && (
          <div className="card flex gap-3 px-3 py-3 text-[13px]">
            <ShieldCheck size={18} className="mt-0.5 shrink-0" style={{ color: "var(--series-1)" }} aria-hidden />
            <div>
              <div className="flex flex-wrap items-center gap-2 font-medium">Trust check: the twin replaying the last 24 h against the CGM
                <Status level={res.fidelity.rating === "good" ? "good" : res.fidelity.rating === "fair" ? "warning" : "serious"}>{res.fidelity.rating === "good" ? "Good agreement" : res.fidelity.rating === "fair" ? "Fair agreement" : "Poor agreement"}</Status>
              </div>
              <div className="ink-2">Mean {res.fidelity.mean_twin} vs {res.fidelity.mean_cgm} mg/dL · time in range {res.fidelity.tir_twin}% vs {res.fidelity.tir_cgm}% · mean absolute error {res.fidelity.mae} mg/dL.</div>
              <div className="mt-1 text-[12px] muted">
                The 24-hour replay tracks average glucose and time in range{res.fidelity.below_70_cgm_min > 0 ? `, but not short lows (${res.fidelity.below_70_cgm_min} min below 70 on CGM vs ${res.fidelity.below_70_twin_min} min on the twin)` : ""}. Hypoglycaemia warnings come from the 2-hour forecast, which is validated for them.
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
