import type { EChartsOption } from "echarts";
import { ArrowLeft, BadgeCheck, Lightbulb, Smartphone } from "lucide-react";
import { useState } from "react";
import { go } from "../App";
import { Agp, InsightsResponse, PatientDetail, State } from "../api";
import AgpPanel from "../components/AgpPanel";
import AskPanel from "../components/AskPanel";
import Chart, { Tokens, baseOption, valueAxis } from "../components/Chart";
import FhirPanel from "../components/FhirPanel";
import GlucoseChart from "../components/GlucoseChart";
import LabsPanel from "../components/LabsPanel";
import VirtualPatient from "../components/VirtualPatient";
import WearablesPanel from "../components/WearablesPanel";
import WhatIfPanel from "../components/WhatIfPanel";
import { RiskMeter, Section, Status, glucoseLevel, riskLevel } from "../components/ui";
import { fmtTime, useData } from "../hooks";

const TABS = ["What-if simulator", "AGP report", "Wearables", "Clinical record", "Ask the twin", "Interoperability (FHIR)"] as const;
const TAB_KEYS: Record<string, (typeof TABS)[number]> = { whatif: "What-if simulator", agp: "AGP report", wearables: "Wearables", record: "Clinical record", ask: "Ask the twin", fhir: "Interoperability (FHIR)" };
type Tab = (typeof TABS)[number];

function RiskCard({ state }: { state: State }) {
  const fc = state.forecast;
  if (!fc) return <p className="text-[13px] muted">No forecast available yet at this time.</p>;
  const peak = Math.max(...(fc.q50.filter((v) => v != null) as number[]));
  const low = Math.min(...(fc.q10.filter((v) => v != null) as number[]));
  const end = fc.q50[fc.q50.length - 1];
  const nowG = [...state.series.cgm].reverse().find((v) => v != null) ?? null;
  const alreadyHigh = nowG != null && nowG > 180;
  const spikeLvl = alreadyHigh ? (nowG > 250 ? "serious" : "warning") : riskLevel(fc.spike, "spike");
  const hypoLvl = riskLevel(fc.hypo, "hypo");
  const lead = hypoLvl !== "good" ? "hypo" : spikeLvl !== "good" ? "spike" : null;
  const why = lead === "hypo" ? fc.why_hypo : fc.why_spike;
  return (
    <div className="space-y-3 text-[13px]">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="mb-1 muted text-[12px]">High &gt;180 for 15+ min</div>
          {alreadyHigh ? <Status level={spikeLvl}>Above range now</Status> : <RiskMeter p={fc.spike} kind="spike" />}
          <div className="mt-1 text-[12px] ink-2">Median peak {Math.round(peak)}{end != null ? `, ${Math.round(end)} in 2 h` : ""}</div>
        </div>
        <div>
          <div className="mb-1 muted text-[12px]">Low &lt;70 for 15+ min</div>
          <RiskMeter p={fc.hypo} kind="hypo" />
          <div className="mt-1 text-[12px] ink-2">10th percentile {Math.round(low)} mg/dL</div>
        </div>
      </div>
      {lead ? (
        <div className="rounded-lg p-3" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
          <Status level={lead === "hypo" ? hypoLvl : spikeLvl}>
            {lead === "hypo" ? "Hypoglycaemia predicted within 2 h" : alreadyHigh ? `Hyperglycaemia: ${Math.round(nowG!)} mg/dL now, forecast peak ${Math.round(peak)}` : "Hyperglycaemic excursion predicted within 2 h"}
          </Status>
          {why && why.length > 0 && (
            <ul className="mt-2 list-disc space-y-0.5 pl-5 text-[12px] ink-2">
              {why.map((w) => <li key={w}>{w}</li>)}
            </ul>
          )}
        </div>
      ) : (
        <Status level="good">No excursion expected in the next 2 hours</Status>
      )}
      <p className="text-[11px] muted">Forecast issued {fmtTime(fc.anchor_time)}. Reasons: SHAP attributions from the event model.</p>
    </div>
  );
}

function TwinEngineCard({ p, state }: { p: PatientDetail; state: State | null }) {
  const si = p.twin.si_daily.map((v, i) => [i + 1, v] as [number, number | null]);
  const option = (t: Tokens): EChartsOption => ({
    ...baseOption(t),
    grid: { left: 36, right: 10, top: 10, bottom: 22 },
    xAxis: { type: "category", data: si.map((d) => `D${d[0]}`), axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false }, axisLabel: { color: t.muted, fontSize: 10 } },
    yAxis: valueAxis(t, { min: 0.3, max: 1.6, splitNumber: 3 }),
    series: [{
      name: "Insulin sensitivity vs baseline", type: "line", data: si.map((d) => d[1]), symbol: "circle", symbolSize: 7,
      lineStyle: { width: 2, color: t.s1 }, itemStyle: { color: t.s1, borderColor: t.surface, borderWidth: 2 },
      markLine: { silent: true, symbol: "none", lineStyle: { color: t.axis, type: "solid" }, label: { color: t.muted, fontSize: 10, formatter: "baseline" }, data: [{ yAxis: 1 }] },
    }],
  });
  const g = p.gate ?? [];
  return (
    <div className="space-y-3 text-[13px]">
      <div className="grid grid-cols-2 gap-2">
        <div className="card px-3 py-2">
          <div className="text-[12px] muted">Model · personalisation</div>
          <div className="font-semibold">{p.twin.fit_rmse_prior != null && p.twin.fit_rmse_personalised != null ? `${p.twin.fit_rmse_prior.toFixed(0)} → ${p.twin.fit_rmse_personalised.toFixed(0)} mg/dL` : "EHR prior"}</div>
          <div className="text-[11px] muted">Twin error before → after fitting to this patient</div>
        </div>
        <div className="card px-3 py-2">
          <div className="text-[12px] muted">Sync · insulin sensitivity today</div>
          <div className="font-semibold">{state?.si_today != null ? `${state.si_today.toFixed(2)}x baseline` : "–"}</div>
          <div className="text-[11px] muted">Unscented Kalman filter, every 5 min</div>
        </div>
      </div>
      <div>
        <div className="mb-1 text-[12px] ink-2">Daily insulin-sensitivity estimate (synced twin)</div>
        <Chart option={option} height={130} ariaLabel="Daily insulin sensitivity relative to baseline" deps={[p]} />
      </div>
      {g.length >= 24 && (
        <div className="text-[12px] ink-2">
          <span className="font-medium">Trust in physics</span> (TwinNet gate): 30 min {Number(g[5]).toFixed(2)} · 60 min {Number(g[11]).toFixed(2)} · 120 min {Number(g[23]).toFixed(2)}
          <div className="muted">The hybrid network learns how much to rely on the mechanistic twin at each horizon.</div>
        </div>
      )}
    </div>
  );
}

export default function PatientView({ id, clock, params }: { id: string; clock: number; params?: URLSearchParams }) {
  const [tab, setTab] = useState<Tab>(TAB_KEYS[params?.get("tab") ?? ""] ?? "What-if simulator");
  const enc = encodeURIComponent(id);
  const { data: p, error } = useData<PatientDetail>(`/api/patients/${enc}`);
  const { data: state, loading } = useData<State>(`/api/patients/${enc}/state?clock=${clock}`);
  const { data: agp } = useData<Agp>(tab === "AGP report" ? `/api/patients/${enc}/agp?clock=${clock}` : null, [tab]);
  const { data: ins } = useData<InsightsResponse>(`/api/patients/${enc}/insights?clock=${Math.floor(clock / 60) * 60}`);

  if (error) return <p style={{ color: "var(--critical)" }}>Could not load patient: {error}</p>;
  if (!p) return <p className="muted">Loading virtual patient…</p>;
  const d = p.display;
  const a1c = p.ehr.labs.find((l) => l.key === "hba1c_pct")?.value;
  const nowG = state ? [...state.series.cgm].reverse().find((v) => v != null) ?? null : null;
  const name = d.name.split(" ")[0];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        <button className="btn" onClick={() => go("/")}><ArrowLeft size={14} /> Panel</button>
        <button className="btn" onClick={() => go(`/companion/${enc}`)}><Smartphone size={14} /> Patient app view</button>
      </div>
      <div className="card flex flex-wrap items-start gap-x-8 gap-y-3 p-4">
        <div>
          <h1 className="text-[20px] font-semibold">{d.name}</h1>
          <p className="text-[13px] ink-2">
            {[d.age && `${Math.round(d.age)} years`, d.sex === "F" ? "Female" : d.sex === "M" ? "Male" : null, d.city, d.language && `prefers ${d.language}`].filter(Boolean).join(" · ")}
          </p>
          <p className="mt-1 text-[12px] muted">{d.label} · {p.story}</p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {p.ehr.conditions.slice(0, 5).map((c) => <span key={c.key} className="chip">{c.display}</span>)}
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-6 text-[13px]">
          <div><div className="text-[12px] muted">Glucose now</div><Status level={glucoseLevel(nowG)}>{nowG == null ? "–" : `${Math.round(nowG)} mg/dL`}</Status></div>
          <div><div className="text-[12px] muted">HbA1c</div><div className="font-semibold">{a1c != null ? `${a1c.toFixed(1)}%` : "–"}</div></div>
          <div><div className="text-[12px] muted">BMI</div><div className="font-semibold">{p.ehr.bmi != null ? p.ehr.bmi.toFixed(1) : "–"}</div></div>
          <div className="flex items-center gap-1 text-[12px] ink-2"><BadgeCheck size={14} color="var(--good)" /> Consent: {p.consent.purpose.join(", ")}</div>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_360px]">
        <Section title="Glucose · last 24 h and 2-hour forecast" right={state && <span className="text-[12px] muted">Now {fmtTime(state.now)}</span>}>
          <div style={{ opacity: loading && state ? 0.75 : 1 }}>{state && <GlucoseChart state={state} />}</div>
        </Section>
        <div className="space-y-4">
          <Section title="Predicted risk (next 2 h)">{state && <RiskCard state={state} />}</Section>
          <Section title={<span className="inline-flex items-center gap-1.5"><Lightbulb size={15} /> Twin insights</span>}>
            {ins ? (
              <ul className="space-y-2.5 text-[13px]">
                {ins.insights.map((i) => (
                  <li key={i.title}>
                    <div className="font-medium">{i.kind === "warning" ? <Status level="warning">{i.title}</Status> : i.title}</div>
                    <div className="text-[12px] ink-2">{i.text}</div>
                  </li>
                ))}
              </ul>
            ) : <p className="text-[13px] muted">Computing…</p>}
          </Section>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="Virtual patient · physiology learned by the twin"><VirtualPatient patient={p} state={state} /></Section>
        <Section title="Twin engine · model, sync, simulate"><TwinEngineCard p={p} state={state} /></Section>
      </div>

      <section className="card">
        <div className="flex gap-1 overflow-x-auto border-b px-2" role="tablist" style={{ borderColor: "var(--grid)" }}>
          {TABS.map((t) => (
            <button key={t} role="tab" className="tab" aria-selected={tab === t} onClick={() => setTab(t)}>{t}</button>
          ))}
        </div>
        <div className="p-4">
          {tab === "What-if simulator" && <WhatIfPanel id={id} clock={clock} insights={ins} preset={params} />}
          {tab === "AGP report" && (agp ? <AgpPanel agp={agp} /> : <p className="muted text-[13px]">Loading…</p>)}
          {tab === "Wearables" && state && <WearablesPanel state={state} />}
          {tab === "Clinical record" && <LabsPanel p={p} />}
          {tab === "Ask the twin" && <AskPanel id={id} clock={clock} name={name} />}
          {tab === "Interoperability (FHIR)" && <FhirPanel id={id} clock={clock} />}
        </div>
      </section>
    </div>
  );
}
