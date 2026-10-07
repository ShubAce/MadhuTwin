import { AlertTriangle, ArrowDown, ArrowDownRight, ArrowRight, ArrowUp, ArrowUpRight, CheckCircle2, OctagonAlert } from "lucide-react";
import type { ReactNode } from "react";

export type Level = "good" | "warning" | "serious" | "critical";

export const LEVEL_COLOR: Record<Level, string> = {
  good: "var(--good)",
  warning: "var(--warning)",
  serious: "var(--serious)",
  critical: "var(--critical)",
};

/** Status always ships with an icon + label, never colour alone. */
export function Status({ level, children }: { level: Level; children: ReactNode }) {
  const Icon = level === "good" ? CheckCircle2 : level === "critical" ? OctagonAlert : AlertTriangle;
  return (
    <span className="inline-flex items-center gap-1 text-[12px] font-medium" style={{ color: "var(--ink)" }}>
      <Icon size={14} color={LEVEL_COLOR[level]} aria-hidden />
      {children}
    </span>
  );
}

export function riskLevel(p: number | null | undefined, kind: "spike" | "hypo"): Level {
  if (p == null) return "good";
  const [w, s, c] = kind === "hypo" ? [0.15, 0.3, 0.5] : [0.3, 0.5, 0.75];
  return p >= c ? "critical" : p >= s ? "serious" : p >= w ? "warning" : "good";
}

export function glucoseLevel(g: number | null | undefined): Level {
  if (g == null) return "good";
  if (g < 70) return "critical";
  if (g > 250) return "serious";
  if (g > 180) return "warning";
  return "good";
}

export function RiskMeter({ p, kind }: { p: number | null | undefined; kind: "spike" | "hypo" }) {
  const level = riskLevel(p, kind);
  return (
    <div className="flex items-center gap-2">
      <div className="meter w-16" aria-hidden>
        <span style={{ width: `${Math.round((p ?? 0) * 100)}%`, background: LEVEL_COLOR[level] }} />
      </div>
      <span className="tabular text-[12px]">{p == null ? "–" : `${Math.round(p * 100)}%`}</span>
    </div>
  );
}

export function TrendArrow({ rate }: { rate: number | null | undefined }) {
  if (rate == null) return null;
  const r = rate;
  const Icon = r > 2 ? ArrowUp : r > 1 ? ArrowUpRight : r < -2 ? ArrowDown : r < -1 ? ArrowDownRight : ArrowRight;
  return (
    <span className="inline-flex items-center muted" title={`${r.toFixed(1)} mg/dL per min`}>
      <Icon size={14} aria-label={`trend ${r.toFixed(1)} mg/dL/min`} />
    </span>
  );
}

export function Sparkline({ values, width = 120, height = 28 }: { values: (number | null)[]; width?: number; height?: number }) {
  const pts = values.map((v, i) => [i, v] as const).filter(([, v]) => v != null) as [number, number][];
  if (pts.length < 2) return <span className="muted text-[12px]">–</span>;
  const lo = 40, hi = 300;
  const x = (i: number) => (i / (values.length - 1)) * (width - 4) + 2;
  const y = (v: number) => height - 2 - ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * (height - 4);
  const d = pts.map(([i, v], k) => `${k ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const last = pts[pts.length - 1];
  return (
    <svg width={width} height={height} aria-hidden>
      <rect x={0} y={y(180)} width={width} height={y(70) - y(180)} fill="var(--range-wash)" />
      <path d={d} fill="none" stroke="var(--deemph)" strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={x(last[0])} cy={y(last[1])} r={3} fill="var(--series-1)" stroke="var(--surface)" strokeWidth={1.5} />
    </svg>
  );
}

export function Stat({ label, value, sub, level }: { label: string; value: ReactNode; sub?: ReactNode; level?: Level }) {
  return (
    <div className="card px-4 py-3">
      <div className="text-[12px] muted">{label}</div>
      <div className="mt-0.5 text-[22px] font-semibold leading-tight">{value}</div>
      {(sub || level) && <div className="mt-1 text-[12px] ink-2">{level ? <Status level={level}>{sub}</Status> : sub}</div>}
    </div>
  );
}

export function Section({ title, right, children, className = "" }: { title: ReactNode; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`card p-4 ${className}`}>
      <div className="mb-3 flex items-center gap-2">
        <h2 className="text-[14px] font-semibold">{title}</h2>
        <div className="ml-auto">{right}</div>
      </div>
      {children}
    </section>
  );
}

export function Legend({ items }: { items: { label: string; color: string; kind?: "line" | "area" | "box" }[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-[12px] ink-2">
      {items.map((it) => (
        <span key={it.label} className="inline-flex items-center gap-1.5">
          {it.kind === "area" ? (
            <span className="inline-block h-2.5 w-4 rounded-sm" style={{ background: it.color }} />
          ) : it.kind === "box" ? (
            <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: it.color }} />
          ) : (
            <span className="inline-block h-[2px] w-4 rounded" style={{ background: it.color }} />
          )}
          {it.label}
        </span>
      ))}
    </div>
  );
}
