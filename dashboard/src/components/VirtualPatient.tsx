import { PatientDetail, State } from "../api";
import { LEVEL_COLOR, Level, Status } from "./ui";

interface Organ {
  key: string;
  name: string;
  value: string;
  detail: string;
  level: Level;
  x: number;
  y: number;
}

const band = (v: number, cuts: [number, number, number], higherIsBetter = true): Level => {
  const [a, b, c] = cuts;
  if (higherIsBetter) return v >= a ? "good" : v >= b ? "warning" : v >= c ? "serious" : "critical";
  return v <= a ? "good" : v <= b ? "warning" : v <= c ? "serious" : "critical";
};

function median(xs: number[]) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

export function organs(p: PatientDetail, s: State | null): Organ[] {
  const tp = p.twin.params;
  const out: Organ[] = [];
  if (tp.beta != null) {
    const pctB = Math.min((tp.beta / 0.13) * 100, 150);
    out.push({ key: "pancreas", name: "Pancreas · beta-cell response", value: `${pctB.toFixed(0)}% of healthy`,
      detail: "Glucose-stimulated insulin secretion, fitted from CGM", level: band(pctB, [70, 40, 20]), x: 104, y: 150 });
  }
  if (tp.SI != null) {
    const pctS = (tp.SI / 6.5e-4) * 100;
    const today = s?.si_today;
    out.push({ key: "muscle", name: "Muscle & liver · insulin sensitivity", value: `${pctS.toFixed(0)}% of healthy`,
      detail: today != null ? `Today ${today.toFixed(2)}x personal baseline (synced)` : "Personal baseline",
      level: today != null && today < 0.8 ? "critical" : band(pctS, [70, 40, 20]), x: 72, y: 250 });
  }
  if (tp.Gb != null) {
    out.push({ key: "liver", name: "Liver · fasting set-point", value: `${tp.Gb.toFixed(0)} mg/dL`,
      detail: tp.dawn != null ? `Dawn rise ~${(tp.dawn * 100).toFixed(0)}% (4–8 am)` : "", level: band(tp.Gb, [100, 125, 180], false), x: 76, y: 128 });
  }
  const tau = p.twin.tau_scale;
  if (tau != null) {
    out.push({ key: "gut", name: "Gut · carbohydrate absorption", value: tau < 0.85 ? "Fast" : tau > 1.2 ? "Slow" : "Typical",
      detail: `${(1 / tau).toFixed(2)}x average speed: ${tau < 0.85 ? "sharper post-meal peaks" : "typical peak timing"}`,
      level: tau < 0.75 ? "warning" : "good", x: 92, y: 182 });
  }
  if (s) {
    const asleep = s.series.sleep.map((v) => (v ?? 0) > 0);
    const hrv = median(s.series.hrv.filter((v, i): v is number => v != null && asleep[i]));
    const hr = median(s.series.hr.filter((v, i): v is number => v != null && asleep[i]));
    if (hrv != null || hr != null) {
      out.push({ key: "heart", name: "Heart · autonomic tone", value: hrv != null ? `HRV ${hrv.toFixed(0)} ms` : `${hr?.toFixed(0)} bpm`,
        detail: hr != null ? `Night heart rate ${hr.toFixed(0)} bpm` : "", level: hrv == null ? "good" : band(hrv, [30, 20, 12]), x: 92, y: 110 });
    }
  }
  const egfr = p.ehr.labs.find((l) => l.key === "egfr")?.value;
  if (egfr != null) {
    out.push({ key: "kidney", name: "Kidneys · eGFR", value: `${egfr.toFixed(0)} mL/min`, detail: "CKD-EPI 2021 (EHR)",
      level: band(egfr, [90, 60, 30]), x: 110, y: 168 });
  }
  return out;
}

export default function VirtualPatient({ patient, state }: { patient: PatientDetail; state: State | null }) {
  const list = organs(patient, state);
  return (
    <div className="flex gap-4">
      <svg viewBox="0 0 184 380" className="h-[300px] w-[140px] shrink-0" aria-hidden>
        <g fill="var(--surface-2)" stroke="var(--axis)" strokeWidth={1.5}>
          <circle cx={92} cy={38} r={24} />
          <rect x={56} y={68} width={72} height={140} rx={30} />
          <rect x={30} y={76} width={22} height={124} rx={11} />
          <rect x={132} y={76} width={22} height={124} rx={11} />
          <rect x={58} y={196} width={30} height={170} rx={14} />
          <rect x={96} y={196} width={30} height={170} rx={14} />
        </g>
        {list.map((o) => (
          <g key={o.key}>
            <circle cx={o.x} cy={o.y} r={9} fill={LEVEL_COLOR[o.level]} stroke="var(--surface)" strokeWidth={2} />
          </g>
        ))}
      </svg>
      <ul className="flex-1 space-y-2.5">
        {list.map((o) => (
          <li key={o.key} className="text-[13px]">
            <div className="flex items-center justify-between gap-2">
              <span className="ink-2">{o.name}</span>
              <Status level={o.level}>{o.value}</Status>
            </div>
            {o.detail && <div className="text-[12px] muted">{o.detail}</div>}
          </li>
        ))}
      </ul>
    </div>
  );
}
