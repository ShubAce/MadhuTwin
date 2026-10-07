import { go } from "../App";
import { PatientSummary } from "../api";
import { RiskMeter, Sparkline, Stat, Status, TrendArrow, glucoseLevel, riskLevel } from "../components/ui";
import { fmtTime, useData } from "../hooks";

const SOURCE_LABEL: Record<string, string> = { synthetic: "Synthetic", cgmacros: "CGMacros", shanghai: "ShanghaiT2DM" };

export default function Panel({ clock }: { clock: number }) {
  const { data, error, loading } = useData<PatientSummary[]>(`/api/patients?clock=${clock}`);
  const rows = data ?? [];
  const alerts = rows.filter((r) => (r.glucose ?? 0) > 180 || (r.glucose ?? 999) < 70 || riskLevel(r.risk_spike, "spike") !== "good" || riskLevel(r.risk_hypo, "hypo") !== "good");
  const hypo = rows.filter((r) => riskLevel(r.risk_hypo, "hypo") !== "good");
  const tirs = rows.map((r) => r.tir_24h).filter((x): x is number => x != null);
  const avgTir = tirs.length ? tirs.reduce((a, b) => a + b, 0) / tirs.length : null;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-2">
        <div>
          <h1 className="text-[20px] font-semibold">Remote monitoring panel</h1>
          <p className="text-[13px] ink-2">Each patient is a live digital twin: EHR + wearable streams fused, re-synced every 5 minutes, ranked by predicted 2-hour risk.</p>
        </div>
      </div>
      {error && <p className="text-[13px]" style={{ color: "var(--critical)" }}>Could not reach the API: {error}</p>}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4" style={{ opacity: loading && data ? 0.7 : 1 }}>
        <Stat label="Patients monitored" value={rows.length || "–"} sub="Virtual patients with synced twins" />
        <Stat label="Need attention (2 h)" value={alerts.length} level={alerts.length ? "warning" : "good"} sub={alerts.length ? "Elevated predicted risk" : "No elevated risk"} />
        <Stat label="Predicted lows" value={hypo.length} level={hypo.length ? "critical" : "good"} sub={hypo.length ? "Hypoglycaemia risk" : "None predicted"} />
        <Stat label="Average time in range (24 h)" value={avgTir == null ? "–" : `${avgTir.toFixed(0)}%`} sub="Target >70%" />
      </div>
      <div className="card overflow-x-auto" style={{ opacity: loading && data ? 0.7 : 1 }}>
        <table className="data min-w-[980px]">
          <thead>
            <tr>
              <th>Patient</th>
              <th>Clinical story</th>
              <th className="num">Glucose</th>
              <th>Spike risk (2 h)</th>
              <th>Hypo risk (2 h)</th>
              <th className="num">TIR 24 h</th>
              <th>Last 6 h</th>
              <th>Last alert</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className="row-link" onClick={() => go(`/patient/${encodeURIComponent(r.id)}`)} tabIndex={0}
                  onKeyDown={(e) => e.key === "Enter" && go(`/patient/${encodeURIComponent(r.id)}`)}>
                <td>
                  <div className="font-medium">{r.display.name}</div>
                  <div className="text-[12px] muted">
                    {[r.display.age && `${Math.round(r.display.age)} y`, r.display.sex, r.display.city, SOURCE_LABEL[r.source]].filter(Boolean).join(" · ")}
                  </div>
                </td>
                <td className="max-w-[260px] text-[12px] ink-2">{r.story}</td>
                <td className="num">
                  <div className="inline-flex items-center gap-1">
                    <Status level={glucoseLevel(r.glucose)}>{r.glucose == null ? "–" : Math.round(r.glucose)}</Status>
                    <TrendArrow rate={r.trend} />
                  </div>
                </td>
                <td>
                  {r.glucose != null && r.glucose > 180 ? (
                    <Status level={r.glucose > 250 ? "serious" : "warning"}>Already high</Status>
                  ) : (
                    <RiskMeter p={r.risk_spike} kind="spike" />
                  )}
                </td>
                <td><RiskMeter p={r.risk_hypo} kind="hypo" /></td>
                <td className="num">{r.tir_24h == null ? "–" : `${r.tir_24h.toFixed(0)}%`}</td>
                <td><Sparkline values={r.sparkline} /></td>
                <td className="text-[12px]">
                  {r.last_alert ? (
                    <Status level={r.last_alert.kind === "hypo" ? "critical" : "warning"}>
                      {r.last_alert.kind === "hypo" ? "Low" : "High"} predicted · {r.last_alert.time ? fmtTime(r.last_alert.time) : ""}
                    </Status>
                  ) : (
                    <span className="muted">None</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[12px] muted">
        Ranking uses the TwinNet event heads (hypoglycaemia weighted 2.5x). Thresholds: spike alert at 50%, hypo alert at 30%. Status colours always carry an icon and label.
      </p>
    </div>
  );
}
