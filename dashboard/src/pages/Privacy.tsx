import { Section } from "../components/ui";
import { useData } from "../hooks";

interface AuditRow {
  ts: string;
  user: string;
  purpose: string;
  method: string;
  resource: string;
  patient: string | null;
  status: number;
  ms: number;
}

const PRINCIPLES: [string, string][] = [
  ["Lawful data only", "Synthetic India-calibrated patients plus de-identified open datasets (CGMacros, CC BY-NC-SA 4.0; ShanghaiT2DM, CC BY 4.0). Raw third-party data is never redistributed."],
  ["Consent and purpose limitation", "Each virtual patient carries a consent artefact (purpose: treatment, care coordination; expiry). In deployment this maps to ABDM's consent manager flow before any EHR pull."],
  ["Data minimisation", "The twin needs CGM, activity, sleep and a short EHR summary: no names or addresses are required for prediction. Real-world records are shown by research ID only."],
  ["Accountability", "Every API access is logged with user, purpose, resource and time (below). Logs are append-only in deployment."],
  ["Security and storage", "Designed for on-premise or in-country cloud hosting; FHIR R4 interfaces; no patient data leaves the deployment for model inference (the optional LLM assistant is off unless configured)."],
  ["Clinical safety", "Decision support for clinicians, not autonomous treatment. Intended regulatory route: Software as a Medical Device under CDSCO Medical Device Rules 2017 (likely Class B/C), with prospective validation."],
];

export default function Privacy() {
  const { data } = useData<AuditRow[]>("/api/audit?limit=200", [Date.now() >> 14]);
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-[20px] font-semibold">Privacy, consent and audit</h1>
        <p className="text-[13px] ink-2">How MadhuTwin aligns with India's Digital Personal Data Protection Act 2023 and the ABDM health data management policy.</p>
      </div>
      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
        {PRINCIPLES.map(([t, d]) => (
          <div key={t} className="card p-4">
            <div className="text-[14px] font-semibold">{t}</div>
            <p className="mt-1 text-[13px] ink-2">{d}</p>
          </div>
        ))}
      </div>
      <Section title="Access audit trail (this session)">
        <div className="overflow-x-auto">
          <table className="data min-w-[720px]">
            <thead><tr><th>Time</th><th>User</th><th>Purpose</th><th>Action</th><th>Patient</th><th className="num">Status</th><th className="num">Latency</th></tr></thead>
            <tbody>
              {(data ?? []).map((r, i) => (
                <tr key={i}>
                  <td className="tabular">{r.ts.replace("T", " ")}</td>
                  <td>{r.user}</td>
                  <td>{r.purpose}</td>
                  <td className="ink-2">{r.method} {r.resource.replace(/^\/api\/patients\/[^/]+/, "") || "/"}</td>
                  <td>{r.patient}</td>
                  <td className="num">{r.status}</td>
                  <td className="num">{r.ms} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}
