import { Download } from "lucide-react";
import { useMemo, useState } from "react";
import { useData } from "../hooks";

interface Bundle {
  resourceType: string;
  entry: { fullUrl: string; resource: { resourceType: string; id: string; code?: { coding?: { code: string; display?: string }[]; text?: string }; prediction?: { probabilityDecimal: number }[] } }[];
}

export default function FhirPanel({ id, clock }: { id: string; clock: number }) {
  const { data } = useData<Bundle>(`/api/patients/${encodeURIComponent(id)}/fhir?clock=${clock}`);
  const [sel, setSel] = useState<number>(-1);
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    data?.entry.forEach((e) => (c[e.resource.resourceType] = (c[e.resource.resourceType] ?? 0) + 1));
    return c;
  }, [data]);
  if (!data) return <p className="text-[13px] muted">Loading FHIR bundle…</p>;
  const ra = data.entry.findIndex((e) => e.resource.resourceType === "RiskAssessment");
  const shown = sel >= 0 ? data.entry[sel] : data.entry[ra >= 0 ? ra : 0];
  const download = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/fhir+json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `${id}-fhir-bundle.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
  return (
    <div className="grid gap-4 lg:grid-cols-[300px_1fr]">
      <div className="space-y-3 text-[13px]">
        <p className="ink-2">
          HL7 FHIR R4 bundle, validated against the FHIR schema. Terminologies follow India's ABDM stack: SNOMED CT conditions, LOINC labs and CGM
          (99504-3, as SampledData), WHO ATC medicines, UCUM units. The twin's live 2-hour forecast is exported as a <b>RiskAssessment</b> resource.
        </p>
        <table className="data">
          <tbody>
            {Object.entries(counts).map(([k, v]) => <tr key={k}><td>{k}</td><td className="num">{v}</td></tr>)}
          </tbody>
        </table>
        <button className="btn" onClick={download}><Download size={14} /> Download bundle</button>
        <label className="block">
          <span className="mb-1 block font-medium">Inspect resource</span>
          <select className="field w-full" value={sel} onChange={(e) => setSel(Number(e.target.value))}>
            <option value={-1}>RiskAssessment (twin forecast)</option>
            {data.entry.map((e, i) => (
              <option key={e.fullUrl} value={i}>
                {e.resource.resourceType} {e.resource.code?.coding?.[0]?.display ?? e.resource.code?.text ?? ""}
              </option>
            ))}
          </select>
        </label>
      </div>
      <pre className="max-h-[520px] overflow-auto rounded-lg p-3 text-[12px] leading-relaxed" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
        {JSON.stringify(shown?.resource, null, 2)}
      </pre>
    </div>
  );
}
