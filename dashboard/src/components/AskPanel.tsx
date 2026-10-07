import { Bot, Send } from "lucide-react";
import { FormEvent, useState } from "react";
import { AskResponse, post } from "../api";

const SUGGESTED = [
  "Why is a spike predicted?",
  "What is the hypoglycaemia risk right now?",
  "Which Indian meals suit this patient best?",
  "Would a walk after dinner help?",
  "Has insulin sensitivity changed recently?",
  "Summarise the last week",
];

interface Msg {
  role: "user" | "twin";
  text: string;
  engine?: string;
}

export default function AskPanel({ id, clock, name }: { id: string; clock: number; name: string }) {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);

  const ask = async (question: string) => {
    if (!question.trim()) return;
    setMsgs((m) => [...m, { role: "user", text: question }]);
    setQ("");
    setBusy(true);
    try {
      const r = await post<AskResponse>(`/api/patients/${encodeURIComponent(id)}/ask`, { question, clock });
      setMsgs((m) => [...m, { role: "twin", text: r.answer, engine: r.engine }]);
    } catch (e) {
      setMsgs((m) => [...m, { role: "twin", text: `Error: ${(e as Error).message}` }]);
    } finally {
      setBusy(false);
    }
  };
  const submit = (e: FormEvent) => {
    e.preventDefault();
    ask(q);
  };

  return (
    <div className="space-y-3">
      <p className="text-[13px] ink-2">
        Ask {name}'s twin in plain language. Answers are composed from the twin's own tools (forecast, AGP, personal meal simulation, insulin-sensitivity
        tracking) and cite numbers. Decision support only; it never prescribes doses.
      </p>
      <div className="flex flex-wrap gap-1.5">
        {SUGGESTED.map((s) => (
          <button key={s} className="chip" style={{ cursor: "pointer" }} onClick={() => ask(s)}>{s}</button>
        ))}
      </div>
      <div className="max-h-[420px] space-y-3 overflow-auto rounded-lg p-3" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }} aria-live="polite">
        {msgs.length === 0 && <p className="text-[13px] muted">No questions yet.</p>}
        {msgs.map((m, i) => (
          <div key={i} className={`flex gap-2 ${m.role === "user" ? "justify-end" : ""}`}>
            {m.role === "twin" && <Bot size={18} className="mt-0.5 shrink-0" color="var(--accent)" aria-hidden />}
            <div className="max-w-[80%] rounded-lg px-3 py-2 text-[13px]" style={{ background: m.role === "user" ? "var(--accent-wash)" : "var(--surface)", border: "1px solid var(--border)" }}>
              {m.text}
              {m.engine && <div className="mt-1 text-[11px] muted">{m.engine === "offline" ? "Grounded offline engine" : `LLM: ${m.engine}`}</div>}
            </div>
          </div>
        ))}
        {busy && <p className="text-[13px] muted">Thinking with the twin…</p>}
      </div>
      <form onSubmit={submit} className="flex gap-2">
        <input className="field flex-1" value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. Why does she spike after dinner?" aria-label="Question for the twin" />
        <button className="btn btn-primary" type="submit" disabled={busy}><Send size={14} /> Ask</button>
      </form>
    </div>
  );
}
