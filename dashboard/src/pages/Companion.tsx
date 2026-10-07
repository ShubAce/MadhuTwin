import type { EChartsOption } from "echarts";
import { ArrowLeft, Footprints, HeartHandshake, Lightbulb } from "lucide-react";
import { useState } from "react";
import { go } from "../App";
import { InsightsResponse, PatientDetail, State } from "../api";
import Chart, { Tokens, alpha, baseOption, timeAxis, valueAxis } from "../components/Chart";
import { Status, TrendArrow, glucoseLevel, riskLevel } from "../components/ui";
import { useData } from "../hooks";

type Lang = "en" | "hi" | "kn";

// Patient-facing text. Hindi and Kannada drafts are to be reviewed by native-speaker clinicians before use.
const T: Record<Lang, Record<string, string>> = {
  en: {
    title: "My glucose", now: "Now", next: "Next 2 hours", inRange: "You are in your target range. Keep it up!",
    riskHigh: "Your glucose may rise above 180 in the next 2 hours. A 15-minute walk after your meal can help.",
    riskLow: "Your glucose may go low in the next 2 hours. Keep a sugar source with you, and follow your doctor's advice if you feel shaky or sweaty.",
    alreadyHigh: "Your glucose is high right now. Drink water and take a short walk if you can. Contact your doctor if it stays high.",
    tipTitle: "Tip for you", tip: "For you, {best} raises glucose less than {worst}.",
    walk: "A 15-minute walk after dinner lowers your peak by about {n} mg/dL.",
    care: "Your care team sees these readings and will contact you if needed.", lang: "Language",
  },
  hi: {
    title: "मेरा ग्लूकोज़", now: "अभी", next: "अगले 2 घंटे", inRange: "आपका ग्लूकोज़ लक्ष्य सीमा में है। ऐसे ही बनाए रखें!",
    riskHigh: "अगले 2 घंटों में आपका ग्लूकोज़ 180 से ऊपर जा सकता है। खाने के बाद 15 मिनट टहलने से मदद मिल सकती है।",
    riskLow: "अगले 2 घंटों में आपका ग्लूकोज़ कम हो सकता है। अपने पास कुछ मीठा रखें, और कंपकंपी या पसीना आने पर अपने डॉक्टर की सलाह मानें।",
    alreadyHigh: "अभी आपका ग्लूकोज़ ज़्यादा है। पानी पिएँ और हो सके तो थोड़ा टहलें। अगर यह ज़्यादा बना रहे तो अपने डॉक्टर से संपर्क करें।",
    tipTitle: "आपके लिए सुझाव", tip: "आपके लिए, {best} से ग्लूकोज़ {worst} की तुलना में कम बढ़ता है।",
    walk: "रात के खाने के बाद 15 मिनट टहलने से आपका ग्लूकोज़ लगभग {n} mg/dL कम बढ़ता है।",
    care: "आपकी देखभाल टीम ये रीडिंग देखती है और ज़रूरत होने पर आपसे संपर्क करेगी।", lang: "भाषा",
  },
  kn: {
    title: "ನನ್ನ ಗ್ಲೂಕೋಸ್", now: "ಈಗ", next: "ಮುಂದಿನ 2 ಗಂಟೆಗಳು", inRange: "ನಿಮ್ಮ ಗ್ಲೂಕೋಸ್ ಗುರಿ ಮಟ್ಟದಲ್ಲಿದೆ. ಹೀಗೆಯೇ ಮುಂದುವರಿಸಿ!",
    riskHigh: "ಮುಂದಿನ 2 ಗಂಟೆಗಳಲ್ಲಿ ನಿಮ್ಮ ಗ್ಲೂಕೋಸ್ 180 ಕ್ಕಿಂತ ಹೆಚ್ಚಾಗಬಹುದು. ಊಟದ ನಂತರ 15 ನಿಮಿಷ ನಡೆಯುವುದು ಸಹಾಯ ಮಾಡಬಹುದು.",
    riskLow: "ಮುಂದಿನ 2 ಗಂಟೆಗಳಲ್ಲಿ ನಿಮ್ಮ ಗ್ಲೂಕೋಸ್ ಕಡಿಮೆಯಾಗಬಹುದು. ಸಿಹಿ ಪದಾರ್ಥವನ್ನು ಜೊತೆಯಲ್ಲಿ ಇಟ್ಟುಕೊಳ್ಳಿ, ನಡುಕ ಅಥವಾ ಬೆವರು ಬಂದರೆ ನಿಮ್ಮ ವೈದ್ಯರ ಸಲಹೆಯನ್ನು ಪಾಲಿಸಿ.",
    alreadyHigh: "ಈಗ ನಿಮ್ಮ ಗ್ಲೂಕೋಸ್ ಹೆಚ್ಚಾಗಿದೆ. ನೀರು ಕುಡಿಯಿರಿ ಮತ್ತು ಸಾಧ್ಯವಾದರೆ ಸ್ವಲ್ಪ ನಡೆಯಿರಿ. ಹೆಚ್ಚಾಗಿಯೇ ಇದ್ದರೆ ನಿಮ್ಮ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
    tipTitle: "ನಿಮಗಾಗಿ ಸಲಹೆ", tip: "ನಿಮಗೆ, {best} ತಿಂದಾಗ {worst} ಗಿಂತ ಗ್ಲೂಕೋಸ್ ಕಡಿಮೆ ಏರುತ್ತದೆ.",
    walk: "ರಾತ್ರಿ ಊಟದ ನಂತರ 15 ನಿಮಿಷ ನಡೆದರೆ ನಿಮ್ಮ ಗ್ಲೂಕೋಸ್ ಸುಮಾರು {n} mg/dL ಕಡಿಮೆ ಏರುತ್ತದೆ.",
    care: "ನಿಮ್ಮ ಆರೈಕೆ ತಂಡ ಈ ರೀಡಿಂಗ್‌ಗಳನ್ನು ನೋಡುತ್ತದೆ ಮತ್ತು ಅಗತ್ಯವಿದ್ದರೆ ನಿಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸುತ್ತದೆ.", lang: "ಭಾಷೆ",
  },
};
const LANGS: { key: Lang; label: string }[] = [{ key: "en", label: "English" }, { key: "hi", label: "हिन्दी" }, { key: "kn", label: "ಕನ್ನಡ" }];

const fill = (s: string, v: Record<string, string | number>) => s.replace(/\{(\w+)\}/g, (_, k) => String(v[k] ?? ""));

export default function Companion({ id, clock }: { id: string; clock: number }) {
  const enc = encodeURIComponent(id);
  const { data: p } = useData<PatientDetail>(`/api/patients/${enc}`);
  const { data: s } = useData<State>(`/api/patients/${enc}/state?clock=${clock}&history_h=3`);
  const { data: ins } = useData<InsightsResponse>(`/api/patients/${enc}/insights?clock=${Math.floor(clock / 60) * 60}`);
  const pref = p?.display.language === "Hindi" ? "hi" : p?.display.language === "Kannada" ? "kn" : "en";
  const [lang, setLang] = useState<Lang | null>(null);
  const L = T[lang ?? pref];
  if (!p || !s) return <p className="muted">Loading…</p>;

  const g = [...s.series.cgm].reverse().find((v) => v != null) ?? null;
  const n = s.series.cgm.length;
  const prev = s.series.cgm[Math.max(n - 4, 0)];
  const trend = g != null && prev != null ? (g - prev) / 15 : null;
  const fc = s.forecast;
  const msg = g != null && g > 180 ? L.alreadyHigh : riskLevel(fc?.hypo, "hypo") !== "good" ? L.riskLow : riskLevel(fc?.spike, "spike") !== "good" ? L.riskHigh : L.inRange;
  const tone = g != null && g > 180 ? "warning" : riskLevel(fc?.hypo, "hypo") !== "good" ? "critical" : riskLevel(fc?.spike, "spike") !== "good" ? "warning" : "good";
  const rank = ins?.meal_ranking ?? [];

  const option = (t: Tokens): EChartsOption => {
    const hist = s.times.map((x, i) => [x, s.series.cgm[i]]);
    const ft = fc ? [fc.anchor_time, ...fc.times] : [];
    const lo = fc ? [g, ...fc.q10] : [];
    const hi = fc ? [g, ...fc.q90] : [];
    return {
      ...baseOption(t),
      tooltip: { show: false },
      grid: { left: 34, right: 10, top: 8, bottom: 22 },
      xAxis: timeAxis(t),
      yAxis: valueAxis(t, { min: 40, max: (v: { max: number }) => Math.max(300, Math.ceil(v.max / 50) * 50) }),
      series: [
        { type: "line", data: hist, showSymbol: false, lineStyle: { width: 2, color: t.s1 },
          markArea: { silent: true, itemStyle: { color: t.rangeWash }, data: [[{ yAxis: 70 }, { yAxis: 180 }]] } },
        { type: "line", data: ft.map((x, i) => [x, lo[i]]), stack: "b", showSymbol: false, lineStyle: { opacity: 0 }, silent: true },
        { type: "line", data: ft.map((x, i) => [x, hi[i] != null && lo[i] != null ? (hi[i] as number) - (lo[i] as number) : null]), stack: "b",
          showSymbol: false, lineStyle: { opacity: 0 }, areaStyle: { color: alpha(t.s2, 0.2) }, silent: true },
        { type: "line", data: ft.map((x, i) => [x, i === 0 ? g : fc!.q50[i - 1]]), showSymbol: false, lineStyle: { width: 2, color: t.s2 } },
      ],
    };
  };

  return (
    <div className="mx-auto max-w-[420px] space-y-3">
      <button className="btn" onClick={() => go(`/patient/${enc}`)}><ArrowLeft size={14} /> Doctor view</button>
      <div className="card space-y-4 p-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-[18px] font-semibold">{L.title}</div>
            <div className="text-[12px] muted">{p.display.name}</div>
          </div>
          <label className="text-[12px]">
            <span className="sr-only">{L.lang}</span>
            <select className="field" value={lang ?? pref} onChange={(e) => setLang(e.target.value as Lang)} aria-label={L.lang}>
              {LANGS.map((l) => <option key={l.key} value={l.key}>{l.label}</option>)}
            </select>
          </label>
        </div>
        <div className="flex items-end gap-3">
          <div>
            <div className="text-[12px] muted">{L.now}</div>
            <div className="flex items-center gap-1 text-[40px] font-semibold leading-none">{g == null ? "–" : Math.round(g)}<span className="text-[14px] font-normal muted">mg/dL</span></div>
          </div>
          <TrendArrow rate={trend} />
          <div className="ml-auto"><Status level={glucoseLevel(g)}>{g == null ? "" : g < 70 ? "< 70" : g > 180 ? "> 180" : "70–180"}</Status></div>
        </div>
        <div>
          <div className="mb-1 text-[12px] muted">{L.next}</div>
          <Chart option={option} height={150} ariaLabel={L.next} deps={[s]} />
        </div>
        <div className="rounded-lg p-3 text-[14px]" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
          <Status level={tone as "good" | "warning" | "critical"}>{msg}</Status>
        </div>
        {rank.length > 1 && (
          <div className="flex gap-2 text-[14px]">
            <Lightbulb size={18} className="shrink-0" color="var(--accent)" aria-hidden />
            <div>
              <div className="font-medium">{L.tipTitle}</div>
              <div className="ink-2">{fill(L.tip, { best: rank[0].name, worst: rank[rank.length - 1].name })}</div>
            </div>
          </div>
        )}
        {ins && ins.walk.reduction >= 3 && (
          <div className="flex gap-2 text-[14px]">
            <Footprints size={18} className="shrink-0" color="var(--accent)" aria-hidden />
            <div className="ink-2">{fill(L.walk, { n: ins.walk.reduction })}</div>
          </div>
        )}
        <div className="flex gap-2 text-[12px] muted">
          <HeartHandshake size={16} className="shrink-0" aria-hidden />
          <span>{L.care}</span>
        </div>
      </div>
      <p className="text-[11px] muted">Patient companion prototype. Hindi and Kannada text drafted for review by native-speaker clinicians before deployment.</p>
    </div>
  );
}
