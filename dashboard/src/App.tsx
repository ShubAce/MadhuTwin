import { Activity, BookOpenCheck, Moon, Pause, Play, ShieldCheck, Sun, Users } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Meta } from "./api";
import { clockLabel, useData } from "./hooks";
import Companion from "./pages/Companion";
import Evidence from "./pages/Evidence";
import Panel from "./pages/Panel";
import PatientView from "./pages/PatientView";
import Privacy from "./pages/Privacy";

type Route = { page: "panel" } | { page: "patient"; id: string; params: URLSearchParams } | { page: "companion"; id: string } | { page: "evidence" } | { page: "privacy" };

function parseHash(): Route {
  const [h, query = ""] = window.location.hash.replace(/^#\/?/, "").split("?");
  const [a, b] = h.split("/");
  if (a === "patient" && b) return { page: "patient", id: decodeURIComponent(b), params: new URLSearchParams(query) };
  if (a === "companion" && b) return { page: "companion", id: decodeURIComponent(b) };
  if (a === "evidence") return { page: "evidence" };
  if (a === "privacy") return { page: "privacy" };
  return { page: "panel" };
}

export function go(path: string) {
  window.location.hash = path;
}

const SPEEDS = [
  { label: "1 min/s", v: 1 },
  { label: "5 min/s", v: 5 },
  { label: "15 min/s", v: 15 },
  { label: "1 h/s", v: 60 },
];

export default function App() {
  const [route, setRoute] = useState<Route>(parseHash());
  const { data: meta } = useData<Meta>("/api/meta");
  const [clock, setClock] = useState<number>(390);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(5);
  const [theme, setTheme] = useState<"light" | "dark" | null>(() => {
    const q = new URLSearchParams(window.location.search).get("theme");
    return q === "light" || q === "dark" ? q : null;
  });

  useEffect(() => {
    const on = () => setRoute(parseHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  useEffect(() => {
    if (!meta) return;
    const q = new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("clock");
    setClock(q != null && !Number.isNaN(Number(q)) ? Number(q) : meta.default_clock);
  }, [meta]);
  useEffect(() => {
    if (!playing || !meta) return;
    const id = window.setInterval(() => setClock((c) => (c + speed) % meta.max_clock), 1000);
    return () => window.clearInterval(id);
  }, [playing, speed, meta]);
  useEffect(() => {
    if (theme) document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    const dark = theme ? theme === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    setTheme(dark ? "light" : "dark");
  }, [theme]);

  const nav = [
    { key: "panel", label: "Patients", icon: <Users size={16} />, href: "/" },
    { key: "evidence", label: "Model evidence", icon: <BookOpenCheck size={16} />, href: "/evidence" },
    { key: "privacy", label: "Privacy & audit", icon: <ShieldCheck size={16} />, href: "/privacy" },
  ];
  const active = route.page === "patient" || route.page === "companion" ? "panel" : route.page;

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
        <div className="mx-auto flex max-w-[1440px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-2.5">
          <button className="flex items-center gap-2" onClick={() => go("/")} aria-label="MadhuTwin home">
            <span className="grid h-8 w-8 place-items-center rounded-lg" style={{ background: "var(--accent)" }}>
              <Activity size={18} color="#fff" />
            </span>
            <span className="text-left leading-tight">
              <span className="block text-[15px] font-semibold">MadhuTwin</span>
              <span className="block text-[11px] muted">Type 2 Diabetes digital twin</span>
            </span>
          </button>
          <nav className="flex gap-1" aria-label="Main">
            {nav.map((n) => (
              <button key={n.key} className="tab flex items-center gap-1.5" aria-selected={active === n.key} onClick={() => go(n.href)}>
                {n.icon}
                {n.label}
              </button>
            ))}
          </nav>
          <div className="ml-auto flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-2 rounded-lg border px-2 py-1" style={{ borderColor: "var(--border)" }}>
              <button className="btn" style={{ padding: "4px 8px" }} onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause live replay" : "Play live replay"}>
                {playing ? <Pause size={14} /> : <Play size={14} />}
                {playing ? "Live" : "Replay"}
              </button>
              <select className="field" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="Replay speed">
                {SPEEDS.map((s) => (
                  <option key={s.v} value={s.v}>
                    {s.label}
                  </option>
                ))}
              </select>
              <input
                type="range"
                min={0}
                max={meta?.max_clock ?? 5760}
                step={15}
                value={clock}
                onChange={(e) => setClock(Number(e.target.value))}
                aria-label="Demo clock"
                className="w-36"
              />
              <span className="tabular min-w-[92px] text-[13px] font-medium">{clockLabel(clock)}</span>
            </div>
            <button className="btn" style={{ padding: "6px" }} onClick={toggleTheme} aria-label="Toggle dark mode">
              {(theme ?? (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")) === "dark" ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </div>
        </div>
      </header>
      <div className="mx-auto max-w-[1440px] px-4 pt-2">
        <p className="text-[12px] muted">
          Research prototype for the Happiest Health Digital Twin Challenge. Synthetic (India-calibrated) and de-identified open data only. Not a medical device.
        </p>
      </div>
      <main className="mx-auto max-w-[1440px] px-4 pb-10 pt-3">
        {route.page === "panel" && <Panel clock={clock} />}
        {route.page === "patient" && <PatientView key={route.id} id={route.id} clock={clock} params={route.params} />}
        {route.page === "companion" && <Companion id={route.id} clock={clock} />}
        {route.page === "evidence" && <Evidence />}
        {route.page === "privacy" && <Privacy />}
      </main>
    </div>
  );
}
