import { useEffect, useRef, useState } from "react";
import { get } from "./api";

/** Fetch JSON; keeps the previous data while refetching (no skeleton flash, no layout jump). */
export function useData<T>(path: string | null, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const seq = useRef(0);
  useEffect(() => {
    if (!path) return;
    const id = ++seq.current;
    setLoading(true);
    get<T>(path)
      .then((d) => {
        if (id === seq.current) {
          setData(d);
          setError(null);
        }
      })
      .catch((e: Error) => id === seq.current && setError(e.message))
      .finally(() => id === seq.current && setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, ...deps]);
  return { data, loading, error };
}

/** Follow the OS / user theme so charts re-read their CSS tokens. */
export function useThemeVersion() {
  const [v, setV] = useState(0);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const bump = () => setV((x) => x + 1);
    mq.addEventListener("change", bump);
    const obs = new MutationObserver(bump);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => {
      mq.removeEventListener("change", bump);
      obs.disconnect();
    };
  }, []);
  return v;
}

export function fmtTime(iso: string) {
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
}

export function clockLabel(clock: number) {
  const day = Math.floor(clock / 1440) + 1;
  const m = Math.floor(clock % 1440);
  return `Day ${day} · ${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
}

export function pct(x: number | null | undefined, digits = 0) {
  return x == null ? "–" : `${(x * 100).toFixed(digits)}%`;
}
