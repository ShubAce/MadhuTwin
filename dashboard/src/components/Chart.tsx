import type { EChartsOption } from "echarts";
import { BarChart, LineChart, ScatterChart } from "echarts/charts";
import {
  AxisPointerComponent,
  GridComponent,
  MarkAreaComponent,
  MarkLineComponent,
  MarkPointComponent,
  TitleComponent,
  TooltipComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { useEffect, useRef } from "react";
import { useThemeVersion } from "../hooks";

// Register only what the dashboard uses (keeps the bundle small).
echarts.use([LineChart, BarChart, ScatterChart, GridComponent, TooltipComponent, TitleComponent, AxisPointerComponent,
  MarkAreaComponent, MarkLineComponent, MarkPointComponent, CanvasRenderer]);

/** Resolved chart tokens (read from CSS custom properties so light/dark swap in one place). */
export function tokens() {
  const s = getComputedStyle(document.documentElement);
  const v = (n: string) => s.getPropertyValue(n).trim();
  return {
    surface: v("--surface"),
    ink: v("--ink"),
    ink2: v("--ink-2"),
    muted: v("--muted"),
    grid: v("--grid"),
    axis: v("--axis"),
    s1: v("--series-1"),
    s2: v("--series-2"),
    s3: v("--series-3"),
    deemph: v("--deemph"),
    good: v("--good"),
    warning: v("--warning"),
    serious: v("--serious"),
    critical: v("--critical"),
    critical2: v("--critical-2"),
    rangeWash: v("--range-wash"),
    accent: v("--accent"),
  };
}
export type Tokens = ReturnType<typeof tokens>;

export function alpha(hex: string, a: number) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

/** Shared axis / tooltip chrome: hairline solid grid, muted labels, crosshair. */
export function baseOption(t: Tokens): EChartsOption {
  return {
    animation: false,
    textStyle: { fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: t.ink2 },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "line", lineStyle: { color: t.axis, width: 1 } },
      backgroundColor: t.surface,
      borderColor: t.grid,
      textStyle: { color: t.ink, fontSize: 12 },
      confine: true,
    },
  };
}

export function timeAxis(t: Tokens, extra: Record<string, unknown> = {}) {
  return {
    type: "time" as const,
    axisLine: { lineStyle: { color: t.axis } },
    axisTick: { show: false },
    axisLabel: { color: t.muted, fontSize: 11, hideOverlap: true, formatter: "{HH}:{mm}" },
    splitLine: { show: false },
    ...extra,
  };
}

export function valueAxis(t: Tokens, extra: Record<string, unknown> = {}) {
  return {
    type: "value" as const,
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { color: t.muted, fontSize: 11 },
    splitLine: { lineStyle: { color: t.grid, width: 1, type: "solid" as const } },
    ...extra,
  };
}

interface Props {
  option: (t: Tokens) => EChartsOption;
  height: number;
  group?: string;
  ariaLabel: string;
  deps?: unknown[];
}

export default function Chart({ option, height, group, ariaLabel, deps = [] }: Props) {
  const el = useRef<HTMLDivElement>(null);
  const inst = useRef<echarts.ECharts | null>(null);
  const themeV = useThemeVersion();

  useEffect(() => {
    if (!el.current) return;
    inst.current = echarts.init(el.current, undefined, { renderer: "canvas" });
    if (group) {
      inst.current.group = group;
      echarts.connect(group);
    }
    const ro = new ResizeObserver(() => inst.current?.resize());
    ro.observe(el.current);
    return () => {
      ro.disconnect();
      inst.current?.dispose();
      inst.current = null;
    };
  }, [group]);

  useEffect(() => {
    inst.current?.setOption(option(tokens()), { notMerge: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [themeV, ...deps]);

  return <div ref={el} role="img" aria-label={ariaLabel} style={{ height, width: "100%" }} />;
}
