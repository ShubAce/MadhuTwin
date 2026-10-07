export type Num = number | null;

export interface Display {
  name: string;
  age: Num;
  sex: string | null;
  city: string | null;
  language: string | null;
  region?: string;
  label: string;
}

export interface Alert {
  bin: number;
  kind: "spike" | "hypo";
  prob: number;
  predicted_value: number;
  in_min: number;
  why: string[];
  time?: string;
}

export interface PatientSummary {
  id: string;
  story: string;
  source: string;
  display: Display;
  status: string;
  now: string;
  glucose: Num;
  trend: Num;
  risk_spike: Num;
  risk_hypo: Num;
  tir_24h: Num;
  sparkline: Num[];
  last_alert: Alert | null;
}

export interface Medication {
  drug: string;
  display: string;
  atc?: string;
  class: string;
  dose?: number;
  unit?: string;
  times?: string[];
}

export interface Lab {
  key: string;
  loinc: string;
  display: string;
  value: number;
  unit: string;
}

export interface Ehr {
  status: string;
  diabetes_years: Num;
  bmi: Num;
  weight_kg: Num;
  height_cm: Num;
  waist_cm?: Num;
  conditions: { key: string; display: string; snomed?: string; years?: number }[];
  medications: Medication[];
  labs: Lab[];
  lab_history: { date: string; hba1c_pct: number; fpg_mgdl: number; weight_kg: number; sbp: number; dbp: number }[];
  family_history: string[];
  genetics: { TCF7L2_rs7903146: string; prs_z: number } | null;
  vitals: { sbp?: number; dbp?: number; resting_hr?: number };
  lifestyle?: Record<string, boolean | number | string>;
  adherence?: number;
}

export interface TwinInfo {
  params: Record<string, Num>;
  tau_scale: Num;
  fit_rmse_prior: Num;
  fit_rmse_personalised: Num;
  si_daily: Num[];
}

export interface PatientDetail {
  id: string;
  story: string;
  source: string;
  display: Display;
  ehr: Ehr;
  twin: TwinInfo;
  gate: Num[] | null;
  consent: { status: string; purpose: string[]; artefact: string; expires: string };
}

export interface EventRow {
  bin: number;
  kind: string;
  label: string;
  food: string | null;
  carbs: Num;
  amount: Num;
  time: string;
}

export interface Forecast {
  anchor_time: string;
  times: string[];
  q10: Num[];
  q50: Num[];
  q90: Num[];
  twin: Num[];
  spike: Num;
  hypo: Num;
  why_spike: string[] | null;
  why_hypo: string[] | null;
}

export interface State {
  id: string;
  now: string;
  bin: number;
  times: string[];
  series: { cgm: Num[]; hr: Num[]; steps: Num[]; hrv: Num[]; sleep: Num[]; mets: Num[] };
  events: EventRow[];
  forecast: Forecast | null;
  alerts: Alert[];
  si_today: Num;
  gate: Num[] | null;
}

export interface Agp {
  days: number;
  active_pct: number;
  mean: number;
  gmi: number;
  cv: number;
  tir: number;
  tar1: number;
  tar2: number;
  tbr1: number;
  tbr2: number;
  profile: { slot: number; time: string; p5?: number; p25?: number; p50?: number; p75?: number; p95?: number }[];
  targets: Record<string, string>;
}

export interface Insight {
  kind: string;
  title: string;
  text: string;
}

export interface MealRank {
  food: string;
  name: string;
  carbs: number;
  gi: number;
  tags: string[];
  rise: number;
  peak_min: number;
  above_180_min: number;
}

export interface InsightsResponse {
  insights: Insight[];
  meal_ranking: MealRank[];
  walk: { meal: string; peak_without: number; peak_with_walk: number; reduction: number };
}

export interface Food {
  key: string;
  name: string;
  serving: string;
  carbs: number;
  protein: number;
  fat: number;
  fiber: number;
  gi: number;
  region: string;
  tags: string[];
  kcal: number;
}

export interface WhatIfRequest {
  clock: number;
  meal?: { food: string; portion: number; in_min: number } | null;
  walk?: { minutes: number; delay_min: number } | null;
  insulin?: { drug: string; units: number; in_min: number } | null;
  sleep_hours?: number | null;
  illness?: boolean;
  horizon?: number;
}

export interface WhatIfResult {
  times: string[];
  baseline: number[];
  scenario: number[];
  baseline_peak: number;
  scenario_peak: number;
  baseline_min: number;
  scenario_min: number;
  baseline_above_180_min: number;
  scenario_above_180_min: number;
  baseline_below_70_min: number;
  scenario_below_70_min: number;
}

export interface AskResponse {
  intent: string;
  answer: string;
  facts: string[];
  engine: string;
}

export interface Meta {
  name: string;
  version: string;
  default_clock: number;
  max_clock: number;
  patients: number;
  disclaimer: string;
}

const HEADERS = { "x-user": "dr.demo@madhutwin", "x-purpose": "treatment" };

export async function get<T>(path: string): Promise<T> {
  const r = await fetch(path, { headers: HEADERS });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json() as Promise<T>;
}

export async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, {
    method: "POST",
    headers: { ...HEADERS, "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json() as Promise<T>;
}
