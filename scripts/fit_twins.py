"""Build processed CGMacros tables and fit a personal mechanistic twin for every participant.

Outputs
    data/processed/cgmacros_{static,series,events}.parquet
    artifacts/twins/cgmacros_twins.parquet     fitted physiology per participant
    artifacts/twins/cgmacros_twins_summary.csv  fit quality (prior-only vs personalised RMSE)
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from twin.ingest import cgmacros
from twin.paths import ARTIFACTS, PROCESSED, ensure_dirs
from twin.physiology.calibrate import fit_patient
from twin.record import PatientRecord


def _fit_one(args):
    static_row, series, events = args
    rec = PatientRecord.from_frames(static_row, series, events)
    return fit_patient(rec)


def main() -> None:
    ensure_dirs()
    static, series, events = cgmacros.load()
    static.to_parquet(PROCESSED / "cgmacros_static.parquet", index=False)
    series.to_parquet(PROCESSED / "cgmacros_series.parquet", index=False)
    events.to_parquet(PROCESSED / "cgmacros_events.parquet", index=False)
    print(f"processed: {len(static)} participants, {len(series)} 5-min bins, {len(events)} meals")

    jobs = [(row, series[series.patient_id == row.patient_id], events[events.patient_id == row.patient_id])
            for _, row in static.iterrows()]
    t0 = time.time()
    results = []
    with ProcessPoolExecutor() as pool:
        futures = [pool.submit(_fit_one, j) for j in jobs]
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            print(f"  {r.patient_id}: RMSE {r.rmse_prior:5.1f} -> {r.rmse_fit:5.1f} mg/dL")
    print(f"fitted {len(results)} twins in {time.time() - t0:.0f}s")

    out = ARTIFACTS / "twins"
    out.mkdir(parents=True, exist_ok=True)
    params = pd.DataFrame([{"patient_id": r.patient_id, **r.params, "tau_scale": r.tau_scale} for r in results])
    params.sort_values("patient_id").to_parquet(out / "cgmacros_twins.parquet", index=False)
    summ = pd.DataFrame([{"patient_id": r.patient_id, "rmse_prior": r.rmse_prior, "rmse_fit": r.rmse_fit,
                          "n_windows": r.n_windows, "converged": r.success} for r in results])
    summ = summ.merge(static[["patient_id", "status", "hba1c_pct", "homa_ir"]], on="patient_id").sort_values("patient_id")
    summ.to_csv(out / "cgmacros_twins_summary.csv", index=False)
    print(summ.groupby("status")[["rmse_prior", "rmse_fit"]].mean().round(1))
    print("overall", summ[["rmse_prior", "rmse_fit"]].mean().round(1).to_dict())


if __name__ == "__main__":
    main()
