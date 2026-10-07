"""Build model-ready datasets (both streams fused + mechanistic-twin features) for every source.

    python scripts/build_dataset.py                 # synthetic, cgmacros, shanghai
    python scripts/build_dataset.py --sources cgmacros

Writes artifacts/datasets/<source>.pkl : list[PatientArrays]
"""

from __future__ import annotations

import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from twin.data.physics import attach_physics
from twin.data.windows import build_patient
from twin.paths import ARTIFACTS, PROCESSED, SYNTHETIC, ensure_dirs

CALIB_DAYS = {"synthetic": 7.0}


def _load(source: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if source == "synthetic":
        d = SYNTHETIC / "full"
        return (pd.read_parquet(d / "static.parquet"), pd.read_parquet(d / "series.parquet"), pd.read_parquet(d / "events.parquet"))
    if source == "cgmacros":
        from twin.ingest import cgmacros
        st, se, ev = cgmacros.load()
    else:
        from twin.ingest import shanghai
        st, se, ev = shanghai.load()
    for name, df in (("static", st), ("series", se), ("events", ev)):
        df.to_parquet(PROCESSED / f"{source}_{name}.parquet", index=False)
    return st, se, ev


def _one(args):
    row, series, events, source = args
    arr = build_patient(row, series)
    span = (series["ts"].max() - series["ts"].min()).total_seconds() / 60
    calib = int(CALIB_DAYS.get(source, 0) * 1440) or int(span // 2)
    return attach_physics(arr, row, series, events, calib)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=["cgmacros", "shanghai", "synthetic"])
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    ensure_dirs()
    out = ARTIFACTS / "datasets"
    out.mkdir(parents=True, exist_ok=True)
    for source in args.sources:
        t0 = time.time()
        st, se, ev = _load(source)
        se_g = dict(tuple(se.groupby("patient_id")))
        ev_g = dict(tuple(ev.groupby("patient_id")))
        empty_ev = ev.iloc[:0]
        jobs = [(row.to_dict(), se_g[row.patient_id], ev_g.get(row.patient_id, empty_ev), source)
                for _, row in st.iterrows() if row.patient_id in se_g]
        arrays = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_one, j) for j in jobs]
            for k, f in enumerate(as_completed(futures), 1):
                arrays.append(f.result())
                if k % 50 == 0 or k == len(jobs):
                    print(f"  {source}: {k}/{len(jobs)} patients ({time.time() - t0:.0f}s)")
        arrays.sort(key=lambda a: a.pid)
        with open(out / f"{source}.pkl", "wb") as fh:
            pickle.dump(arrays, fh, protocol=5)
        n_anchor = sum(len(a.anchors) for a in arrays)
        print(f"{source}: {len(arrays)} patients, {n_anchor} anchors in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
