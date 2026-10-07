"""Generate the synthetic India-calibrated cohort and its FHIR R4 bundles.

    python scripts/generate_cohort.py --n 1000 --days 14

Outputs (data/synthetic/full is git-ignored; a small sample is committed for reviewers):
    data/synthetic/full/{static,series,events,truth,lab_history,nights}.parquet
    data/synthetic/full/profiles.json
    data/synthetic/fhir/<patient>.json        (FHIR R4 Bundles for the first --fhir patients)
    data/synthetic/sample/...                  (first 5 patients, for quick inspection)
"""

from __future__ import annotations

import argparse
import json
import time

from twin.cohort import generate_cohort
from twin.ehr.fhir import patient_bundle
from twin.paths import SYNTHETIC


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fhir", type=int, default=25, help="number of patients to export as FHIR bundles")
    args = ap.parse_args()

    t0 = time.time()
    c = generate_cohort(n=args.n, days=args.days, seed=args.seed)
    print(f"generated {args.n} patients x {args.days} days in {time.time() - t0:.0f}s")

    full = SYNTHETIC / "full"
    full.mkdir(parents=True, exist_ok=True)
    for name in ("static", "series", "events", "truth", "lab_history", "nights"):
        getattr(c, name).to_parquet(full / f"{name}.parquet", index=False)
    (full / "profiles.json").write_text(json.dumps([p.to_dict() for p in c.profiles], indent=1), encoding="utf-8")

    sample = SYNTHETIC / "sample"
    sample.mkdir(parents=True, exist_ok=True)
    ids = set(c.static.patient_id.head(5))
    for name in ("static", "series", "events", "lab_history"):
        df = getattr(c, name)
        df[df.patient_id.isin(ids)].to_csv(sample / f"{name}.csv", index=False)

    fhir_dir = SYNTHETIC / "fhir"
    fhir_dir.mkdir(parents=True, exist_ok=True)
    static = c.static.set_index("patient_id")
    for p in c.profiles[: args.fhir]:
        b = patient_bundle(p, static.loc[p.patient_id], c.lab_history,
                           c.series if p.patient_id in ids else None)
        (fhir_dir / f"{p.patient_id}.json").write_text(json.dumps(b, indent=1), encoding="utf-8")
    print(f"wrote {min(args.fhir, args.n)} validated FHIR R4 bundles to {fhir_dir}")
    print(c.static.groupby("status")[["hba1c_pct", "fpg_mgdl", "bmi"]].mean().round(2))


if __name__ == "__main__":
    main()
