"""Reproduce the whole MadhuTwin pipeline end to end.

    python scripts/run_all.py            # everything (about 2-3 hours on a 16-core CPU)
    python scripts/run_all.py --from build_dataset

Each step is a script in this folder; the pipeline stops at the first failure.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    ("download_data", []),
    ("fit_twins", []),
    ("generate_cohort", ["--n", "1000", "--days", "14"]),
    ("build_dataset", []),
    ("exp_synthetic", ["--twinnet-ablations"]),
    ("exp_real", []),
    ("exp_extra", []),
    ("exp_learning", []),
    ("exp_cgm_light", []),
    ("exp_fidelity", []),
    ("exp_robustness", []),
    ("bench", []),
    ("build_demo", []),
    ("make_report", []),
    ("make_techreport", []),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=[s for s, _ in STEPS], default=STEPS[0][0])
    args = ap.parse_args()
    names = [s for s, _ in STEPS]
    t_all = time.time()
    for name, extra in STEPS[names.index(args.start):]:
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        rc = subprocess.call([sys.executable, str(HERE / f"{name}.py"), *extra])
        if rc != 0:
            print(f"step {name} failed with exit code {rc}")
            return rc
        print(f"=== {name} done in {time.time() - t0:.0f}s ===", flush=True)
    print(f"\npipeline finished in {(time.time() - t_all) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
