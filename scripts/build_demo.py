"""Build the dashboard's demo cohort (artifacts/demo/*.json) from the trained models.

    python scripts/build_demo.py
"""

from twin.service.demo import build_demo

if __name__ == "__main__":
    idx = build_demo()
    print(f"{len(idx)} virtual patients written to artifacts/demo/")
