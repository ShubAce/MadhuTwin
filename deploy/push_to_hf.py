"""Deploy MadhuTwin to a free Hugging Face Space (Docker, CPU basic).

    pip install huggingface_hub
    python deploy/push_to_hf.py --token hf_xxx              # creates/updates <your-username>/madhutwin
    python deploy/push_to_hf.py --token hf_xxx --space my-name/my-space

The token needs "write" access (https://huggingface.co/settings/tokens). Instead of --token you
can set HF_TOKEN or run `huggingface-cli login` once.

What is uploaded: the Dockerfile, the API code (twin/), the dashboard source (built inside the
Space), the precomputed demo cohort and evaluation results, LICENSE and NOTICE. No raw datasets,
no trained-model pickles, no node_modules.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["Dockerfile", ".dockerignore", "requirements-api.txt", "LICENSE", "NOTICE.md", "twin", "dashboard",
           "artifacts/demo", "artifacts/results"]
SKIP = shutil.ignore_patterns("node_modules", "dist", "__pycache__", "*.pyc", ".vite", "*.try.png")

SPACE_README = """---
title: MadhuTwin
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: true
license: apache-2.0
short_description: Hybrid digital twin for type 2 diabetes (research demo)
---

# MadhuTwin: a hybrid digital twin for type 2 diabetes

Live demo for the Happiest Health Digital Twin Challenge 2026. Research prototype on synthetic
(India-calibrated) and de-identified open data only. **Not a medical device.**

Open the app, pick a patient from the risk-ranked panel, and try the what-if and therapy simulators,
the 3D virtual patient, the AGP report and the FHIR export.

Code: Apache-2.0. Demo bundles derived from CGMacros are CC BY-NC-SA 4.0. The 3D body is from
MakeHuman (CC0); the 3D organs are from BodyParts3D, (c) 2008 Life Science Integrated Database
Center, CC BY-SA 2.1 Japan. See NOTICE.md.
"""


def stage(dst: Path) -> None:
    for rel in INCLUDE:
        src = ROOT / rel
        if not src.exists():
            raise SystemExit(f"missing {rel}: build the demo first (python scripts/build_demo.py)")
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, target, ignore=SKIP)
        else:
            shutil.copy2(src, target)
    (dst / "README.md").write_text(SPACE_README, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", help="user/space-name (default: <your-username>/madhutwin)")
    ap.add_argument("--token", help="Hugging Face token with write access (or set HF_TOKEN / huggingface-cli login)")
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    a = ap.parse_args()

    api = HfApi(token=a.token)
    user = api.whoami()["name"]
    space = a.space or f"{user}/madhutwin"
    api.create_repo(space, repo_type="space", space_sdk="docker", private=a.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        size = sum(f.stat().st_size for f in Path(tmp).rglob("*") if f.is_file()) / 1e6
        print(f"uploading {size:.0f} MB to https://huggingface.co/spaces/{space} ...")
        api.upload_folder(folder_path=tmp, repo_id=space, repo_type="space", commit_message="Deploy MadhuTwin demo",
                          delete_patterns=["*"])  # mirror exactly: files removed locally disappear from the Space
    owner, name = space.split("/")
    print("done. The Space now builds (about 5-10 minutes).")
    print(f"  build logs and status: https://huggingface.co/spaces/{space}")
    print(f"  app, once running:     https://{owner}-{name}.hf.space".lower().replace("_", "-"))


if __name__ == "__main__":
    main()
