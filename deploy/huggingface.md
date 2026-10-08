# Deployment on Hugging Face Spaces (needs a PRO plan)

> **Note (2026):** Hugging Face now requires a paid PRO subscription to host Docker Spaces. For a free
> deployment use Render instead: see [`render.md`](render.md). The steps below still work with PRO.

A Hugging Face Docker Space (CPU basic: 2 vCPU, 16 GB RAM) runs the whole app (the API and the dashboard,
including the 3D virtual patient) from this repository's `Dockerfile`. Judges get a link they can open
in any browser.

## Steps

1. **Create a free account** at https://huggingface.co/join and confirm your email.

2. **Create an access token** at https://huggingface.co/settings/tokens. Click *Create new token*,
   choose **Write**, give it a name (for example `madhutwin-deploy`), and copy the token (it starts with `hf_`).

3. **Install the uploader** (once), in the project's virtual environment:

   ```
   pip install huggingface_hub
   ```

4. **Make sure the demo exists.** `artifacts/demo/index.json` must be present (it is, if the dashboard
   works locally). If not, run `python scripts/build_demo.py`.

5. **Deploy** from the project folder:

   ```
   python deploy/push_to_hf.py --token hf_your_token_here
   ```

   This creates the public Space `<your-username>/madhutwin` (or updates it) and uploads about 33 MB:
   the Dockerfile, the API code, the dashboard source, the demo cohort and the evaluation results.
   Use `--space your-username/another-name` for a different name, or `--private` for a private Space.

6. **Wait for the build** (about 5–10 minutes). Open `https://huggingface.co/spaces/<your-username>/madhutwin`.
   The status badge goes *Building*, then *Running*; the *Logs* tab shows progress.

7. **Open the app** at `https://<your-username>-madhutwin.hf.space` (the script prints the exact link).
   Share this link in the README table, on the presentation's closing slide and in the video description.

To update the live demo after changing code or results, run step 5 again.

## Notes

- **Cost:** free. A free Space sleeps after about 48 hours without visitors and wakes on the next visit
  (the first page load then takes 1–2 minutes). Open it once before a judging session.
- **The "Ask the twin" assistant** uses the built-in offline engine. Do not add an `ANTHROPIC_API_KEY`
  secret to a public Space, or anyone could spend your API credit.
- **Licences:** the Space README carries the attributions (CGMacros-derived demo bundles are
  CC BY-NC-SA 4.0; the 3D organs are BodyParts3D, CC BY-SA 2.1 JP; the 3D body is MakeHuman, CC0).
  Keep `NOTICE.md` in the Space.
- **If the build fails**, the *Logs* tab shows why. The same image can be tested locally with
  `docker compose up --build`, then opening http://localhost:8000.
