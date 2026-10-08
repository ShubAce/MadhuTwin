# Free public deployment on Render

Render's free web service runs this repository's `Dockerfile` (dashboard + API + 3D virtual patient).
The app needs about 250 MB of memory, well inside the free 512 MB. No credit card is needed.

A free service sleeps after 15 minutes without visitors; the next visit wakes it in about a minute.
Open the link yourself shortly before judges look at it.

## Steps

1. **Put the code on GitHub** (skip if it is already there).
   - Create a free account at https://github.com and a new repository, for example `madhutwin`
     (public or private both work). Do not add a README or licence on GitHub; the project has them.
   - In `cmd`, from the project folder:
     ```
     cd "F:\Digital Twin"
     git add -A
     git commit -m "MadhuTwin: hybrid digital twin for type 2 diabetes"
     git branch -M main
     git remote add origin https://github.com/<your-username>/madhutwin.git
     git push -u origin main
     ```
   Raw datasets, trained-model pickles and `node_modules` are excluded by `.gitignore`; the demo
   cohort, results and 3D models that the app needs are included.

2. **Create a free Render account** at https://render.com, choosing *Sign up with GitHub*.

3. **Create the service from the blueprint.** In the Render dashboard click **New → Blueprint**,
   connect your GitHub account if asked, pick the `madhutwin` repository, and click **Apply**.
   Render reads `render.yaml` (free plan, Docker, health check on `/api/meta`).

   Without the blueprint: **New → Web Service**, pick the repository, *Language: Docker*,
   *Instance type: Free*, then **Deploy Web Service**.

4. **Wait for the first build** (about 5–10 minutes). The *Logs* tab shows progress; the deploy is
   done when it shows "Your service is live".

5. **Open the app** at the address shown at the top of the service page, such as
   `https://madhutwin.onrender.com` (Render adds a suffix if the name is taken). Put this link in
   the README's "Live demo" row, on the closing slide and in the video description.

Every `git push` to `main` redeploys automatically.

## Notes

- **No API key:** "Ask the twin" uses the built-in offline engine. Do not add `ANTHROPIC_API_KEY`
  to a public deployment, or anyone could spend your API credit.
- **Never paste tokens or keys into chats, issues or code.** If one leaks, revoke it at once.
- **Other free hosts** that run the same Docker image: Koyeb (free instance, also 512 MB).
  Hugging Face Docker Spaces now need a paid PRO plan (see `huggingface.md`).
