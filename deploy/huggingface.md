# Public live demo on Hugging Face Spaces (free)

A public link means judges can try the dashboard without installing anything. The repository's `Dockerfile` builds the dashboard and serves it together with the API on port 8000.

1. Create a free account at https://huggingface.co, then click **New Space**: SDK **Docker**, template **Blank**, visibility **Public**, hardware **CPU basic (free)**.
2. Clone the Space and copy in what it needs:
   ```bash
   git clone https://huggingface.co/spaces/<you>/madhutwin && cd madhutwin
   cp -r "<repo>/twin" "<repo>/dashboard" "<repo>/requirements-api.txt" "<repo>/Dockerfile" "<repo>/.dockerignore" .
   mkdir -p artifacts && cp -r "<repo>/artifacts/demo" "<repo>/artifacts/results" artifacts/
   rm -rf dashboard/node_modules dashboard/dist
   ```
3. Put this front matter at the top of the Space's `README.md` (it tells Spaces which port to expose):
   ```yaml
   ---
   title: MadhuTwin
   emoji: 🩺
   colorFrom: blue
   colorTo: indigo
   sdk: docker
   app_port: 8000
   pinned: true
   ---
   ```
4. `git add . && git commit -m "MadhuTwin demo" && git push`. The Space builds in about 5–10 minutes and is then live at `https://huggingface.co/spaces/<you>/madhutwin`.
5. Add the link to the README table, the presentation's closing slide and the video description.

**Notes**

- The demo serves precomputed, out-of-sample predictions plus live physics simulations (what-if, meal ranking), so the free CPU tier is enough.
- "Ask the twin" uses the offline engine unless you add `MADHUTWIN_LLM=on` and an `ANTHROPIC_API_KEY` as Space secrets. Keep it off for a public demo to avoid API costs.
- The two CGMacros-derived demo bundles are CC BY-NC-SA 4.0. Keep the `NOTICE.md` attribution in the Space.
