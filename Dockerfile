# --- dashboard build ---
FROM node:22-slim AS web
WORKDIR /web
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY dashboard/ ./
RUN npm run build

# --- API runtime ---
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MADHUTWIN_ROOT=/app
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
# hosts such as Hugging Face Spaces run the container as a non-root user: Numba's on-disk cache must
# go somewhere writable
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
COPY twin/ twin/
COPY artifacts/demo/ artifacts/demo/
COPY artifacts/results/ artifacts/results/
COPY --from=web /web/dist dashboard/dist
EXPOSE 8000
# cloud hosts (Render, Koyeb, Cloud Run) pass the port in $PORT; locally it defaults to 8000
CMD ["sh", "-c", "exec uvicorn twin.service.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
