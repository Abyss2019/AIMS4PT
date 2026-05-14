# AIMS4PT_cpx Web Interface

Lightweight FastAPI interface for running the main `calculator.ipynb` workflow locally or on a small single-worker server.

## Local Run

```bash
uvicorn web.main:app --host 127.0.0.1 --port 8000 --workers 1 --reload
```

Install web dependencies with:

```bash
pip install -e .
pip install -r requirements-web.txt
```

Configuration is read from environment variables:

- `MAX_UPLOAD_MB`, default `10`
- `SESSION_TTL_SECONDS`, default `7200`
- `MAX_CONCURRENT_CALCULATIONS`, default `3`
- `WEB_WORKERS_RECOMMENDED`, default `1`
- `APP_ENV`, default `local`
- `AIMS4PT_WEB_ENABLE_R_MODELS`, default `false`
- `AIMS4PT_WEB_ENABLE_TENSORFLOW_MODELS`, default `false`

Uploaded workbooks, generated reports, and calculation results are kept in memory only. Reports are generated on demand from the current session cache.
