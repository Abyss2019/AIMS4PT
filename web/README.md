# AIMS4PT_cpx Web Interface

A user-friendly web interface for AIMS4PT_cpx, an AI-assisted framework for clinopyroxene-based pressure–temperature estimation.

## Local Run

```bash
uvicorn web.main:app --host 127.0.0.1 --port 8003 --workers 1
```

Install web dependencies with:

```bash
pip install -e ".[web]"
```

Configuration is read from environment variables:

- `MAX_UPLOAD_MB`, default `10`
- `SESSION_TTL_SECONDS`, default `7200`
- `MAX_CONCURRENT_CALCULATIONS`, default `3`
- `WEB_WORKERS_RECOMMENDED`, default `1`
- `APP_ENV`, default `local`
- `AIMS4PT_WEB_ENABLE_R_MODELS`, default `true`
- `AIMS4PT_WEB_ENABLE_TENSORFLOW_MODELS`, default `true`

For small-memory servers, disable optional R-backed or TensorFlow-backed models explicitly:

```bash
export AIMS4PT_WEB_ENABLE_R_MODELS=false
export AIMS4PT_WEB_ENABLE_TENSORFLOW_MODELS=false
```

Uploaded workbooks, generated reports, and calculation results are kept in memory only. Reports are generated on demand from the current session cache.
