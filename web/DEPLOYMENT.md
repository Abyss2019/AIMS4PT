# AIMS4PT_cpx Web Deployment

Use a clean server checkout as the deployment target. Server-specific settings should live in ignored `.env` files or systemd environment settings, not in tracked source code.

Recommended update sequence:

```bash
git fetch origin
git reset --hard origin/main
source .venv/bin/activate
pip install -e .
pip install -r requirements-web.txt
sudo systemctl restart aims4pt-web
```

Default single-worker command:

```bash
uvicorn web.main:app --host 127.0.0.1 --port 8000 --workers 1
```

For small-memory servers, reduce calculation concurrency without changing code:

```bash
export MAX_CONCURRENT_CALCULATIONS=1
```

R-backed and TensorFlow-backed models are disabled by default in the web service to keep the first deployment lightweight and stable. Enable them only on a server with the required runtime and memory headroom:

```bash
export AIMS4PT_WEB_ENABLE_R_MODELS=1
export AIMS4PT_WEB_ENABLE_TENSORFLOW_MODELS=1
```
