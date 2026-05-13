"""Cache helpers shared by paper notebooks."""

from __future__ import annotations

import hashlib
import pickle
import re
from pathlib import Path

import pandas as pd


def _find_project_root(start: Path | None = None) -> Path:
    """Find the repository root by walking upward from a starting path."""
    path = Path.cwd() if start is None else Path(start).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not find pyproject.toml above the current directory.")


CACHE_DIR = _find_project_root() / "paper" / ".cache"


def configure_cache_dir(cache_dir: str | Path) -> None:
    """Set the cache directory used by notebook helper functions."""
    global CACHE_DIR
    CACHE_DIR = Path(cache_dir)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _fast_df_fingerprint(df: pd.DataFrame, cols=None) -> str:
    """Return a stable short fingerprint for selected DataFrame content."""
    if df is None:
        return "none"
    data = df if cols is None else df.loc[:, cols]
    hashed = pd.util.hash_pandas_object(data, index=True).values
    return hashlib.md5(hashed.tobytes()).hexdigest()[:12]


def _cache_path(prefix: str, key: str) -> Path:
    """Build a filesystem-safe cache path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix)
    return CACHE_DIR / f"{safe}__{key}.pkl"


def cache_load(prefix: str, key: str):
    """Load a cached pickle object, returning None when it is absent."""
    path = _cache_path(prefix, key)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def cache_save(prefix: str, key: str, obj):
    """Persist a pickle object in the configured cache directory."""
    path = _cache_path(prefix, key)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)
    return path
