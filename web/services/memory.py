"""Lightweight process memory logging."""

from __future__ import annotations

import logging

logger = logging.getLogger("web.memory")


def log_memory(label: str) -> None:
    """Log process RSS memory in MB when psutil is available."""
    try:
        import psutil

        process = psutil.Process()
        rss_mb = process.memory_info().rss / (1024 * 1024)
        logger.info("memory label=%s rss_mb=%.1f", label, rss_mb)
    except Exception:
        logger.info("memory label=%s rss_mb=unavailable", label)

