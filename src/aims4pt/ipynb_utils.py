"""Helpers for persisting and restoring notebook state with ``dill``."""

from __future__ import annotations

import os
from typing import Dict



def find_root_path(base_name: str = "Database") -> str:
    """Find the project root path by climbing parent directories until ``base_name``."""

    current_path = os.getcwd()
    while True:
        if os.path.basename(current_path) == base_name:
            return current_path
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            raise FileNotFoundError("Root path not found.")
        current_path = parent_path



