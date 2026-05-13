"""Helpers for configuring the R runtime before importing :mod:`rpy2`."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

PACKAGE_R_HOME_ENV = "MY_ANALYSIS_TOOLS_R_HOME"
PACKAGE_CONFIG_ENV = "MY_ANALYSIS_TOOLS_CONFIG"
DEFAULT_LANG = "en_US.UTF-8"
DEFAULT_RENCODING = "UTF-8"
REQUIRED_R_VERSION = (4, 4, 3)


def default_config_path() -> Path:
    """Return the package-level config file used for R settings."""
    configured = os.environ.get(PACKAGE_CONFIG_ENV)
    if configured:
        return Path(os.path.expandvars(configured)).expanduser()
    return Path.home() / ".aims4pt" / "config.json"


def load_user_config(config_path: Optional[str | Path] = None) -> dict[str, Any]:
    """Load the user config file if it exists and contains a JSON object."""
    if config_path is None:
        path = default_config_path()
    else:
        path = Path(os.path.expandvars(str(config_path))).expanduser()
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    return data if isinstance(data, dict) else {}


def save_r_home_config(r_home: str | Path, config_path: Optional[str | Path] = None) -> Path:
    """Persist ``R_HOME`` to the package-level config file."""
    normalized = _normalize_r_home(r_home)
    if normalized is None or not _looks_like_r_home(normalized):
        raise ValueError(f"Invalid R_HOME path: {r_home!s}")

    if config_path is None:
        path = default_config_path()
    else:
        path = Path(os.path.expandvars(str(config_path))).expanduser()
    config = load_user_config(path)
    config["R_HOME"] = str(normalized)
    config.setdefault("LANG", DEFAULT_LANG)
    config.setdefault("RENCODING", DEFAULT_RENCODING)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def configure_r_environment(r_home: Optional[str | Path] = None) -> Optional[str]:
    """Best-effort R runtime discovery used before importing :mod:`rpy2`."""
    config = load_user_config()
    configured_home: Optional[Path]

    if r_home is not None:
        configured_home = _normalize_r_home(r_home)
        if configured_home is None or not _is_supported_r_home(configured_home):
            raise ValueError(f"Invalid R_HOME path: {r_home!s}")
    else:
        configured_home = discover_r_home(config)

    if configured_home is not None:
        os.environ["R_HOME"] = str(configured_home)
        _ensure_r_runtime_on_path(configured_home)

    os.environ.setdefault("RENCODING", str(config.get("RENCODING", DEFAULT_RENCODING)))
    os.environ.setdefault("LANG", str(config.get("LANG", DEFAULT_LANG)))
    return os.environ.get("R_HOME")


def discover_r_home(config: Optional[dict[str, Any]] = None) -> Optional[Path]:
    """Locate a plausible R home directory."""
    config = {} if config is None else config

    configured_candidates = [
        os.environ.get("R_HOME"),
        os.environ.get(PACKAGE_R_HOME_ENV),
        config.get("R_HOME"),
        config.get("r_home"),
        config.get(PACKAGE_R_HOME_ENV),
    ]
    for raw_candidate in configured_candidates:
        candidate = _normalize_r_home(raw_candidate)
        if _is_supported_r_home(candidate):
            return candidate

    for raw_candidate in _conda_r_home_candidates():
        candidate = _normalize_r_home(raw_candidate)
        if _is_supported_r_home(candidate):
            return candidate

    for raw_candidate in _r_home_from_commands():
        candidate = _normalize_r_home(raw_candidate)
        if _is_supported_r_home(candidate):
            return candidate

    if platform.system() == "Windows":
        for raw_candidate in _windows_registry_candidates():
            candidate = _normalize_r_home(raw_candidate)
            if _is_supported_r_home(candidate):
                return candidate

    for raw_candidate in _well_known_install_locations():
        candidate = _normalize_r_home(raw_candidate)
        if _is_supported_r_home(candidate):
            return candidate

    return None


def format_rpy2_setup_error(exc: BaseException) -> str:
    """Return a compact error message with concrete next steps."""
    config_path = default_config_path()
    return (
        "R-based models require both a working R installation and the optional "
        "Python dependency 'rpy2'. Automatic discovery checks R_HOME, "
        f"{PACKAGE_R_HOME_ENV}, the active conda environment, {config_path}, "
        "the R/Rscript commands on PATH, and common install locations. "
        "AIMS4PT's R-backed models are tested with R 4.4.3; other R homes "
        "are ignored because they are not ABI-compatible with the pinned rpy2. "
        "If discovery fails, set R_HOME, set "
        f"{PACKAGE_R_HOME_ENV}, or create {config_path} with "
        '{"R_HOME": "path/to/R"}. '
        f"Original error: {exc}"
    )


def _normalize_r_home(raw_path: Optional[str | Path]) -> Optional[Path]:
    if raw_path is None:
        return None

    path_text = os.path.expandvars(str(raw_path).strip().strip('"').strip("'"))
    if not path_text:
        return None

    path = Path(path_text).expanduser()

    if path.is_file():
        name = path.name.lower()
        if name in {"r", "r.exe", "rscript", "rscript.exe"}:
            path = path.parent

    if path.name.lower() in {"x64", "i386"} and path.parent.name.lower() == "bin":
        path = path.parent.parent
    elif path.name.lower() == "bin":
        path = path.parent

    return path


def _looks_like_r_home(path: Optional[Path]) -> bool:
    if path is None or not path.exists():
        return False

    markers = (
        path / "bin" / "R",
        path / "bin" / "R.exe",
        path / "bin" / "Rscript",
        path / "bin" / "Rscript.exe",
        path / "lib" / "libR.so",
        path / "lib" / "libR.dylib",
    )
    return any(marker.exists() for marker in markers)


def _is_supported_r_home(path: Optional[Path]) -> bool:
    if not _looks_like_r_home(path):
        return False

    version = _r_home_version(path)
    if version is None:
        return True
    return version[:3] == REQUIRED_R_VERSION


def _r_home_version(path: Optional[Path]) -> Optional[tuple[int, ...]]:
    if path is None:
        return None

    path_version = _parse_version_text(str(path))
    if path_version is not None:
        return path_version

    rscript = path / "bin" / ("Rscript.exe" if platform.system() == "Windows" else "Rscript")
    if not rscript.exists():
        return None

    try:
        completed = subprocess.run(
            [str(rscript), "--version"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    return _parse_version_text(f"{completed.stdout}\n{completed.stderr}")


def _parse_version_text(text: str) -> Optional[tuple[int, ...]]:
    match = re.search(r"(?<!\d)(\d+)\.(\d+)(?:\.(\d+))?", text)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups(default="0"))


def _conda_r_home_candidates() -> list[Path]:
    candidates: list[Path] = []
    prefixes = [
        os.environ.get("CONDA_PREFIX"),
        sys.prefix,
        sys.base_prefix,
    ]

    for raw_prefix in prefixes:
        if not raw_prefix:
            continue
        prefix = Path(raw_prefix)
        candidates.extend([prefix / "Lib" / "R", prefix / "lib" / "R"])

    return _dedupe_paths(candidates)


def _r_home_from_commands() -> list[str]:
    commands = (
        ["R", "RHOME"],
        ["Rscript", "-e", "cat(R.home())"],
    )
    discovered: list[str] = []
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            continue

        if completed.returncode != 0:
            continue

        output_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not output_lines and completed.stdout.strip():
            output_lines = [completed.stdout.strip()]
        if output_lines:
            discovered.append(output_lines[-1])
    return discovered


def _windows_registry_candidates() -> list[str]:
    try:
        import winreg
    except ImportError:
        return []

    candidates: list[str] = []
    roots = (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE)
    key_paths = (
        r"SOFTWARE\R-core\R",
        r"SOFTWARE\WOW6432Node\R-core\R",
    )

    for root in roots:
        for key_path in key_paths:
            try:
                with winreg.OpenKey(root, key_path) as key:
                    candidates.extend(_read_registry_install_paths(winreg, key))
            except OSError:
                continue

    return _dedupe_strings(candidates)


def _read_registry_install_paths(winreg_module: Any, key: Any) -> list[str]:
    install_paths: list[str] = []

    try:
        current_version, _ = winreg_module.QueryValueEx(key, "Current Version")
    except OSError:
        current_version = None

    if current_version:
        try:
            with winreg_module.OpenKey(key, str(current_version)) as version_key:
                install_path, _ = winreg_module.QueryValueEx(version_key, "InstallPath")
                install_paths.append(install_path)
        except OSError:
            pass

    index = 0
    while True:
        try:
            subkey_name = winreg_module.EnumKey(key, index)
        except OSError:
            break

        index += 1
        try:
            with winreg_module.OpenKey(key, subkey_name) as version_key:
                install_path, _ = winreg_module.QueryValueEx(version_key, "InstallPath")
                install_paths.append(install_path)
        except OSError:
            continue

    return install_paths


def _well_known_install_locations() -> list[Path]:
    candidates: list[Path] = []
    system = platform.system()

    if system == "Windows":
        root_env_vars = ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")
        for env_var in root_env_vars:
            root_value = os.environ.get(env_var)
            if not root_value:
                continue

            root_path = Path(root_value)
            direct_r_root = root_path / "R"
            if direct_r_root.exists():
                candidates.extend(direct_r_root.glob("R-*"))

            programs_r_root = root_path / "Programs" / "R"
            if programs_r_root.exists():
                candidates.extend(programs_r_root.glob("R-*"))

    elif system == "Darwin":
        candidates.append(Path("/Library/Frameworks/R.framework/Resources"))
        for cellar_root in (Path("/opt/homebrew/Cellar/r"), Path("/usr/local/Cellar/r")):
            if cellar_root.exists():
                candidates.extend(cellar_root.glob("*/lib/R"))

    else:
        candidates.extend(
            [
                Path("/usr/lib/R"),
                Path("/usr/local/lib/R"),
                Path("/opt/R"),
            ]
        )
        opt_r_root = Path("/opt/R")
        if opt_r_root.exists():
            candidates.extend(opt_r_root.glob("*/lib/R"))

    return _sort_versioned_paths(_dedupe_paths(candidates))


def _sort_versioned_paths(paths: list[Path]) -> list[Path]:
    def version_key(path: Path) -> tuple[int, ...]:
        return tuple(int(part) for part in re.findall(r"\d+", str(path)))

    return sorted(paths, key=version_key, reverse=True)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path).lower() if platform.system() == "Windows" else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower() if platform.system() == "Windows" else value
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _ensure_r_runtime_on_path(r_home: Path) -> None:
    if platform.system() != "Windows":
        return

    path_entries = [r_home / "bin" / "x64", r_home / "bin"]
    current_path = os.environ.get("PATH", "")
    existing = current_path.split(os.pathsep) if current_path else []

    for entry in reversed(path_entries):
        if not entry.exists():
            continue
        entry_text = str(entry)
        if entry_text not in existing:
            existing.insert(0, entry_text)

    os.environ["PATH"] = os.pathsep.join(existing)
