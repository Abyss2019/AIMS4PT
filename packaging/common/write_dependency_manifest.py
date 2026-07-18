"""Write an exact dependency manifest for a packaged AIMS4PT runtime."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import distributions
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix", default=sys.prefix)
    parser.add_argument("--label", default="AIMS4PT release runtime")
    return parser.parse_args()


def _read_conda_packages(prefix: Path) -> list[dict[str, str]]:
    packages: list[dict[str, str]] = []
    metadata_dir = prefix / "conda-meta"
    if not metadata_dir.is_dir():
        return packages

    for metadata_path in sorted(metadata_dir.glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        packages.append(
            {
                "name": str(metadata.get("name", "")),
                "version": str(metadata.get("version", "")),
                "build": str(metadata.get("build", "")),
                "channel": str(metadata.get("channel", "")),
                "subdir": str(metadata.get("subdir", "")),
            }
        )
    return packages


def _read_python_distributions() -> list[dict[str, str]]:
    packages = {
        (
            distribution.metadata.get("Name", distribution.name),
            distribution.version,
        )
        for distribution in distributions()
    }
    return [
        {"name": name, "version": version}
        for name, version in sorted(
            packages,
            key=lambda item: item[0].lower(),
        )
    ]


def main() -> None:
    """Record platform, conda package, and Python distribution versions."""
    args = _parse_args()
    prefix = Path(args.prefix).resolve()
    output = Path(args.output).resolve()
    manifest = {
        "schema_version": 1,
        "label": args.label,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        },
        "prefix_at_build_time": str(prefix),
        "conda_packages": _read_conda_packages(prefix),
        "python_distributions": _read_python_distributions(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "dependency_manifest_written "
        f"{output} conda={len(manifest['conda_packages'])} "
        f"python={len(manifest['python_distributions'])}"
    )


if __name__ == "__main__":
    main()
