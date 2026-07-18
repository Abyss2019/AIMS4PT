"""Validate a packaged AIMS4PT Windows runtime."""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from importlib.metadata import version as distribution_version
from importlib.resources import files
from pathlib import Path


REQUIRED_MODULES = [
    "aims4pt_web",
    "aims4pt",
    "numpy",
    "pandas",
    "sklearn",
    "xgboost",
    "tensorflow",
    "keras",
    "rpy2",
    "onnxruntime",
]

REQUIRED_RESOURCES = [
    ("aims4pt.reporting.data", "independent_composition_kde.npz"),
    ("aims4pt.reporting.data", "independent_data_final.xlsx"),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--expected-prefix", required=True)
    return parser.parse_args()


def main() -> None:
    """Run version, location, import, resource, and model checks."""
    args = _parse_args()
    expected_prefix = Path(args.expected_prefix).resolve()
    if platform.system() != "Windows":
        raise RuntimeError(f"Expected Windows, found {platform.system()}.")

    installed_version = distribution_version("AIMS4PT")
    print(f"package_version {installed_version}")
    if installed_version != args.expected_version:
        raise RuntimeError(
            f"Expected AIMS4PT {args.expected_version}, "
            f"found {installed_version}."
        )
    if Path(sys.prefix).resolve() != expected_prefix:
        raise RuntimeError(
            f"sys.prefix is {sys.prefix}, expected {expected_prefix}."
        )

    import aims4pt

    package_path = Path(aims4pt.__file__).resolve()
    if not package_path.is_relative_to(expected_prefix):
        raise RuntimeError(
            f"AIMS4PT was imported from {package_path}, "
            f"outside {expected_prefix}."
        )

    for module_name in REQUIRED_MODULES:
        module = importlib.import_module(module_name)
        print(
            f"import_ok {module_name} "
            f"{getattr(module, '__file__', '<built-in>')}"
        )

    for package_name, resource_name in REQUIRED_RESOURCES:
        resource = files(package_name).joinpath(resource_name)
        print(
            f"resource_ok {package_name} {resource_name} "
            f"{resource.is_file()}"
        )
        if not resource.is_file():
            raise RuntimeError(
                f"Missing packaged resource: "
                f"{package_name}/{resource_name}"
            )

    import tensorflow as tf

    tensorflow_result = tf.reduce_sum(
        tf.constant([1.0, 2.0])
    ).numpy().item()
    if tensorflow_result != 3.0:
        raise RuntimeError(
            f"Unexpected TensorFlow result: {tensorflow_result}"
        )

    from aims4pt.model_tools.model_registry import import_all_models
    from aims4pt_web.main import app

    imported, failed = import_all_models(skip_failed=True)
    print(f"web_app_ok {app.title} {app.version}")
    print(f"model_registry_imported {len(imported)}")
    print(f"model_registry_failed {len(failed)}")
    if len(imported) != 8 or failed:
        raise RuntimeError(
            f"Expected all 8 model modules to import; "
            f"imported={imported}, failed={failed}."
        )
    if app.version != args.expected_version:
        raise RuntimeError(
            f"Expected web app version {args.expected_version}, "
            f"found {app.version}."
        )


if __name__ == "__main__":
    main()
