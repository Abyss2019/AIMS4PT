# AIMS4PT release packaging

This directory contains version-controlled release sources only. Generated
payloads, wheels, installer scripts, logs, and temporary archives belong under
`build/`; final release artifacts belong under `dist/`.

- `common/`: shared release metadata helpers.
- `regression/`: fixed cross-platform numerical reference and validator.
- `macos/`: Apple Silicon environment, installer builder, and smoke tests.
- `windows/`: Windows x64 installer builder and smoke tests.

Both platform builders:

1. read the product version from `pyproject.toml`;
2. record the exact build dependency manifest;
3. package a non-editable AIMS4PT release environment;
4. validate imports, resources, and web-service readiness;
5. run the same fixed numerical regression case;
6. produce checksummed release artifacts under `dist/`.
