# macOS arm64 packaging

This directory builds the Apple Silicon edition of AIMS4PT_cpx on a real
macOS runner. Windows cannot produce or validate the native Python, R,
TensorFlow, XGBoost, and ONNX Runtime binaries used by this package.

## GitHub Actions

The workflow at `.github/workflows/build-macos-arm64.yml` runs on the native
arm64 `macos-15` image. It can be started manually, by pushing the
`codex/macos-arm64` branch, or by pushing a version tag such as `v0.2.0`.

The workflow:

1. creates the release environment;
2. installs AIMS4PT non-editably;
3. validates Python, TensorFlow, R, model imports, resources, and uvicorn;
4. compares representative analytical, ML, ONNX, and TensorFlow predictions
   with the shared fixed cross-platform reference case;
5. records the exact conda and Python dependency manifest;
6. packages the relocatable environment into a macOS installer;
7. installs the generated pkg on the ephemeral runner and repeats the checks;
8. wraps the verified pkg in a disk image;
9. uploads the dmg, dependency manifest, and SHA-256 metadata as a workflow
   artifact.

## Output

The normal output is:

`AIMS4PT_cpx-<version>-macOS-arm64.dmg`

Packages larger than 1900 MiB are uploaded as numbered parts together with a
checksum file and a reassembly script.

## Signing

The default CI output uses ad-hoc signing for the small `.app` launcher and
does not sign or notarize the pkg inside the dmg. It is suitable for internal
testing, but macOS may require Control-click > Open. Public distribution
should use an Apple Developer ID Application certificate, a Developer ID
Installer certificate, and Apple notarization.

## Local Apple Silicon build

After creating the environment from `environment.macos-arm64.yml` and
installing the project non-editably:

```bash
bash packaging/macos/build_macos_installer.sh \
  --env-prefix "${CONDA_PREFIX}"
```

Use `--help` for signing and verification options.
