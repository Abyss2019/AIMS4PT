#!/usr/bin/env bash

set -euo pipefail

PRODUCT_NAME="AIMS4PT_cpx"
PKG_IDENTIFIER="org.aims4pt.cpx"
INSTALL_DIR="/Applications/AIMS4PT_cpx"
APP_BUNDLE="/Applications/AIMS4PT_cpx.app"
SPLIT_LIMIT_BYTES=$((1900 * 1024 * 1024))

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
DIST_DIR="${REPO_ROOT}/dist"

ENV_PREFIX="${CONDA_PREFIX:-}"
STAGE_ROOT="${TMPDIR:-/tmp}/A4PTm"
CLEANUP_ROOT=""
INSTALLER_SIGN_IDENTITY=""
APP_SIGN_IDENTITY="-"
SKIP_SMOKE_TEST=0
VERIFY_INSTALL=0
FORCE_SPLIT=0

usage() {
  cat <<EOF
Usage:
  bash packaging/macos/build_macos_installer.sh [options]

Options:
  --env-prefix PATH              Apple Silicon conda environment to package.
  --stage-root PATH              Temporary staging directory.
                                 Default: ${STAGE_ROOT}
  --cleanup-root PATH            Delete this ephemeral root after conda-pack.
                                 The environment must be inside this root.
  --installer-sign-identity NAME Optional Developer ID Installer identity.
  --app-sign-identity NAME       Optional Developer ID Application identity.
                                 Default: ad-hoc signing.
  --skip-smoke-test              Skip runtime and uvicorn smoke tests.
  --verify-install               Install the pkg on this machine and test it.
  --force-split                  Split the pkg into 1900 MiB parts.
  -h, --help                     Show this help.

The --cleanup-root option is intended for ephemeral CI runners with limited
disk space. The path must be under RUNNER_TEMP.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

write_step() {
  echo "==> $*"
}

abs_path() {
  local path="$1"
  local parent
  if [[ "${path}" = /* ]]; then
    parent="$(dirname -- "${path}")"
  else
    parent="$(pwd -P)/$(dirname -- "${path}")"
  fi
  mkdir -p -- "${parent}"
  printf '%s/%s\n' "$(cd -- "${parent}" && pwd -P)" "$(basename -- "${path}")"
}

path_is_inside() {
  local path="$1"
  local root="$2"
  local full_path
  local full_root
  full_path="$(abs_path "${path}")"
  full_root="$(abs_path "${root}")"
  [[ "${full_path}/" == "${full_root}/"* ]]
}

remove_directory_safe() {
  local path="$1"
  local root="$2"
  path_is_inside "${path}" "${root}" ||
    die "Refusing to remove '${path}' because it is outside '${root}'."
  rm -rf -- "${path}"
}

file_size_bytes() {
  stat -f%z "$1"
}

dir_size_bytes() {
  du -sk "$1" | awk '{ print $1 * 1024 }'
}

format_gib() {
  awk -v bytes="$1" 'BEGIN { printf "%.2f GiB", bytes / 1024 / 1024 / 1024 }'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-prefix)
      [[ $# -ge 2 ]] || die "--env-prefix requires a value"
      ENV_PREFIX="$2"
      shift 2
      ;;
    --stage-root)
      [[ $# -ge 2 ]] || die "--stage-root requires a value"
      STAGE_ROOT="$2"
      shift 2
      ;;
    --cleanup-root)
      [[ $# -ge 2 ]] || die "--cleanup-root requires a value"
      CLEANUP_ROOT="$2"
      shift 2
      ;;
    --installer-sign-identity)
      [[ $# -ge 2 ]] || die "--installer-sign-identity requires a value"
      INSTALLER_SIGN_IDENTITY="$2"
      shift 2
      ;;
    --app-sign-identity)
      [[ $# -ge 2 ]] || die "--app-sign-identity requires a value"
      APP_SIGN_IDENTITY="$2"
      shift 2
      ;;
    --skip-smoke-test)
      SKIP_SMOKE_TEST=1
      shift
      ;;
    --verify-install)
      VERIFY_INSTALL=1
      shift
      ;;
    --force-split)
      FORCE_SPLIT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

[[ "$(uname -s)" == "Darwin" ]] ||
  die "This script must be run on macOS."
[[ "$(uname -m)" == "arm64" ]] ||
  die "This script only builds Apple Silicon arm64 packages."
[[ -n "${ENV_PREFIX}" ]] ||
  die "--env-prefix is required when CONDA_PREFIX is not set."

ENV_PREFIX="$(abs_path "${ENV_PREFIX}")"
STAGE_ROOT="$(abs_path "${STAGE_ROOT}")"
[[ -d "${ENV_PREFIX}" ]] ||
  die "Environment prefix does not exist: ${ENV_PREFIX}"
[[ -x "${ENV_PREFIX}/bin/python" ]] ||
  die "Python was not found at ${ENV_PREFIX}/bin/python."

if [[ -n "${CLEANUP_ROOT}" ]]; then
  [[ -n "${RUNNER_TEMP:-}" ]] ||
    die "--cleanup-root is allowed only when RUNNER_TEMP is set."
  CLEANUP_ROOT="$(abs_path "${CLEANUP_ROOT}")"
  RUNNER_TEMP="$(abs_path "${RUNNER_TEMP}")"
  path_is_inside "${CLEANUP_ROOT}" "${RUNNER_TEMP}" ||
    die "--cleanup-root must be inside RUNNER_TEMP."
  path_is_inside "${ENV_PREFIX}" "${CLEANUP_ROOT}" ||
    die "The environment must be inside --cleanup-root."
  if path_is_inside "${STAGE_ROOT}" "${CLEANUP_ROOT}"; then
    die "The stage root must not be inside --cleanup-root."
  fi
  if path_is_inside "${REPO_ROOT}" "${CLEANUP_ROOT}"; then
    die "The repository must not be inside --cleanup-root."
  fi
fi

read_product_version() {
  "${ENV_PREFIX}/bin/python" - "${REPO_ROOT}/pyproject.toml" <<'PY'
from __future__ import annotations

import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(
    r'(?ms)^\[project\]\s*.*?^version\s*=\s*"(?P<version>[^"]+)"',
    text,
)
if match is None:
    raise SystemExit("Could not read [project].version from pyproject.toml.")
print(match.group("version"))
PY
}

PRODUCT_VERSION="$(read_product_version)"
[[ "${PRODUCT_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z]+)*$ ]] ||
  die "Unsupported product version: ${PRODUCT_VERSION}"

PACKAGE_BASE_NAME="${PRODUCT_NAME}-${PRODUCT_VERSION}-macOS-arm64"
PKG_PATH="${STAGE_ROOT}/${PACKAGE_BASE_NAME}.pkg"
DMG_PATH="${DIST_DIR}/${PACKAGE_BASE_NAME}.dmg"
CHECKSUM_PATH="${DIST_DIR}/${PACKAGE_BASE_NAME}-SHA256.txt"
PARTS_CHECKSUM_PATH="${DIST_DIR}/${PACKAGE_BASE_NAME}-PARTS-SHA256.txt"
REASSEMBLE_PATH="${DIST_DIR}/${PACKAGE_BASE_NAME}-reassemble.sh"
BUILD_INFO_PATH="${DIST_DIR}/${PACKAGE_BASE_NAME}-BUILD-INFO.txt"
DEPENDENCY_MANIFEST_PATH="${DIST_DIR}/${PACKAGE_BASE_NAME}-DEPENDENCIES.json"

ARCHIVE_PATH="${STAGE_ROOT}/env.tar.gz"
PKG_ROOT="${STAGE_ROOT}/pkg-root"
PKG_SCRIPTS_DIR="${STAGE_ROOT}/pkg-scripts"
DMG_ROOT="${STAGE_ROOT}/dmg-root"
STAGED_RUNTIME="${PKG_ROOT}${INSTALL_DIR}"
STAGED_APP="${PKG_ROOT}${APP_BUNDLE}"
SMOKE_TEST="${SCRIPT_DIR}/runtime_smoke.py"
REGRESSION_TEST="${REPO_ROOT}/packaging/regression/cross_platform_regression.py"
DEPENDENCY_MANIFEST_SCRIPT="${REPO_ROOT}/packaging/common/write_dependency_manifest.py"
SERVER_STDOUT="${STAGE_ROOT}/uvicorn-smoke.stdout.log"
SERVER_STDERR="${STAGE_ROOT}/uvicorn-smoke.stderr.log"

mkdir -p -- "${DIST_DIR}" "${STAGE_ROOT}"

find_conda_pack() {
  if [[ -x "${ENV_PREFIX}/bin/conda-pack" ]]; then
    printf '%s\n' "${ENV_PREFIX}/bin/conda-pack"
    return
  fi
  if command -v conda-pack >/dev/null 2>&1; then
    command -v conda-pack
    return
  fi
  die "conda-pack was not found in the release environment or PATH."
}

runtime_env_args() {
  local prefix="$1"
  printf '%s\0' \
    "CONDA_PREFIX=${prefix}" \
    "CONDA_DEFAULT_ENV=AIMS4PT_cpx" \
    "PYTHONNOUSERSITE=1" \
    "MPLBACKEND=Agg" \
    "PATH=${prefix}/bin:${prefix}/condabin:/usr/bin:/bin:/usr/sbin:/sbin" \
    "DYLD_FALLBACK_LIBRARY_PATH=${prefix}/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"

  if [[ -d "${prefix}/lib/R" ]]; then
    printf '%s\0' "R_HOME=${prefix}/lib/R"
  fi
  if [[ -x "${prefix}/bin/java" ]]; then
    printf '%s\0' "JAVA_HOME=${prefix}"
  fi
}

run_with_runtime() {
  local prefix="$1"
  shift
  local env_args=()
  while IFS= read -r -d '' item; do
    env_args+=("${item}")
  done < <(runtime_env_args "${prefix}")
  (
    cd -- "${HOME}"
    exec env "${env_args[@]}" "$@"
  )
}

run_runtime_smoke_test() {
  local prefix="$1"
  run_with_runtime \
    "${prefix}" \
    "${prefix}/bin/python" \
    "${SMOKE_TEST}" \
    --expected-version "${PRODUCT_VERSION}" \
    --expected-prefix "${prefix}"
}

run_numerical_regression_test() {
  local prefix="$1"
  run_with_runtime \
    "${prefix}" \
    "${prefix}/bin/python" \
    "${REGRESSION_TEST}"
}

run_uvicorn_smoke_test() {
  local prefix="$1"
  local python="${prefix}/bin/python"
  local port
  local url
  local pid
  local deadline

  port="$(
    run_with_runtime "${prefix}" "${python}" -c \
      'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
  )"
  url="http://127.0.0.1:${port}/"
  rm -f -- "${SERVER_STDOUT}" "${SERVER_STDERR}"

  run_with_runtime \
    "${prefix}" \
    "${python}" -m uvicorn aims4pt_web.main:app \
    --host 127.0.0.1 --port "${port}" --workers 1 \
    >"${SERVER_STDOUT}" 2>"${SERVER_STDERR}" &
  pid=$!
  deadline=$((SECONDS + 180))

  while (( SECONDS < deadline )); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      wait "${pid}" || true
      sed -n '1,200p' "${SERVER_STDERR}" >&2 || true
      die "uvicorn exited before readiness."
    fi
    if /usr/bin/curl --noproxy "*" -fsS --max-time 2 "${url}" >/dev/null 2>&1; then
      echo "uvicorn_ready ${url}"
      kill -TERM "${pid}" >/dev/null 2>&1 || true
      wait "${pid}" >/dev/null 2>&1 || true
      return
    fi
    sleep 0.5
  done

  kill -TERM "${pid}" >/dev/null 2>&1 || true
  wait "${pid}" >/dev/null 2>&1 || true
  sed -n '1,200p' "${SERVER_STDERR}" >&2 || true
  die "Timed out waiting for uvicorn at ${url}."
}

write_start_command() {
  local path="$1"
  cat > "${path}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

RUNTIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export CONDA_PREFIX="${RUNTIME_DIR}"
export CONDA_DEFAULT_ENV="AIMS4PT_cpx"
export PYTHONNOUSERSITE="1"
export MPLBACKEND="Agg"
export PATH="${RUNTIME_DIR}/bin:${RUNTIME_DIR}/condabin:/usr/bin:/bin:/usr/sbin:/sbin"
export DYLD_FALLBACK_LIBRARY_PATH="${RUNTIME_DIR}/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"

if [[ -d "${RUNTIME_DIR}/lib/R" ]]; then
  export R_HOME="${RUNTIME_DIR}/lib/R"
fi
if [[ -x "${RUNTIME_DIR}/bin/java" ]]; then
  export JAVA_HOME="${RUNTIME_DIR}"
fi

cd -- "${HOME}"
exec "${RUNTIME_DIR}/bin/python" -m aims4pt_web.launcher
EOF
  chmod +x "${path}"
}

write_app_bundle() {
  local app_path="$1"
  local executable="${app_path}/Contents/MacOS/${PRODUCT_NAME}"
  mkdir -p -- "${app_path}/Contents/MacOS" "${app_path}/Contents/Resources"

  cat > "${executable}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

RUNTIME_DIR="/Applications/AIMS4PT_cpx"
LOG_DIR="${HOME}/Library/Logs/AIMS4PT_cpx"
LOG_FILE="${LOG_DIR}/launcher.log"
mkdir -p -- "${LOG_DIR}"

show_error() {
  local message="$1"
  /usr/bin/osascript - "${message}" "${LOG_FILE}" <<'APPLESCRIPT' || true
on run argv
  display alert "AIMS4PT_cpx" message ((item 1 of argv) & return & "Log: " & (item 2 of argv)) as critical
end run
APPLESCRIPT
}

if [[ ! -x "${RUNTIME_DIR}/bin/python" ]]; then
  show_error "The AIMS4PT runtime is missing. Reinstall the application."
  exit 1
fi

export CONDA_PREFIX="${RUNTIME_DIR}"
export CONDA_DEFAULT_ENV="AIMS4PT_cpx"
export PYTHONNOUSERSITE="1"
export MPLBACKEND="Agg"
export PATH="${RUNTIME_DIR}/bin:${RUNTIME_DIR}/condabin:/usr/bin:/bin:/usr/sbin:/sbin"
export DYLD_FALLBACK_LIBRARY_PATH="${RUNTIME_DIR}/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"

if [[ -d "${RUNTIME_DIR}/lib/R" ]]; then
  export R_HOME="${RUNTIME_DIR}/lib/R"
fi
if [[ -x "${RUNTIME_DIR}/bin/java" ]]; then
  export JAVA_HOME="${RUNTIME_DIR}"
fi

cd -- "${HOME}"
"${RUNTIME_DIR}/bin/python" -m aims4pt_web.launcher >>"${LOG_FILE}" 2>&1 &
child_pid=$!

terminate_child() {
  if kill -0 "${child_pid}" >/dev/null 2>&1; then
    kill -TERM "${child_pid}" >/dev/null 2>&1 || true
    wait "${child_pid}" >/dev/null 2>&1 || true
  fi
}
trap terminate_child TERM INT HUP

set +e
wait "${child_pid}"
status=$?
set -e
trap - TERM INT HUP

if [[ "${status}" -ne 0 && "${status}" -ne 130 && "${status}" -ne 143 ]]; then
  show_error "AIMS4PT_cpx could not start."
fi
exit "${status}"
EOF
  chmod +x "${executable}"

  cat > "${app_path}/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleDisplayName</key>
  <string>${PRODUCT_NAME}</string>
  <key>CFBundleExecutable</key>
  <string>${PRODUCT_NAME}</string>
  <key>CFBundleIdentifier</key>
  <string>${PKG_IDENTIFIER}</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>${PRODUCT_NAME}</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>${PRODUCT_VERSION}</string>
  <key>CFBundleVersion</key>
  <string>${PRODUCT_VERSION}</string>
  <key>LSMinimumSystemVersion</key>
  <string>13.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
EOF

  plutil -lint "${app_path}/Contents/Info.plist"

  local codesign_args=(--force --deep --sign "${APP_SIGN_IDENTITY}")
  if [[ "${APP_SIGN_IDENTITY}" != "-" ]]; then
    codesign_args+=(--timestamp --options runtime)
  fi
  codesign "${codesign_args[@]}" "${app_path}"
  codesign --verify --deep --strict --verbose=2 "${app_path}"
}

write_pkg_scripts() {
  remove_directory_safe "${PKG_SCRIPTS_DIR}" "${STAGE_ROOT}"
  mkdir -p -- "${PKG_SCRIPTS_DIR}"

  cat > "${PKG_SCRIPTS_DIR}/preinstall" <<EOF
#!/bin/bash
set -e

rm -rf -- "${INSTALL_DIR}" "${APP_BUNDLE}"
exit 0
EOF

  cat > "${PKG_SCRIPTS_DIR}/postinstall" <<EOF
#!/bin/bash
set -e

RUNTIME_DIR="${INSTALL_DIR}"
export CONDA_PREFIX="\${RUNTIME_DIR}"
export CONDA_DEFAULT_ENV="AIMS4PT_cpx"
export PYTHONNOUSERSITE="1"
export PATH="\${RUNTIME_DIR}/bin:\${RUNTIME_DIR}/condabin:/usr/bin:/bin:/usr/sbin:/sbin"
export DYLD_FALLBACK_LIBRARY_PATH="\${RUNTIME_DIR}/lib:\${DYLD_FALLBACK_LIBRARY_PATH:-}"

if [ -d "\${RUNTIME_DIR}/lib/R" ]; then
  export R_HOME="\${RUNTIME_DIR}/lib/R"
fi

if [ -x "\${RUNTIME_DIR}/bin/conda-unpack" ]; then
  "\${RUNTIME_DIR}/bin/conda-unpack"
fi

chmod +x "\${RUNTIME_DIR}/AIMS4PT_cpx.command"
chmod +x "${APP_BUNDLE}/Contents/MacOS/${PRODUCT_NAME}"
exit 0
EOF

  chmod +x "${PKG_SCRIPTS_DIR}/preinstall" "${PKG_SCRIPTS_DIR}/postinstall"
}

write_step "Product version: ${PRODUCT_VERSION}"
write_step "Release environment: ${ENV_PREFIX}"

if [[ "${SKIP_SMOKE_TEST}" -eq 0 ]]; then
  write_step "Checking release environment dependencies"
  run_with_runtime "${ENV_PREFIX}" "${ENV_PREFIX}/bin/python" -m pip check

  write_step "Running release environment smoke test"
  run_runtime_smoke_test "${ENV_PREFIX}"

  write_step "Running release environment numerical regression test"
  run_numerical_regression_test "${ENV_PREFIX}"

  write_step "Running release environment uvicorn smoke test"
  run_uvicorn_smoke_test "${ENV_PREFIX}"
fi

rm -f -- "${DEPENDENCY_MANIFEST_PATH}"
write_step "Recording the exact macOS release dependency manifest"
run_with_runtime \
  "${ENV_PREFIX}" \
  "${ENV_PREFIX}/bin/python" \
  "${DEPENDENCY_MANIFEST_SCRIPT}" \
  --output "${DEPENDENCY_MANIFEST_PATH}" \
  --prefix "${ENV_PREFIX}" \
  --label "${PRODUCT_NAME} ${PRODUCT_VERSION} macOS arm64"

CONDA_PACK="$(find_conda_pack)"
remove_directory_safe "${PKG_ROOT}" "${STAGE_ROOT}"
rm -f -- "${ARCHIVE_PATH}"

write_step "Creating temporary conda-pack archive"
exclude_patterns=(
  "**/__pycache__/**"
  "**/*.pyc"
  "**/*.pyo"
  "**/*.pdb"
  "include/**"
  "lib/**/*.a"
  "lib/**/*.la"
  "lib/cmake/**"
  "lib/pkgconfig/**"
  "lib/python*/site-packages/tensorflow/include/**"
  "lib/python*/site-packages/debugpy/**"
  "lib/python*/site-packages/debugpy-*.dist-info/**"
  "lib/python*/site-packages/ipykernel/**"
  "lib/python*/site-packages/ipykernel-*.dist-info/**"
  "lib/python*/site-packages/jedi/**"
  "lib/python*/site-packages/jedi-*.dist-info/**"
  "lib/python*/site-packages/sphinx/**"
  "lib/python*/site-packages/sphinx-*.dist-info/**"
  "lib/python*/site-packages/PyQt5/**"
  "lib/python*/site-packages/PyQt5-*.dist-info/**"
  "lib/python*/site-packages/qtpy/**"
  "lib/python*/site-packages/qtpy-*.dist-info/**"
)

pack_args=(
  --prefix "${ENV_PREFIX}"
  --output "${ARCHIVE_PATH}"
  --dest-prefix "${INSTALL_DIR}"
  --format tar.gz
  --compress-level 1
  --force
  --ignore-editable-packages
)
for pattern in "${exclude_patterns[@]}"; do
  pack_args+=(--exclude "${pattern}")
done

PYTHONUTF8=1 PYTHONIOENCODING=utf-8 "${CONDA_PACK}" "${pack_args[@]}"
archive_size="$(file_size_bytes "${ARCHIVE_PATH}")"
write_step "Temporary archive size: $(format_gib "${archive_size}")"

if [[ -n "${CLEANUP_ROOT}" ]]; then
  write_step "Removing ephemeral environment root to free disk space"
  remove_directory_safe "${CLEANUP_ROOT}" "${RUNNER_TEMP}"
  hash -r
fi

write_step "Creating installer payload"
mkdir -p -- "${STAGED_RUNTIME}"
tar -xzf "${ARCHIVE_PATH}" -C "${STAGED_RUNTIME}"
rm -f -- "${ARCHIVE_PATH}"

[[ -x "${STAGED_RUNTIME}/bin/python" ]] ||
  die "Staged runtime is missing bin/python."

write_start_command "${STAGED_RUNTIME}/AIMS4PT_cpx.command"
write_app_bundle "${STAGED_APP}"
write_pkg_scripts

payload_size="$(dir_size_bytes "${PKG_ROOT}")"
write_step "Installer payload size: $(format_gib "${payload_size}")"

write_step "Building macOS installer package"
rm -f -- \
  "${PKG_PATH}" \
  "${DMG_PATH}" \
  "${CHECKSUM_PATH}" \
  "${PARTS_CHECKSUM_PATH}" \
  "${REASSEMBLE_PATH}" \
  "${BUILD_INFO_PATH}" \
  "${DMG_PATH}.part-"*

pkgbuild_args=(
  --root "${PKG_ROOT}"
  --install-location "/"
  --identifier "${PKG_IDENTIFIER}"
  --version "${PRODUCT_VERSION}"
  --ownership recommended
  --scripts "${PKG_SCRIPTS_DIR}"
)
if [[ -n "${INSTALLER_SIGN_IDENTITY}" ]]; then
  pkgbuild_args+=(--sign "${INSTALLER_SIGN_IDENTITY}")
fi

pkgbuild "${pkgbuild_args[@]}" "${PKG_PATH}"

pkgutil --payload-files "${PKG_PATH}" |
  grep -F "Applications/AIMS4PT_cpx/bin/python" >/dev/null ||
  die "The pkg does not contain the Python runtime."
pkgutil --payload-files "${PKG_PATH}" |
  grep -F "Applications/AIMS4PT_cpx.app/Contents/MacOS/AIMS4PT_cpx" >/dev/null ||
  die "The pkg does not contain the application launcher."

if [[ "${VERIFY_INSTALL}" -eq 1 ]]; then
  write_step "Removing staging payload before installation verification"
  remove_directory_safe "${PKG_ROOT}" "${STAGE_ROOT}"

  write_step "Installing pkg on the ephemeral build machine"
  sudo /usr/sbin/installer -pkg "${PKG_PATH}" -target /

  [[ -x "${INSTALL_DIR}/bin/python" ]] ||
    die "Installed runtime is missing bin/python."
  [[ -x "${APP_BUNDLE}/Contents/MacOS/${PRODUCT_NAME}" ]] ||
    die "Installed application launcher is missing."

  codesign --verify --deep --strict --verbose=2 "${APP_BUNDLE}"
  installed_version="$(
    defaults read "${APP_BUNDLE}/Contents/Info" CFBundleShortVersionString
  )"
  [[ "${installed_version}" == "${PRODUCT_VERSION}" ]] ||
    die "Installed app version is ${installed_version}, expected ${PRODUCT_VERSION}."

  if [[ "${SKIP_SMOKE_TEST}" -eq 0 ]]; then
    write_step "Running installed runtime smoke test"
    run_runtime_smoke_test "${INSTALL_DIR}"

    write_step "Running installed numerical regression test"
    run_numerical_regression_test "${INSTALL_DIR}"

    write_step "Running installed uvicorn smoke test"
    run_uvicorn_smoke_test "${INSTALL_DIR}"
  fi

  write_step "Cleaning verified installation from the ephemeral build machine"
  sudo rm -rf -- "${INSTALL_DIR}" "${APP_BUNDLE}"
fi

write_step "Creating macOS disk image"
remove_directory_safe "${DMG_ROOT}" "${STAGE_ROOT}"
mkdir -p -- "${DMG_ROOT}"
cp -p -- "${PKG_PATH}" "${DMG_ROOT}/Install ${PRODUCT_NAME}.pkg"
cat > "${DMG_ROOT}/Read Me.txt" <<EOF
${PRODUCT_NAME} ${PRODUCT_VERSION} for Apple Silicon Macs

1. Double-click "Install ${PRODUCT_NAME}.pkg".
2. Complete the macOS Installer steps.
3. Open ${PRODUCT_NAME} from the Applications folder.

This test build is not notarized. If macOS blocks it, Control-click the
installer, choose Open, and confirm that you trust this package.
EOF

hdiutil create \
  -volname "${PRODUCT_NAME} ${PRODUCT_VERSION}" \
  -srcfolder "${DMG_ROOT}" \
  -format UDZO \
  -ov \
  "${DMG_PATH}"
hdiutil verify "${DMG_PATH}"

if [[ "${APP_SIGN_IDENTITY}" != "-" ]]; then
  codesign --force --timestamp --sign "${APP_SIGN_IDENTITY}" "${DMG_PATH}"
  codesign --verify --verbose=2 "${DMG_PATH}"
fi

remove_directory_safe "${DMG_ROOT}" "${STAGE_ROOT}"
rm -f -- "${PKG_PATH}"

(
  cd -- "${DIST_DIR}"
  shasum -a 256 "$(basename -- "${DMG_PATH}")" > "$(basename -- "${CHECKSUM_PATH}")"
)

dmg_size="$(file_size_bytes "${DMG_PATH}")"
write_step "Disk image size: $(format_gib "${dmg_size}")"

cat > "${BUILD_INFO_PATH}" <<EOF
ProductName=${PRODUCT_NAME}
ProductVersion=${PRODUCT_VERSION}
Architecture=arm64
MinimumMacOS=13.0
PackageIdentifier=${PKG_IDENTIFIER}
InstallerInsideDMG=yes
InstallerSigned=$([[ -n "${INSTALLER_SIGN_IDENTITY}" ]] && echo yes || echo no)
ApplicationSigningIdentity=${APP_SIGN_IDENTITY}
Notarized=no
DependencyManifest=$(basename -- "${DEPENDENCY_MANIFEST_PATH}")
NumericalRegressionReference=$(basename -- "${REPO_ROOT}/packaging/regression/cpx_only_temperature_reference.json")
EOF

if [[ "${FORCE_SPLIT}" -eq 1 || "${dmg_size}" -gt "${SPLIT_LIMIT_BYTES}" ]]; then
  write_step "Splitting disk image into 1900 MiB parts"
  split -b 1900m "${DMG_PATH}" "${DMG_PATH}.part-"
  (
    cd -- "${DIST_DIR}"
    shasum -a 256 "$(basename -- "${DMG_PATH}")".part-* \
      > "$(basename -- "${PARTS_CHECKSUM_PATH}")"
  )

  dmg_name="$(basename -- "${DMG_PATH}")"
  checksum_name="$(basename -- "${CHECKSUM_PATH}")"
  cat > "${REASSEMBLE_PATH}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd -- "\$(dirname -- "\${BASH_SOURCE[0]}")"
cat "${dmg_name}.part-"* > "${dmg_name}"
shasum -a 256 -c "${checksum_name}"
EOF
  chmod +x "${REASSEMBLE_PATH}"
  rm -f -- "${DMG_PATH}"
fi

write_step "Release artifacts"
for artifact in "${DIST_DIR}/${PACKAGE_BASE_NAME}"*; do
  [[ -e "${artifact}" ]] || continue
  ls -lh "${artifact}"
done
