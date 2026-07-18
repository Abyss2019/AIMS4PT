# Windows x64 packaging

This directory contains the reusable Windows installer sources. Generated
payloads, Inno Setup files, logs, wheels, and archives belong under `build/`
or `dist/` and are intentionally ignored by Git.

The release environment must already contain a non-editable AIMS4PT install,
the web runtime, native Python/R dependencies, and `conda-pack`.

Run from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File ".\packaging\windows\build_windows_installer.ps1" `
  -EnvPrefix "C:\Users\13493\miniconda3\envs\AIMS4PT_release"
```

The script:

1. records the exact release dependency manifest;
2. creates a relocatable, trimmed conda payload;
3. validates package imports and resources;
4. runs the shared cross-platform numerical regression case;
5. checks uvicorn readiness;
6. builds the x64 installer with Inno Setup;
7. writes SHA-256 metadata under `dist/`.
