param(
    [string]$EnvPrefix = "C:\Users\13493\miniconda3\envs\AIMS4PT_release",
    [string]$ProductVersion = "",
    [string]$IsccPath = "",
    [string]$StageRoot = "",
    [switch]$SkipStage,
    [switch]$SkipSmokeTest,
    [switch]$ForceDiskSpanning
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProductName = "AIMS4PT_cpx"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
if (-not $ProductVersion) {
    $pyprojectPath = Join-Path $RepoRoot "pyproject.toml"
    $pyprojectText = Get-Content -Raw -LiteralPath $pyprojectPath
    $versionMatch = [regex]::Match(
        $pyprojectText,
        '(?ms)^\[project\]\s*.*?^version\s*=\s*"(?<version>[^"]+)"'
    )
    if (-not $versionMatch.Success) {
        throw "Could not read [project].version from $pyprojectPath"
    }
    $ProductVersion = $versionMatch.Groups["version"].Value
}
$SetupBaseName = "$ProductName-$ProductVersion-Windows-x64-Setup"
$BuildRoot = Join-Path $RepoRoot "build\installer"
if (-not $StageRoot) {
    # Keep staging paths short to avoid Inno Setup MAX_PATH failures.
    $StageRoot = Join-Path $env:TEMP "A4PTi"
}
$PayloadDir = Join-Path $StageRoot "p"
$SlimZipPath = Join-Path $StageRoot "env.zip"
$DistDir = Join-Path $RepoRoot "dist"
$SmokeTestPath = Join-Path $PSScriptRoot "runtime_smoke.py"
$RegressionTestPath = Join-Path $RepoRoot "packaging\regression\cross_platform_regression.py"
$DependencyManifestScript = Join-Path $RepoRoot "packaging\common\write_dependency_manifest.py"
$DependencyManifestPath = Join-Path $DistDir "$ProductName-$ProductVersion-Windows-x64-DEPENDENCIES.json"
$IssPath = Join-Path $BuildRoot "$ProductName.iss"
$ServerStdout = Join-Path $BuildRoot "uvicorn-smoke.stdout.log"
$ServerStderr = Join-Path $BuildRoot "uvicorn-smoke.stderr.log"

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message"
}

function Get-FullPath {
    param([string]$Path)
    return [System.IO.Path]::GetFullPath($Path)
}

function Assert-PathInside {
    param(
        [string]$Path,
        [string]$Root
    )
    $fullPath = Get-FullPath $Path
    $fullRoot = (Get-FullPath $Root).TrimEnd('\') + '\'
    if (-not $fullPath.StartsWith($fullRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to operate outside expected root. Path='$fullPath' Root='$fullRoot'"
    }
}

function Remove-DirectorySafe {
    param(
        [string]$Path,
        [string]$Root
    )
    Assert-PathInside -Path $Path -Root $Root
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

function Get-DirectorySizeBytes {
    param([string]$Path)
    $total = 0L
    Get-ChildItem -LiteralPath $Path -Recurse -File | ForEach-Object {
        $total += $_.Length
    }
    return $total
}

function Expand-ZipArchive {
    param(
        [string]$ZipPath,
        [string]$DestinationPath
    )
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory(
        (Get-FullPath $ZipPath),
        (Get-FullPath $DestinationPath)
    )
}

function Find-InnoCompiler {
    param([string]$RequestedPath)
    if ($RequestedPath -and (Test-Path -LiteralPath $RequestedPath)) {
        return (Resolve-Path -LiteralPath $RequestedPath).Path
    }

    try {
        $whereResult = & where.exe ISCC 2>$null
    }
    catch {
        $whereResult = @()
    }
    if ($LASTEXITCODE -eq 0 -and $whereResult) {
        return ($whereResult | Select-Object -First 1)
    }

    $commonPaths = @(
        "$env:LOCALAPPDATA\Programs\Inno Setup 7\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "C:\Program Files (x86)\Inno Setup 7\ISCC.exe",
        "C:\Program Files\Inno Setup 7\ISCC.exe",
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "C:\Program Files\Inno Setup 6\ISCC.exe"
    )
    foreach ($candidate in $commonPaths) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    return $null
}

function Write-StartScript {
    param([string]$Path)
    $lines = @(
        "@echo off",
        "setlocal",
        "set ""APP_DIR=%~dp0""",
        "if ""%APP_DIR:~-1%""==""\"" set ""APP_DIR=%APP_DIR:~0,-1%""",
        "set ""CONDA_PREFIX=%APP_DIR%""",
        "set ""CONDA_DEFAULT_ENV=AIMS4PT_cpx""",
        "set ""PYTHONNOUSERSITE=1""",
        "set ""PATH=%APP_DIR%;%APP_DIR%\Library\mingw-w64\bin;%APP_DIR%\Library\usr\bin;%APP_DIR%\Library\bin;%APP_DIR%\Scripts;%PATH%""",
        "cd /d ""%USERPROFILE%""",
        """%APP_DIR%\python.exe"" -m aims4pt_web.launcher",
        "if errorlevel 1 (",
        "  echo.",
        "  echo AIMS4PT_cpx failed to start. Press any key to close.",
        "  pause >nul",
        ")"
    )
    Set-Content -LiteralPath $Path -Value $lines -Encoding ASCII
}

function Invoke-WithPayloadEnvironment {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )
    $oldCondaPrefix = $env:CONDA_PREFIX
    $oldCondaDefaultEnv = $env:CONDA_DEFAULT_ENV
    $oldPythonNoUserSite = $env:PYTHONNOUSERSITE
    $oldPath = $env:PATH
    try {
        $env:CONDA_PREFIX = $PayloadDir
        $env:CONDA_DEFAULT_ENV = "AIMS4PT_cpx"
        $env:PYTHONNOUSERSITE = "1"
        $env:PATH = "$PayloadDir;$PayloadDir\Library\mingw-w64\bin;$PayloadDir\Library\usr\bin;$PayloadDir\Library\bin;$PayloadDir\Scripts;$oldPath"
        & $Executable @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $Executable $($Arguments -join ' ')"
        }
    }
    finally {
        $env:CONDA_PREFIX = $oldCondaPrefix
        $env:CONDA_DEFAULT_ENV = $oldCondaDefaultEnv
        $env:PYTHONNOUSERSITE = $oldPythonNoUserSite
        $env:PATH = $oldPath
    }
}

function Invoke-UvicornSmokeTest {
    $python = Join-Path $PayloadDir "python.exe"
    $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, 0)
    $listener.Start()
    $port = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
    $listener.Stop()
    $url = "http://127.0.0.1:$port/"
    Remove-Item -LiteralPath $ServerStdout, $ServerStderr -Force -ErrorAction SilentlyContinue

    $oldPath = $env:PATH
    $oldCondaPrefix = $env:CONDA_PREFIX
    $oldCondaDefaultEnv = $env:CONDA_DEFAULT_ENV
    $oldPythonNoUserSite = $env:PYTHONNOUSERSITE
    $process = $null
    try {
        $env:CONDA_PREFIX = $PayloadDir
        $env:CONDA_DEFAULT_ENV = "AIMS4PT_cpx"
        $env:PYTHONNOUSERSITE = "1"
        $env:PATH = "$PayloadDir;$PayloadDir\Library\mingw-w64\bin;$PayloadDir\Library\usr\bin;$PayloadDir\Library\bin;$PayloadDir\Scripts;$oldPath"
        $process = Start-Process `
            -FilePath $python `
            -ArgumentList @("-m", "uvicorn", "aims4pt_web.main:app", "--host", "127.0.0.1", "--port", "$port", "--workers", "1") `
            -WorkingDirectory $PayloadDir `
            -WindowStyle Hidden `
            -PassThru `
            -RedirectStandardOutput $ServerStdout `
            -RedirectStandardError $ServerStderr

        $deadline = (Get-Date).AddSeconds(180)
        while ((Get-Date) -lt $deadline) {
            if ($process.HasExited) {
                $stderr = if (Test-Path -LiteralPath $ServerStderr) { Get-Content -Raw -LiteralPath $ServerStderr } else { "" }
                throw "uvicorn exited before readiness. ExitCode=$($process.ExitCode)`n$stderr"
            }
            try {
                $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
                if ($response.StatusCode -eq 200) {
                    Write-Host "uvicorn_ready $url"
                    return
                }
            }
            catch {
                Start-Sleep -Milliseconds 500
            }
        }
        throw "Timed out waiting for uvicorn at $url"
    }
    finally {
        if ($process -and -not $process.HasExited) {
            Stop-Process -Id $process.Id -Force
            $process.WaitForExit()
        }
        $env:PATH = $oldPath
        $env:CONDA_PREFIX = $oldCondaPrefix
        $env:CONDA_DEFAULT_ENV = $oldCondaDefaultEnv
        $env:PYTHONNOUSERSITE = $oldPythonNoUserSite
    }
}

function Write-InnoScript {
    param([bool]$UseDiskSpanning)

    $diskSpanningValue = if ($UseDiskSpanning) { "yes" } else { "no" }
    $diskSliceLine = if ($UseDiskSpanning) { "DiskSliceSize=1900000000" } else { "; DiskSliceSize is only used with DiskSpanning=yes" }
    $compression = "lzma2/ultra64"
    $solidCompression = "yes"
    $lzmaThreads = "1"
    $versionInfoVersion = if ($ProductVersion -match '^\d+\.\d+\.\d+$') { "$ProductVersion.0" } else { $ProductVersion }
    $innoPayloadSource = Join-Path $PayloadDir "*"
    $script = @"
#define MyAppName "$ProductName"
#define MyAppVersion "$ProductVersion"
#define MyAppPublisher "Xiaoyu Liu"
#define MyOutputBaseFilename "$SetupBaseName"

[Setup]
AppId={{8B1F3F9F-0B95-4C24-A43A-1A2B3C4D0100}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
VersionInfoVersion=$versionInfoVersion
VersionInfoProductVersion=$ProductVersion
VersionInfoProductName=$ProductName
VersionInfoCompany=Xiaoyu Liu
VersionInfoDescription=$ProductName Setup
DefaultDirName={autopf}\AIMS4PT_cpx
DefaultGroupName=AIMS4PT_cpx
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=$DistDir
OutputBaseFilename={#MyOutputBaseFilename}
Compression=$compression
SolidCompression=$solidCompression
LZMANumBlockThreads=$lzmaThreads
DiskSpanning=$diskSpanningValue
$diskSliceLine
WizardStyle=modern
SetupLogging=yes
UninstallDisplayName=AIMS4PT_cpx

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "$innoPayloadSource"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
Type: filesandordirs; Name: "{app}\Lib\site-packages\aims4pt"
Type: filesandordirs; Name: "{app}\Lib\site-packages\aims4pt_web"
Type: filesandordirs; Name: "{app}\Lib\site-packages\aims4pt-*.dist-info"

[Icons]
Name: "{group}\AIMS4PT_cpx"; Filename: "{app}\Start AIMS4PT_cpx.cmd"; WorkingDir: "{app}"
Name: "{autodesktop}\AIMS4PT_cpx"; Filename: "{app}\Start AIMS4PT_cpx.cmd"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\Scripts\conda-unpack.exe"; StatusMsg: "Finalizing Python environment..."; Flags: runhidden waituntilterminated; Check: CondaUnpackExists
Filename: "{app}\Start AIMS4PT_cpx.cmd"; Description: "Launch AIMS4PT_cpx"; Flags: nowait postinstall skipifsilent

[Code]
function CondaUnpackExists: Boolean;
begin
  Result := FileExists(ExpandConstant('{app}\Scripts\conda-unpack.exe'));
end;
"@
    Set-Content -LiteralPath $IssPath -Value $script -Encoding UTF8
}

New-Item -ItemType Directory -Force -Path $BuildRoot, $DistDir, $StageRoot | Out-Null

$envPrefixFull = Get-FullPath $EnvPrefix
if (-not (Test-Path -LiteralPath $envPrefixFull)) {
    throw "Environment prefix does not exist: $envPrefixFull"
}

$condaPack = Join-Path $envPrefixFull "Scripts\conda-pack.exe"
if (-not (Test-Path -LiteralPath $condaPack)) {
    throw "conda-pack.exe was not found in $envPrefixFull\Scripts"
}
$envPython = Join-Path $envPrefixFull "python.exe"
if (-not (Test-Path -LiteralPath $envPython)) {
    throw "python.exe was not found in $envPrefixFull"
}
$installedVersionOutput = & $envPython -c "import importlib.metadata as m; print(m.version('AIMS4PT'))"
if ($LASTEXITCODE -ne 0) {
    throw "Could not read the installed AIMS4PT version from $envPrefixFull"
}
$installedVersion = ($installedVersionOutput | Select-Object -Last 1).Trim()
if ($installedVersion -ne $ProductVersion) {
    throw "Release environment has AIMS4PT $installedVersion, expected $ProductVersion"
}
Write-Step "Release environment contains AIMS4PT $installedVersion"

Write-Step "Recording the exact Windows release dependency manifest"
& $envPython $DependencyManifestScript `
    --output $DependencyManifestPath `
    --prefix $envPrefixFull `
    --label "$ProductName $ProductVersion Windows x64"
if ($LASTEXITCODE -ne 0) {
    throw "Dependency manifest generation failed with exit code $LASTEXITCODE"
}

if (-not $SkipStage) {
    Write-Step "Creating slim conda-pack archive from $envPrefixFull"
    Remove-DirectorySafe -Path $PayloadDir -Root $StageRoot
    Assert-PathInside -Path $SlimZipPath -Root $StageRoot
    Remove-Item -LiteralPath $SlimZipPath -Force -ErrorAction SilentlyContinue
    $excludePatterns = @(
        "**/__pycache__/**",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.pdb",
        "**/tests/**",
        "Library/include/**",
        "Library/lib/**/*.lib",
        "Library/lib/**/*.a",
        "Library/lib/**/*.la",
        "Library/lib/cmake/**",
        "Library/lib/pkgconfig/**",
        "Library/libexec/gcc/**",
        "Library/x86_64-w64-mingw32/**",
        "Library/mingw-w64/**",
        "Lib/site-packages/tensorflow/include/**",
        "Lib/site-packages/tensorflow/**/*.lib",
        "Lib/site-packages/debugpy/**",
        "Lib/site-packages/debugpy-*.dist-info/**",
        "Lib/site-packages/ipykernel/**",
        "Lib/site-packages/ipykernel-*.dist-info/**",
        "Lib/site-packages/jedi/**",
        "Lib/site-packages/jedi-*.dist-info/**",
        "Lib/site-packages/sphinx/**",
        "Lib/site-packages/sphinx-*.dist-info/**",
        "Lib/site-packages/PyQt5/**",
        "Lib/site-packages/PyQt5-*.dist-info/**",
        "Lib/site-packages/qtpy/**",
        "Lib/site-packages/qtpy-*.dist-info/**"
    )
    $includePatterns = @(
        "Library/mingw-w64/bin/xgboost.dll"
    )
    $packArgs = @(
        "--prefix", $envPrefixFull,
        "--output", $SlimZipPath,
        "--format", "zip",
        "--compress-level", "9",
        "--force",
        "--ignore-editable-packages"
    )
    foreach ($pattern in $excludePatterns) {
        $packArgs += @("--exclude", $pattern)
    }
    foreach ($pattern in $includePatterns) {
        $packArgs += @("--include", $pattern)
    }
    $oldPythonUtf8 = $env:PYTHONUTF8
    $oldPythonIoEncoding = $env:PYTHONIOENCODING
    try {
        $env:PYTHONUTF8 = "1"
        $env:PYTHONIOENCODING = "utf-8"
        & $condaPack @packArgs
        if ($LASTEXITCODE -ne 0) {
            throw "conda-pack failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        $env:PYTHONUTF8 = $oldPythonUtf8
        $env:PYTHONIOENCODING = $oldPythonIoEncoding
    }

    $slimZipSizeGiB = [math]::Round((Get-Item -LiteralPath $SlimZipPath).Length / 1GB, 2)
    Write-Step "Slim archive size: $slimZipSizeGiB GiB"
    Write-Step "Extracting slim archive to payload"
    New-Item -ItemType Directory -Force -Path $PayloadDir | Out-Null
    Expand-ZipArchive -ZipPath $SlimZipPath -DestinationPath $PayloadDir
    Write-StartScript -Path (Join-Path $PayloadDir "Start AIMS4PT_cpx.cmd")
}
else {
    Write-Step "Reusing existing payload at $PayloadDir"
}

if (-not (Test-Path -LiteralPath (Join-Path $PayloadDir "python.exe"))) {
    throw "Payload is missing python.exe: $PayloadDir"
}
if (-not (Test-Path -LiteralPath (Join-Path $PayloadDir "Scripts\conda-unpack.exe"))) {
    throw "Payload is missing Scripts\conda-unpack.exe"
}
if (-not (Test-Path -LiteralPath (Join-Path $PayloadDir "Start AIMS4PT_cpx.cmd"))) {
    Write-StartScript -Path (Join-Path $PayloadDir "Start AIMS4PT_cpx.cmd")
}

$payloadSize = Get-DirectorySizeBytes -Path $PayloadDir
$payloadSizeGiB = [math]::Round($payloadSize / 1GB, 2)
Write-Step "Payload size: $payloadSizeGiB GiB"

if (-not $SkipSmokeTest) {
    Write-Step "Running packaged runtime smoke test"
    Invoke-WithPayloadEnvironment `
        -Executable (Join-Path $PayloadDir "python.exe") `
        -Arguments @(
            $SmokeTestPath,
            "--expected-version", $ProductVersion,
            "--expected-prefix", $PayloadDir
        )

    Write-Step "Running cross-platform numerical regression test"
    Invoke-WithPayloadEnvironment `
        -Executable (Join-Path $PayloadDir "python.exe") `
        -Arguments @($RegressionTestPath)

    Write-Step "Running uvicorn readiness smoke test"
    Invoke-UvicornSmokeTest
}

$compiler = Find-InnoCompiler -RequestedPath $IsccPath
if (-not $compiler) {
    throw "Inno Setup compiler ISCC.exe was not found. Install Inno Setup or pass -IsccPath."
}

$githubAssetLimitBytes = 1990MB
$useDiskSpanning = [bool]$ForceDiskSpanning
Get-ChildItem -LiteralPath $DistDir -Filter "$SetupBaseName*" -File -ErrorAction SilentlyContinue |
    Remove-Item -Force
Write-InnoScript -UseDiskSpanning $useDiskSpanning
Write-Step "Compiling installer with Inno Setup: $compiler"
& $compiler $IssPath
$firstExitCode = $LASTEXITCODE
if ($firstExitCode -ne 0 -and -not $useDiskSpanning) {
    Write-Step "Single-file installer compile failed; retrying with disk spanning"
    $useDiskSpanning = $true
    Get-ChildItem -LiteralPath $DistDir -Filter "$SetupBaseName*" -File -ErrorAction SilentlyContinue |
        Remove-Item -Force
    Write-InnoScript -UseDiskSpanning $true
    & $compiler $IssPath
    if ($LASTEXITCODE -ne 0) {
        throw "Inno Setup disk-spanning compile failed with exit code $LASTEXITCODE"
    }
}
elseif ($firstExitCode -ne 0) {
    throw "Inno Setup compile failed with exit code $firstExitCode"
}

$setupExe = Join-Path $DistDir "$SetupBaseName.exe"
if (-not (Test-Path -LiteralPath $setupExe)) {
    throw "Expected setup executable was not created: $setupExe"
}

if ((-not $useDiskSpanning) -and ((Get-Item -LiteralPath $setupExe).Length -gt $githubAssetLimitBytes)) {
    Write-Step "Single-file installer exceeds the GitHub asset limit; rebuilding with disk spanning"
    $useDiskSpanning = $true
    Get-ChildItem -LiteralPath $DistDir -Filter "$SetupBaseName*" -File -ErrorAction SilentlyContinue |
        Remove-Item -Force
    Write-InnoScript -UseDiskSpanning $true
    & $compiler $IssPath
    if ($LASTEXITCODE -ne 0) {
        throw "Inno Setup disk-spanning compile failed with exit code $LASTEXITCODE"
    }
}

$artifacts = Get-ChildItem -LiteralPath $DistDir -Filter "$SetupBaseName*" -File
if (-not $artifacts) {
    throw "No installer artifacts were created for $SetupBaseName"
}
foreach ($artifact in $artifacts) {
    if ($artifact.Length -gt $githubAssetLimitBytes) {
        throw "Installer artifact exceeds the GitHub asset limit: $($artifact.FullName)"
    }
}

$checksumPath = Join-Path $DistDir "$SetupBaseName-SHA256.txt"
$checksumLines = foreach ($artifact in $artifacts) {
    $hash = Get-FileHash -LiteralPath $artifact.FullName -Algorithm SHA256
    "$($hash.Hash)  $($artifact.Name)"
}
Set-Content -LiteralPath $checksumPath -Value $checksumLines -Encoding ASCII

Write-Step "Installer created: $setupExe"
Get-ChildItem -LiteralPath $DistDir -Filter "$SetupBaseName*" |
    Select-Object FullName, Length, LastWriteTime
Get-Item -LiteralPath $DependencyManifestPath |
    Select-Object FullName, Length, LastWriteTime


# powershell -NoProfile -ExecutionPolicy Bypass -File ".\packaging\windows\build_windows_installer.ps1" `
#   -EnvPrefix "C:\Users\13493\miniconda3\envs\AIMS4PT_release" `
#   -IsccPath "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
