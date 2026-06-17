# One-shot installer for gimpsegany on Windows.
#
# Creates (or reuses) a Python virtualenv with the SAM2 backend (PyTorch /
# OpenCV / huggingface_hub / sam2), copies the plugin files into GIMP's
# per-user plug-ins folder, and seeds a default segany_settings.json pointing
# at SAM 2.1 Large on Hugging Face so the first run in GIMP works without any
# extra configuration.
#
# Uses uv (https://docs.astral.sh/uv/) when available -- it provisions Python
# 3.11 itself and installs fast. Falls back to the stdlib venv + pip when uv
# is absent (needs a Python 3.10+ already on the system, via the 'py' launcher
# or 'python' on PATH). No conda required. On Windows pip pulls the
# CUDA-enabled torch wheels, so NVIDIA GPUs work out of the box.
#
# Re-runs are idempotent: an existing venv is kept, already-installed packages
# are skipped, and an existing segany_settings.json is NOT overwritten.
#
# Run from PowerShell (in the repo folder):
#     powershell -ExecutionPolicy Bypass -File install-windows.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$RepoDir = (Get-Location).Path
# Kept in sync with SEGANY_VENV_PY in seganyplugin.py.
$SeganyVenv = Join-Path $env:USERPROFILE ".gimp-segany\venv"
$EnvPy = Join-Path $SeganyVenv "Scripts\python.exe"

# sam2 isn't on PyPI; pin it to a known-good commit so the build is
# reproducible (a bare git URL floats to the repo tip). Python deps are pinned
# in requirements-lock.txt.
$Sam2Git = "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"

Write-Host "==> gimpsegany installer (Windows)"
Write-Host "    repo: $RepoDir"
Write-Host "    venv: $SeganyVenv"
Write-Host ""

# --- 1. Backend: uv or venv --------------------------------------------------
$UseUv = $false
if (Get-Command uv -ErrorAction SilentlyContinue) {
    $UseUv = $true
    Write-Host "==> using uv: $((Get-Command uv).Source)"
} else {
    Write-Host "==> uv not found -- falling back to the stdlib venv + pip"
}

# Install into the venv with whichever backend. Takes the pip arguments as an
# explicit array so leading-dash tokens (e.g. -r) are treated as data, not as
# parameters to this function.
function Invoke-PipInstall([string[]] $PipArgs) {
    if ($UseUv) {
        & uv pip install --python $EnvPy @PipArgs
    } else {
        & $EnvPy -m pip install @PipArgs
    }
    if ($LASTEXITCODE -ne 0) { throw "pip install failed: $PipArgs" }
}

# --- 2. Virtualenv -----------------------------------------------------------
if (Test-Path $EnvPy) {
    Write-Host "==> reusing existing venv at $SeganyVenv"
} else {
    New-Item -ItemType Directory -Force -Path (Split-Path $SeganyVenv) | Out-Null
    if ($UseUv) {
        Write-Host "==> creating venv with uv (python 3.11)"
        & uv venv --python 3.11 $SeganyVenv
        if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }
    } else {
        # Find a Python 3.10+ to bootstrap the venv: prefer the 'py' launcher
        # with an explicit version, then plain python. Hashtables (not nested
        # arrays) keep the version-arg lists intact — PowerShell flattens
        # nested array literals.
        $candidates = @(
            @{ Exe = "py";     Args = @("-3.12") },
            @{ Exe = "py";     Args = @("-3.11") },
            @{ Exe = "py";     Args = @("-3.10") },
            @{ Exe = "py";     Args = @("-3")    },
            @{ Exe = "python"; Args = @()        }
        )
        $bootExe = $null
        $bootArgs = @()
        foreach ($c in $candidates) {
            $exe = $c.Exe
            $verArgs = $c.Args
            if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
            try {
                & $exe @verArgs -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)"
                if ($LASTEXITCODE -eq 0) { $bootExe = $exe; $bootArgs = $verArgs; break }
            } catch { }
        }
        if ($null -eq $bootExe) {
            Write-Host "Error: need uv or a Python 3.10+ interpreter."
            Write-Host "Install uv (recommended -- it provisions Python for you):"
            Write-Host "    powershell -c ""irm https://astral.sh/uv/install.ps1 | iex"""
            Write-Host "or install Python 3.11 from https://www.python.org/downloads/"
            exit 1
        }
        Write-Host "==> creating venv with '$bootExe $bootArgs -m venv'"
        & $bootExe @bootArgs -m venv $SeganyVenv
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
        & $EnvPy -m pip install --upgrade pip
    }
}
Write-Host "==> env python: $EnvPy"

# --- 3. Dependencies ---------------------------------------------------------
Write-Host "==> installing pinned python dependencies"
Invoke-PipInstall @("-r", "requirements-lock.txt")

# sam2 isn't on PyPI -- install the pinned commit from GitHub if missing.
& $EnvPy -c "import sam2" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "    [ok] sam2 already installed"
} else {
    Write-Host "    ... installing sam2 from GitHub (this may take a few minutes)"
    $env:SAM2_BUILD_CUDA = "0"
    Invoke-PipInstall @($Sam2Git)
}

# --- 4. Plugin directory -----------------------------------------------------
$GimpBase = Join-Path $env:APPDATA "GIMP"
if (-not (Test-Path $GimpBase)) {
    Write-Host ""
    Write-Host "Error: $GimpBase does not exist."
    Write-Host "Install GIMP 3 first: https://www.gimp.org/downloads/"
    exit 1
}

# Pick the newest 3.x folder already present on disk.
$GimpVersion = Get-ChildItem -Path $GimpBase -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^3\.' } |
    Sort-Object { [version]$_.Name } |
    Select-Object -Last 1
if ($null -eq $GimpVersion) {
    Write-Host ""
    Write-Host "Error: no GIMP 3.x folder under $GimpBase."
    Write-Host "Launch GIMP 3 once so it creates its config, then re-run this script."
    exit 1
}

$PluginDir = Join-Path $GimpBase ("{0}\plug-ins\seganyplugin" -f $GimpVersion.Name)
Write-Host "==> installing into GIMP $($GimpVersion.Name)"
Write-Host "    $PluginDir"

New-Item -ItemType Directory -Force -Path $PluginDir | Out-Null
Copy-Item -Path seganyplugin.py, seganybridge.py, segany_backend.py -Destination $PluginDir -Force

# --- 5. Default settings -----------------------------------------------------
$Settings = Join-Path $PluginDir "segany_settings.json"
if (Test-Path $Settings) {
    Write-Host "==> keeping existing $Settings"
} else {
    Write-Host "==> writing default settings"
    # Use forward slashes in the JSON path so no backslash escaping is needed;
    # Python on Windows accepts them fine.
    $EnvPyJson = $EnvPy -replace '\\', '/'
    @"
{
  "pythonPath": "$EnvPyJson",
  "modelType": "sam2_hiera_large (SAM2)",
  "checkPtPath": "facebook/sam2.1-hiera-large",
  "modelSource": "SAM 2.1 Large (~900 MB, best quality)",
  "segType": "Box",
  "maskType": "Single",
  "autoSelectTopMask": true
}
"@ | Set-Content -Path $Settings -Encoding utf8
}

Write-Host ""
Write-Host "==> Done."
Write-Host "    Open GIMP, load an image, and use: Image -> Segment Anything Layers"
Write-Host "    The first run downloads the SAM 2.1 Large weights from Hugging Face"
Write-Host "    (~900 MB, cached under %USERPROFILE%\.cache\huggingface)."
