#!/bin/bash
# One-shot installer for gimpsegany on Linux.
#
# Creates (or reuses) a Python virtualenv with the SAM2 backend (PyTorch /
# OpenCV / huggingface_hub / sam2), copies the plugin files into GIMP's
# per-user plug-ins folder, and seeds a default segany_settings.json pointing
# at SAM 2.1 Large on Hugging Face so the first run in GIMP works without any
# extra configuration.
#
# Uses uv (https://docs.astral.sh/uv/) when available — it provisions Python
# 3.11 itself and installs fast. Falls back to the stdlib venv + pip when uv
# is absent (needs a Python 3.10+ already on the system). No conda required.
# On Linux pip pulls the CUDA-enabled torch wheels, so GPU machines work out
# of the box (and CPU-only machines still run).
#
# Re-runs are idempotent: an existing venv is kept, already-installed packages
# are skipped, and an existing segany_settings.json is NOT overwritten.
#
# Run from a terminal:
#     bash install-linux.sh

set -e
cd "$(dirname "$0")"

REPO_DIR="$(pwd)"
# Kept in sync with SEGANY_VENV_PY in seganyplugin.py.
SEGANY_VENV="$HOME/.gimp-segany/venv"
ENV_PY="$SEGANY_VENV/bin/python"

# sam2 isn't on PyPI; pin it to a known-good commit so the build is
# reproducible (a bare git URL floats to the repo tip). Python deps are pinned
# in requirements-lock.txt.
SAM2_GIT="git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"

echo "==> gimpsegany installer (Linux)"
echo "    repo: $REPO_DIR"
echo "    venv: $SEGANY_VENV"
echo

# --- 1. Backend: uv or venv --------------------------------------------------
USE_UV=0
if command -v uv >/dev/null 2>&1; then
    USE_UV=1
    echo "==> using uv: $(command -v uv)"
else
    echo "==> uv not found — falling back to the stdlib venv + pip"
fi

# pip_install <pip args…> — install into the venv with whichever backend.
pip_install() {
    if [ "$USE_UV" = "1" ]; then
        uv pip install --python "$ENV_PY" "$@"
    else
        "$ENV_PY" -m pip install "$@"
    fi
}

# --- 2. Virtualenv -----------------------------------------------------------
if [ -x "$ENV_PY" ]; then
    echo "==> reusing existing venv at $SEGANY_VENV"
else
    mkdir -p "$(dirname "$SEGANY_VENV")"
    if [ "$USE_UV" = "1" ]; then
        echo "==> creating venv with uv (python 3.11)"
        uv venv --python 3.11 "$SEGANY_VENV"
    else
        PYBOOT=""
        for cand in python3.12 python3.11 python3.10 python3; do
            if command -v "$cand" >/dev/null 2>&1; then PYBOOT="$cand"; break; fi
        done
        if [ -z "$PYBOOT" ] || ! "$PYBOOT" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)'; then
            echo "Error: need uv or a Python 3.10+ interpreter."
            echo "Install uv (recommended — it provisions Python for you):"
            echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
            echo "or install Python 3.11 from your distro / https://www.python.org/downloads/"
            exit 1
        fi
        echo "==> creating venv with '$PYBOOT -m venv'"
        "$PYBOOT" -m venv "$SEGANY_VENV"
        "$ENV_PY" -m pip install --upgrade pip
    fi
fi
echo "==> env python: $ENV_PY"

# --- 3. Dependencies ---------------------------------------------------------
echo "==> installing pinned python dependencies"
pip_install -r requirements-lock.txt

# sam2 isn't on PyPI — install the pinned commit from GitHub if missing.
if "$ENV_PY" -c "import sam2" >/dev/null 2>&1; then
    echo "    ✓ sam2 already installed"
else
    echo "    … installing sam2 from GitHub (this may take a few minutes)"
    SAM2_BUILD_CUDA=0 pip_install "$SAM2_GIT"
fi

# --- 4. Plugin directory ---------------------------------------------------
# Standard install keeps config under ~/.config/GIMP; a Flatpak install
# sandboxes it under ~/.var/app/org.gimp.GIMP/. Use whichever has a 3.x
# folder, preferring the standard location.
GIMP_BASE=""
for base in \
    "$HOME/.config/GIMP" \
    "$HOME/.var/app/org.gimp.GIMP/config/GIMP"
do
    if [ -d "$base" ] && ls -1 "$base" 2>/dev/null | grep -qE '^3\.'; then
        GIMP_BASE="$base"
        break
    fi
done

if [ -z "$GIMP_BASE" ]; then
    echo
    echo "Error: no GIMP 3.x config folder found under:"
    echo "    $HOME/.config/GIMP"
    echo "    $HOME/.var/app/org.gimp.GIMP/config/GIMP   (Flatpak)"
    echo "Install GIMP 3 (https://www.gimp.org/downloads/) and launch it once"
    echo "so it creates its config, then re-run this script."
    exit 1
fi

# Pick the newest 3.x folder already present on disk.
GIMP_VERSION=$(ls -1 "$GIMP_BASE" 2>/dev/null | grep -E '^3\.' | sort -V | tail -1)

PLUGIN_DIR="$GIMP_BASE/$GIMP_VERSION/plug-ins/seganyplugin"
echo "==> installing into GIMP $GIMP_VERSION"
echo "    $PLUGIN_DIR"

mkdir -p "$PLUGIN_DIR"
cp -v seganyplugin.py seganybridge.py segany_backend.py "$PLUGIN_DIR/"
chmod +x "$PLUGIN_DIR/seganyplugin.py"

# --- 5. Default settings ---------------------------------------------------
SETTINGS="$PLUGIN_DIR/segany_settings.json"
if [ -f "$SETTINGS" ]; then
    echo "==> keeping existing $SETTINGS"
else
    echo "==> writing default settings"
    cat > "$SETTINGS" <<JSON
{
  "pythonPath": "$ENV_PY",
  "modelType": "sam2_hiera_large (SAM2)",
  "checkPtPath": "facebook/sam2.1-hiera-large",
  "modelSource": "SAM 2.1 Large (~900 MB, best quality)",
  "segType": "Box",
  "maskType": "Single",
  "autoSelectTopMask": true
}
JSON
fi

echo
echo "==> Done."
echo "    Open GIMP, load an image, and use: Image → Segment Anything Layers"
echo "    The first run downloads the SAM 2.1 Large weights from Hugging Face"
echo "    (~900 MB, cached under ~/.cache/huggingface)."
