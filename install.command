#!/bin/bash
# One-shot installer for gimpsegany on macOS.
#
# Creates (or reuses) a "sam2" conda env, installs PyTorch / opencv /
# huggingface_hub / sam2, copies the plugin files into GIMP's per-user
# plug-ins folder, and seeds a default segany_settings.json pointing at
# SAM 2.1 Large on Hugging Face so the first run in GIMP works without
# any extra configuration.
#
# Re-runs are idempotent: existing env is kept, already-installed packages
# are skipped, and an existing segany_settings.json is NOT overwritten.
#
# Double-click in Finder, or run from Terminal:
#     bash install.command

set -e
cd "$(dirname "$0")"

ENV_NAME="sam2"
REPO_DIR="$(pwd)"

echo "==> gimpsegany installer"
echo "    repo: $REPO_DIR"
echo

# --- 1. Conda ---------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "Error: 'conda' not found on PATH."
    echo "Install Miniforge (recommended) or Miniconda first:"
    echo "    https://github.com/conda-forge/miniforge"
    exit 1
fi
echo "==> conda: $(command -v conda)"

# --- 2. Env ------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "==> conda env '${ENV_NAME}' already exists — keeping it"
else
    if [ -f "environment-macos.yml" ]; then
        echo "==> creating conda env '${ENV_NAME}' from environment-macos.yml"
        conda env create -f environment-macos.yml
    else
        echo "==> creating conda env '${ENV_NAME}' (python 3.11) from scratch"
        conda create -y -n "${ENV_NAME}" python=3.11
        conda run -n "${ENV_NAME}" pip install \
            torch torchvision torchaudio opencv-python huggingface_hub
    fi
fi

ENV_PY=$(conda run -n "${ENV_NAME}" python -c "import sys; print(sys.executable)")
echo "==> env python: $ENV_PY"

# --- 3. Dependencies (belt-and-suspenders, in case the env is older) -------
ensure_pkg() {
    local import_name="$1"
    local pip_target="$2"
    if conda run -n "${ENV_NAME}" python -c "import ${import_name}" >/dev/null 2>&1; then
        echo "    ✓ ${import_name} already installed"
    else
        echo "    … installing ${pip_target}"
        conda run -n "${ENV_NAME}" pip install "${pip_target}"
    fi
}

echo "==> verifying python dependencies"
ensure_pkg "torch"             "torch"
ensure_pkg "cv2"               "opencv-python"
ensure_pkg "huggingface_hub"   "huggingface_hub"

# sam2 isn't on PyPI — install from GitHub if missing.
if conda run -n "${ENV_NAME}" python -c "import sam2" >/dev/null 2>&1; then
    echo "    ✓ sam2 already installed"
else
    echo "    … installing sam2 from GitHub (this may take a few minutes)"
    SAM2_BUILD_CUDA=0 conda run -n "${ENV_NAME}" pip install \
        "git+https://github.com/facebookresearch/sam2.git"
fi

# --- 4. Plugin directory ---------------------------------------------------
GIMP_BASE="$HOME/Library/Application Support/GIMP"
if [ ! -d "$GIMP_BASE" ]; then
    echo
    echo "Error: $GIMP_BASE does not exist."
    echo "Install GIMP 3 first: https://www.gimp.org/downloads/"
    exit 1
fi

# Pick the newest 3.x folder already present on disk.
GIMP_VERSION=$(ls -1 "$GIMP_BASE" 2>/dev/null | grep -E '^3\.' | sort -V | tail -1)
if [ -z "$GIMP_VERSION" ]; then
    echo
    echo "Error: no GIMP 3.x folder under $GIMP_BASE."
    echo "Launch GIMP 3 once so it creates its config, then re-run this script."
    exit 1
fi

PLUGIN_DIR="$GIMP_BASE/$GIMP_VERSION/plug-ins/seganyplugin"
echo "==> installing into GIMP $GIMP_VERSION"
echo "    $PLUGIN_DIR"

mkdir -p "$PLUGIN_DIR"
cp -v seganyplugin.py seganybridge.py "$PLUGIN_DIR/"
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
