## GIMP Plugin For Integration With Meta Segment Anything

> **Fork notice:** This is a fork of [Shriinivas/gimpsegany](https://github.com/Shriinivas/gimpsegany) with macOS Apple Silicon support and UI improvements. See [What's new in this fork](#whats-new-in-this-fork).

This GIMP plugin integrates with Meta's AI-based tool Segment Anything, which enables you to effortlessly isolate objects within raster images directly from GIMP.

The plugin supports both **Segment Anything 1 (SAM1)** and **Segment Anything 2 (SAM2)** — including SAM 2.1 checkpoints.

---

## What's new in this fork

- **Apple Silicon / MPS support** — SAM2 models load and run on macOS Apple Silicon via PyTorch's MPS backend. Falls back to CUDA, then CPU. If a run hits the known MPS `Placeholder storage` error, the bridge automatically reloads on CPU and retries — no source edits needed. Set `SEGANY_FORCE_CPU=1` to skip MPS entirely.
- **SAM1 imports are optional** — the bridge no longer requires `segment_anything` to be installed when you only use SAM2.
- **SAM 2.1 checkpoints** — the bridge auto-detects `sam2.1_*` checkpoint filenames and loads the matching configs from `configs/sam2.1/`.
- **Editable path fields** — the Python and Checkpoint path fields in the plugin dialog accept typed/pasted paths (Entry + Browse button) instead of being limited to a file picker. The browse dialog also shows hidden files, which is required to reach `~/Library` on macOS.
- **Persistent backend** — the bridge runs as a long-living subprocess and caches the loaded model. The first segmentation pays the model-load cost; subsequent ones skip it (≈8× faster on Apple Silicon).
- **Hugging Face Hub support** — instead of a local `.pt` path you can paste a HF id like `facebook/sam2.1-hiera-large` into the Checkpoint field. Requires `pip install huggingface_hub` in the SAM environment.

---

## Installation

### Quick install (macOS)

Clone this repo, then run:

```bash
bash install.command
```

It creates (or reuses) a Python virtualenv at `~/.gimp-segany/venv` with PyTorch / OpenCV / huggingface_hub / sam2, copies `seganyplugin.py` and `seganybridge.py` into the right GIMP plug-ins folder, and seeds a default `segany_settings.json` pointing at `facebook/sam2.1-hiera-large` on Hugging Face — so the first run in GIMP works without any extra configuration. The script is idempotent and preserves an existing settings file.

It uses [`uv`](https://docs.astral.sh/uv/) when present (it provisions Python 3.11 itself and installs fast) and falls back to the stdlib `venv` + `pip` otherwise. **No conda required.**

Requirements: GIMP 3 already installed, and either `uv` on your `PATH` (`curl -LsSf https://astral.sh/uv/install.sh | sh`) or an existing Python 3.10+.

Then restart GIMP and use `Image → Segment Anything Layers`. The first segmentation downloads the SAM 2.1 Large weights from Hugging Face (~900 MB, cached under `~/.cache/huggingface`); the GIMP status bar shows the download and load progress.

If something goes wrong, click **Run Setup Check** in the plugin dialog — it verifies the interpreter, PyTorch + device, sam2, and the checkpoint load in one go and points at the exact missing piece.

### Quick install (Linux)

Clone this repo, then run:

```bash
bash install-linux.sh
```

It does the same as the macOS installer (uv when present, stdlib `venv` + `pip` otherwise — **no conda**): builds the `~/.gimp-segany/venv` backend, copies the plugin files into GIMP's per-user plug-ins folder, and seeds a default `segany_settings.json`. On Linux pip pulls the CUDA-enabled torch wheels, so GPU machines work out of the box (and CPU-only machines still run). It auto-detects both a standard `~/.config/GIMP` install and a Flatpak install under `~/.var/app/org.gimp.GIMP/`. The script is idempotent and preserves an existing settings file.

Requirements: GIMP 3 already installed and launched once, and either `uv` on your `PATH` (`curl -LsSf https://astral.sh/uv/install.sh | sh`) or an existing Python 3.10+.

### Quick install (Windows)

Clone this repo, then in PowerShell (from the repo folder) run:

```powershell
powershell -ExecutionPolicy Bypass -File install-windows.ps1
```

Same behaviour as the other installers (uv when present, stdlib `venv` + `pip` otherwise — **no conda**): builds the `%USERPROFILE%\.gimp-segany\venv` backend, copies the plugin files into `%APPDATA%\GIMP\3.x\plug-ins\seganyplugin\`, and seeds a default `segany_settings.json`. pip pulls the CUDA-enabled torch wheels, so NVIDIA GPUs work out of the box.

Requirements: GIMP 3 already installed and launched once, and either `uv` on your `PATH` (`irm https://astral.sh/uv/install.ps1 | iex`) or an existing Python 3.10+ (the `py` launcher or `python` on `PATH`).

### Manual install

If you prefer to set things up by hand (Windows, or a custom environment), follow the sections below.

#### 1. Plugin files

Copy this repo's `seganyplugin.py` and `seganybridge.py` into a `seganyplugin` folder inside GIMP's user plug-ins directory:

- **Windows:** `C:\Users\[YourUsername]\AppData\Roaming\GIMP\3.x\plug-ins\seganyplugin\`
- **Linux:** `~/.config/GIMP/3.x/plug-ins/seganyplugin/`
- **macOS:** `~/Library/Application Support/GIMP/3.x/plug-ins/seganyplugin/`

Replace `3.x` with the minor version that matches your GIMP install (e.g. `3.2`). You can find the exact path under `Edit → Preferences → Folders → Plug-ins`. On Linux and macOS, `chmod +x seganyplugin.py`.

#### 2. Python environment

Create a Python environment with the SAM2 backend. The dependencies are pinned in [`requirements-lock.txt`](requirements-lock.txt) (a known-good set captured from a working env) and `sam2` is pinned to an exact git commit, so installs are reproducible. Any Python 3.10+ works (3.11 recommended); no conda needed.

With [`uv`](https://docs.astral.sh/uv/) (recommended — provisions Python and installs fast):

```bash
uv venv --python 3.11 ~/.gimp-segany/venv
uv pip install --python ~/.gimp-segany/venv/bin/python -r requirements-lock.txt
SAM2_BUILD_CUDA=0 uv pip install --python ~/.gimp-segany/venv/bin/python \
  "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
```

Or with the stdlib `venv` + `pip` (needs Python 3.10+ already installed):

```bash
python3 -m venv ~/.gimp-segany/venv
~/.gimp-segany/venv/bin/pip install -r requirements-lock.txt
SAM2_BUILD_CUDA=0 ~/.gimp-segany/venv/bin/pip install \
  "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
```

On Linux pip pulls the CUDA-enabled torch wheels automatically. Set the plugin's **Python3 Path** to `~/.gimp-segany/venv/bin/python` (the **Detect…** button finds it).

For SAM1 support (optional), additionally run `pip install "git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf"` and download one of the `sam_vit_*.pth` checkpoints.

To bump the pins later: install the latest versions, verify a segmentation works, then regenerate the `==` lines in `requirements-lock.txt` from the working env and update the `sam2` commit.

#### 3. Point the plugin at it

Open GIMP, then `Image → Segment Anything Layers`, and set:

- **Python3 Path** — the interpreter with SAM2 installed (e.g. `~/.gimp-segany/venv/bin/python` from the installer). The **Detect…** button lists it (and any other envs it finds) automatically.
- **Model Source** — pick one of the curated SAM 2.1 entries to load via Hugging Face, or choose *Custom checkpoint* and point at a local `.pt` / `.safetensors`.

Click **Run Setup Check** to verify the combo works before running a real segmentation. Values are persisted to `segany_settings.json` next to the plugin files.

---

## Plugin Usage

### Dialog options

At the top of the dialog, the **Preset** bar lets you save the current option combo under a name and reload it later. *Save as…* snapshots everything below (except the python interpreter, which is machine-specific) into `segany_settings.json`; *Delete* removes the active preset.

**Interpreter & model:**

- **Python3 Path** — the python used to run `seganybridge.py`. Type/paste a path, click *Browse…*, or click *Detect…* to pick the `~/.gimp-segany/venv` interpreter the installer creates, plus any legacy conda envs and Homebrew/system python3s it still finds.
- **Model Source** — curated Hugging Face presets (SAM 2.1 Large / Base+ / Small / Tiny). Picking one fills *Checkpoint Path* with the HF id and aligns *Model Type*; on first run the weights are downloaded via `huggingface_hub` and cached under `~/.cache/huggingface`. Choose *— Custom checkpoint —* to point at a local file instead.
- **Model Type** — `Auto` infers from the checkpoint name (`sam_*` → SAM1, `sam2*` / `sam2.1*` → SAM2). Override when auto-detection picks the wrong variant.
- **Model Checkpoint** — either a full path to a `.pt` / `.pth` / `.safetensors` file, or a Hugging Face repo id like `facebook/sam2.1-hiera-large`.

**Segmentation:**

- **Segmentation Type:**
  - **Auto** — segments the entire image.
  - **Box** — segments objects inside a rectangular selection.
  - **Selection** — segments objects seeded by sample points drawn from the current selection.
- **Mask Type:**
  - **Multiple** — one layer per candidate mask.
  - **Single** — only the highest-scoring mask.
- **Selection Points** (Selection mode only) — how many seed points to sample from the current selection.

**SAM2-specific options (Auto segmentation):**

- **Segmentation Resolution** — auto-segmentation grid density. Higher = more masks, slower.
- **Crop n Layers** — run segmentation on overlapping crops for better small-object recall.
- **Minimum Mask Area** — drop masks smaller than this many pixels.

**Output:**

- **Random Mask Color** / **Mask Color** — colorize each mask layer randomly, or with a single fixed color.
- **Auto-select from top mask** (on by default) — after segmentation, replace the GIMP selection with the best mask so you can crop / copy / paint against it directly. Turn off to keep your pre-run selection.
- **Show all masks** (on by default) — keep every generated mask layer visible (and the mask group visible) so you can compare candidates in the layer panel. Turn off to only show the top mask and, when combined with *Auto-select*, also hide the mask group entirely — useful when you only want the selection and nothing on screen.

**Run Setup Check** — launches the bridge in test mode with the current interpreter and checkpoint and reports a checklist (python, torch + device, sam2, checkpoint loads, test inference passes). Failed checks surface the actionable hint from the translation table instead of a raw traceback. Use this first whenever a run fails.

### Workflow

1. Pick your options (or load a preset) and click *OK*.
2. The GIMP status bar shows progress: *Loading model…* (or the HF download warning on first run) → *Running … segmentation…*. The first segmentation pays the model-load cost; subsequent ones with the same checkpoint reuse the cached model and start in under a second.
3. **Default (Auto-select + Show all masks both on):** a *Segment Anything – …* group is created with every candidate mask visible as a colored overlay, and the GIMP selection is already set to the top (best-scoring) mask. Layer names include coverage percentage (`Mask - Box #1 (17.3%)`) to help pick a different one if the top mask isn't what you wanted.
4. **Show all masks off + Auto-select on:** the mask group is hidden entirely; you just get the top-mask selection. Good for a pure "give me the selection" workflow.
5. **Auto-select off:** your pre-run selection is restored and the masks stay on screen (or just the top mask, depending on *Show all masks*). Pick a mask by hand and turn it into a selection via *Select → By Color* or *Fuzzy Select*.

---

## Troubleshooting

**Click *Run Setup Check* first.** It translates most install-time errors (missing `sam2`, missing `huggingface_hub`, missing `cv2`, MPS OOM, checkpoint/config mismatch, network errors) into an actionable hint with the exact `pip install` line you need. Run-time failures in the main dialog go through the same translator, so the error dialog already tells you what to fix in almost every case.

If *Run Setup Check* doesn't help, the common less-obvious failure modes are:

- **`Placeholder storage has not been allocated on MPS device`** — known SAM2/MPS issue on Apple Silicon. The bridge now detects this error automatically, reloads the model on CPU, and retries the same job once (you'll see *"MPS backend failed; retrying on CPU…"* in the status bar). If you'd rather skip MPS entirely from the start, launch GIMP with `SEGANY_FORCE_CPU=1` in the environment — CPU is slower but always works.
- **`Failed to build the SAM 2 CUDA extension`** during install — harmless on macOS; SAM2 falls back to its pure-PyTorch path.
- **GIMP does not list the plugin** — check `Edit → Preferences → Folders → Plug-ins`. The installer places files under `…/GIMP/3.x/plug-ins/seganyplugin/`, where `3.x` must match GIMP's minor version. On Linux/macOS the `.py` files must be executable.
- **Mask group is empty / only hidden layers** — you're on an older commit without the *Auto-select* default. Re-run `install.command` or copy the latest `seganyplugin.py` into the plug-ins folder.
