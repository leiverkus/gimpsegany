## GIMP Plugin For Integration With Meta Segment Anything

> **Fork notice:** This is a fork of [Shriinivas/gimpsegany](https://github.com/Shriinivas/gimpsegany) with macOS Apple Silicon support and UI improvements. See [What's new in this fork](#whats-new-in-this-fork).

This GIMP plugin integrates with Meta's AI-based tool Segment Anything, which enables you to effortlessly isolate objects within raster images directly from GIMP.

The plugin supports both **Segment Anything 1 (SAM1)** and **Segment Anything 2 (SAM2)** — including SAM 2.1 checkpoints.

---

## What's new in this fork

- **Apple Silicon / MPS support** — SAM2 models load and run on macOS Apple Silicon via PyTorch's MPS backend. Falls back to CUDA, then CPU.
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

It creates (or reuses) a `sam2` conda env, installs PyTorch / OpenCV / huggingface_hub / sam2, copies `seganyplugin.py` and `seganybridge.py` into the right GIMP plug-ins folder, and seeds a default `segany_settings.json` pointing at `facebook/sam2.1-hiera-large` on Hugging Face — so the first run in GIMP works without any extra configuration. The script is idempotent and preserves an existing settings file.

Requirements: GIMP 3 already installed, and Miniforge/Miniconda on your `PATH` (https://github.com/conda-forge/miniforge).

Then restart GIMP and use `Image → Segment Anything Layers`. The first segmentation downloads the SAM 2.1 Large weights from Hugging Face (~900 MB, cached under `~/.cache/huggingface`); the GIMP status bar shows the download and load progress.

If something goes wrong, click **Run Setup Check** in the plugin dialog — it verifies the interpreter, PyTorch + device, sam2, and the checkpoint load in one go and points at the exact missing piece.

### Manual install

If you prefer to set things up by hand (Linux, Windows, or a custom environment), follow the sections below.

#### 1. Plugin files

Copy this repo's `seganyplugin.py` and `seganybridge.py` into a `seganyplugin` folder inside GIMP's user plug-ins directory:

- **Windows:** `C:\Users\[YourUsername]\AppData\Roaming\GIMP\3.x\plug-ins\seganyplugin\`
- **Linux:** `~/.config/GIMP/3.x/plug-ins/seganyplugin/`
- **macOS:** `~/Library/Application Support/GIMP/3.x/plug-ins/seganyplugin/`

Replace `3.x` with the minor version that matches your GIMP install (e.g. `3.2`). You can find the exact path under `Edit → Preferences → Folders → Plug-ins`. On Linux and macOS, `chmod +x seganyplugin.py`.

#### 2. Python environment

Create a Python environment with the SAM2 backend. On macOS the bundled `environment-macos.yml` is the easiest path:

```bash
conda env create -f environment-macos.yml
conda activate sam2
SAM2_BUILD_CUDA=0 pip install git+https://github.com/facebookresearch/sam2.git
```

On Linux with CUDA, follow Meta's instructions at https://github.com/facebookresearch/sam2 and also `pip install opencv-python huggingface_hub`.

For SAM1 support (optional), additionally run `pip install git+https://github.com/facebookresearch/segment-anything.git` and download one of the `sam_vit_*.pth` checkpoints.

#### 3. Point the plugin at it

Open GIMP, then `Image → Segment Anything Layers`, and set:

- **Python3 Path** — the interpreter with SAM2 installed (e.g. `/opt/miniconda3/envs/sam2/bin/python`). The **Detect…** button lists conda envs it finds automatically.
- **Model Source** — pick one of the curated SAM 2.1 entries to load via Hugging Face, or choose *Custom checkpoint* and point at a local `.pt` / `.safetensors`.

Click **Run Setup Check** to verify the combo works before running a real segmentation. Values are persisted to `segany_settings.json` next to the plugin files.

---

## Plugin Usage

### Options

- **Python3 Path:** The Python executable used to run the `seganybridge.py` backend.
- **Model Type:** SAM model variant. `Auto` infers the type from the checkpoint filename (`sam_*` → SAM1, `sam2*` / `sam2.1*` → SAM2).
- **Checkpoint Path:** Full path to the model checkpoint (`.pth`, `.pt`, or `.safetensors`).
- **Segmentation Type:**
  - **Auto** — segments the entire image automatically.
  - **Box** — segments objects within a rectangular selection.
  - **Selection** — segments objects based on sample points from the current selection.
- **Mask Type:**
  - **Multiple** — one layer per candidate mask.
  - **Single** — only the highest-probability mask.
- **Random Mask Color:** If checked, layers get random colors. Otherwise pick a fixed color.

#### SAM2-specific options (Auto segmentation)

- **Segmentation Resolution:** Density of the auto-segmentation grid. Higher = more masks, slower. (Low / Medium / High)
- **Crop n Layers:** Run segmentation on overlapping crops. Improves accuracy on small objects.
- **Minimum Mask Area:** Discards masks smaller than this area.

### Workflow

1. Pick your options and click OK.
2. The plugin creates a layer group with one or more mask layers.
3. Pick the mask layer for the object you want, then use Fuzzy Selection on it to turn the mask into a selection.
4. Hide the mask group, switch back to the original image layer, and cut/copy/edit as usual.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'cv2'`** — run `pip install opencv-python` in the SAM environment.

**`ModuleNotFoundError: No module named 'sam2'`** — SAM2 is not installed in the Python interpreter you pointed the plugin at. Verify with `which python` after `conda activate sam2`.

**`Torch not compiled with CUDA enabled` (macOS)** — make sure you are running this fork's `seganybridge.py`. The upstream version assumes CUDA.

**`Cannot find primary config 'configs/sam2.1/...'`** — your installed `sam2` package is missing the 2.1 configs. Reinstall from a recent `facebookresearch/sam2` checkout.

**`Unexpected key(s) in state_dict: no_obj_embed_spatial, ...`** — you are loading a SAM 2.1 checkpoint with SAM 2.0 configs. This fork auto-detects 2.1 from filename — make sure the file is named `sam2.1_*`.

**`Placeholder storage has not been allocated on MPS device`** — known SAM2 issue on Apple Silicon. Workaround: force CPU by editing `seganybridge.py` (`device = "cpu"`). Slower but functional.

**`Failed to build the SAM 2 CUDA extension`** — harmless on macOS.

**GIMP does not list the plugin** — check `Edit → Preferences → Folders → Plug-ins`; the folder name must match the GIMP minor version (e.g. `3.2`). Make sure the `.py` files are executable on Linux/macOS.
