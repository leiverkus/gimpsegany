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

---

## Installation

### 1. Plugin installation

Copy or symlink the contents of this repository into a `seganyplugin` folder inside GIMP's user plug-ins directory:

- **Windows:** `C:\Users\[YourUsername]\AppData\Roaming\GIMP\3.0\plug-ins\seganyplugin\`
- **Linux:** `~/.config/GIMP/3.0/plug-ins/seganyplugin/`
- **macOS:** `~/Library/Application Support/GIMP/3.0/plug-ins/seganyplugin/` (use the matching minor version, e.g. `3.2/plug-ins/`)

You can find your exact plugin location in GIMP under `Edit → Preferences → Folders → Plug-ins`.

On Linux and macOS, make the scripts executable:

```bash
chmod +x ~/Library/Application\ Support/GIMP/3.2/plug-ins/seganyplugin/*.py
```

### 2. Backend installation

You need a Python environment with the SAM backend installed. The plugin can use either SAM1 or SAM2 (or both). The platform-specific guides below are recommended.

#### macOS Apple Silicon (SAM2 with MPS)

Tested on M3 Max, macOS 14+, GIMP 3.2, Miniconda.

**Create a Conda environment with Python 3.11.** SAM2 is most stable on 3.11; 3.13 can break PyTorch extensions.

```bash
conda create -n sam2 python=3.11 -y
conda activate sam2
```

**Install PyTorch with MPS support.** PyTorch automatically enables the MPS (Metal Performance Shaders) backend on Apple Silicon.

```bash
pip install torch torchvision torchaudio
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
# expected: MPS available: True
```

**Install SAM2** (skip the CUDA extension build; macOS has no CUDA):

```bash
cd ~/Library
git clone https://github.com/facebookresearch/sam2.git
cd sam2
SAM2_BUILD_CUDA=0 pip install -e .
```

> The "Failed to build the SAM 2 CUDA extension" warning is expected and harmless on macOS.

**Install OpenCV** (used by the bridge):

```bash
pip install opencv-python
```

**Optional — only if you also want SAM1:**

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

This fork's bridge no longer requires `segment_anything` for SAM2-only usage.

**Download a checkpoint.** Both SAM 2.0 and SAM 2.1 work:

```bash
mkdir -p ~/Library/sam2/checkpoints

# SAM 2.0 Large
curl -L -o ~/Library/sam2/checkpoints/sam2_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# OR SAM 2.1 Large
curl -L -o ~/Library/sam2/checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

#### Linux / Windows (SAM2 with CUDA)

Follow Meta's instructions at https://github.com/facebookresearch/segment-anything-2.

**Prerequisites:** Python ≥ 3.10, PyTorch ≥ 2.3.1.

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
pip install opencv-python
```

Then download a checkpoint (Tiny / Small / Base Plus / Large, version 2.0 or 2.1).

#### SAM1 backend (optional)

Follow Meta's instructions at https://github.com/facebookresearch/segment-anything.

```bash
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
```

Then download a SAM1 checkpoint (`sam_vit_h_4b8939.pth`, `sam_vit_l_0b3195.pth`, or `sam_vit_b_01ec64.pth`).

### 3. Bridge test

Verify your installation by running the bridge directly. Open a terminal and `cd` into the plugin folder.

**SAM2 (or SAM 2.1):**

```bash
/path/to/python ./seganybridge.py auto /path/to/sam2_hiera_large.pt
```

**SAM1:**

```bash
/path/to/python ./seganybridge.py auto /path/to/sam_vit_h_4b8939.pth
```

A `Success!!` message indicates the backend is wired up correctly. On Apple Silicon you should also see `Model moved to MPS (Apple Silicon)`.

### 4. First run in GIMP

Open GIMP and load any image. From `Image → Segment Anything Layers`, open the dialog and fill in:

- **Python3 Path** — the Python interpreter that has the SAM backend installed (e.g. `/opt/miniconda3/envs/sam2/bin/python` on macOS). You can paste the path directly into the field.
- **Checkpoint Path** — the full path to your `.pt` / `.pth` / `.safetensors` checkpoint. Paste or use the Browse button.

The values are persisted to `segany_settings.json` next to the plugin and restored on the next run.

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
