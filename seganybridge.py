"""
Script to generate Meta Segment Anything masks.

Adapted from:
https://github.com/facebookresearch/segment-anything-2
https://github.com/facebookresearch/segment-anything

Author: Shrinivas Kulkarni

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

import os
import sys
import json
import argparse

# Let PyTorch fall back to CPU for MPS ops that aren't implemented yet in the
# Apple Silicon backend instead of crashing. Must be set before torch is
# imported. Users can override by exporting the variable themselves.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Heavy, install-specific dependencies are imported defensively so the pure
# helper logic in this module stays importable (e.g. by the unit tests) even
# when torch / OpenCV / sam2 aren't present. main() fails fast with the
# original ImportError text — which the plugin knows how to translate — if a
# runtime dependency is actually missing; see _require_runtime_deps().
_IMPORT_ERRORS = {}

try:
    import torch
except ImportError as e:  # pragma: no cover - exercised only without torch
    torch = None
    _IMPORT_ERRORS["torch"] = e

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    np = None
    _IMPORT_ERRORS["numpy"] = e

try:
    import cv2
except ImportError as e:  # pragma: no cover
    cv2 = None
    _IMPORT_ERRORS["cv2"] = e

# SAM2 imports
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError as e:  # pragma: no cover
    build_sam2 = None
    SAM2ImagePredictor = None
    SAM2AutomaticMaskGenerator = None
    _IMPORT_ERRORS["sam2"] = e

# build_sam2_hf is absent on older sam2 installs that still provide build_sam2,
# so it gets its own guard rather than being lumped with the block above.
try:
    from sam2.build_sam import build_sam2_hf
except ImportError:
    build_sam2_hf = None

# Maps Hugging Face model ids to the internal model_type the SAM2 strategy
# expects. Both 2.0 and 2.1 ids are recognised.
HF_MODEL_TYPE_LOOKUP = {
    "facebook/sam2-hiera-tiny": "sam2_hiera_tiny",
    "facebook/sam2-hiera-small": "sam2_hiera_small",
    "facebook/sam2-hiera-base-plus": "sam2_hiera_base_plus",
    "facebook/sam2-hiera-large": "sam2_hiera_large",
    "facebook/sam2.1-hiera-tiny": "sam2_hiera_tiny",
    "facebook/sam2.1-hiera-small": "sam2_hiera_small",
    "facebook/sam2.1-hiera-base-plus": "sam2_hiera_base_plus",
    "facebook/sam2.1-hiera-large": "sam2_hiera_large",
}

# SAM1 imports (optional – only needed when using SAM1 models)
try:
    from segment_anything import (
        sam_model_registry,
        SamAutomaticMaskGenerator as SamAutomaticMaskGenerator_SAM1,
        SamPredictor,
    )
except ImportError:
    sam_model_registry = None
    SamAutomaticMaskGenerator_SAM1 = None
    SamPredictor = None

# --- Utility Functions ---


def packBoolArray(filepath, arr):
    packed_data = bytearray()
    num_rows = len(arr)
    num_cols = len(arr[0])
    packed_data.extend(
        [num_rows >> 24, (num_rows >> 16) & 255, (num_rows >> 8) & 255, num_rows & 255]
    )
    packed_data.extend(
        [num_cols >> 24, (num_cols >> 16) & 255, (num_cols >> 8) & 255, num_cols & 255]
    )
    current_byte = 0
    bit_position = 0
    for row in arr:
        for boolean_value in row:
            if boolean_value:
                current_byte |= 1 << bit_position
            bit_position += 1
            if bit_position == 8:
                packed_data.append(current_byte)
                current_byte = 0
                bit_position = 0
    if bit_position > 0:
        packed_data.append(current_byte)
    with open(filepath, "wb") as f:
        f.write(packed_data)
    return packed_data


def saveMask(filepath, maskArr, formatBinary):
    if formatBinary:
        packBoolArray(filepath, maskArr)
    else:
        with open(filepath, "w") as f:
            for row in maskArr:
                f.write("".join(str(int(val)) for val in row) + "\n")


def saveMasks(masks, saveFileNoExt, formatBinary):
    for i, mask in enumerate(masks):
        filepath = saveFileNoExt + str(i) + ".seg"
        arr = [[val for val in row] for row in mask]
        saveMask(filepath, arr, formatBinary)


# --- Strategy Pattern Implementation ---


class SegmentationStrategy:
    def get_model_type_from_filename(self, model_filename):
        raise NotImplementedError

    def load_model(self, checkPtFilePath, modelType):
        raise NotImplementedError

    def segment_auto(self, sam, cvImage, saveFileNoExt, formatBinary, **kwargs):
        raise NotImplementedError

    def segment_box(self, sam, cvImage, maskType, boxCos, saveFileNoExt, formatBinary):
        raise NotImplementedError

    def segment_sel(
        self, sam, cvImage, maskType, selFile, boxCos, saveFileNoExt, formatBinary
    ):
        raise NotImplementedError

    def run_test(self, sam):
        raise NotImplementedError

    def cleanup(self):
        pass


class SAM1Strategy(SegmentationStrategy):
    MODEL_TYPE_LOOKUP = {
        "sam_vit_h_4b8939": "vit_h",
        "sam_vit_l_0b3195": "vit_l",
        "sam_vit_b_01ec64": "vit_b",
    }

    def get_model_type_from_filename(self, model_filename):
        filename_stem = os.path.splitext(model_filename)[0]
        model_type = self.MODEL_TYPE_LOOKUP.get(filename_stem)
        if model_type:
            print(f"Auto-detected SAM1 model type: {model_type}")
            return model_type
        else:
            print(
                f"Error: Could not auto-detect model type from SAM1 filename: {model_filename}"
            )
            print(
                f"Please use one of the following file names: {list(self.MODEL_TYPE_LOOKUP.keys())}"
            )
            return None

    def load_model(self, checkPtFilePath, modelType, device):
        if sam_model_registry is None:
            print(
                "Error: SAM1 is not installed. Install with: "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
            return None
        try:
            sam = sam_model_registry[modelType](checkpoint=checkPtFilePath)
            sam.to(device=device)
            print("SAM1 Model loaded successfully!")
            return sam
        except Exception as e:
            print(f"Error loading SAM1 model: {e}")
            return None

    def segment_auto(self, sam, cvImage, saveFileNoExt, formatBinary, **kwargs):
        mask_generator = SamAutomaticMaskGenerator_SAM1(sam)
        masks = mask_generator.generate(cvImage)
        masks = [mask["segmentation"] for mask in masks]
        saveMasks(masks, saveFileNoExt, formatBinary)

    def segment_box(self, sam, cvImage, maskType, boxCos, saveFileNoExt, formatBinary):
        predictor = SamPredictor(sam)
        predictor.set_image(cvImage)
        input_box = np.array(boxCos)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=(maskType == "Multiple"),
        )
        saveMasks(masks, saveFileNoExt, formatBinary)

    def segment_sel(
        self, sam, cvImage, maskType, selFile, boxCos, saveFileNoExt, formatBinary
    ):
        pts = []
        with open(selFile, "r") as f:
            lines = f.readlines()
            for line in lines:
                cos = line.split(" ")
                pts.append([int(cos[0]), int(cos[1])])
        predictor = SamPredictor(sam)
        predictor.set_image(cvImage)
        input_point = np.array(pts)
        input_label = np.array([1] * len(input_point))
        input_box = np.array(boxCos) if boxCos else None
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=(maskType == "Multiple"),
        )
        saveMasks(masks, saveFileNoExt, formatBinary)

    def run_test(self, sam):
        npArr = np.zeros((50, 50), np.uint8)
        cvImage = cv2.cvtColor(npArr, cv2.COLOR_GRAY2BGR)
        predictor = SamPredictor(sam)
        predictor.set_image(cvImage)
        input_box = np.array([10, 10, 20, 20])
        predictor.predict(
            point_coords=None, point_labels=None, box=input_box, multimask_output=False
        )


# SAM 2.0 configs live in the sam2 package root; SAM 2.1 configs live under
# configs/sam2.1/. Keyed by the internal model_type.
SAM2_CONFIGS_V2 = {
    "sam2_hiera_tiny": "sam2_hiera_t.yaml",
    "sam2_hiera_small": "sam2_hiera_s.yaml",
    "sam2_hiera_base_plus": "sam2_hiera_b+.yaml",
    "sam2_hiera_large": "sam2_hiera_l.yaml",
}
SAM2_CONFIGS_V21 = {
    "sam2_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


def select_sam2_config(checkpoint_path, model_type):
    """Pick the SAM2 Hydra config for a checkpoint.

    SAM 2.1 checkpoints (filenames starting with ``sam2.1``) need the
    ``configs/sam2.1/`` yamls; SAM 2.0 checkpoints use the package-root yamls.
    Falls back to the large 2.0 config for an unknown model type.
    """
    is_sam21 = os.path.basename(checkpoint_path).startswith("sam2.1")
    configs = SAM2_CONFIGS_V21 if is_sam21 else SAM2_CONFIGS_V2
    return configs.get(model_type, "sam2_hiera_l.yaml")


class SAM2Strategy(SegmentationStrategy):
    MODEL_TYPE_LOOKUP = {
        "sam2_hiera_large": "sam2_hiera_large",
        "sam2_hiera_base_plus": "sam2_hiera_base_plus",
        "sam2_hiera_small": "sam2_hiera_small",
        "sam2_hiera_tiny": "sam2_hiera_tiny",
        "sam2.1_hiera_large": "sam2_hiera_large",
        "sam2.1_hiera_base_plus": "sam2_hiera_base_plus",
        "sam2.1_hiera_small": "sam2_hiera_small",
        "sam2.1_hiera_tiny": "sam2_hiera_tiny",
    }

    def __init__(self):
        self._temp_pth_path = None

    def get_model_type_from_filename(self, model_filename):
        filename_stem = os.path.splitext(model_filename)[0]
        model_type = self.MODEL_TYPE_LOOKUP.get(filename_stem)
        if model_type:
            print(f"Auto-detected SAM2 model type: {model_type}")
            return model_type
        else:
            print(
                f"Error: Could not auto-detect model type from SAM2 filename: {model_filename}"
            )
            print(
                f"Please use one of the following file names (or their .safetensors/.pt equivalents): {list(self.MODEL_TYPE_LOOKUP.keys())}"
            )
            return None

    def _convert_safetensors_to_pth(self, safetensors_path, pth_path):
        try:
            from safetensors.torch import load_file

            state_dict = load_file(safetensors_path)
            checkpoint = {"model": state_dict}
            torch.save(checkpoint, pth_path)
            return True
        except Exception as e:
            print(f"Error converting safetensors to pth: {e}")
            return False

    def load_from_hf(self, hf_id, device):
        if "sam2" in _IMPORT_ERRORS:
            print(str(_IMPORT_ERRORS["sam2"]))
            return None
        if build_sam2_hf is None:
            print(
                "Error: this sam2 install has no build_sam2_hf. "
                "Update sam2 (pip install -U git+https://github.com/facebookresearch/sam2.git)."
            )
            return None
        try:
            print(f"Loading SAM2 from Hugging Face Hub: {hf_id}")
            sam = build_sam2_hf(hf_id, device=device)
            print("SAM2 Model loaded successfully!")
            return sam
        except ImportError as e:
            print(
                f"Error loading SAM2 model from HF: {e}\n"
                "Hint: install the Hugging Face client with: pip install huggingface_hub"
            )
            return None
        except Exception as e:
            print(f"Error loading SAM2 model from HF: {e}")
            return None

    def load_model(self, checkPtFilePath, modelType, device):
        if build_sam2 is None:
            print(str(_IMPORT_ERRORS.get("sam2", "Error: No module named 'sam2'")))
            return None
        config_file = select_sam2_config(checkPtFilePath, modelType)
        actual_checkpoint_path = checkPtFilePath
        if checkPtFilePath.endswith(".safetensors"):
            print("Converting safetensors to pth format...")
            self._temp_pth_path = checkPtFilePath.replace(".safetensors", "_temp.pth")
            if self._convert_safetensors_to_pth(checkPtFilePath, self._temp_pth_path):
                actual_checkpoint_path = self._temp_pth_path
                print(f"Converted to: {self._temp_pth_path}")
            else:
                print("Failed to convert safetensors file")
                return None
        try:
            sam = build_sam2(config_file, actual_checkpoint_path, device=device)
            print("SAM2 Model loaded successfully!")
            return sam
        except Exception as e:
            print(f"Error loading SAM2 model: {e}")
            self.cleanup()
            return None

    def segment_auto(self, sam, cvImage, saveFileNoExt, formatBinary, **kwargs):
        points_per_side = 32
        if kwargs.get("segRes") == "Low":
            points_per_side = 16
        elif kwargs.get("segRes") == "High":
            points_per_side = 64
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            crop_n_layers=kwargs.get("cropNLayers", 0),
            min_mask_region_area=kwargs.get("minMaskArea", 0),
        )
        masks = mask_generator.generate(cvImage)
        masks = [mask["segmentation"] for mask in masks]
        saveMasks(masks, saveFileNoExt, formatBinary)

    def segment_box(self, sam, cvImage, maskType, boxCos, saveFileNoExt, formatBinary):
        predictor = SAM2ImagePredictor(sam)
        predictor.set_image(cvImage)
        input_box = np.array(boxCos)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=(maskType == "Multiple"),
        )
        saveMasks(masks, saveFileNoExt, formatBinary)

    def segment_sel(
        self, sam, cvImage, maskType, selFile, boxCos, saveFileNoExt, formatBinary
    ):
        pts = []
        with open(selFile, "r") as f:
            lines = f.readlines()
            for line in lines:
                cos = line.split(" ")
                pts.append([int(cos[0]), int(cos[1])])
        predictor = SAM2ImagePredictor(sam)
        predictor.set_image(cvImage)
        input_point = np.array(pts)
        input_label = np.array([1] * len(input_point))
        input_box = np.array(boxCos) if boxCos else None
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=(maskType == "Multiple"),
        )
        saveMasks(masks, saveFileNoExt, formatBinary)

    def run_test(self, sam):
        npArr = np.zeros((50, 50), np.uint8)
        cvImage = cv2.cvtColor(npArr, cv2.COLOR_GRAY2BGR)
        predictor = SAM2ImagePredictor(sam)
        predictor.set_image(cvImage)
        input_box = np.array([10, 10, 20, 20])
        predictor.predict(
            point_coords=None, point_labels=None, box=input_box, multimask_output=False
        )

    def cleanup(self):
        if self._temp_pth_path and os.path.exists(self._temp_pth_path):
            os.remove(self._temp_pth_path)
            print(f"Removed temporary file: {self._temp_pth_path}")


# Flipped to True once an MPS run fails with the known "Placeholder storage"
# error so every subsequent (re)load in this process is built on CPU.
_mps_disabled = False


def _force_cpu_requested():
    """Honour an explicit SEGANY_FORCE_CPU=1 escape hatch from the environment."""
    return os.environ.get("SEGANY_FORCE_CPU", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _is_mps_failure(exc):
    """Match the known Apple Silicon MPS errors that a CPU retry can recover."""
    msg = str(exc).lower()
    return (
        "placeholder storage has not been allocated on mps" in msg
        or "mpsndarray" in msg
        or ("mps" in msg and "not currently implemented" in msg)
    )


def _select_device():
    if _force_cpu_requested() or _mps_disabled or torch is None:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _print_device(device):
    if device == "mps":
        print("Model moved to MPS (Apple Silicon)")
    elif device == "cuda":
        print("Model moved to CUDA")
    else:
        print("Model running on CPU")


def _detect_strategy(model_filename):
    name = model_filename.lower()
    if name.startswith("sam_"):
        return SAM1Strategy()
    if name.startswith("sam2"):
        return SAM2Strategy()
    return None


def _is_hf_id(path):
    """Heuristic: looks like ``org/model-name`` (no extension, not a real path)."""
    if os.path.exists(path):
        return False
    if os.path.isabs(path) or path.startswith((".", "~")):
        return False
    if path.endswith((".pt", ".pth", ".safetensors")):
        return False
    return path.count("/") == 1


def _prepare_model(checkpoint_path, model_type):
    device = _select_device()

    if _is_hf_id(checkpoint_path):
        strategy = SAM2Strategy()
        sam = strategy.load_from_hf(checkpoint_path, device)
        if sam is None:
            return None, None, None
        _print_device(device)
        return strategy, sam, device

    model_filename = os.path.basename(checkpoint_path)
    strategy = _detect_strategy(model_filename)
    if strategy is None:
        print(
            f"Error: Could not determine model family from filename: {model_filename}"
        )
        print(
            "Filename must start with 'sam_' for SAM1 or 'sam2' for SAM2, "
            "or be a Hugging Face id like 'facebook/sam2.1-hiera-large'."
        )
        return None, None, None

    if model_type.lower() == "auto":
        model_type = strategy.get_model_type_from_filename(model_filename)
        if not model_type:
            return None, None, None

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return None, None, None

    sam = strategy.load_model(checkpoint_path, model_type, device)
    if sam is None:
        return None, None, None
    _print_device(device)
    return strategy, sam, device


def _run_test(model_type, checkpoint_path):
    global _mps_disabled
    strategy, sam, device = _prepare_model(checkpoint_path, model_type)
    if sam is None:
        return 1
    try:
        strategy.run_test(sam)
    except Exception as e:
        if _mps_disabled or _force_cpu_requested() or not _is_mps_failure(e):
            raise
        print(
            "MPS backend failed during test; retrying on CPU (slower)…",
            file=sys.stderr,
        )
        _mps_disabled = True
        strategy.cleanup()
        strategy, sam, device = _prepare_model(checkpoint_path, model_type)
        if sam is None:
            return 1
        strategy.run_test(sam)
    print("Success!!")
    # Structured line for the plugin's Setup Check button. Kept as the
    # final stdout line so the plugin can grep for `"setup_check"` and
    # extract the device without depending on the surrounding log prose.
    print(json.dumps({
        "setup_check": "ok",
        "device": device,
        "checkpoint": checkpoint_path,
    }))
    strategy.cleanup()
    return 0


def _run_inference(strategy, sam, params):
    """Run a single segmentation using an already-loaded model."""
    image_path = params["image_path"]
    seg_type = params["seg_type"]
    mask_type = params.get("mask_type", "Multiple")
    save_prefix = params["save_prefix"]
    format_binary = params.get("format_binary", True)

    cvImage = cv2.imread(image_path)
    if cvImage is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    cvImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)

    if seg_type == "Auto":
        auto_kwargs = {}
        if isinstance(strategy, SAM2Strategy):
            if "seg_res" in params:
                auto_kwargs["segRes"] = params["seg_res"]
            if "crop_n_layers" in params:
                auto_kwargs["cropNLayers"] = int(params["crop_n_layers"])
            if "min_mask_area" in params:
                auto_kwargs["minMaskArea"] = int(params["min_mask_area"])
        strategy.segment_auto(
            sam, cvImage, save_prefix, format_binary, **auto_kwargs
        )
    elif seg_type in {"Selection", "Box-Selection"}:
        strategy.segment_sel(
            sam,
            cvImage,
            mask_type,
            params["sel_file"],
            params.get("box_coords"),
            save_prefix,
            format_binary,
        )
    elif seg_type == "Box":
        strategy.segment_box(
            sam,
            cvImage,
            mask_type,
            params["box_coords"],
            save_prefix,
            format_binary,
        )
    else:
        raise ValueError(f"Unknown segmentation type: {seg_type}")


def _process_job(params, cache, emit):
    """Load (or reuse a cached) model and run one segmentation job."""
    checkpoint_path = params["checkpoint_path"]
    model_type = params.get("model_type", "auto")
    cache_key = (checkpoint_path, model_type)
    if cache_key in cache:
        strategy, sam = cache[cache_key]
        print(f"Reusing cached model for {os.path.basename(checkpoint_path)}")
    else:
        # First-load messaging: HF ids may trigger a ~900 MB download,
        # local files just stream from disk — tell the user which.
        if _is_hf_id(checkpoint_path):
            emit({
                "status": "progress",
                "stage": "loading_model",
                "text": f"Loading model from Hugging Face ({checkpoint_path})… (first run may download up to ~900 MB)",
            })
        else:
            emit({
                "status": "progress",
                "stage": "loading_model",
                "text": "Loading model…",
            })
        strategy, sam, _ = _prepare_model(checkpoint_path, model_type)
        if sam is None:
            raise RuntimeError("model load failed (see log above)")
        # Temporary .pth (from .safetensors) is no longer needed once
        # the model is built — drop it but keep the loaded model.
        strategy.cleanup()
        cache[cache_key] = (strategy, sam)
    emit({
        "status": "progress",
        "stage": "inferring",
        "text": f"Running {params.get('seg_type', 'segmentation').lower()} segmentation…",
    })
    _run_inference(strategy, sam, params)


def _serve(real_stdout):
    """Daemon mode. Read JSON jobs from stdin, keep models loaded across jobs.

    All informational output goes to stderr; only single-line JSON status
    messages are written to ``real_stdout`` so the plugin can parse them.
    """
    global _mps_disabled
    sys.stdout = sys.stderr  # any stray prints inside strategies go to stderr

    def _emit(obj):
        real_stdout.write(json.dumps(obj) + "\n")
        real_stdout.flush()

    cache = {}  # (checkpoint_path, model_type) -> (strategy, sam)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            params = json.loads(line)
        except json.JSONDecodeError as e:
            _emit({"status": "error", "message": f"invalid JSON: {e}"})
            continue

        if params.get("action") == "shutdown":
            break

        try:
            _process_job(params, cache, _emit)
            print("Done!")
            _emit({"status": "done"})
        except Exception as e:
            # Known Apple Silicon MPS failures are recoverable: rebuild the
            # model on CPU (the cached one lives on the dead device) and retry
            # the same job once.
            if not _mps_disabled and not _force_cpu_requested() and _is_mps_failure(e):
                print(
                    "MPS backend failed; falling back to CPU (slower) and retrying…",
                    file=sys.stderr,
                )
                _mps_disabled = True
                cache.clear()
                _emit({
                    "status": "progress",
                    "stage": "loading_model",
                    "text": "MPS backend failed; retrying on CPU (slower)…",
                })
                try:
                    _process_job(params, cache, _emit)
                    print("Done!")
                    _emit({"status": "done"})
                    continue
                except Exception as retry_exc:
                    e = retry_exc

            import traceback

            traceback.print_exc(file=sys.stderr)
            _emit({"status": "error", "message": str(e)})


def _require_runtime_deps():
    """Abort with the original ImportError text if a dependency needed to load
    or run any model is missing. The text (e.g. ``No module named 'torch'``)
    is what the plugin's error translator matches to show an actionable hint.
    """
    for name in ("torch", "numpy", "cv2"):
        if name in _IMPORT_ERRORS:
            raise SystemExit(str(_IMPORT_ERRORS[name]))


def main():
    _require_runtime_deps()
    parser = argparse.ArgumentParser(
        prog="seganybridge.py",
        description=(
            "SAM segmentation backend for the GIMP plugin. With no arguments "
            "it runs as a daemon, reading JSON jobs from stdin and keeping "
            "models loaded across jobs. Given MODEL_TYPE and CHECKPOINT it "
            "runs a one-shot backend self-test instead."
        ),
        epilog=(
            "Environment: SEGANY_FORCE_CPU=1 skips the MPS/CUDA backends and "
            "runs on CPU."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Both arguments are optional so the bare `seganybridge.py` daemon call and
    # the `seganybridge.py <model_type> <checkpoint>` test call (as issued by
    # the plugin) keep working unchanged.
    parser.add_argument(
        "model_type",
        nargs="?",
        metavar="MODEL_TYPE",
        help=(
            "model type (e.g. sam2_hiera_large) or 'auto' to infer it from the "
            "checkpoint filename; switches to test mode when given"
        ),
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        metavar="CHECKPOINT",
        help=(
            "path to a .pt/.pth/.safetensors checkpoint, or a Hugging Face id "
            "like facebook/sam2.1-hiera-large"
        ),
    )
    args = parser.parse_args()

    if args.model_type is None and args.checkpoint is None:
        _serve(real_stdout=sys.stdout)
        return

    if args.model_type is None or args.checkpoint is None:
        parser.error("test mode requires both MODEL_TYPE and CHECKPOINT")

    sys.exit(_run_test(args.model_type, args.checkpoint))


if __name__ == "__main__":
    main()

