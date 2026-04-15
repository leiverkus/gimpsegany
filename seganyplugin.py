#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gi

gi.require_version("Gimp", "3.0")
from gi.repository import Gimp

gi.require_version("Gdk", "3.0")
from gi.repository import Gdk

gi.require_version("Gegl", "0.4")
from gi.repository import Gio, Gegl
from gi.repository import GLib


import tempfile
import subprocess
import shutil
import threading
import collections
import atexit
from os.path import exists
import random
import os
import sys
import glob
import struct
import json
import logging


def _parse_int(text, default):
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return default


def _looks_like_hf_id(path):
    if not path or os.path.exists(path):
        return False
    if os.path.isabs(path) or path.startswith((".", "~")):
        return False
    if path.endswith((".pt", ".pth", ".safetensors")):
        return False
    return path.count("/") == 1


def _default_python_search_dir():
    """Best-guess start directory for the python interpreter file picker."""
    for candidate in (
        "/opt/miniconda3/envs",
        os.path.expanduser("~/miniconda3/envs"),
        os.path.expanduser("~/anaconda3/envs"),
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
    ):
        if os.path.isdir(candidate):
            return candidate
    return None


def _discover_python_candidates():
    """Enumerate plausible python3 interpreters for the Detect menu.

    Returns (label, absolute_path) tuples for every conda/mamba env under
    the usual roots plus a few system pythons. Used to populate a dropdown
    menu next to the Python path field so users don't have to hand-type or
    click through ``~/Library`` to find their sam2 env.
    """
    candidates = []
    seen = set()

    env_roots = [
        "/opt/miniconda3/envs",
        os.path.expanduser("~/miniconda3/envs"),
        os.path.expanduser("~/anaconda3/envs"),
        os.path.expanduser("~/mambaforge/envs"),
        os.path.expanduser("~/miniforge3/envs"),
    ]
    for root in env_roots:
        if not os.path.isdir(root):
            continue
        for env in sorted(os.listdir(root)):
            py = os.path.join(root, env, "bin", "python")
            if os.path.exists(py) and py not in seen:
                candidates.append((f"{env}  ({py})", py))
                seen.add(py)

    for sys_py in (
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3",
        "/usr/bin/python3",
    ):
        if os.path.exists(sys_py) and sys_py not in seen:
            candidates.append((sys_py, sys_py))
            seen.add(sys_py)

    return candidates


# Curated SAM 2.1 models on Hugging Face. Selecting one of these in the
# Model Source dropdown populates the checkpoint path with the HF repo id;
# the bridge loads via build_sam2_hf() on first use and caches the weights
# under ~/.cache/huggingface, so users don't need to download .pt files
# manually. Index 0 is the "custom" sentinel which leaves the path field
# untouched so users can still point at a local file.
HF_MODELS = [
    # (label shown in dropdown, hf repo id, internal model_type)
    ("— Custom checkpoint —", None, None),
    ("SAM 2.1 Large (~900 MB, best quality)", "facebook/sam2.1-hiera-large", "sam2_hiera_large"),
    ("SAM 2.1 Base+ (~320 MB, balanced)", "facebook/sam2.1-hiera-base-plus", "sam2_hiera_base_plus"),
    ("SAM 2.1 Small (~180 MB, fast)", "facebook/sam2.1-hiera-small", "sam2_hiera_small"),
    ("SAM 2.1 Tiny (~150 MB, fastest)", "facebook/sam2.1-hiera-tiny", "sam2_hiera_tiny"),
]


# Common bridge errors translated into actionable hints. The first substring
# found in the raw stderr wins, so ordering matters — put the most specific
# needles first. The raw traceback is always appended underneath so power
# users can still copy it.
ERROR_HINTS = [
    (
        "No module named 'sam2'",
        "The SAM2 package is not installed in the selected Python env.\n"
        "Pick a different Python3 Path (e.g. your 'sam2' conda env) or install it with:\n"
        "    pip install git+https://github.com/facebookresearch/sam2.git",
    ),
    (
        "No module named 'huggingface_hub'",
        "Hugging Face client is missing in the selected Python env — needed\n"
        "to download models from HF. Install it with:\n"
        "    pip install huggingface_hub",
    ),
    (
        "No module named 'segment_anything'",
        "SAM1 package is not installed. If you want SAM2 instead, pick one of\n"
        "the 'SAM 2.1 …' entries in the Model Source dropdown.",
    ),
    (
        "No module named 'cv2'",
        "OpenCV is not installed in the selected Python env. Install with:\n"
        "    pip install opencv-python",
    ),
    (
        "No module named 'torch'",
        "PyTorch is not installed in the selected Python env. On Apple Silicon\n"
        "install the MPS-enabled build:\n"
        "    pip install torch torchvision",
    ),
    (
        "Torch not compiled with CUDA enabled",
        "PyTorch has no CUDA support. On macOS Apple Silicon this is expected —\n"
        "the plugin uses MPS automatically. If you still see this, your\n"
        "seganybridge.py may be out of date.",
    ),
    (
        "MPS backend out of memory",
        "Apple Silicon GPU ran out of memory. Try a smaller model\n"
        "(SAM 2.1 Small or Tiny) or close other GPU-heavy apps.",
    ),
    (
        "CUDA out of memory",
        "GPU ran out of memory. Try a smaller model (SAM 2.1 Small or Tiny).",
    ),
    (
        "Could not read image",
        "The image file could not be read. This is usually an internal PNG\n"
        "export problem — try flattening the image and retrying.",
    ),
    (
        "ConnectionError",
        "No internet connection for the Hugging Face download. Connect and\n"
        "retry, or point the Checkpoint field at a local .pt file.",
    ),
    (
        "HTTPError",
        "Hugging Face request failed. The model id may be wrong, or the HF\n"
        "service is temporarily unavailable.",
    ),
    (
        "model load failed",
        "The bridge could not load the checkpoint. Common causes:\n"
        "  • wrong Model Type for the checkpoint (SAM 2.0 vs 2.1 mismatch)\n"
        "  • .pt file incomplete or corrupted\n"
        "  • HF id typo\n"
        "Try 'Model Source → SAM 2.1 Large' to get a known-good default.",
    ),
]


def _translate_error(raw):
    """Prepend an actionable hint to a raw bridge error if we recognize it.

    Returns the raw text unchanged when nothing matches, so unknown errors
    still reach the user verbatim.
    """
    if not raw:
        return raw
    for needle, hint in ERROR_HINTS:
        if needle in raw:
            return f"{hint}\n\n--- technical details ---\n{raw}"
    return raw

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk


class DialogValue:
    def __init__(self, filepath):
        data = None
        self.pythonPath = None
        self.modelType = "Auto"
        self.checkPtPath = None
        self.modelSource = None  # label of selected HF preset, None for custom
        self.maskType = "Multiple"
        self.segType = "Auto"
        self.isRandomColor = False
        self.maskColor = [255, 0, 0, 255]
        self.selPtCnt = 10
        self.selBoxPathName = None
        self.segRes = "Medium"
        self.cropNLayers = 0
        self.minMaskArea = 0
        self.autoSelectTopMask = True

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                self.pythonPath = data.get("pythonPath", self.pythonPath)
                self.modelType = data.get("modelType", self.modelType)
                self.checkPtPath = data.get("checkPtPath", self.checkPtPath)
                self.modelSource = data.get("modelSource", self.modelSource)
                self.maskType = data.get("maskType", self.maskType)
                self.segType = data.get("segType", self.segType)
                self.isRandomColor = data.get("isRandomColor", self.isRandomColor)
                self.maskColor = data.get("maskColor", self.maskColor)
                self.selPtCnt = data.get("selPtCnt", self.selPtCnt)
                self.segRes = data.get("segRes", self.segRes)
                self.cropNLayers = data.get("cropNLayers", self.cropNLayers)
                self.minMaskArea = data.get("minMaskArea", self.minMaskArea)
                self.autoSelectTopMask = data.get(
                    "autoSelectTopMask", self.autoSelectTopMask
                )
        except Exception as e:
            logging.info("Error reading json : %s" % e)

    def persist(self, filepath):
        data = {
            "pythonPath": self.pythonPath,
            "modelType": self.modelType,
            "checkPtPath": self.checkPtPath,
            "modelSource": self.modelSource,
            "maskType": self.maskType,
            "segType": self.segType,
            "isRandomColor": self.isRandomColor,
            "maskColor": self.maskColor,
            "selPtCnt": self.selPtCnt,
            "segRes": self.segRes,
            "cropNLayers": self.cropNLayers,
            "minMaskArea": self.minMaskArea,
            "autoSelectTopMask": self.autoSelectTopMask,
        }
        with open(filepath, "w") as f:
            json.dump(data, f)


class OptionsDialog(Gtk.Dialog):
    def __init__(self, image, boxPathDict):
        Gtk.Dialog.__init__(self, title="Segment Anything", transient_for=None, flags=0)
        self.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OK, Gtk.ResponseType.OK
        )

        self.set_default_size(400, 200)

        self.boxPathNames = sorted(boxPathDict.keys())
        self.isGrayScale = image.get_base_type() == Gimp.ImageType.GRAYA_IMAGE
        scriptDir = os.path.dirname(os.path.abspath(__file__))
        self.configFilePath = os.path.join(scriptDir, "segany_settings.json")

        self.values = DialogValue(self.configFilePath)

        grid = Gtk.Grid()
        grid.set_column_spacing(10)
        grid.set_row_spacing(10)
        grid.set_margin_start(10)
        grid.set_margin_end(10)
        grid.set_margin_top(10)
        grid.set_margin_bottom(10)
        self.get_content_area().add(grid)

        # Python Path
        pythonFileLbl = Gtk.Label(label="Python3 Path:", xalign=1)
        self.pythonPathBox, self.pythonPathEntry = self._create_path_chooser(
            "Select Python Path",
            self.values.pythonPath,
            default_folder=_default_python_search_dir(),
            detect_candidates=_discover_python_candidates(),
        )
        grid.attach(pythonFileLbl, 0, 0, 1, 1)
        grid.attach(self.pythonPathBox, 1, 0, 1, 1)

        # Model Source (Hugging Face preset or custom checkpoint)
        modelSourceLbl = Gtk.Label(label="Model Source:", xalign=1)
        self.modelSourceDropDown = Gtk.ComboBoxText()
        for label, _hf_id, _mtype in HF_MODELS:
            self.modelSourceDropDown.append_text(label)
        self.modelSourceDropDown.set_active(self._initial_model_source_index())
        grid.attach(modelSourceLbl, 0, 1, 1, 1)
        grid.attach(self.modelSourceDropDown, 1, 1, 1, 1)

        # Model Type
        modelTypeLbl = Gtk.Label(label="Model Type:", xalign=1)
        self.modelTypeDropDown = Gtk.ComboBoxText()
        self.modelTypeVals = [
            "Auto",
            "vit_h (SAM1)",
            "vit_l (SAM1)",
            "vit_b (SAM1)",
            "sam2_hiera_large (SAM2)",
            "sam2_hiera_base_plus (SAM2)",
            "sam2_hiera_small (SAM2)",
            "sam2_hiera_tiny (SAM2)",
        ]
        for value in self.modelTypeVals:
            self.modelTypeDropDown.append_text(value)

        try:
            active_index = self.modelTypeVals.index(self.values.modelType)
        except ValueError:
            active_index = 0  # Default to Auto
        self.modelTypeDropDown.set_active(active_index)

        grid.attach(modelTypeLbl, 0, 2, 1, 1)
        grid.attach(self.modelTypeDropDown, 1, 2, 1, 1)

        # Checkpoint Path
        checkPtFileLbl = Gtk.Label(
            label="Model Checkpoint (.pth/.safetensors or HF id):", xalign=1
        )
        self.checkPtPathBox, self.checkPtPathEntry = self._create_path_chooser(
            "Select Model Checkpoint Path", self.values.checkPtPath
        )
        grid.attach(checkPtFileLbl, 0, 3, 1, 1)
        grid.attach(self.checkPtPathBox, 1, 3, 1, 1)

        # Segmentation Type
        segTypeLbl = Gtk.Label(label="Segmentation Type:", xalign=1)
        self.segTypeDropDown = Gtk.ComboBoxText()
        self.segTypeVals = ["Auto", "Box", "Selection"]
        for value in self.segTypeVals:
            self.segTypeDropDown.append_text(value)
        self.segTypeDropDown.set_active(self.segTypeVals.index(self.values.segType))
        grid.attach(segTypeLbl, 0, 4, 1, 1)
        grid.attach(self.segTypeDropDown, 1, 4, 1, 1)

        # Mask Type
        self.maskTypeLbl = Gtk.Label(label="Mask Type:", xalign=1)
        self.maskTypeDropDown = Gtk.ComboBoxText()
        self.maskTypeVals = ["Multiple", "Single"]
        for value in self.maskTypeVals:
            self.maskTypeDropDown.append_text(value)
        self.maskTypeDropDown.set_active(self.maskTypeVals.index(self.values.maskType))
        grid.attach(self.maskTypeLbl, 0, 5, 1, 1)
        grid.attach(self.maskTypeDropDown, 1, 5, 1, 1)

        # Selection Points
        self.selPtsLbl = Gtk.Label(label="Selection Points:", xalign=1)
        self.selPtsEntry = Gtk.Entry()
        self.selPtsEntry.set_text(str(self.values.selPtCnt))
        grid.attach(self.selPtsLbl, 0, 6, 1, 1)
        grid.attach(self.selPtsEntry, 1, 6, 1, 1)

        # SAM2 Specific Auto-Segmentation Options
        self.segResLbl = Gtk.Label(label="Segmentation Resolution:", xalign=1)
        self.segResDropDown = Gtk.ComboBoxText()
        self.segResVals = ["Low", "Medium", "High"]
        for value in self.segResVals:
            self.segResDropDown.append_text(value)
        self.segResDropDown.set_active(self.segResVals.index(self.values.segRes))
        grid.attach(self.segResLbl, 0, 7, 1, 1)
        grid.attach(self.segResDropDown, 1, 7, 1, 1)

        self.cropNLayersLbl = Gtk.Label(label="Crop n Layers:", xalign=1)
        self.cropNLayersChk = Gtk.CheckButton()
        self.cropNLayersChk.set_active(self.values.cropNLayers > 0)
        grid.attach(self.cropNLayersLbl, 0, 8, 1, 1)
        grid.attach(self.cropNLayersChk, 1, 8, 1, 1)

        self.minMaskAreaLbl = Gtk.Label(label="Minimum Mask Area:", xalign=1)
        self.minMaskAreaEntry = Gtk.Entry()
        self.minMaskAreaEntry.set_text(str(self.values.minMaskArea))
        grid.attach(self.minMaskAreaLbl, 0, 9, 1, 1)
        grid.attach(self.minMaskAreaEntry, 1, 9, 1, 1)

        # Mask Color
        if not self.isGrayScale:
            self.randColBtn = Gtk.CheckButton(label="Random Mask Color")
            self.randColBtn.set_active(self.values.isRandomColor)
            grid.attach(self.randColBtn, 1, 10, 1, 1)

            self.maskColorLbl = Gtk.Label(label="Mask Color:", xalign=1)
            self.maskColorBtn = Gtk.ColorButton()
            rgba = Gdk.RGBA()
            rgba.parse(
                f"rgb({self.values.maskColor[0]},{self.values.maskColor[1]},{self.values.maskColor[2]})"
            )
            self.maskColorBtn.set_rgba(rgba)
            grid.attach(self.maskColorLbl, 0, 11, 1, 1)
            grid.attach(self.maskColorBtn, 1, 11, 1, 1)

        # Auto-select top mask: after segmentation, convert the best mask
        # into a GIMP selection and hide the mask group so the user can
        # immediately operate on it (crop, copy, paint behind, etc.).
        self.autoSelectChk = Gtk.CheckButton(label="Auto-select from top mask")
        self.autoSelectChk.set_active(self.values.autoSelectTopMask)
        grid.attach(self.autoSelectChk, 1, 12, 1, 1)

        self.connect("map-event", self.on_map_event)
        self.segTypeDropDown.connect("changed", self.update_options_visibility)
        self.modelTypeDropDown.connect("changed", self.update_options_visibility)
        self.checkPtPathEntry.connect("changed", self.update_options_visibility)
        self.modelSourceDropDown.connect("changed", self.on_model_source_changed)
        if not self.isGrayScale:
            self.randColBtn.connect("toggled", self.on_random_toggled)

        self.show_all()

    def _create_path_chooser(
        self,
        title,
        initial_path=None,
        is_folder=False,
        default_folder=None,
        detect_candidates=None,
    ):
        """Combined Entry + Browse button so paths can be typed/pasted or picked.

        When ``detect_candidates`` is a non-empty list of (label, path) tuples,
        a "Detect…" button is added that pops up a menu of those entries;
        picking one sets the entry text. An empty list renders the button
        greyed out so users still see the affordance.
        """
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)

        entry = Gtk.Entry()
        entry.set_hexpand(True)
        if initial_path:
            entry.set_text(initial_path)
        box.pack_start(entry, True, True, 0)

        if detect_candidates is not None:
            detect_btn = Gtk.Button(label="Detect…")
            detect_btn.set_sensitive(bool(detect_candidates))
            menu = Gtk.Menu()
            for cand_label, cand_path in detect_candidates:
                item = Gtk.MenuItem(label=cand_label)

                def _on_activate(_widget, path=cand_path):
                    entry.set_text(path)

                item.connect("activate", _on_activate)
                menu.append(item)
            menu.show_all()

            def on_detect(widget):
                menu.popup_at_widget(
                    widget,
                    Gdk.Gravity.SOUTH_WEST,
                    Gdk.Gravity.NORTH_WEST,
                    None,
                )

            detect_btn.connect("clicked", on_detect)
            box.pack_start(detect_btn, False, False, 0)

        button = Gtk.Button(label="Browse…")

        def on_browse(widget):
            action = (
                Gtk.FileChooserAction.SELECT_FOLDER
                if is_folder
                else Gtk.FileChooserAction.OPEN
            )
            dialog = Gtk.FileChooserDialog(
                title=title,
                parent=self,
                action=action,
            )
            dialog.add_buttons(
                Gtk.STOCK_CANCEL,
                Gtk.ResponseType.CANCEL,
                Gtk.STOCK_OPEN,
                Gtk.ResponseType.OK,
            )
            current = entry.get_text()
            start_folder = None
            if current and os.path.exists(os.path.dirname(current)):
                start_folder = os.path.dirname(current)
            elif default_folder and os.path.isdir(default_folder):
                start_folder = default_folder
            if start_folder:
                dialog.set_current_folder(start_folder)
            # ~/Library is hidden by default on macOS
            dialog.set_show_hidden(True)
            response = dialog.run()
            if response == Gtk.ResponseType.OK:
                entry.set_text(dialog.get_filename())
            dialog.destroy()

        button.connect("clicked", on_browse)
        box.pack_start(button, False, False, 0)

        return box, entry

    def _initial_model_source_index(self):
        """Pick the Model Source dropdown entry matching the persisted state.

        Preference order: explicit ``modelSource`` label from settings → HF id
        match against the current checkpoint path → fall back to Custom (0).
        """
        if self.values.modelSource:
            for i, (label, _hf, _mt) in enumerate(HF_MODELS):
                if label == self.values.modelSource:
                    return i
        if self.values.checkPtPath:
            for i, (_label, hf_id, _mt) in enumerate(HF_MODELS):
                if hf_id and hf_id == self.values.checkPtPath:
                    return i
        return 0

    def on_model_source_changed(self, widget):
        """When a curated HF model is chosen, populate the path + model type."""
        idx = self.modelSourceDropDown.get_active()
        if idx <= 0:
            return  # Custom: leave user input alone
        _label, hf_id, internal_type = HF_MODELS[idx]
        self.checkPtPathEntry.set_text(hf_id)
        if internal_type:
            target = next(
                (v for v in self.modelTypeVals if v.startswith(internal_type + " ")),
                None,
            )
            if target is not None:
                self.modelTypeDropDown.set_active(self.modelTypeVals.index(target))

    def update_options_visibility(self, widget):
        segType = self.segTypeVals[self.segTypeDropDown.get_active()]
        modelType = self.modelTypeVals[self.modelTypeDropDown.get_active()]

        isAuto = segType == "Auto"

        # Determine if SAM1 is being used
        checkpoint_path = self.checkPtPathEntry.get_text().strip() or None
        isSam1_by_filename = (
            modelType == "Auto"
            and checkpoint_path
            and os.path.basename(checkpoint_path).lower().startswith("sam_")
        )
        isSam1_by_type = "(SAM1)" in modelType
        isSam1 = isSam1_by_filename or isSam1_by_type

        self.selPtsLbl.set_visible(segType in ["Selection"])
        self.selPtsEntry.set_visible(segType in ["Selection"])
        self.maskTypeLbl.set_visible(segType not in ["Auto"])
        self.maskTypeDropDown.set_visible(segType not in ["Auto"])

        # Show SAM2-specific options only for Auto mode and not SAM1
        show_sam2_options = isAuto and not isSam1
        self.segResLbl.set_visible(show_sam2_options)
        self.segResDropDown.set_visible(show_sam2_options)
        self.cropNLayersLbl.set_visible(show_sam2_options)
        self.cropNLayersChk.set_visible(show_sam2_options)
        self.minMaskAreaLbl.set_visible(show_sam2_options)
        self.minMaskAreaEntry.set_visible(show_sam2_options)

    def on_random_toggled(self, widget):
        is_random = self.randColBtn.get_active()
        self.maskColorLbl.set_visible(not is_random)
        self.maskColorBtn.set_visible(not is_random)

    def on_map_event(self, widget, event):
        self.update_options_visibility(None)
        if not self.isGrayScale:
            self.on_random_toggled(self.randColBtn)

    def get_values(self):
        python_path = self.pythonPathEntry.get_text().strip()
        self.values.pythonPath = python_path if python_path else None

        # Persist the full model type string for UI restoration
        self.values.modelType = self.modelTypeVals[self.modelTypeDropDown.get_active()]

        checkpoint_path = self.checkPtPathEntry.get_text().strip()
        self.values.checkPtPath = checkpoint_path if checkpoint_path else None

        # Persist the Model Source choice so the dropdown restores next time.
        # Only remember a curated preset if its HF id still matches the path —
        # if the user edited the path by hand, drop back to "custom".
        source_idx = self.modelSourceDropDown.get_active()
        if source_idx > 0:
            label, hf_id, _mt = HF_MODELS[source_idx]
            self.values.modelSource = label if hf_id == self.values.checkPtPath else None
        else:
            self.values.modelSource = None
        self.values.segType = self.segTypeVals[self.segTypeDropDown.get_active()]
        self.values.maskType = self.maskTypeVals[self.maskTypeDropDown.get_active()]
        if hasattr(self, "randColBtn"):
            self.values.isRandomColor = self.randColBtn.get_active()
            rgba = self.maskColorBtn.get_rgba()
            self.values.maskColor = [
                int(rgba.red * 255),
                int(rgba.green * 255),
                int(rgba.blue * 255),
                255,
            ]
        self.values.selPtCnt = _parse_int(
            self.selPtsEntry.get_text(), self.values.selPtCnt
        )
        self.values.segRes = self.segResVals[self.segResDropDown.get_active()]
        self.values.cropNLayers = 1 if self.cropNLayersChk.get_active() else 0
        self.values.minMaskArea = _parse_int(
            self.minMaskAreaEntry.get_text(), self.values.minMaskArea
        )
        self.values.autoSelectTopMask = self.autoSelectChk.get_active()
        self.values.persist(self.configFilePath)

        # Return a copy with the parsed model type for the bridge script
        run_values = self.values
        if run_values.modelType == "Auto":
            run_values.modelType = "auto"
        else:
            run_values.modelType = run_values.modelType.split(" ")[0]

        return run_values


def getPathDict(image):
    return {}


# --- Persistent bridge process -------------------------------------------
#
# GIMP 3 keeps the plugin's Python interpreter alive between menu invocations,
# so we can keep the SAM backend loaded as a long-running subprocess. The
# first segmentation pays the model-load cost; subsequent ones (even with
# different parameters, as long as the same checkpoint is used) skip it.

_bridge_proc = None
_bridge_python = None
_bridge_script = None
_bridge_stderr_buf = collections.deque(maxlen=400)
_bridge_lock = threading.Lock()


def _drain_bridge_stderr(proc, buf):
    """Continuously copy bridge stderr into a ring buffer + the plugin log."""
    try:
        for line in iter(proc.stderr.readline, ""):
            if not line:
                break
            buf.append(line)
            sys.stderr.write("[bridge] " + line)
    except Exception:
        pass


def _shutdown_bridge():
    global _bridge_proc, _bridge_python, _bridge_script
    proc = _bridge_proc
    if proc is None:
        return
    _bridge_proc = None
    _bridge_python = None
    _bridge_script = None
    if proc.poll() is None:
        try:
            proc.stdin.write(json.dumps({"action": "shutdown"}) + "\n")
            proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


atexit.register(_shutdown_bridge)


def _spawn_bridge(pythonPath, scriptPath):
    global _bridge_proc, _bridge_python, _bridge_script, _bridge_stderr_buf
    _bridge_stderr_buf = collections.deque(maxlen=400)
    try:
        proc = subprocess.Popen(
            [pythonPath, scriptPath],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
    except FileNotFoundError as e:
        return None, f"Could not launch python interpreter: {e}"

    _bridge_proc = proc
    _bridge_python = pythonPath
    _bridge_script = scriptPath
    threading.Thread(
        target=_drain_bridge_stderr,
        args=(proc, _bridge_stderr_buf),
        daemon=True,
    ).start()
    return proc, None


def bridgeRun(pythonPath, scriptPath, params, progress_cb=None):
    """Send a job to the (long-running) bridge and wait for the JSON result.

    Returns (success, message). On failure ``message`` contains the most
    recent bridge stderr output, which is shown to the user.

    ``progress_cb`` — if supplied, called as ``progress_cb(text, stage)``
    for every ``{"status": "progress", ...}`` line the bridge emits before
    the terminal ``done`` or ``error`` message. The callback runs on the
    plugin thread, so it must be quick and non-blocking.
    """
    with _bridge_lock:
        proc = _bridge_proc
        if (
            proc is None
            or proc.poll() is not None
            or _bridge_python != pythonPath
            or _bridge_script != scriptPath
        ):
            if proc is not None:
                _shutdown_bridge()
            proc, err = _spawn_bridge(pythonPath, scriptPath)
            if proc is None:
                return False, err

        try:
            proc.stdin.write(json.dumps(params) + "\n")
            proc.stdin.flush()
        except Exception as e:
            _shutdown_bridge()
            return False, f"Failed to send job to bridge: {e}"

        # Read status lines until we see a terminal one (done/error). The
        # bridge may emit any number of {"status": "progress"} lines first.
        while True:
            status_line = proc.stdout.readline()
            if not status_line:
                # bridge died
                _shutdown_bridge()
                tail = "".join(_bridge_stderr_buf)
                return False, "Bridge died unexpectedly.\n\n" + tail

            try:
                status = json.loads(status_line)
            except json.JSONDecodeError:
                # Stray non-JSON line (shouldn't happen — stdout is JSON-only
                # in daemon mode — but be forgiving rather than crashing).
                continue

            kind = status.get("status")
            if kind == "progress":
                if progress_cb is not None:
                    try:
                        progress_cb(status.get("text", ""), status.get("stage"))
                    except Exception:
                        pass  # never let a UI hiccup break the bridge loop
                continue
            if kind == "done":
                return True, ""
            # Error path: combine the JSON summary with the tail of stderr,
            # because the bridge's generic message ("model load failed (see
            # log above)") hides the real traceback from the user — and from
            # _translate_error, which matches substrings against whatever we
            # return here.
            msg = status.get("message", "unknown bridge error")
            stderr_tail = "".join(_bridge_stderr_buf).strip()
            if stderr_tail:
                return False, f"{msg}\n\n{stderr_tail}"
            return False, msg


def unpackBoolArray(filepath):
    with open(filepath, "rb") as file:
        packed_data = bytearray(file.read())

    byte_index = 8  # Skip the first 8 bytes for num_rows and num_cols

    num_rows = struct.unpack(">I", packed_data[:4])[0]
    num_cols = struct.unpack(">I", packed_data[4:8])[0]

    unpacked_data = []
    bit_position = 0

    for _ in range(num_rows):
        unpacked_row = []
        for _ in range(num_cols):
            if bit_position == 0:
                current_byte = packed_data[byte_index]
                byte_index += 1

            boolean_value = (current_byte >> bit_position) & 1
            unpacked_row.append(boolean_value)
            bit_position += 1

            if bit_position == 8:
                bit_position = 0

        unpacked_data.append(unpacked_row)

    return unpacked_data


def readMaskFile(filepath, formatBinary):
    if formatBinary:
        return unpackBoolArray(filepath)
    else:
        mask = []
        with open(filepath, "r") as f:
            lines = f.readlines()
        for line in lines:
            mask.append([val == "1" for val in line])
        return mask


def exportSelection(image, expfile, exportCnt):
    procedure = Gimp.get_pdb().lookup_procedure("gimp-selection-bounds")
    config = procedure.create_config()
    config.set_property("image", image)
    result = procedure.run(config)
    non_empty = result.index(1)
    x1 = result.index(2)
    y1 = result.index(3)
    x2 = result.index(4)
    y2 = result.index(5)

    if not non_empty:
        return

    coords = []
    numPts = (x2 - x1) * (y2 - y1)
    if exportCnt >= numPts:
        selIdxs = range(numPts)
    else:
        selIdxs = random.sample(range(numPts), exportCnt)
    for selIdx in selIdxs:
        x = x1 + selIdx % (x2 - x1)
        y = y1 + int(selIdx / (x2 - x1))

        procedure = Gimp.get_pdb().lookup_procedure("gimp-selection-value")
        config = procedure.create_config()
        config.set_property("image", image)
        config.set_property("x", float(x))
        config.set_property("y", float(y))
        result = procedure.run(config)
        value = result.index(1)

        if value > 200:
            coords.append((x, y))
    with open(expfile, "w") as f:
        for co in coords:
            f.write(str(co[0]) + " " + str(co[1]) + "\n")


def getRandomColor(layerCnt):
    uniqueColors = set()
    while len(uniqueColors) < layerCnt:
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)

        color = (red, green, blue)

        if color not in uniqueColors:
            uniqueColors.add(color)
    return list(uniqueColors)


def createLayers(image, maskFileNoExt, userSelColor, formatBinary, values):
    width, height = image.get_width(), image.get_height()
    total_pixels = max(width * height, 1)

    idx = 0
    maxLayers = 99999

    parent = Gimp.GroupLayer.new(image)
    parent.set_name(f"Segment Anything - {values.segType}")
    image.insert_layer(parent, None, 0)
    parent.set_opacity(50)

    uniqueColors = getRandomColor(layerCnt=999)

    if image.get_base_type() == Gimp.ImageType.GRAYA_IMAGE:
        layerType = Gimp.ImageType.GRAYA_IMAGE
        userSelColor = [100, 255]
        babl_format = "YA u8"
        pix_size = 2
    else:
        layerType = Gimp.ImageType.RGBA_IMAGE
        babl_format = "RGBA u8"
        pix_size = 4

    top_layer = None  # first mask created (SAM ranks by score, so it's the best)

    while idx < maxLayers:
        filepath = maskFileNoExt + str(idx) + ".seg"
        if exists(filepath):
            print("Creating Layer..", (idx + 1))
            newlayer = Gimp.Layer.new(
                image,
                f"Mask - {values.segType} #{idx + 1}",
                width,
                height,
                layerType,
                100.0,
                Gimp.LayerMode.NORMAL,
            )
            buffer = newlayer.get_buffer()
            image.insert_layer(newlayer, parent, 0)

            rect = Gegl.Rectangle.new(0, 0, width, height)

            maskVals = readMaskFile(filepath, formatBinary)
            maskColor = (
                userSelColor
                if userSelColor is not None
                else list(uniqueColors[idx]) + [255]
            )

            mask_color_bytes = bytes(maskColor)
            transparent_pixel = bytes(pix_size)
            row_byte_strings = []
            white_pixels = 0
            for row in maskVals:
                row_pixels = []
                for p in row:
                    if p:
                        white_pixels += 1
                        row_pixels.append(mask_color_bytes)
                    else:
                        row_pixels.append(transparent_pixel)
                row_byte_strings.append(b"".join(row_pixels))
            pixels = b"".join(row_byte_strings)

            buffer.set(rect, babl_format, pixels)

            # Rename to include coverage so the layer list is self-describing
            # and easy to sort/pick from — the top mask isn't always #1.
            coverage_pct = (white_pixels / total_pixels) * 100.0
            newlayer.set_name(
                f"Mask - {values.segType} #{idx + 1} ({coverage_pct:.1f}%)"
            )

            # Hide all masks except the first (best-scoring) one so the user
            # sees something immediately instead of an empty layer group.
            if idx == 0:
                top_layer = newlayer
            else:
                newlayer.set_visible(False)

            idx += 1
            newlayer.update(0, 0, width, height)
        else:
            break
    # Gimp.displays_flush()  # turn on only if needed

    return idx, parent, top_layer


def cleanup(filepathPrefix):
    for f in glob.glob(filepathPrefix + "*"):
        os.remove(f)


def showError(message):
    dialog = Gtk.MessageDialog(
        None,
        Gtk.DialogFlags.MODAL | Gtk.DialogFlags.DESTROY_WITH_PARENT,
        Gtk.MessageType.ERROR,
        Gtk.ButtonsType.OK,
        message,
    )

    dialog.run()
    dialog.destroy()


def validateOptions(image, values):
    if not values.checkPtPath or (
        not os.path.exists(values.checkPtPath)
        and not _looks_like_hf_id(values.checkPtPath)
    ):
        showError(
            "Checkpoint path is not set, does not exist, "
            "and is not a Hugging Face model id:\n"
            + (values.checkPtPath or "(empty)")
        )
        return False

    pythonPath = values.pythonPath or "python"
    if not (os.path.isabs(pythonPath) and os.path.exists(pythonPath)):
        if shutil.which(pythonPath) is None:
            showError(
                "Python interpreter not found:\n"
                + pythonPath
                + "\n\nSet the full path in the Python3 Path field."
            )
            return False

    if values.segType in {"Selection", "Box"}:
        procedure = Gimp.get_pdb().lookup_procedure("gimp-selection-is-empty")
        config = procedure.create_config()
        config.set_property("image", image)
        result = procedure.run(config)
        isSelEmpty = result.index(1)
        if isSelEmpty:
            showError(
                "No Selection! For the Segmentation Types: "
                + "Selection to work you need "
                + "to select an area on the image"
            )
            return False
    return True


def configLogging(level):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_segmentation(image, values):
    configLogging(logging.DEBUG)
    if not validateOptions(image, values):
        return

    pythonPath = values.pythonPath or "python"

    formatBinary = True
    filePrefix = "__seg__"
    filepathPrefix = os.path.join(tempfile.gettempdir(), filePrefix)
    selFile = filepathPrefix + "sel__.txt"
    maskFileNoExt = filepathPrefix + "mask__"

    cleanup(filepathPrefix)

    currDir = os.path.dirname(os.path.realpath(__file__))
    scriptFilepath = os.path.join(currDir, "seganybridge.py")

    ipFilePath = filepathPrefix + next(tempfile._get_candidate_names()) + ".png"

    params = {
        "checkpoint_path": values.checkPtPath,
        "model_type": values.modelType,
        "image_path": ipFilePath,
        "seg_type": values.segType,
        "mask_type": values.maskType,
        "save_prefix": maskFileNoExt,
        "format_binary": formatBinary,
    }

    if values.segType == "Auto":
        isSam1_by_type = values.modelType in ["vit_h", "vit_l", "vit_b"]
        isSam1_by_filename = (
            values.modelType == "auto"
            and values.checkPtPath
            and os.path.basename(values.checkPtPath).lower().startswith("sam_")
        )
        if not (isSam1_by_type or isSam1_by_filename):
            params["seg_res"] = values.segRes
            params["crop_n_layers"] = values.cropNLayers
            params["min_mask_area"] = values.minMaskArea

    newImage = image.duplicate()
    newImage.merge_visible_layers(Gimp.MergeType.CLIP_TO_IMAGE)

    procedure = Gimp.get_pdb().lookup_procedure("file-png-export")
    config = procedure.create_config()
    config.set_property("run-mode", Gimp.RunMode.NONINTERACTIVE)
    config.set_property("image", newImage)

    gfile = Gio.File.new_for_path(ipFilePath)
    config.set_property("file", gfile)
    config.set_property("interlaced", False)
    config.set_property("compression", 9)
    config.set_property("bkgd", False)
    config.set_property("offs", False)
    config.set_property("phys", False)
    config.set_property("time", False)
    config.set_property("save-transparent", True)
    config.set_property("optimize-palette", False)
    procedure.run(config)

    newImage.delete()

    procedure = Gimp.get_pdb().lookup_procedure("gimp-selection-save")
    config = procedure.create_config()
    config.set_property("image", image)
    result = procedure.run(config)
    channel = result.index(1)

    if values.segType == "Selection":
        exportSelection(image, selFile, values.selPtCnt)
        params["sel_file"] = selFile
    elif values.segType == "Box":
        procedure = Gimp.get_pdb().lookup_procedure("gimp-selection-bounds")
        config = procedure.create_config()
        config.set_property("image", image)
        result = procedure.run(config)
        x1 = result.index(2)
        y1 = result.index(3)
        x2 = result.index(4)
        y2 = result.index(5)
        params["box_coords"] = [x1, y1, x2, y2]

    procedure = Gimp.get_pdb().lookup_procedure("gimp-selection-none")
    config = procedure.create_config()
    config.set_property("image", image)
    procedure.run(config)

    Gimp.progress_init("Running Segment Anything…")

    def _progress(text, _stage):
        # GIMP 3 Python bindings expose both set_text + pulse; fall back to
        # re-initing if a given install lacks set_text.
        try:
            Gimp.progress_set_text(text)
        except Exception:
            try:
                Gimp.progress_init(text)
            except Exception:
                pass
        try:
            Gimp.progress_pulse()
        except Exception:
            pass

    try:
        success, stderr_text = bridgeRun(
            pythonPath, scriptFilepath, params, progress_cb=_progress
        )
    finally:
        Gimp.progress_end()

    if not success:
        cleanup(filepathPrefix)
        raw = stderr_text.strip() or "(no stderr)"
        showError("Segment Anything bridge failed:\n\n" + _translate_error(raw))
        return

    layerMaskColor = None if values.isRandomColor else values.maskColor
    _count, parent_group, top_layer = createLayers(
        image, maskFileNoExt, layerMaskColor, formatBinary, values
    )
    cleanup(filepathPrefix)

    if values.autoSelectTopMask and top_layer is not None:
        # Convert the top mask directly into a GIMP selection and hide the
        # overlay group so the user can immediately crop/copy/paint — this
        # is what most people want to do with the result anyway. We
        # deliberately skip restoring the pre-run selection (saved in
        # `channel`) because the new selection is more useful.
        parent_group.set_visible(False)
        procedure = Gimp.get_pdb().lookup_procedure("gimp-image-select-item")
        config = procedure.create_config()
        config.set_property("image", image)
        config.set_property("operation", Gimp.ChannelOps.REPLACE)
        config.set_property("item", top_layer)
        procedure.run(config)
    elif channel is not None:
        procedure = Gimp.get_pdb().lookup_procedure("gimp-image-select-item")
        config = procedure.create_config()
        config.set_property("image", image)
        config.set_property("operation", Gimp.ChannelOps.REPLACE)
        config.set_property("item", channel)
        procedure.run(config)

    logging.debug("Finished creating segments!")


class SegAnyPlugin(Gimp.PlugIn):
    def do_query_procedures(self):
        return ["seg-any-gimp3"]

    def do_set_i18n(self, procname):
        return False, None, None  # Returning False disables localization

    def do_create_procedure(self, name):
        procedure = Gimp.ImageProcedure.new(
            self, name, Gimp.PDBProcType.PLUGIN, self.seg_any_run, None
        )
        procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
        procedure.set_menu_label("Segment Anything Layers")
        procedure.set_attribution("Shrinivas Kulkarni", "Shrinivas Kulkarni", "2024")
        procedure.add_menu_path("<Image>/Image")
        return procedure

    def seg_any_run(self, procedure, run_mode, image, drawables, config, data):
        boxPathDict = getPathDict(image)
        dialog = OptionsDialog(image, boxPathDict)
        response = dialog.run()

        if response == Gtk.ResponseType.OK:
            values = dialog.get_values()
            image.undo_group_start()
            run_segmentation(image, values)
            image.undo_group_end()

        dialog.destroy()

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


Gimp.main(SegAnyPlugin.__gtype__, sys.argv)
