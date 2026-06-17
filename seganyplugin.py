#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gi

gi.require_version("Gimp", "3.0")
from gi.repository import Gimp

gi.require_version("Gdk", "3.0")
from gi.repository import Gdk

from gi.repository import Gio
from gi.repository import GLib

# Give the plugin process a recognizable name. On Linux the task list and
# the window's WM_CLASS pick this up; on macOS GLib's name never reaches
# NSProcessInfo, so the menu bar keeps saying "Python" until the Carbon
# TransformProcessType call in _macos_make_accessory_and_front() below.
GLib.set_prgname("gimp-segment-anything")
GLib.set_application_name("GIMP Segment Anything")


def _macos_make_accessory_and_front():
    """Hide the python-launcher Dock icon and pull the dialog to the front.

    GIMP 3 runs python plugins as detached subprocesses, so on macOS the
    plugin shows up as its own app in the Dock (with the python-launcher
    rocket) and loses focus behind GIMP's main window. GTK's quartz
    backend ignores keep_above / skip_taskbar_hint / present_with_time,
    so the fix has to go directly through AppKit or Carbon.

    GIMP's bundled python doesn't ship pyobjc, so we can't talk to
    NSApplication via AppKit. Instead we use the Carbon Process Manager
    through ctypes: TransformProcessType turns this process into a
    "UIElement" (accessory) app — no Dock icon, no app menu, windows
    can still receive focus — and SetFrontProcess raises it above GIMP.
    Both calls are wrapped in try/except so a failure silently no-ops.
    """
    if sys.platform != "darwin":
        return
    try:
        import ctypes
        fw = ctypes.CDLL(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        )

        class ProcessSerialNumber(ctypes.Structure):
            _fields_ = [
                ("highLongOfPSN", ctypes.c_uint32),
                ("lowLongOfPSN", ctypes.c_uint32),
            ]

        fw.GetCurrentProcess.argtypes = [ctypes.POINTER(ProcessSerialNumber)]
        fw.GetCurrentProcess.restype = ctypes.c_int32
        fw.TransformProcessType.argtypes = [
            ctypes.POINTER(ProcessSerialNumber),
            ctypes.c_uint32,
        ]
        fw.TransformProcessType.restype = ctypes.c_int32
        fw.SetFrontProcess.argtypes = [ctypes.POINTER(ProcessSerialNumber)]
        fw.SetFrontProcess.restype = ctypes.c_int32

        psn = ProcessSerialNumber(0, 0)
        if fw.GetCurrentProcess(ctypes.byref(psn)) != 0:
            return
        # kProcessTransformToUIElementApplication = 4 → accessory app
        fw.TransformProcessType(ctypes.byref(psn), 4)
        fw.SetFrontProcess(ctypes.byref(psn))
    except Exception:
        pass


import tempfile
import subprocess
import shutil
from os.path import exists
import random
import os
import sys
import json
import logging

# GIMP-independent backend logic lives in the sibling module (same plug-in
# folder; the running script's directory is on sys.path).
from segany_backend import (
    DialogValue,
    HF_MODELS,
    bridgeRun,
    cleanup,
    configLogging,
    _default_python_search_dir,
    _discover_python_candidates,
    _looks_like_hf_id,
    _parse_int,
    _translate_error,
)

# Fork version. Keep in sync with __version__ in seganybridge.py and the latest
# entry in CHANGELOG.md.
__version__ = "3.1.0"


gi.require_version("Gtk", "3.0")
from gi.repository import Gtk


class OptionsDialog(Gtk.Dialog):
    def __init__(self, image):
        Gtk.Dialog.__init__(
            self,
            title=f"GIMP — Segment Anything (v{__version__})",
            transient_for=None,
            flags=Gtk.DialogFlags.MODAL,
        )
        self.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OK, Gtk.ResponseType.OK
        )

        self.set_default_size(400, 200)
        self.set_modal(True)

        self.isGrayScale = image.get_base_type() == Gimp.ImageType.GRAYA_IMAGE
        scriptDir = os.path.dirname(os.path.abspath(__file__))
        self.configFilePath = os.path.join(scriptDir, "segany_settings.json")

        self.values = DialogValue(self.configFilePath)

        # Preset bar lives above the options grid so preset switching
        # doesn't visually sit in the middle of the option flow.
        self._preset_sentinel = "— no preset —"
        preset_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        preset_box.set_margin_start(10)
        preset_box.set_margin_end(10)
        preset_box.set_margin_top(10)
        preset_box.set_margin_bottom(0)
        preset_box.pack_start(Gtk.Label(label="Preset:"), False, False, 0)
        self.presetDropDown = Gtk.ComboBoxText()
        self.presetDropDown.set_hexpand(True)
        self._rebuild_preset_dropdown()
        preset_box.pack_start(self.presetDropDown, True, True, 0)
        self.presetSaveBtn = Gtk.Button(label="Save as…")
        preset_box.pack_start(self.presetSaveBtn, False, False, 0)
        self.presetDeleteBtn = Gtk.Button(label="Delete")
        preset_box.pack_start(self.presetDeleteBtn, False, False, 0)
        self.get_content_area().pack_start(preset_box, False, False, 0)

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

        # Show all masks: keep every generated mask layer visible instead
        # of hiding all but #1. Overrides the hide-group step of the
        # auto-select flow too, so the colored overlay stays on screen.
        self.showAllMasksChk = Gtk.CheckButton(label="Show all masks")
        self.showAllMasksChk.set_active(self.values.showAllMasks)
        grid.attach(self.showAllMasksChk, 1, 13, 1, 1)

        # Setup check: runs the bridge in test mode with the current
        # python/checkpoint and reports a checklist. Spans both columns so
        # it visually separates from the option rows above.
        self.setupCheckBtn = Gtk.Button(label="Run Setup Check")
        self.setupCheckBtn.connect("clicked", self.on_setup_check_clicked)
        grid.attach(self.setupCheckBtn, 0, 14, 2, 1)

        self.connect("map-event", self.on_map_event)
        self.segTypeDropDown.connect("changed", self.update_options_visibility)
        self.modelTypeDropDown.connect("changed", self.update_options_visibility)
        self.checkPtPathEntry.connect("changed", self.update_options_visibility)
        self.modelSourceDropDown.connect("changed", self.on_model_source_changed)
        self.presetDropDown.connect("changed", self.on_preset_changed)
        self.presetSaveBtn.connect("clicked", self.on_save_preset_clicked)
        self.presetDeleteBtn.connect("clicked", self.on_delete_preset_clicked)
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

    def on_setup_check_clicked(self, widget):
        """Run seganybridge.py in test mode and show a checklist.

        This intentionally spawns a fresh subprocess (not the cached
        daemon) so the test exercises the exact python path + checkpoint
        the user has in the dialog right now, and a bad selection doesn't
        poison the running daemon.
        """
        python_path = self.pythonPathEntry.get_text().strip() or "python"
        checkpoint = self.checkPtPathEntry.get_text().strip()
        if not checkpoint:
            showError(
                "Set a Checkpoint path or pick a Model Source before running "
                "the setup check."
            )
            return

        full_model_type = self.modelTypeVals[self.modelTypeDropDown.get_active()]
        if full_model_type == "Auto":
            model_type = "auto"
        else:
            model_type = full_model_type.split(" ")[0]

        # Warn the user if the check may trigger a large HF download.
        if _looks_like_hf_id(checkpoint):
            warn = Gtk.MessageDialog(
                self,
                Gtk.DialogFlags.MODAL,
                Gtk.MessageType.QUESTION,
                Gtk.ButtonsType.YES_NO,
                (
                    "Setup check will load the model via Hugging Face.\n"
                    "If it isn't cached yet this can download up to ~900 MB "
                    "and take several minutes.\n\nContinue?"
                ),
            )
            resp = warn.run()
            warn.destroy()
            if resp != Gtk.ResponseType.YES:
                return

        bridge_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "seganybridge.py"
        )

        try:
            result = subprocess.run(
                [python_path, bridge_path, model_type, checkpoint],
                capture_output=True,
                text=True,
                timeout=600,
            )
        except FileNotFoundError:
            showError(
                f"Python interpreter not found or not executable:\n{python_path}"
            )
            return
        except subprocess.TimeoutExpired:
            showError(
                "Setup check timed out after 10 minutes. If this is a first-run "
                "Hugging Face download on a slow connection, run the bridge "
                "manually once to cache the weights."
            )
            return

        device = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("{") and "setup_check" in line:
                try:
                    device = json.loads(line).get("device")
                except json.JSONDecodeError:
                    pass

        if result.returncode == 0:
            device_txt = f" (device: {device})" if device else ""
            message = (
                "Setup check passed.\n\n"
                f"  ✓ Python interpreter starts\n"
                f"  ✓ torch available{device_txt}\n"
                f"  ✓ sam2 package importable\n"
                f"  ✓ Checkpoint loads: {checkpoint}\n"
                f"  ✓ Test inference succeeds"
            )
            dlg = Gtk.MessageDialog(
                self,
                Gtk.DialogFlags.MODAL,
                Gtk.MessageType.INFO,
                Gtk.ButtonsType.OK,
                message,
            )
        else:
            raw = (result.stderr + result.stdout).strip() or "(no output)"
            dlg = Gtk.MessageDialog(
                self,
                Gtk.DialogFlags.MODAL,
                Gtk.MessageType.ERROR,
                Gtk.ButtonsType.OK,
                "Setup check failed:\n\n" + _translate_error(raw),
            )
        dlg.run()
        dlg.destroy()

    # ---- Preset handling ------------------------------------------------

    def _rebuild_preset_dropdown(self):
        """Reload the preset combo from ``self.values.presets``."""
        self.presetDropDown.remove_all()
        self.presetDropDown.append_text(self._preset_sentinel)
        for name in sorted(self.values.presets.keys()):
            self.presetDropDown.append_text(name)
        self.presetDropDown.set_active(0)

    def _capture_preset(self):
        """Snapshot current widget state into a dict for storage as a preset.

        pythonPath is deliberately excluded: it's a machine-level setting,
        shared presets shouldn't change which interpreter runs.
        """
        data = {
            "modelType": self.modelTypeVals[self.modelTypeDropDown.get_active()],
            "checkPtPath": self.checkPtPathEntry.get_text().strip() or None,
            "modelSource": None,
            "segType": self.segTypeVals[self.segTypeDropDown.get_active()],
            "maskType": self.maskTypeVals[self.maskTypeDropDown.get_active()],
            "selPtCnt": _parse_int(self.selPtsEntry.get_text(), self.values.selPtCnt),
            "segRes": self.segResVals[self.segResDropDown.get_active()],
            "cropNLayers": 1 if self.cropNLayersChk.get_active() else 0,
            "minMaskArea": _parse_int(
                self.minMaskAreaEntry.get_text(), self.values.minMaskArea
            ),
            "autoSelectTopMask": self.autoSelectChk.get_active(),
            "showAllMasks": self.showAllMasksChk.get_active(),
        }
        source_idx = self.modelSourceDropDown.get_active()
        if source_idx > 0:
            label, hf_id, _mt = HF_MODELS[source_idx]
            if hf_id == data["checkPtPath"]:
                data["modelSource"] = label
        if hasattr(self, "randColBtn"):
            data["isRandomColor"] = self.randColBtn.get_active()
            rgba = self.maskColorBtn.get_rgba()
            data["maskColor"] = [
                int(rgba.red * 255),
                int(rgba.green * 255),
                int(rgba.blue * 255),
                255,
            ]
        return data

    def _apply_preset(self, data):
        """Push a preset dict back into the dialog widgets."""
        # Model source first so its cascade fills path + model type, then
        # explicit values from the preset override the cascade if present.
        source_label = data.get("modelSource")
        source_idx = 0
        if source_label:
            for i, (lbl, _hf, _mt) in enumerate(HF_MODELS):
                if lbl == source_label:
                    source_idx = i
                    break
        self.modelSourceDropDown.set_active(source_idx)

        if data.get("checkPtPath") is not None:
            self.checkPtPathEntry.set_text(data["checkPtPath"])
        model_type = data.get("modelType")
        if model_type in self.modelTypeVals:
            self.modelTypeDropDown.set_active(self.modelTypeVals.index(model_type))

        seg_type = data.get("segType")
        if seg_type in self.segTypeVals:
            self.segTypeDropDown.set_active(self.segTypeVals.index(seg_type))
        mask_type = data.get("maskType")
        if mask_type in self.maskTypeVals:
            self.maskTypeDropDown.set_active(self.maskTypeVals.index(mask_type))

        if "selPtCnt" in data:
            self.selPtsEntry.set_text(str(data["selPtCnt"]))
        seg_res = data.get("segRes")
        if seg_res in self.segResVals:
            self.segResDropDown.set_active(self.segResVals.index(seg_res))
        if "cropNLayers" in data:
            self.cropNLayersChk.set_active(bool(data["cropNLayers"]))
        if "minMaskArea" in data:
            self.minMaskAreaEntry.set_text(str(data["minMaskArea"]))
        if "autoSelectTopMask" in data:
            self.autoSelectChk.set_active(bool(data["autoSelectTopMask"]))
        if "showAllMasks" in data:
            self.showAllMasksChk.set_active(bool(data["showAllMasks"]))

        if hasattr(self, "randColBtn"):
            if "isRandomColor" in data:
                self.randColBtn.set_active(bool(data["isRandomColor"]))
            if "maskColor" in data and isinstance(data["maskColor"], list):
                mc = data["maskColor"]
                if len(mc) >= 3:
                    rgba = Gdk.RGBA()
                    rgba.parse(f"rgb({mc[0]},{mc[1]},{mc[2]})")
                    self.maskColorBtn.set_rgba(rgba)

        self.update_options_visibility(None)
        if hasattr(self, "randColBtn"):
            self.on_random_toggled(self.randColBtn)

    def _prompt_name(self, title, default=""):
        """Tiny modal asking for a string. Returns str or None on cancel."""
        dlg = Gtk.Dialog(title=title, transient_for=self, flags=Gtk.DialogFlags.MODAL)
        dlg.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OK, Gtk.ResponseType.OK,
        )
        entry = Gtk.Entry()
        entry.set_text(default)
        entry.set_activates_default(True)
        entry.set_width_chars(30)
        box = dlg.get_content_area()
        box.set_spacing(10)
        box.set_margin_start(10)
        box.set_margin_end(10)
        box.set_margin_top(10)
        box.set_margin_bottom(10)
        box.pack_start(entry, False, False, 0)
        dlg.set_default_response(Gtk.ResponseType.OK)
        dlg.show_all()
        resp = dlg.run()
        name = entry.get_text().strip() if resp == Gtk.ResponseType.OK else None
        dlg.destroy()
        return name or None

    def on_preset_changed(self, widget):
        idx = self.presetDropDown.get_active()
        if idx <= 0:
            return  # sentinel — leave widgets alone
        name = self.presetDropDown.get_active_text()
        preset = self.values.presets.get(name)
        if preset is not None:
            self._apply_preset(preset)

    def on_save_preset_clicked(self, widget):
        current = self.presetDropDown.get_active_text() or ""
        if current == self._preset_sentinel:
            current = ""
        name = self._prompt_name("Save preset", default=current)
        if not name:
            return
        self.values.presets[name] = self._capture_preset()
        # Persist right away — users expect Save as… to stick even if they
        # later cancel the main dialog.
        self.values.persist(self.configFilePath)
        self._rebuild_preset_dropdown()
        # Re-select the just-saved preset. Values already match the widgets,
        # so the resulting on_preset_changed re-apply is a no-op.
        for i in range(self.presetDropDown.get_model().iter_n_children(None)):
            self.presetDropDown.set_active(i)
            if self.presetDropDown.get_active_text() == name:
                break

    def on_delete_preset_clicked(self, widget):
        idx = self.presetDropDown.get_active()
        if idx <= 0:
            return
        name = self.presetDropDown.get_active_text()
        if name in self.values.presets:
            del self.values.presets[name]
            self.values.persist(self.configFilePath)
        self._rebuild_preset_dropdown()

    def on_map_event(self, widget, event):
        self.update_options_visibility(None)
        if not self.isGrayScale:
            self.on_random_toggled(self.randColBtn)
        # On macOS, hide the rocket Dock icon and pull the window to the
        # front via Carbon APIs (GTK's quartz backend can't do either).
        # No-op on Linux/Windows.
        _macos_make_accessory_and_front()

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
        self.values.showAllMasks = self.showAllMasksChk.get_active()
        self.values.persist(self.configFilePath)

        # Return a copy with the parsed model type for the bridge script
        run_values = self.values
        if run_values.modelType == "Auto":
            run_values.modelType = "auto"
        else:
            run_values.modelType = run_values.modelType.split(" ")[0]

        return run_values


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


def _loadMaskLayer(image, filepath):
    """Load a mask PNG straight into ``image`` as a layer.

    The bridge already wrote the mask as a colored RGBA PNG (opaque over the
    mask, transparent elsewhere), so we just drop it in — GIMP converts the
    layer to the image's type, and its alpha channel doubles as the selection
    mask for "auto-select". Returns the new Gimp.Layer.
    """
    procedure = Gimp.get_pdb().lookup_procedure("gimp-file-load-layer")
    config = procedure.create_config()
    config.set_property("run-mode", Gimp.RunMode.NONINTERACTIVE)
    config.set_property("image", image)
    config.set_property("file", Gio.File.new_for_path(filepath))
    result = procedure.run(config)
    return result.index(1)


def createLayers(image, maskMeta, values):
    """Build the mask layer group from the bridge's per-mask PNG list.

    ``maskMeta`` is the list the bridge returns: ``{"file", "coverage"}`` per
    mask, already ranked by SAM score (first = best). Returns
    ``(count, parentGroup, topLayer)``.
    """
    parent = Gimp.GroupLayer.new(image)
    parent.set_name(f"Segment Anything - {values.segType}")
    image.insert_layer(parent, None, 0)
    parent.set_opacity(50)

    top_layer = None  # first mask (SAM ranks by score, so it's the best)

    for idx, entry in enumerate(maskMeta):
        filepath = entry.get("file")
        if not filepath or not exists(filepath):
            continue
        print("Creating Layer..", (idx + 1))
        newlayer = _loadMaskLayer(image, filepath)
        image.insert_layer(newlayer, parent, 0)

        # Name includes coverage so the layer list is self-describing and easy
        # to sort/pick from — the top mask isn't always #1.
        coverage_pct = float(entry.get("coverage", 0.0))
        newlayer.set_name(
            f"Mask - {values.segType} #{idx + 1} ({coverage_pct:.1f}%)"
        )

        # Hide all masks except the first (best-scoring) one so the user sees
        # something immediately instead of an empty layer group — unless "Show
        # all masks" is on, in which case keep every layer visible so the user
        # can compare candidates directly.
        if idx == 0:
            top_layer = newlayer
        elif not values.showAllMasks:
            newlayer.set_visible(False)

    return len(maskMeta), parent, top_layer


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


def run_segmentation(image, values):
    configLogging(logging.DEBUG)
    if not validateOptions(image, values):
        return

    pythonPath = values.pythonPath or "python"

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
        # [r, g, b] for a fixed mask colour, or None for a random colour per
        # mask. The bridge bakes the colour into each PNG.
        "mask_color": None if values.isRandomColor else values.maskColor[:3],
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
        success, stderr_text, maskMeta = bridgeRun(
            pythonPath, scriptFilepath, params, progress_cb=_progress
        )
    finally:
        Gimp.progress_end()

    if not success:
        cleanup(filepathPrefix)
        raw = stderr_text.strip() or "(no stderr)"
        showError("Segment Anything bridge failed:\n\n" + _translate_error(raw))
        return

    _count, parent_group, top_layer = createLayers(image, maskMeta, values)
    cleanup(filepathPrefix)

    if values.autoSelectTopMask and top_layer is not None:
        # Convert the top mask directly into a GIMP selection so the user
        # can immediately crop/copy/paint. We usually hide the colored
        # overlay group too (more useful), but if "Show all masks" is on
        # the user explicitly wants to see the masks, so keep the group
        # visible. We deliberately skip restoring the pre-run selection
        # (saved in `channel`) because the new selection is more useful.
        if not values.showAllMasks:
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
        # Only enable on RGB/GRAY images (with or without alpha) — the formats
        # the segmentation pipeline supports.
        procedure.set_image_types("RGB*, GRAY*")
        procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
        procedure.set_menu_label("Segment Anything Layers")
        procedure.set_documentation(
            "Segment objects with Meta's Segment Anything",
            "Adds masks as layers using SAM1/SAM2/SAM2.1 via an external Python "
            "backend. Configure the interpreter and model in the dialog; "
            "non-interactive runs reuse the last saved settings.",
            name,
        )
        procedure.set_attribution("Shrinivas Kulkarni", "Shrinivas Kulkarni", "2024")
        procedure.add_menu_path("<Image>/Image")
        return procedure

    def seg_any_run(self, procedure, run_mode, image, drawables, config, data):
        if run_mode == Gimp.RunMode.INTERACTIVE:
            # GimpUi.init() is the documented way to wire a plug-in's GTK UI
            # into GIMP; import it lazily so headless runs don't need it.
            gi.require_version("GimpUi", "3.0")
            from gi.repository import GimpUi

            GimpUi.init("seg-any-gimp3")
            dialog = OptionsDialog(image)
            response = dialog.run()
            values = dialog.get_values() if response == Gtk.ResponseType.OK else None
            dialog.destroy()
            if values is None:
                return procedure.new_return_values(
                    Gimp.PDBStatusType.CANCEL, GLib.Error()
                )
        else:
            # NONINTERACTIVE / WITH_LAST_VALS: this plug-in keeps its parameters
            # in segany_settings.json (not in declared PDB args), so reuse the
            # last saved settings instead of popping a dialog.
            scriptDir = os.path.dirname(os.path.abspath(__file__))
            values = DialogValue(os.path.join(scriptDir, "segany_settings.json"))

        image.undo_group_start()
        run_segmentation(image, values)
        image.undo_group_end()

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


Gimp.main(SegAnyPlugin.__gtype__, sys.argv)
