#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GIMP-independent backend logic for the Segment Anything plug-in.

Everything here is free of GIMP / GTK (``gi``) imports: the bridge IPC client,
interpreter discovery, error-message translation, settings persistence, and
small helpers. Keeping it separate makes it unit-testable without GIMP and
keeps seganyplugin.py focused on the GTK UI and GIMP calls.

seganyplugin.py imports from this module; it sits next to it in the plug-in
folder (Python puts the running script's directory on sys.path, so a plain
``import segany_backend`` resolves).
"""

import os
import sys
import glob
import json
import logging
import subprocess
import threading
import collections
import atexit


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


# Where the bundled installers (install.command / install-linux.sh /
# install-windows.ps1) create the SAM2 backend virtualenv. Kept in sync with
# the SEGANY_VENV in those scripts. venv lays out the interpreter differently
# on Windows (Scripts\python.exe) than on Unix (bin/python).
if os.name == "nt":
    SEGANY_VENV_PY = os.path.expanduser(r"~\.gimp-segany\venv\Scripts\python.exe")
else:
    SEGANY_VENV_PY = os.path.expanduser("~/.gimp-segany/venv/bin/python")


def _default_python_search_dir():
    """Best-guess start directory for the python interpreter file picker."""
    for candidate in (
        os.path.dirname(SEGANY_VENV_PY),
        os.path.expanduser("~/.gimp-segany"),
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
    ):
        if os.path.isdir(candidate):
            return candidate
    return None


def _discover_python_candidates():
    """Enumerate plausible python3 interpreters for the Detect menu.

    Lists the gimp-segany venv first (what the installers create), then any
    legacy conda/mamba envs still on disk, then a few system pythons. Used to
    populate a dropdown next to the Python path field so users don't have to
    hand-type or click through ``~/Library`` to find their backend.
    """
    candidates = []
    seen = set()

    if os.path.exists(SEGANY_VENV_PY):
        candidates.append((f"gimp-segany  ({SEGANY_VENV_PY})", SEGANY_VENV_PY))
        seen.add(SEGANY_VENV_PY)

    # Legacy: surface existing conda/mamba envs so users who set things up the
    # old way (or have other envs) can still pick them. Not required anymore.
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
        "Run the bundled installer (install.command / install-linux.sh), or\n"
        "click Detect… to pick the gimp-segany env, or install it manually:\n"
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
        self.segRes = "Medium"
        self.cropNLayers = 0
        self.minMaskArea = 0
        self.autoSelectTopMask = True
        self.showAllMasks = True
        self.presets = {}  # name -> dict of option values (see _capture_preset)

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
                self.showAllMasks = data.get("showAllMasks", self.showAllMasks)
                loaded_presets = data.get("presets")
                if isinstance(loaded_presets, dict):
                    self.presets = loaded_presets
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
            "showAllMasks": self.showAllMasks,
            "presets": self.presets,
        }
        with open(filepath, "w") as f:
            json.dump(data, f)


# --- Bridge IPC client -------------------------------------------------------
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

    Returns (success, message, masks). On failure ``message`` contains the
    most recent bridge stderr output, which is shown to the user, and ``masks``
    is an empty list. On success ``masks`` is the bridge's per-mask metadata
    list (``{"file", "coverage"}`` each, ranked best-first).

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
                return False, err, []

        try:
            proc.stdin.write(json.dumps(params) + "\n")
            proc.stdin.flush()
        except Exception as e:
            _shutdown_bridge()
            return False, f"Failed to send job to bridge: {e}", []

        # Read status lines until we see a terminal one (done/error). The
        # bridge may emit any number of {"status": "progress"} lines first.
        while True:
            status_line = proc.stdout.readline()
            if not status_line:
                # bridge died
                _shutdown_bridge()
                tail = "".join(_bridge_stderr_buf)
                return False, "Bridge died unexpectedly.\n\n" + tail, []

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
                return True, "", status.get("masks", [])
            # Error path: combine the JSON summary with the tail of stderr,
            # because the bridge's generic message ("model load failed (see
            # log above)") hides the real traceback from the user — and from
            # _translate_error, which matches substrings against whatever we
            # return here.
            msg = status.get("message", "unknown bridge error")
            stderr_tail = "".join(_bridge_stderr_buf).strip()
            if stderr_tail:
                return False, f"{msg}\n\n{stderr_tail}", []
            return False, msg, []


def cleanup(filepathPrefix):
    for f in glob.glob(filepathPrefix + "*"):
        os.remove(f)


def configLogging(level):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
