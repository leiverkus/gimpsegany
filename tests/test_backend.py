"""Unit tests for the GIMP-independent logic in segany_backend.

segany_backend has no `gi` import, so it loads in plain CI Python — these
cover the error-message translation, the HF-id heuristic, int parsing,
interpreter discovery shape, and DialogValue settings persistence.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import segany_backend as b  # noqa: E402


class TestTranslateError:
    def test_known_needle_prepends_hint(self):
        out = b._translate_error("Traceback ... No module named 'torch'")
        assert "PyTorch is not installed" in out
        assert "--- technical details ---" in out
        assert "No module named 'torch'" in out  # raw kept underneath

    def test_first_matching_needle_wins(self):
        # both 'model load failed' and 'No module named cv2' could appear;
        # the most specific (cv2) is listed first and should win.
        out = b._translate_error("No module named 'cv2' ... model load failed")
        assert "OpenCV is not installed" in out

    def test_unknown_error_passes_through(self):
        assert b._translate_error("some novel error") == "some novel error"

    def test_empty(self):
        assert b._translate_error("") == ""
        assert b._translate_error(None) is None


class TestLooksLikeHfId:
    def test_typical_hf_id(self):
        assert b._looks_like_hf_id("facebook/sam2.1-hiera-large") is True

    def test_local_extensions(self):
        assert b._looks_like_hf_id("model.pt") is False
        assert b._looks_like_hf_id("model.safetensors") is False

    def test_abs_and_home_paths(self):
        assert b._looks_like_hf_id("/abs/model") is False
        assert b._looks_like_hf_id("~/model") is False

    def test_too_many_slashes(self):
        assert b._looks_like_hf_id("a/b/c") is False

    def test_empty(self):
        assert b._looks_like_hf_id("") is False
        assert b._looks_like_hf_id(None) is False


class TestParseInt:
    def test_valid(self):
        assert b._parse_int("7", 0) == 7
        assert b._parse_int("  3 ", 0) == 3

    def test_invalid_returns_default(self):
        assert b._parse_int("x", 5) == 5
        assert b._parse_int(None, 9) == 9
        assert b._parse_int("", 4) == 4


class TestDiscoverPythonCandidates:
    def test_returns_label_path_tuples(self):
        cands = b._discover_python_candidates()
        assert isinstance(cands, list)
        for entry in cands:
            assert isinstance(entry, tuple) and len(entry) == 2
            label, path = entry
            assert isinstance(label, str) and isinstance(path, str)


class TestHfModels:
    def test_custom_sentinel_first(self):
        assert b.HF_MODELS[0][1] is None  # custom sentinel has no repo id
        # every other entry maps a HF id to an internal model_type
        for _label, hf_id, mtype in b.HF_MODELS[1:]:
            assert hf_id and mtype


class TestDialogValue:
    def test_defaults_on_missing_file(self, tmp_path):
        v = b.DialogValue(str(tmp_path / "does-not-exist.json"))
        assert v.segType == "Auto"
        assert v.maskType == "Multiple"
        assert v.maskColor == [255, 0, 0, 255]
        assert v.autoSelectTopMask is True
        assert v.presets == {}

    def test_round_trip(self, tmp_path):
        f = str(tmp_path / "settings.json")
        v = b.DialogValue(f)
        v.segType = "Box"
        v.maskType = "Single"
        v.checkPtPath = "facebook/sam2.1-hiera-large"
        v.isRandomColor = True
        v.presets = {"fast": {"segType": "Box"}}
        v.persist(f)

        loaded = b.DialogValue(f)
        assert loaded.segType == "Box"
        assert loaded.maskType == "Single"
        assert loaded.checkPtPath == "facebook/sam2.1-hiera-large"
        assert loaded.isRandomColor is True
        assert loaded.presets == {"fast": {"segType": "Box"}}
