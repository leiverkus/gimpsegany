"""Unit tests for the pure (dependency-free) logic in seganybridge.

These cover the fork's core behaviour changes — SAM 2.0 vs 2.1 config
selection, model-type auto-detection, Hugging Face id heuristics, the MPS
failure detector and the force-CPU switch — without importing torch, OpenCV or
sam2. seganybridge guards those heavy imports, so it stays importable here.
"""

import os
import re
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import seganybridge as b  # noqa: E402


def _grep_version(filename):
    """Read __version__ from a source file without importing it (the plugin
    needs GIMP's gi bindings, which aren't available in CI)."""
    text = open(os.path.join(REPO, filename), encoding="utf-8").read()
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return m.group(1) if m else None


class TestSelectSam2Config:
    def test_sam2_0_large(self):
        assert (
            b.select_sam2_config("sam2_hiera_large.pt", "sam2_hiera_large")
            == "sam2_hiera_l.yaml"
        )

    def test_sam2_1_large_uses_v21_config(self):
        assert (
            b.select_sam2_config("sam2.1_hiera_large.pt", "sam2_hiera_large")
            == "configs/sam2.1/sam2.1_hiera_l.yaml"
        )

    def test_sam2_1_tiny_uses_v21_config(self):
        assert (
            b.select_sam2_config("sam2.1_hiera_tiny.pt", "sam2_hiera_tiny")
            == "configs/sam2.1/sam2.1_hiera_t.yaml"
        )

    def test_version_decided_by_basename_not_dirname(self):
        # A "sam2.1" directory must not flip a plain 2.0 checkpoint to v2.1.
        assert (
            b.select_sam2_config("/models/sam2.1/sam2_hiera_small.pt", "sam2_hiera_small")
            == "sam2_hiera_s.yaml"
        )

    def test_unknown_model_type_falls_back_to_large(self):
        assert b.select_sam2_config("sam2_mystery.pt", "nope") == "sam2_hiera_l.yaml"
        assert (
            b.select_sam2_config("sam2.1_mystery.pt", "nope") == "sam2_hiera_l.yaml"
        )


class TestModelTypeAutoDetect:
    def test_sam2_1_filename_maps_to_internal_type(self):
        strat = b.SAM2Strategy()
        assert (
            strat.get_model_type_from_filename("sam2.1_hiera_large.pt")
            == "sam2_hiera_large"
        )

    def test_safetensors_extension_stripped(self):
        strat = b.SAM2Strategy()
        assert (
            strat.get_model_type_from_filename("sam2_hiera_base_plus.safetensors")
            == "sam2_hiera_base_plus"
        )

    def test_unknown_filename_returns_none(self):
        assert b.SAM2Strategy().get_model_type_from_filename("random.pt") is None

    def test_hf_lookup_covers_both_2_0_and_2_1(self):
        assert b.HF_MODEL_TYPE_LOOKUP["facebook/sam2-hiera-large"] == "sam2_hiera_large"
        assert (
            b.HF_MODEL_TYPE_LOOKUP["facebook/sam2.1-hiera-large"] == "sam2_hiera_large"
        )


class TestDetectStrategy:
    def test_sam1_prefix(self):
        assert isinstance(b._detect_strategy("sam_vit_h_4b8939.pth"), b.SAM1Strategy)

    def test_sam2_prefix(self):
        assert isinstance(b._detect_strategy("sam2_hiera_large.pt"), b.SAM2Strategy)

    def test_sam2_1_prefix(self):
        assert isinstance(b._detect_strategy("sam2.1_hiera_large.pt"), b.SAM2Strategy)

    def test_unknown_prefix_returns_none(self):
        assert b._detect_strategy("mystery_model.pt") is None


class TestIsHfId:
    def test_typical_hf_id(self):
        assert b._is_hf_id("facebook/sam2.1-hiera-large") is True

    def test_local_checkpoint_extension(self):
        assert b._is_hf_id("sam2.1_hiera_large.pt") is False
        assert b._is_hf_id("model.safetensors") is False

    def test_absolute_path(self):
        assert b._is_hf_id("/Users/me/models/sam2.pt") is False

    def test_home_relative_path(self):
        assert b._is_hf_id("~/models/sam2") is False

    def test_too_many_slashes(self):
        assert b._is_hf_id("a/b/c") is False

    def test_existing_path_is_not_hf_id(self, tmp_path):
        f = tmp_path / "weird"
        f.write_text("x")
        assert b._is_hf_id(str(f)) is False


class TestIsMpsFailure:
    def test_placeholder_storage(self):
        assert b._is_mps_failure(
            RuntimeError("Placeholder storage has not been allocated on MPS device")
        )

    def test_mpsndarray(self):
        assert b._is_mps_failure(RuntimeError("MPSNDArray initialization failed"))

    def test_unimplemented_op(self):
        assert b._is_mps_failure(
            NotImplementedError("aten::foo not currently implemented for the MPS device")
        )

    def test_unrelated_error_is_not_mps(self):
        assert not b._is_mps_failure(RuntimeError("out of memory"))
        assert not b._is_mps_failure(ValueError("bad checkpoint"))


class TestForceCpuRequested:
    def test_unset(self, monkeypatch):
        monkeypatch.delenv("SEGANY_FORCE_CPU", raising=False)
        assert b._force_cpu_requested() is False

    def test_truthy_values(self, monkeypatch):
        for v in ("1", "true", "YES", "On"):
            monkeypatch.setenv("SEGANY_FORCE_CPU", v)
            assert b._force_cpu_requested() is True

    def test_falsy_values(self, monkeypatch):
        for v in ("0", "false", "", "no"):
            monkeypatch.setenv("SEGANY_FORCE_CPU", v)
            assert b._force_cpu_requested() is False


class TestVersionDiscipline:
    """The two source files and the changelog must declare the same version."""

    def test_bridge_version_is_semver(self):
        assert re.fullmatch(r"\d+\.\d+\.\d+", b.__version__), b.__version__

    def test_plugin_matches_bridge(self):
        assert _grep_version("seganyplugin.py") == b.__version__

    def test_changelog_top_entry_matches(self):
        text = open(os.path.join(REPO, "CHANGELOG.md"), encoding="utf-8").read()
        # First "## [X.Y.Z]" heading is the current version.
        m = re.search(r"^##\s*\[(\d+\.\d+\.\d+)\]", text, re.MULTILINE)
        assert m, "no versioned heading found in CHANGELOG.md"
        assert m.group(1) == b.__version__


class TestSaveMasks:
    """Round-trip the mask PNG writer. Needs numpy + OpenCV, so it skips in the
    dependency-free CI run and exercises the real writer where they exist."""

    @pytest.fixture(autouse=True)
    def _deps(self):
        self.np = pytest.importorskip("numpy")
        self.cv2 = pytest.importorskip("cv2")

    def _mask(self, h, w, true_cells):
        m = self.np.zeros((h, w), dtype=bool)
        for (y, x) in true_cells:
            m[y, x] = True
        return m

    def test_fixed_color_and_alpha(self, tmp_path):
        prefix = str(tmp_path / "m_")
        mask = self._mask(4, 5, [(0, 0), (1, 2), (3, 4)])
        meta = b.saveMasks([mask], prefix, color=[10, 20, 30])

        assert len(meta) == 1
        assert meta[0]["file"] == prefix + "0.png"
        img = self.cv2.imread(meta[0]["file"], self.cv2.IMREAD_UNCHANGED)
        assert img.shape == (4, 5, 4)  # H, W, BGRA
        # mask cell: BGRA == (b=30, g=20, r=10, a=255)
        assert list(img[0, 0]) == [30, 20, 10, 255]
        # background cell: fully transparent
        assert list(img[0, 1]) == [0, 0, 0, 0]

    def test_coverage_percentage(self, tmp_path):
        prefix = str(tmp_path / "m_")
        mask = self._mask(2, 2, [(0, 0)])  # 1 of 4 pixels
        meta = b.saveMasks([mask], prefix, color=[255, 0, 0])
        assert meta[0]["coverage"] == pytest.approx(25.0)

    def test_multiple_masks_random_color(self, tmp_path):
        prefix = str(tmp_path / "m_")
        masks = [self._mask(3, 3, [(0, 0)]), self._mask(3, 3, [(1, 1), (2, 2)])]
        meta = b.saveMasks(masks, prefix, color=None)
        assert [m["file"] for m in meta] == [prefix + "0.png", prefix + "1.png"]
        for m in meta:
            assert os.path.exists(m["file"])
