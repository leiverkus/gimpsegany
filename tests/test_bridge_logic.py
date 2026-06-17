"""Unit tests for the pure (dependency-free) logic in seganybridge.

These cover the fork's core behaviour changes — SAM 2.0 vs 2.1 config
selection, model-type auto-detection, Hugging Face id heuristics, the MPS
failure detector and the force-CPU switch — without importing torch, OpenCV or
sam2. seganybridge guards those heavy imports, so it stays importable here.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import seganybridge as b  # noqa: E402


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
