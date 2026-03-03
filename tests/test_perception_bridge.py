"""Tests for kernel/perception_bridge.py — perception → grid fill translation."""

import pytest
from kernel.perception_bridge import (
    MODALITY_POSTCODES,
    perception_to_fill,
    environment_snapshot_to_fills,
    fusion_signal_to_fill,
    modality_for_postcode,
)


class TestPerceptionToFill:
    """perception_to_fill() conversion tests."""

    def test_screen_modality(self):
        result = perception_to_fill("screen", "VS Code - main.py", 0.8, 1740000000.0)
        assert result["postcode_key"] == "OBS.ENV.APP.WHAT.USR"
        assert result["primitive"] == "VS Code - main.py"
        assert result["confidence"] == 0.8
        assert result["source"] == ("observation:screen:1740000000.0",)

    def test_speech_modality(self):
        result = perception_to_fill("speech", "user speaking about project", 0.9, 1740000001.0)
        assert result["postcode_key"] == "OBS.USR.APP.WHAT.USR"
        assert "user speaking" in result["primitive"]

    def test_camera_modality(self):
        result = perception_to_fill("camera", "user at desk", 0.7, 1740000002.0)
        assert result["postcode_key"] == "OBS.USR.APP.WHERE.USR"

    def test_fusion_modality(self):
        result = perception_to_fill("fusion", "activity detected", 0.6, 1740000003.0)
        assert result["postcode_key"] == "OBS.ENV.APP.HOW.USR"

    def test_unknown_modality_raises(self):
        with pytest.raises(ValueError, match="Unknown modality"):
            perception_to_fill("lidar", "3d scan", 0.5, 1740000000.0)

    def test_confidence_clamped_high(self):
        result = perception_to_fill("screen", "test", 1.5, 1740000000.0)
        assert result["confidence"] == 1.0

    def test_confidence_clamped_low(self):
        result = perception_to_fill("screen", "test", -0.3, 1740000000.0)
        assert result["confidence"] == 0.0

    def test_empty_summary_uses_modality(self):
        result = perception_to_fill("screen", "", 0.5, 1740000000.0)
        assert result["primitive"] == "screen"
        assert result["content"] == ""

    def test_long_summary_truncated_primitive(self):
        long_summary = "A" * 200
        result = perception_to_fill("screen", long_summary, 0.5, 1740000000.0)
        assert len(result["primitive"]) <= 60

    def test_long_summary_truncated_content(self):
        long_summary = "B" * 1000
        result = perception_to_fill("screen", long_summary, 0.5, 1740000000.0)
        assert len(result["content"]) <= 500

    def test_source_provenance_format(self):
        result = perception_to_fill("camera", "test", 0.5, 1740000099.5)
        source = result["source"][0]
        assert source.startswith("observation:camera:")
        assert "1740000099.5" in source

    def test_result_has_all_required_keys(self):
        result = perception_to_fill("screen", "test", 0.5, 1740000000.0)
        required = {"postcode_key", "primitive", "content", "confidence", "source"}
        assert required == set(result.keys())


class TestEnvironmentSnapshotToFills:
    """environment_snapshot_to_fills() batch conversion tests."""

    def test_empty_list(self):
        assert environment_snapshot_to_fills([]) == []

    def test_single_entry(self):
        entries = [("screen", "VS Code", 0.8, 1740000000.0)]
        fills = environment_snapshot_to_fills(entries)
        assert len(fills) == 1
        assert fills[0]["postcode_key"] == "OBS.ENV.APP.WHAT.USR"

    def test_multiple_modalities(self):
        entries = [
            ("screen", "VS Code", 0.8, 1740000000.0),
            ("speech", "talking", 0.7, 1740000001.0),
            ("camera", "at desk", 0.6, 1740000002.0),
        ]
        fills = environment_snapshot_to_fills(entries)
        assert len(fills) == 3

    def test_unknown_modality_skipped(self):
        entries = [
            ("screen", "VS Code", 0.8, 1740000000.0),
            ("lidar", "3d scan", 0.9, 1740000001.0),  # unknown
        ]
        fills = environment_snapshot_to_fills(entries)
        assert len(fills) == 1

    def test_duplicate_modalities_all_included(self):
        entries = [
            ("screen", "VS Code", 0.8, 1740000000.0),
            ("screen", "Terminal", 0.6, 1740000001.0),
        ]
        fills = environment_snapshot_to_fills(entries)
        # Both are valid, both produce fills (same postcode = AX3 revision)
        assert len(fills) == 2

    def test_preserves_confidence(self):
        entries = [("screen", "test", 0.42, 1740000000.0)]
        fills = environment_snapshot_to_fills(entries)
        assert fills[0]["confidence"] == 0.42


class TestFusionSignalToFill:
    """fusion_signal_to_fill() tests."""

    def test_presenting_pattern(self):
        result = fusion_signal_to_fill("presenting", 0.8, ("screen", "speech"), 1740000000.0)
        assert result["postcode_key"] == "OBS.ENV.APP.HOW.USR"
        assert "presenting" in result["primitive"]
        assert "screen, speech" in result["content"]

    def test_away_pattern_no_evidence(self):
        result = fusion_signal_to_fill("away", 0.9, (), 1740000000.0)
        assert "inferred" in result["primitive"]
        assert result["confidence"] == 0.9

    def test_idle_pattern(self):
        result = fusion_signal_to_fill("idle", 0.5, (), 1740000000.0)
        assert "idle" in result["primitive"]

    def test_source_uses_fusion_prefix(self):
        result = fusion_signal_to_fill("focused", 0.6, ("screen",), 1740000000.0)
        assert result["source"][0].startswith("fusion:focused:")

    def test_confidence_clamped(self):
        result = fusion_signal_to_fill("test", 2.0, (), 1740000000.0)
        assert result["confidence"] == 1.0

    def test_long_evidence_truncated_primitive(self):
        evidence = tuple(f"mod_{i}" for i in range(20))
        result = fusion_signal_to_fill("pattern", 0.5, evidence, 1740000000.0)
        assert len(result["primitive"]) <= 60


class TestModalityForPostcode:
    """modality_for_postcode() reverse lookup tests."""

    def test_screen_lookup(self):
        assert modality_for_postcode("OBS.ENV.APP.WHAT.USR") == "screen"

    def test_speech_lookup(self):
        assert modality_for_postcode("OBS.USR.APP.WHAT.USR") == "speech"

    def test_camera_lookup(self):
        assert modality_for_postcode("OBS.USR.APP.WHERE.USR") == "camera"

    def test_fusion_lookup(self):
        assert modality_for_postcode("OBS.ENV.APP.HOW.USR") == "fusion"

    def test_unknown_postcode_returns_none(self):
        assert modality_for_postcode("INT.SEM.ECO.WHAT.SFT") is None

    def test_all_modalities_have_valid_postcodes(self):
        from kernel.cell import parse_postcode
        for modality, pk in MODALITY_POSTCODES.items():
            pc = parse_postcode(pk)
            assert pc.layer == "OBS"
            assert modality_for_postcode(pk) == modality


class TestModalityPostcodesConsistency:
    """Verify MODALITY_POSTCODES use valid axis values."""

    def test_all_postcodes_valid(self):
        from kernel.cell import parse_postcode
        for modality, pk in MODALITY_POSTCODES.items():
            pc = parse_postcode(pk)
            assert pc is not None, f"Invalid postcode for modality {modality}: {pk}"

    def test_no_duplicate_postcodes(self):
        postcodes = list(MODALITY_POSTCODES.values())
        # screen and fusion share ENV concern but different dimensions
        # speech and camera share USR concern but different dimensions
        # All 4 should be unique
        assert len(postcodes) == len(set(postcodes))
