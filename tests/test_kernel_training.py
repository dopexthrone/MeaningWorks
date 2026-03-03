"""Tests for kernel/training.py — training data emission."""

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel.training import (
    TrainingExample,
    extract_training_examples,
    emit_jsonl,
    training_stats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_training_dir(tmp_path):
    return tmp_path / "training"


class _FillState(Enum):
    F = "filled"
    P = "partial"
    E = "empty"
    Q = "quarantined"
    C = "candidate"


def _make_cell(fill="F", content="some content", confidence=0.9):
    return SimpleNamespace(
        fill=_FillState[fill],
        content=content,
        confidence=confidence,
    )


def _make_grid(cells_dict=None):
    g = SimpleNamespace()
    g.cells = cells_dict or {}
    return g


def _make_state(known=None):
    s = SimpleNamespace()
    s.known = known or {}
    return s


def _make_result(success=True):
    return SimpleNamespace(success=success, verification={}, insights=[])


# ---------------------------------------------------------------------------
# TrainingExample dataclass
# ---------------------------------------------------------------------------

class TestTrainingExample:
    def test_frozen(self):
        ex = TrainingExample(
            type="positive", postcode="INT.ENT.LOC.SEM.GEN",
            intent_context="build X", output_text="component A",
            confidence=0.9, rejection_reason="", domain="software",
            run_id="run-1",
        )
        with pytest.raises(AttributeError):
            ex.type = "negative"

    def test_fields(self):
        ex = TrainingExample(
            type="negative", postcode="SEM.BHV.LOC.SEM.GEN",
            intent_context="ctx", output_text="out",
            confidence=0.3, rejection_reason="quarantined",
            domain="api", run_id="run-2",
        )
        assert ex.type == "negative"
        assert ex.domain == "api"


# ---------------------------------------------------------------------------
# extract_training_examples
# ---------------------------------------------------------------------------

class TestExtraction:
    def test_filled_cells_produce_positive(self):
        grid = _make_grid({
            "INT.ENT.LOC.SEM.GEN": _make_cell("F", "User component", 0.95),
        })
        state = _make_state({"input": "Build a CRM", "domain": "software"})
        result = _make_result()

        examples = extract_training_examples(grid, result, state, "run-1")
        assert len(examples) == 1
        assert examples[0].type == "positive"
        assert examples[0].confidence == 0.95
        assert examples[0].intent_context == "Build a CRM"

    def test_quarantined_cells_produce_negative(self):
        grid = _make_grid({
            "INT.ENT.LOC.SEM.GEN": _make_cell("Q", "bad content", 0.2),
        })
        state = _make_state({"input": "test"})
        examples = extract_training_examples(grid, _make_result(), state, "run-2")
        assert len(examples) == 1
        assert examples[0].type == "negative"

    def test_partial_cells_produce_instruction(self):
        grid = _make_grid({
            "INT.ENT.LOC.SEM.GEN": _make_cell("P", "partial content", 0.5),
        })
        state = _make_state({"input": "test"})
        examples = extract_training_examples(grid, _make_result(), state, "run-3")
        assert len(examples) == 1
        assert examples[0].type == "instruction"

    def test_empty_cells_skipped(self):
        grid = _make_grid({
            "INT.ENT.LOC.SEM.GEN": _make_cell("E", "", 0.0),
        })
        state = _make_state({"input": "test"})
        examples = extract_training_examples(grid, _make_result(), state, "run-4")
        assert len(examples) == 0

    def test_filled_but_empty_content_skipped(self):
        grid = _make_grid({
            "INT.ENT.LOC.SEM.GEN": _make_cell("F", "", 0.9),
        })
        state = _make_state({"input": "test"})
        examples = extract_training_examples(grid, _make_result(), state, "run-5")
        assert len(examples) == 0

    def test_mixed_cells(self):
        grid = _make_grid({
            "pc1": _make_cell("F", "filled content", 0.9),
            "pc2": _make_cell("Q", "rejected", 0.1),
            "pc3": _make_cell("E", "", 0.0),
            "pc4": _make_cell("P", "partial", 0.4),
            "pc5": _make_cell("C", "candidate", 0.6),
        })
        state = _make_state({"input": "test"})
        examples = extract_training_examples(grid, _make_result(), state, "run-6")
        types = [e.type for e in examples]
        assert "positive" in types
        assert "negative" in types
        assert "instruction" in types
        assert len(examples) == 3  # F, Q, P only

    def test_content_truncation(self):
        long_content = "x" * 5000
        grid = _make_grid({
            "pc1": _make_cell("F", long_content, 0.9),
        })
        state = _make_state({"input": "test"})
        examples = extract_training_examples(grid, _make_result(), state, "run-7")
        assert len(examples[0].output_text) <= 2000

    def test_intent_context_truncation(self):
        long_input = "y" * 1000
        grid = _make_grid({
            "pc1": _make_cell("F", "content", 0.9),
        })
        state = _make_state({"input": long_input})
        examples = extract_training_examples(grid, _make_result(), state, "run-8")
        assert len(examples[0].intent_context) <= 500

    def test_empty_grid(self):
        grid = _make_grid({})
        state = _make_state({"input": "test"})
        examples = extract_training_examples(grid, _make_result(), state, "run-9")
        assert examples == []

    def test_no_cells_attr(self):
        grid = SimpleNamespace()  # no cells attribute
        state = _make_state({"input": "test"})
        examples = extract_training_examples(grid, _make_result(), state, "run-10")
        assert examples == []

    def test_cap_at_200(self):
        cells = {f"pc{i}": _make_cell("F", f"content {i}", 0.9) for i in range(300)}
        grid = _make_grid(cells)
        state = _make_state({"input": "test"})
        examples = extract_training_examples(grid, _make_result(), state, "run-11")
        assert len(examples) == 200

    def test_domain_propagation(self):
        grid = _make_grid({"pc1": _make_cell("F", "x", 0.9)})
        state = _make_state({"input": "test", "domain": "api"})
        examples = extract_training_examples(grid, _make_result(), state, "run-12")
        assert examples[0].domain == "api"

    def test_run_id_propagation(self):
        grid = _make_grid({"pc1": _make_cell("F", "x", 0.9)})
        state = _make_state({"input": "test"})
        examples = extract_training_examples(grid, _make_result(), state, "my-run-id")
        assert examples[0].run_id == "my-run-id"


# ---------------------------------------------------------------------------
# emit_jsonl
# ---------------------------------------------------------------------------

class TestEmitJsonl:
    def test_creates_file(self, tmp_training_dir):
        examples = [TrainingExample(
            type="positive", postcode="pc1", intent_context="ctx",
            output_text="out", confidence=0.9, rejection_reason="",
            domain="software", run_id="r1",
        )]
        path = emit_jsonl(examples, output_dir=tmp_training_dir)
        assert path.exists()
        assert path.suffix == ".jsonl"

    def test_file_content_valid_jsonl(self, tmp_training_dir):
        examples = [
            TrainingExample("positive", "pc1", "ctx", "out1", 0.9, "", "software", "r1"),
            TrainingExample("negative", "pc2", "ctx", "out2", 0.1, "quarantined", "api", "r2"),
        ]
        path = emit_jsonl(examples, output_dir=tmp_training_dir)
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2

        for line in lines:
            rec = json.loads(line)
            assert "type" in rec
            assert "postcode" in rec
            assert "run_id" in rec

    def test_append_mode(self, tmp_training_dir):
        ex1 = [TrainingExample("positive", "pc1", "ctx", "out1", 0.9, "", "software", "r1")]
        ex2 = [TrainingExample("negative", "pc2", "ctx", "out2", 0.1, "rej", "api", "r2")]

        path1 = emit_jsonl(ex1, output_dir=tmp_training_dir)
        path2 = emit_jsonl(ex2, output_dir=tmp_training_dir)
        assert path1 == path2  # same date, same file

        lines = path1.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_empty_examples(self, tmp_training_dir):
        path = emit_jsonl([], output_dir=tmp_training_dir)
        assert path.exists()
        assert path.read_text() == ""

    def test_creates_directory(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "training"
        examples = [TrainingExample("positive", "pc1", "ctx", "out1", 0.9, "", "sw", "r1")]
        path = emit_jsonl(examples, output_dir=nested)
        assert path.exists()


# ---------------------------------------------------------------------------
# training_stats
# ---------------------------------------------------------------------------

class TestTrainingStats:
    def test_empty_dir(self, tmp_training_dir):
        stats = training_stats(output_dir=tmp_training_dir)
        assert stats["total_files"] == 0
        assert stats["total_examples"] == 0

    def test_nonexistent_dir(self):
        stats = training_stats(output_dir=Path("/nonexistent/abc123"))
        assert stats["total_files"] == 0

    def test_counts(self, tmp_training_dir):
        examples = [
            TrainingExample("positive", "pc1", "ctx", "out1", 0.9, "", "software", "r1"),
            TrainingExample("positive", "pc2", "ctx", "out2", 0.8, "", "software", "r2"),
            TrainingExample("negative", "pc3", "ctx", "out3", 0.1, "rej", "api", "r3"),
        ]
        emit_jsonl(examples, output_dir=tmp_training_dir)
        stats = training_stats(output_dir=tmp_training_dir)
        assert stats["total_files"] == 1
        assert stats["total_examples"] == 3
        assert stats["type_counts"]["positive"] == 2
        assert stats["type_counts"]["negative"] == 1
        assert stats["domain_counts"]["software"] == 2
        assert stats["domain_counts"]["api"] == 1

    def test_malformed_line_skipped(self, tmp_training_dir):
        tmp_training_dir.mkdir(parents=True, exist_ok=True)
        f = tmp_training_dir / "2026-01-01.jsonl"
        f.write_text('{"type":"positive","postcode":"pc1","run_id":"r1","domain":"sw"}\nnot-json\n')
        stats = training_stats(output_dir=tmp_training_dir)
        assert stats["total_examples"] == 1
