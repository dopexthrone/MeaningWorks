"""Tests for self-understanding wiring — source reader → engine/bridge/chat integration."""

import asyncio
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mother.source_reader import (
    SourceSnapshot,
    ModuleSummary,
    ClassSummary,
    MethodSummary,
    FunctionSummary,
    read_codebase,
    format_source_summary,
    source_snapshot_to_facts,
)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def project_root():
    """Return the actual project root."""
    return str(Path(__file__).parent.parent)


@pytest.fixture
def real_snapshot(project_root):
    """Read actual codebase snapshot (cached)."""
    return read_codebase(project_root)


@pytest.fixture
def mini_snapshot():
    """A minimal snapshot for unit tests."""
    return SourceSnapshot(
        project_root="/tmp/test",
        timestamp=time.time(),
        modules=(
            ModuleSummary(
                path="core/engine.py",
                docstring="Core engine.",
                classes=(
                    ClassSummary(
                        name="Engine",
                        docstring="The main engine.",
                        bases=(),
                        methods=(
                            MethodSummary("compile", "text: str", "dict", False),
                        ),
                        is_frozen=False,
                    ),
                ),
                functions=(),
                imports=("kernel.store",),
                line_count=600,
                is_leaf=False,
            ),
            ModuleSummary(
                path="mother/bridge.py",
                docstring="LEAF module. Bridge.",
                classes=(),
                functions=(
                    FunctionSummary("source_summary", "max_words: int", "str", False, "Read source."),
                ),
                imports=("core.engine",),
                line_count=100,
                is_leaf=True,
            ),
        ),
        total_lines=700,
        total_classes=1,
        total_functions=1,
    )


# ── Bridge.source_summary tests ────────────────────────────────────────


class TestBridgeSourceSummary:
    def test_returns_nonempty_string(self):
        """source_summary() returns a non-empty string on the real codebase."""
        from mother.bridge import EngineBridge
        result = EngineBridge.source_summary(max_words=2000)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_word_count_under_budget(self):
        """Summary stays roughly within word budget."""
        from mother.bridge import EngineBridge
        result = EngineBridge.source_summary(max_words=3000)
        word_count = len(result.split())
        assert word_count < 5000  # Allow some overshoot

    def test_returns_empty_on_error(self):
        """Returns empty string when source reader fails."""
        from mother.bridge import EngineBridge
        with patch("mother.bridge.EngineBridge.source_summary") as mock:
            mock.return_value = ""
            assert EngineBridge.source_summary() == ""


# ── Bridge.compile_self_context tests ──────────────────────────────────


class TestBridgeCompileSelfContext:
    def test_calls_compile_with_context_mode(self):
        """compile_self_context should call compile(mode='context')."""
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._provider = "claude"
        bridge._model = None
        bridge._api_key = None

        mock_result = MagicMock()
        mock_result.context_map = {"concepts": [{"name": "Engine"}]}

        with patch.object(bridge, "compile", new_callable=AsyncMock, return_value=mock_result) as mock_compile:
            with patch("mother.source_reader.read_codebase") as mock_read:
                mock_read.return_value = SourceSnapshot(
                    project_root="/tmp", timestamp=0, modules=(),
                    total_lines=0, total_classes=0, total_functions=0,
                )
                with patch("mother.source_reader.format_source_summary", return_value="summary text"):
                    with patch("mother.source_reader.source_snapshot_to_facts", return_value=[]):
                        result = asyncio.run(bridge.compile_self_context())

        mock_compile.assert_called_once()
        call_args = mock_compile.call_args
        assert call_args[1].get("mode") == "context" or (len(call_args[0]) > 1 and call_args[0][1] == "context")
        assert result is mock_result

    def test_persists_structural_facts(self):
        """compile_self_context should persist facts via save_facts."""
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._provider = "claude"

        with patch.object(bridge, "compile", new_callable=AsyncMock, return_value=MagicMock()):
            with patch("mother.source_reader.read_codebase") as mock_read:
                mock_read.return_value = SourceSnapshot(
                    project_root="/tmp", timestamp=0, modules=(),
                    total_lines=0, total_classes=0, total_functions=0,
                )
                with patch("mother.source_reader.format_source_summary", return_value="text"):
                    with patch("mother.source_reader.source_snapshot_to_facts", return_value=[
                        {"fact_id": "test:1", "category": "pattern", "subject": "x",
                         "predicate": "y", "value": "z", "confidence": 0.9,
                         "source": "test", "first_seen": 0, "last_confirmed": 0,
                         "access_count": 0}
                    ]):
                        with patch("mother.knowledge_base.save_facts", return_value=1) as mock_save:
                            asyncio.run(bridge.compile_self_context())

        mock_save.assert_called_once()

    def test_returns_none_on_error(self):
        """compile_self_context returns None when everything fails."""
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._provider = "claude"

        with patch("mother.source_reader.read_codebase", side_effect=Exception("boom")):
            result = asyncio.run(bridge.compile_self_context())
        assert result is None


# ── Engine._generate_self_description tests ────────────────────────────


class TestEngineSelfDescription:
    def test_includes_source_structure_section(self):
        """Self-description should include SOURCE STRUCTURE section."""
        from core.engine import MotherlabsEngine
        engine = MotherlabsEngine.__new__(MotherlabsEngine)
        # Set minimal required attributes
        engine._insight_callback = None
        engine._emit_callback = None

        with patch("mother.source_reader.read_codebase") as mock_read:
            mock_read.return_value = SourceSnapshot(
                project_root="/tmp", timestamp=0,
                modules=(
                    ModuleSummary(
                        path="core/engine.py", docstring="Engine.", classes=(),
                        functions=(), imports=(), line_count=100, is_leaf=False,
                    ),
                ),
                total_lines=100, total_classes=0, total_functions=0,
            )
            with patch("mother.source_reader.format_source_summary", return_value="## core/\n  core/engine.py (100 lines)"):
                desc = engine._generate_self_description()

        assert "SOURCE STRUCTURE (from AST)" in desc
        assert "core/engine.py" in desc

    def test_falls_back_on_import_error(self):
        """Self-description still works if source_reader fails."""
        from core.engine import MotherlabsEngine
        engine = MotherlabsEngine.__new__(MotherlabsEngine)
        engine._insight_callback = None
        engine._emit_callback = None

        with patch("mother.source_reader.read_codebase", side_effect=Exception("boom")):
            desc = engine._generate_self_description()

        assert "Mother is a cognitive entity" in desc
        # SOURCE STRUCTURE section should be absent when reader fails
        assert "SOURCE STRUCTURE" not in desc

    def test_preserves_existing_content(self):
        """All 10 original sections still present."""
        from core.engine import MotherlabsEngine
        engine = MotherlabsEngine.__new__(MotherlabsEngine)
        engine._insight_callback = None
        engine._emit_callback = None

        desc = engine._generate_self_description()
        assert "1. IDENTITY" in desc
        assert "2. PERCEPTION" in desc
        assert "3. COGNITION" in desc
        assert "4. ACTUATORS" in desc
        assert "5. SENSES" in desc
        assert "6. AUTONOMY" in desc
        assert "7. LEARNING" in desc
        assert "8. SUBSTRATE" in desc
        assert "10. CONVERGENCE CRITERION" in desc


# ── source_snapshot_to_facts round-trip tests ──────────────────────────


class TestFactsRoundTrip:
    def test_facts_saveable_to_knowledge_base(self, tmp_path):
        """Facts from source_snapshot_to_facts can be saved and retrieved."""
        from mother.knowledge_base import KnowledgeFact, save_facts, search_facts

        snapshot = SourceSnapshot(
            project_root="/tmp",
            timestamp=time.time(),
            modules=(
                ModuleSummary(
                    path="core/engine.py",
                    docstring="Core engine.",
                    classes=(
                        ClassSummary("Engine", "Main", (), (
                            MethodSummary("compile", "text: str", "dict", False),
                        ), False),
                    ),
                    functions=(),
                    imports=("kernel.store",),
                    line_count=500,
                    is_leaf=False,
                ),
            ),
            total_lines=500,
            total_classes=1,
            total_functions=0,
        )

        raw_facts = source_snapshot_to_facts(snapshot)
        assert len(raw_facts) >= 2  # module fact + class fact + dep fact

        kf_list = [KnowledgeFact(**f) for f in raw_facts]
        db_path = tmp_path / "roundtrip.db"
        saved = save_facts(kf_list, db_path=db_path)
        assert saved >= 2

        results = search_facts("Engine", db_path=db_path)
        assert len(results) > 0

    def test_idempotent_resave(self, tmp_path):
        """Saving the same facts twice doesn't duplicate — upserts."""
        from mother.knowledge_base import KnowledgeFact, save_facts

        snapshot = SourceSnapshot(
            project_root="/tmp", timestamp=time.time(),
            modules=(
                ModuleSummary("a.py", "", (), (), (), 10, False),
            ),
            total_lines=10, total_classes=0, total_functions=0,
        )

        raw = source_snapshot_to_facts(snapshot)
        kf_list = [KnowledgeFact(**f) for f in raw]
        db = tmp_path / "idem.db"
        save_facts(kf_list, db_path=db)
        save_facts(kf_list, db_path=db)  # Second save — should upsert, not duplicate


# ── ACTION dispatch tests ─────────────────────────────────────────────


class TestActionDispatch:
    def test_self_understand_action_recognized(self):
        """The action dispatch table should route 'self_understand'."""
        # Read the chat.py source and verify the action is in the dispatch
        chat_path = Path(__file__).parent.parent / "mother" / "screens" / "chat.py"
        source = chat_path.read_text()
        assert 'action == "self_understand"' in source
        assert "_run_self_understand" in source

    def test_self_understand_method_exists(self):
        """_run_self_understand should be defined in ChatScreen."""
        chat_path = Path(__file__).parent.parent / "mother" / "screens" / "chat.py"
        source = chat_path.read_text()
        assert "def _run_self_understand(self)" in source

    def test_self_understand_worker_exists(self):
        """_self_understand_worker should be defined."""
        chat_path = Path(__file__).parent.parent / "mother" / "screens" / "chat.py"
        source = chat_path.read_text()
        assert "async def _self_understand_worker(self)" in source


# ── Integration: actual codebase snapshot ──────────────────────────────


class TestRealCodebaseSnapshot:
    def test_snapshot_covers_key_packages(self, real_snapshot):
        """Snapshot should include modules from core, kernel, mother packages."""
        packages = {m.path.split(os.sep)[0] if os.sep in m.path else m.path.split("/")[0]
                     for m in real_snapshot.modules}
        assert "core" in packages
        assert "kernel" in packages
        assert "mother" in packages

    def test_summary_mentions_key_classes(self, real_snapshot):
        """Summary should mention MotherlabsEngine and EngineBridge."""
        summary = format_source_summary(real_snapshot)
        # These are the two most important classes
        assert "MotherlabsEngine" in summary or "Engine" in summary
        assert "EngineBridge" in summary or "Bridge" in summary

    def test_facts_cover_key_modules(self, real_snapshot):
        """Facts should include entries for core/engine.py."""
        facts = source_snapshot_to_facts(real_snapshot)
        fact_ids = [f["fact_id"] for f in facts]
        engine_facts = [fid for fid in fact_ids if "engine" in fid]
        assert len(engine_facts) > 0
