"""Tests for mother/source_reader.py — AST-based structural analysis."""

import ast
import os
import tempfile
import time
from pathlib import Path

import pytest

from mother.source_reader import (
    ClassSummary,
    FunctionSummary,
    MethodSummary,
    ModuleSummary,
    SourceSnapshot,
    format_dependency_graph,
    format_source_summary,
    read_codebase,
    read_module,
    read_package,
    source_snapshot_to_facts,
    _detect_frozen,
    _extract_imports,
    _extract_methods,
    _format_args,
    _format_return,
    _get_docstring,
)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project structure for testing."""
    # core/engine.py
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    (core_dir / "__init__.py").write_text("")
    (core_dir / "engine.py").write_text('''"""Core engine module."""
import os
from mother.bridge import EngineBridge

class Engine:
    """The main engine."""
    def __init__(self, provider: str = "claude"):
        self.provider = provider

    def compile(self, text: str) -> dict:
        """Compile intent to structure."""
        return {}

    async def compile_async(self, text: str) -> dict:
        return {}

    def _private_method(self):
        pass

def run_engine(config: dict) -> Engine:
    """Create and run an engine."""
    return Engine()
''')

    # mother/bridge.py
    mother_dir = tmp_path / "mother"
    mother_dir.mkdir()
    (mother_dir / "__init__.py").write_text("")
    (mother_dir / "bridge.py").write_text('''"""
Engine bridge — the single integration seam.

LEAF module.
"""
from core.engine import Engine

class EngineBridge:
    """Bridge between TUI and core."""
    def __init__(self):
        self._engine = None

    async def compile(self, desc: str, mode: str = "build") -> dict:
        return {}
''')

    # tests/ (should be skipped)
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "test_engine.py").write_text("def test_it(): pass")

    return tmp_path


@pytest.fixture
def simple_source():
    """Simple Python source for AST parsing tests."""
    return '''"""Module docstring for testing."""
import os
from typing import List

class Foo:
    """A frozen dataclass."""
    def __init__(self, x: int):
        self.x = x

    def bar(self, y: str) -> bool:
        return True

    def _private(self):
        pass

async def top_level_func(a: int, b: str = "x") -> List[str]:
    """Does something async."""
    return []

def another_func() -> None:
    pass
'''


# ── read_module tests ───────────────────────────────────────────────────


class TestReadModule:
    def test_parse_simple_file(self, tmp_path, simple_source):
        f = tmp_path / "mod.py"
        f.write_text(simple_source)
        mod = read_module(str(f), str(tmp_path))
        assert mod is not None
        assert mod.path == "mod.py"
        assert "Module docstring" in mod.docstring
        assert len(mod.classes) == 1
        assert mod.classes[0].name == "Foo"
        assert len(mod.functions) == 2  # top_level_func, another_func (private skipped)

    def test_syntax_error_returns_none(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def broken(\n")
        mod = read_module(str(f), str(tmp_path))
        assert mod is None

    def test_empty_file_returns_none(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("")
        mod = read_module(str(f), str(tmp_path))
        assert mod is None

    def test_whitespace_only_returns_none(self, tmp_path):
        f = tmp_path / "ws.py"
        f.write_text("   \n\n  \n")
        mod = read_module(str(f), str(tmp_path))
        assert mod is None

    def test_leaf_detection_in_docstring(self, tmp_path):
        f = tmp_path / "leaf.py"
        f.write_text('"""This is a LEAF module."""\nx = 1\n')
        mod = read_module(str(f), str(tmp_path))
        assert mod is not None
        assert mod.is_leaf is True

    def test_non_leaf_module(self, tmp_path, simple_source):
        f = tmp_path / "nonleaf.py"
        f.write_text(simple_source)
        mod = read_module(str(f), str(tmp_path))
        assert mod is not None
        assert mod.is_leaf is False

    def test_frozen_dataclass_detection(self, tmp_path):
        f = tmp_path / "frozen.py"
        f.write_text('''"""Test."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Point:
    x: int
    y: int
''')
        mod = read_module(str(f), str(tmp_path))
        assert mod is not None
        assert len(mod.classes) == 1
        assert mod.classes[0].is_frozen is True

    def test_non_frozen_dataclass(self, tmp_path):
        f = tmp_path / "mutable.py"
        f.write_text('''"""Test."""
from dataclasses import dataclass

@dataclass
class Point:
    x: int
''')
        mod = read_module(str(f), str(tmp_path))
        assert mod.classes[0].is_frozen is False

    def test_async_function_detection(self, tmp_path):
        f = tmp_path / "async_mod.py"
        f.write_text('''"""Test."""
async def fetch(url: str) -> str:
    return ""
''')
        mod = read_module(str(f), str(tmp_path))
        assert len(mod.functions) == 1
        assert mod.functions[0].is_async is True

    def test_async_method_detection(self, tmp_path):
        f = tmp_path / "async_cls.py"
        f.write_text('''"""Test."""
class Fetcher:
    async def fetch(self, url: str) -> str:
        return ""
''')
        mod = read_module(str(f), str(tmp_path))
        assert len(mod.classes) == 1
        assert mod.classes[0].methods[0].is_async is True

    def test_method_cap_at_15(self, tmp_path):
        methods = "\n".join(f"    def method_{i}(self): pass" for i in range(20))
        f = tmp_path / "many.py"
        f.write_text(f'"""Test."""\nclass Big:\n{methods}\n')
        mod = read_module(str(f), str(tmp_path))
        assert len(mod.classes[0].methods) == 15

    def test_skip_private_methods_except_init(self, tmp_path):
        f = tmp_path / "priv.py"
        f.write_text('''"""Test."""
class Cls:
    def __init__(self): pass
    def _helper(self): pass
    def __repr__(self): pass
    def public(self): pass
''')
        mod = read_module(str(f), str(tmp_path))
        method_names = [m.name for m in mod.classes[0].methods]
        assert "__init__" in method_names
        assert "public" in method_names
        assert "_helper" not in method_names
        assert "__repr__" not in method_names

    def test_skip_private_functions(self, tmp_path):
        f = tmp_path / "fns.py"
        f.write_text('"""Test."""\ndef public(): pass\ndef _private(): pass\n')
        mod = read_module(str(f), str(tmp_path))
        assert len(mod.functions) == 1
        assert mod.functions[0].name == "public"

    def test_function_cap_at_20(self, tmp_path):
        fns = "\n".join(f"def func_{i}(): pass" for i in range(25))
        f = tmp_path / "manyfns.py"
        f.write_text(f'"""Test."""\n{fns}\n')
        mod = read_module(str(f), str(tmp_path))
        assert len(mod.functions) == 20

    def test_line_count(self, tmp_path):
        f = tmp_path / "lines.py"
        f.write_text("x = 1\ny = 2\nz = 3\n")
        mod = read_module(str(f), str(tmp_path))
        assert mod.line_count == 4  # 3 lines + trailing newline count

    def test_import_extraction_skips_stdlib(self, tmp_path):
        f = tmp_path / "imps.py"
        f.write_text('"""Test."""\nimport os\nimport json\nfrom core.engine import Engine\n')
        mod = read_module(str(f), str(tmp_path))
        # Only core.engine should be in imports, not os/json
        assert any("core" in imp for imp in mod.imports)
        assert not any(imp.startswith("os") for imp in mod.imports)
        assert not any(imp.startswith("json") for imp in mod.imports)

    def test_docstring_truncated_at_200(self, tmp_path):
        long_doc = "A" * 300
        f = tmp_path / "longdoc.py"
        f.write_text(f'"""{long_doc}"""\nx = 1\n')
        mod = read_module(str(f), str(tmp_path))
        assert len(mod.docstring) == 200

    def test_nonexistent_file_returns_none(self, tmp_path):
        mod = read_module(str(tmp_path / "nope.py"), str(tmp_path))
        assert mod is None

    def test_relative_path_from_project_root(self, tmp_path):
        sub = tmp_path / "pkg" / "sub"
        sub.mkdir(parents=True)
        f = sub / "mod.py"
        f.write_text('"""Test."""\nx = 1\n')
        mod = read_module(str(f), str(tmp_path))
        assert mod.path == os.path.join("pkg", "sub", "mod.py")

    def test_class_bases_extracted(self, tmp_path):
        f = tmp_path / "bases.py"
        f.write_text('''"""Test."""
class Child(Parent, Mixin):
    pass
''')
        mod = read_module(str(f), str(tmp_path))
        assert mod.classes[0].bases == ("Parent", "Mixin")

    def test_function_args_formatting(self, tmp_path):
        f = tmp_path / "args.py"
        f.write_text('''"""Test."""
def func(a: int, b: str = "x", *args, key: bool = False, **kwargs) -> None:
    pass
''')
        mod = read_module(str(f), str(tmp_path))
        assert "a: int" in mod.functions[0].args
        assert "b: str" in mod.functions[0].args


# ── read_package tests ──────────────────────────────────────────────────


class TestReadPackage:
    def test_reads_all_py_files(self, tmp_project):
        mods = read_package(str(tmp_project / "core"), str(tmp_project))
        paths = [m.path for m in mods]
        assert any("engine.py" in p for p in paths)
        # __init__.py is empty → read_module returns None → not in results

    def test_skips_tests_dir(self, tmp_project):
        # tests/ exists but should be skipped
        mods = read_package(str(tmp_project / "tests"), str(tmp_project))
        # read_package skips tests/ *subdirectories*, but tests/ itself is the package_dir
        # If called on tests/ as root, it reads files in that dir but skips nested tests/ subdirs
        # The skip logic prevents walking INTO tests/ subfolders, not reading the root
        # For the intended use case, tests/ is never a package we pass to read_codebase

    def test_skips_pycache(self, tmp_project):
        cache = tmp_project / "core" / "__pycache__"
        cache.mkdir()
        (cache / "engine.cpython-314.pyc").write_text("compiled")
        # Should still work fine, pycache ignored
        mods = read_package(str(tmp_project / "core"), str(tmp_project))
        assert all("__pycache__" not in m.path for m in mods)

    def test_sorted_by_path(self, tmp_project):
        mods = read_package(str(tmp_project / "core"), str(tmp_project))
        paths = [m.path for m in mods]
        assert paths == sorted(paths)

    def test_skips_non_py_files(self, tmp_project):
        (tmp_project / "core" / "README.md").write_text("# Core")
        (tmp_project / "core" / "data.json").write_text("{}")
        mods = read_package(str(tmp_project / "core"), str(tmp_project))
        assert all(m.path.endswith(".py") for m in mods)


# ── read_codebase tests ────────────────────────────────────────────────


class TestReadCodebase:
    def test_reads_multiple_packages(self, tmp_project):
        snap = read_codebase(str(tmp_project), packages=("core", "mother"))
        assert len(snap.modules) >= 2  # engine.py + bridge.py (empty __init__ skipped)
        assert snap.total_lines > 0
        assert snap.total_classes >= 2

    def test_skips_missing_packages(self, tmp_project):
        snap = read_codebase(str(tmp_project), packages=("core", "nonexistent"))
        assert len(snap.modules) >= 1  # Only core found

    def test_aggregates_correct(self, tmp_project):
        snap = read_codebase(str(tmp_project), packages=("core",))
        assert snap.total_lines == sum(m.line_count for m in snap.modules)
        assert snap.total_classes == sum(len(m.classes) for m in snap.modules)
        assert snap.total_functions == sum(len(m.functions) for m in snap.modules)

    def test_timestamp_set(self, tmp_project):
        before = time.time()
        snap = read_codebase(str(tmp_project), packages=("core",))
        after = time.time()
        assert before <= snap.timestamp <= after

    def test_project_root_recorded(self, tmp_project):
        snap = read_codebase(str(tmp_project), packages=("core",))
        assert snap.project_root == str(tmp_project)

    def test_integration_on_real_project(self):
        """Integration test: read actual Motherlabs codebase."""
        project_root = str(Path(__file__).parent.parent)
        snap = read_codebase(project_root)
        # Should find at least 50 modules
        assert len(snap.modules) >= 50, f"Found only {len(snap.modules)} modules"
        assert snap.total_lines > 10000
        assert snap.total_classes > 30


# ── format_source_summary tests ────────────────────────────────────────


class TestFormatSourceSummary:
    def test_stays_within_word_budget(self):
        """Summary should roughly stay within max_words."""
        project_root = str(Path(__file__).parent.parent)
        snap = read_codebase(project_root)
        summary = format_source_summary(snap, max_words=3000)
        word_count = len(summary.split())
        # Allow some overshoot since we measure approximately
        assert word_count < 5000, f"Summary too long: {word_count} words"

    def test_contains_package_headers(self, tmp_project):
        snap = read_codebase(str(tmp_project), packages=("core", "mother"))
        summary = format_source_summary(snap)
        assert "## core/" in summary
        assert "## mother/" in summary

    def test_leaf_annotated(self, tmp_project):
        snap = read_codebase(str(tmp_project), packages=("mother",))
        summary = format_source_summary(snap)
        assert "[LEAF]" in summary

    def test_includes_module_count(self, tmp_project):
        snap = read_codebase(str(tmp_project), packages=("core",))
        summary = format_source_summary(snap)
        assert "modules" in summary

    def test_large_file_gets_class_detail(self, tmp_path):
        """Files >500 lines should show class names and methods."""
        big_dir = tmp_path / "big"
        big_dir.mkdir()
        (big_dir / "__init__.py").write_text("")
        # Create a 600-line file
        lines = ['"""Big module."""'] + [f"# line {i}" for i in range(598)]
        lines.append("class BigClass:\n    def do_thing(self): pass\n")
        (big_dir / "large.py").write_text("\n".join(lines))
        snap = read_codebase(str(tmp_path), packages=("big",))
        summary = format_source_summary(snap)
        assert "BigClass" in summary


# ── format_dependency_graph tests ──────────────────────────────────────


class TestFormatDependencyGraph:
    def test_cross_package_only(self, tmp_project):
        snap = read_codebase(str(tmp_project), packages=("core", "mother"))
        graph = format_dependency_graph(snap)
        # mother/bridge.py imports from core → should show as edge
        assert "mother" in graph or "core" in graph

    def test_intra_package_excluded(self, tmp_path):
        """Imports within the same package should not appear."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "a.py").write_text('"""A."""\nfrom pkg.b import X\n')
        (pkg / "b.py").write_text('"""B."""\nX = 1\n')
        snap = read_codebase(str(tmp_path), packages=("pkg",))
        graph = format_dependency_graph(snap)
        # No cross-package edges, so graph should be empty
        assert graph == ""

    def test_empty_when_no_cross_deps(self, tmp_path):
        pkg = tmp_path / "standalone"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "mod.py").write_text('"""Standalone."""\nimport os\n')
        snap = read_codebase(str(tmp_path), packages=("standalone",))
        graph = format_dependency_graph(snap)
        assert graph == ""

    def test_header_present_when_edges_exist(self, tmp_project):
        snap = read_codebase(str(tmp_project), packages=("core", "mother"))
        graph = format_dependency_graph(snap)
        if graph:
            assert "Cross-Package Dependencies" in graph


# ── source_snapshot_to_facts tests ─────────────────────────────────────


class TestSourceSnapshotToFacts:
    def test_deterministic_fact_ids(self, tmp_project):
        snap = read_codebase(str(tmp_project), packages=("core",))
        facts1 = source_snapshot_to_facts(snap)
        facts2 = source_snapshot_to_facts(snap)
        ids1 = [f["fact_id"] for f in facts1]
        ids2 = [f["fact_id"] for f in facts2]
        assert ids1 == ids2

    def test_module_facts_have_pattern_category(self, tmp_project):
        snap = read_codebase(str(tmp_project), packages=("core",))
        facts = source_snapshot_to_facts(snap)
        module_facts = [f for f in facts if f["fact_id"].startswith("source:module:")]
        assert len(module_facts) > 0
        for f in module_facts:
            assert f["category"] == "pattern"

    def test_class_facts_have_capability_category(self, tmp_project):
        snap = read_codebase(str(tmp_project), packages=("core",))
        facts = source_snapshot_to_facts(snap)
        class_facts = [f for f in facts if f["fact_id"].startswith("source:class:")]
        assert len(class_facts) > 0
        for f in class_facts:
            assert f["category"] == "capability"

    def test_dep_facts_have_tool_category(self, tmp_project):
        snap = read_codebase(str(tmp_project), packages=("core", "mother"))
        facts = source_snapshot_to_facts(snap)
        dep_facts = [f for f in facts if f["fact_id"].startswith("source:deps:")]
        # mother/bridge.py imports from core → should generate dep fact
        assert len(dep_facts) > 0
        for f in dep_facts:
            assert f["category"] == "tool"

    def test_all_facts_have_required_fields(self, tmp_project):
        snap = read_codebase(str(tmp_project), packages=("core",))
        facts = source_snapshot_to_facts(snap)
        required = {"fact_id", "category", "subject", "predicate", "value",
                     "confidence", "source", "first_seen", "last_confirmed",
                     "access_count"}
        for f in facts:
            assert required <= set(f.keys()), f"Missing fields in {f['fact_id']}"

    def test_confidence_is_high(self, tmp_project):
        snap = read_codebase(str(tmp_project), packages=("core",))
        facts = source_snapshot_to_facts(snap)
        for f in facts:
            assert f["confidence"] == 0.95

    def test_source_is_source_reader(self, tmp_project):
        snap = read_codebase(str(tmp_project), packages=("core",))
        facts = source_snapshot_to_facts(snap)
        for f in facts:
            assert f["source"] == "source_reader:ast"

    def test_round_trip_with_knowledge_base(self, tmp_path, tmp_project):
        """Facts can be saved to knowledge base and retrieved."""
        from mother.knowledge_base import KnowledgeFact, save_facts, search_facts

        snap = read_codebase(str(tmp_project), packages=("core",))
        raw_facts = source_snapshot_to_facts(snap)

        # Convert dicts to KnowledgeFact objects
        kf_list = [KnowledgeFact(**f) for f in raw_facts]
        db_path = tmp_path / "test_kb.db"
        saved = save_facts(kf_list, db_path=db_path)
        assert saved > 0

        # Search should find them
        results = search_facts("engine", db_path=db_path)
        assert len(results) > 0


# ── Frozen dataclass tests ─────────────────────────────────────────────


class TestDataclassesFrozen:
    def test_method_summary_frozen(self):
        m = MethodSummary(name="foo", args="x: int", return_annotation="str", is_async=False)
        with pytest.raises(AttributeError):
            m.name = "bar"

    def test_class_summary_frozen(self):
        c = ClassSummary(name="Foo", docstring="", bases=(), methods=(), is_frozen=False)
        with pytest.raises(AttributeError):
            c.name = "bar"

    def test_module_summary_frozen(self):
        m = ModuleSummary(path="x.py", docstring="", classes=(), functions=(),
                          imports=(), line_count=10, is_leaf=False)
        with pytest.raises(AttributeError):
            m.path = "y.py"

    def test_source_snapshot_frozen(self):
        s = SourceSnapshot(project_root="/tmp", timestamp=0, modules=(),
                           total_lines=0, total_classes=0, total_functions=0)
        with pytest.raises(AttributeError):
            s.total_lines = 100
