"""
Real-LLM Integration Tests for Motherlabs Semantic Compiler.

All tests require ANTHROPIC_API_KEY and are marked @pytest.mark.slow.
They NEVER run by default — only when explicitly invoked:

    ANTHROPIC_API_KEY=sk-ant-... pytest tests/test_real_llm.py -m slow -s -v --timeout=900

Design principles:
- Loose assertions (ranges, substrings) — LLM output is non-deterministic
- Shared module-scoped compilations to minimize cost (~$2-3 total)
- Cost tracking per test with per-test budget assertions
"""

import pytest
from typing import Dict, Any, List

pytestmark = [pytest.mark.slow, pytest.mark.timeout(900)]

# ---------------------------------------------------------------------------
# Test inputs — shared across tests to minimize redundant API calls
# ---------------------------------------------------------------------------

SIMPLE_INPUT = "A todo app with tasks, deadlines, and priority levels"

MEDIUM_INPUT = (
    "User authentication system with JWT tokens, OAuth2 login, "
    "session management, and role-based access control"
)

PROCESS_INPUT = "Employee onboarding process for a 50-person company"

COMPLEX_INPUT = (
    "An e-commerce platform with product catalog, shopping cart, checkout, "
    "payment processing via Stripe, order tracking, inventory management, "
    "customer reviews with ratings, and an admin dashboard for analytics"
)

MINIMAL_INPUT = "A simple notes app with folders"


# ---------------------------------------------------------------------------
# Module-scoped shared compilation results
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _simple_result(request):
    """
    Compile SIMPLE_INPUT once and share across TestStructuralInvariants
    and TestCodeEmission. Module-scoped to avoid redundant API calls.
    """
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    from core.llm import ClaudeClient
    from core.engine import MotherlabsEngine
    from persistence.corpus import Corpus
    import tempfile

    from pathlib import Path
    tmp = tempfile.mkdtemp()
    client = ClaudeClient(
        api_key=api_key,
        model="claude-sonnet-4-20250514",
        deterministic=True,
    )
    corpus = Corpus(Path(tmp) / "corpus.db")
    engine = MotherlabsEngine(
        llm_client=client,
        pipeline_mode="staged",
        corpus=corpus,
        auto_store=False,
    )
    result = engine.compile(SIMPLE_INPUT)
    return {"result": result, "engine": engine}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _component_names(result) -> list:
    """Extract component names from a CompileResult."""
    return [c["name"] for c in result.blueprint.get("components", [])]


def _has_anchor(names: list, anchor: str) -> bool:
    """Check if anchor appears as substring of any component name (case-insensitive)."""
    anchor_lower = anchor.lower()
    return any(anchor_lower in n.lower() for n in names)


# ===========================================================================
# TEST CLASSES
# ===========================================================================


class TestSimpleCompilation:
    """Compile a simple todo app and validate structure."""

    def test_compile_todo_app(self, real_engine, cost_tracker):
        result = real_engine.compile(SIMPLE_INPUT)
        report = cost_tracker(real_engine)

        # Tier 1: boolean success
        assert result.success is True, f"Compilation failed: {result.error}"

        # Tier 2: component count in range
        components = result.blueprint.get("components", [])
        assert 3 <= len(components) <= 15, (
            f"Expected 3-15 components, got {len(components)}"
        )

        # Tier 3: fuzzy anchor — "Task" should appear somewhere
        names = _component_names(result)
        assert _has_anchor(names, "Task"), (
            f"Expected 'Task' as anchor in component names: {names}"
        )

        # Tier 2: relationships
        rels = result.blueprint.get("relationships", [])
        assert 2 <= len(rels) <= 25, (
            f"Expected 2-25 relationships, got {len(rels)}"
        )

        # Structural: all components have required fields
        for comp in components:
            assert comp.get("name"), f"Component missing name: {comp}"
            assert comp.get("type"), f"Component missing type: {comp}"
            assert comp.get("derived_from"), f"Component missing derived_from: {comp}"

        # Cost guard
        assert report["total_cost_usd"] < 1.0, (
            f"Simple compilation cost ${report['total_cost_usd']:.2f} (budget: $1.00)"
        )


class TestMediumCompilation:
    """Compile a more complex auth system."""

    def test_compile_auth_system(self, real_engine, cost_tracker):
        result = real_engine.compile(MEDIUM_INPUT)
        report = cost_tracker(real_engine)

        assert result.success is True, f"Compilation failed: {result.error}"

        components = result.blueprint.get("components", [])
        assert 4 <= len(components) <= 20, (
            f"Expected 4-20 components, got {len(components)}"
        )

        names = _component_names(result)
        assert _has_anchor(names, "User") or _has_anchor(names, "Auth"), (
            f"Expected 'User' or 'Auth' as anchor in: {names}"
        )
        assert _has_anchor(names, "Session") or _has_anchor(names, "Token"), (
            f"Expected 'Session' or 'Token' as anchor in: {names}"
        )

        rels = result.blueprint.get("relationships", [])
        assert 3 <= len(rels) <= 30, (
            f"Expected 3-30 relationships, got {len(rels)}"
        )

        # At least one constraint
        constraints = result.blueprint.get("constraints", [])
        assert len(constraints) >= 1, "Expected at least 1 constraint"

        assert report["total_cost_usd"] < 1.0


@pytest.fixture(scope="module")
def _process_result(request):
    """
    Compile PROCESS_INPUT with PROCESS_ADAPTER once and share across TestProcessDomain.
    Module-scoped to avoid redundant API calls.
    """
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    from core.llm import ClaudeClient
    from core.engine import MotherlabsEngine
    from core.adapter_registry import get_adapter
    from persistence.corpus import Corpus
    import adapters  # noqa: F401
    import tempfile

    from pathlib import Path
    tmp = tempfile.mkdtemp()
    client = ClaudeClient(
        api_key=api_key,
        model="claude-sonnet-4-20250514",
        deterministic=True,
    )
    corpus = Corpus(Path(tmp) / "corpus.db")
    adapter = get_adapter("process")
    engine = MotherlabsEngine(
        llm_client=client,
        pipeline_mode="staged",
        corpus=corpus,
        auto_store=False,
        domain_adapter=adapter,
    )
    result = engine.compile(PROCESS_INPUT)
    return {"result": result, "engine": engine, "adapter": adapter}


class TestProcessDomain:
    """Compile with the PROCESS_ADAPTER to prove domain generalization."""

    def test_compile_onboarding(self, _process_result):
        result = _process_result["result"]

        assert result.success is True, f"Compilation failed: {result.error}"

        components = result.blueprint.get("components", [])
        assert 3 <= len(components) <= 20, (
            f"Expected 3-20 components, got {len(components)}"
        )

        rels = result.blueprint.get("relationships", [])
        assert 2 <= len(rels) <= 25, (
            f"Expected 2-25 relationships, got {len(rels)}"
        )

    def test_process_component_types(self, _process_result):
        """Components should include process-domain types (activity, gateway, event, etc.)."""
        result = _process_result["result"]
        assert result.success is True

        components = result.blueprint.get("components", [])
        types = {c.get("type", "").lower() for c in components}
        # At least one process-domain type should appear
        process_types = {"activity", "gateway", "event", "participant", "artifact", "subprocess", "process"}
        overlap = types & process_types
        assert overlap, (
            f"No process-domain types found. Types: {types}. "
            f"Expected at least one of: {process_types}"
        )

    def test_process_relationships_reference_components(self, _process_result):
        """All relationship endpoints should reference known components."""
        result = _process_result["result"]
        assert result.success is True

        names = set(_component_names(result))
        rels = result.blueprint.get("relationships", [])
        for rel in rels:
            from_name = rel.get("from", "")
            to_name = rel.get("to", "")
            assert from_name in names, (
                f"Relationship 'from' references unknown: '{from_name}'. Known: {names}"
            )
            assert to_name in names, (
                f"Relationship 'to' references unknown: '{to_name}'. Known: {names}"
            )

    def test_process_trust_indicators(self, _process_result):
        """Trust indicators should compute with non-zero scores for process domain."""
        result = _process_result["result"]
        assert result.success is True

        from core.trust import compute_trust_indicators

        intent_kw = list(result.context_graph.get("keywords", [])) if result.context_graph else []
        if not intent_kw:
            intent_kw = [w for w in PROCESS_INPUT.lower().split() if len(w) >= 3]

        trust = compute_trust_indicators(
            blueprint=result.blueprint,
            verification=result.verification or {},
            context_graph=result.context_graph or {},
            dimensional_metadata=result.dimensional_metadata or {},
            intent_keywords=intent_kw,
        )

        assert 0 <= trust.overall_score <= 100
        assert trust.verification_badge in ("verified", "partial", "unverified")
        assert trust.component_count >= 3
        assert trust.provenance_depth >= 1

    def test_process_emission_produces_yaml(self, _process_result, cost_tracker):
        """Emission for process domain should produce YAML, not Python."""
        result = _process_result["result"]
        engine = _process_result["engine"]
        assert result.success is True

        # Reset tokens for emission-only cost
        engine._compilation_tokens = []

        emission = engine.emit_code(result.blueprint)
        report = cost_tracker(engine)

        all_node_emissions = []
        for batch in emission.batch_emissions:
            all_node_emissions.extend(batch.emissions)

        assert len(all_node_emissions) >= 1, "No nodes emitted"

        # At least some should succeed
        passed = sum(1 for ne in all_node_emissions if ne.success)
        assert passed >= 1, (
            f"No successful emissions ({len(all_node_emissions)} attempted)"
        )

        # Successful emissions should contain YAML-like content (not Python class defs)
        for ne in all_node_emissions:
            if ne.success and ne.code.strip():
                # YAML typically has key: value lines, not "class X:" or "def f():"
                lines = ne.code.strip().split('\n')
                non_empty = [l for l in lines if l.strip()]
                if non_empty:
                    # Should NOT start with typical Python patterns
                    first = non_empty[0].strip()
                    assert not first.startswith("class "), (
                        f"Expected YAML, got Python class in emission for {ne.component_name}"
                    )

        assert report["total_cost_usd"] < 2.0


class TestStructuralInvariants:
    """
    Validate structural invariants on the shared SIMPLE_INPUT result.
    These tests cost $0 — they reuse the module-scoped compilation.
    """

    def test_relationships_reference_components(self, _simple_result):
        result = _simple_result["result"]
        assert result.success is True

        names = set(_component_names(result))
        rels = result.blueprint.get("relationships", [])

        for rel in rels:
            from_name = rel.get("from", "")
            to_name = rel.get("to", "")
            assert from_name in names, (
                f"Relationship 'from' references unknown component '{from_name}'. "
                f"Known: {names}"
            )
            assert to_name in names, (
                f"Relationship 'to' references unknown component '{to_name}'. "
                f"Known: {names}"
            )

    def test_no_duplicate_names(self, _simple_result):
        result = _simple_result["result"]
        assert result.success is True

        names = _component_names(result)
        assert len(names) == len(set(names)), (
            f"Duplicate component names: {[n for n in names if names.count(n) > 1]}"
        )

    def test_context_graph_populated(self, _simple_result):
        result = _simple_result["result"]
        assert result.success is True

        cg = result.context_graph
        has_insights = bool(cg.get("insights"))
        has_trace = bool(cg.get("decision_trace"))
        assert has_insights or has_trace, (
            "context_graph has neither insights nor decision_trace"
        )

    def test_dimensional_metadata_populated(self, _simple_result):
        result = _simple_result["result"]
        assert result.success is True

        dm = result.dimensional_metadata
        assert dm, "dimensional_metadata is empty"

    def test_interface_map_populated(self, _simple_result):
        result = _simple_result["result"]
        assert result.success is True

        im = result.interface_map
        assert im, "interface_map is empty"


class TestCodeEmission:
    """
    Test emit_code on the shared compilation result.
    Emission calls the LLM per-node, so this has its own cost.
    """

    def test_emit_produces_valid_python(self, _simple_result, cost_tracker):
        result = _simple_result["result"]
        engine = _simple_result["engine"]
        assert result.success is True

        # Reset token tracking for emission-only cost measurement
        engine._compilation_tokens = []

        emission = engine.emit_code(result.blueprint)
        report = cost_tracker(engine)

        # Flatten all NodeEmission objects from batch_emissions
        all_node_emissions = []
        for batch in emission.batch_emissions:
            all_node_emissions.extend(batch.emissions)

        # At least 1 node was emitted
        assert len(all_node_emissions) >= 1, (
            f"Expected at least 1 node emission, got {len(all_node_emissions)}"
        )

        # Pass rate >= 30%
        total = len(all_node_emissions)
        passed = sum(1 for ne in all_node_emissions if ne.success)
        pass_rate = passed / total if total > 0 else 0
        assert pass_rate >= 0.30, (
            f"Pass rate {pass_rate:.0%} below 30% threshold "
            f"({passed}/{total} nodes succeeded)"
        )

        # Syntax error rate <= 20%
        import ast
        syntax_errors = 0
        for ne in all_node_emissions:
            if ne.success and ne.code.strip():
                try:
                    ast.parse(ne.code)
                except SyntaxError:
                    syntax_errors += 1
        code_nodes = sum(1 for ne in all_node_emissions if ne.success and ne.code.strip())
        if code_nodes > 0:
            syntax_error_rate = syntax_errors / code_nodes
            assert syntax_error_rate <= 0.20, (
                f"Syntax error rate {syntax_error_rate:.0%} exceeds 20% threshold "
                f"({syntax_errors}/{code_nodes} files)"
            )

        # Cost guard
        assert report["total_cost_usd"] < 2.0, (
            f"Emission cost ${report['total_cost_usd']:.2f} (budget: $2.00)"
        )


class TestTrustIndicators:
    """
    Validate trust computation on real compilation output.
    Reuses the module-scoped _simple_result — $0 additional cost.
    """

    def test_trust_indicators_populated(self, _simple_result):
        result = _simple_result["result"]
        assert result.success is True

        from core.trust import compute_trust_indicators

        blueprint = result.blueprint
        verification = result.verification if hasattr(result, "verification") else {}
        context_graph = result.context_graph if hasattr(result, "context_graph") else {}
        dimensional_metadata = result.dimensional_metadata if hasattr(result, "dimensional_metadata") else {}

        # Extract intent keywords from context_graph
        intent_keywords = context_graph.get("keywords", [])
        if not intent_keywords:
            # Fallback: derive from SIMPLE_INPUT
            intent_keywords = [w for w in SIMPLE_INPUT.lower().split() if len(w) >= 3]

        trust = compute_trust_indicators(
            blueprint=blueprint,
            verification=verification,
            context_graph=context_graph,
            dimensional_metadata=dimensional_metadata,
            intent_keywords=intent_keywords,
        )

        # Overall score is populated (0-100 range)
        assert 0 <= trust.overall_score <= 100, (
            f"overall_score {trust.overall_score} outside 0-100"
        )

        # Badge is one of the valid values
        assert trust.verification_badge in ("verified", "partial", "unverified"), (
            f"Unexpected badge: {trust.verification_badge}"
        )

        # Fidelity scores has the 7 expected keys
        expected_dims = {
            "completeness", "consistency", "coherence", "traceability",
            "actionability", "specificity", "codegen_readiness",
        }
        assert expected_dims.issubset(set(trust.fidelity_scores.keys())), (
            f"Missing fidelity dimensions: {expected_dims - set(trust.fidelity_scores.keys())}"
        )

        # Provenance depth >= 1 (we compiled with a real LLM, so insights should exist)
        assert trust.provenance_depth >= 1, (
            f"Expected provenance_depth >= 1, got {trust.provenance_depth}"
        )

        # Gap report and silence zones are tuples
        assert isinstance(trust.gap_report, tuple), "gap_report should be a tuple"
        assert isinstance(trust.silence_zones, tuple), "silence_zones should be a tuple"

        # Component count matches blueprint
        bp_components = len(blueprint.get("components", []))
        assert trust.component_count == bp_components, (
            f"component_count {trust.component_count} != blueprint components {bp_components}"
        )


class TestAgentOrchestratorE2E:
    """Full pipeline: input -> compile -> emit -> project on disk."""

    def test_orchestrator_produces_project(self, real_orchestrator, cost_tracker):
        orchestrator = real_orchestrator
        engine = orchestrator.engine

        result = orchestrator.run(SIMPLE_INPUT)
        report = cost_tracker(engine)

        # Compilation succeeded
        assert result.success is True, f"Orchestrator failed: {result.error}"

        # Blueprint was produced
        assert result.blueprint, "No blueprint produced"
        components = result.blueprint.get("components", [])
        assert len(components) >= 2, (
            f"Expected >= 2 components, got {len(components)}"
        )

        # Code was generated
        assert result.generated_code, "No code generated"
        assert len(result.generated_code) >= 1, (
            f"Expected >= 1 generated files, got {len(result.generated_code)}"
        )

        # Compile result is accessible
        assert result.compile_result is not None
        assert result.compile_result.success is True

        # Project was written to disk (manifest exists)
        assert result.project_manifest is not None, "No project manifest"
        assert result.project_manifest.files_written, "No files written"

        # Fuzzy anchor
        names = _component_names(result.compile_result)
        assert _has_anchor(names, "Task"), (
            f"Expected 'Task' anchor in orchestrator output: {names}"
        )

        # Cost guard — full pipeline (enrich + compile + emit + write) under $2
        assert report["total_cost_usd"] < 2.0, (
            f"Orchestrator cost ${report['total_cost_usd']:.2f} (budget: $2.00)"
        )


class TestStability:
    """
    Run the same input twice and verify both succeed with similar shape.
    Tests LLM non-determinism resilience of the pipeline.
    """

    def test_two_runs_both_succeed(self, real_engine, cost_tracker):
        result1 = real_engine.compile(SIMPLE_INPUT)
        cost1 = cost_tracker(real_engine)

        # Reset tokens for second run
        real_engine._compilation_tokens = []
        result2 = real_engine.compile(SIMPLE_INPUT)
        cost2 = cost_tracker(real_engine)

        # Both must succeed
        assert result1.success is True, f"Run 1 failed: {result1.error}"
        assert result2.success is True, f"Run 2 failed: {result2.error}"

        # Component counts within 3x of each other
        count1 = len(result1.blueprint.get("components", []))
        count2 = len(result2.blueprint.get("components", []))
        assert count1 > 0 and count2 > 0, "Both runs must produce components"

        ratio = max(count1, count2) / min(count1, count2)
        assert ratio <= 3.0, (
            f"Component counts too divergent: {count1} vs {count2} (ratio {ratio:.1f}x)"
        )

        # Total cost for both runs
        total = cost1["total_cost_usd"] + cost2["total_cost_usd"]
        assert total < 2.0, f"Stability test cost ${total:.2f} (budget: $2.00)"


class TestComplexCompilation:
    """Compile a complex e-commerce system to test scale behavior."""

    def test_compile_ecommerce(self, real_engine, cost_tracker):
        result = real_engine.compile(COMPLEX_INPUT)
        report = cost_tracker(real_engine)

        assert result.success is True, f"Compilation failed: {result.error}"

        components = result.blueprint.get("components", [])
        assert 5 <= len(components) <= 25, (
            f"Expected 5-25 components, got {len(components)}"
        )

        names = _component_names(result)
        # At least one commerce-related anchor
        commerce_anchors = ["Product", "Cart", "Order", "Payment", "Checkout"]
        found = any(_has_anchor(names, a) for a in commerce_anchors)
        assert found, (
            f"No commerce anchor found in: {names}. "
            f"Expected at least one of: {commerce_anchors}"
        )

        rels = result.blueprint.get("relationships", [])
        assert 4 <= len(rels) <= 40, (
            f"Expected 4-40 relationships, got {len(rels)}"
        )

        constraints = result.blueprint.get("constraints", [])
        assert len(constraints) >= 1, "Expected at least 1 constraint for complex input"

        # Cost guard
        assert report["total_cost_usd"] < 1.5, (
            f"Complex compilation cost ${report['total_cost_usd']:.2f} (budget: $1.50)"
        )


class TestMinimalCompilation:
    """Compile a minimal input to test lower-bound behavior."""

    def test_compile_notes_app(self, real_engine, cost_tracker):
        result = real_engine.compile(MINIMAL_INPUT)
        report = cost_tracker(real_engine)

        assert result.success is True, f"Compilation failed: {result.error}"

        components = result.blueprint.get("components", [])
        assert 2 <= len(components) <= 10, (
            f"Expected 2-10 components, got {len(components)}"
        )

        names = _component_names(result)
        assert _has_anchor(names, "Note"), (
            f"Expected 'Note' as anchor in: {names}"
        )

        rels = result.blueprint.get("relationships", [])
        assert 1 <= len(rels) <= 15, (
            f"Expected 1-15 relationships, got {len(rels)}"
        )

        assert report["total_cost_usd"] < 1.0


class TestVerificationQuality:
    """Verify that verification data is populated for real compilations."""

    def test_verification_populated(self, _simple_result):
        result = _simple_result["result"]
        assert result.success is True

        verification = result.verification
        assert verification, "Verification is empty"

        # Should have at least some scoring data
        if isinstance(verification, dict):
            # Could be deterministic verification dict
            assert len(verification) > 0, "Verification dict is empty"

    def test_compilation_metadata_complete(self, _simple_result):
        """All CompileResult metadata fields are populated."""
        result = _simple_result["result"]
        assert result.success is True

        # Blueprint has core_need
        assert result.blueprint.get("core_need"), "Blueprint missing core_need"
        assert result.blueprint.get("domain"), "Blueprint missing domain"

        # Components have required fields
        for comp in result.blueprint.get("components", []):
            assert comp.get("name"), f"Component missing name: {comp}"
            assert comp.get("type") in ("entity", "process", "external", "data_store",
                                         "activity", "gateway", "event", "participant",
                                         "artifact", "subprocess"), (
                f"Component type unexpected: {comp.get('type')}"
            )

    def test_trust_nonzero_for_real_compilation(self, _simple_result):
        """Real compilations should produce non-zero trust scores."""
        result = _simple_result["result"]
        assert result.success is True

        from core.trust import compute_trust_indicators

        intent_kw = list(result.context_graph.get("keywords", [])) if result.context_graph else []
        if not intent_kw:
            intent_kw = [w for w in SIMPLE_INPUT.lower().split() if len(w) >= 3]

        trust = compute_trust_indicators(
            blueprint=result.blueprint,
            verification=result.verification or {},
            context_graph=result.context_graph or {},
            dimensional_metadata=result.dimensional_metadata or {},
            intent_keywords=intent_kw,
        )

        # At least one fidelity score should be > 0
        nonzero_scores = [v for v in trust.fidelity_scores.values() if v > 0]
        assert len(nonzero_scores) >= 3, (
            f"Expected at least 3 non-zero fidelity scores, got {len(nonzero_scores)}: "
            f"{dict(trust.fidelity_scores)}"
        )

        assert trust.overall_score > 0, f"overall_score is 0"
        assert trust.component_count > 0, f"component_count is 0"
