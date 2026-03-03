"""Tests for tiered compilation routing — cheap model for dialogue, quality model for synthesis."""

import pytest
from unittest.mock import MagicMock, patch

from core.llm import MockClient, FailoverClient, RouteTier


# ── Engine: quality_llm parameter ──────────────────────────────────

class TestEngineQualityLLM:
    def test_default_uses_same_llm_for_all(self):
        """Without quality_llm, all agents share the same LLM."""
        from core.engine import MotherlabsEngine
        mock = MockClient()
        engine = MotherlabsEngine(llm_client=mock)

        assert engine.intent_agent.llm is mock
        assert engine.persona_agent.llm is mock
        assert engine.entity_agent.llm is mock
        assert engine.process_agent.llm is mock
        assert engine.synthesis_agent.llm is mock
        assert engine.verify_agent.llm is mock

    def test_quality_llm_routes_to_synthesis_and_verify(self):
        """With quality_llm, synthesis and verify use it; others use default."""
        from core.engine import MotherlabsEngine
        cheap = MockClient()
        quality = MockClient()
        engine = MotherlabsEngine(llm_client=cheap, quality_llm=quality)

        # Dialogue agents use cheap LLM
        assert engine.intent_agent.llm is cheap
        assert engine.persona_agent.llm is cheap
        assert engine.entity_agent.llm is cheap
        assert engine.process_agent.llm is cheap

        # Synthesis and verify use quality LLM
        assert engine.synthesis_agent.llm is quality
        assert engine.verify_agent.llm is quality

        # Governor uses cheap (structural, not LLM-intensive)
        assert engine.governor.llm is cheap

    def test_quality_llm_none_falls_back_to_default(self):
        """quality_llm=None should use the default LLM for everything."""
        from core.engine import MotherlabsEngine
        mock = MockClient()
        engine = MotherlabsEngine(llm_client=mock, quality_llm=None)

        assert engine.synthesis_agent.llm is mock
        assert engine.verify_agent.llm is mock

    def test_quality_llm_stored_on_engine(self):
        """Engine stores quality_llm for inspection."""
        from core.engine import MotherlabsEngine
        cheap = MockClient()
        quality = MockClient()
        engine = MotherlabsEngine(llm_client=cheap, quality_llm=quality)

        assert engine.quality_llm is quality

    def test_quality_llm_same_as_default_when_not_set(self):
        """quality_llm property equals self.llm when not explicitly set."""
        from core.engine import MotherlabsEngine
        mock = MockClient()
        engine = MotherlabsEngine(llm_client=mock)

        assert engine.quality_llm is mock


# ── FailoverClient: tier passthrough fix ────────────────────────────

class TestFailoverTierPassthrough:
    def test_complete_with_system_passes_tier(self):
        """complete_with_system should pass tier to complete."""
        mock1 = MockClient()
        mock2 = MockClient()
        client = FailoverClient(
            [mock1, mock2],
            tier_map={RouteTier.CRITICAL: [1], RouteTier.STANDARD: [0]},
        )

        # Monkey-patch complete to track tier
        original_complete = client.complete
        captured_tiers = []

        def tracking_complete(*args, **kwargs):
            captured_tiers.append(kwargs.get("tier"))
            return original_complete(*args, **kwargs)

        client.complete = tracking_complete

        client.complete_with_system(
            "system prompt", "user content", tier=RouteTier.CRITICAL
        )
        assert captured_tiers == [RouteTier.CRITICAL]

    def test_complete_with_system_none_tier(self):
        """complete_with_system with no tier should pass None."""
        mock = MockClient()
        client = FailoverClient([mock])

        original_complete = client.complete
        captured_tiers = []

        def tracking_complete(*args, **kwargs):
            captured_tiers.append(kwargs.get("tier"))
            return original_complete(*args, **kwargs)

        client.complete = tracking_complete

        client.complete_with_system("sys", "user")
        assert captured_tiers == [None]

    def test_tier_map_routes_to_preferred_provider(self):
        """Tier map should route CRITICAL to provider at index 1."""
        cheap = MockClient()
        quality = MockClient()

        # Make cheap fail so we can verify routing
        cheap_calls = []
        quality_calls = []

        original_cheap_complete = cheap.complete
        original_quality_complete = quality.complete

        def cheap_tracking(*args, **kwargs):
            cheap_calls.append(1)
            return original_cheap_complete(*args, **kwargs)

        def quality_tracking(*args, **kwargs):
            quality_calls.append(1)
            return original_quality_complete(*args, **kwargs)

        cheap.complete = cheap_tracking
        quality.complete = quality_tracking

        client = FailoverClient(
            [cheap, quality],
            tier_map={RouteTier.CRITICAL: [1]},
        )

        # CRITICAL tier should try quality (index 1) first
        client.complete(
            [{"role": "user", "content": "test"}],
            tier=RouteTier.CRITICAL,
        )
        assert len(quality_calls) == 1
        assert len(cheap_calls) == 0

    def test_no_tier_uses_default_order(self):
        """Without tier, providers are tried in original order."""
        first = MockClient()
        second = MockClient()

        first_calls = []
        original_first = first.complete

        def first_tracking(*args, **kwargs):
            first_calls.append(1)
            return original_first(*args, **kwargs)

        first.complete = first_tracking

        client = FailoverClient(
            [first, second],
            tier_map={RouteTier.CRITICAL: [1]},
        )

        # No tier — should use index 0 (first) first
        client.complete([{"role": "user", "content": "test"}])
        assert len(first_calls) == 1


# ── Bridge: tiered engine creation ─────────────────────────────────

class TestBridgeTieredEngine:
    def test_no_chat_model_uses_single_llm(self):
        """Without chat_model, engine gets no quality_llm."""
        from mother.bridge import EngineBridge

        bridge = EngineBridge(provider="claude", model="claude-sonnet-4-20250514")
        # Don't set chat_model — should use single model

        with patch("core.engine.MotherlabsEngine") as MockEngine:
            MockEngine.return_value = MagicMock()
            bridge._get_engine()

            call_kwargs = MockEngine.call_args[1]
            assert call_kwargs.get("quality_llm") is None

    def test_same_model_no_split(self):
        """When chat_model == model, no quality split."""
        from mother.bridge import EngineBridge

        bridge = EngineBridge(provider="claude", model="claude-sonnet-4-20250514")
        bridge.set_chat_model("claude-sonnet-4-20250514")  # same as main

        with patch("core.engine.MotherlabsEngine") as MockEngine:
            MockEngine.return_value = MagicMock()
            bridge._get_engine()

            call_kwargs = MockEngine.call_args[1]
            assert call_kwargs.get("quality_llm") is None

    def test_different_chat_model_creates_quality_llm(self):
        """When chat_model != model, engine gets cheap + quality split."""
        from mother.bridge import EngineBridge

        bridge = EngineBridge(provider="claude", model="claude-sonnet-4-20250514")
        bridge.set_chat_model("claude-haiku-4-5-20251001")

        with patch("core.engine.MotherlabsEngine") as MockEngine, \
             patch("core.llm.create_client") as mock_create:
            MockEngine.return_value = MagicMock()
            mock_quality = MagicMock()
            mock_create.return_value = mock_quality

            bridge._get_engine()

            # Engine should be created with cheap model
            call_kwargs = MockEngine.call_args[1]
            assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
            # And quality_llm should be the main model
            assert call_kwargs["quality_llm"] is mock_quality

            # create_client should have been called for quality LLM
            mock_create.assert_called_once_with(
                provider="claude",
                model="claude-sonnet-4-20250514",
                api_key=None,
            )

    def test_quality_llm_creation_failure_falls_back(self):
        """If quality LLM creation fails, engine uses single model."""
        from mother.bridge import EngineBridge

        bridge = EngineBridge(provider="claude", model="claude-sonnet-4-20250514")
        bridge.set_chat_model("claude-haiku-4-5-20251001")

        with patch("core.engine.MotherlabsEngine") as MockEngine, \
             patch("core.llm.create_client", side_effect=Exception("No API key")):
            MockEngine.return_value = MagicMock()
            bridge._get_engine()

            # Should fall back to main model for everything
            call_kwargs = MockEngine.call_args[1]
            assert call_kwargs["model"] == "claude-sonnet-4-20250514"
            assert call_kwargs["quality_llm"] is None


# ── Hollow artifact fallback to legacy ─────────────────────────────


class TestHollowArtifactFallback:
    def test_hollow_artifact_error_caught_by_fallback(self):
        """Engine code catches 'hollow artifact' CompilationError and continues."""
        import ast

        with open("core/engine.py") as f:
            source = f.read()

        assert "hollow artifact" in source
        assert "_staged_ok" in source

        lines = source.split("\n")
        found_catch = False
        found_check = False
        found_fallback = False
        for line in lines:
            if "except CompilationError" in line:
                found_catch = True
            if found_catch and '"hollow artifact"' in line:
                found_check = True
            if "_staged_ok" in line and "not" in line and "if" in line:
                found_fallback = True

        assert found_catch
        assert found_check
        assert found_fallback

    def test_non_hollow_error_re_raised(self):
        """Non-hollow CompilationError in staged pipeline should re-raise."""
        with open("core/engine.py") as f:
            source = f.read()

        lines = source.split("\n")
        in_catch = False
        found_raise = False
        for line in lines:
            if "except CompilationError" in line:
                in_catch = True
            elif in_catch and line.strip() == "raise":
                found_raise = True
                break
            elif in_catch and "if not _staged_ok" in line:
                break

        assert found_raise


# ── Non-blocking compile ───────────────────────────────────────────


class TestNonBlockingCompile:
    def test_compile_running_flag_initialized(self):
        """ChatScreen should initialize _compile_running to False."""
        with open("mother/screens/chat.py") as f:
            source = f.read()

        assert "self._compile_running = False" in source

    def test_compile_worker_is_non_exclusive(self):
        """Compile worker should run with exclusive=False."""
        with open("mother/screens/chat.py") as f:
            source = f.read()

        for line in source.split("\n"):
            if "_compile_worker" in line and "run_worker" in line:
                assert "exclusive=False" in line
                return
        pytest.fail("run_worker(_compile_worker...) not found")

    def test_compile_guard_prevents_double_compile(self):
        """_run_compile should reject if _compile_running is True."""
        with open("mother/screens/chat.py") as f:
            source = f.read()

        in_method = False
        found_guard = False
        for line in source.split("\n"):
            if "def _run_compile" in line:
                in_method = True
            elif in_method and line.strip().startswith("def "):
                break
            elif in_method and "_compile_running" in line and "if" in line:
                found_guard = True
                break

        assert found_guard

    def test_compile_running_cleared_in_finally(self):
        """_compile_running should be set to False in the finally block."""
        with open("mother/screens/chat.py") as f:
            source = f.read()

        in_method = False
        in_finally = False
        found_clear = False
        for line in source.split("\n"):
            if "async def _compile_worker" in line:
                in_method = True
            elif in_method and line.strip() == "finally:":
                in_finally = True
            elif in_finally and "_compile_running = False" in line:
                found_clear = True
                break
            elif in_method and not in_finally and line.strip().startswith("def "):
                break

        assert found_clear
