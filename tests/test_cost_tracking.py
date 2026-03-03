"""
Phase 21: Cost Optimization tests.

Tests for:
- LLM client last_usage side-channel (core/llm.py)
- Token/cost data structures (core/telemetry.py)
- Protocol spec CostSpec (core/protocol_spec.py)
- Engine cost accumulation (core/engine.py)
- CostCapExceededError (core/exceptions.py)
- Error catalog E8xxx (core/error_catalog.py)
"""

import ast
import time
import pytest

from core.llm import BaseLLMClient, MockClient, FailoverClient
from core.telemetry import (
    TokenUsage,
    CostEstimate,
    PRICING_TABLE,
    estimate_cost,
    CompilationMetrics,
    AggregateMetrics,
    HealthSnapshot,
    aggregate_metrics,
    compute_health,
    metrics_to_dict,
    aggregate_to_dict,
    health_to_dict,
    token_usage_to_dict,
    cost_estimate_to_dict,
)
from core.protocol_spec import PROTOCOL, CostSpec
from core.exceptions import CostCapExceededError
from core.error_catalog import get_entry


# =============================================================================
# HELPERS
# =============================================================================


def make_usage(
    input_tokens: int = 100,
    output_tokens: int = 200,
    provider: str = "mock",
    model: str = "mock-1",
) -> TokenUsage:
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        provider=provider,
        model=model,
    )


def make_metrics(
    compilation_id: str = "abc123",
    success: bool = True,
    total_duration: float = 10.0,
    total_input_tokens: int = 0,
    total_output_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
    provider: str = "mock",
    model: str = "mock-1",
    timestamp: float = None,
) -> CompilationMetrics:
    return CompilationMetrics(
        compilation_id=compilation_id,
        timestamp=timestamp or time.time(),
        success=success,
        total_duration=total_duration,
        stage_timings=(("intent", 1.0), ("synthesis", 5.0)),
        dialogue_turns=4,
        component_count=5,
        insight_count=3,
        verification_score=80,
        verification_mode="deterministic",
        cache_hits=1,
        cache_misses=2,
        retry_count=0,
        provider=provider,
        model=model,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        estimated_cost_usd=estimated_cost_usd,
    )


# =============================================================================
# 21.1: LLM CLIENT last_usage TESTS
# =============================================================================


class TestBaseLLMClientLastUsage:
    """BaseLLMClient and subclass last_usage tests."""

    def test_base_client_has_last_usage(self):
        """BaseLLMClient.__init__ sets self.last_usage = {}."""
        client = MockClient()
        assert hasattr(client, "last_usage")
        # Before any call, it's the initial value from BaseLLMClient.__init__
        # MockClient's first call will set it. Check type at least.
        assert isinstance(client.last_usage, dict)

    def test_mock_client_sets_last_usage(self):
        """MockClient.complete() sets deterministic last_usage."""
        client = MockClient()
        client.complete([{"role": "user", "content": "test"}])
        assert client.last_usage["input_tokens"] == 10
        assert client.last_usage["output_tokens"] == 20
        assert client.last_usage["total_tokens"] == 30

    def test_mock_client_last_usage_keys(self):
        """MockClient.last_usage has the expected keys."""
        client = MockClient()
        client.complete_with_system("sys", "user")
        assert set(client.last_usage.keys()) == {"input_tokens", "output_tokens", "total_tokens"}

    def test_mock_client_last_usage_resets_each_call(self):
        """Each call overwrites last_usage (fresh dict each time)."""
        client = MockClient()
        client.complete([{"role": "user", "content": "a"}])
        first = client.last_usage
        client.complete([{"role": "user", "content": "b"}])
        second = client.last_usage
        # They should be equal in value but be separate dict instances
        assert first == second
        assert first is not second

    def test_mock_client_complete_with_system_sets_usage(self):
        """complete_with_system also sets last_usage."""
        client = MockClient()
        client.complete_with_system("system", "user content")
        assert client.last_usage["total_tokens"] == 30

    def test_failover_relays_last_usage(self):
        """FailoverClient copies last_usage from successful provider."""
        mock1 = MockClient()
        failover = FailoverClient([mock1])
        failover.complete([{"role": "user", "content": "test"}])
        assert failover.last_usage == {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}

    def test_failover_relays_usage_after_failover(self):
        """After failover, last_usage comes from the successful provider."""

        class FailingClient(BaseLLMClient):
            def __init__(self):
                super().__init__()

            def complete(self, messages, **kwargs):
                raise RuntimeError("fail")

            def complete_with_system(self, system_prompt, user_content, **kwargs):
                raise RuntimeError("fail")

        failing = FailingClient()
        mock = MockClient()
        failover = FailoverClient([failing, mock])
        failover.complete([{"role": "user", "content": "test"}])
        assert failover.last_usage == {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}


# =============================================================================
# 21.2: TOKEN/COST DATA STRUCTURES
# =============================================================================


class TestTokenUsage:

    def test_token_usage_frozen(self):
        """TokenUsage is immutable."""
        u = make_usage()
        with pytest.raises(AttributeError):
            u.input_tokens = 999

    def test_token_usage_fields(self):
        u = make_usage(input_tokens=50, output_tokens=100, provider="claude", model="claude-sonnet-4")
        assert u.input_tokens == 50
        assert u.output_tokens == 100
        assert u.total_tokens == 150
        assert u.provider == "claude"
        assert u.model == "claude-sonnet-4"


class TestCostEstimate:

    def test_cost_estimate_frozen(self):
        """CostEstimate is immutable."""
        u = make_usage()
        c = estimate_cost(u)
        with pytest.raises(AttributeError):
            c.total_cost = 999.0

    def test_cost_estimate_fields(self):
        u = make_usage(model="claude-sonnet-4-20250514")
        c = estimate_cost(u)
        assert c.input_cost >= 0
        assert c.output_cost >= 0
        assert c.total_cost == c.input_cost + c.output_cost
        assert c.token_usage is u


class TestEstimateCost:

    def test_estimate_cost_claude_sonnet(self):
        """Claude Sonnet pricing: $3/1M input, $15/1M output."""
        u = make_usage(input_tokens=1_000_000, output_tokens=1_000_000, model="claude-sonnet-4-20250514")
        c = estimate_cost(u)
        assert abs(c.input_cost - 3.0) < 0.01
        assert abs(c.output_cost - 15.0) < 0.01

    def test_estimate_cost_unknown_model(self):
        """Unknown model yields zero cost."""
        u = make_usage(input_tokens=1000, output_tokens=2000, model="unknown-model-xyz")
        c = estimate_cost(u)
        assert c.total_cost == 0.0

    def test_estimate_cost_gemini_flash(self):
        """Gemini Flash: $0.10/1M input, $0.40/1M output."""
        u = make_usage(input_tokens=1_000_000, output_tokens=1_000_000, model="gemini-2.0-flash")
        c = estimate_cost(u)
        assert abs(c.input_cost - 0.10) < 0.01
        assert abs(c.output_cost - 0.40) < 0.01

    def test_estimate_cost_grok(self):
        """Grok-4 pricing."""
        u = make_usage(input_tokens=1_000_000, output_tokens=1_000_000, model="grok-4-1-fast-reasoning")
        c = estimate_cost(u)
        assert abs(c.input_cost - 2.0) < 0.01
        assert abs(c.output_cost - 10.0) < 0.01

    def test_pricing_table_not_empty(self):
        assert len(PRICING_TABLE) >= 8


class TestCompilationMetricsWithCost:

    def test_default_tokens(self):
        """New cost fields default to 0."""
        m = CompilationMetrics(
            compilation_id="x",
            timestamp=time.time(),
            success=True,
            total_duration=1.0,
            stage_timings=(),
            dialogue_turns=0,
            component_count=0,
            insight_count=0,
            verification_score=0,
            verification_mode="deterministic",
            cache_hits=0,
            cache_misses=0,
            retry_count=0,
            provider="mock",
            model="mock-1",
        )
        assert m.total_input_tokens == 0
        assert m.total_output_tokens == 0
        assert m.estimated_cost_usd == 0.0

    def test_with_tokens(self):
        m = make_metrics(total_input_tokens=500, total_output_tokens=1000, estimated_cost_usd=0.05)
        assert m.total_input_tokens == 500
        assert m.total_output_tokens == 1000
        assert m.estimated_cost_usd == 0.05


class TestAggregateMetricsCost:

    def test_aggregate_cost_fields(self):
        """Aggregate includes cost totals and averages."""
        m1 = make_metrics(total_input_tokens=100, total_output_tokens=200, estimated_cost_usd=0.01)
        m2 = make_metrics(total_input_tokens=300, total_output_tokens=400, estimated_cost_usd=0.03)
        agg = aggregate_metrics([m1, m2])
        assert agg.total_input_tokens == 400
        assert agg.total_output_tokens == 600
        assert abs(agg.total_cost_usd - 0.04) < 1e-9
        assert abs(agg.avg_cost_usd - 0.02) < 1e-9
        assert abs(agg.avg_tokens_per_compilation - 500.0) < 1e-9

    def test_aggregate_empty_cost_fields(self):
        """Empty aggregate has zero cost fields."""
        agg = aggregate_metrics([])
        assert agg.total_input_tokens == 0
        assert agg.total_output_tokens == 0
        assert agg.total_cost_usd == 0.0
        assert agg.avg_cost_usd == 0.0
        assert agg.avg_tokens_per_compilation == 0.0


class TestHealthSnapshotCost:

    def test_health_snapshot_cost_field(self):
        """HealthSnapshot has recent_total_cost_usd."""
        m = make_metrics(estimated_cost_usd=1.5, timestamp=time.time())
        h = compute_health([m], uptime=100.0, corpus_size=1)
        assert h.recent_total_cost_usd == 1.5

    def test_health_cost_warning(self):
        """High avg cost adds issue."""
        m = make_metrics(estimated_cost_usd=5.0, timestamp=time.time())
        h = compute_health([m], uptime=100.0, corpus_size=1, cost_warn_threshold=3.0)
        assert any("cost" in issue.lower() for issue in h.issues)


class TestCostSerialization:

    def test_metrics_to_dict_includes_tokens(self):
        m = make_metrics(total_input_tokens=42, total_output_tokens=84, estimated_cost_usd=0.1)
        d = metrics_to_dict(m)
        assert d["total_input_tokens"] == 42
        assert d["total_output_tokens"] == 84
        assert d["estimated_cost_usd"] == 0.1

    def test_aggregate_to_dict_includes_cost(self):
        m = make_metrics(estimated_cost_usd=0.5)
        agg = aggregate_metrics([m])
        d = aggregate_to_dict(agg)
        assert "total_cost_usd" in d
        assert "avg_cost_usd" in d
        assert "avg_tokens_per_compilation" in d

    def test_health_to_dict_includes_cost(self):
        m = make_metrics(estimated_cost_usd=2.0, timestamp=time.time())
        h = compute_health([m], uptime=100.0, corpus_size=1)
        d = health_to_dict(h)
        assert "recent_total_cost_usd" in d
        assert d["recent_total_cost_usd"] == 2.0

    def test_token_usage_to_dict(self):
        u = make_usage(input_tokens=10, output_tokens=20, provider="p", model="m")
        d = token_usage_to_dict(u)
        assert d == {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "provider": "p",
            "model": "m",
        }

    def test_cost_estimate_to_dict(self):
        u = make_usage(model="claude-sonnet-4-x")
        c = estimate_cost(u)
        d = cost_estimate_to_dict(c)
        assert "input_cost" in d
        assert "output_cost" in d
        assert "total_cost" in d
        assert "token_usage" in d


# =============================================================================
# 21.2: PROTOCOL SPEC COST
# =============================================================================


class TestCostSpec:

    def test_cost_spec_defaults(self):
        """CostSpec has expected defaults matching documented $5/$50 caps."""
        cs = CostSpec()
        assert cs.per_compilation_cap_usd == 5.0
        assert cs.per_compilation_warn_usd == 2.5
        assert cs.session_cap_usd == 50.0
        assert cs.health_cost_warn_threshold == 2.0

    def test_cost_spec_in_protocol(self):
        """PROTOCOL singleton has cost spec."""
        assert hasattr(PROTOCOL, "cost")
        assert isinstance(PROTOCOL.cost, CostSpec)
        assert PROTOCOL.cost.per_compilation_cap_usd == 5.0

    def test_protocol_spec_still_leaf_module(self):
        """protocol_spec.py must import only stdlib."""
        with open("core/protocol_spec.py", "r") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module is None or not node.module.startswith("core"), \
                    f"protocol_spec imports from project: {node.module}"


class TestTelemetryLeafModule:

    def test_telemetry_still_leaf_module(self):
        """telemetry.py must import only stdlib."""
        with open("core/telemetry.py", "r") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module is None or not node.module.startswith("core"), \
                    f"telemetry imports from project: {node.module}"


# =============================================================================
# 21.3: ENGINE COST ACCUMULATION
# =============================================================================


class TestEngineCostTracking:

    def _make_engine(self):
        from core.engine import MotherlabsEngine
        return MotherlabsEngine(llm_client=MockClient())

    def test_collect_usage_from_mock(self):
        """_collect_usage reads MockClient.last_usage."""
        engine = self._make_engine()
        # Trigger a call so last_usage is populated
        engine.llm.complete([{"role": "user", "content": "test"}])
        engine._collect_usage()
        assert len(engine._compilation_tokens) == 1
        assert engine._compilation_tokens[0].input_tokens == 10

    def test_collect_usage_empty_when_no_data(self):
        """_collect_usage does nothing when last_usage is empty."""
        engine = self._make_engine()
        # last_usage is {} before any call
        engine._compilation_tokens = []
        engine.llm.last_usage = {}
        engine._collect_usage()
        assert len(engine._compilation_tokens) == 0

    def test_check_cost_cap_under_limit(self):
        """No exception when cost is under cap."""
        engine = self._make_engine()
        # Mock tokens: 10 in + 20 out = 30 tokens at mock model = $0
        engine.llm.complete([{"role": "user", "content": "test"}])
        engine._collect_usage()
        engine._check_cost_cap()  # Should not raise

    def test_check_cost_cap_exceeds_raises(self):
        """CostCapExceededError when cost exceeds cap."""
        engine = self._make_engine()
        # Inject a huge usage manually
        huge = TokenUsage(
            input_tokens=10_000_000,
            output_tokens=10_000_000,
            total_tokens=20_000_000,
            provider="claude",
            model="claude-sonnet-4-20250514",
        )
        engine._compilation_tokens = [huge]
        with pytest.raises(CostCapExceededError) as exc_info:
            engine._check_cost_cap()
        assert exc_info.value.cap == PROTOCOL.cost.per_compilation_cap_usd

    def test_compile_accumulates_tokens(self):
        """After compile(), _compilation_tokens is populated."""
        engine = self._make_engine()
        engine.compile("Build a library management system with books, members, and loans")
        # MockClient sets usage on each call, _collect_usage is called after each stage
        assert len(engine._compilation_tokens) > 0

    def test_record_metrics_includes_cost(self):
        """_record_metrics stores token/cost data."""
        engine = self._make_engine()
        engine._record_metrics(
            compilation_id="test",
            success=True,
            total_duration=1.0,
            stage_timings={"intent": 0.5},
            dialogue_turns=2,
            component_count=3,
            insight_count=2,
            verification_score=80,
            verification_mode="deterministic",
            cache_hits=0,
            cache_misses=1,
            retry_count=0,
            total_input_tokens=100,
            total_output_tokens=200,
            estimated_cost_usd=0.05,
        )
        assert len(engine._metrics_buffer) == 1
        m = engine._metrics_buffer[0]
        assert m.total_input_tokens == 100
        assert m.total_output_tokens == 200
        assert m.estimated_cost_usd == 0.05

    def test_session_cost_accumulates(self):
        """Session cost grows across compilations."""
        engine = self._make_engine()
        initial = engine._session_cost_usd
        engine.compile("Build a restaurant reservation system with tables, guests, and bookings")
        # Session cost may be 0 (mock model has no pricing) but should not decrease
        assert engine._session_cost_usd >= initial


# =============================================================================
# 21.3: COST CAP ERROR
# =============================================================================


class TestCostCapExceededError:

    def test_cost_cap_error_attributes(self):
        err = CostCapExceededError("over budget", current_cost=6.0, cap=5.0)
        assert err.current_cost == 6.0
        assert err.cap == 5.0
        assert err.error_code == "E8001"

    def test_cost_cap_error_user_dict(self):
        err = CostCapExceededError("over budget", current_cost=6.0, cap=5.0)
        d = err.to_user_dict()
        assert "error" in d
        assert d["error_code"] == "E8001"


class TestErrorCatalogE8xxx:

    def test_error_catalog_e8001(self):
        entry = get_entry("E8001")
        assert entry is not None
        assert "cost" in entry.title.lower() or "cap" in entry.title.lower()

    def test_error_catalog_e8002(self):
        entry = get_entry("E8002")
        assert entry is not None
        assert "session" in entry.title.lower()
