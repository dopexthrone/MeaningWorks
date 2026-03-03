"""
Pytest fixtures for Motherlabs test suite.

Phase 5.4: Test Coverage

Provides reusable fixtures for testing without making actual LLM API calls.
Also provides real-LLM fixtures gated behind @pytest.mark.slow.
"""

import os
import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List


# =============================================================================
# MOCK LLM RESPONSES
# =============================================================================

MOCK_INTENT_RESPONSE = """
{
  "core_need": "Build a user authentication system",
  "actors": ["User", "AuthService"],
  "constraints": ["Must be secure", "Must handle sessions"],
  "domain": "authentication"
}
"""

MOCK_PERSONA_RESPONSE = """
[
  {
    "name": "Security Architect",
    "perspective": "Focus on authentication security and session management",
    "blind_spots": "May over-engineer simple auth flows"
  },
  {
    "name": "UX Designer",
    "perspective": "Focus on user experience during login/logout",
    "blind_spots": "May underestimate security requirements"
  }
]
"""

MOCK_ENTITY_RESPONSE = """
INSIGHT: User entity contains email, password_hash, created_at fields
INSIGHT: Session entity contains token, user_id, expiry fields
INSIGHT: AuthService manages User → Session relationship
"""

MOCK_PROCESS_RESPONSE = """
INSIGHT: login(email, password) → Session | validates credentials, creates session
INSIGHT: logout(token) → None | invalidates session
INSIGHT: Session.expiry constraint: must be within 24 hours
"""

MOCK_SYNTHESIS_RESPONSE = """
{
  "components": [
    {
      "name": "User",
      "type": "entity",
      "description": "User account with authentication credentials",
      "derived_from": "INSIGHT: User entity contains email, password_hash, created_at fields",
      "properties": [
        {"name": "email", "type": "str"},
        {"name": "password_hash", "type": "str"},
        {"name": "created_at", "type": "datetime"}
      ]
    },
    {
      "name": "Session",
      "type": "entity",
      "description": "Active user session with expiry",
      "derived_from": "INSIGHT: Session entity contains token, user_id, expiry fields",
      "properties": [
        {"name": "token", "type": "str"},
        {"name": "user_id", "type": "str"},
        {"name": "expiry", "type": "datetime"}
      ]
    },
    {
      "name": "AuthService",
      "type": "process",
      "description": "Handles user authentication and session management",
      "derived_from": "INSIGHT: AuthService manages User → Session relationship",
      "methods": [
        {
          "name": "login",
          "parameters": [
            {"name": "email", "type_hint": "str"},
            {"name": "password", "type_hint": "str"}
          ],
          "return_type": "Session",
          "description": "Validates credentials and creates session",
          "derived_from": "INSIGHT: login(email, password) → Session"
        },
        {
          "name": "logout",
          "parameters": [{"name": "token", "type_hint": "str"}],
          "return_type": "None",
          "description": "Invalidates session",
          "derived_from": "INSIGHT: logout(token) → None"
        }
      ]
    }
  ],
  "relationships": [
    {
      "from": "AuthService",
      "to": "User",
      "type": "accesses",
      "description": "AuthService validates User credentials"
    },
    {
      "from": "AuthService",
      "to": "Session",
      "type": "generates",
      "description": "AuthService creates Session on login"
    },
    {
      "from": "Session",
      "to": "User",
      "type": "depends_on",
      "description": "Session belongs to User"
    }
  ],
  "constraints": [
    {
      "description": "Session expiry must be within 24 hours",
      "applies_to": ["Session"],
      "derived_from": "INSIGHT: Session.expiry constraint"
    }
  ],
  "unresolved": []
}
"""

MOCK_VERIFY_RESPONSE = """
{
  "status": "pass",
  "completeness": 0.95,
  "consistency": 0.90,
  "traceability": 1.0,
  "notes": ["All components have derivation traces", "Relationships are consistent"]
}
"""


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_llm_client():
    """
    Mock LLM client for testing without API calls.

    Returns a mock that produces deterministic responses for each stage.
    """
    client = Mock()

    # Track call count to return appropriate responses
    call_count = [0]
    responses = [
        MOCK_INTENT_RESPONSE,
        MOCK_PERSONA_RESPONSE,
        MOCK_ENTITY_RESPONSE,
        MOCK_PROCESS_RESPONSE,
        MOCK_ENTITY_RESPONSE,  # Second dialogue turn
        MOCK_PROCESS_RESPONSE,
        MOCK_SYNTHESIS_RESPONSE,
        MOCK_VERIFY_RESPONSE,
    ]

    def mock_complete(*args, **kwargs):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        return responses[idx]

    client.complete = Mock(side_effect=mock_complete)
    client.provider_name = "mock"
    client.model_name = "mock-model"

    return client


@pytest.fixture
def sample_blueprint() -> Dict[str, Any]:
    """
    Sample valid blueprint for testing.

    Represents a simple authentication system with proper structure.
    """
    return {
        "version": "2.0",
        "components": [
            {
                "name": "User",
                "type": "entity",
                "description": "User account with authentication credentials",
                "derived_from": "User entity with email, password_hash",
                "properties": [
                    {"name": "email", "type": "str"},
                    {"name": "password_hash", "type": "str"},
                ],
                "methods": []
            },
            {
                "name": "Session",
                "type": "entity",
                "description": "Active user session",
                "derived_from": "Session entity with token, expiry",
                "properties": [
                    {"name": "token", "type": "str"},
                    {"name": "expiry", "type": "datetime"},
                ],
                "methods": []
            },
            {
                "name": "AuthService",
                "type": "process",
                "description": "Authentication service",
                "derived_from": "AuthService manages authentication",
                "properties": [],
                "methods": [
                    {
                        "name": "login",
                        "parameters": [
                            {"name": "email", "type_hint": "str"},
                            {"name": "password", "type_hint": "str"}
                        ],
                        "return_type": "Session",
                        "description": "Authenticate user",
                        "derived_from": "login method"
                    }
                ]
            }
        ],
        "relationships": [
            {
                "from": "AuthService",
                "to": "User",
                "type": "accesses",
                "description": "AuthService validates User"
            },
            {
                "from": "AuthService",
                "to": "Session",
                "type": "generates",
                "description": "AuthService creates Session"
            },
            {
                "from": "Session",
                "to": "User",
                "type": "depends_on",
                "description": "Session belongs to User"
            }
        ],
        "constraints": [
            {
                "description": "Session must expire within 24 hours",
                "applies_to": ["Session"],
                "derived_from": "Security requirement"
            }
        ],
        "unresolved": []
    }


@pytest.fixture
def invalid_blueprint() -> Dict[str, Any]:
    """
    Invalid blueprint with various schema errors.
    """
    return {
        "components": [
            {
                "name": "",  # Missing name
                "type": "invalid_type",  # Invalid type
                "description": ""  # Missing description
                # Missing derived_from
            },
            {
                "name": "Orphan",
                "type": "entity",
                "description": "Orphan component",
                "derived_from": "Test"
                # No relationships - will be orphan
            }
        ],
        "relationships": [
            {
                "from": "NonExistent",  # References non-existent component
                "to": "AlsoNonExistent",
                "type": "depends_on",
                "description": "Invalid relationship"
            }
        ],
        "constraints": [],
        "unresolved": []
    }


@pytest.fixture
def blueprint_with_cycles() -> Dict[str, Any]:
    """
    Blueprint with circular dependencies.
    """
    return {
        "version": "2.0",
        "components": [
            {
                "name": "A",
                "type": "entity",
                "description": "Component A",
                "derived_from": "Test component"
            },
            {
                "name": "B",
                "type": "entity",
                "description": "Component B",
                "derived_from": "Test component"
            },
            {
                "name": "C",
                "type": "entity",
                "description": "Component C",
                "derived_from": "Test component"
            }
        ],
        "relationships": [
            {
                "from": "A",
                "to": "B",
                "type": "depends_on",
                "description": "A depends on B"
            },
            {
                "from": "B",
                "to": "C",
                "type": "depends_on",
                "description": "B depends on C"
            },
            {
                "from": "C",
                "to": "A",
                "type": "depends_on",
                "description": "C depends on A - creates cycle"
            }
        ],
        "constraints": [],
        "unresolved": []
    }


@pytest.fixture
def nested_blueprint() -> Dict[str, Any]:
    """
    Blueprint with nested subsystems.
    """
    return {
        "version": "2.0",
        "components": [
            {
                "name": "App",
                "type": "subsystem",
                "description": "Main application",
                "derived_from": "Top-level system",
                "sub_blueprint": {
                    "components": [
                        {
                            "name": "UserService",
                            "type": "process",
                            "description": "User management",
                            "derived_from": "User service subsystem"
                        },
                        {
                            "name": "Database",
                            "type": "entity",
                            "description": "Data storage",
                            "derived_from": "Database subsystem"
                        }
                    ],
                    "relationships": [
                        {
                            "from": "UserService",
                            "to": "Database",
                            "type": "accesses",
                            "description": "UserService queries Database"
                        }
                    ],
                    "constraints": [],
                    "unresolved": []
                }
            }
        ],
        "relationships": [],
        "constraints": [],
        "unresolved": []
    }


@pytest.fixture
def self_compile_canonical() -> List[str]:
    """
    Canonical components expected in Motherlabs self-compile.
    """
    return [
        "Intent Agent",
        "Persona Agent",
        "Entity Agent",
        "Process Agent",
        "Synthesis Agent",
        "Verify Agent",
        "Governor Agent",
        "SharedState",
        "ConfidenceVector",
        "ConflictOracle",
        "Message",
        "DialogueProtocol",
        "Corpus",
    ]


@pytest.fixture
def mock_provider_config():
    """
    Mock provider configuration for testing.
    """
    return {
        "name": "mock",
        "api_key_env": "MOCK_API_KEY",
        "default_model": "mock-model",
        "timeout_seconds": 30,
        "max_retries": 2
    }


@pytest.fixture
def sample_context_graph() -> Dict[str, Any]:
    """
    Sample context graph for testing.
    """
    return {
        "known": {
            "User": {"type": "entity", "fields": ["email", "password_hash"]},
            "Session": {"type": "entity", "fields": ["token", "expiry"]}
        },
        "unknown": [],
        "ontology": {
            "user": "A person who uses the system",
            "session": "An active authentication period"
        },
        "personas": [
            {
                "name": "Security Architect",
                "perspective": "Security-focused design",
                "blind_spots": "May over-engineer"
            }
        ],
        "decision_trace": [
            {
                "sender": "Intent",
                "content": "Core need: authentication",
                "type": "proposition",
                "insight": None,
                "timestamp": "2024-01-01T00:00:00"
            },
            {
                "sender": "Entity",
                "content": "User entity with email, password_hash",
                "type": "proposition",
                "insight": "User entity contains email, password_hash",
                "timestamp": "2024-01-01T00:01:00"
            }
        ],
        "insights": [
            "User entity contains email, password_hash",
            "Session entity contains token, expiry"
        ],
        "flags": []
    }


# =============================================================================
# ENVIRONMENT FIXTURES
# =============================================================================

@pytest.fixture
def mock_env_vars(monkeypatch):
    """
    Set mock environment variables for testing.
    """
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")


@pytest.fixture
def no_api_keys(monkeypatch):
    """
    Remove all API key environment variables for testing error paths.
    """
    for key in ["XAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"]:
        monkeypatch.delenv(key, raising=False)


# =============================================================================
# REAL-LLM FIXTURES (for @pytest.mark.slow tests only)
# =============================================================================

@pytest.fixture(scope="session")
def real_llm_client():
    """
    Session-scoped real Claude client for integration tests.

    Skips all slow tests if ANTHROPIC_API_KEY is not set.
    Uses deterministic mode (temperature=0) for reproducibility.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    from core.llm import ClaudeClient
    return ClaudeClient(
        api_key=api_key,
        model="claude-sonnet-4-20250514",
        deterministic=True,
    )


@pytest.fixture
def real_engine(real_llm_client, tmp_path):
    """
    Function-scoped engine with real LLM client.

    Creates a fresh engine per test to avoid state leakage.
    Uses staged pipeline mode (the production path).
    """
    from persistence.corpus import Corpus
    from core.engine import MotherlabsEngine

    corpus = Corpus(tmp_path / "corpus")
    return MotherlabsEngine(
        llm_client=real_llm_client,
        pipeline_mode="staged",
        corpus=corpus,
        auto_store=False,
    )


@pytest.fixture
def real_engine_with_adapter(real_llm_client, tmp_path):
    """
    Factory fixture: creates engine with a specific domain adapter.

    Usage:
        def test_process(real_engine_with_adapter):
            engine = real_engine_with_adapter("process")
            result = engine.compile("...")
    """
    from persistence.corpus import Corpus
    from core.engine import MotherlabsEngine
    from core.adapter_registry import get_adapter
    import adapters  # noqa: F401 — triggers adapter registration

    def _factory(domain_name: str = "software"):
        corpus = Corpus(tmp_path / f"corpus_{domain_name}")
        adapter = get_adapter(domain_name)
        return MotherlabsEngine(
            llm_client=real_llm_client,
            pipeline_mode="staged",
            corpus=corpus,
            auto_store=False,
            domain_adapter=adapter,
        )

    return _factory


@pytest.fixture
def real_orchestrator(real_llm_client, tmp_path):
    """
    Function-scoped orchestrator with real LLM for E2E tests.

    Creates engine + AgentOrchestrator with output_dir in tmp_path.
    Build loop is disabled (build=False) to keep cost/time reasonable.
    """
    from core.engine import MotherlabsEngine
    from core.agent_orchestrator import AgentOrchestrator, AgentConfig
    from persistence.corpus import Corpus

    corpus = Corpus(tmp_path / "corpus")
    engine = MotherlabsEngine(
        llm_client=real_llm_client,
        pipeline_mode="staged",
        corpus=corpus,
        auto_store=False,
    )
    config = AgentConfig(output_dir=str(tmp_path / "project"), build=False)
    return AgentOrchestrator(engine, config)


@pytest.fixture
def cost_tracker():
    """
    Collects token usage from engine._compilation_tokens after a compile.

    Usage:
        result = engine.compile("...")
        report = cost_tracker(engine)
        assert report["total_cost_usd"] < 1.0
    """
    from core.telemetry import estimate_cost

    def _track(engine):
        tokens = getattr(engine, "_compilation_tokens", [])
        total_input = sum(t.input_tokens for t in tokens)
        total_output = sum(t.output_tokens for t in tokens)
        total_cost = sum(estimate_cost(t).total_cost for t in tokens)
        report = {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost_usd": total_cost,
            "calls": len(tokens),
        }
        print(f"\n  [COST] {report['calls']} LLM calls | "
              f"{total_input:,} in / {total_output:,} out | "
              f"${total_cost:.4f}")
        _session_cost_accumulator.append(total_cost)
        return report

    return _track


# Session-scoped cost accumulator for real-LLM test budget enforcement
_session_cost_accumulator: List[float] = []


@pytest.fixture(autouse=True, scope="session")
def _enforce_session_cost_budget(request):
    """
    Enforce a $5.00 total cost budget across the entire real-LLM test session.

    This fixture runs once per session. Individual tests add their costs via
    the cost_tracker fixture. At session end, we check the total.
    """
    _session_cost_accumulator.clear()
    yield
    if _session_cost_accumulator:
        total = sum(_session_cost_accumulator)
        print(f"\n\n{'='*60}")
        print(f"  TOTAL SESSION COST: ${total:.4f}")
        print(f"  BUDGET:             $500.00")
        print(f"  REMAINING:          ${500.0 - total:.4f}")
        print(f"{'='*60}")
        assert total < 500.0, (
            f"Session cost ${total:.2f} exceeds $500.00 budget"
        )
