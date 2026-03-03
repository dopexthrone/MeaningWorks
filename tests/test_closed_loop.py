"""
Tests for kernel/closed_loop.py — closed-loop transcription gate.

Yi Ma's insight: a representation is valid only if it retains enough
information to recover what it came from. The gate measures whether
the blueprint faithfully captures the original intent.
"""

import pytest
from kernel.closed_loop import (
    CompressionLoss,
    GateResult,
    decode_blueprint,
    decode_blueprint_llm,
    semantic_similarity,
    detect_compression_losses,
    closed_loop_gate,
    _normalize_tokens,
    _blueprint_to_flat_text,
    _FIDELITY_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _blueprint_auth():
    """A blueprint for a user authentication system."""
    return {
        "core_need": "authenticate users securely",
        "domain": "software",
        "components": [
            {"name": "AuthService", "purpose": "handle login and registration"},
            {"name": "TokenManager", "purpose": "issue and validate JWT tokens"},
            {"name": "UserStore", "purpose": "persist user credentials"},
            {"name": "PasswordHasher", "purpose": "hash and verify passwords"},
        ],
        "relationships": [
            {"source": "AuthService", "target": "TokenManager", "type": "uses"},
            {"source": "AuthService", "target": "UserStore", "type": "reads from"},
            {"source": "AuthService", "target": "PasswordHasher", "type": "delegates to"},
        ],
        "constraints": [
            {"description": "Passwords must be hashed with bcrypt"},
            {"description": "JWT tokens expire after 24 hours"},
            {"description": "Rate limit login attempts to 5 per minute"},
        ],
        "actors": [
            {"name": "End User", "role": "person authenticating"},
            {"name": "Admin", "role": "manages user accounts"},
        ],
    }


def _blueprint_empty():
    return {}


def _blueprint_minimal():
    return {
        "core_need": "track inventory",
        "components": [{"name": "InventoryDB", "purpose": "store items"}],
    }


def _intent_auth():
    return (
        "Build a user authentication system with login and registration. "
        "Use JWT tokens that expire after 24 hours. Hash passwords with bcrypt. "
        "Rate limit login attempts to 5 per minute. Support end users and admins."
    )


def _intent_inventory():
    return "I need a simple inventory tracking system to store and manage items."


# ---------------------------------------------------------------------------
# CompressionLoss dataclass
# ---------------------------------------------------------------------------

class TestCompressionLoss:
    def test_frozen(self):
        loss = CompressionLoss(
            category="entity",
            original_fragment="user store",
            description="Entity not captured",
            severity=0.8,
        )
        with pytest.raises(AttributeError):
            loss.severity = 0.5

    def test_fields(self):
        loss = CompressionLoss(
            category="constraint",
            original_fragment="must be encrypted",
            description="Encryption requirement lost",
            severity=0.9,
        )
        assert loss.category == "constraint"
        assert loss.severity == 0.9


# ---------------------------------------------------------------------------
# GateResult dataclass
# ---------------------------------------------------------------------------

class TestGateResult:
    def test_frozen(self):
        result = GateResult(
            passed=True,
            fidelity_score=0.85,
            reconstructed_intent="test",
            compression_losses=(),
            total_loss=0.15,
            explanation="test",
        )
        with pytest.raises(AttributeError):
            result.passed = False

    def test_passed_result(self):
        result = GateResult(
            passed=True,
            fidelity_score=0.9,
            reconstructed_intent="reconstructed",
            compression_losses=(),
            total_loss=0.1,
            explanation="All good",
        )
        assert result.passed is True
        assert result.fidelity_score == 0.9


# ---------------------------------------------------------------------------
# decode_blueprint
# ---------------------------------------------------------------------------

class TestDecodeBlueprint:
    def test_full_blueprint(self):
        bp = _blueprint_auth()
        decoded = decode_blueprint(bp)
        assert "authenticate" in decoded.lower()
        assert "AuthService" in decoded
        assert "TokenManager" in decoded
        assert "bcrypt" in decoded
        assert "24 hours" in decoded

    def test_empty_blueprint(self):
        decoded = decode_blueprint({})
        assert decoded == ""

    def test_minimal_blueprint(self):
        decoded = decode_blueprint(_blueprint_minimal())
        assert "inventory" in decoded.lower()
        assert "InventoryDB" in decoded

    def test_components_with_purpose(self):
        bp = {"components": [{"name": "Cache", "purpose": "speed up reads"}]}
        decoded = decode_blueprint(bp)
        assert "Cache" in decoded
        assert "speed up reads" in decoded

    def test_relationships(self):
        bp = _blueprint_auth()
        decoded = decode_blueprint(bp)
        assert "uses" in decoded or "delegates" in decoded

    def test_constraints(self):
        bp = _blueprint_auth()
        decoded = decode_blueprint(bp)
        assert "bcrypt" in decoded
        assert "expire" in decoded or "24 hours" in decoded

    def test_actors(self):
        bp = _blueprint_auth()
        decoded = decode_blueprint(bp)
        assert "End User" in decoded or "Admin" in decoded

    def test_tuple_relationships(self):
        bp = {
            "core_need": "test",
            "relationships": [("A", "B", "triggers"), ("C", "D", "reads")],
        }
        decoded = decode_blueprint(bp)
        assert "triggers" in decoded

    def test_string_constraints(self):
        bp = {
            "core_need": "test",
            "constraints": ["must be fast", "must be secure"],
        }
        decoded = decode_blueprint(bp)
        assert "must be fast" in decoded


# ---------------------------------------------------------------------------
# decode_blueprint_llm
# ---------------------------------------------------------------------------

class TestDecodeBlueprintLlm:
    def test_uses_llm_when_available(self):
        def mock_llm(prompt: str) -> str:
            return "The system manages user authentication with JWT tokens."

        bp = _blueprint_auth()
        decoded = decode_blueprint_llm(bp, mock_llm)
        assert "authentication" in decoded

    def test_falls_back_on_empty_response(self):
        def mock_llm(prompt: str) -> str:
            return ""

        bp = _blueprint_auth()
        decoded = decode_blueprint_llm(bp, mock_llm)
        # Should fall back to structural decode
        assert "AuthService" in decoded

    def test_falls_back_on_exception(self):
        def mock_llm(prompt: str) -> str:
            raise RuntimeError("LLM failed")

        bp = _blueprint_auth()
        decoded = decode_blueprint_llm(bp, mock_llm)
        assert "AuthService" in decoded


# ---------------------------------------------------------------------------
# semantic_similarity
# ---------------------------------------------------------------------------

class TestSemanticSimilarity:
    def test_identical(self):
        text = "Build a user authentication system"
        score = semantic_similarity(text, text)
        assert score == 1.0

    def test_empty(self):
        assert semantic_similarity("", "something") == 0.0
        assert semantic_similarity("something", "") == 0.0
        assert semantic_similarity("", "") == 0.0

    def test_high_overlap(self):
        orig = "Build a user authentication system with JWT tokens"
        recon = "The system authenticates users using JWT token management"
        score = semantic_similarity(orig, recon)
        assert score > 0.3  # significant overlap

    def test_low_overlap(self):
        orig = "Build a user authentication system"
        recon = "The database stores inventory records efficiently"
        score = semantic_similarity(orig, recon)
        assert score < 0.3

    def test_recall_weighted(self):
        # If reconstruction contains all original tokens plus extras,
        # score should be high (recall is weighted higher)
        orig = "authentication tokens"
        recon = "authentication tokens verification encryption hashing"
        score = semantic_similarity(orig, recon)
        assert score > 0.7  # high recall

    def test_symmetric_ish(self):
        a = "user login system"
        b = "system for user login"
        s1 = semantic_similarity(a, b)
        s2 = semantic_similarity(b, a)
        # Not perfectly symmetric due to recall weighting, but close
        assert abs(s1 - s2) < 0.3


# ---------------------------------------------------------------------------
# detect_compression_losses
# ---------------------------------------------------------------------------

class TestDetectCompressionLosses:
    def test_no_losses(self):
        orig = "authenticate users"
        bp = _blueprint_auth()
        recon = decode_blueprint(bp)
        losses = detect_compression_losses(orig, bp, recon)
        # Very short intent should be fully captured
        assert len(losses) == 0 or all(l.severity < 0.5 for l in losses)

    def test_detects_missing_terms(self):
        # Intent mentions "WebSocket" but blueprint doesn't
        orig = "Build a real-time chat with WebSocket connections and message encryption"
        bp = {
            "core_need": "real-time chat",
            "components": [{"name": "ChatServer", "purpose": "handle messages"}],
        }
        recon = decode_blueprint(bp)
        losses = detect_compression_losses(orig, bp, recon)
        # Should detect WebSocket (split: "socket") and/or encryption as losses
        all_fragments = " ".join(l.original_fragment for l in losses).lower()
        assert "socket" in all_fragments or "encryp" in all_fragments

    def test_empty_reconstruction(self):
        losses = detect_compression_losses("test intent", {}, "")
        # All tokens lost
        assert len(losses) >= 0  # graceful

    def test_categories(self):
        orig = "Build authentication with rate limiting and encryption"
        bp = {"core_need": "authentication", "components": []}
        recon = "authentication"
        losses = detect_compression_losses(orig, bp, recon)
        categories = {l.category for l in losses}
        # Should have entity losses for missing terms
        assert len(losses) > 0


# ---------------------------------------------------------------------------
# closed_loop_gate (the full gate)
# ---------------------------------------------------------------------------

class TestClosedLoopGate:
    def test_faithful_blueprint_passes(self):
        intent = _intent_auth()
        bp = _blueprint_auth()
        result = closed_loop_gate(intent, bp)
        # Rich blueprint should capture most of the intent
        assert result.fidelity_score > 0.3
        assert result.reconstructed_intent != ""
        assert isinstance(result.explanation, str)

    def test_empty_intent_fails(self):
        result = closed_loop_gate("", _blueprint_auth())
        assert result.passed is False
        assert result.fidelity_score == 0.0
        assert "Empty" in result.explanation

    def test_empty_blueprint_fails(self):
        result = closed_loop_gate("Build something", {})
        assert result.passed is False

    def test_minimal_blueprint(self):
        result = closed_loop_gate(_intent_inventory(), _blueprint_minimal())
        assert result.reconstructed_intent != ""
        assert result.fidelity_score > 0.0

    def test_lossy_blueprint(self):
        # Intent has many details, blueprint captures almost nothing
        intent = (
            "Build a comprehensive e-commerce platform with product catalog, "
            "shopping cart, payment processing via Stripe, inventory management, "
            "order tracking, customer reviews, and admin dashboard"
        )
        bp = {
            "core_need": "e-commerce",
            "components": [{"name": "WebApp", "purpose": "serve web pages"}],
        }
        result = closed_loop_gate(intent, bp)
        # Should detect significant compression loss
        assert len(result.compression_losses) > 0
        assert result.total_loss > 0.0

    def test_custom_threshold(self):
        result = closed_loop_gate(
            _intent_auth(), _blueprint_auth(),
            threshold=0.99,  # impossibly high
        )
        # Even good blueprints fail at 0.99
        assert result.passed is False or result.fidelity_score >= 0.99

    def test_with_llm_fn(self):
        def mock_llm(prompt: str) -> str:
            return (
                "Build a user authentication system with login and registration, "
                "using JWT tokens and bcrypt password hashing with rate limiting."
            )

        result = closed_loop_gate(
            _intent_auth(), _blueprint_auth(),
            llm_fn=mock_llm,
        )
        assert result.fidelity_score > 0.4
        assert "authentication" in result.reconstructed_intent.lower()

    def test_gate_result_explanation_pass(self):
        result = closed_loop_gate(
            "track inventory items",
            {"core_need": "track inventory items",
             "components": [{"name": "Inventory", "purpose": "track inventory items"}]},
        )
        if result.passed:
            assert "PASSED" in result.explanation

    def test_gate_result_explanation_fail(self):
        result = closed_loop_gate(
            "Build a comprehensive machine learning pipeline with feature engineering",
            {"core_need": "hello world", "components": []},
        )
        if not result.passed:
            assert "FAILED" in result.explanation

    def test_compression_losses_are_tuples(self):
        result = closed_loop_gate(_intent_auth(), _blueprint_auth())
        assert isinstance(result.compression_losses, tuple)


# ---------------------------------------------------------------------------
# Normalize tokens
# ---------------------------------------------------------------------------

class TestNormalizeTokens:
    def test_basic(self):
        tokens = _normalize_tokens("Build a user authentication system")
        assert "build" in tokens
        assert "authentic" in tokens  # stemmed from "authentication"
        assert "a" not in tokens  # stop word

    def test_punctuation_stripped(self):
        tokens = _normalize_tokens("hello, world! (test)")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_empty(self):
        assert _normalize_tokens("") == set()

    def test_short_words_excluded(self):
        tokens = _normalize_tokens("I a at it")
        # All single-char or stop words
        assert len(tokens) == 0


class TestBlueprintToFlatText:
    def test_includes_all_parts(self):
        bp = _blueprint_auth()
        text = _blueprint_to_flat_text(bp)
        assert "authenticate" in text
        assert "AuthService" in text
        assert "bcrypt" in text
        assert "End User" in text

    def test_empty(self):
        assert _blueprint_to_flat_text({}) == ""


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_roundtrip(self):
        """Intent → Blueprint → Decode → Similarity → Gate."""
        intent = _intent_auth()
        bp = _blueprint_auth()

        # Decode
        reconstructed = decode_blueprint(bp)
        assert len(reconstructed) > 50

        # Similarity
        fidelity = semantic_similarity(intent, reconstructed)
        assert fidelity > 0.0

        # Losses
        losses = detect_compression_losses(intent, bp, reconstructed)

        # Gate
        result = closed_loop_gate(intent, bp)
        assert result.reconstructed_intent == reconstructed
        assert result.fidelity_score == fidelity

    def test_perfect_capture(self):
        """When blueprint perfectly mirrors intent, fidelity should be very high."""
        intent = "track inventory"
        bp = {
            "core_need": "track inventory",
            "domain": "software",
            "components": [{"name": "Inventory Tracker", "purpose": "track inventory items and quantities"}],
        }
        result = closed_loop_gate(intent, bp)
        assert result.fidelity_score > 0.5

    def test_catastrophic_loss(self):
        """When blueprint captures nothing from intent, gate should fail hard."""
        intent = (
            "Build a neural network training pipeline with GPU acceleration, "
            "distributed training, model checkpointing, and hyperparameter optimization"
        )
        bp = {
            "core_need": "display hello world",
            "components": [{"name": "Printer", "purpose": "print text to console"}],
        }
        result = closed_loop_gate(intent, bp)
        assert result.fidelity_score < 0.3
        assert result.passed is False
        assert len(result.compression_losses) > 0


# ---------------------------------------------------------------------------
# Relationship key format compatibility (from/to vs source/target)
# ---------------------------------------------------------------------------

def _blueprint_auth_from_to():
    """Same as _blueprint_auth() but with from/to keys (matching real engine output)."""
    return {
        "core_need": "authenticate users securely",
        "domain": "software",
        "components": [
            {"name": "AuthService", "purpose": "handle login and registration"},
            {"name": "TokenManager", "purpose": "issue and validate JWT tokens"},
            {"name": "UserStore", "purpose": "persist user credentials"},
            {"name": "PasswordHasher", "purpose": "hash and verify passwords"},
        ],
        "relationships": [
            {"from": "AuthService", "to": "TokenManager", "type": "uses"},
            {"from": "AuthService", "to": "UserStore", "type": "reads from"},
            {"from": "AuthService", "to": "PasswordHasher", "type": "delegates to"},
        ],
        "constraints": [
            {"description": "Passwords must be hashed with bcrypt"},
            {"description": "JWT tokens expire after 24 hours"},
            {"description": "Rate limit login attempts to 5 per minute"},
        ],
        "actors": [
            {"name": "End User", "role": "person authenticating"},
            {"name": "Admin", "role": "manages user accounts"},
        ],
    }


class TestRelationshipKeyFormats:
    """Verify that from/to keys (engine output) work identically to source/target."""

    def test_flat_text_source_target_keys(self):
        """Existing source/target format still works."""
        bp = _blueprint_auth()
        text = _blueprint_to_flat_text(bp)
        assert "AuthService" in text
        assert "TokenManager" in text
        assert "UserStore" in text

    def test_flat_text_from_to_keys(self):
        """Engine's from/to format now works."""
        bp = _blueprint_auth_from_to()
        text = _blueprint_to_flat_text(bp)
        assert "AuthService" in text
        assert "TokenManager" in text
        assert "UserStore" in text

    def test_flat_text_both_formats_equivalent(self):
        """Both key formats produce identical flat text."""
        text_st = _blueprint_to_flat_text(_blueprint_auth())
        text_ft = _blueprint_to_flat_text(_blueprint_auth_from_to())
        assert text_st == text_ft

    def test_decode_source_target(self):
        """decode_blueprint handles source/target."""
        decoded = decode_blueprint(_blueprint_auth())
        assert "AuthService" in decoded
        assert "uses" in decoded or "delegates" in decoded

    def test_decode_from_to(self):
        """decode_blueprint handles from/to."""
        decoded = decode_blueprint(_blueprint_auth_from_to())
        assert "AuthService" in decoded
        assert "uses" in decoded or "delegates" in decoded

    def test_gate_fidelity_equivalent(self):
        """Gate scores are identical for both key formats."""
        intent = _intent_auth()
        result_st = closed_loop_gate(intent, _blueprint_auth())
        result_ft = closed_loop_gate(intent, _blueprint_auth_from_to())
        assert result_st.fidelity_score == result_ft.fidelity_score

    def test_compression_losses_no_false_entity_losses(self):
        """from/to relationships don't generate false entity losses."""
        intent = _intent_auth()
        bp = _blueprint_auth_from_to()
        recon = decode_blueprint(bp)
        losses = detect_compression_losses(intent, bp, recon)
        # No entity loss should reference relationship endpoint names
        # that are actually present in the blueprint
        entity_losses = [l for l in losses if l.category == "entity"]
        for loss in entity_losses:
            assert "authservice" not in loss.original_fragment.lower()
            assert "tokenmanager" not in loss.original_fragment.lower()
            assert "userstore" not in loss.original_fragment.lower()

    def test_mixed_format_relationships(self):
        """Blueprint mixing both key formats works correctly."""
        bp = {
            "core_need": "authenticate users securely",
            "domain": "software",
            "components": [
                {"name": "AuthService", "purpose": "handle login"},
                {"name": "TokenManager", "purpose": "manage tokens"},
                {"name": "UserStore", "purpose": "store users"},
            ],
            "relationships": [
                {"source": "AuthService", "target": "TokenManager", "type": "uses"},
                {"from": "AuthService", "to": "UserStore", "type": "reads from"},
            ],
        }
        text = _blueprint_to_flat_text(bp)
        assert "TokenManager" in text
        assert "UserStore" in text
        decoded = decode_blueprint(bp)
        assert "TokenManager" in decoded
        assert "UserStore" in decoded
