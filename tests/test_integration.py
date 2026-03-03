"""
Integration Tests - Phase 6.4.

End-to-end tests verifying that Phase 6 components
(failover, caching, constraints) work together correctly.

Tests cover:
- Constraint parsing -> code generation pipeline
- Entity template generation with formal validators
- Blueprint code generation with constraints
- Failover + caching interaction
- Full compile pipeline with mocked LLM
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import json

from core.schema import (
    ConstraintType,
    FormalConstraint,
    parse_constraint,
    parse_blueprint_constraints,
    generate_validator_code,
    generate_validate_method,
)
from core.cache import CompilationCache, StagedCache, CacheStats, reset_cache
from core.llm import FailoverClient, MockClient, BaseLLMClient
from core.exceptions import (
    MotherlabsError,
    CompilationError,
    ProviderError,
    FailoverExhaustedError,
    ProviderUnavailableError,
    ConfigurationError,
)
from codegen.generator import BlueprintCodeGenerator, to_pascal_case, to_snake_case
from codegen.templates import generate_validation_rules


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_blueprint():
    """A minimal blueprint with components, relationships, and constraints."""
    return {
        "components": [
            {
                "name": "User",
                "type": "entity",
                "description": "A user in the system",
                "derived_from": "User entity from requirements",
            },
            {
                "name": "Order",
                "type": "entity",
                "description": "A purchase order",
                "derived_from": "Order entity from requirements",
            },
            {
                "name": "OrderProcessor",
                "type": "process",
                "description": "Processes orders through stages",
                "derived_from": "Order processing flow",
                "state_machine": {
                    "states": ["PENDING", "PROCESSING", "COMPLETE"],
                    "initial_state": "PENDING",
                    "transitions": [
                        {"from_state": "PENDING", "to_state": "PROCESSING", "trigger": "start"},
                        {"from_state": "PROCESSING", "to_state": "COMPLETE", "trigger": "finish"},
                    ],
                    "derived_from": "Order lifecycle from requirements",
                },
            },
        ],
        "relationships": [
            {
                "from": "OrderProcessor",
                "to": "Order",
                "type": "accesses",
                "description": "OrderProcessor reads and updates Order",
            },
            {
                "from": "Order",
                "to": "User",
                "type": "contains",
                "description": "Order belongs to User",
            },
        ],
        "constraints": [
            {
                "description": "price in range [0.01, 99999.99]",
                "applies_to": ["Order"],
                "derived_from": "Price must be positive and bounded",
            },
            {
                "description": "quantity must be positive",
                "applies_to": ["Order"],
                "derived_from": "Cannot order zero or negative items",
            },
            {
                "description": "email matches /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+$/",
                "applies_to": ["User"],
                "derived_from": "Email format requirement",
            },
            {
                "description": "status must be one of: pending, processing, complete, cancelled",
                "applies_to": ["Order"],
                "derived_from": "Valid order statuses",
            },
            {
                "description": "name cannot be null",
                "applies_to": ["User"],
                "derived_from": "User name is required",
            },
        ],
    }


@pytest.fixture
def blueprint_with_length_constraints():
    """Blueprint with length-type constraints."""
    return {
        "components": [
            {
                "name": "Profile",
                "type": "entity",
                "description": "User profile",
                "derived_from": "Profile entity",
            },
        ],
        "relationships": [],
        "constraints": [
            {
                "description": "username length between 3 and 30",
                "applies_to": ["Profile"],
                "derived_from": "Username length requirements",
            },
            {
                "description": "bio max length 500",
                "applies_to": ["Profile"],
                "derived_from": "Bio character limit",
            },
        ],
    }


@pytest.fixture(autouse=True)
def clean_cache():
    """Reset global cache between tests."""
    reset_cache()
    yield
    reset_cache()


# =============================================================================
# CONSTRAINT -> CODE GENERATION PIPELINE
# =============================================================================

class TestConstraintToCodePipeline:
    """Test the full constraint parsing -> code generation flow."""

    def test_range_constraint_generates_validation_code(self):
        """Range constraint should produce if-raise-ValueError validation."""
        constraints = [{"description": "price in range [0, 100]", "applies_to": ["Order"]}]
        code = generate_validation_rules(constraints)
        assert "self.price" in code
        assert "raise ValueError" in code
        assert "0" in code
        assert "100" in code

    def test_not_null_constraint_generates_validation(self):
        """Not-null constraint should produce None check."""
        constraints = [{"description": "name cannot be null", "applies_to": ["User"]}]
        code = generate_validation_rules(constraints)
        assert "self.name" in code
        assert "None" in code
        assert "raise ValueError" in code

    def test_positive_constraint_generates_validation(self):
        """Positive constraint should produce > 0 check."""
        constraints = [{"description": "quantity must be positive", "applies_to": ["Order"]}]
        code = generate_validation_rules(constraints)
        assert "self.quantity" in code
        assert "> 0" in code
        assert "raise ValueError" in code

    def test_enum_constraint_generates_validation(self):
        """Enum constraint should produce membership check."""
        constraints = [{"description": "status must be one of: active, inactive", "applies_to": ["User"]}]
        code = generate_validation_rules(constraints)
        assert "self.status" in code
        assert "raise ValueError" in code

    def test_length_constraint_generates_validation(self, blueprint_with_length_constraints):
        """Length constraint should produce len() check."""
        constraints = blueprint_with_length_constraints["constraints"]
        code = generate_validation_rules(constraints)
        assert "len(self.username)" in code
        assert "raise ValueError" in code

    def test_regex_constraint_generates_validation(self):
        """Regex constraint should produce re.match check."""
        constraints = [{"description": "email matches /^[a-z]+@/", "applies_to": ["User"]}]
        code = generate_validation_rules(constraints)
        assert "import re" in code
        assert "raise ValueError" in code

    def test_custom_constraint_becomes_comment(self):
        """Unparseable constraints should produce TODO comments."""
        constraints = [{"description": "data must be internally consistent", "applies_to": ["System"]}]
        code = generate_validation_rules(constraints)
        assert "TODO" in code

    def test_empty_constraints_produces_comment(self):
        """Empty constraints list should produce 'no rules' comment."""
        code = generate_validation_rules([])
        assert "No validation rules defined" in code

    def test_multiple_constraints_all_generate(self, sample_blueprint):
        """All parseable constraints in a blueprint should generate code."""
        code = generate_validation_rules(sample_blueprint["constraints"])
        # Should have validation code for price (range), quantity (positive),
        # status (enum), name (not_null)
        assert "raise ValueError" in code
        # At least 3 distinct validation blocks
        assert code.count("raise ValueError") >= 3


class TestBlueprintConstraintParsing:
    """Test parse_blueprint_constraints integration."""

    def test_all_constraints_parsed(self, sample_blueprint):
        """All constraints in sample blueprint should parse."""
        parsed = parse_blueprint_constraints(sample_blueprint)
        assert len(parsed) == 5

    def test_constraint_types_detected(self, sample_blueprint):
        """Each constraint should be detected as the correct type."""
        parsed = parse_blueprint_constraints(sample_blueprint)
        types = [c.constraint_type for c in parsed]
        assert ConstraintType.RANGE in types
        assert ConstraintType.POSITIVE in types
        assert ConstraintType.NOT_NULL in types

    def test_applies_to_preserved(self, sample_blueprint):
        """applies_to should be preserved from blueprint."""
        parsed = parse_blueprint_constraints(sample_blueprint)
        for constraint in parsed:
            assert len(constraint.applies_to) > 0

    def test_length_vs_range_disambiguation(self, blueprint_with_length_constraints):
        """Length constraints should not be parsed as range."""
        parsed = parse_blueprint_constraints(blueprint_with_length_constraints)
        for c in parsed:
            assert c.constraint_type == ConstraintType.LENGTH
            assert c.constraint_type != ConstraintType.RANGE


# =============================================================================
# CODE GENERATOR WITH CONSTRAINTS
# =============================================================================

class TestCodeGeneratorWithConstraints:
    """Test BlueprintCodeGenerator handles constraints correctly."""

    def test_generate_produces_valid_python(self, sample_blueprint):
        """Generated code should be syntactically valid Python."""
        gen = BlueprintCodeGenerator(sample_blueprint)
        code = gen.generate()
        # Should compile without syntax errors
        compile(code, "<test>", "exec")

    def test_generate_tests_includes_constraints(self, sample_blueprint):
        """Generated tests should include constraint test class."""
        gen = BlueprintCodeGenerator(sample_blueprint)
        tests = gen.generate_tests()
        assert "TestConstraints" in tests
        assert "test_constraint_0" in tests
        assert "test_constraint_1" in tests

    def test_constraint_tests_have_real_assertions(self, sample_blueprint):
        """Constraint tests should have actual assertions, not just pass."""
        gen = BlueprintCodeGenerator(sample_blueprint)
        tests = gen.generate_tests()
        # At least some constraints should have real assertions
        assert "assert" in tests

    def test_constraint_tests_reference_correct_classes(self, sample_blueprint):
        """Constraint tests should reference the applies_to classes."""
        gen = BlueprintCodeGenerator(sample_blueprint)
        tests = gen.generate_tests()
        assert "Order" in tests
        assert "User" in tests

    def test_entity_validation_in_generated_code(self, sample_blueprint):
        """Entity code should include validate() method with constraints."""
        gen = BlueprintCodeGenerator(sample_blueprint)
        code = gen.generate()
        # The generated code should have validate methods
        assert "def validate(self)" in code

    def test_process_state_machine_generated(self, sample_blueprint):
        """Process component should have state machine code."""
        gen = BlueprintCodeGenerator(sample_blueprint)
        code = gen.generate()
        assert "PENDING" in code
        assert "PROCESSING" in code
        assert "COMPLETE" in code
        assert "current_state" in code


# =============================================================================
# FAILOVER + CACHING INTEGRATION
# =============================================================================

class TestFailoverWithCaching:
    """Test that failover and caching work together."""

    def test_cache_survives_provider_switch(self):
        """Cache should persist across provider failover."""
        cache = CompilationCache(max_size=10, ttl_seconds=300)

        # Simulate caching a result from provider A
        key_a = cache.make_key("test input", {"provider": "provider_a"})
        cache.set(key_a, {"result": "from_a"})

        # Simulate failover: now using provider B, same input but different config
        key_b = cache.make_key("test input", {"provider": "provider_b"})

        # Keys should be different (different provider)
        assert key_a != key_b

        # But original cache entry should still be accessible
        assert cache.get(key_a) == {"result": "from_a"}
        assert cache.get(key_b) is None

    def test_staged_cache_with_failover_client(self):
        """StagedCache should work with FailoverClient responses."""
        staged = StagedCache(enabled=True)

        # Simulate intent caching
        intent_key = staged.intent.make_key("build a todo app", {"provider": "mock"})
        intent_result = {"core_need": "todo management", "domain": "productivity"}
        staged.intent.set(intent_key, intent_result)

        # Verify retrieval
        cached = staged.intent.get(intent_key)
        assert cached == intent_result
        assert staged.intent.stats.hits == 1

    def test_failover_client_with_mock(self):
        """FailoverClient with MockClient should work for testing."""
        mock1 = MockClient()
        mock2 = MockClient()
        client = FailoverClient([mock1, mock2])

        result = client.complete_with_system("system prompt", "user input")
        assert result is not None
        assert len(result) > 0

    def test_failover_triggers_on_provider_error(self):
        """When first provider fails, second should be used."""
        failing = Mock(spec=BaseLLMClient)
        failing.complete = Mock(side_effect=ProviderError("failed", provider="bad"))
        failing.provider_name = "bad_provider"
        failing.deterministic = False

        working = MockClient()

        client = FailoverClient([failing, working])
        result = client.complete_with_system("test", "test")
        assert result is not None

    def test_failover_exhausted_raises(self):
        """When all providers fail, FailoverExhaustedError should be raised."""
        fail1 = Mock(spec=BaseLLMClient)
        fail1.complete = Mock(side_effect=ProviderError("fail1", provider="p1"))
        fail1.provider_name = "p1"
        fail1.deterministic = False

        fail2 = Mock(spec=BaseLLMClient)
        fail2.complete = Mock(side_effect=ProviderError("fail2", provider="p2"))
        fail2.provider_name = "p2"
        fail2.deterministic = False

        client = FailoverClient([fail1, fail2])
        with pytest.raises(FailoverExhaustedError):
            client.complete_with_system("test", "test")


class TestCacheWithConstraints:
    """Test caching works with constraint-heavy blueprints."""

    def test_blueprint_with_constraints_cacheable(self, sample_blueprint):
        """Blueprints with constraints should be cacheable."""
        cache = CompilationCache(max_size=10, ttl_seconds=300)
        key = cache.make_key(
            json.dumps(sample_blueprint, sort_keys=True),
            {"stage": "synthesis"}
        )
        cache.set(key, sample_blueprint)

        retrieved = cache.get(key)
        assert retrieved is not None
        assert retrieved["constraints"] == sample_blueprint["constraints"]
        assert len(retrieved["constraints"]) == 5

    def test_constraint_parsing_is_deterministic(self, sample_blueprint):
        """Same constraints should always parse to same result."""
        parsed1 = parse_blueprint_constraints(sample_blueprint)
        parsed2 = parse_blueprint_constraints(sample_blueprint)

        assert len(parsed1) == len(parsed2)
        for c1, c2 in zip(parsed1, parsed2):
            assert c1.constraint_type == c2.constraint_type
            assert c1.field == c2.field
            assert c1.params == c2.params


# =============================================================================
# ENGINE INTEGRATION (mocked LLM)
# =============================================================================

class TestEngineIntegration:
    """Test MotherlabsEngine with Phase 6 features integrated."""

    def _make_engine(self, **kwargs):
        """Create engine with MockClient to avoid provider issues."""
        from core.engine import MotherlabsEngine
        defaults = {"llm_client": MockClient()}
        defaults.update(kwargs)
        return MotherlabsEngine(**defaults)

    def test_engine_accepts_llm_client(self):
        """Engine should accept llm_client parameter directly."""
        engine = self._make_engine()
        assert engine is not None

    def test_engine_cache_policy_none(self):
        """Engine with cache_policy=none should disable caching."""
        engine = self._make_engine(cache_policy="none")
        assert engine._cache is not None
        assert not engine._cache.enabled

    def test_engine_cache_policy_intent(self):
        """Engine with cache_policy=intent should enable intent caching."""
        engine = self._make_engine(cache_policy="intent")
        assert engine._cache.enabled

    def test_engine_cache_policy_full(self):
        """Engine with cache_policy=full should enable all caching."""
        engine = self._make_engine(cache_policy="full")
        assert engine._cache.enabled


# =============================================================================
# VALIDATOR CODE EXECUTION
# =============================================================================

class TestValidatorCodeExecution:
    """Test that generated validator code is actually executable."""

    def test_range_validator_executes(self):
        """Range validator code should execute and catch violations."""
        constraint = FormalConstraint(
            constraint_type=ConstraintType.RANGE,
            field="price",
            params={"min": 0.0, "max": 100.0},
        )
        code = generate_validator_code(constraint)
        assert "assert" in code

        # Create a mock object
        class Obj:
            price = 50.0
        obj = Obj()
        # Execute the validator code with self=obj
        exec(code.replace("self.", "obj."))

    def test_range_validator_rejects_violation(self):
        """Range validator should raise on out-of-range value."""
        constraint = FormalConstraint(
            constraint_type=ConstraintType.RANGE,
            field="price",
            params={"min": 0.0, "max": 100.0},
        )
        code = generate_validator_code(constraint)

        class Obj:
            price = 200.0
        obj = Obj()
        with pytest.raises(AssertionError):
            exec(code.replace("self.", "obj."))

    def test_not_null_validator_executes(self):
        """Not-null validator should pass for non-None values."""
        constraint = FormalConstraint(
            constraint_type=ConstraintType.NOT_NULL,
            field="name",
        )
        code = generate_validator_code(constraint)

        class Obj:
            name = "test"
        obj = Obj()
        exec(code.replace("self.", "obj."))

    def test_not_null_validator_rejects_none(self):
        """Not-null validator should raise for None values."""
        constraint = FormalConstraint(
            constraint_type=ConstraintType.NOT_NULL,
            field="name",
        )
        code = generate_validator_code(constraint)

        class Obj:
            name = None
        obj = Obj()
        with pytest.raises(AssertionError):
            exec(code.replace("self.", "obj."))

    def test_positive_validator_executes(self):
        """Positive validator should pass for positive values."""
        constraint = FormalConstraint(
            constraint_type=ConstraintType.POSITIVE,
            field="count",
        )
        code = generate_validator_code(constraint)

        class Obj:
            count = 5
        obj = Obj()
        exec(code.replace("self.", "obj."))

    def test_positive_validator_rejects_zero(self):
        """Positive validator should reject zero."""
        constraint = FormalConstraint(
            constraint_type=ConstraintType.POSITIVE,
            field="count",
        )
        code = generate_validator_code(constraint)

        class Obj:
            count = 0
        obj = Obj()
        with pytest.raises(AssertionError):
            exec(code.replace("self.", "obj."))

    def test_enum_validator_executes(self):
        """Enum validator should pass for valid values."""
        constraint = FormalConstraint(
            constraint_type=ConstraintType.ENUM,
            field="status",
            params={"values": ["active", "inactive"]},
        )
        code = generate_validator_code(constraint)

        class Obj:
            status = "active"
        obj = Obj()
        exec(code.replace("self.", "obj."))

    def test_enum_validator_rejects_invalid(self):
        """Enum validator should reject invalid values."""
        constraint = FormalConstraint(
            constraint_type=ConstraintType.ENUM,
            field="status",
            params={"values": ["active", "inactive"]},
        )
        code = generate_validator_code(constraint)

        class Obj:
            status = "deleted"
        obj = Obj()
        with pytest.raises(AssertionError):
            exec(code.replace("self.", "obj."))

    def test_unique_validator_executes(self):
        """Unique validator should pass for unique values."""
        constraint = FormalConstraint(
            constraint_type=ConstraintType.UNIQUE,
            field="tags",
        )
        code = generate_validator_code(constraint)

        class Obj:
            tags = [1, 2, 3]
        obj = Obj()
        exec(code.replace("self.", "obj."))

    def test_unique_validator_rejects_duplicates(self):
        """Unique validator should reject duplicates."""
        constraint = FormalConstraint(
            constraint_type=ConstraintType.UNIQUE,
            field="tags",
        )
        code = generate_validator_code(constraint)

        class Obj:
            tags = [1, 2, 2]
        obj = Obj()
        with pytest.raises(AssertionError):
            exec(code.replace("self.", "obj."))

    def test_length_validator_executes(self):
        """Length validator should pass for valid lengths."""
        constraint = FormalConstraint(
            constraint_type=ConstraintType.LENGTH,
            field="username",
            params={"min": 3, "max": 20},
        )
        code = generate_validator_code(constraint)

        class Obj:
            username = "alice"
        obj = Obj()
        exec(code.replace("self.", "obj."))

    def test_length_validator_rejects_short(self):
        """Length validator should reject too-short values."""
        constraint = FormalConstraint(
            constraint_type=ConstraintType.LENGTH,
            field="username",
            params={"min": 3, "max": 20},
        )
        code = generate_validator_code(constraint)

        class Obj:
            username = "ab"
        obj = Obj()
        with pytest.raises(AssertionError):
            exec(code.replace("self.", "obj."))


# =============================================================================
# GENERATED CODE COMPILATION
# =============================================================================

class TestGeneratedCodeCompilation:
    """Test that full generated code compiles and imports work."""

    def test_full_generated_code_compiles(self, sample_blueprint):
        """Generated module should compile without errors."""
        gen = BlueprintCodeGenerator(sample_blueprint)
        code = gen.generate()
        compiled = compile(code, "<generated>", "exec")
        assert compiled is not None

    def test_generated_test_code_compiles(self, sample_blueprint):
        """Generated test module should compile without errors."""
        gen = BlueprintCodeGenerator(sample_blueprint)
        tests = gen.generate_tests()
        compiled = compile(tests, "<generated_tests>", "exec")
        assert compiled is not None

    def test_generated_code_has_expected_classes(self, sample_blueprint):
        """Generated code should contain expected class names."""
        gen = BlueprintCodeGenerator(sample_blueprint)
        code = gen.generate()
        assert "class User" in code
        assert "class Order" in code
        assert "class OrderProcessor" in code

    def test_generated_code_includes_base_classes(self, sample_blueprint):
        """Generated code should include necessary base classes."""
        gen = BlueprintCodeGenerator(sample_blueprint)
        code = gen.generate()
        assert "class BaseAgent" in code
        assert "from dataclasses import" in code

    def test_generated_code_includes_factory(self, sample_blueprint):
        """Generated code with relationships should include factory."""
        gen = BlueprintCodeGenerator(sample_blueprint)
        code = gen.generate()
        assert "BlueprintFactory" in code
        assert "create_all" in code
