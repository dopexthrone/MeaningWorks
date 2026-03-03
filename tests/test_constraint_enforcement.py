"""
Tests for Phase 20: Constraint Enforcement — End-to-End.

These tests generate code from blueprints with constraints,
execute the generated code, and verify that constraints are
actually enforced at runtime (ValueError raised on violation).
"""

import pytest
from codegen.generator import BlueprintCodeGenerator


def make_blueprint(components=None, relationships=None, constraints=None):
    """Helper to create a minimal blueprint."""
    return {
        "components": components or [],
        "relationships": relationships or [],
        "constraints": constraints or [],
    }


def generate_and_exec(bp):
    """Generate code from blueprint and execute it, returning the namespace."""
    gen = BlueprintCodeGenerator(bp)
    code = gen.generate()
    ns = {}
    exec(compile(code, "<test>", "exec"), ns)
    return ns


class TestConstraintEnforcementEndToEnd:
    """Execute generated code and verify constraint enforcement."""

    def test_range_constraint_enforced_at_construction(self):
        """Entity with range constraint rejects out-of-range values."""
        bp = make_blueprint(
            components=[{
                "name": "Score",
                "type": "entity",
                "description": "A numeric score",
            }],
            relationships=[
                {"from": "Score", "to": "Value", "type": "contains"},
            ],
        )
        bp["components"].append({
            "name": "Value",
            "type": "entity",
            "description": "float = 0.0 - The score value",
            "type_hint": "float",
            "default_value": "0.0",
        })
        bp["constraints"] = [{
            "description": "value in range [0, 100]",
            "applies_to": ["Score"],
        }]
        ns = generate_and_exec(bp)
        Score = ns["Score"]

        # Valid construction should work
        s = Score()
        assert s.validate() is True

        # Out-of-range should raise
        s.value = 101.0
        with pytest.raises(ValueError):
            s.validate()

    def test_enum_constraint_enforced(self):
        """Entity with enum constraint rejects invalid values."""
        bp = make_blueprint(
            components=[{
                "name": "Light",
                "type": "entity",
                "description": "A traffic light",
            }],
            relationships=[
                {"from": "Light", "to": "Color", "type": "contains"},
            ],
        )
        bp["components"].append({
            "name": "Color",
            "type": "entity",
            "description": "str = 'red' - The light color",
            "type_hint": "str",
            "default_value": "'red'",
        })
        bp["constraints"] = [{
            "description": "color must be one of: red, yellow, green",
            "applies_to": ["Light"],
        }]
        ns = generate_and_exec(bp)
        Light = ns["Light"]

        # Valid construction
        light = Light()
        assert light.validate() is True

        # Invalid value
        light.color = "purple"
        with pytest.raises(ValueError):
            light.validate()

    def test_not_null_constraint_enforced(self):
        """Entity with not_null constraint rejects None."""
        bp = make_blueprint(
            components=[{
                "name": "User",
                "type": "entity",
                "description": "A user",
            }],
            relationships=[
                {"from": "User", "to": "Username", "type": "contains"},
            ],
        )
        bp["components"].append({
            "name": "Username",
            "type": "entity",
            "description": "str = '' - The username",
            "type_hint": "str",
            "default_value": "''",
        })
        bp["constraints"] = [{
            "description": "username must not be null",
            "applies_to": ["User"],
        }]
        ns = generate_and_exec(bp)
        User = ns["User"]

        # Valid construction (empty string is not None)
        user = User()
        assert user.validate() is True

        # None should raise
        user.username = None
        with pytest.raises(ValueError):
            user.validate()

    def test_length_constraint_enforced(self):
        """Entity with length constraint rejects too-long values."""
        bp = make_blueprint(
            components=[{
                "name": "Tag",
                "type": "entity",
                "description": "A tag",
            }],
            relationships=[
                {"from": "Tag", "to": "Label", "type": "contains"},
            ],
        )
        bp["components"].append({
            "name": "Label",
            "type": "entity",
            "description": "str = 'ok' - The label text",
            "type_hint": "str",
            "default_value": "'ok'",
        })
        bp["constraints"] = [{
            "description": "label length between 1 and 50",
            "applies_to": ["Tag"],
        }]
        ns = generate_and_exec(bp)
        Tag = ns["Tag"]

        # Valid
        tag = Tag()
        assert tag.validate() is True

        # Too long should raise
        tag.label = "x" * 51
        with pytest.raises(ValueError):
            tag.validate()

    def test_positive_constraint_enforced(self):
        """Entity with positive constraint rejects zero and negative."""
        bp = make_blueprint(
            components=[{
                "name": "Invoice",
                "type": "entity",
                "description": "An invoice",
            }],
            relationships=[
                {"from": "Invoice", "to": "Amount", "type": "contains"},
            ],
        )
        bp["components"].append({
            "name": "Amount",
            "type": "entity",
            "description": "float = 1.0 - The invoice amount",
            "type_hint": "float",
            "default_value": "1.0",
        })
        bp["constraints"] = [{
            "description": "amount must be positive",
            "applies_to": ["Invoice"],
        }]
        ns = generate_and_exec(bp)
        Invoice = ns["Invoice"]

        # Valid
        inv = Invoice()
        assert inv.validate() is True

        # Zero should raise
        inv.amount = 0
        with pytest.raises(ValueError):
            inv.validate()

        # Negative should raise
        inv.amount = -5
        with pytest.raises(ValueError):
            inv.validate()

    def test_non_negative_constraint_enforced(self):
        """Entity with non_negative constraint rejects negative."""
        bp = make_blueprint(
            components=[{
                "name": "Counter",
                "type": "entity",
                "description": "A counter",
            }],
            relationships=[
                {"from": "Counter", "to": "Count", "type": "contains"},
            ],
        )
        bp["components"].append({
            "name": "Count",
            "type": "entity",
            "description": "int = 0 - The count value",
            "type_hint": "int",
            "default_value": "0",
        })
        bp["constraints"] = [{
            "description": "count >= 0",
            "applies_to": ["Counter"],
        }]
        ns = generate_and_exec(bp)
        Counter = ns["Counter"]

        # Valid (zero is ok)
        c = Counter()
        assert c.validate() is True

        # Negative should raise
        c.count = -1
        with pytest.raises(ValueError):
            c.validate()

    def test_multiple_constraints_all_enforced(self):
        """Entity with multiple constraints checks all."""
        bp = make_blueprint(
            components=[{
                "name": "Product",
                "type": "entity",
                "description": "A product",
            }],
            relationships=[
                {"from": "Product", "to": "Price", "type": "contains"},
                {"from": "Product", "to": "Qty", "type": "contains"},
            ],
        )
        bp["components"].extend([
            {
                "name": "Price",
                "type": "entity",
                "description": "float = 1.0 - The product price",
                "type_hint": "float",
                "default_value": "1.0",
            },
            {
                "name": "Qty",
                "type": "entity",
                "description": "int = 1 - Quantity in stock",
                "type_hint": "int",
                "default_value": "1",
            },
        ])
        bp["constraints"] = [
            {"description": "price in range [0, 10000]", "applies_to": ["Product"]},
            {"description": "qty >= 0", "applies_to": ["Product"]},
        ]
        ns = generate_and_exec(bp)
        Product = ns["Product"]

        # Valid
        p = Product()
        assert p.validate() is True

        # Violate price
        p.price = 20000.0
        with pytest.raises(ValueError):
            p.validate()

        # Fix price, violate qty
        p.price = 5.0
        p.qty = -1
        with pytest.raises(ValueError):
            p.validate()

    def test_valid_values_pass_construction(self):
        """Entity with constraints accepts valid default values."""
        bp = make_blueprint(
            components=[{
                "name": "Rating",
                "type": "entity",
                "description": "A rating",
            }],
            relationships=[
                {"from": "Rating", "to": "Stars", "type": "contains"},
            ],
        )
        bp["components"].append({
            "name": "Stars",
            "type": "entity",
            "description": "int = 3 - Star rating",
            "type_hint": "int",
            "default_value": "3",
        })
        bp["constraints"] = [{
            "description": "stars in range [1, 5]",
            "applies_to": ["Rating"],
        }]
        ns = generate_and_exec(bp)
        Rating = ns["Rating"]

        # Default value (3) is in range [1, 5] — construction should succeed
        r = Rating()
        assert r.stars == 3

    def test_validate_method_still_returns_true(self):
        """validate() returns True when all constraints pass."""
        bp = make_blueprint(
            components=[{
                "name": "Config",
                "type": "entity",
                "description": "Configuration",
            }],
            relationships=[
                {"from": "Config", "to": "Timeout", "type": "contains"},
            ],
        )
        bp["components"].append({
            "name": "Timeout",
            "type": "entity",
            "description": "int = 30 - Timeout in seconds",
            "type_hint": "int",
            "default_value": "30",
        })
        bp["constraints"] = [{
            "description": "timeout in range [1, 3600]",
            "applies_to": ["Config"],
        }]
        ns = generate_and_exec(bp)
        Config = ns["Config"]

        c = Config()
        result = c.validate()
        assert result is True

    def test_error_message_includes_constraint_description(self):
        """ValueError message describes the violated constraint."""
        bp = make_blueprint(
            components=[{
                "name": "Temp",
                "type": "entity",
                "description": "Temperature",
            }],
            relationships=[
                {"from": "Temp", "to": "Celsius", "type": "contains"},
            ],
        )
        bp["components"].append({
            "name": "Celsius",
            "type": "entity",
            "description": "float = 20.0 - Temperature in celsius",
            "type_hint": "float",
            "default_value": "20.0",
        })
        bp["constraints"] = [{
            "description": "celsius in range [0, 1000]",
            "applies_to": ["Temp"],
        }]
        ns = generate_and_exec(bp)
        Temp = ns["Temp"]

        t = Temp()
        t.celsius = -5.0
        with pytest.raises(ValueError, match="celsius"):
            t.validate()

    def test_post_init_enforcement_blocks_invalid_construction(self):
        """__post_init__ should prevent construction with invalid defaults."""
        bp = make_blueprint(
            components=[{
                "name": "Bounded",
                "type": "entity",
                "description": "A bounded value",
            }],
            relationships=[
                {"from": "Bounded", "to": "Val", "type": "contains"},
            ],
        )
        bp["components"].append({
            "name": "Val",
            "type": "entity",
            "description": "float = -1.0 - A value that defaults to invalid",
            "type_hint": "float",
            "default_value": "-1.0",
        })
        bp["constraints"] = [{
            "description": "val in range [0, 100]",
            "applies_to": ["Bounded"],
        }]
        ns = generate_and_exec(bp)
        Bounded = ns["Bounded"]

        # Construction should fail because default -1.0 violates [0, 100]
        with pytest.raises(ValueError):
            Bounded()
