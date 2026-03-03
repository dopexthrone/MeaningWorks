"""
Tests for core/blueprint_editor.py — blueprint edit operations.

Phase 16: Human-in-the-Loop Iteration
~25 tests — rename, remove, merge, add_constraint, flag_hollow,
add_component, apply_edits orchestrator, lineage tracking.
"""

import pytest

from core.blueprint_editor import (
    rename_component,
    remove_component,
    merge_components,
    add_constraint,
    flag_hollow,
    add_component,
    apply_edits,
    EditResult,
    EditOperation,
)


# =============================================================================
# Helpers
# =============================================================================

def _blueprint(*comp_names, relationships=None, constraints=None):
    """Build a minimal blueprint dict."""
    components = [
        {"name": n, "type": "entity", "description": f"The {n} component",
         "derived_from": "test", "methods": []}
        for n in comp_names
    ]
    return {
        "components": components,
        "relationships": relationships or [],
        "constraints": constraints or [],
        "unresolved": [],
    }


# =============================================================================
# Rename Tests
# =============================================================================

class TestRenameComponent:
    def test_rename_updates_name(self):
        bp = _blueprint("Auth", "Database")
        result, warnings = rename_component(bp, "Auth", "AuthService")
        names = [c["name"] for c in result["components"]]
        assert "AuthService" in names
        assert "Auth" not in names
        assert warnings == []

    def test_rename_updates_relationships(self):
        bp = _blueprint("Auth", "DB", relationships=[
            {"from": "Auth", "to": "DB", "type": "accesses", "description": "test"},
        ])
        result, _ = rename_component(bp, "Auth", "AuthService")
        assert result["relationships"][0]["from"] == "AuthService"

    def test_rename_updates_constraints(self):
        bp = _blueprint("Auth", constraints=[
            {"description": "Auth must use JWT", "applies_to": ["Auth"]},
        ])
        result, _ = rename_component(bp, "Auth", "AuthService")
        assert result["constraints"][0]["applies_to"] == ["AuthService"]

    def test_rename_not_found_warns(self):
        bp = _blueprint("Auth")
        result, warnings = rename_component(bp, "Missing", "NewName")
        assert len(warnings) == 1
        assert "not found" in warnings[0]

    def test_rename_case_insensitive(self):
        bp = _blueprint("Auth")
        result, warnings = rename_component(bp, "auth", "AuthService")
        names = [c["name"] for c in result["components"]]
        assert "AuthService" in names
        assert warnings == []

    def test_rename_does_not_mutate_original(self):
        bp = _blueprint("Auth")
        result, _ = rename_component(bp, "Auth", "AuthService")
        assert bp["components"][0]["name"] == "Auth"


# =============================================================================
# Remove Tests
# =============================================================================

class TestRemoveComponent:
    def test_remove_component(self):
        bp = _blueprint("Auth", "Database")
        result, warnings = remove_component(bp, "Auth")
        names = [c["name"] for c in result["components"]]
        assert "Auth" not in names
        assert "Database" in names
        assert warnings == []

    def test_remove_cleans_relationships(self):
        bp = _blueprint("Auth", "DB", relationships=[
            {"from": "Auth", "to": "DB", "type": "accesses"},
            {"from": "DB", "to": "Cache", "type": "flows_to"},
        ])
        result, _ = remove_component(bp, "Auth")
        assert len(result["relationships"]) == 1
        assert result["relationships"][0]["from"] == "DB"

    def test_remove_not_found_warns(self):
        bp = _blueprint("Auth")
        _, warnings = remove_component(bp, "Missing")
        assert len(warnings) == 1

    def test_remove_cleans_constraint_applies_to(self):
        bp = _blueprint("Auth", "DB", constraints=[
            {"description": "test", "applies_to": ["Auth", "DB"]},
        ])
        result, _ = remove_component(bp, "Auth")
        assert result["constraints"][0]["applies_to"] == ["DB"]


# =============================================================================
# Merge Tests
# =============================================================================

class TestMergeComponents:
    def test_merge_two_components(self):
        bp = _blueprint("BookingManager", "ReservationHandler")
        result, warnings = merge_components(bp, ["BookingManager", "ReservationHandler"], "BookingService")
        names = [c["name"] for c in result["components"]]
        assert "BookingService" in names
        assert "BookingManager" not in names
        assert "ReservationHandler" not in names
        assert warnings == []

    def test_merge_combines_methods(self):
        bp = _blueprint("A", "B")
        bp["components"][0]["methods"] = [{"name": "create"}]
        bp["components"][1]["methods"] = [{"name": "delete"}]
        result, _ = merge_components(bp, ["A", "B"], "AB")
        merged = next(c for c in result["components"] if c["name"] == "AB")
        method_names = [m["name"] for m in merged["methods"]]
        assert "create" in method_names
        assert "delete" in method_names

    def test_merge_updates_relationships(self):
        bp = _blueprint("A", "B", "C", relationships=[
            {"from": "A", "to": "C", "type": "triggers"},
            {"from": "B", "to": "C", "type": "accesses"},
        ])
        result, _ = merge_components(bp, ["A", "B"], "AB")
        for rel in result["relationships"]:
            assert rel["from"] == "AB"

    def test_merge_removes_self_referential(self):
        bp = _blueprint("A", "B", relationships=[
            {"from": "A", "to": "B", "type": "triggers"},
        ])
        result, _ = merge_components(bp, ["A", "B"], "AB")
        # A→B becomes AB→AB → removed
        assert len(result["relationships"]) == 0

    def test_merge_fewer_than_2_warns(self):
        bp = _blueprint("A")
        _, warnings = merge_components(bp, ["A", "Missing"], "AM")
        assert len(warnings) == 1

    def test_merge_keeps_richest_description(self):
        bp = _blueprint("A", "B")
        bp["components"][0]["description"] = "Short"
        bp["components"][1]["description"] = "This is a much longer and more detailed description"
        result, _ = merge_components(bp, ["A", "B"], "AB")
        merged = next(c for c in result["components"] if c["name"] == "AB")
        assert "much longer" in merged["description"]


# =============================================================================
# Add Constraint Tests
# =============================================================================

class TestAddConstraint:
    def test_add_constraint(self):
        bp = _blueprint("Auth")
        result, warnings = add_constraint(bp, "Must use HTTPS", ["Auth"])
        assert len(result["constraints"]) == 1
        assert result["constraints"][0]["description"] == "Must use HTTPS"
        assert warnings == []

    def test_add_constraint_orphaned_warns(self):
        bp = _blueprint("Auth")
        _, warnings = add_constraint(bp, "test", ["Missing"])
        assert any("not found" in w for w in warnings)


# =============================================================================
# Flag Hollow Tests
# =============================================================================

class TestFlagHollow:
    def test_flag_sets_hollow(self):
        bp = _blueprint("PlaceholderService")
        result, warnings = flag_hollow(bp, "PlaceholderService")
        comp = result["components"][0]
        assert comp.get("hollow") is True
        assert any("HOLLOW" in u for u in result["unresolved"])
        assert warnings == []

    def test_flag_not_found_warns(self):
        bp = _blueprint("Auth")
        _, warnings = flag_hollow(bp, "Missing")
        assert len(warnings) == 1


# =============================================================================
# Add Component Tests
# =============================================================================

class TestAddComponent:
    def test_add_component(self):
        bp = _blueprint("Auth")
        result, warnings = add_component(bp, "Cache", "entity", "Caching layer")
        names = [c["name"] for c in result["components"]]
        assert "Cache" in names
        assert warnings == []

    def test_add_duplicate_warns(self):
        bp = _blueprint("Auth")
        _, warnings = add_component(bp, "Auth")
        assert len(warnings) == 1
        assert "already exists" in warnings[0]


# =============================================================================
# Apply Edits Orchestrator Tests
# =============================================================================

class TestApplyEdits:
    def test_single_edit(self):
        bp = _blueprint("Auth", "Database")
        result = apply_edits(bp, [
            {"operation": "rename", "old_name": "Auth", "new_name": "AuthService"},
        ])
        assert isinstance(result, EditResult)
        assert len(result.operations_applied) == 1
        names = [c["name"] for c in result.blueprint["components"]]
        assert "AuthService" in names

    def test_multiple_edits_sequential(self):
        bp = _blueprint("Auth", "Database", "Cache")
        result = apply_edits(bp, [
            {"operation": "rename", "old_name": "Auth", "new_name": "AuthService"},
            {"operation": "remove", "name": "Cache"},
        ])
        assert len(result.operations_applied) == 2
        assert result.components_before == 3
        assert result.components_after == 2

    def test_unknown_operation_warns(self):
        bp = _blueprint("Auth")
        result = apply_edits(bp, [
            {"operation": "explode"},
        ])
        assert len(result.warnings) == 1
        assert "Unknown" in result.warnings[0]

    def test_add_and_constrain(self):
        bp = _blueprint("Auth")
        result = apply_edits(bp, [
            {"operation": "add_component", "name": "Cache", "type": "entity", "description": "Cache layer"},
            {"operation": "add_constraint", "description": "Cache TTL max 1 hour", "applies_to": ["Cache"]},
        ])
        assert result.components_after == 2
        assert len(result.blueprint["constraints"]) == 1

    def test_empty_edits(self):
        bp = _blueprint("Auth")
        result = apply_edits(bp, [])
        assert len(result.operations_applied) == 0
        assert result.components_before == result.components_after

    def test_edit_result_has_operation_details(self):
        bp = _blueprint("Auth")
        result = apply_edits(bp, [
            {"operation": "rename", "old_name": "Auth", "new_name": "AuthService"},
        ])
        op = result.operations_applied[0]
        assert op.operation == "rename"
        assert op.target == "Auth"
        assert op.details == {"new_name": "AuthService"}

    def test_does_not_mutate_original(self):
        bp = _blueprint("Auth", "Database")
        _ = apply_edits(bp, [
            {"operation": "remove", "name": "Auth"},
        ])
        assert len(bp["components"]) == 2
