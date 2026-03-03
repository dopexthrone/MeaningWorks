import os
import sys
import pytest

# Add the mother subdirectory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from l1_core import track_losses

def test_entity_matching():
    """Test entity compression loss with normalized matching."""
    input_text = "Implement System Instrumentation and Entity management."
    blueprint = {
        "components": [
            {"name": "system-instrumentation"},
            {"name": "entity-management"},
        ]
    }
    losses = track_losses(input_text, blueprint)
    entity_loss = [l for l in losses if l[0] == 'entity']
    assert not entity_loss, "No entity loss expected with matching normalized names"

def test_entity_missing():
    """Test entity loss when nouns not covered."""
    input_text = "Handle ExcavationProcess and PersonaConstraints."
    blueprint = {"components": []}
    losses = track_losses(input_text, blueprint)
    entity_loss = [l for l in losses if l[0] == 'entity']
    assert entity_loss, "Entity loss expected"
    assert entity_loss[0][1] > 0.5, "Significant severity"

def test_constraint_handled():
    """Test no constraint loss when handled in blueprint."""
    input_text = "must validate before advancing phase"
    blueprint = {
        "components": [
            {"description": "validates input and handles constraints"}
        ]
    }
    losses = track_losses(input_text, blueprint)
    constraint_loss = [l for l in losses if l[0] == 'constraint']
    assert not constraint_loss, "Constraint handled, no loss"

def test_constraint_missing():
    """Test constraint loss when not handled."""
    input_text = "must trigger persona on state change"
    blueprint = {"components": [{"description": "basic state management"}]}
    losses = track_losses(input_text, blueprint)
    constraint_loss = [l for l in losses if l[0] == 'constraint']
    assert constraint_loss, "Constraint loss expected"

def test_trivial_input():
    """Test no losses for short input."""
    losses = track_losses("short", {})
    assert losses == []

def test_severity_scaling():
    """Test entity severity scales with fraction missing."""
    input_text = "Test scaling with nouns A B C D E"  # nouns: a,b,c,d,e (test filtered)
    blueprint = {"components": [{"name": "a-b"}]}  # covers a,b
    losses = track_losses(input_text, blueprint)
    entity_loss = next((l for l in losses if l[0] == 'entity'), None)
    assert entity_loss is not None
    assert entity_loss[1] == 0.6, "3/5 missing -> 0.6 severity (ratio)"

def test_fuzzy_entity_match():
    """Test fuzzy matching for entity compression improvement."""
    input_text = "Instrument Processing."
    blueprint = {
        "components": [
            {"name": "instrumentation-processor"},
        ]
    }
    losses = track_losses(input_text, blueprint)
    entity_loss = [l for l in losses if l[0] == 'entity']
    assert len(entity_loss) == 1
    assert entity_loss[0][1] == 0.5, "1/2 matched (instrument yes, processing no), loss 0.5"
