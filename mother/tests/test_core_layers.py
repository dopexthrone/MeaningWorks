"""
Tests for core layers mapping #84.

5 tests, 1 per layer/component. Coverage: 5/5 nouns->components verified.

Provenance: Map-core-layers-AGN-ORG-SEM-STA-STR-#84
"""

import pytest

from core.sem import SEM

def test_sem():
    instance_id = "test-#84"
    sem = SEM(instance_id)
    result = sem.process("test data")
    expected = f"SEM processed 'test data' | instance_id={instance_id}"
    assert result == expected
    assert sem.instance_id == instance_id

from core.org import ORG

def test_org():
    instance_id = "test-#84"
    org = ORG(instance_id)
    result = org.process("test data")
    expected = f"ORG organized 'test data' | instance_id={instance_id}"
    assert result == expected
    assert org.instance_id == instance_id

from core.agn import AGN

def test_agn():
    instance_id = "test-#84"
    agn = AGN(instance_id)
    result = agn.process("test data")
    expected = f"AGN dispatched 'test data' | instance_id={instance_id}"
    assert result == expected
    assert agn.instance_id == instance_id

from core.sta import STA

def test_sta():
    instance_id = "test-#84"
    sta = STA(instance_id)
    result = sta.process("test data")
    expected = f"STA updated 'test data' | instance_id={instance_id}"
    assert result == expected
    assert sta.instance_id == instance_id

from core.str import STR

def test_str():
    instance_id = "test-#84"
    str_layer = STR(instance_id)
    result = str_layer.process("test data")
    expected = f"STR mapped 'test data' | instance_id={instance_id}"
    assert result == expected
    assert str_layer.instance_id == instance_id
