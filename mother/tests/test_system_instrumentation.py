import pytest
from core.system_instrumentation import *

def test_cells_get_confidence():
    cells = Cells()
    assert cells.get_confidence('nonexistent') == 0.0
    cells.update_confidence('test', 0.25)
    assert cells.get_confidence('test') == 0.25
    cells.update_confidence('test', 1.5)
    assert cells.get_confidence('test') == 1.0

def test_recent_append_entry():
    recent = RECENT()
    recent.append_entry('entry1')
    assert len(recent.entries) == 1
    assert recent.entries[0] == 'entry1'

def test_alerts_generate_alert():
    alert = Alerts.generate_alert('test msg')
    assert 'CRITICAL ALERT' in alert
    assert '5 cells' in alert
    assert 'TASK: strengthen' in alert

def test_perspective_analyze_via_personas():
    p = PERSPECTIVE('test_id')
    analyses = p.analyze_via_personas('test alert')
    assert len(analyses) == 3
    assert 'Monitoring Engineer' in analyses[0]
    assert 'Incident Responder' in analyses[1]
    assert 'Reliability Manager' in analyses[2]

def test_system_instrumentation_propagate_alerts_below_threshold():
    si = SystemInstrumentation('test_id')
    cells = Cells()
    cells.update_confidence('low1', 0.2)
    analyses = si.propagate_alerts(cells)
    assert len(analyses) == 0
    assert len(si.recent.entries) == 0

def test_system_instrumentation_propagate_alerts_critical():
    si = SystemInstrumentation('test_id')
    cells = Cells()
    cells.initialize()
    analyses = si.propagate_alerts(cells)
    assert len(analyses) == 3
    assert len(si.recent.entries) == 4
    assert 'CRITICAL ALERT' in si.recent.entries[-1]
    assert 'SYSTEM_INSTRUMENTATION.initialize' in si.recent.entries[-1]

def test_cells_initialize_critical_five():
    cells = Cells()
    cells.initialize()
    assert cells.get_confidence("SYSTEM_INSTRUMENTATION.initialize") == 0.25
    cm = CellMonitoring()
    low_count, low_pks = cm.process(cells)
    assert low_count == 5
    assert "Dynamic.read_flow" in low_pks

def test_recent_process_layers():
    recent = RECENT()
    cells = Cells()
    cells.initialize()
    outputs = recent.process(cells)
    assert len(outputs) == 3
    assert "L1-CORE: 5 critical cells detected." == outputs[0]
    assert len(recent.entries) == 3

def test_dynamic_read_flow():
    dyn = Dynamic()
    assert dyn.read_flow("test") == {}
    dyn.configure({"flows": {"testflow": {"step": "read"}}})
    assert dyn.read_flow("testflow") == {"step": "read"}

def test_system_instrumentation_full_process():
    si = SystemInstrumentation("test_id")
    si.initialize()
    cells = Cells()
    cells.initialize()
    analyses = si.process(cells)
    assert len(analyses) == 3
    assert si.recent.l1_core.status == "ready"

def test_perspective_process():
    p = PERSPECTIVE("test")
    analyses = p.process("alert")
    assert len(analyses) == 3

def test_cells_critical_cells():
    cells = Cells()
    cells.initialize()
    crit = cells.critical_cells
    assert len(crit) == 5
    assert "SYSTEM_INSTRUMENTATION.initialize" in crit

def test_cells_snapshot_state():
    cells = Cells()
    cells.update_confidence("test", 0.8)
    snapshot = cells.snapshot_state()
    assert "test" in snapshot
    assert snapshot["test"]["confidence"] == 0.8
    assert snapshot["test"]["fill_state"] == "EMPTY"

def test_l1core_initialize_monitoring():
    l1 = L1Core()
    cells = Cells()
    cells.initialize()
    l1.initialize_monitoring(cells)
    # Test runs without error

def test_perspective_trigger_escalation():
    p = PERSPECTIVE("test_id")
    cells = Cells()
    cells.initialize()
    res = p.trigger_escalation(cells)
    assert len(res) == 3

# MonitoringSystem Tests
def test_monitoring_system_evaluate_confidence():
    ms = MonitoringSystem("test_id")
    cells = Cells()
    cells.initialize()
    evaluation = ms.evaluate_confidence(cells)
    assert evaluation['critical_count'] == 5
    assert evaluation['total_cells'] == 15
    assert len(evaluation['critical']) == 5
    assert len(evaluation['warning']) == 10

def test_monitoring_system_identify_critical_items():
    ms = MonitoringSystem("test_id")
    cells = Cells()
    cells.initialize()
    critical_items = ms.identify_critical_items(cells)
    assert len(critical_items) == 5
    assert all(item['severity'] == 'critical' for item in critical_items)
    assert all(item['requires_immediate_attention'] for item in critical_items)
    # Should be sorted by confidence (lowest first)
    confidences = [item['confidence'] for item in critical_items]
    assert confidences == sorted(confidences)

def test_monitoring_system_scan_all_cells():
    ms = MonitoringSystem("test_id")
    cells = Cells()
    cells.initialize()
    scan_result = ms.scan_all_cells(cells)
    assert 'timestamp' in scan_result
    assert 'evaluation' in scan_result
    assert 'critical_items' in scan_result
    assert 'queue_status' in scan_result
    assert 'recommendations' in scan_result
    assert scan_result['queue_status']['critical_count'] == 5
    assert len(scan_result['recommendations']) > 0

# CriticalItemsQueue Tests
def test_critical_items_queue_add_item():
    queue = CriticalItemsQueue("test_id")
    item = {'pk': 'test_cell', 'confidence': 0.25, 'severity': 'critical'}
    result = queue.add_critical_item(item)
    assert result is True
    assert queue.get_critical_count() == 1

def test_critical_items_queue_overflow():
    queue = CriticalItemsQueue("test_id")
    # Fill queue to capacity
    for i in range(5):
        item = {'pk': f'cell_{i}', 'confidence': 0.20 + i * 0.01, 'severity': 'critical'}
        queue.add_critical_item(item)
    
    assert queue.get_critical_count() == 5
    
    # Add one more to trigger overflow
    overflow_item = {'pk': 'overflow_cell', 'confidence': 0.15, 'severity': 'critical'}
    result = queue.add_critical_item(overflow_item)
    assert result is False  # Overflow occurred
    assert queue.emergency_escalated is True
    assert len(queue.overflow_items) == 1

def test_critical_items_queue_update_existing():
    queue = CriticalItemsQueue("test_id")
    item = {'pk': 'test_cell', 'confidence': 0.25, 'severity': 'critical'}
    queue.add_critical_item(item)
    
    # Update same item
    updated_item = {'pk': 'test_cell', 'confidence': 0.20, 'severity': 'critical'}
    result = queue.add_critical_item(updated_item)
    assert result is True
    assert queue.get_critical_count() == 1
    assert queue.critical_items[0]['confidence'] == 0.20

def test_critical_items_queue_clear_resolved():
    queue = CriticalItemsQueue("test_id")
    items = [
        {'pk': 'cell_1', 'confidence': 0.25, 'severity': 'critical'},
        {'pk': 'cell_2', 'confidence': 0.20, 'severity': 'critical'},
        {'pk': 'cell_3', 'confidence': 0.15, 'severity': 'critical'}
    ]
    for item in items:
        queue.add_critical_item(item)
    
    assert queue.get_critical_count() == 3
    cleared = queue.clear_resolved_items(['cell_1', 'cell_3'])
    assert cleared == 2
    assert queue.get_critical_count() == 1
    assert queue.critical_items[0]['pk'] == 'cell_2'

# SystemOperator Tests
def test_system_operator_coordinate_response():
    so = SystemOperator("test_id")
    cells = Cells()
    cells.initialize()
    response = so.coordinate_response(cells)
    
    assert 'timestamp' in response
    assert 'scan_result' in response
    assert 'blast_radius' in response
    assert 'business_impact' in response
    assert 'prioritized_items' in response
    assert 'team_responses' in response
    assert response['action_required'] is True
    assert len(response['team_responses']) == 3

def test_system_operator_calculate_blast_radius():
    so = SystemOperator("test_id")
    critical_items = [
        {'pk': 'SYSTEM_INSTRUMENTATION.initialize', 'confidence': 0.25},
        {'pk': 'PERSPECTIVE.process', 'confidence': 0.20},
        {'pk': 'Dynamic.read_flow', 'confidence': 0.15}
    ]
    blast_radius = so.calculate_blast_radius(critical_items)
    
    assert blast_radius['radius'] == 3
    assert blast_radius['risk_level'] == 'high'
    assert 'instrumentation' in blast_radius['affected_systems']
    assert 'oversight' in blast_radius['affected_systems']
    assert 'flow_control' in blast_radius['affected_systems']

def test_system_operator_prioritize_critical_items():
    so = SystemOperator("test_id")
    critical_items = [
        {'pk': 'test_cell_1', 'confidence': 0.25},
        {'pk': 'SYSTEM_INSTRUMENTATION.initialize', 'confidence': 0.20},
        {'pk': 'test_cell_2', 'confidence': 0.15}
    ]
    prioritized = so.prioritize_critical_items(critical_items)
    
    assert len(prioritized) == 3
    assert all('priority_score' in item for item in prioritized)
    assert all('priority_rank' in item for item in prioritized)
    # Should be sorted by priority score (highest first)
    scores = [item['priority_score'] for item in prioritized]
    assert scores == sorted(scores, reverse=True)

def test_system_operator_assess_business_impact():
    so = SystemOperator("test_id")
    evaluation = {
        'critical_count': 5,
        'total_cells': 15,
        'critical': [{'pk': f'cell_{i}', 'confidence': 0.25} for i in range(5)],
        'warning': [{'pk': f'warn_{i}', 'confidence': 0.40} for i in range(10)],
        'stable': []
    }
    impact = so.assess_business_impact(evaluation)
    
    assert impact['impact_level'] == 'severe'
    assert impact['impact_score'] == 10
    assert impact['stability_ratio'] == 0.0
    assert len(impact['implications']) > 0
    assert len(impact['recommended_actions']) > 0

def test_system_operator_handle_capacity_overflow():
    so = SystemOperator("test_id")
    cells = Cells()
    cells.initialize()
    
    # Force overflow by filling queue beyond capacity
    for i in range(7):  # More than max capacity of 5
        item = {'pk': f'overflow_cell_{i}', 'confidence': 0.20, 'severity': 'critical'}
        so.monitoring_system.critical_queue.add_critical_item(item)
    
    overflow_response = so.handle_capacity_overflow(cells)
    assert overflow_response['overflow_detected'] is True
    assert len(overflow_response['emergency_actions']) > 0
    assert 'emergency_reinforcement' in overflow_response

# SystemHealthMonitor Tests
def test_system_health_monitor_scan_confidence_levels():
    shm = SystemHealthMonitor("test_id")
    cells = Cells()
    cells.initialize()
    scan = shm.scan_confidence_levels(cells)
    
    assert 'timestamp' in scan
    assert 'cell_confidence' in scan
    assert 'summary_stats' in scan
    assert 'health_indicators' in scan
    assert scan['summary_stats']['critical_count'] == 5
    assert scan['summary_stats']['warning_count'] == 10
    assert scan['health_indicators']['health_status'] in ['critical', 'warning', 'good', 'excellent']

def test_system_health_monitor_detect_degradation():
    shm = SystemHealthMonitor("test_id")
    cells = Cells()
    cells.initialize()
    
    # First scan
    shm.scan_confidence_levels(cells)
    
    # Degrade some cells
    cells.update_confidence("SYSTEM_INSTRUMENTATION.initialize", 0.05)  # Drop from 0.25 to 0.05
    cells.update_confidence("PERSPECTIVE.process", 0.10)  # Drop from 0.25 to 0.10
    
    # Second scan to detect degradation
    degradation = shm.detect_degradation(cells)
    
    assert degradation['degradation_detected'] is True
    assert len(degradation['degraded_cells']) == 2
    assert degradation['alert_level'] in ['medium', 'high', 'critical']
    assert len(degradation['recommendations']) > 0

def test_system_health_monitor_trend_analysis():
    shm = SystemHealthMonitor("test_id")
    cells = Cells()
    cells.initialize()
    
    # Multiple scans to build history
    for i in range(5):
        # Gradually improve confidence
        for pk in cells.cells:
            current = cells.get_confidence(pk)
            cells.update_confidence(pk, min(1.0, current + 0.05))
        shm.scan_confidence_levels(cells)
    
    degradation = shm.detect_degradation(cells)
    trend = degradation['trend_analysis']
    
    assert 'trend' in trend
    assert trend['trend'] in ['improving', 'degrading', 'stable', 'insufficient_data']

# Cell Reinforcement System Tests
def test_cell_reinforcement_system_analyze_stability():
    crs = CellReinforcementSystem("test_id")
    cells = Cells()
    cells.initialize()
    stability = crs.analyze_stability(cells)
    
    assert 'critical_count' in stability
    assert 'reinforcement_candidates' in stability
    assert 'stable_cells' in stability
    assert 'stability_score' in stability
    assert 'needs_reinforcement' in stability
    assert stability['critical_count'] == 5
    assert stability['reinforcement_candidates'] == 10

def test_cell_reinforcement_system_execute_reinforcement():
    crs = CellReinforcementSystem("test_id")
    cells = Cells()
    cells.initialize()
    
    # Get initial confidence levels
    initial_confidences = {pk: cells.get_confidence(pk) for pk in cells.get_reinforcement_candidates()}
    
    results = crs.execute_reinforcement(cells, target_count=5)
    
    assert len(results) == 5
    for pk, result in results.items():
        assert 'old' in result
        assert 'new' in result
        assert 'boost' in result
        assert result['new'] > result['old']
        assert result['new'] > initial_confidences[pk]

def test_cell_reinforcement_system_monitor_progress():
    crs = CellReinforcementSystem("test_id")
    cells = Cells()
    cells.initialize()
    
    # Execute reinforcement first
    crs.execute_reinforcement(cells, target_count=3)
    
    progress = crs.monitor_progress(cells)
    
    assert 'current_stability' in progress
    assert 'reinforcement_cycles' in progress
    assert 'last_reinforcement' in progress
    assert progress['reinforcement_cycles'] == 1
    assert progress['last_reinforcement'] is not None

# Integration Tests
def test_full_system_integration():
    """Test complete system integration with all components working together"""
    instance_id = "TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)"
    
    # Initialize all components
    cells = Cells()
    cells.initialize()
    
    system_operator = SystemOperator(instance_id)
    health_monitor = SystemHealthMonitor(instance_id)
    
    # Initial system scan
    initial_scan = health_monitor.scan_confidence_levels(cells)
    assert initial_scan['summary_stats']['critical_count'] == 5
    
    # Coordinate response
    response = system_operator.coordinate_response(cells)
    assert response['action_required'] is True
    assert len(response['prioritized_items']) == 5
    
    # Execute reinforcement
    reinforcement_results = system_operator.monitoring_system.critical_queue
    assert reinforcement_results.get_critical_count() == 5
    
    # Verify system improvement after reinforcement
    si = SystemInstrumentation(instance_id)
    reinforcement_outcome = si.reinforce_low_confidence_cells(cells)
    assert reinforcement_outcome['reinforcement_executed'] is True
    
    # Final health check
    final_scan = health_monitor.scan_confidence_levels(cells)
    assert final_scan['summary_stats']['critical_count'] <= 5  # Should be same or better

def test_emergency_escalation_scenario():
    """Test emergency escalation when critical queue overflows"""
    instance_id = "TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)"
    
    cells = Cells()
    cells.initialize()
    
    # Add more critical cells to force overflow
    for i in range(10):
        cells.update_confidence(f"emergency_cell_{i}", 0.15)
    
    system_operator = SystemOperator(instance_id)
    response = system_operator.coordinate_response(cells)
    
    # Should trigger overflow handling
    assert response['overflow_response'] is not None
    assert response['overflow_response']['overflow_detected'] is True
    
    # Emergency reinforcement should be executed
    assert 'emergency_reinforcement' in response['overflow_response']
    emergency_reinforcement = response['overflow_response']['emergency_reinforcement']
    # The emergency reinforcement returns results from execute_reinforcement
    assert len(emergency_reinforcement) > 0  # Should have reinforced some cells

def test_degradation_detection_and_response():
    """Test degradation detection and automated response"""
    instance_id = "TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)"
    
    cells = Cells()
    cells.initialize()
    
    health_monitor = SystemHealthMonitor(instance_id)
    
    # Initial scan
    health_monitor.scan_confidence_levels(cells)
    
    # Simulate significant degradation
    for pk in list(cells.cells.keys())[:3]:
        current = cells.get_confidence(pk)
        cells.update_confidence(pk, max(0.0, current - 0.20))  # Drop by 20%
    
    # Detect degradation
    degradation = health_monitor.detect_degradation(cells)
    
    assert degradation['degradation_detected'] is True
    assert len(degradation['degraded_cells']) >= 3
    assert degradation['alert_level'] in ['high', 'critical']
    
    # System should respond to degradation
    system_operator = SystemOperator(instance_id)
    response = system_operator.coordinate_response(cells)
    
    assert response['action_required'] is True
    assert response['blast_radius']['risk_level'] in ['high', 'critical']
