from dataclasses import dataclass, asdict
from typing import Dict, List, Any
from enum import Enum
import logging
logger = logging.getLogger("mother.core.system_instrumentation")

@dataclass
class Cell:
    confidence: float = 0.0
    fill_state: str = 'EMPTY'
    primitive: str = ''

class Cells:
    '''
    Cell State
    Central data entities holding confidence metrics (S,B,C); owned by Core Instrumentation, accessed by all for resolution prioritization.
    methods: get_confidence, update_confidence, snapshot_state, critical_cells
    instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
    '''
    def __init__(self):
        self.cells: Dict[str, Cell] = {}
    
    def get_confidence(self, pk: str) -> float:
        cell = self.cells.get(pk)
        return cell.confidence if cell else 0.0
    
    def update_confidence(self, pk: str, confidence: float) -> None:
        if pk not in self.cells:
            self.cells[pk] = Cell()
        self.cells[pk].confidence = max(0.0, min(1.0, confidence))

    def initialize(self) -> None:
        """Initialize with 5 critical low-confidence cells and 10 reinforcement cells 30-50% (#98)."""
        critical_pks = [
            "SYSTEM_INSTRUMENTATION.initialize",
            "PERSPECTIVE.process",
            "DOMAIN_ENTITIES.configure",
            "RECENT.process",
            "Dynamic.read_flow"
        ]
        for pk in critical_pks:
            self.update_confidence(pk, 0.25)
        
        # 10 low-confidence cells between 30-50% requiring reinforcement
        reinforcement_pks = [
            "L1Core.process",
            "L2State.process", 
            "L3Now.process",
            "CellMonitoring.check_low_confidence",
            "MonitoringEngineer.analyze",
            "IncidentResponder.analyze",
            "ReliabilityManager.analyze",
            "SystemInstrumentation.propagate_alerts",
            "PERSPECTIVE.trigger_escalation",
            "Cells.snapshot_state"
        ]
        # Initialize with varying confidence levels in 30-50% range
        confidence_levels = [0.32, 0.35, 0.38, 0.41, 0.44, 0.47, 0.33, 0.36, 0.39, 0.42]
        for pk, conf in zip(reinforcement_pks, confidence_levels):
            self.update_confidence(pk, conf)

    def process(self, updates: Dict[str, float]) -> Dict[str, float]:
        """Process confidence updates and return applied values."""
        applied = {}
        for pk, conf in updates.items():
            self.update_confidence(pk, conf)
            applied[pk] = self.get_confidence(pk)
        return applied

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure cell parameters."""
        pass

    @property
    def critical_cells(self) -> List[str]:
        '''
        critical_cells (contains)
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        return [pk for pk, cell in self.cells.items() if cell.confidence < 0.3]

    def snapshot_state(self) -> Dict[str, Dict[str, Any]]:
        '''
        snapshot_state
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        return {pk: asdict(cell) for pk, cell in self.cells.items()}

    def get_reinforcement_candidates(self) -> List[str]:
        '''
        Get cells with confidence between 30-50% that need reinforcement
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        return [pk for pk, cell in self.cells.items() if 0.30 <= cell.confidence <= 0.50]

    def reinforce_cell(self, pk: str, boost: float = 0.15) -> float:
        '''
        Reinforce a specific cell by boosting its confidence
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        if pk in self.cells:
            old_conf = self.cells[pk].confidence
            new_conf = min(1.0, old_conf + boost)
            self.update_confidence(pk, new_conf)
            return new_conf
        return 0.0

    def bulk_reinforce(self, target_pks: List[str] = None, boost: float = 0.15) -> Dict[str, float]:
        '''
        Reinforce multiple cells to improve system stability
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        if target_pks is None:
            target_pks = self.get_reinforcement_candidates()
        
        results = {}
        for pk in target_pks:
            results[pk] = self.reinforce_cell(pk, boost)
        return results

class RECENT:
    '''
    Recent process and entity entries folded as history
    methods: append_entry
    instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
    '''
    def __init__(self):
        self.entries: List[str] = []
        self.l1_core = L1Core()
        self.l2_state = L2State()
        self.l3_now = L3Now()
    
    def append_entry(self, entry: str) -> None:
        self.entries.append(entry)

    def initialize(self) -> None:
        """Initialize recent layers."""
        self.l1_core.initialize()
        self.l2_state.initialize()
        self.l3_now.initialize()

    def process(self, cells: Cells) -> List[str]:
        """Process through L1-L3 layers."""
        l1 = self.l1_core.process(cells)
        l2 = self.l2_state.process(l1)
        l3 = self.l3_now.process(l2)
        outputs = [l1, l2, l3]
        self.entries.extend(outputs)
        return outputs

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure recent subsystems."""
        pass

class Alerts:
    '''
    Alerts generated for low-confidence critical cells
    methods: generate_alert
    instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
    '''
    @staticmethod
    def generate_alert(message: str) -> str:
        instance_id = 'TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)'
        return f'CRITICAL ALERT | instance_id={instance_id} | 5 cells <30% confidence: {message}'

class L1Core:
    '''
    Core Instrumentation.L1-CORE (contains)
    instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
    methods: initialize_monitoring, initialize, process, configure
    '''
    def __init__(self):
        self.status: str = "idle"

    def initialize(self) -> None:
        self.status = "ready"

    def process(self, cells: Cells) -> str:
        low_count = sum(1 for c in cells.cells.values() if c.confidence < 0.3)
        return f"L1-CORE: {low_count} critical cells detected."

    def configure(self, threshold: float = 0.3) -> None:
        pass

    def initialize_monitoring(self, cells: Cells) -> None:
        '''
        initialize_monitoring
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        self.initialize()
        crit_count = len(cells.critical_cells)
        logger.info(f"L1-CORE: Initialized monitoring for {crit_count} critical cells.")

class L2State:
    '''
    L2-STATE (contains)
    instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
    methods: initialize, process, configure
    '''
    def __init__(self):
        self.state_data: Dict[str, Any] = {}
        self.dynamic = Dynamic()

    def initialize(self) -> None:
        self.state_data = {}

    def process(self, l1_output: str) -> str:
        self.state_data["l1"] = l1_output
        return f"L2-STATE: State updated from L1: {l1_output}"

    def configure(self, config: Dict[str, Any]) -> None:
        self.state_data.update(config)

class L3Now:
    '''
    L3-NOW (contains)
    instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
    methods: initialize, process, configure
    '''
    def __init__(self):
        self.now: str = ""

    def initialize(self) -> None:
        self.now = ""

    def process(self, l2_output: str) -> str:
        self.now = l2_output
        return f"L3-NOW: Current state: {l2_output}"

    def configure(self, params: Dict[str, Any]) -> None:
        pass

class Dynamic:
    '''
    Dynamic
    Dynamic read/write flows; referenced in L2-STATE discovered
    methods: read_flow, initialize, process, configure
    instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
    '''
    def __init__(self):
        self.flows: Dict[str, Dict[str, Any]] = {}

    def read_flow(self, flow_id: str) -> Dict[str, Any]:
        return self.flows.get(flow_id, {})

    def initialize(self) -> None:
        pass

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data

    def configure(self, config: Dict[str, Any]) -> None:
        self.flows.update(config.get('flows', {}))

class CellMonitoring:
    '''
    CellMonitoring (contains)
    instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
    methods: initialize, process, configure
    '''
    def __init__(self):
        self.threshold: float = 0.3

    def initialize(self) -> None:
        pass

    def check_low_confidence(self, cells: Cells) -> tuple[int, List[str]]:
        low_pks = [pk for pk, c in cells.cells.items() if c.confidence < self.threshold]
        return len(low_pks), low_pks

    def process(self, cells: Cells) -> tuple[int, List[str]]:
        return self.check_low_confidence(cells)

    def configure(self, threshold: float) -> None:
        self.threshold = max(0.0, min(1.0, threshold))

class MonitoringEngineer:
    '''
    MonitoringEngineer (contains)
    instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
    '''
    def analyze(self, alert: str) -> str:
        return 'Monitoring Engineer: Deploy additional sensors to low-confidence cells.'

class IncidentResponder:
    '''
    Incident_Responder (contains)
    instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
    '''
    def analyze(self, alert: str) -> str:
        return 'Incident Responder: Isolate affected cells and initiate recovery protocols.'

class ReliabilityManager:
    '''
    Reliability_Manager (contains)
    instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
    '''
    def analyze(self, alert: str) -> str:
        return 'Reliability Manager: Reallocate resources to boost cell confidence above 30%.'

class PERSPECTIVE:
    '''
    Persona Oversight
    Behavioral layer owned by PERSPECTIVE for triage/escalation; depends on Cell State post-instrumentation init.
    methods: trigger_escalation, analyze_via_personas
    instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
    '''
    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        self.monitoring_engineer = MonitoringEngineer()
        self.incident_responder = IncidentResponder()
        self.reliability_manager = ReliabilityManager()
    
    def analyze_via_personas(self, alert: str) -> List[str]:
        return [
            self.monitoring_engineer.analyze(alert),
            self.incident_responder.analyze(alert),
            self.reliability_manager.analyze(alert)
        ]

    def initialize(self) -> None:
        pass

    def process(self, alert: str) -> List[str]:
        return self.analyze_via_personas(alert)

    def configure(self, config: Dict[str, Any]) -> None:
        pass

    def trigger_escalation(self, cells: Cells) -> List[str]:
        '''
        trigger_escalation
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        cm = CellMonitoring()
        low_count, low_pks = cm.check_low_confidence(cells)
        if low_count >= 5:
            message = f"need immediate attention. Low confidence cells: {', '.join(low_pks)} (Total critical: {low_count})"
            alert = Alerts.generate_alert(message)
            return self.analyze_via_personas(alert)
        return []

class CellReinforcementSystem:
    '''
    Cell Reinforcement System for improving stability of low-confidence cells
    methods: analyze_stability, execute_reinforcement, monitor_progress
    instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
    '''
    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        self.reinforcement_history: List[Dict[str, Any]] = []
        
    def analyze_stability(self, cells: Cells) -> Dict[str, Any]:
        '''
        Analyze system stability based on cell confidence distribution
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        critical_count = len(cells.critical_cells)
        reinforcement_candidates = cells.get_reinforcement_candidates()
        stable_cells = [pk for pk, cell in cells.cells.items() if cell.confidence > 0.70]
        
        stability_score = len(stable_cells) / max(1, len(cells.cells))
        
        return {
            'critical_count': critical_count,
            'reinforcement_candidates': len(reinforcement_candidates),
            'stable_cells': len(stable_cells),
            'stability_score': stability_score,
            'needs_reinforcement': len(reinforcement_candidates) >= 5
        }
    
    def execute_reinforcement(self, cells: Cells, target_count: int = 10) -> Dict[str, Any]:
        '''
        Execute reinforcement on low-confidence cells to improve stability
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        candidates = cells.get_reinforcement_candidates()
        target_cells = candidates[:target_count]
        
        # Apply graduated reinforcement based on current confidence
        results = {}
        for pk in target_cells:
            current_conf = cells.get_confidence(pk)
            if current_conf < 0.35:
                boost = 0.20  # Stronger boost for very low confidence
            elif current_conf < 0.45:
                boost = 0.15  # Medium boost
            else:
                boost = 0.10  # Light boost for borderline cases
                
            new_conf = cells.reinforce_cell(pk, boost)
            results[pk] = {'old': current_conf, 'new': new_conf, 'boost': boost}
        
        # Record reinforcement action
        self.reinforcement_history.append({
            'timestamp': 'now',
            'target_cells': target_cells,
            'results': results
        })
        
        return results
    
    def monitor_progress(self, cells: Cells) -> Dict[str, Any]:
        '''
        Monitor reinforcement progress and system stability improvements
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        stability = self.analyze_stability(cells)
        
        progress = {
            'current_stability': stability,
            'reinforcement_cycles': len(self.reinforcement_history),
            'last_reinforcement': self.reinforcement_history[-1] if self.reinforcement_history else None
        }
        
        return progress

class SystemInstrumentation:
    '''
    Hierarchical alerting layers that read from cells, compute confidence, and propagate critical alerts (e.g., 5 cells <30%)
    methods: propagate_alerts
    instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
    '''
    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        logger.info("SystemInstrumentation initialized", extra={"instance_id": instance_id})
        self.recent = RECENT()
        self.cell_monitoring = CellMonitoring()
        self.reinforcement_system = CellReinforcementSystem(instance_id)
    
    def propagate_alerts(self, cells: Cells) -> List[str]:
        low_count, low_pks = self.cell_monitoring.check_low_confidence(cells)
        logger.info("Alert check", extra={"low_count": low_count, "low_pks_count": len(low_pks)})
        if low_count >= 5:
            logger.critical("CRITICAL low conf cells", extra={"count": low_count, "pks": low_pks[:3], "instance_id": self.instance_id})
            message = f"need immediate attention. Low confidence cells: {', '.join(low_pks)} (Total critical: {low_count})"
            alert = Alerts.generate_alert(message)
            self.recent.process(cells)
            self.recent.append_entry(alert)
            perspective = PERSPECTIVE(self.instance_id)
            return perspective.analyze_via_personas(alert)
        return []

    def initialize(self) -> None:
        """Initialize system instrumentation."""
        self.recent.initialize()
        self.cell_monitoring.initialize()

    def process(self, cells: Cells) -> List[str]:
        """Full process: recent layers + alerts."""
        self.recent.process(cells)
        return self.propagate_alerts(cells)

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure subsystems."""
        self.cell_monitoring.configure(config.get('threshold', 0.3))
        self.recent.configure(config)

    def reinforce_low_confidence_cells(self, cells: Cells) -> Dict[str, Any]:
        """
        Reinforce 10 low-confidence cells between 30-50% to improve system stability
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        """
        stability_analysis = self.reinforcement_system.analyze_stability(cells)
        
        if stability_analysis['needs_reinforcement']:
            logger.info("Executing cell reinforcement", extra={
                "instance_id": self.instance_id,
                "candidates": stability_analysis['reinforcement_candidates']
            })
            
            reinforcement_results = self.reinforcement_system.execute_reinforcement(cells, target_count=10)
            progress = self.reinforcement_system.monitor_progress(cells)
            
            return {
                'reinforcement_executed': True,
                'cells_reinforced': len(reinforcement_results),
                'results': reinforcement_results,
                'stability_improvement': progress
            }
        
        return {
            'reinforcement_executed': False,
            'reason': 'Insufficient candidates for reinforcement',
            'stability': stability_analysis
        }

class MonitoringSystem:
    '''
    Continuously evaluates cell confidence levels against 30% threshold and identifies critical items requiring attention
    methods: evaluate_confidence, identify_critical_items, scan_all_cells
    instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
    '''
    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        self.threshold: float = 0.30
        self.critical_queue = CriticalItemsQueue(instance_id)
        
    def evaluate_confidence(self, cells: Cells) -> Dict[str, Any]:
        '''
        Evaluate confidence levels across all cells and categorize by severity
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        critical_cells = []
        warning_cells = []
        stable_cells = []
        
        for pk, cell in cells.cells.items():
            if cell.confidence < self.threshold:
                critical_cells.append({'pk': pk, 'confidence': cell.confidence})
            elif cell.confidence < 0.50:
                warning_cells.append({'pk': pk, 'confidence': cell.confidence})
            else:
                stable_cells.append({'pk': pk, 'confidence': cell.confidence})
        
        return {
            'critical': critical_cells,
            'warning': warning_cells,
            'stable': stable_cells,
            'total_cells': len(cells.cells),
            'critical_count': len(critical_cells)
        }
    
    def identify_critical_items(self, cells: Cells) -> List[Dict[str, Any]]:
        '''
        Identify critical items requiring immediate attention based on confidence threshold
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        evaluation = self.evaluate_confidence(cells)
        critical_items = []
        
        for item in evaluation['critical']:
            critical_items.append({
                'pk': item['pk'],
                'confidence': item['confidence'],
                'severity': 'critical',
                'priority': 1,
                'requires_immediate_attention': True
            })
        
        # Sort by confidence (lowest first - most critical)
        critical_items.sort(key=lambda x: x['confidence'])
        return critical_items
    
    def scan_all_cells(self, cells: Cells) -> Dict[str, Any]:
        '''
        Comprehensive scan of all cells with detailed analysis
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        evaluation = self.evaluate_confidence(cells)
        critical_items = self.identify_critical_items(cells)
        
        # Update critical queue with new items
        for item in critical_items:
            self.critical_queue.add_critical_item(item)
        
        scan_result = {
            'timestamp': 'now',
            'evaluation': evaluation,
            'critical_items': critical_items,
            'queue_status': {
                'critical_count': self.critical_queue.get_critical_count(),
                'overflow_triggered': self.critical_queue.get_critical_count() > 5
            },
            'recommendations': self._generate_recommendations(evaluation)
        }
        
        return scan_result
    
    def _generate_recommendations(self, evaluation: Dict[str, Any]) -> List[str]:
        '''Generate actionable recommendations based on evaluation results'''
        recommendations = []
        
        if evaluation['critical_count'] >= 5:
            recommendations.append("URGENT: 5+ cells below 30% confidence - immediate reinforcement required")
        elif evaluation['critical_count'] > 0:
            recommendations.append(f"WARNING: {evaluation['critical_count']} critical cells need attention")
        
        if len(evaluation['warning']) > 10:
            recommendations.append("Consider bulk reinforcement for warning-level cells")
        
        stability_ratio = len(evaluation['stable']) / evaluation['total_cells']
        if stability_ratio < 0.5:
            recommendations.append("System stability compromised - less than 50% stable cells")
        
        return recommendations

class CriticalItemsQueue:
    '''
    Maintains ordered list of exactly 5 critical items requiring immediate attention with overflow handling
    methods: add_critical_item, get_critical_count, overflow_to_emergency
    instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
    '''
    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        self.critical_items: List[Dict[str, Any]] = []
        self.max_capacity: int = 5
        self.overflow_items: List[Dict[str, Any]] = []
        self.emergency_escalated: bool = False
        
    def add_critical_item(self, item: Dict[str, Any]) -> bool:
        '''
        Add critical item to queue, handling overflow to emergency escalation
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        # Check if item already exists (by pk)
        existing_pks = [existing['pk'] for existing in self.critical_items]
        if item['pk'] in existing_pks:
            # Update existing item with latest data
            for i, existing in enumerate(self.critical_items):
                if existing['pk'] == item['pk']:
                    self.critical_items[i] = item
                    return True
        
        # Add new item
        if len(self.critical_items) < self.max_capacity:
            self.critical_items.append(item)
            # Sort by confidence (lowest first)
            self.critical_items.sort(key=lambda x: x['confidence'])
            return True
        else:
            # Queue is full - trigger overflow
            return self.overflow_to_emergency(item)
    
    def get_critical_count(self) -> int:
        '''
        Get current count of critical items in queue
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        return len(self.critical_items)
    
    def overflow_to_emergency(self, item: Dict[str, Any]) -> bool:
        '''
        Handle overflow when queue exceeds capacity - escalate to emergency protocols
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        self.overflow_items.append(item)
        self.emergency_escalated = True
        
        logger.critical("Critical queue overflow - emergency escalation triggered", extra={
            "instance_id": self.instance_id,
            "queue_size": len(self.critical_items),
            "overflow_count": len(self.overflow_items),
            "new_item_pk": item['pk'],
            "new_item_confidence": item['confidence']
        })
        
        return False  # Indicates overflow occurred
    
    def get_queue_status(self) -> Dict[str, Any]:
        '''Get comprehensive queue status'''
        return {
            'critical_count': len(self.critical_items),
            'overflow_count': len(self.overflow_items),
            'capacity': self.max_capacity,
            'emergency_escalated': self.emergency_escalated,
            'queue_full': len(self.critical_items) >= self.max_capacity,
            'items': self.critical_items.copy()
        }
    
    def clear_resolved_items(self, resolved_pks: List[str]) -> int:
        '''Remove resolved items from queue'''
        initial_count = len(self.critical_items)
        self.critical_items = [item for item in self.critical_items if item['pk'] not in resolved_pks]
        return initial_count - len(self.critical_items)

class SystemOperator:
    '''
    Command and control layer that orchestrates overall system behavior and coordinates cross-team response when critical thresholds are reached
    methods: coordinate_response, calculate_blast_radius, prioritize_critical_items, assess_business_impact, handle_capacity_overflow
    instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
    '''
    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        self.monitoring_system = MonitoringSystem(instance_id)
        self.response_teams = {
            'monitoring': MonitoringEngineer(),
            'incident': IncidentResponder(),
            'reliability': ReliabilityManager()
        }
        self.active_responses: List[Dict[str, Any]] = []
        
    def coordinate_response(self, cells: Cells) -> Dict[str, Any]:
        '''
        Coordinate comprehensive response to critical cell conditions
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        # Scan system state
        scan_result = self.monitoring_system.scan_all_cells(cells)
        
        # Calculate impact
        blast_radius = self.calculate_blast_radius(scan_result['critical_items'])
        business_impact = self.assess_business_impact(scan_result['evaluation'])
        
        # Prioritize response
        prioritized_items = self.prioritize_critical_items(scan_result['critical_items'])
        
        # Handle overflow if needed
        overflow_response = None
        queue_status = self.monitoring_system.critical_queue.get_queue_status()
        if queue_status['overflow_count'] > 0 or queue_status['emergency_escalated']:
            overflow_response = self.handle_capacity_overflow(cells)
        
        # Coordinate team responses
        team_responses = {}
        if scan_result['critical_items']:
            alert_message = f"Critical system state: {len(scan_result['critical_items'])} cells below 30% confidence"
            for team_name, team in self.response_teams.items():
                team_responses[team_name] = team.analyze(alert_message)
        
        response = {
            'timestamp': 'now',
            'scan_result': scan_result,
            'blast_radius': blast_radius,
            'business_impact': business_impact,
            'prioritized_items': prioritized_items,
            'team_responses': team_responses,
            'overflow_response': overflow_response,
            'action_required': len(scan_result['critical_items']) > 0
        }
        
        self.active_responses.append(response)
        return response
    
    def calculate_blast_radius(self, critical_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        '''
        Calculate potential impact radius of critical cell failures
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        if not critical_items:
            return {'radius': 0, 'affected_systems': [], 'risk_level': 'none'}
        
        # Analyze interconnected systems based on cell types
        system_categories = {}
        for item in critical_items:
            pk = item['pk']
            if 'SYSTEM_INSTRUMENTATION' in pk:
                system_categories['instrumentation'] = system_categories.get('instrumentation', 0) + 1
            elif 'PERSPECTIVE' in pk:
                system_categories['oversight'] = system_categories.get('oversight', 0) + 1
            elif 'Dynamic' in pk:
                system_categories['flow_control'] = system_categories.get('flow_control', 0) + 1
            else:
                system_categories['core_processing'] = system_categories.get('core_processing', 0) + 1
        
        # Calculate risk level
        total_critical = len(critical_items)
        if total_critical >= 5:
            risk_level = 'critical'
        elif total_critical >= 3:
            risk_level = 'high'
        elif total_critical >= 1:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'radius': total_critical,
            'affected_systems': list(system_categories.keys()),
            'system_breakdown': system_categories,
            'risk_level': risk_level,
            'cascade_potential': total_critical >= 3
        }
    
    def prioritize_critical_items(self, critical_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        '''
        Prioritize critical items based on confidence level and system importance
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        if not critical_items:
            return []
        
        # Add priority scoring
        for item in critical_items:
            base_score = (0.30 - item['confidence']) * 100  # Lower confidence = higher score
            
            # System importance multiplier
            pk = item['pk']
            if 'SYSTEM_INSTRUMENTATION' in pk:
                importance_multiplier = 1.5  # Critical infrastructure
            elif 'PERSPECTIVE' in pk:
                importance_multiplier = 1.3  # Oversight systems
            elif 'Dynamic' in pk:
                importance_multiplier = 1.2  # Flow control
            else:
                importance_multiplier = 1.0  # Standard processing
            
            item['priority_score'] = base_score * importance_multiplier
            item['priority_rank'] = 0  # Will be set after sorting
        
        # Sort by priority score (highest first)
        prioritized = sorted(critical_items, key=lambda x: x['priority_score'], reverse=True)
        
        # Assign ranks
        for i, item in enumerate(prioritized):
            item['priority_rank'] = i + 1
        
        return prioritized
    
    def assess_business_impact(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        '''
        Assess business impact of current system state
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        critical_count = evaluation['critical_count']
        total_cells = evaluation['total_cells']
        stability_ratio = len(evaluation['stable']) / total_cells if total_cells > 0 else 0
        
        # Impact scoring
        if critical_count >= 5:
            impact_level = 'severe'
            impact_score = 10
        elif critical_count >= 3:
            impact_level = 'high'
            impact_score = 7
        elif critical_count >= 1:
            impact_level = 'moderate'
            impact_score = 4
        else:
            impact_level = 'minimal'
            impact_score = 1
        
        # Business implications
        implications = []
        if stability_ratio < 0.3:
            implications.append("System reliability severely compromised")
        if critical_count >= 5:
            implications.append("Service degradation likely")
            implications.append("Customer impact probable")
        if len(evaluation['warning']) > 15:
            implications.append("Preventive maintenance required")
        
        return {
            'impact_level': impact_level,
            'impact_score': impact_score,
            'stability_ratio': stability_ratio,
            'implications': implications,
            'recommended_actions': self._get_business_actions(impact_level)
        }
    
    def handle_capacity_overflow(self, cells: Cells) -> Dict[str, Any]:
        '''
        Handle critical queue capacity overflow with emergency protocols
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        queue_status = self.monitoring_system.critical_queue.get_queue_status()
        
        # Emergency response protocols
        emergency_actions = []
        if queue_status['overflow_count'] > 0:
            emergency_actions.append("Activate emergency response team")
            emergency_actions.append("Implement immediate cell reinforcement")
            emergency_actions.append("Escalate to senior operations")
        
        # Attempt emergency reinforcement
        reinforcement_system = CellReinforcementSystem(self.instance_id)
        emergency_reinforcement = reinforcement_system.execute_reinforcement(cells, target_count=15)
        
        return {
            'overflow_detected': True,
            'queue_status': queue_status,
            'emergency_actions': emergency_actions,
            'emergency_reinforcement': emergency_reinforcement,
            'escalation_required': queue_status['overflow_count'] > 3
        }
    
    def _get_business_actions(self, impact_level: str) -> List[str]:
        '''Get recommended business actions based on impact level'''
        actions = {
            'severe': [
                "Immediate executive notification",
                "Customer communication preparation",
                "Service degradation protocols",
                "Emergency resource allocation"
            ],
            'high': [
                "Operations team alert",
                "Accelerated remediation",
                "Stakeholder notification",
                "Resource reallocation"
            ],
            'moderate': [
                "Standard remediation procedures",
                "Monitoring intensification",
                "Preventive measures"
            ],
            'minimal': [
                "Routine monitoring",
                "Scheduled maintenance"
            ]
        }
        return actions.get(impact_level, [])

class SystemHealthMonitor:
    '''
    Tracks component confidence levels and detects degradation patterns with emergency escalation capabilities
    methods: scan_confidence_levels, detect_degradation
    instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
    '''
    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        self.confidence_history: List[Dict[str, Any]] = []
        self.degradation_threshold: float = 0.15  # 15% drop triggers alert
        self.monitoring_window: int = 10  # Track last 10 scans
        
    def scan_confidence_levels(self, cells: Cells) -> Dict[str, Any]:
        '''
        Comprehensive scan of confidence levels across all cells
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        current_scan = {
            'timestamp': 'now',
            'cell_confidence': {},
            'summary_stats': {},
            'health_indicators': {}
        }
        
        # Collect current confidence levels
        confidences = []
        for pk, cell in cells.cells.items():
            current_scan['cell_confidence'][pk] = cell.confidence
            confidences.append(cell.confidence)
        
        # Calculate summary statistics
        if confidences:
            current_scan['summary_stats'] = {
                'mean_confidence': sum(confidences) / len(confidences),
                'min_confidence': min(confidences),
                'max_confidence': max(confidences),
                'critical_count': len([c for c in confidences if c < 0.30]),
                'warning_count': len([c for c in confidences if 0.30 <= c < 0.50]),
                'stable_count': len([c for c in confidences if c >= 0.50])
            }
        
        # Health indicators
        current_scan['health_indicators'] = self._calculate_health_indicators(current_scan['summary_stats'])
        
        # Store in history
        self.confidence_history.append(current_scan)
        if len(self.confidence_history) > self.monitoring_window:
            self.confidence_history.pop(0)
        
        return current_scan
    
    def detect_degradation(self, cells: Cells) -> Dict[str, Any]:
        '''
        Detect confidence degradation patterns and trigger alerts
        instance_id: TASK: strengthen implementation and add test coverage for 5 cells below 30% confidence need immediate attention. (Total critical: 5)
        '''
        current_scan = self.scan_confidence_levels(cells)
        
        degradation_analysis = {
            'degradation_detected': False,
            'degraded_cells': [],
            'trend_analysis': {},
            'alert_level': 'none',
            'recommendations': []
        }
        
        if len(self.confidence_history) < 2:
            return degradation_analysis
        
        # Compare with previous scan
        previous_scan = self.confidence_history[-2]
        degraded_cells = []
        
        for pk in current_scan['cell_confidence']:
            current_conf = current_scan['cell_confidence'][pk]
            previous_conf = previous_scan['cell_confidence'].get(pk, 0.0)
            
            confidence_drop = previous_conf - current_conf
            if confidence_drop >= self.degradation_threshold:
                degraded_cells.append({
                    'pk': pk,
                    'current_confidence': current_conf,
                    'previous_confidence': previous_conf,
                    'degradation_amount': confidence_drop,
                    'degradation_percent': (confidence_drop / previous_conf * 100) if previous_conf > 0 else 0
                })
        
        if degraded_cells:
            degradation_analysis['degradation_detected'] = True
            degradation_analysis['degraded_cells'] = degraded_cells
            
            # Determine alert level
            max_degradation = max(cell['degradation_amount'] for cell in degraded_cells)
            if max_degradation >= 0.30:
                degradation_analysis['alert_level'] = 'critical'
            elif max_degradation >= 0.20:
                degradation_analysis['alert_level'] = 'high'
            else:
                degradation_analysis['alert_level'] = 'medium'
            
            # Generate recommendations
            degradation_analysis['recommendations'] = self._generate_degradation_recommendations(degraded_cells)
        
        # Trend analysis
        degradation_analysis['trend_analysis'] = self._analyze_confidence_trends()
        
        return degradation_analysis
    
    def _calculate_health_indicators(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        '''Calculate system health indicators'''
        if not stats:
            return {}
        
        total_cells = stats['critical_count'] + stats['warning_count'] + stats['stable_count']
        
        health_score = 0
        if total_cells > 0:
            health_score = (stats['stable_count'] * 1.0 + stats['warning_count'] * 0.5) / total_cells
        
        health_status = 'critical' if health_score < 0.3 else \
                      'warning' if health_score < 0.6 else \
                      'good' if health_score < 0.8 else 'excellent'
        
        return {
            'health_score': health_score,
            'health_status': health_status,
            'mean_confidence': stats['mean_confidence'],
            'confidence_range': stats['max_confidence'] - stats['min_confidence'],
            'critical_ratio': stats['critical_count'] / total_cells if total_cells > 0 else 0
        }
    
    def _analyze_confidence_trends(self) -> Dict[str, Any]:
        '''Analyze confidence trends over monitoring window'''
        if len(self.confidence_history) < 3:
            return {'trend': 'insufficient_data'}
        
        # Calculate mean confidence over time
        mean_confidences = [scan['summary_stats'].get('mean_confidence', 0) 
                           for scan in self.confidence_history if scan['summary_stats']]
        
        if len(mean_confidences) < 2:
            return {'trend': 'insufficient_data'}
        
        # Simple trend analysis
        recent_mean = sum(mean_confidences[-3:]) / min(3, len(mean_confidences))
        earlier_mean = sum(mean_confidences[:3]) / min(3, len(mean_confidences))
        
        trend_direction = 'improving' if recent_mean > earlier_mean else \
                         'degrading' if recent_mean < earlier_mean else 'stable'
        
        trend_magnitude = abs(recent_mean - earlier_mean)
        
        return {
            'trend': trend_direction,
            'magnitude': trend_magnitude,
            'recent_mean': recent_mean,
            'earlier_mean': earlier_mean,
            'confidence_history_length': len(mean_confidences)
        }
    
    def _generate_degradation_recommendations(self, degraded_cells: List[Dict[str, Any]]) -> List[str]:
        '''Generate recommendations for addressing degradation'''
        recommendations = []
        
        severe_degradation = [cell for cell in degraded_cells if cell['degradation_amount'] >= 0.25]
        if severe_degradation:
            recommendations.append("URGENT: Immediate reinforcement required for severely degraded cells")
        
        if len(degraded_cells) >= 3:
            recommendations.append("Consider system-wide stability analysis")
        
        recommendations.append("Implement targeted confidence boosting for degraded cells")
        recommendations.append("Monitor degradation patterns for systemic issues")
        
        return recommendations
