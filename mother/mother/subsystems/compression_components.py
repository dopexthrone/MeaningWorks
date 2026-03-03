from typing import List, Dict
from collections import Counter

class CompressionLoss:
    _losses: List[tuple[str, float]] = []
    @classmethod
    def reset(cls) -> None:
        cls._losses = []
    @classmethod
    def create_compression_loss(cls, category: str, severity: float) -> None:
        cls._losses.append((category, severity))
    @classmethod
    def get_compression_loss(cls) -> List[tuple[str, float]]:
        return cls._losses[:]
    @classmethod
    def update_compression_loss(cls, index: int, severity: float) -> None:
        if 0 &lt;= index &lt; len(cls._losses):
            category = cls._losses[index][0]
            cls._losses[index] = (category, severity)
    @classmethod
    def delete_compression_loss(cls, index: int) -> None:
        if 0 &lt;= index &lt; len(cls._losses):
            del cls._losses[index]

class EntityPayloadRedundancyPatterns:
    _patterns: List[str] = []
    @classmethod
    def reset(cls) -> None:
        cls._patterns = []
    @classmethod
    def create_entity_payload_redundancy_patterns(cls, patterns: List[str]) -> None:
        cls._patterns.extend(patterns)
    @classmethod
    def get_entity_payload_redundancy_patterns(cls) -> List[str]:
        return cls._patterns[:]
    @classmethod
    def update_entity_payload_redundancy_patterns(cls, index: int, pattern: str) -> None:
        if 0 &lt;= index &lt; len(cls._patterns):
            cls._patterns[index] = pattern
    @classmethod
    def delete_entity_payload_redundancy_patterns(cls, index: int) -> None:
        if 0 &lt;= index &lt; len(cls._patterns):
            del cls._patterns[index]

class RecursiveStructures:
    _structures: List[Dict] = []
    @classmethod
    def reset(cls) -> None:
        cls._structures = []
    @classmethod
    def create_recursive_structures(cls, struct: Dict) -> None:
        cls._structures.append(struct)
    @classmethod
    def get_recursive_structures(cls) -> List[Dict]:
        return cls._structures[:]
    @classmethod
    def update_recursive_structures(cls, index: int, struct: Dict) -&gt; None:
        if 0 &lt;= index &lt; len(cls._structures):
            cls._structures[index] = struct
    @classmethod
    def delete_recursive_structures(cls, index: int) -> None:
        if 0 &lt;= index &lt; len(cls._structures):
            del cls._structures[index]

class CompressionOptimization:
    def __init__(self):
        self.loss_manager = CompressionLoss
        self.redun_manager = EntityPayloadRedundancyPatterns
        self.rec_manager = RecursiveStructures
        from compression_improvement_system import CompressionMetrics
        self.metrics = CompressionMetrics()
    def analyze_redundancy(self, entity_payloads: List[Dict]) -> Dict:
        redundancy_patterns = self._detect_redundancy_patterns(entity_payloads)
        self.redun_manager.create_entity_payload_redundancy_patterns(redundancy_patterns)
        recursive_structs = self._detect_recursive_structures(entity_payloads)
        for struct in recursive_structs:
            self.rec_manager.create_recursive_structures(struct)
        for pattern in redundancy_patterns:
            self.loss_manager.create_compression_loss(&quot;entity_payload_redundancy&quot;, 0.4)
        return {
            &quot;redundancy_patterns&quot;: self.redun_manager.get_entity_payload_redundancy_patterns(),
            &quot;recursive_structures&quot;: self.rec_manager.get_recursive_structures(),
            &quot;compression_losses&quot;: self.loss_manager.get_compression_loss()
        }
    def _detect_redundancy_patterns(self, payloads: List[Dict]) -> List[str]:
        if not payloads:
            return []
        all_keys = []
        all_values = []
        for p in payloads:
            if isinstance(p, dict):
                all_keys.extend(p.keys())
                all_values.extend([str(v) for v in p.values()])
        key_counts = Counter(all_keys)
        value_counts = Counter(all_values)
        threshold = len(payloads) / 2
        redundant_keys = [k for k, c in key_counts.items() if c &gt; threshold]
        redundant_values = [v for v, c in value_counts.items() if c &gt; threshold and len(v) &gt; 2]
        return redundant_keys + redundant_values
    def _detect_recursive_structures(self, payloads: List[Dict]) -> List[Dict]:
        structs = []
        def get_depth(d: dict, depth: int = 0) -&gt; int:
            if not isinstance(d, dict):
                return depth
            max_sub = max((get_depth(v, depth + 1) for v in d.values()), default=depth)
            return max_sub
        for i, p in enumerate(payloads):
            if isinstance(p, dict):
                depth = get_depth(p)
                if depth &gt; 2:
                    structs.append({&quot;index&quot;: i, &quot;depth&quot;: depth, &quot;payload&quot;: p})
        return structs
    def optimize_payloads(self, analysis: Dict) -&gt; List[Dict]:
        current = self.metrics.get_compression_entity()
        improvement = min(10.0 * len(analysis.get(&quot;redundancy_patterns&quot;, [])), 30.0)
        new_value = min(current + improvement, 50.0)
        self.metrics.update_compression_entity(new_value)
        return []

class StateMonitor:
    def __init__(self):
        self.optimization = CompressionOptimization()
        self.phase = &quot;init&quot;
    def poll_discovery(self) -&gt; Dict:
        discoveries = len(self.optimization.redun_manager.get_entity_payload_redundancy_patterns())
        confidence = self.optimization.metrics.get_compression_entity() / 100.0
        conflicts = len(self.optimization.loss_manager.get_compression_loss())
        unknowns = len(self.optimization.rec_manager.get_recursive_structures())
        return {
            &quot;discoveries&quot;: discoveries,
            &quot;confidence&quot;: round(confidence, 2),
            &quot;conflicts&quot;: conflicts,
            &quot;unknowns&quot;: unknowns,
            &quot;phase&quot;: self.phase
        }
    def update_phase(self, phase: str) -&gt; None:
        self.phase = phase