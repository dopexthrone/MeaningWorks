"""
Motherlabs Corpus - Local storage and indexing for compilations.

Derived from: PROJECT-PLAN.md Phase 3.6

Responsibilities:
- Store compilations in ~/motherlabs/corpus/
- Index: compilation metadata (input, domain, timestamp, components count)
- Query: list compilations by domain
- Re-export: read old compilation as input to new compilation

This is BASIC corpus integration. Full query interface deferred to Phase 4.
"""

import json
import os
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


@dataclass
class CompilationRecord:
    """
    Metadata record for a stored compilation.

    Derived from: PROJECT-PLAN.md Phase 3.6
    """
    id: str                           # Unique hash of input
    input_text: str                   # Original user input
    domain: str                       # Extracted domain
    timestamp: str                    # ISO8601
    components_count: int             # Number of components in blueprint
    insights_count: int               # Number of insights extracted
    success: bool                     # Did verification pass?
    file_path: str                    # Path to full context graph
    # Provider tuning fields (P2)
    provider: str = "unknown"         # LLM provider used
    model: str = "unknown"            # Model name used
    stage_timings: Optional[Dict[str, float]] = None   # Time per stage in seconds
    retry_counts: Optional[Dict[str, int]] = None      # Retries per stage
    # Process telemetry fields (Phase 12.2a)
    dialogue_turns: Optional[int] = None
    confidence_trajectory: Optional[List[float]] = None    # per-turn overall confidence
    message_type_counts: Optional[Dict[str, int]] = None   # {PROPOSITION: N, CHALLENGE: M, ...}
    conflict_count: Optional[int] = None                    # total conflicts detected
    structural_conflicts_resolved: Optional[int] = None     # resolved by reframing
    unknown_count: Optional[int] = None                     # unknowns at end
    dialogue_depth_config: Optional[Dict[str, int]] = None  # {min_turns, min_insights, max_turns}
    # Phase 16: Lineage tracking
    parent_id: Optional[str] = None                          # ID of parent compilation (if edited variant)
    edit_operations: Optional[List[Dict[str, Any]]] = None   # Edit ops applied to create this variant
    # Corpus quality weighting
    verification_score: Optional[float] = None               # Average of completeness + consistency (0-100)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompilationRecord":
        # Handle backward compatibility - old records may not have new fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


# Domain keyword → canonical category mapping.
# Intent agent produces fine-grained strings like "Software architecture/design automation".
# Normalize these to the 4 canonical adapter domains so corpus samples accumulate.
_DOMAIN_KEYWORDS: List[tuple] = [
    # Most specific multi-word patterns first to avoid partial matches
    ("multi-instance", "agent_system"),
    ("multi-agent", "agent_system"),
    ("multi agent", "agent_system"),
    ("service mesh", "api"),
    ("microservice", "api"),
    ("data pipeline", "process"),
    # Single-word agent patterns
    ("autonomous", "agent_system"),
    # API patterns
    ("graphql", "api"),
    ("webhook", "api"),
    ("rest", "api"),
    # Process patterns — check AFTER software patterns to avoid "software automation" → process
    ("workflow", "process"),
    ("orchestration", "process"),
    ("etl", "process"),
    # Software-specific patterns that should NOT fall through to process
    ("software", "software"),
    ("architecture", "software"),
    ("compilation", "software"),
    ("compiler", "software"),
    ("developer", "software"),
    ("development", "software"),
    ("design", "software"),
    ("tooling", "software"),
    ("code", "software"),
    # Agent patterns (less specific — after software)
    ("agent", "agent_system"),
    # API patterns (less specific)
    ("api", "api"),
    ("endpoint", "api"),
    # Process patterns (less specific — check last to avoid over-matching)
    ("pipeline", "process"),
    ("automation", "process"),
    ("process", "process"),
    ("build", "process"),
    ("system", "software"),
]


def _normalize_domain(raw_domain: str) -> str:
    """Normalize a free-form domain string to one of the 4 canonical adapter categories.

    Canonical categories: software, api, process, agent_system.
    Falls back to 'software' for unrecognized domains.
    """
    if not raw_domain or raw_domain == "unknown":
        return "software"
    lowered = raw_domain.lower()
    for keyword, canonical in _DOMAIN_KEYWORDS:
        if keyword in lowered:
            return canonical
    return "software"


class Corpus:
    """
    Local corpus storage for Motherlabs compilations.

    Derived from: PROJECT-PLAN.md Phase 3.6

    Storage structure:
    ~/motherlabs/corpus/
    ├── index.json           # Metadata index for all compilations
    ├── {id}/
    │   ├── context-graph.json
    │   ├── blueprint.json
    │   └── trace.md
    """

    DEFAULT_PATH = Path.home() / "motherlabs" / "corpus"

    def __init__(self, corpus_path: Optional[Path] = None):
        """
        Initialize corpus.

        Args:
            corpus_path: Custom path for corpus storage (default: ~/motherlabs/corpus/)
        """
        self.path = corpus_path or self.DEFAULT_PATH
        self.index_path = self.path / "index.json"
        self._ensure_corpus_exists()
        self._index: List[CompilationRecord] = self._load_index()

    def _ensure_corpus_exists(self):
        """Create corpus directory if it doesn't exist."""
        self.path.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> List[CompilationRecord]:
        """Load index from disk."""
        if self.index_path.exists():
            with open(self.index_path, "r") as f:
                data = json.load(f)
                return [CompilationRecord.from_dict(r) for r in data]
        return []

    def _save_index(self):
        """Persist index to disk."""
        with open(self.index_path, "w") as f:
            json.dump([r.to_dict() for r in self._index], f, indent=2)

    def _generate_id(self, input_text: str) -> str:
        """Generate unique ID from input text."""
        return hashlib.sha256(input_text.encode()).hexdigest()[:12]

    def store(
        self,
        input_text: str,
        context_graph: Dict[str, Any],
        blueprint: Dict[str, Any],
        insights: List[str],
        success: bool,
        provider: str = "unknown",
        model: str = "unknown",
        stage_timings: Optional[Dict[str, float]] = None,
        retry_counts: Optional[Dict[str, int]] = None,
        # Phase 12.2a: Process telemetry
        dialogue_turns: Optional[int] = None,
        confidence_trajectory: Optional[List[float]] = None,
        message_type_counts: Optional[Dict[str, int]] = None,
        conflict_count: Optional[int] = None,
        structural_conflicts_resolved: Optional[int] = None,
        unknown_count: Optional[int] = None,
        dialogue_depth_config: Optional[Dict[str, int]] = None,
        # Phase 16: Lineage tracking
        parent_id: Optional[str] = None,
        edit_operations: Optional[List[Dict[str, Any]]] = None,
        # Corpus quality weighting
        verification_score: Optional[float] = None,
        # L2 feedback loop (stored in SQLiteCorpus, accepted here for compat)
        corpus_feedback: Optional[str] = None,
    ) -> CompilationRecord:
        """
        Store a compilation in the corpus.

        Derived from: PROJECT-PLAN.md Phase 3.6

        Args:
            input_text: Original user input
            context_graph: Full context graph from compilation
            blueprint: Synthesized blueprint
            insights: List of extracted insights
            success: Whether verification passed
            provider: LLM provider used (e.g., "grok", "openai")
            model: Model name used (e.g., "grok-4-1-fast-reasoning")
            stage_timings: Time per stage in seconds
            retry_counts: Number of retries per stage

        Returns:
            CompilationRecord with storage metadata
        """
        # Generate ID and paths
        compilation_id = self._generate_id(input_text)
        compilation_dir = self.path / compilation_id
        compilation_dir.mkdir(exist_ok=True)

        # Extract and normalize domain from context graph.
        # Intent extracts fine-grained domain strings ("Software architecture/design automation").
        # Normalize to canonical adapter categories so build_domain_model() can accumulate
        # enough samples (min_samples=3) to produce patterns.
        intent = context_graph.get("known", {}).get("intent", {})
        raw_domain = intent.get("domain", "unknown")
        domain = _normalize_domain(raw_domain)

        # Write files
        context_path = compilation_dir / "context-graph.json"
        blueprint_path = compilation_dir / "blueprint.json"
        trace_path = compilation_dir / "trace.md"

        with open(context_path, "w") as f:
            json.dump(context_graph, f, indent=2)

        with open(blueprint_path, "w") as f:
            json.dump(blueprint, f, indent=2)

        with open(trace_path, "w") as f:
            f.write(f"# Compilation Trace\n\n")
            f.write(f"*ID: {compilation_id}*\n")
            f.write(f"*Generated: {datetime.now().isoformat()}*\n\n")
            f.write(f"## Input\n\n{input_text}\n\n")
            f.write(f"## Insights\n\n")
            for i, insight in enumerate(insights, 1):
                f.write(f"{i}. {insight}\n")

        # Create record
        record = CompilationRecord(
            id=compilation_id,
            input_text=input_text[:500],  # Truncate for index
            domain=domain,
            timestamp=datetime.now().isoformat(),
            components_count=len(blueprint.get("components", [])),
            insights_count=len(insights),
            success=success,
            file_path=str(compilation_dir),
            provider=provider,
            model=model,
            stage_timings=stage_timings,
            retry_counts=retry_counts,
            dialogue_turns=dialogue_turns,
            confidence_trajectory=confidence_trajectory,
            message_type_counts=message_type_counts,
            conflict_count=conflict_count,
            structural_conflicts_resolved=structural_conflicts_resolved,
            unknown_count=unknown_count,
            dialogue_depth_config=dialogue_depth_config,
            parent_id=parent_id,
            edit_operations=edit_operations,
            verification_score=verification_score,
        )

        # Update index (replace if exists)
        self._index = [r for r in self._index if r.id != compilation_id]
        self._index.append(record)
        self._save_index()

        return record

    def list_all(self) -> List[CompilationRecord]:
        """
        List all compilations in corpus.

        Derived from: PROJECT-PLAN.md Phase 3.6
        """
        return sorted(self._index, key=lambda r: r.timestamp, reverse=True)

    def list_by_domain(self, domain: str) -> List[CompilationRecord]:
        """
        List compilations filtered by domain.

        Derived from: PROJECT-PLAN.md Phase 3.6
        """
        return [r for r in self.list_all() if r.domain.lower() == domain.lower()]

    def get(self, compilation_id: str) -> Optional[CompilationRecord]:
        """Get compilation record by ID."""
        for record in self._index:
            if record.id == compilation_id:
                return record
        return None

    def load_context_graph(self, compilation_id: str) -> Optional[Dict[str, Any]]:
        """
        Load full context graph for a compilation.

        Derived from: PROJECT-PLAN.md Phase 3.6 (re-export capability)
        """
        record = self.get(compilation_id)
        if not record:
            return None

        context_path = Path(record.file_path) / "context-graph.json"
        if context_path.exists():
            with open(context_path, "r") as f:
                return json.load(f)
        return None

    def load_blueprint(self, compilation_id: str) -> Optional[Dict[str, Any]]:
        """Load blueprint for a compilation."""
        record = self.get(compilation_id)
        if not record:
            return None

        blueprint_path = Path(record.file_path) / "blueprint.json"
        if blueprint_path.exists():
            with open(blueprint_path, "r") as f:
                return json.load(f)
        return None

    def search(self, query: str) -> List[CompilationRecord]:
        """
        Search compilations by input text (basic substring match).

        Args:
            query: Search string

        Returns:
            Matching compilation records
        """
        query_lower = query.lower()
        return [
            r for r in self.list_all()
            if query_lower in r.input_text.lower() or query_lower in r.domain.lower()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get corpus statistics including suggestion readiness."""
        if not self._index:
            return {
                "total_compilations": 0,
                "domains": {},
                "success_rate": 0.0,
                "total_components": 0,
                "total_insights": 0,
                "domains_with_suggestions": []
            }

        domains: Dict[str, int] = {}
        domain_success: Dict[str, int] = {}
        for r in self._index:
            domains[r.domain] = domains.get(r.domain, 0) + 1
            if r.success:
                domain_success[r.domain] = domain_success.get(r.domain, 0) + 1

        successful = sum(1 for r in self._index if r.success)

        # Find domains with enough successful samples for suggestions
        min_samples = 3
        domains_with_suggestions = [
            domain for domain, success_count in domain_success.items()
            if success_count >= min_samples
        ]

        return {
            "total_compilations": len(self._index),
            "domains": domains,
            "success_rate": successful / len(self._index) if self._index else 0.0,
            "total_components": sum(r.components_count for r in self._index),
            "total_insights": sum(r.insights_count for r in self._index),
            "domains_with_suggestions": domains_with_suggestions
        }

    def get_provider_stats(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance statistics by provider.

        Derived from: NEXT-STEPS.md - Provider-Specific Tuning

        Args:
            provider: Optional filter for specific provider

        Returns:
            Dict mapping provider name to stats:
            {
                "grok": {
                    "total_compilations": 10,
                    "success_rate": 0.9,
                    "avg_synthesis_time": 12.5,
                    "avg_retries": 0.5,
                    "models_used": ["grok-4-1-fast-reasoning"]
                },
                ...
            }
        """
        # Group records by provider
        provider_records: Dict[str, List[CompilationRecord]] = {}
        for r in self._index:
            p = r.provider
            if provider and p != provider:
                continue
            if p not in provider_records:
                provider_records[p] = []
            provider_records[p].append(r)

        result: Dict[str, Any] = {}
        for p, records in provider_records.items():
            successful = sum(1 for r in records if r.success)
            models = list(set(r.model for r in records))

            # Calculate average synthesis time if available
            synthesis_times = [
                r.stage_timings.get("synthesis", 0)
                for r in records
                if r.stage_timings
            ]
            avg_synthesis = (
                sum(synthesis_times) / len(synthesis_times)
                if synthesis_times else 0.0
            )

            # Calculate average retries if available
            retry_counts = [
                r.retry_counts.get("synthesis", 0)
                for r in records
                if r.retry_counts
            ]
            avg_retries = (
                sum(retry_counts) / len(retry_counts)
                if retry_counts else 0.0
            )

            result[p] = {
                "total_compilations": len(records),
                "success_rate": successful / len(records) if records else 0.0,
                "avg_synthesis_time": round(avg_synthesis, 2),
                "avg_retries": round(avg_retries, 2),
                "models_used": models
            }

        return result

    def export_for_recompile(self, compilation_id: str) -> Optional[str]:
        """
        Export a prior compilation as input for re-compilation.

        Derived from: PROJECT-PLAN.md Phase 3.6 (re-export capability)

        This enables dogfooding: take a prior compilation's insights
        and use them to prime a new compilation.

        Returns:
            Formatted string suitable as input to a new compile() call
        """
        context = self.load_context_graph(compilation_id)
        if not context:
            return None

        record = self.get(compilation_id)
        if not record:
            return None

        # Build re-compile prompt from prior context
        insights = context.get("insights", [])
        known = context.get("known", {})
        intent = known.get("intent", {})

        output = f"""Prior compilation: {record.id}
Domain: {record.domain}

Original input: {record.input_text}

Core need identified: {intent.get('core_need', 'unknown')}

Key insights from prior compilation:
"""
        for i, insight in enumerate(insights[:10], 1):  # Top 10 insights
            output += f"  {i}. {insight}\n"

        output += f"""
Components discovered: {record.components_count}

Build on this prior understanding. Refine, extend, or challenge these insights.
"""
        return output

    # =========================================================================
    # CORPUS-DRIVEN SUGGESTIONS
    # Derived from: NEXT-STEPS.md - Corpus Learning
    # =========================================================================

    def _analyze_component_frequencies(
        self,
        records: List[CompilationRecord]
    ) -> Dict[str, float]:
        """
        Analyze component frequencies across compilations.

        Args:
            records: List of compilation records to analyze

        Returns:
            Dict mapping component name -> frequency (0.0 to 1.0)
        """
        if not records:
            return {}

        component_counts: Dict[str, int] = {}
        total = len(records)

        for record in records:
            blueprint = self.load_blueprint(record.id)
            if not blueprint:
                continue

            # Get unique component names from this blueprint
            seen_in_blueprint: set = set()
            for comp in blueprint.get("components", []):
                name = comp.get("name")
                if name and name not in seen_in_blueprint:
                    seen_in_blueprint.add(name)
                    component_counts[name] = component_counts.get(name, 0) + 1

        # Convert counts to frequencies
        return {
            name: count / total
            for name, count in component_counts.items()
        }

    def _analyze_relationship_frequencies(
        self,
        records: List[CompilationRecord]
    ) -> Dict[tuple, float]:
        """
        Analyze relationship frequencies across compilations.

        Args:
            records: List of compilation records to analyze

        Returns:
            Dict mapping (from, to, type) tuple -> frequency (0.0 to 1.0)
        """
        if not records:
            return {}

        rel_counts: Dict[tuple, int] = {}
        total = len(records)

        for record in records:
            blueprint = self.load_blueprint(record.id)
            if not blueprint:
                continue

            # Get unique relationships from this blueprint
            seen_in_blueprint: set = set()
            for rel in blueprint.get("relationships", []):
                key = (rel.get("from"), rel.get("to"), rel.get("type"))
                if key not in seen_in_blueprint and all(key):
                    seen_in_blueprint.add(key)
                    rel_counts[key] = rel_counts.get(key, 0) + 1

        # Convert counts to frequencies
        return {
            rel: count / total
            for rel, count in rel_counts.items()
        }

    def get_domain_suggestions(
        self,
        domain: str,
        min_frequency: float = 0.6,
        min_samples: int = 3
    ) -> Dict[str, Any]:
        """
        Get suggested canonical components for a domain based on corpus patterns.

        Derived from: NEXT-STEPS.md - Corpus-Driven Canonical Sets

        Args:
            domain: Target domain to find suggestions for
            min_frequency: Minimum % of successful compilations a component
                          must appear in (default 0.6 = 60%)
            min_samples: Minimum successful compilations required before
                        making suggestions (default 3)

        Returns:
            {
                "domain": str,
                "sample_size": int,
                "suggested_components": List[str],
                "suggested_relationships": List[tuple],
                "component_frequencies": Dict[str, float],
                "has_suggestions": bool
            }
        """
        # Get successful compilations in this domain
        records = self.list_by_domain(domain)
        successful = [r for r in records if r.success]

        # Check minimum sample size
        if len(successful) < min_samples:
            return {
                "domain": domain,
                "sample_size": len(successful),
                "suggested_components": [],
                "suggested_relationships": [],
                "component_frequencies": {},
                "has_suggestions": False,
                "reason": f"Insufficient samples ({len(successful)}/{min_samples})"
            }

        # Analyze component frequencies
        comp_frequencies = self._analyze_component_frequencies(successful)

        # Filter by threshold
        suggested_components = [
            name for name, freq in comp_frequencies.items()
            if freq >= min_frequency
        ]
        # Sort by frequency (highest first)
        suggested_components.sort(
            key=lambda n: comp_frequencies.get(n, 0),
            reverse=True
        )

        # Analyze relationship frequencies (lower threshold)
        rel_frequencies = self._analyze_relationship_frequencies(successful)
        rel_threshold = min_frequency * 0.67  # ~40% if min_frequency is 60%
        suggested_relationships = [
            rel for rel, freq in rel_frequencies.items()
            if freq >= rel_threshold
        ]

        return {
            "domain": domain,
            "sample_size": len(successful),
            "suggested_components": suggested_components,
            "suggested_relationships": suggested_relationships,
            "component_frequencies": comp_frequencies,
            "has_suggestions": len(suggested_components) > 0
        }
