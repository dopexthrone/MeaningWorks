"""
Motherlabs Corpus Analysis - Structural pattern extraction from compilation history.

Phase 22: Corpus Pattern Synthesis

Upgrades the corpus from passive frequency counting to structural pattern extraction:
- Component archetypes with methods, constraints, relationships
- Relationship chains across compilations
- Vocabulary normalization
- Anti-pattern detection from low-verification compilations

This is Layer 2 activation: the compiler compiling its own compilation history
into reusable knowledge.

Architecture:
    SQLiteCorpus (storage) ← reads from ← CorpusAnalyzer (analysis)
    CorpusAnalyzer → feeds into → engine.compile() synthesis prompt SECTION 2c

Design decisions:
- Composition over inheritance: takes SQLiteCorpus instance, doesn't subclass
- All analysis is on-demand from stored blueprints/context_graphs (no schema changes)
- Deterministic: no LLM calls, pure structural extraction
- Provenance: all outputs carry source_ids tracing to specific compilations
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from persistence.sqlite_corpus import SQLiteCorpus


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class ComponentArchetype:
    """
    A recurring component pattern extracted from multiple compilations.

    Represents the "archetype" of a component that appears across compilations
    in a domain — capturing its canonical name, common methods, relationships,
    and constraints.
    """
    canonical_name: str                    # Most frequent variant (original casing)
    type: str                              # entity/process/agent/subsystem
    variants: List[str] = field(default_factory=list)
    frequency: float = 0.0                 # 0.0-1.0 across domain
    common_methods: List[str] = field(default_factory=list)
    common_relationships: List[Dict[str, Any]] = field(default_factory=list)
    common_constraints: List[Dict[str, Any]] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)


@dataclass
class RelationshipPattern:
    """
    A recurring relationship chain extracted from multiple compilations.

    Captures ordered component chains (A→B→C) that appear together across
    compilations in a domain.
    """
    components: List[str] = field(default_factory=list)
    relationships: List[Dict[str, str]] = field(default_factory=list)
    frequency: float = 0.0
    source_ids: List[str] = field(default_factory=list)


@dataclass
class DomainModel:
    """
    Complete structural model for a domain, synthesized from corpus history.

    Provenance: stratum 2 (corpus patterns) — derived from prior compilations,
    not from user input (stratum 0) or domain entailment (stratum 1).
    """
    domain: str = ""
    sample_size: int = 0
    archetypes: List[ComponentArchetype] = field(default_factory=list)
    relationship_patterns: List[RelationshipPattern] = field(default_factory=list)
    constraint_templates: List[Dict[str, Any]] = field(default_factory=list)
    vocabulary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    anti_patterns: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=lambda: {"stratum": 2, "source_ids": []})


# =============================================================================
# CORPUS ANALYZER
# =============================================================================


class CorpusAnalyzer:
    """
    Structural pattern extraction from compilation history.

    Takes a SQLiteCorpus instance (composition, not inheritance) and computes
    patterns on-demand from stored blueprints and context graphs.

    All analysis is deterministic — no LLM calls. Results carry provenance
    (source_ids) tracing to specific compilations.
    """

    def __init__(self, corpus: SQLiteCorpus):
        self.corpus = corpus

    # =========================================================================
    # NAME NORMALIZATION
    # =========================================================================

    @staticmethod
    def normalize_name(name: str) -> str:
        """
        Normalize a component name for comparison.

        'Auth Service' / 'AuthService' / 'auth_service' / 'auth-service' → 'authservice'

        Deterministic, no LLM. Strips spaces, underscores, hyphens, lowercases.
        """
        if not name:
            return ""
        return name.lower().replace(" ", "").replace("_", "").replace("-", "")

    # =========================================================================
    # CORE EXTRACTION
    # =========================================================================

    def _load_successful_blueprints(
        self, domain: str
    ) -> List[Tuple[str, Dict[str, Any], Optional[float]]]:
        """
        Load all successful blueprints for a domain.

        Returns list of (compilation_id, blueprint_dict, verification_score) tuples.
        verification_score may be None for older records.
        """
        records = self.corpus.list_by_domain(domain)
        successful = [r for r in records if r.success]
        results = []
        for record in successful:
            bp = self.corpus.load_blueprint(record.id)
            if bp:
                v_score = getattr(record, "verification_score", None)
                results.append((record.id, bp, v_score))
        return results

    def compute_adoption_weights(self, domain: str) -> Dict[str, float]:
        """
        Compute adoption-weighted multipliers for archetype names.

        Uses stored corpus_feedback from prior compilations to boost
        archetypes that are consistently adopted and demote ones that
        are consistently ignored.

        Returns dict mapping normalized archetype name → weight multiplier:
            adoption_rate >= 0.7 and avg trust >= 70 → 1.5x (strong signal)
            adoption_rate >= 0.4 → 1.0x (neutral)
            adoption_rate < 0.3 and times_suggested >= 3 → 0.5x (ignored)
            no data → 1.0x (no penalty for new patterns)
        """
        adoption_data = self.corpus.get_adoption_data(domain)
        if not adoption_data:
            return {}

        # archetype_name → {suggested: int, adopted: int, trust_scores: [float]}
        archetype_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"suggested": 0, "adopted": 0, "trust_scores": []}
        )

        for record in adoption_data:
            feedback = record["feedback"]
            v_score = record.get("verification_score", 0.0) or 0.0

            for name in feedback.get("suggestions_used", []):
                norm = self.normalize_name(name)
                stats = archetype_stats[norm]
                stats["suggested"] += 1
                stats["adopted"] += 1
                stats["trust_scores"].append(v_score)

            for name in feedback.get("suggestions_ignored", []):
                norm = self.normalize_name(name)
                archetype_stats[norm]["suggested"] += 1

        weights: Dict[str, float] = {}
        for norm_name, stats in archetype_stats.items():
            suggested = stats["suggested"]
            if suggested == 0:
                continue
            adopted = stats["adopted"]
            adoption_rate = adopted / suggested
            trust_scores = stats["trust_scores"]
            avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0.0

            if adoption_rate >= 0.7 and avg_trust >= 70:
                weights[norm_name] = 1.5
            elif adoption_rate >= 0.4:
                weights[norm_name] = 1.0
            elif suggested >= 3:
                weights[norm_name] = 0.5
            # else: no data or insufficient → default 1.0 (not stored)

        return weights

    def extract_archetypes(
        self,
        domain: str,
        min_frequency: float = 0.4,
        min_samples: int = 3,
        adoption_weights: Optional[Dict[str, float]] = None,
    ) -> List[ComponentArchetype]:
        """
        Extract component archetypes from corpus history for a domain.

        Algorithm:
        1. Load all successful blueprints for domain
        2. Normalize each component name → group variants
        3. Per group: count frequency, collect methods/relationships/constraints
        4. Filter: sub-item appears in ≥50% of instances → "common"
        5. canonical_name = most frequent variant (original casing)
        6. Return archetypes above min_frequency threshold
        """
        blueprints = self._load_successful_blueprints(domain)
        if len(blueprints) < min_samples:
            return []

        total = len(blueprints)

        # Build quality weight map: compilation_id → weight (0.0-1.0)
        # NULL verification_score → default 0.5; scores are 0-100 scale
        _quality_weights: Dict[str, float] = {}
        for comp_id, bp, v_score in blueprints:
            if v_score is not None:
                _quality_weights[comp_id] = max(0.0, min(v_score / 100.0, 1.0))
            else:
                _quality_weights[comp_id] = 0.5

        # Group components by normalized name
        # normalized_name → {variants: Counter, types: Counter, methods: {name: count},
        #                     relationships: {(norm_to, type): {to_variants: Counter, count}},
        #                     constraints: {(type, desc_norm): {desc: str, count}},
        #                     source_ids: set, instance_count: int, blueprints_seen: set,
        #                     weighted_score: float}
        groups: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "variants": Counter(),
            "types": Counter(),
            "methods": Counter(),
            "relationships": defaultdict(lambda: {"to_variants": Counter(), "count": 0}),
            "constraints": defaultdict(lambda: {"desc": "", "count": 0}),
            "source_ids": set(),
            "blueprints_seen": set(),
            "weighted_score": 0.0,
        })

        for comp_id, bp, v_score in blueprints:
            # Index components in this blueprint
            comp_names_in_bp: Dict[str, str] = {}  # original_name → normalized
            for comp in bp.get("components", []):
                name = comp.get("name", "")
                if not name:
                    continue
                norm = self.normalize_name(name)
                comp_names_in_bp[name] = norm

                g = groups[norm]
                g["variants"][name] += 1
                g["types"][comp.get("type", "entity")] += 1
                g["source_ids"].add(comp_id)
                if comp_id not in g["blueprints_seen"]:
                    g["weighted_score"] += _quality_weights.get(comp_id, 0.5)
                g["blueprints_seen"].add(comp_id)

                # Collect methods
                for method in comp.get("methods", []):
                    method_name = method.get("name", "") if isinstance(method, dict) else str(method)
                    if method_name:
                        g["methods"][method_name] += 1

                # Collect constraints applicable to this component
                for constraint in bp.get("constraints", []):
                    target = constraint.get("target", "")
                    if self.normalize_name(target) == norm:
                        c_type = constraint.get("type", "")
                        c_desc = constraint.get("description", "")
                        key = (c_type, self.normalize_name(c_desc))
                        entry = g["constraints"][key]
                        entry["desc"] = c_desc
                        entry["type"] = c_type
                        entry["count"] += 1

            # Collect relationships
            for rel in bp.get("relationships", []):
                from_name = rel.get("from", "")
                to_name = rel.get("to", "")
                rel_type = rel.get("type", "")
                if not from_name or not to_name:
                    continue
                from_norm = self.normalize_name(from_name)
                to_norm = self.normalize_name(to_name)
                if from_norm in groups:
                    key = (to_norm, rel_type)
                    entry = groups[from_norm]["relationships"][key]
                    entry["to_variants"][to_name] += 1
                    entry["count"] += 1

        # Build archetypes — frequency weighted by verification quality + adoption
        max_possible_weight = sum(_quality_weights.values())
        _adoption = adoption_weights or {}
        archetypes = []
        for norm, g in groups.items():
            instance_count = len(g["blueprints_seen"])
            # Quality-weighted frequency: sum of weights / max possible weight
            if max_possible_weight > 0:
                freq = g["weighted_score"] / max_possible_weight
            else:
                freq = instance_count / total
            # Apply adoption weight multiplier (L2 feedback loop)
            freq *= _adoption.get(norm, 1.0)
            if freq < min_frequency:
                continue

            # Canonical name = most frequent variant
            canonical = g["variants"].most_common(1)[0][0]
            comp_type = g["types"].most_common(1)[0][0]

            # Common methods: appear in ≥50% of instances
            threshold = instance_count * 0.5
            common_methods = [
                name for name, count in g["methods"].most_common()
                if count >= threshold
            ]

            # Common relationships: appear in ≥50% of instances
            common_rels = []
            for (to_norm, rel_type), entry in g["relationships"].items():
                if entry["count"] >= threshold:
                    to_canonical = entry["to_variants"].most_common(1)[0][0]
                    common_rels.append({
                        "to": to_canonical,
                        "type": rel_type,
                        "frequency": entry["count"] / instance_count,
                    })

            # Common constraints: appear in ≥50% of instances
            common_constraints = []
            for (c_type, _), entry in g["constraints"].items():
                if entry["count"] >= threshold:
                    common_constraints.append({
                        "type": entry.get("type", c_type),
                        "description": entry["desc"],
                        "frequency": entry["count"] / instance_count,
                    })

            archetypes.append(ComponentArchetype(
                canonical_name=canonical,
                type=comp_type,
                variants=list(g["variants"].keys()),
                frequency=freq,
                common_methods=common_methods,
                common_relationships=common_rels,
                common_constraints=common_constraints,
                source_ids=sorted(g["source_ids"]),
            ))

        # Sort by frequency descending
        archetypes.sort(key=lambda a: a.frequency, reverse=True)
        return archetypes

    def extract_relationship_patterns(
        self,
        domain: str,
        min_frequency: float = 0.3,
        min_samples: int = 3,
    ) -> List[RelationshipPattern]:
        """
        Extract recurring relationship patterns from corpus history.

        Algorithm:
        1. For each blueprint, extract all relationship triplets with normalized names
        2. Count frequency of each normalized triplet across compilations
        3. Detect chains: if A→B and B→C both appear in same blueprint, record [A,B,C]
        4. Return patterns above min_frequency
        """
        blueprints = self._load_successful_blueprints(domain)
        if len(blueprints) < min_samples:
            return []

        total = len(blueprints)

        # Track individual triplets per blueprint for chain detection
        # (from_norm, to_norm, type) → {source_ids, from_canonical, to_canonical}
        triplet_info: Dict[Tuple[str, str, str], Dict[str, Any]] = defaultdict(
            lambda: {"source_ids": set(), "from_variants": Counter(), "to_variants": Counter()}
        )
        # Per-blueprint adjacency for chain detection
        per_bp_adjacency: Dict[str, Dict[str, Set[str]]] = {}  # comp_id → {from_norm → {to_norm}}

        for comp_id, bp, _v_score in blueprints:
            adjacency: Dict[str, Set[str]] = defaultdict(set)
            for rel in bp.get("relationships", []):
                from_name = rel.get("from", "")
                to_name = rel.get("to", "")
                rel_type = rel.get("type", "")
                if not from_name or not to_name:
                    continue
                from_norm = self.normalize_name(from_name)
                to_norm = self.normalize_name(to_name)
                key = (from_norm, to_norm, rel_type)
                info = triplet_info[key]
                info["source_ids"].add(comp_id)
                info["from_variants"][from_name] += 1
                info["to_variants"][to_name] += 1
                adjacency[from_norm].add(to_norm)
            per_bp_adjacency[comp_id] = adjacency

        # Filter triplets by frequency
        patterns = []
        for (from_norm, to_norm, rel_type), info in triplet_info.items():
            freq = len(info["source_ids"]) / total
            if freq < min_frequency:
                continue
            from_canonical = info["from_variants"].most_common(1)[0][0]
            to_canonical = info["to_variants"].most_common(1)[0][0]
            patterns.append(RelationshipPattern(
                components=[from_canonical, to_canonical],
                relationships=[{
                    "from": from_canonical,
                    "to": to_canonical,
                    "type": rel_type,
                }],
                frequency=freq,
                source_ids=sorted(info["source_ids"]),
            ))

        # Detect chains: A→B and B→C in same blueprint, both above threshold
        # Build normalized triplet frequency map
        triplet_freq: Dict[Tuple[str, str], float] = {}
        for (from_norm, to_norm, _), info in triplet_info.items():
            pair = (from_norm, to_norm)
            freq = len(info["source_ids"]) / total
            # Keep highest freq if multiple rel types
            if pair not in triplet_freq or freq > triplet_freq[pair]:
                triplet_freq[pair] = freq

        # Find 3-node chains
        chain_counts: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
        for comp_id, adjacency in per_bp_adjacency.items():
            for a, a_targets in adjacency.items():
                for b in a_targets:
                    if b in adjacency:
                        for c in adjacency[b]:
                            if c != a:
                                chain_counts[(a, b, c)].add(comp_id)

        for (a, b, c), source_ids in chain_counts.items():
            freq = len(source_ids) / total
            if freq < min_frequency:
                continue
            # Get canonical names from any existing triplet info
            a_name = a
            b_name = b
            c_name = c
            for (fn, tn, _), info in triplet_info.items():
                if fn == a:
                    a_name = info["from_variants"].most_common(1)[0][0]
                if fn == b:
                    b_name = info["from_variants"].most_common(1)[0][0]
                if tn == c:
                    c_name = info["to_variants"].most_common(1)[0][0]
            patterns.append(RelationshipPattern(
                components=[a_name, b_name, c_name],
                relationships=[
                    {"from": a_name, "to": b_name, "type": "chain"},
                    {"from": b_name, "to": c_name, "type": "chain"},
                ],
                frequency=freq,
                source_ids=sorted(source_ids),
            ))

        patterns.sort(key=lambda p: p.frequency, reverse=True)
        return patterns

    def extract_constraint_templates(
        self,
        domain: str,
        min_samples: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Extract recurring constraint patterns from corpus history.

        Groups constraints by normalized target and type, tracks frequency.
        """
        blueprints = self._load_successful_blueprints(domain)
        if len(blueprints) < min_samples:
            return []

        total = len(blueprints)

        # (type, target_norm) → {description, source_ids, count}
        constraint_groups: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
            lambda: {"description": "", "source_ids": set(), "count": 0, "type": "", "target": ""}
        )

        for comp_id, bp, _v_score in blueprints:
            seen_in_bp: Set[Tuple[str, str]] = set()
            for constraint in bp.get("constraints", []):
                c_type = constraint.get("type", "")
                target = constraint.get("target", "")
                target_norm = self.normalize_name(target)
                key = (c_type, target_norm)
                if key in seen_in_bp:
                    continue
                seen_in_bp.add(key)
                entry = constraint_groups[key]
                entry["description"] = constraint.get("description", "")
                entry["type"] = c_type
                entry["target"] = target
                entry["source_ids"].add(comp_id)
                entry["count"] += 1

        # Filter: appeared in ≥2 compilations
        results = []
        for (c_type, target_norm), entry in constraint_groups.items():
            if entry["count"] < 2:
                continue
            results.append({
                "type": entry["type"],
                "target_pattern": entry["target"],
                "description": entry["description"],
                "frequency": entry["count"] / total,
                "source_ids": sorted(entry["source_ids"]),
            })

        results.sort(key=lambda r: r["frequency"], reverse=True)
        return results

    def extract_vocabulary(
        self,
        domain: str,
        min_samples: int = 3,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract domain vocabulary from context graphs.

        Collects terms from insights, component descriptions, and conflict resolutions.
        Returns terms appearing in ≥2 compilations.
        """
        records = self.corpus.list_by_domain(domain)
        successful = [r for r in records if r.success]
        if len(successful) < min_samples:
            return {}

        # term → {compilations: set, first_seen_id, definition}
        term_tracker: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"compilations": set(), "first_seen_id": "", "definition": ""}
        )

        for record in successful:
            cg = self.corpus.load_context_graph(record.id)
            if not cg:
                continue

            # Extract terms from insights
            for insight in cg.get("insights", []):
                if not isinstance(insight, str):
                    continue
                # Extract quoted terms or capitalized multi-word phrases
                words = insight.split()
                for i, word in enumerate(words):
                    # Quoted terms
                    stripped = word.strip('"\'.,;:!?()[]')
                    if len(stripped) >= 4 and stripped[0].isupper():
                        entry = term_tracker[stripped.lower()]
                        entry["compilations"].add(record.id)
                        if not entry["first_seen_id"]:
                            entry["first_seen_id"] = record.id
                        # Use the insight as a rough definition
                        if not entry["definition"]:
                            entry["definition"] = insight[:200]

            # Extract terms from component descriptions
            bp = self.corpus.load_blueprint(record.id)
            if bp:
                for comp in bp.get("components", []):
                    desc = comp.get("description", "")
                    if desc and len(desc) >= 10:
                        name = comp.get("name", "")
                        if name:
                            entry = term_tracker[name.lower()]
                            entry["compilations"].add(record.id)
                            if not entry["first_seen_id"]:
                                entry["first_seen_id"] = record.id
                            if not entry["definition"]:
                                entry["definition"] = desc[:200]

        # Filter: appeared in ≥2 compilations
        results = {}
        for term, info in term_tracker.items():
            if len(info["compilations"]) < 2:
                continue
            results[term] = {
                "definition": info["definition"],
                "frequency": len(info["compilations"]),
                "first_seen_id": info["first_seen_id"],
            }

        return results

    def detect_anti_patterns(
        self,
        min_samples: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Detect anti-patterns from low-verification compilations.

        Algorithm:
        1. Load all compilations (cross-domain)
        2. Load context_graphs to get verification scores
        3. For compilations with low verification (completeness < 70):
           - Extract component features: method_count=0, no relationships, no description
           - Count co-occurrence of these features with low scores
        4. Return features that correlate with low verification
        """
        all_records = self.corpus.list_all()
        if len(all_records) < min_samples:
            return []

        # Track feature co-occurrence with low scores
        feature_counts: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_score": 0.0, "source_ids": []}
        )

        low_score_count = 0

        for record in all_records:
            cg = self.corpus.load_context_graph(record.id)
            bp = self.corpus.load_blueprint(record.id)
            if not cg or not bp:
                continue

            # Get verification scores
            verification = cg.get("known", {}).get("verification", {})
            scores = verification.get("scores", {})
            completeness = scores.get("completeness", 100)

            if completeness >= 70:
                continue

            low_score_count += 1

            # Check for hollow components (0 methods)
            for comp in bp.get("components", []):
                methods = comp.get("methods", [])
                desc = comp.get("description", "")
                name = comp.get("name", "unknown")

                if len(methods) == 0:
                    f = feature_counts["hollow_component"]
                    f["count"] += 1
                    f["total_score"] += completeness
                    if record.id not in f["source_ids"]:
                        f["source_ids"].append(record.id)

                if not desc or len(desc) < 10:
                    f = feature_counts["missing_description"]
                    f["count"] += 1
                    f["total_score"] += completeness
                    if record.id not in f["source_ids"]:
                        f["source_ids"].append(record.id)

            # Check for orphan components (no relationships)
            rel_endpoints = set()
            for rel in bp.get("relationships", []):
                rel_endpoints.add(rel.get("from", ""))
                rel_endpoints.add(rel.get("to", ""))
            for comp in bp.get("components", []):
                name = comp.get("name", "")
                if name and name not in rel_endpoints:
                    f = feature_counts["orphan_component"]
                    f["count"] += 1
                    f["total_score"] += completeness
                    if record.id not in f["source_ids"]:
                        f["source_ids"].append(record.id)

        # Build results
        results = []
        for feature_name, info in feature_counts.items():
            if info["count"] < 2:
                continue
            avg_score = info["total_score"] / info["count"] if info["count"] > 0 else 0
            results.append({
                "description": feature_name,
                "avg_score": round(avg_score, 1),
                "count": info["count"],
                "source_ids": info["source_ids"],
            })

        results.sort(key=lambda r: r["count"], reverse=True)
        return results

    # =========================================================================
    # ORCHESTRATOR
    # =========================================================================

    def build_domain_model(
        self,
        domain: str,
        min_samples: int = 3,
    ) -> Optional[DomainModel]:
        """
        Build complete domain model from corpus history.

        Orchestrates all extraction methods and assembles into a DomainModel
        with stratum-2 provenance.

        Returns None if insufficient samples for the domain.
        """
        records = self.corpus.list_by_domain(domain)
        successful = [r for r in records if r.success]
        if len(successful) < min_samples:
            return None

        adoption_weights = self.compute_adoption_weights(domain)
        archetypes = self.extract_archetypes(
            domain, min_samples=min_samples, adoption_weights=adoption_weights,
        )
        relationship_patterns = self.extract_relationship_patterns(
            domain, min_samples=min_samples
        )
        constraint_templates = self.extract_constraint_templates(
            domain, min_samples=min_samples
        )
        vocabulary = self.extract_vocabulary(domain, min_samples=min_samples)
        anti_patterns = self.detect_anti_patterns(min_samples=min_samples)

        # Collect all source IDs
        all_source_ids: Set[str] = set()
        for a in archetypes:
            all_source_ids.update(a.source_ids)
        for p in relationship_patterns:
            all_source_ids.update(p.source_ids)
        for ct in constraint_templates:
            all_source_ids.update(ct.get("source_ids", []))
        for v_info in vocabulary.values():
            if v_info.get("first_seen_id"):
                all_source_ids.add(v_info["first_seen_id"])

        return DomainModel(
            domain=domain,
            sample_size=len(successful),
            archetypes=archetypes,
            relationship_patterns=relationship_patterns,
            constraint_templates=constraint_templates,
            vocabulary=vocabulary,
            anti_patterns=anti_patterns,
            provenance={
                "stratum": 2,
                "source_ids": sorted(all_source_ids),
            },
        )

    def summarize_pattern_health(self, domain: str) -> str:
        """
        Summarize pattern health for a domain based on adoption data.

        Classifies archetypes with 3+ suggestions into:
            Strong: adoption >= 0.6, avg trust >= 70
            Declining: adoption < 0.3, suggested 3+ times

        Returns compact string like "Strong: Auth, API Gateway. Declining: Logger."
        Returns empty string if < 3 compilations with feedback data.
        """
        adoption_data = self.corpus.get_adoption_data(domain)
        if len(adoption_data) < 3:
            return ""

        # Build per-archetype stats
        archetype_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"suggested": 0, "adopted": 0, "trust_scores": [], "display_name": ""}
        )

        for record in adoption_data:
            feedback = record["feedback"]
            v_score = record.get("verification_score", 0.0) or 0.0

            for name in feedback.get("suggestions_used", []):
                norm = self.normalize_name(name)
                stats = archetype_stats[norm]
                stats["suggested"] += 1
                stats["adopted"] += 1
                stats["trust_scores"].append(v_score)
                if not stats["display_name"]:
                    stats["display_name"] = name

            for name in feedback.get("suggestions_ignored", []):
                norm = self.normalize_name(name)
                stats = archetype_stats[norm]
                stats["suggested"] += 1
                if not stats["display_name"]:
                    stats["display_name"] = name

        strong = []
        declining = []

        for norm_name, stats in archetype_stats.items():
            suggested = stats["suggested"]
            if suggested < 3:
                continue
            adopted = stats["adopted"]
            adoption_rate = adopted / suggested
            trust_scores = stats["trust_scores"]
            avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0.0
            display = stats["display_name"]

            if adoption_rate >= 0.6 and avg_trust >= 70:
                strong.append(display)
            elif adoption_rate < 0.3:
                declining.append(display)

        if not strong and not declining:
            return ""

        parts = []
        if strong:
            parts.append(f"Strong: {', '.join(strong)}")
        if declining:
            parts.append(f"Declining: {', '.join(declining)}")
        return ". ".join(parts) + "."

    # =========================================================================
    # CROSS-DOMAIN
    # =========================================================================

    def find_isomorphisms(
        self,
        domain_a: str,
        domain_b: str,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Find structural isomorphisms between two domains.

        Compares component archetypes by normalized name and type.
        Returns overlap information.
        """
        archetypes_a = self.extract_archetypes(domain_a, min_frequency=0.3)
        archetypes_b = self.extract_archetypes(domain_b, min_frequency=0.3)

        if not archetypes_a or not archetypes_b:
            return {"shared_archetypes": [], "overlap_score": 0.0}

        # Build lookup by normalized name
        norm_a = {self.normalize_name(a.canonical_name): a for a in archetypes_a}
        norm_b = {self.normalize_name(b.canonical_name): b for b in archetypes_b}

        shared = []
        for norm_name in set(norm_a.keys()) & set(norm_b.keys()):
            a = norm_a[norm_name]
            b = norm_b[norm_name]
            # Type match bonus
            type_match = a.type == b.type
            shared.append({
                "name": a.canonical_name,
                "type_a": a.type,
                "type_b": b.type,
                "type_match": type_match,
                "frequency_a": a.frequency,
                "frequency_b": b.frequency,
            })

        total_unique = len(set(norm_a.keys()) | set(norm_b.keys()))
        overlap = len(shared) / total_unique if total_unique > 0 else 0.0

        return {
            "shared_archetypes": shared,
            "overlap_score": round(overlap, 3),
        }

    # =========================================================================
    # FORMAT FOR SYNTHESIS
    # =========================================================================

    @staticmethod
    def format_anti_pattern_warnings(domain_model: DomainModel) -> Optional[str]:
        """Format anti-patterns as terse warnings for the synthesis prompt.

        Returns None if no anti-patterns detected.
        Pure string function — no side effects.
        """
        if not domain_model.anti_patterns:
            return None

        _DESCRIPTIONS = {
            "hollow_component": "Hollow components (no methods): ensure every component has ≥1 method.",
            "missing_description": "Missing descriptions: ensure every component has a meaningful description.",
            "orphan_component": "Orphan components: ensure every component has ≥1 relationship.",
        }

        lines = ["WARNINGS from prior compilations:"]
        for ap in domain_model.anti_patterns:
            desc = ap.get("description", "")
            readable = _DESCRIPTIONS.get(desc, desc)
            count = ap.get("count", 0)
            source_count = len(ap.get("source_ids", []))
            lines.append(f"- {readable} (appeared in {source_count} build{'s' if source_count != 1 else ''})")
        return "\n".join(lines)

    @staticmethod
    def format_constraint_hints(domain_model: DomainModel) -> Optional[str]:
        """Format constraint templates as structural priors for synthesis.

        Returns None if no constraint templates found.
        Pure string function — no side effects.
        """
        if not domain_model.constraint_templates:
            return None

        lines = ["PROVEN CONSTRAINTS from prior builds:"]
        for ct in domain_model.constraint_templates:
            target = ct.get("target_pattern", "")
            c_type = ct.get("type", "")
            desc = ct.get("description", "")
            freq = ct.get("frequency", 0.0)
            freq_pct = int(freq * 100)
            label = f"{target}: {c_type}" if c_type else target
            if desc:
                label += f" — {desc}"
            lines.append(f"- {label} ({freq_pct}% of builds)")
        return "\n".join(lines)

    def format_corpus_patterns_section(
        self,
        domain_model: DomainModel,
    ) -> Optional[str]:
        """
        Format a DomainModel into a string for the synthesis prompt.

        Returns None if the model has no useful content to display.
        """
        if not domain_model.archetypes and not domain_model.relationship_patterns:
            return None

        lines = [
            f"DOMAIN: {domain_model.domain} (from {domain_model.sample_size} prior compilations)"
        ]

        # Archetypes
        if domain_model.archetypes:
            lines.append("")
            lines.append("COMPONENT ARCHETYPES (high confidence):")
            for arch in domain_model.archetypes:
                count = int(arch.frequency * domain_model.sample_size + 0.5)
                lines.append(
                    f"- {arch.canonical_name} [{arch.type}] "
                    f"(frequency: {arch.frequency:.2f}, seen in {count}/{domain_model.sample_size} compilations)"
                )
                if arch.common_methods:
                    methods_str = ", ".join(f"{m}()" for m in arch.common_methods)
                    lines.append(f"  common methods: {methods_str}")
                if arch.common_relationships:
                    for rel in arch.common_relationships:
                        lines.append(
                            f"  common relationships: -> {rel['to']} ({rel['type']})"
                        )
                if arch.common_constraints:
                    for con in arch.common_constraints:
                        lines.append(
                            f"  common constraints: {con['type']} on {con.get('description', '')}"
                        )

        # Relationship patterns
        rel_patterns = [
            p for p in domain_model.relationship_patterns
            if len(p.components) >= 3  # Only chains
        ]
        if rel_patterns:
            lines.append("")
            lines.append("RELATIONSHIP PATTERNS:")
            for pat in rel_patterns:
                chain_str = " -> ".join(pat.components)
                lines.append(
                    f"- {chain_str} (chain, frequency: {pat.frequency:.2f})"
                )

        # Vocabulary
        if domain_model.vocabulary:
            lines.append("")
            lines.append("VOCABULARY:")
            # Sort by frequency descending, take top 10
            sorted_vocab = sorted(
                domain_model.vocabulary.items(),
                key=lambda x: x[1].get("frequency", 0),
                reverse=True,
            )[:10]
            for term, info in sorted_vocab:
                definition = info.get("definition", "")
                freq = info.get("frequency", 0)
                if definition:
                    # Truncate long definitions
                    short_def = definition[:100]
                    if len(definition) > 100:
                        short_def += "..."
                    lines.append(f'- "{term}" = {short_def} ({freq} compilations)')

        return "\n".join(lines)
