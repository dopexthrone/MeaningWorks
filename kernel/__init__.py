"""
Motherlabs Semantic Map Kernel.

Primitives: Cell, Grid, fill, connect.
Navigator: next_cell, descend, detect_emergence.
Agents: memory, author, verifier, observer, emergence, governor.
Orchestrator: compile.
Nav: grid_to_nav, nav_to_grid, budget_nav.
Emission: emit, Manifest, Escalation.
Store: save_grid, load_grid, list_maps, delete_map.
Ground Truth: load_ground_truth, ground_truth_stats.
LLM Bridge: make_llm_function, parse_extractions.
Observer: record_observation, apply_observation, apply_batch, ObservationDelta, ObservationBatch.
Closed-Loop: closed_loop_gate, decode_blueprint, semantic_similarity, GateResult, CompressionLoss.
"""

from kernel.cell import Cell, FillState, Postcode, parse_postcode
from kernel.grid import Grid
from kernel.ops import fill, connect, FillResult, ConnectResult
from kernel.navigator import (
    next_cell,
    score_candidates,
    is_converged,
    should_descend,
    descend,
    descend_selective,
    detect_emergence,
    promote_emergence,
    ScoredCell,
    EmergenceSignal,
)
from kernel.agents import (
    memory,
    author,
    verifier,
    observer,
    emergence,
    governor,
    deduplicate_dimensional,
    compile,
    CompileConfig,
    CompileResult,
    SimulationResult,
    SimulationIssue,
    VerifyResult,
    ObserveResult,
    EmergeResult,
    DedupeResult,
    AuthorExtraction,
)
from kernel.nav import (
    grid_to_nav,
    nav_to_grid,
    budget_nav,
    estimate_tokens,
)
from kernel.emission import (
    emit,
    Manifest,
    ManifestNode,
    Escalation,
    extract_escalations,
)
from kernel.store import save_grid, load_grid, list_maps, delete_map, map_cell_count
from kernel.ground_truth import load_ground_truth, ground_truth_stats
from kernel.llm_bridge import make_llm_function, parse_extractions
from kernel.closed_loop import (
    closed_loop_gate,
    decode_blueprint,
    decode_blueprint_llm,
    semantic_similarity,
    detect_compression_losses,
    GateResult,
    CompressionLoss,
)
from kernel.observer import (
    record_observation,
    apply_observation,
    apply_batch,
    compute_confidence_drift,
    find_low_confidence_cells,
    find_anomalous_cells,
    ObservationDelta,
    ObservationBatch,
)

__all__ = [
    # Primitives
    "Cell", "FillState", "Postcode", "parse_postcode",
    "Grid",
    "fill", "connect", "FillResult", "ConnectResult",
    # Navigator
    "next_cell", "score_candidates", "is_converged",
    "should_descend", "descend", "descend_selective",
    "detect_emergence", "promote_emergence",
    "ScoredCell", "EmergenceSignal",
    # Agents
    "memory", "author", "verifier", "observer", "emergence", "governor",
    "deduplicate_dimensional",
    "compile", "CompileConfig", "CompileResult",
    "SimulationResult", "SimulationIssue",
    "VerifyResult", "ObserveResult", "EmergeResult", "DedupeResult",
    "AuthorExtraction",
    # Nav
    "grid_to_nav", "nav_to_grid", "budget_nav", "estimate_tokens",
    # Emission
    "emit", "Manifest", "ManifestNode", "Escalation", "extract_escalations",
    # Store
    "save_grid", "load_grid", "list_maps", "delete_map", "map_cell_count",
    # Ground Truth
    "load_ground_truth", "ground_truth_stats",
    # LLM Bridge
    "make_llm_function", "parse_extractions",
    # Closed-Loop
    "closed_loop_gate", "decode_blueprint", "decode_blueprint_llm",
    "semantic_similarity", "detect_compression_losses",
    "GateResult", "CompressionLoss",
    # Observer
    "record_observation", "apply_observation", "apply_batch",
    "compute_confidence_drift", "find_low_confidence_cells", "find_anomalous_cells",
    "ObservationDelta", "ObservationBatch",
]
