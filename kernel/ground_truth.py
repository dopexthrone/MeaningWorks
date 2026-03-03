"""
kernel/ground_truth.py — Load the MTH-USM-001 ground truth map.

96 nodes across 16 layers. The self-describing semantic map:
"the map of the map." Every vector is a node.

Source: ChatGPT ideation session 2026-02-18, simulation-verified.
"""

from __future__ import annotations

from kernel.cell import Cell, FillState, parse_postcode
from kernel.grid import Grid
from kernel.ops import fill, connect

MAP_ID = "MTH-USM-001"
MAP_NAME = "Universal Semantic Map — Self-Description"
MAP_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Raw node definitions: (postcode, fill_state, confidence, primitive, content,
#                        connections)
# Where postcode uses shared postcodes for multiple primitives in same cell,
# we differentiate by primitive name.
# ---------------------------------------------------------------------------

_NODES = [
    # =========================================================================
    # LAYER: INT — INTENT (5 nodes)
    # =========================================================================
    ("INT.SEM.ECO.WHY.SFT", "F", 0.99, "system_purpose",
     "Mother translates human intent into structured coordinate manifests for any complex adaptive system",
     ("INT.SEM.ECO.WHAT.SFT", "AGN.ORC.ECO.WHO.SFT")),

    ("INT.SEM.ECO.WHO.SFT", "F", 0.99, "primary_actors",
     "Mother (compiler), Map Agent (populator), Claude Code (executor), Human (curator)",
     ("IDN.ACT.ECO.WHO.SFT",)),

    ("INT.SEM.ECO.WHAT.SFT", "F", 0.99, "outcome",
     "exported manifest: coordinate map + provenance + agent sequence + simulation report. Claude Code executes with zero ambiguity",
     ("EXC.FNC.ECO.HOW.SFT",)),

    ("INT.SEM.ECO.HOW.SFT", "F", 0.97, "success_condition",
     "human pre-build time < 10 minutes, post-build manual intervention = 0, map never hits ceiling — extensible on all axes",
     ("CTR.CFG.ECO.HOW.SFT",)),

    ("INT.SEM.ECO.IF.SFT", "F", 0.95, "ceiling_condition",
     "ceiling hit IF: new domain requires schema change. prevented BY: domain as 5th axis (extensible), layer/concern/scope all extensible",
     ()),

    # =========================================================================
    # LAYER: SEM — SEMANTIC (6 nodes)
    # =========================================================================
    ("SEM.SEM.ECO.WHAT.SFT", "F", 0.99, "semantic_map",
     "sparse tensor of coordinate nodes. each node: primitive + postcode + connections + fill_state + confidence + provenance. fractal: same schema at every zoom level",
     ("STR.ENT.ECO.WHAT.SFT",)),

    ("SEM.SEM.ECO.HOW.SFT", "F", 0.97, "compilation_flow",
     "human intent → intent_contract (human approves) → coordinate_map (agents populate) → simulation (verified clean) → manifest (Claude Code executes) → observations (feed back to map)",
     ("EXC.FNC.ECO.HOW.SFT",)),

    ("SEM.SEM.APP.HOW.SFT", "F", 0.97, "gradient_descent_navigation",
     "high uncertainty nodes resolved first. confidence < 0.60 → human escalation. confidence 0.60-0.85 → emergence resolves. confidence > 0.85 → auto-promote",
     ("AGN.AGT.CMP.WHO.SFT",)),

    ("SEM.SEM.APP.HOW.SFT", "F", 0.95, "fractal_property",
     "every node contains same 4-axis structure. zoom in → same schema at higher resolution. stop condition: fill_score = 1.0, no children",
     ("EXC.FNC.CMP.HOW.SFT",)),

    ("SEM.SEM.ECO.WHY.ORG", "F", 0.90, "org_semantic_layer",
     "same coordinate schema applies to org design. domain = ORG, vocabulary changes, axes identical. team = entity, process = function, role = actor",
     ("STR.ENT.CMP.WHAT.ORG",)),

    ("SEM.SEM.ECO.WHY.COG", "F", 0.88, "cognitive_semantic_layer",
     "same coordinate schema applies to AI cognition. domain = COG, belief = state, reasoning = function, attention = resource, memory = persistence layer",
     ("STR.ENT.CMP.WHAT.COG",)),

    # =========================================================================
    # LAYER: STR — STRUCTURE (14 nodes)
    # =========================================================================
    ("STR.ENT.ECO.WHAT.SFT", "F", 0.99, "entity.node",
     "core data structure: id, postcode, primitive, description, fill_state, confidence, version, lock, provenance, connections, constraints, observations, children",
     ()),

    ("STR.ENT.ECO.WHAT.SFT", "F", 0.99, "entity.map",
     "collection of nodes, append-only. one map per blueprint. nodes never deleted — superseded by version",
     ("STR.REL.ECO.WHAT.SFT",)),

    ("STR.ENT.ECO.WHAT.SFT", "F", 0.99, "entity.intent_contract",
     "human-approved statement of purpose. fields: purpose, actors, outcome, constraints, out_of_scope, assumptions, approved, timestamp",
     ()),

    ("STR.ENT.ECO.WHAT.SFT", "F", 0.99, "entity.manifest",
     "export package for Claude Code. fields: all nodes + contracts + axioms + agent_sequence + simulation_report + escalation_log",
     ()),

    ("STR.ENT.ECO.WHAT.SFT", "F", 0.99, "entity.delta",
     "observation record written post-execution. fields: run_id, agent_id, expected, actual, confidence, anomalies[], timestamp",
     ()),

    ("STR.ENT.ECO.WHAT.SFT", "F", 0.99, "entity.event",
     "bus message between agents. fields: id, type, node_id, agent_id, run_id, payload, timestamp",
     ()),

    ("STR.ENT.ECO.WHAT.SFT", "F", 0.99, "entity.escalation_question",
     "human micro-question on low-confidence node. format: single question + max 3 options. never: context dump, ambiguous phrasing",
     ("EXC.FNC.STP.HOW.SFT",)),

    ("STR.ENT.ECO.WHAT.SFT", "F", 0.97, "entity.constraint",
     "rule a node must not violate. fields: id, rule, hard bool, source, propagates. hard = halt on violation. soft = flag and continue",
     ()),

    ("STR.REL.ECO.WHAT.SFT", "F", 0.99, "relation.map_to_nodes",
     "one map → many nodes. map is append-only container",
     ()),

    ("STR.REL.ECO.WHAT.SFT", "F", 0.99, "relation.node_to_children",
     "one parent → many children. fill_score = resolved / total. parent [F] only when all children [F]",
     ()),

    ("STR.REL.ECO.WHAT.SFT", "F", 0.97, "relation.node_connections",
     "nodes connected by postcode reference. connection = semantic wire. Claude Code implements connection as code dependency",
     ()),

    ("STR.ENM.ECO.WHAT.SFT", "F", 0.99, "enum.fill_state",
     "F filled | P partial | E empty | B blocked | Q quarantined | C candidate",
     ()),

    ("STR.ENM.ECO.WHAT.SFT", "F", 0.99, "enum.node_status",
     "authored | candidate | quarantined | promoted",
     ()),

    ("STR.ENM.ECO.WHAT.SFT", "F", 0.99, "enum.event_types",
     "NODE_CREATED | NODE_WRITTEN | NODE_VERIFIED | NODE_OBSERVED | NODE_SCANNED | NODE_PROMOTED | NODE_QUARANTINED | NODE_VIOLATED | NODE_PROPOSED | NODE_BLOCKED | RUN_STARTED | RUN_COMPLETE | CONSTRAINT_BREACH | ALREADY_PROCESSED | ESCALATION_REQUIRED | ESCALATION_ANSWERED | SIMULATION_STARTED | SIMULATION_COMPLETE | MANIFEST_EXPORTED | LOCK_ACQUIRED | LOCK_RELEASED | LOCK_TIMEOUT | VERSION_CONFLICT | DEPTH_LIMIT_REACHED",
     ()),

    ("STR.SCH.ECO.WHAT.SFT", "F", 0.99, "schema.coordinate",
     "postcode format: [LAYER].[CONCERN].[SCOPE].[DIM].[DOMAIN]. all fields: 3-char uppercase codes. extensible: new codes appended, never replaced",
     ()),

    # =========================================================================
    # LAYER: IDN — IDENTITY (5 nodes)
    # =========================================================================
    ("IDN.ACT.ECO.WHO.SFT", "F", 0.99, "actor.mother",
     "can: author nodes, compile maps, emit manifests, run simulations, generate intent contracts. cannot: self-approve candidates, override governor halt",
     ()),

    ("IDN.ACT.ECO.WHO.SFT", "F", 0.99, "actor.map_agent",
     "can: receive domain + intent, populate nodes, descend coordinate space, flag blocked. cannot: approve own candidates, bypass verifier. sub-agent of Mother",
     ()),

    ("IDN.ACT.ECO.WHO.SFT", "F", 0.99, "actor.claude_code",
     "can: read manifest, implement nodes, write observations to map. cannot: modify map structure, approve candidates, implement [B] or [Q] nodes. receives: manifest only",
     ()),

    ("IDN.ACT.ECO.WHO.SFT", "F", 0.99, "actor.human",
     "can: approve intent contracts, answer micro-questions, review simulation reports, promote quarantined nodes, add new domain vocabulary. cannot: skip simulation gate, bypass provenance check",
     ()),

    ("IDN.PRM.ECO.WHO.SFT", "F", 0.99, "permissions",
     "write map: mother, map_agent only. read map: all actors. approve nodes: human only. approve candidates: verifier + governor co-sign. execute: claude_code only. halt: governor only",
     ()),

    # =========================================================================
    # LAYER: AGN — AGENCY (8 nodes)
    # =========================================================================
    ("AGN.ORC.ECO.WHO.SFT", "F", 0.99, "orchestrator",
     "owns: agent sequence enforcement. routes: all events on event bus to correct agent. manages: parallel groups. escalates: low-confidence nodes to human queue",
     ()),

    ("AGN.AGT.CMP.WHO.SFT", "F", 0.99, "agent.author",
     "axiom: AX2 DESCENT. fills nodes, spawns children. stops when atomic reached",
     ()),

    ("AGN.AGT.CMP.WHO.SFT", "F", 0.99, "agent.verifier",
     "axiom: AX1 PROVENANCE. checks source_ref[], scores confidence. quarantines on failure",
     ()),

    ("AGN.AGT.CMP.WHO.SFT", "F", 0.99, "agent.observer",
     "axiom: AX3 FEEDBACK. computes delta (expected vs actual). writes to node.observations[]",
     ()),

    ("AGN.AGT.CMP.WHO.SFT", "F", 0.99, "agent.emergence",
     "axiom: AX4 EMERGENCE. scans for undiscovered nodes. proposes candidates — cannot self-approve",
     ()),

    ("AGN.AGT.CMP.WHO.SFT", "F", 0.99, "agent.governor",
     "axiom: AX5 CONSTRAINT. validates all constraints. halts all writers on hard breach",
     ()),

    ("AGN.MEM.CMP.WHO.SFT", "F", 0.95, "agent.memory",
     "persists: context across runs. stores: node.history[], run_log[], observation_log[]. provides: context lookup for all agents on request",
     ()),

    ("AGN.GOL.CMP.WHY.SFT", "F", 0.97, "goal.map_agent",
     "receive domain + intent → identify which layers activate → descend through coordinate space → populate until atomic → flag what cannot be resolved → return verified manifest",
     ()),

    # =========================================================================
    # LAYER: STA — STATE (5 nodes)
    # =========================================================================
    ("STA.BHV.CMP.WHEN.SFT", "F", 0.99, "node_state_machine",
     "EMPTY → WRITTEN → VERIFIED → OBSERVED → SCANNED → PROMOTED. any step → QUARANTINED (provenance fail). any step → BLOCKED (external dependency). QUARANTINED → PROMOTED (human override only). CANDIDATE → PROMOTED (verifier + governor)",
     ()),

    ("STA.TRN.STP.WHEN.SFT", "F", 0.99, "transition.written",
     "requires: lock acquired, provenance.source_ref[] present. action: increment version, append to run_history[]",
     ()),

    ("STA.TRN.STP.WHEN.SFT", "F", 0.99, "transition.verified",
     "requires: confidence > threshold, all source_refs resolved. action: mark verified, release for observation",
     ()),

    ("STA.TRN.STP.WHEN.SFT", "F", 0.99, "transition.promoted",
     "requires: all 5 agents passed, constraints satisfied. action: node live in map, children triggerable",
     ()),

    ("STA.TRN.STP.WHEN.SFT", "F", 0.99, "transition.quarantined",
     "trigger: provenance_check fails OR constraint hard-fail. action: halt agent, suspend children, escalate",
     ()),

    ("STA.TRN.STP.WHEN.SFT", "F", 0.97, "transition.blocked",
     "trigger: external dependency not yet resolved. action: node suspended, escalation_question generated. resume: when escalation answered",
     ()),

    # =========================================================================
    # LAYER: TME — TIME (5 nodes)
    # =========================================================================
    ("TME.TMO.CMP.WHEN.SFT", "F", 0.97, "timeout.lock",
     "default: 30 seconds. on expiry: auto-release lock, agent retries x 3",
     ()),

    ("TME.TMO.CMP.WHEN.SFT", "F", 0.97, "timeout.agent",
     "max execution per agent per node: 60 seconds. on expiry: NODE_VIOLATED, rollback, escalate",
     ()),

    ("TME.TMO.CMP.WHEN.SFT", "F", 0.95, "timeout.escalation",
     "escalation unanswered: 24 hours → reminder. 72 hours → blocking flag. map branch paused until answered",
     ()),

    ("TME.SCH.CMP.WHEN.SFT", "P", 0.75, "schedule.feedback_aggregation",
     "aggregate observations → map update. BLOCKED: run frequency decision required. options: [realtime] [batched hourly] [manual trigger]",
     ()),

    ("TME.VRS.ECO.WHEN.SFT", "F", 0.97, "versioning",
     "every node write increments node.version. map maintains full history — nothing deleted. any version replayable from node.run_history[]",
     ()),

    # =========================================================================
    # LAYER: EXC — EXECUTION (14 nodes)
    # =========================================================================
    ("EXC.FNC.ECO.HOW.SFT", "F", 0.99, "fn.compile",
     "input: raw human intent. steps: parse → contract → coordinate_map → populate_agents → simulation → export. output: manifest (verified, simulation-passed). gate: provenance_check + simulation_pass required",
     ()),

    ("EXC.FNC.ECO.HOW.SFT", "F", 0.99, "fn.agent_pipeline",
     "sequence: author → verifier → observer → emergence → governor. per node: strict order, no skipping. parallel: sibling nodes with no shared edges only. dedup: check agent_log before each step",
     ()),

    ("EXC.FNC.CMP.HOW.SFT", "F", 0.99, "fn.write_node",
     "1. check agent_log (dedup) 2. acquire lock 3. read current version 4. gate.provenance_check 5. write + increment version 6. emit NODE_WRITTEN 7. release lock. on fail: rollback → retry x 3 → quarantine",
     ()),

    ("EXC.FNC.CMP.HOW.SFT", "F", 0.97, "fn.descent_check",
     "fill_score = resolved_children / total_children. fill_score = 1.0 → mark [F]. fill_score < 1.0 → mark [P], spawn children. no children + no ambiguity → mark [F] atomic",
     ()),

    ("EXC.FNC.CMP.HOW.SFT", "F", 0.97, "fn.simulate",
     "traverse all edges in map. execute all transitions with synthetic data. collect: conflict_log[], gap_log[], assumption_log[], dead_end_log[], cycle_log[]. pass condition: all logs empty. fail: block export until resolved",
     ()),

    ("EXC.FNC.CMP.HOW.SFT", "F", 0.97, "fn.export_manifest",
     "requires: simulation_report.passed = true. packages: coordinate_map + intent_contract + axioms + agent_sequence + parallel_groups + blocked_nodes[] + escalation_log + simulation_report. emits: MANIFEST_EXPORTED",
     ()),

    ("EXC.FNC.STP.HOW.SFT", "F", 0.97, "fn.escalate",
     "trigger: confidence < 0.60 OR blocked node. format: single precise question + max 3 options. never: context dump, vague phrasing. on answer: re-score node, re-enter pipeline",
     ()),

    ("EXC.FNC.CMP.HOW.SFT", "F", 0.95, "fn.rollback",
     "trigger: write_node failure x 3. action: restore node to previous version, release lock, emit NODE_VIOLATED, log to run_history[]",
     ()),

    ("EXC.FNC.ECO.HOW.SFT", "F", 0.95, "fn.resolve_conflict",
     "trigger: version_mismatch on concurrent write. process: second writer re-reads, rebases, retries x 3. governor: merge | discard | escalate_human",
     ()),

    ("EXC.GTE.ECO.HOW.SFT", "F", 0.99, "gate.provenance_check",
     "every output node must have source_ref[] present. all source_refs must resolve to promoted node or human-approved statement on record. fail: quarantine + halt",
     ()),

    ("EXC.GTE.ECO.HOW.SFT", "F", 0.99, "gate.no_repeat",
     "before agent writes: check node.agent_log[]. agent_id already present → skip. emit ALREADY_PROCESSED",
     ()),

    ("EXC.GTE.ECO.HOW.SFT", "F", 0.99, "gate.simulation_pass",
     "manifest not exported until simulation passed. Claude Code never receives unverified blueprint",
     ()),

    ("EXC.GTE.CMP.HOW.SFT", "F", 0.97, "gate.depth_limit",
     "max depth: 7 (configurable, governor override required). depth exceeded → mark [B], emit DEPTH_LIMIT_REACHED",
     ()),

    ("EXC.LCK.CMP.HOW.SFT", "F", 0.99, "lock.node",
     "mutex per node. one writer at a time — no exceptions. ttl: 30s auto-release. prevents concurrent write corruption",
     ()),

    ("EXC.RTY.CMP.HOW.SFT", "F", 0.97, "retry.write",
     "max: 3 attempts. backoff: exponential (1s, 2s, 4s). final failure: quarantine + escalate",
     ()),

    # =========================================================================
    # LAYER: DAT — DATA (4 nodes)
    # =========================================================================
    ("DAT.FLW.ECO.HOW.SFT", "F", 0.97, "flow.intent_to_manifest",
     "raw string → intent_contract (structured) → coordinate_map (nodes) → simulation_report (verified) → manifest (packaged) → claude_code (executes) → observations (fed back)",
     ()),

    ("DAT.TRF.CMP.HOW.SFT", "F", 0.95, "transform.confidence_score",
     "input: provenance_strength, ambiguity_count, assumption_count, observation_history. output: float 0.0-1.0. formula: weighted average (weights configurable)",
     ()),

    ("DAT.TRF.CMP.HOW.SFT", "F", 0.95, "transform.fill_score",
     "input: node.children[], each child.status. output: float 0.0-1.0. formula: promoted_children / total_children",
     ()),

    ("DAT.COL.ECO.WHAT.SFT", "F", 0.97, "collection.map_store",
     "all nodes, append-only. indexed by: postcode, id, domain, layer, status. queryable by: postcode prefix, connection graph, fill_state, confidence range",
     ()),

    # =========================================================================
    # LAYER: SFX — SIDE EFFECTS (4 nodes)
    # =========================================================================
    ("SFX.WRT.ECO.HOW.SFT", "F", 0.99, "write.map_store",
     "append-only. every write versioned + timestamped. no hard deletes — supersede only",
     ("DAT.COL.ECO.WHAT.SFT",)),

    ("SFX.WRT.ECO.HOW.SFT", "F", 0.97, "write.run_log",
     "every run: run_id, timestamp, nodes_touched[], agents_run[], escalations[], simulation_result",
     ()),

    ("SFX.EMT.ECO.HOW.SFT", "F", 0.99, "emit.event_bus",
     "FIFO per node. agents never call each other directly. all inter-agent communication through bus only",
     ("STR.ENT.ECO.WHAT.SFT",)),

    ("SFX.RED.ECO.HOW.SFT", "F", 0.99, "read.map",
     "any actor can read any node. reads never blocked. reads never increment version. reads do not require lock",
     ()),

    # =========================================================================
    # LAYER: NET — NETWORK (4 nodes)
    # =========================================================================
    ("NET.FLW.ECO.HOW.SFT", "F", 0.95, "flow.event_bus",
     "all agent communication through event bus. bus is FIFO per node. bus is ordered — sequence preserved",
     ("SFX.EMT.ECO.HOW.SFT",)),

    ("NET.FLW.CMP.HOW.SFT", "P", 0.75, "flow.multi_agent_coordination",
     "agents on different processes/machines. shared map store as coordination point. no direct agent-to-agent calls. PARTIAL: distributed deployment spec pending",
     ()),

    ("NET.FLW.ECO.HOW.NET", "C", 0.70, "flow.cross_system_map",
     "CANDIDATE: maps from different systems connecting via shared postcode namespace. requires: governance for cross-map connections",
     ()),

    # =========================================================================
    # LAYER: RES — RESOURCE (4 nodes)
    # =========================================================================
    ("RES.MET.ECO.HOW_MUCH.SFT", "P", 0.75, "metric.map_size",
     "estimated nodes per app: 5K-200K. growth rate: function of domain complexity + run frequency + emergence rate. PARTIAL: sizing model not yet defined",
     ()),

    ("RES.MET.ECO.HOW_MUCH.SFT", "P", 0.75, "metric.agent_cost",
     "cost per node per run: author + verifier + observer + emergence + governor. PARTIAL: benchmark data needed",
     ()),

    ("RES.LMT.ECO.HOW_MUCH.SFT", "F", 0.90, "limit.depth",
     "max depth per branch: 7 (configurable). prevents infinite descent",
     ("EXC.GTE.CMP.HOW.SFT",)),

    ("RES.LMT.ECO.HOW_MUCH.SFT", "F", 0.90, "limit.retry",
     "max retries: 3 per operation. prevents infinite retry loops",
     ("EXC.RTY.CMP.HOW.SFT",)),

    # =========================================================================
    # LAYER: OBS — OBSERVABILITY (8 nodes)
    # =========================================================================
    ("OBS.LOG.ECO.HOW.SFT", "F", 0.97, "log.all_events",
     "every bus event logged. fields: event_type, node_id, agent_id, run_id, timestamp, payload. retention: indefinite (append-only)",
     ()),

    ("OBS.LOG.ECO.HOW.SFT", "F", 0.97, "log.agent_decisions",
     "every agent decision logged. fields: input_state, output_state, confidence_before, confidence_after, reasoning",
     ()),

    ("OBS.MET.ECO.HOW.SFT", "F", 0.95, "metric.fill_rate",
     "promoted_nodes / total_nodes. per: run, layer, domain, scope. tracks: map quality over time",
     ()),

    ("OBS.MET.ECO.HOW.SFT", "F", 0.95, "metric.escalation_rate",
     "escalations / total_nodes. high rate → intent contract quality low. indicator: human should refine intent upfront",
     ()),

    ("OBS.MET.ECO.HOW.SFT", "P", 0.75, "metric.confidence_distribution",
     "histogram of node confidence scores per run. tracks: map quality + learning trajectory",
     ()),

    ("OBS.MET.ECO.HOW.SFT", "P", 0.75, "metric.emergence_rate",
     "candidate_nodes_proposed / total_nodes. tracks: how much map self-discovers vs authored",
     ()),

    ("OBS.TRC.CMP.HOW.SFT", "F", 0.95, "trace.execution_path",
     "for any node: full path from Level 0 to VAL. for any run: sequence of agent steps. for any output: provenance chain to source",
     ()),

    ("OBS.ALT.ECO.HOW.SFT", "F", 0.90, "alert.constraint_breach",
     "immediate notification on CONSTRAINT_BREACH. blocks all map writes until resolved. human required",
     ()),

    # =========================================================================
    # LAYER: CTR — CONTROL (6 nodes)
    # =========================================================================
    ("CTR.CFG.ECO.HOW.SFT", "F", 0.99, "config.confidence_thresholds",
     "auto_promote: > 0.85. emergence_resolve: 0.60 - 0.85. escalate_human: < 0.60. configurable per: domain, layer, scope",
     ()),

    ("CTR.CFG.ECO.HOW.SFT", "F", 0.97, "config.depth_limits",
     "max_depth: 7 (default). max_width: configurable per layer. parallel_limit: sibling nodes, no shared edges",
     ()),

    ("CTR.CFG.ECO.HOW.SFT", "F", 0.97, "config.retry_policy",
     "max_retries: 3. backoff: exponential. final_failure: quarantine + escalate",
     ()),

    ("CTR.PLY.ECO.HOW.SFT", "F", 0.99, "policy.append_only",
     "map is append-only. no node deleted — only superseded by new version. enables: full history, replay, audit",
     ()),

    ("CTR.PLY.ECO.HOW.SFT", "F", 0.99, "policy.simulation_gate",
     "manifest not exported unless simulation passed. Claude Code never receives broken blueprint",
     ()),

    ("CTR.PLY.ECO.HOW.SFT", "F", 0.97, "policy.extension_rule",
     "ADD new axis value: append + assign new code. NEVER: reuse code, rename code, change axis order. this policy is what prevents the ceiling",
     ()),

    # =========================================================================
    # LAYER: EMG — EMERGENCE (3 nodes)
    # =========================================================================
    ("EMG.CND.ECO.WHAT.SFT", "F", 0.90, "emergence.cross_domain_connection",
     "connections between SFT and ORG domains. e.g. software booking fn ↔ org approval process. surfaces automatically via shared postcode prefix",
     ("NET.FLW.ECO.HOW.NET",)),

    ("EMG.CND.ECO.WHAT.SFT", "C", 0.72, "candidate.map_as_agent",
     "CANDIDATE: the map itself as an autonomous agent. observes its own fill state. proposes its own next descent targets. self-organizes population order by gradient",
     ("AGN.AGT.CMP.WHO.SFT",)),

    ("EMG.CND.ECO.WHAT.SFT", "C", 0.68, "candidate.cross_blueprint_learning",
     "CANDIDATE: observations from blueprint A inform confidence scores in blueprint B. shared domain vocabulary bootstraps new blueprints. requires: governed postcode namespace",
     ()),

    # =========================================================================
    # LAYER: MET — META (4 nodes)
    # =========================================================================
    ("MET.INT.ECO.WHY.SFT", "F", 0.99, "meta.self_description",
     "this map describes the system that builds maps. it is itself a valid map. it compiles itself via the same pipeline",
     ("INT.SEM.ECO.WHY.SFT",)),

    ("MET.VRS.ECO.WHEN.SFT", "F", 0.99, "meta.version",
     "export_id: MTH-USM-001. version: 1.0. timestamp: 2026-02-18. status: simulation_verified",
     ()),

    ("MET.CNS.ECO.IF.SFT", "F", 0.99, "meta.constraints",
     "1. every node traces to source (AX1). 2. fill until atomic (AX2). 3. every run writes delta (AX3). 4. agents propose, never self-approve (AX4). 5. Level 0 constraints propagate (AX5). 6. map append-only. 7. simulation gate before export. 8. schema extensible, never breaking",
     ()),

    ("MET.PRV.ECO.HOW.SFT", "F", 0.99, "meta.provenance",
     "this blueprint authored by: Mother / human conversation. source: semantic map ideation session 2026-02-18. validated by: self-compilation (map describes itself)",
     ()),
]

# Fill state lookup
_FILL_MAP = {"F": FillState.F, "P": FillState.P, "C": FillState.C, "B": FillState.B, "Q": FillState.Q, "E": FillState.E}


def load_ground_truth() -> Grid:
    """Load the MTH-USM-001 ground truth map into a Grid.

    96 nodes across 16 layers. Self-describing semantic map.
    The reference compilation that all future compilations can inherit from.

    Returns a fully populated Grid with all nodes, connections, and fill states.
    """
    grid = Grid()
    grid.set_intent(
        "Mother translates human intent into structured coordinate manifests for any complex adaptive system",
        "INT.SEM.ECO.WHY.SFT",
        "system_purpose",
    )

    # First pass: create all cells (handling duplicate postcodes via primitive suffix)
    seen_postcodes: dict[str, int] = {}
    cell_keys: list[str] = []  # track insertion order for connections

    for postcode_str, fill_str, confidence, primitive, content, connections in _NODES:
        pc = parse_postcode(postcode_str)
        fill = _FILL_MAP[fill_str]

        # Ensure layer is activated (direct add, no root cell creation)
        grid.activated_layers.add(pc.layer)

        # Handle duplicate postcodes (multiple primitives at same coordinate)
        # Grid stores by postcode key — for duplicates, we merge content
        pk = pc.key
        if pk in grid.cells:
            # Append to existing cell's content
            existing = grid.cells[pk]
            merged_content = existing.content + " | " + f"[{primitive}] {content}"
            merged_primitive = existing.primitive + "+" + primitive
            merged_confidence = max(existing.confidence, confidence)
            merged_source = existing.source + (f"ground_truth:{primitive}",)
            merged_connections = existing.connections + tuple(connections)

            # Determine merged fill state (worst wins)
            if fill == FillState.C or existing.fill == FillState.C:
                merged_fill = FillState.C
            elif fill == FillState.P or existing.fill == FillState.P:
                merged_fill = FillState.P
            else:
                merged_fill = fill

            grid.cells[pk] = Cell(
                postcode=pc,
                primitive=merged_primitive,
                content=merged_content,
                fill=merged_fill,
                confidence=merged_confidence,
                connections=merged_connections,
                parent=existing.parent,
                source=merged_source,
            )
        else:
            cell = Cell(
                postcode=pc,
                primitive=primitive,
                content=content,
                fill=fill,
                confidence=confidence,
                connections=tuple(connections),
                source=(f"ground_truth:{primitive}",),
            )
            grid.cells[pk] = cell

    return grid


def ground_truth_stats() -> dict[str, object]:
    """Return summary statistics about the ground truth map.

    Useful for quick verification without loading the full grid.
    """
    total = len(_NODES)
    filled = sum(1 for n in _NODES if n[1] == "F")
    partial = sum(1 for n in _NODES if n[1] == "P")
    candidate = sum(1 for n in _NODES if n[1] == "C")

    layers: set[str] = set()
    concerns: set[str] = set()
    for n in _NODES:
        parts = n[0].split(".")
        layers.add(parts[0])
        concerns.add(parts[1])

    return {
        "map_id": MAP_ID,
        "version": MAP_VERSION,
        "total_node_definitions": total,
        "filled": filled,
        "partial": partial,
        "candidate": candidate,
        "fill_rate": round(filled / total, 4) if total > 0 else 0.0,
        "layers_used": sorted(layers),
        "concerns_used": sorted(concerns),
        "layer_count": len(layers),
        "concern_count": len(concerns),
    }
