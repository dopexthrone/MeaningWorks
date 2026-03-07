# MOTHERLABS BLUEPRINTS — SINGLE SOURCE OF TRUTH
## Version: 1.0.0
## Date: 2026-03-07
## Status: Canonical reference. All other docs defer to this.

---

# 1. WHAT A BLUEPRINT IS

A blueprint is the compiled output of the Motherlabs semantic compiler.
It transforms natural language intent into a deterministic, navigable,
provenance-traced technical specification.

A blueprint is NOT a requirements doc, NOT a prompt, NOT a README.
It is a compiled artifact — the same way gcc produces machine code from C.

The compiler doesn't generate. It excavates.
The solution exists in probability space.
The compiler locates it through progressive constraint narrowing.

---

# 2. THE THREE LAYERS

Every Motherlabs blueprint has three layers. Most "blueprints" elsewhere
only have Layer 3. A Motherlabs blueprint requires all three.

```
┌───────────────────────────────────────────────────────┐
│  LAYER 3: COMPILED OUTPUTS                            │
│  entities, functions, rules, relationships,           │
│  state machines, gaps, silences                       │
│  → what coding agents consume                         │
├───────────────────────────────────────────────────────┤
│  LAYER 2: SEMANTIC MAP (coordinate DAG)               │
│  nodes with postcodes, filled by 5 map agents         │
│  Author → Verifier → Observer → Emergence → Governor  │
│  → the derivation path from seed to outputs           │
├───────────────────────────────────────────────────────┤
│  LAYER 1: PROVENANCE & VERIFICATION                   │
│  hashes, sources, failed paths, coverage,             │
│  rejected alternatives, human decisions               │
│  → the proof that this blueprint is valid             │
└───────────────────────────────────────────────────────┘
```

---

# 3. THE POSTCODE COORDINATE SYSTEM

Every concept in any domain gets a 5-axis address.
The address IS the content. An agent that knows the postcode
can excavate the node's meaning from the coordinate alone.

## 3.1 Format

```
[LAYER].[CONCERN].[SCOPE].[DIMENSION].[DOMAIN]

Example:  EXC.FNC.CMP.HOW.SFT
          ───  ───  ───  ───  ───
          │    │    │    │    └── DOMAIN:    software
          │    │    │    └─────── DIMENSION: how it works
          │    │    └──────────── SCOPE:     component level
          │    └───────────────── CONCERN:   function
          └────────────────────── LAYER:     execution
```

## 3.2 Axis 1 — LAYER (19 layers, extensible)

```
INT   intent          why — human goal
SEM   semantic        what it means — concepts
ORG   organization    how humans structure around it
COG   cognitive       how intelligence reasons about it
AGN   agency          who acts and decides
STR   structure       what exists — entities, types
STA   state           what phase — lifecycle
IDN   identity        who is allowed — auth, permissions
TME   time            when — schedule, sequence, timeout
EXC   execution       what happens — functions, logic
DAT   data            how information is shaped and moved
SFX   side effects    what reaches outside — IO, events
NET   network         how systems connect across boundaries
RES   resource        what it costs — compute, money, time
OBS   observability   what can be seen — logs, metrics
SEC   security        what protects — auth, validation, access, safeguards
CTR   control         what governs — config, limits, policies
EMG   emergence       what was discovered, not designed
MET   meta            the blueprint describing itself
```

## 3.2.1 Node References

Postcode identifies the coordinate cell.
Node reference identifies the specific primitive inside that cell.

```
[POSTCODE]/[NAME]

Example:
EXC.FNC.CMP.HOW.SFT/divide
CTR.PLY.APP.HOW.SFT/division_guard
```

Use postcode for:
- coordinate queries
- layer coverage
- prefix navigation

Use node reference for:
- clickable semantic links
- node-to-node connections
- read_before / read_after / warns links
- UI navigation to a specific compiled primitive

## 3.3 Axis 2 — CONCERN (40 types)

```
SEM semantic    ENT entity      REL relation    SCH schema
ENM enum        COL collection  BHV behavior    FNC function
TRG trigger     STP step        LGC logic       STA state
TRN transition  DAT data        TRF transform   FLW flow
ACT actor       PRM permission  AUT auth        SCO scope
MEM memory      PLN plan        DLG delegation  NGT negotiation
CNS consensus   HND handoff     ORC orchestration
OBS observation CTR control     CFG config      LMT limit
PLY policy      LOG log         MET metric      TRC trace
ALT alert       CND candidate   PRV provenance  VRS version
SIM simulation  RPT report      WRT write       EMT emit
RED read
```

## 3.4 Axis 3 — SCOPE (10 levels)

```
ECO   ecosystem       all systems, all apps
APP   application     one complete application
DOM   domain          one business domain
FET   feature         one capability
CMP   component       one module
FNC   function        one unit of work
STP   step            one action
OPR   operation       one computation
EXP   expression      one condition/transform
VAL   value           atomic — stop descent
```

## 3.5 Axis 4 — DIMENSION (8 questions)

```
WHY        causal / motivational
WHO        actors, agents, stakeholders
WHAT       identity, definition
WHEN       temporal, lifecycle
WHERE      spatial, boundary, location
HOW        mechanism, process
HOW_MUCH   quantity, cost, measurement
IF         conditional, constraint
```

## 3.6 Axis 5 — DOMAIN (10+, extensible)

```
SFT   software       ORG   organization    COG   cognitive
ECN   economic       PHY   physical        SOC   social
NET   network        EDU   educational     CRE   creative
LGL   legal
```

## 3.7 Extension Rules

```
ADD NEW LAYER:    append to Axis 1, assign 3-char code
ADD NEW CONCERN:  append to Axis 2
ADD NEW DOMAIN:   append to Axis 5, assign 3-char code
NEVER:            reuse a code, rename a code, change axis order
```

---

# 4. NODE SCHEMA

Every node in the semantic map conforms to this structure. No exceptions.

```typescript
interface Node {
  id:              string
  postcode:        string
  primitive:       string
  description:     string
  notes:           string[]

  fill_state:      FillState
  confidence:      number
  status:          NodeStatus

  version:         number
  created_at:      string
  updated_at:      string

  last_verified:   string
  freshness: {
    decay_rate:    number
    floor:         number
    stale_after:   number
  }

  parent:          string | null
  children:        string[]
  connections:     string[]        // node_ref[] or postcode[] during transition
  depth:           number

  references: {
    read_before:   string[]        // node_ref[]
    read_after:    string[]        // node_ref[]
    see_also:      string[]        // node_ref[]
    deep_dive:     string[]
    warns:         string[]        // node_ref[]
  }

  provenance: {
    source_ref:    string[]
    agent_id:      string
    run_id:        string
    timestamp:     string
    human_input:   boolean
  }

  token_cost:      number

  constraints:     Constraint[]
  constraint_source: string[]      // node_ref[] or postcode[] during transition

  layer:           string
  concern:         string
  scope:           string
  dimension:       string
  domain:          string
}
```

## 4.1 Fill States (Epistemic Type System)

```
[F]  filled        confidence > 0.85, all children resolved
[P]  partial       some children unresolved
[E]  empty         not yet touched
[B]  blocked       external dependency missing
[Q]  quarantined   provenance check failed
[C]  candidate     proposed by emergence, awaiting approval
```

## 4.2 Taint Propagation Rules

```
[Q] parent   → all children inherit [Q]
[B] dependency → all dependents become [B]

RECOVERY:
[Q]→[F] healed → children re-enter VERIFICATION
[B]→[F] unblocked → dependents re-enter AUTHORING
propagation = immediate. recovery = gated.
```

## 4.3 Confidence Thresholds

```
> 0.85     auto-promote
0.60-0.85  emergence resolves
< 0.60     escalate to human
```

## 4.4 Effective Confidence (with Freshness Decay)

```
effective_confidence = max(
  confidence - (days_since_verified × decay_rate),
  freshness.floor
)
```

---

# 5. THE FIVE AXIOMS

```
AX1  PROVENANCE   every output traces to verified input
AX2  DESCENT      go deeper until atomic (VAL scope = stop)
AX3  FEEDBACK     every write reports back as observation delta
AX4  EMERGENCE    proposals need external approval, never self-approve
AX5  CONSTRAINT   parent constraints own all children
```

One failure = node does not promote.

---

# 6. THE 7-AGENT COMPILATION PIPELINE

Sequential. Cannot reorder. Each produces artifacts the next requires.

```
Intent ──gate──▶ Persona ──gate──▶ Entity ◀──dialogue──▶ Process
                                                            │
                              ◀──gate── Synthesis ◀─────────┘
                                  │
                                  ▼
                               Verify ──fail──▶ recompile loop (max 3)
                                  │
                                  ▼
                               Governor ──▶ BLUEPRINT OUT
```

## 6.1 Agent Definitions

| Agent     | Lens          | Temp | Mode           | Produces                |
|-----------|---------------|------|----------------|-------------------------|
| Intent    | requirements  | 0.5  | analytical     | intent graph + contract |
| Persona   | domain context| 0.6  | empathic       | domain context model    |
| Entity    | structural    | 0.7  | spatial        | entity map (nouns)      |
| Process   | behavioral    | 0.7  | temporal       | process flows (verbs)   |
| Synthesis | integrative   | 0.4  | compositional  | unified blueprint draft |
| Verify    | adversarial   | 0.2  | forensic       | audit report            |
| Governor  | governance    | 0.0  | legal          | accept/reject decision  |

## 6.2 Entity↔Process Dialogue

Alternating challenge loop. Min 6 turns, min 8 insights, max 12 turns.
Challenge types:

```
MISSING
CONTRADICTION
ASSUMPTION
DEPTH
META
```

Convergence = no new insights for 2 consecutive rounds.

## 6.3 Provenance Gates

Checkpoint between every agent transition.
Every output artifact traces to validated input.

```
gate: {
  source_agent:    string
  source_nodes:    postcode[]
  transformation:  string
  hash:            string
}
```

## 6.4 Commit Ordering Rule

A node is citable ONLY after `NODE_WRITTEN`.

---

# 7. THE INTENT CONTRACT

Frozen artifact from Intent Agent. Immutable after Intent completes.

```typescript
interface IntentContract {
  seed_text:        string
  goals:            string[]
  constraints:      string[]
  layers_in_scope:  string[]
  domains_in_scope: string[]
  known_unknowns:   string[]
  budget_limit:     number

  anti_goals: [{
    description:    string
    derived_from:   string
    severity:       'critical' | 'high' | 'medium'
    detection:      string
  }]

  runtime_anchor: {
    enabled:        boolean
    invariants:     string[]
    check_mode:     'startup' | 'periodic' | 'continuous'
  }

  context_budget: {
    total:          number
    reserved:       number
    available:      number
    per_agent:      number
    compression_trigger: number
  }

  seed_hash:        string
  contract_hash:    string
}
```

---

# 8. SILENCE ZONES

```typescript
interface Silence {
  layer:       string
  reason:      string
  type:        'intentional' | 'deferred' | 'out_of_scope'
  decided_by:  'agent' | 'human' | 'intent_contract'
}
```

ABSENCE WITHOUT SILENCE MARKER = GAP.

---

# 9. DEFAULT LAYERS

Every compilation automatically includes OBS and SEC-adjacent concerns.
User never asks for logging or security. They get it.

---

# 10. LAYER ACTIVATION RULES

When a promoted node implies an unactivated layer, Governor approves activation
or logs it as silence/gap.

---

# 11. COMPILED OUTPUT FORMAT

What Synthesis Agent produces. What coding agents consume.

```typescript
interface CompiledBlueprint {
  metadata: {
    id:               string
    seed:             string
    seed_hash:        string
    blueprint_hash:   string
    created_at:       string
    version:          string
    compilation_depth: DepthReport
  }

  intent_contract:    IntentContract

  layers: [{
    layer:            string
    nodeCount:        number
    coverage:         string[]
  }]

  entities: [{
    name:             string
    postcode:         string
    description:      string
    attributes:       string[]
    confidence:       number
  }]

  functions: [{
    name:             string
    postcode:         string
    description:      string
    inputs:           string[]
    outputs:          string[]
    rules:            string[]
    confidence:       number
  }]

  rules: [{
    name:             string
    postcode:         string
    description:      string
    type:             'constraint' | 'policy' | 'trigger' | 'condition'
    confidence:       number
  }]

  relationships: [{
    from:             string
    to:               string
    relation:         string
    postcode:         string
    confidence:       number
  }]

  state_machines: [{
    name:             string
    postcode:         string
    states:           string[]
    transitions:      [{ from: string, to: string, trigger: string }]
  }]

  silences: Silence[]
  gaps: any[]
  failed_paths: any
  governance_report: GovernanceReport
  tests: any[]
}
```

---

# 12. COMPILATION DEPTH METRIC

```
depth 1-2:  sketch
depth 3-4:  demo
depth 5-6:  standard
depth 7+:   production
```

Measured by average scope depth, fill-state ratios, activated layers,
anti-goals, dialogue rounds, and gaps.

---

# 13. COST MODEL

- Estimate before compile.
- Hard ceiling during compile.
- Per-agent report after compile.

---

# 14. CONTEXT WINDOW — VIRTUAL MEMORY PROTOCOL

Postcodes = memory addresses.
Context window = RAM.
Map store = disk.

ALWAYS LOADED:
- intent_contract
- axioms
- anti_goals
- active node parent chain

HOT:
- current build layer
- connected nodes depth 1

WARM:
- adjacent layers
- sibling nodes

COLD:
- everything else

Lossless paging. Postcode is the retrieval key.

---

# 15. CODING AGENT INTERFACE

The coding agent is the renderer.
It does not design, architect, or improve. It renders.

## Five Rendering Laws

```
R1  BLUEPRINT SUPREMACY
R2  COMPLETE FILE EMISSION
R3  VECTOR-TO-CODE TRACE
R4  INTERIOR COMPLETENESS
R5  DEPENDENCY DECLARATION
```

## Build Sequence

```
1. FOUNDATION
2. TYPE
3. CORE
4. INTEGRATION
5. OPERATIONAL
```

## Transport Modes

```
MODE 1 — FILE
MODE 2 — API
MODE 3 — CONTEXT
```

---

# 16. BLUEPRINT LIFECYCLE

Blueprints are living artifacts.
- freshness decay
- semantic diff
- continuous compilation

---

# 17. CROSS-BLUEPRINT PROTOCOL

Blueprints reference each other via namespace prefix.

```
calc:EXC.FNC.APP.HOW.SFT/divide
auth:IDN.ACT.APP.WHO.SFT/authenticate
```

---

# 18. REVERSE COMPILATION

Existing code can be reverse-compiled into a partial blueprint.
Reverse-compiled nodes max at `[P] 0.80` until human verification.

---

# 19. GOVERNANCE REPORT

```typescript
interface GovernanceReport {
  total_nodes:       number
  promoted:          number
  quarantined:       any[]
  escalated:         any[]
  axiom_violations:  any[]
  human_decisions:   any[]
  coverage:          number
  anti_goals_checked: number
  compilation_depth: DepthReport
  cost_report:       CostReport
}
```

---

# 20. SHARED BLACKBOARD & EVENT BUS

Agents never call each other directly.
All communication through shared blackboard + event bus.

Event types:

```
AGENT_STARTED
AGENT_COMPLETED
GATE_PASSED
GATE_FAILED
NODE_WRITTEN
CHALLENGE_ISSUED
CONVERGENCE
ESCALATION
RECOMPILE_TRIGGERED
BLUEPRINT_EMITTED
```

---

# 21. PIPELINE STATES

```
idle → intent_phase → persona_phase → dialogue_phase →
synthesis_phase → verification_phase → governor_phase →
complete | rejected | halted
```

---

# 22. RECOMPILATION LOOP

When Verify fails, route the gap report back to the earliest implicated agent.
Max 3 recompilation loops, then escalate to human.

---

# 23. FRACTAL SELF-SIMILARITY

The architecture is isomorphic to its output.
Same pipeline at every level. Only provenance gate constraints change.

---

# 24. TERMINOLOGY

## Use
- semantic compiler / semantic compilation
- provenance gates / provenance-guided
- 7-agent pipeline
- intent degradation
- context constraints / constraint engineering
- semantic entropy reduction
- excavation
- postcode / coordinate
- blueprint

## Do NOT use
- SEED Protocol
- QRPT
- Oracles
- "AI tool" / "chatbot" / "assistant" for Mother

---

# 25. OPEN GAPS

1. Layer activation lookup table validation
2. Recompilation re-entry routing
3. Context chunking validation at 200+ nodes
4. Blueprint transport format implementation
5. Impossibility reporting
6. Blueprint-to-blueprint conflict resolution

---

# END OF SSOT

This document is the canonical reference for Motherlabs blueprints.
All agent implementations, coding agent prompts, and product decisions
defer to this document. Changes require versioning.
