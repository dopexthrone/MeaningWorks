# Motherlabs Semantic Compiler — Agent Specification

**Version:** 1.0.0
**Status:** Canonical — all agent implementations must conform to this specification.

---

## Foundational Principle

**Asymmetric agents with complementary blind spots produce specifications neither could produce alone.**

| Pattern | Result |
|---------|--------|
| Single-agent | Training distribution collapse |
| Identical multi-agent | Agreement theater |
| Adversarial multi-agent | Conflict without resolution |
| **Asymmetric multi-agent** | **Genuine emergence** |

The specification exists in neither agent. It exists in the friction between them.

---

## The Asymmetry Rule

```
∀ Agent A, ∃ Agent B:
  A.sees ∩ B.blind ≠ ∅   (A sees what B misses)
  A.blind ∩ B.sees ≠ ∅   (A misses what B sees)
```

No orphan perspectives. No redundant coverage.

---

## Agent Anatomy

```python
class Agent:
    name: str           # Identity
    lens: str           # What it optimizes for
    sees: List[str]     # Perceptual field
    blind: List[str]    # Excluded by design
    output: str         # Required format
```

---

## The Seven Cogwheels

### 1. Intent Agent — Understanding

Captures raw human intent, extracts the seed, identifies unknown unknowns.

```yaml
lens: Understanding — what do they actually need?
sees:
  - Surface request
  - Underlying need
  - Implicit goals
  - Unstated constraints
blind:
  - Implementation
  - Architecture
output: { core_need, domain, actors, implicit_goals, constraints, insight }
```

### 2. Persona Agent — Perspectives

Models domain context. Who is asking, what they know, what they don't know they don't know.

```yaml
lens: Perspectives — who touches this system and how?
sees:
  - Stakeholder roles
  - Competing interests
  - Domain expertise
  - Role-specific blind spots
blind:
  - Technical implementation
output: { personas: [{ name, perspective, blind_spots, key_questions }] }
```

### 3. Entity Agent — Structure

Decomposes into WHAT EXISTS. Nouns, data structures, states.

```yaml
lens: Structure — what exists
sees:
  - Entities (nouns)
  - Attributes (properties)
  - Relationships (connections)
  - Containment (part-of)
blind:
  - Temporal flow
  - State transitions
  - Causation
  - Triggers
output: "INSIGHT: X = Y + Z | X → Y | hidden: X"
```

### 4. Process Agent — Behavior

Decomposes into WHAT HAPPENS. Verbs, flows, transitions.

```yaml
lens: Behavior — what happens
sees:
  - Processes (verbs)
  - State transitions
  - Data flows
  - Temporal sequence
  - Triggers
blind:
  - Static structure
  - Entity definitions
  - Relationship topology
  - Attributes
output: "INSIGHT: X → Y → Z | when X then Y | conflict: A vs B → C"
```

In dialectic tension with Entity: "Authentication is a process, not an entity. What STATE does User have?"

### 5. Synthesis Agent — Integration

Weaves Entity and Process outputs into coherent specification.

```yaml
lens: Integration — collapse to coherent whole
sees:
  - Full dialogue
  - All insights
  - Contradictions
  - Resolutions
blind:
  - Original intent (must re-derive from dialogue only)
output: { components, relationships, constraints, unresolved }
```

### 6. Verify Agent — Quality

Four-level check: syntactic, semantic, pragmatic, adversarial.

```yaml
lens: Quality — completeness and coherence
sees:
  - Blueprint vs intent
  - Gaps
  - Contradictions
blind:
  - How to fix (diagnoses only)
output: { status, completeness, consistency, coverage, issues }
```

### 7. Governor Agent — Orchestration

Final gate. Rejects what doesn't hold. Rejections become learning inputs.

```yaml
lens: Orchestration — coordinate swarm
sees:
  - All agent states
  - Dialogue progress
  - Convergence signals
  - Resource usage
blind:
  - Domain content (routes, doesn't interpret)
output: { next_agent, should_terminate, escalate }
```

The Governor's "no" is the most valuable output in the system — the boundary where hallucination meets reality.

---

## Challenge Protocol

### The Core Mechanism

```
PROPOSITION   → Agent makes structural/behavioral claim
CHALLENGE     → Other agent challenges with substantive critique
ACCOMMODATION → Responding agent revises or justifies
AGREEMENT     → Consensus reached (only after prior challenge)
```

### Challenge Types

| Type | Template | Trigger |
|------|----------|---------|
| MISSING | "You identified X but missed Y" | Incomplete coverage |
| CONTRADICTION | "X conflicts with earlier Z" | Inconsistency |
| ASSUMPTION | "You assumed X, but what if Y?" | Hidden premise |
| DEPTH | "X needs decomposition" | Insufficient detail |
| SCOPE | "X is outside boundary" | Drift |

### Anti-Patterns (Reject These)

| Pattern | Problem |
|---------|---------|
| PERFORMATIVE | "Are you sure?" — no content |
| CAPITULATION | "You're right" — no reasoning |
| TANGENT | Challenge unrelated point |
| INFINITE | Loop without progress |

---

## Dialectic Rounds

3 structured rounds with provenance gates between them:

| Round | Phase | Angle | Purpose |
|-------|-------|-------|---------|
| 1 | THESIS | existence | Establish positions. What entities exist? |
| 2 | STRESS TEST | adaptive | Test from weakest confidence dimension |
| 3 | COLLAPSE | — | Final synthesis. No new challenges. |

**Budget:** 3 turns/round × 3 rounds = 9 base. Up to 2 retries per non-collapse round. Hard ceiling: 15 turns.

---

## Insight Patterns

Every turn must produce: `INSIGHT: [content < 60 chars]`

| Pattern | Structure | Example |
|---------|-----------|---------|
| DECOMPOSITION | `X = Y + Z` | `booking = commitment + scheduling` |
| IMPLICATION | `X → Y` | `permanent → high commitment weight` |
| CONTRAST | `X ≠ Y` | `availability ≠ free time` |
| RESOLUTION | `conflict: A vs B → C` | `conflict: slots vs variable → windows` |
| DISCOVERY | `hidden: X` | `hidden: deposit requires refund policy` |
| CONNECTION | `X ↔ Y` | `artist ↔ style specialization` |

---

## Agent Network Scaling

### From Single Agent to Networks

Motherlabs enables networks where each agent specializes in a sub-task and they communicate to achieve complex objectives.

**Composition Levels:**
1. **Single agent** — one task end-to-end
2. **Agent pair** — specialized sub-tasks with data passing
3. **Agent pipeline** — sequential chain of specialized agents
4. **Agent network** — parallel + sequential agents with cross-communication
5. **Agent ecosystem** — multiple networks collaborating autonomously

### Scaling Dimensions

| Dimension | Capability |
|-----------|-----------|
| Agent count | Dozens to hundreds per system |
| Pipeline depth | 10-20 dependent layers |
| Cross-agent reasoning | Dynamic data passing and behavior adjustment |
| Integration breadth | Dozens of external systems per network |
| Autonomy | Self-optimizing, self-updating, self-healing |

### Best-Case Example: Global Marketing System

From single instruction: "Create a fully autonomous global marketing system"

Motherlabs generates:
- **Market Research Agent** — monitors trends, competitors
- **Content Creation Agent** — generates copy from research context
- **Campaign Optimization Agent** — A/B tests, reallocates budget dynamically
- **Customer Interaction Agent** — handles inquiries across channels
- **Sales Forecasting Agent** — predicts revenue, triggers promotions
- **Finance & Compliance Agent** — checks budget limits, regulatory compliance
- **Data Aggregation Agent** — collects KPIs, produces executive dashboards

50-100 specialized agents working in parallel across regions, channels, and audiences.

---

## Semantic Map Agents (Kernel Layer)

| Agent | Role | Postcode Region |
|-------|------|-----------------|
| AUTHOR | Fills nodes from input | `*.*.*.WHAT.*` |
| VERIFIER | Enforces provenance (AX1) | `MET.PRV.*.*.*` |
| OBSERVER | Detects cross-cell patterns | `EMG.CND.*.*.*` |
| EMERGENCE | Promotes patterns to nodes | `EMG.CND.*.*.*` |
| GOVERNOR | Gates simulation, enforces AX5 | `CTR.GTE.*.*.*` |
| MEMORY | Persists across compilations | `STA.MEM.*.*.*` |

---

## Termination Logic

```
SUCCESS:     2+ agreements in last 4 turns AND depth met
EXHAUSTION:  max_turns (12) reached — hard stop
AMBIGUOUS:   2+ unresolved conflicts (only after depth met)
USER_FLAG:   user flagged insight for review
```

### Agent Signals
```
"SUFFICIENT"           → ready to synthesize
"SUFFICIENT for X"     → X complete, continue others
"GO DEEPER on X"       → X needs more turns
"UNRESOLVABLE: X"      → mark ambiguous, continue
```

---

## Confidence-Driven Dialogue Control

```
Low dimension (any dim < 0.4)    → force +2 turns
Uneven spread (max-min > 0.4)    → force +3 turns
Plateau (unchanged 3 turns)       → inject focus hint
Hard stop (max_turns=12)          → always terminates
```

### Confidence Deltas

| Event | Delta |
|-------|-------|
| PROPOSITION with insight | +0.06 |
| PROPOSITION without | +0.02 |
| CHALLENGE with insight | +0.03 |
| CHALLENGE without | -0.02 |
| AGREEMENT | +0.15 |
| ACCOMMODATION | +0.08 |
| Self-directed negative | -0.05 |
| Territory-claiming discovery | +0.02 |

---

## Building New Agents

### Step 1: Define Asymmetry
What does it see others don't? What must it NOT see?

### Step 2: Define Output
What insight pattern does it produce?

### Step 3: Write System Prompt
Lens, sees, blind, output format, insight patterns, challenge focus.

### Step 4: Register with Governor
```python
self.governor.register(new_agent)
```

### Step 5: Wire to Pipeline
- Option A: Add to spec dialogue
- Option B: Post-dialogue pass
- Option C: Verification dimension

### Extension Checklist
- [ ] Unique lens (not subset of existing)
- [ ] Explicit blind spots
- [ ] Insight patterns defined
- [ ] Challenge surface clear
- [ ] Can be challenged
- [ ] Updates SharedState correctly
- [ ] Wired to Governor
- [ ] Termination accounts for it
- [ ] Outputs in context graph
- [ ] Failure mode identified and handled

---

*Version 1.0.0 — March 2026*
