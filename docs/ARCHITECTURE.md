# MeaningWorks Architecture

This document explains how the system works, from the ground up. No fluff — just the actual machinery.

---

## The one-sentence version

Natural language goes in. A 7-agent pipeline excavates the specification that was already implied. A verified, traceable, running system comes out.

---

## The pipeline

```
Input (what you said)
    ↓
[1] Intent Extraction
    What do you actually need? (not what you said — what you meant)
    ↓
[2] Persona Generation
    Who touches this system? What does each person care about?
    ↓
[3] Spec Dialogue (the heart)
    Entity Agent ↔ Process Agent
    Structure vs Behavior — asymmetric, challenged, productive friction
    ↓
[4] Synthesis
    Collapse dialogue into coherent blueprint
    ↓
[5] Verification
    Does this hold up? Completeness, consistency, traceability
    ↓
[6] Emission
    Blueprint → running code with provenance
    ↓
Output: blueprint + context graph + trace + running system
```

### Phase 3: Where the magic happens

The Entity Agent sees **structure** — nouns, attributes, relationships, containment. It's blind to temporal flow, causation, and triggers.

The Process Agent sees **behavior** — verbs, state transitions, data flows, temporal sequence. It's blind to static structure and relationships.

Neither can see the complete picture. That's the design. They challenge each other:

- Entity: "Booking is a data structure with User, Session, Artist"
- Process: "But sessions vary 1-8 hours — your fixed-slot model breaks"
- Entity: "Fine. Session becomes dynamic: duration_min, duration_max"
- Both: "Now we see it"

The specification emerges from the space between them. Not from either agent. From the friction.

### Challenge protocol

Every challenge must be:
1. **Substantive** — references specific content, not "are you sure?"
2. **Constructive** — identifies a gap AND offers an alternative
3. **Accountable** — the challenged agent must revise or justify (never ignore)

Anti-patterns that get rejected:
- "Are you sure?" (performative — no content)
- "You're right" (capitulation — no reasoning)
- Challenge on unrelated point (tangent)
- Same challenge repeated (infinite loop)

---

## SharedState

The single source of truth all agents read and write:

```
S = (K, U, O, P, H)

K: Known      — resolved specifications
U: Unknown    — remaining ambiguities (shrinks over time)
O: Ontology   — shared vocabulary
P: Personas   — domain perspectives
H: History    — complete decision trace (append-only)
```

**Key properties:**
- Unknowns are monotonically decreasing — they move to Known or stay explicit
- History is append-only — for auditability
- Every Known item traces back through History to the original input

---

## The three layers

```
L1:  F(I)    → compile intent to output
L2:  F({O})  → compile compilations to patterns
L3:  F(F)    → compile the compiler to evolution
```

**L1** is the product. User describes something, gets a working system.

**L2** is the moat. After 50 compilations in the same domain, the system knows patterns — "booking systems need scheduling, matching, state management, and payments." After 500 compilations, it knows anti-patterns too — "fixed time slots always fail for variable-length sessions."

**L3** is the long game. The compiler can compile itself. It finds its own gaps, compiles solutions, and evolves. Recursive self-improvement that's bounded and traceable (not unbounded and scary).

---

## Semantic postcodes

Every concept has a 5-axis coordinate:

```
[LAYER].[CONCERN].[SCOPE].[DIMENSION].[DOMAIN]

Example:
INT.SEM.ECO.WHY.ORG  →  "Why the organization exists" (Intent, Semantic, Ecosystem, Why, Organization)
AGN.AGT.CMP.WHO.SFT  →  "Which software agent handles this" (Agency, Agent, Component, Who, Software)
```

**16 layers** (Intent, Semantic, Organization, Cognitive, Agency, Structure, State, Identity, Time, Execution, Control, Resource, Observability, Network, Emergence, Meta)

**6 fill states:**
- **F**illed — verified, confidence >= 0.85
- **P**artial — exists but incomplete
- **E**mpty — cell exists, nothing there yet
- **B**locked — dependency unresolved
- **Q**uarantined — Governor rejected, trace preserved
- **C**andidate — pattern detected, awaiting promotion

The grid makes gaps visible. If a cell is Empty, you can see it. If it's Blocked, you know why. No hidden assumptions.

---

## Provenance gates

Between every pipeline stage sits a gate that records:

```
{ input, output, added, rejected, reason, timestamp, agent }
```

The chain of all gates = **compilation receipt**. This is the trust artifact.

You can ask: "Why does this component exist?" and follow the chain:
→ Component traces to insight in turn 7
→ Turn 7: Entity challenged Process on state representation
→ Process: "Sessions vary 1-8 hours, not fixed slots"
→ Resolution: time windows instead of fixed slots
→ Derived from: original input "multi-hour tattoo sessions with variable duration"

This chain is **not recoverable from the blueprint alone.** It only exists in the context graph. That's the moat.

---

## Domain adapters

The pipeline is domain-invariant. Domain-specific behavior is injected through adapters:

| Adapter | Output type | Vocabulary |
|---------|-------------|------------|
| Software | Code + infrastructure | entities, services, APIs |
| Process | YAML workflows | participants, steps, artifacts |
| API | OpenAPI specifications | endpoints, schemas, auth |
| Agent System | Running agents | components, dispatch, state |

New domains = new 3-character code + adapter implementation. The pipeline doesn't change.

---

## Emission layer

Blueprints become running systems:

- **Runtime LLM client** — provider-agnostic, async, with retry
- **Stateful event loop** — TCP server with component dispatch
- **Persistent state** — SQLite/JSON store, async CRUD
- **Self-modification** — recompilation with gap detection and safe rollback
- **Sandboxed tool execution** — allowlist-based, path traversal protection
- **Event coordination** — message dispatch, component registration

---

## Termination

The system always terminates. Four possible endings:

| State | Meaning |
|-------|---------|
| **SUCCESS** | K saturated, U empty, agents agree |
| **EXHAUSTION** | Budget depleted before convergence (hard ceiling: 15 turns) |
| **AMBIGUOUS** | Irreconcilable conflicts that dialogue can't resolve |
| **USER_FLAG** | User disagreed with an insight — needs human input |

No infinite loops. No runaway costs. Bounded by design.

---

## Confidence system

The Governor tracks 4 confidence dimensions (structural, behavioral, coverage, consistency) and uses them for dialogue control:

- Any dimension below 0.4 → force +2 turns
- Spread between max and min > 0.4 → force +3 turns
- Confidence unchanged for 3 turns → inject focus hint
- max_turns (12) → hard stop, always honored

This prevents both premature convergence (stopping before it's ready) and infinite dialogue (going nowhere).

---

## Cost

~$0.04 per compilation run. Per-compilation cap: $5. Per-session cap: $50.

The system tracks token usage across all LLM calls and stops before exceeding limits. No surprises.

---

## Invariants

These must hold across every change, every version, every domain:

1. **Asymmetry preserved** — no agent is a superset of another
2. **Challenge enforced** — no agreement without prior substantive challenge
3. **Insights traceable** — component → insight → turn → agent → input
4. **Unknowns explicit** — ambiguity surfaced, never buried
5. **Convergence deterministic** — same input + seed = same output

Break any of these and the trust guarantees collapse. They're not nice-to-haves.

---

*Architecture v1.0.0 — March 2026*
