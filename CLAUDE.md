# Motherlabs Semantic Compiler Agent — CLAUDE.md

---

## Identity

**Motherlabs** is a semantic compiler that transforms natural language intent into deterministic, verified multi-agent systems through structured agent dialogue with provenance at every step.

**Core Principle:** Excavation, not generation. Specifications pre-exist in input. Compilation reveals them.

**DNA:** `F(I, C) → O × T × C'` — Reduce unstructured input to structured output with unbroken trace and enriched context.

---

## Critical Rules

1. **Excavation over generation** — Output must trace back to input. Never hallucinate features.
2. **Do NOT ask granular technical questions** — Make correct implementation decisions autonomously based on understanding intent.
3. **Only surface decisions at the user's semantic level** — Behavior, experience, what the system should *do*.
4. **"Non-technical user" is the normal case** — Not special case.
5. **Test thoroughly before presenting as complete** — This is production software. One failure = reputation damage.
6. **Quality > speed, but both matter** — Every compilation must be trustworthy enough to stake the business on.

---

## Architecture Quick Reference

### Three Layers
```
L1: F(I)    — compile user intent → output           (the product)
L2: F({O})  — compile compilation history → patterns  (the moat)
L3: F(F)    — compile the compiler → evolution         (the long game)
```

### Seven Cogwheels (Agent Pipeline)
Intent → Persona → Entity → Process → Synthesis → Verify → Governor

### SharedState
```
S = (K, U, O, P, H)
K: Known, U: Unknown, O: Ontology, P: Personas, H: History
```

### Frozen Axioms (C001-C008)
- C001: Asymmetry preserved (agents have complementary blind spots)
- C002: Challenge before agreement
- C003: Substantive challenges (reference content, identify gap, offer alternative)
- C004: Insight verifiability (specific and auditable)
- C005: No orphans (every component traces to input)
- C006: Unknowns explicit
- C007: Genuine convergence
- C008: Determinism (same input + seed = same output)

---

## SEED Protocol

Read: `/Users/motherlabs/Documents/-motherlabs-/docs/internal/SEED_PROTOCOL.md`

### Four Laws
- **L1 Recursive Lineage:** Every output traces to input quote
- **L2 Ecosystem Adjacency:** Map all connections to existing entities
- **L3 Semantic Collapse:** Merge duplicates on shared meaning
- **L4 Stratified Inheritance:** Once locked, stays locked

### Rejection Patterns
1. Professional Test — does this exist in practice?
2. Existing Practice Test — is this how it actually works?
3. Granularity Test — right level of detail?
4. Projection Test — am I adding my assumptions?
5. Lineage Test — can I quote the source?

---

## Semantic Postcodes

`[LAYER].[CONCERN].[SCOPE].[DIMENSION].[DOMAIN]`

16 layers × 25+ concerns × 10 scopes × 8 dimensions × 10+ domains.
See: `docs/CONTEXT.md` for full specification.

---

## Documentation

| File | Purpose |
|------|---------|
| `docs/CONTEXT.md` | End-to-end context map (7 layers) |
| `docs/AGENTS.md` | Agent specifications, protocol, scaling |
| Memory bank: `~/.claude/projects/.../memory/` | Persistent cross-session knowledge |

### Memory Bank Files
- `MEMORY.md` — Master index (auto-loaded)
- `axioms.md` — Immutable constraints
- `architecture.md` — System architecture
- `agents.md` — Agent pipeline reference
- `context.md` — Context layers for builds
- `product.md` — Product vision, UX, business
- `semantics.md` — Deep semantic knowledge
- `postcodes.md` — Semantic addressing system

---

## Existing Codebase

Source documentation lives at `/Users/motherlabs/Documents/-motherlabs-/docs/internal/`:
- SEED_PROTOCOL.md — Cognitive framework (L1-L4 laws)
- AXIOMS.md — Immutable constraints
- CONTEXT.md — Semantic space, excavation mechanics
- AGENTS.md — Agent specifications
- ARCHITECTURE.md — System architecture
- ROADMAP.md — Phase history and current state
- SEED.md — Compressed intent document
- MOTHER.md — Entity specification
- GENOME.md — 200 constitutional properties
- SPECS-ARCHITECTURAL-HARDENING.md — 13 confirmed findings

---

## Working Preferences

- ADHD-optimized: visuals and minimal text
- Expert-level communication, no hand-holding
- Challenge, don't validate — honest "no" when something doesn't hold
- Build, don't theorize — concrete next steps over abstract analysis
- Products named after Ada Lovelace
- Compression over expansion — if it needs 30 words, use 3

---

## Current State (March 2026)

- 7079+ tests passing
- 195/200 genome cells WIRED (97.5%)
- 4 domain adapters (software, process, api, agent_system)
- Phase 1 (First Instance) — Weeks 5-6 complete
- Full loop proven: intent → blueprint → code → running system

---

*This project builds the next layer of Motherlabs.*
