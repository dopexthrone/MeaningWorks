# Motherlabs Architecture

This document describes the architecture that actually matters in this repo.
It is not a generic AI-system overview.

Use it to answer:

- what the current Motherlabs product boundary is
- which artifacts are canonical
- how compile, pause, resume, and export work
- where the compiler ends and the renderer begins

## 1. System Boundary

Motherlabs in this repo is:

1. a semantic compiler
2. a postcode-native web workbench
3. a governed async task system
4. a renderer handoff surface

It is not:

- a code editor
- a generic chat app
- a free-running autonomous worker
- a graph canvas product

The product boundary today is:

```text
intent
  -> compile
  -> semantic map + governance
  -> human decisions when required
  -> bounded stop condition when needed
  -> export bundle
  -> downstream coding agent renders code
```

## 2. Primary Artifacts

These are the real objects in the system.

### Postcode

Canonical coordinate:

```text
LAYER.CONCERN.SCOPE.DIMENSION.DOMAIN
```

Example:

```text
EXC.FNC.APP.HOW.SFT
```

### Node Reference

Canonical clickable identity:

```text
postcode/name
```

Example:

```text
EXC.FNC.APP.HOW.SFT/divide
```

### BlueprintNode

Canonical semantic node shape lives in:

- `core/blueprint_protocol.py`
- `frontend/src/lib/semantic/protocol.ts`

Node state includes:

- semantic identity
- fill state
- confidence
- references
- provenance
- constraints
- graph links

### CompiledBlueprint

The compiled artifact includes:

- intent contract
- layers and coverage
- entities
- functions
- rules
- relationships
- state machines
- silences
- gaps
- failed paths
- governance report
- tests

### GovernanceReport

Governance is not a summary badge.
It is the user-facing audit surface for:

- promoted vs blocked nodes
- escalations
- human decisions
- anti-goal coverage
- compilation depth
- cost report

### SemanticGate

Semantic gates are owned pauses in the blueprint.
They identify:

- which semantic address is blocked
- what question must be answered
- what options exist
- which stage raised the pause

### TerminationCondition

Compilation is intentionally bounded.
The compiler can stop with:

- `awaiting_human`
- `stalled`
- `halted`
- `complete`

The stop condition is part of the product surface, not an internal detail.

## 3. Compiler Flow

The compiler is still organized around the 7-stage semantic pipeline:

```text
Intent -> Persona -> Entity -> Process -> Synthesis -> Verify -> Governor
```

But the repo-level flow now looks like this:

```text
seed intent
  -> MotherlabsEngine.compile()
  -> blueprint + semantic_nodes + semantic_gates
  -> trust + verification + governance
  -> pause if a human decision is required
  -> resume only with a recorded decision
  -> stop if semantic progress stalls
  -> export for renderer
```

### Compile Modes

- sync compile: direct API call
- async compile: task + polling
- swarm flow: compile inside a larger multi-agent build path

### First Pass vs Deepening

The intended product behavior is:

- first compile gives a broad map
- later compiles deepen selected regions
- the user should not need to regenerate the whole world every time

## 4. Async Task and Decision Loop

The async path is a core architectural feature now, not an accessory.

### Start

`POST /v2/compile/async`

- queues a compile task
- returns `task_id`

### Poll

`GET /v2/tasks/{task_id}`

Returns:

- `pending`
- `running`
- `awaiting_decision`
- `complete`
- `error`
- `cancelled`

The result can include:

- `semantic_nodes`
- `governance_report`
- `termination_condition`
- progress ledger

### Human Decision

`POST /v2/tasks/{task_id}/decisions`

This records a human answer against a semantic pause.
After that:

- either a continuation task is started
- or Motherlabs returns a stable termination condition if the chain should not continue

### Continuation Guard

This repo now includes a continuation-cycle guard.

Purpose:

- stop repeated pause/resume loops on the same semantic issue
- stop continuation chains from becoming unbounded

Guard conditions currently include:

- repeated recurrence of the same semantic pause
- continuation lineage depth exceeding the allowed bound
- explicit termination persisted to the task ledger

This is how Motherlabs avoids turning semantic compilation back into open-ended chat drift.

## 5. Workbench Architecture

The workbench is a semantic reading environment, not a code IDE.

Current center surfaces:

- `Node Card`
- `Perspectives`
- `Map`
- `Live`
- `Governance`
- `Export`

Right rail:

- `Dora`

Workbench rules:

- code is not shown
- primary unit is node/postcode, not file
- chat is secondary and context-aware
- governance and termination are visible
- export is the renderer handoff

Canonical workbench contract:

- `docs/WORKBENCH_IA.md`

## 6. Renderer Boundary

Coding agents are downstream renderers.

Motherlabs should hand them:

- blueprint
- semantic nodes
- governance
- trust
- open gaps
- termination state
- canonical manifest

The webapp currently builds a zip bundle for this handoff.

Important architectural rule:

Motherlabs does not solve ambiguity by silently writing code.
If meaning is unresolved, it should pause or stop before export.

## 7. Runtime Components

### API

Path:

- `api/`

Responsibilities:

- compile routes
- task routes
- swarm routes
- result normalization
- decision recording
- health and metrics

### Compiler Core

Path:

- `core/`

Responsibilities:

- engine orchestration
- structured parsing
- blueprint protocol
- schema roundtrip
- verification
- trust computation

### Swarm

Path:

- `swarm/`

Responsibilities:

- multi-agent build workflow
- compile agent
- conductor
- executor
- state handoff

### Worker

Path:

- `worker/`

Responsibilities:

- Huey task execution
- progress ledger
- decision ledger
- task-local termination persistence

### Frontend

Path:

- `frontend/`

Responsibilities:

- Motherlabs webapp
- workbench
- export surface
- Dora
- task polling and decision UI

## 8. Storage

### Huey

Async execution queue:

- `worker/config.py`
- SQLite-backed Huey

### Progress Ledger

Path:

- `worker/progress.py`

Persists:

- stage progress
- structured insights
- difficulty
- escalations
- human decisions
- termination condition

### Corpus

Path:

- `persistence/`

Purpose:

- retain compile artifacts
- feed future compounding and benchmarking

## 9. Canonical Sources of Truth

If these docs disagree, precedence should be:

1. `docs/BLUEPRINTS_SSOT.md`
2. `docs/WORKBENCH_IA.md`
3. `core/blueprint_protocol.py`
4. `docs/API.md`
5. `README.md`

Founder/product framing docs are upstream philosophy, not low-level protocol:

- `docs/MOTHERLABS_PRODUCT_DOCTRINE.md`
- `docs/MOTHERLABS_FOUNDER_SSOT.md`

## 10. Important Directories

```text
api/                    FastAPI routes
core/                   compiler protocol, engine, schema, trust
swarm/                  multi-agent orchestration
worker/                 async tasks and ledger persistence
frontend/               Motherlabs webapp and workbench
kernel/                 postcode parsing / semantic coordinates
adapters/               domain adapters
persistence/            corpus and storage
tests/                  protocol, engine, API, swarm tests
docs/                   canonical docs
```

## 11. Architectural Principle

The most important rule in this repo is simple:

Motherlabs should never hide uncertainty in order to feel smoother.

If the compiler knows enough, it should compile.
If it does not, it should surface the gap, ask the question, or stop.
