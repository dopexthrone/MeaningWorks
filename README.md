# Motherlabs

> Clarity You Can Ship

Motherlabs is a semantic compiler and no-code workbench for turning domain intent into buildable software blueprints.

It is not a chatbot wrapper.  
It is not a code IDE with AI attached.  
It is not a graph toy.

This repo contains the compiler core, the governed async compile loop, the postcode-native web workbench, and the renderer handoff bundle used to push resolved context into Codex, Claude Code, or another coding agent.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## What This Repo Actually Does

Motherlabs takes a seed intent and compiles it into:

- a `Blueprint`
- a `semantic map` of postcode-addressed nodes
- a `governance report`
- a `termination condition` when the system should stop instead of looping
- an `export bundle` for a downstream coding agent

Core flow:

```text
seed intent
  -> first-pass compilation
  -> semantic nodes + governance + trust
  -> deeper recompilation on selected regions
  -> human decisions when needed
  -> bounded termination instead of open-ended looping
  -> export bundle for renderer
```

## Product Boundary

Current Motherlabs boundary:

- semantic compiler
- web workbench
- governed async compile loop
- export to renderer

Not in scope for this repo's public product surface:

- a code editor inside the workbench
- generic chat as the main interaction
- Mother as a separate persistent worker runtime

The workbench is for reading, navigating, branching, governing, and exporting meaning.
Code is downstream.

## Core Objects

- `Postcode`: `LAYER.CONCERN.SCOPE.DIMENSION.DOMAIN`
- `NodeRef`: `postcode/name`
- `BlueprintNode`: canonical semantic node model
- `CompiledBlueprint`: blueprint artifact with semantic map + outputs
- `GovernanceReport`: what was checked, escalated, approved, or blocked
- `SemanticGate`: explicit human-in-the-loop pause surface
- `TerminationCondition`: why compilation stopped, and what should happen next

## Why It Exists

Motherlabs is built around one claim:

> Reliability comes from constraints, not capability.

The system exists to prevent intent degradation between what a domain expert means and what a coding agent eventually builds.

That means:

- unknowns must be explicit
- provenance must stay intact
- a compile should stop when ambiguity is irreducible
- the user should never lose semantic state in an open-ended conversation

## Repo Map

```text
api/                    FastAPI surface for compile, task, and swarm routes
core/                   compiler core, protocol, schema, verification, trust
swarm/                  multi-agent orchestration and async build flow
worker/                 Huey task execution + task progress ledger
frontend/               Next.js Motherlabs webapp + workbench
kernel/                 postcode parsing and semantic coordinate primitives
adapters/               domain adapters
persistence/            SQLite-backed corpus and storage
docs/                   canonical product, blueprint, workbench, API docs
tests/                  protocol, engine, API, swarm, and UI-facing tests
```

## Read This First

These are the docs that matter:

| File | Why it exists |
| --- | --- |
| [docs/README.md](docs/README.md) | Docs folder map and reading paths |
| [docs/BLUEPRINTS_SSOT.md](docs/BLUEPRINTS_SSOT.md) | Canonical blueprint contract |
| [docs/WORKBENCH_IA.md](docs/WORKBENCH_IA.md) | Workbench interaction model |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Real system structure in this repo |
| [docs/API.md](docs/API.md) | Public API contract |
| [docs/SSOT.md](docs/SSOT.md) | Documentation index and precedence |
| [docs/MOTHERLABS_PRODUCT_DOCTRINE.md](docs/MOTHERLABS_PRODUCT_DOCTRINE.md) | Product truth and tone |

## Local Development

### 1. Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Simplest local backend run:

```bash
MOTHERLABS_HUEY_IMMEDIATE=1 \
MOTHERLABS_DATA_DIR=$(pwd)/.data \
python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8001
```

That runs async compile requests in immediate mode so you do not need a separate worker while developing the webapp.

Worker-backed mode:

```bash
python -m huey.bin.huey_consumer worker.config.huey --workers=2 --worker-type=thread
```

### 2. Frontend

```bash
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://127.0.0.1:8001 npm run dev -- --hostname 127.0.0.1 --port 3000
```

Workbench:

- `http://127.0.0.1:3000/workbench`

API health:

- `http://127.0.0.1:8001/v2/health`

### 3. Tests

```bash
pytest tests/test_api_v2.py tests/test_engine.py tests/test_output_parser.py -q
cd frontend && npm run type-check && npm run build
```

## Main API Surface

The GitHub-facing primary flow is V2:

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v2/compile` | sync compile |
| `POST` | `/v2/compile/async` | async compile returning task id |
| `GET` | `/v2/tasks/{task_id}` | task polling with governance, gates, and termination |
| `POST` | `/v2/tasks/{task_id}/decisions` | record a human decision and continue, or stop with a guard |
| `DELETE` | `/v2/tasks/{task_id}` | cancel pending task |
| `POST` | `/v2/swarm/execute` | async swarm workflow |
| `GET` | `/v2/health` | health |

Important response surfaces:

- `semantic_nodes`
- `governance_report`
- `termination_condition`
- `structured_insights`
- `stage_results`

See [docs/API.md](docs/API.md) for the actual contract.

## Workbench Surface

The workbench is postcode-native and no-code:

- `Node Card`
- `Perspective Views`
- `Map View`
- `Compilation Live View`
- `Governance`
- `Export`
- `Dora` as the deep reading layer

The export bundle is the renderer handoff. It currently includes:

- blueprint JSON
- compile response
- trust + verification
- canonical renderer manifest
- semantic nodes
- governance and stage data
- open gaps preserved as explicit context

## Repository

Canonical GitHub location:

```bash
git clone https://github.com/alexrozex/Motherlabs-Semantic-Compiler.git
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
