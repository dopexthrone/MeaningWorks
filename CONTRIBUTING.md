# Contributing to Motherlabs

If you want to contribute here, start by understanding what the product is trying to protect.

Motherlabs is not:

- a generic AI framework
- a prompt wrapper
- a code editor with chat attached
- a graph demo

Motherlabs in this repo is:

- a semantic compiler
- a postcode-native workbench
- a governed async decision loop
- a renderer handoff surface for coding agents

## Read Before You Touch Code

Do not open a PR blind. Read these first:

1. [README.md](README.md)
2. [docs/BLUEPRINTS_SSOT.md](docs/BLUEPRINTS_SSOT.md)
3. [docs/WORKBENCH_IA.md](docs/WORKBENCH_IA.md)
4. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
5. [docs/API.md](docs/API.md)

If your change contradicts those documents, it is probably wrong.

## What Good Contributions Look Like

- tighten the semantic compiler contract
- improve provenance, governance, or bounded termination
- improve the workbench as a reading and compiling environment
- add tests that catch semantic drift or regressions
- improve docs so GitHub reflects the real product

## What Not To Add

These are common ways to dilute the repo:

- generic chatbot surfaces as the main interaction
- code-editor behavior inside the workbench
- UI that looks like a themed VS Code clone
- features that bypass provenance or human-decision gates
- docs that describe Motherlabs as a vague AI platform

## Workflow

1. Branch from `main`.
2. Keep the change set narrow and explain the user-facing consequence.
3. Update the canonical contract docs first if the product truth changed.
4. Add or update tests for protocol, API, engine, or workbench behavior.
5. Run the smallest relevant verification set before opening the PR.

Examples:

- backend protocol/API change:

```bash
pytest tests/test_output_parser.py tests/test_schema.py tests/test_engine.py tests/test_api_v2.py -q
```

- frontend/workbench change:

```bash
cd frontend
npm run type-check
npm run build
```

- mixed change:

```bash
pytest tests/test_api_v2.py tests/test_engine.py -q
cd frontend && npm run type-check && npm run build
```

## Documentation Rule

If you change the product contract, update docs in this order:

1. `docs/BLUEPRINTS_SSOT.md`
2. `docs/WORKBENCH_IA.md`
3. implementation
4. `docs/API.md` / `docs/ARCHITECTURE.md`
5. `README.md`

Do not update README first and hope the rest catches up later.

## Project Areas

```text
core/         compiler contracts, engine, verification, trust, schema
api/          FastAPI V1/V2 routes and response models
swarm/        multi-agent compile orchestration
worker/       Huey task execution and progress ledger
frontend/     Motherlabs webapp and postcode-native workbench
kernel/       postcode parsing and semantic coordinate primitives
adapters/     domain adapters
persistence/  corpus and storage
docs/         canonical product and protocol docs
tests/        regression coverage
```

## Pull Request Checklist

- the change matches the current Motherlabs product boundary
- provenance or governance did not get weaker
- the workbench still reads as no-code and postcode-native
- docs were updated if the contract changed
- tests cover the regression you are fixing or the behavior you are adding

## Questions

Open an issue with:

- what you are changing
- why it belongs inside Motherlabs
- which canonical docs it touches

Or email [motherlabsai@gmail.com](mailto:motherlabsai@gmail.com).
