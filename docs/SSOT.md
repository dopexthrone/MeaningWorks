# Motherlabs Documentation SSOT

This file is the documentation map for the repo.

It answers one question:

> Which document is canonical for which kind of truth?

The previous version of this file mixed founder narrative, philosophy, product identity, and implementation.
That made the GitHub docs harder to trust.

Founder worldview still matters, but it should not be the public-facing documentation entry point.
That material now lives in the dedicated founder docs.

## 1. Read Order

If you are new to the repo, read in this order:

1. [README.md](../README.md)
2. [docs/README.md](README.md)
3. [docs/BLUEPRINTS_SSOT.md](BLUEPRINTS_SSOT.md)
4. [docs/WORKBENCH_IA.md](WORKBENCH_IA.md)
5. [docs/ARCHITECTURE.md](ARCHITECTURE.md)
6. [docs/API.md](API.md)

## 2. Canonical Documents

### Public Product and Repo Surface

- [docs/README.md](README.md)
  - docs folder landing page
  - reading paths by intent
  - quick map of canonical sources

- [README.md](../README.md)
  - what the repo is
  - what is in scope
  - local run instructions
  - main links

- [docs/ARCHITECTURE.md](ARCHITECTURE.md)
  - current implementation shape
  - system boundary
  - compiler/workbench/task/export flow

- [docs/API.md](API.md)
  - public API contract
  - async compile loop
  - decision and termination behavior

### Canonical Product Contracts

- [docs/BLUEPRINTS_SSOT.md](BLUEPRINTS_SSOT.md)
  - blueprint contract
  - postcode system
  - node schema
  - governance semantics

- [docs/WORKBENCH_IA.md](WORKBENCH_IA.md)
  - workbench object model
  - shell layout
  - interaction rules

### Upstream Philosophy

- [docs/MOTHERLABS_PRODUCT_DOCTRINE.md](MOTHERLABS_PRODUCT_DOCTRINE.md)
  - product identity
  - product laws
  - interface doctrine
  - tone and aesthetic constraints

- [docs/MOTHERLABS_FOUNDER_SSOT.md](MOTHERLABS_FOUNDER_SSOT.md)
  - founder worldview
  - origin story
  - long-horizon direction
  - conceptual framing

## 3. Precedence Rules

If two docs disagree, use this order:

1. `docs/BLUEPRINTS_SSOT.md`
2. `docs/WORKBENCH_IA.md`
3. source code protocol and schema
4. `docs/API.md`
5. `docs/ARCHITECTURE.md`
6. `README.md`
7. founder/product doctrine docs

Why:

- blueprint and workbench contracts define the product truth
- source code enforces runtime truth
- README should explain, not override
- philosophy docs guide decisions, but should not silently replace protocol

## 4. What This Repo Should Never Do Again

Documentation should not:

- describe Motherlabs as a generic AI system
- present founder narrative as the main technical entry point
- call the workbench a code IDE
- imply chat is the primary interaction
- hide termination, governance, or open gaps
- drift away from the postcode/node blueprint contract

## 5. Source of Implementation Truth

When docs need to be checked against implementation, use:

- `core/blueprint_protocol.py`
- `core/engine.py`
- `api/v2/models.py`
- `api/v2/routes.py`
- `frontend/src/lib/semantic/protocol.ts`
- `frontend/src/app/workbench/page.tsx`

## 6. Maintenance Rule

When the product changes:

- update the canonical contract first
- then update architecture/API docs
- then update README

Not the reverse.
