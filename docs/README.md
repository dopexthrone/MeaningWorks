# Motherlabs Docs

This folder is the product and protocol map for the repo.

If you want to understand Motherlabs from GitHub without falling into generic AI-doc mush, read in this order:

1. [README.md](../README.md)
2. [BLUEPRINTS_SSOT.md](BLUEPRINTS_SSOT.md)
3. [WORKBENCH_IA.md](WORKBENCH_IA.md)
4. [ARCHITECTURE.md](ARCHITECTURE.md)
5. [API.md](API.md)

## Pick A Path

### I want to understand the product

- [MOTHERLABS_PRODUCT_DOCTRINE.md](MOTHERLABS_PRODUCT_DOCTRINE.md)
- [WORKBENCH_IA.md](WORKBENCH_IA.md)
- [BLUEPRINTS_SSOT.md](BLUEPRINTS_SSOT.md)

### I want to understand the implementation

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [API.md](API.md)

### I want to run or deploy it

- [README.md](../README.md)
- [DEPLOYMENT.md](DEPLOYMENT.md)

### I want the canonical truth

- [SSOT.md](SSOT.md)
- [BLUEPRINTS_SSOT.md](BLUEPRINTS_SSOT.md)
- [WORKBENCH_IA.md](WORKBENCH_IA.md)

## What These Docs Are Trying To Prevent

Motherlabs should not read like:

- a generic AI framework
- a chatbot wrapper
- a code editor with prompts
- a graph demo

The product in this repo is:

- a semantic compiler
- a postcode-native workbench
- a governed async decision loop
- a renderer handoff bundle for downstream coding agents

## Source Files Behind The Docs

When a doc needs to be checked against implementation, start here:

- `core/blueprint_protocol.py`
- `core/engine.py`
- `api/v2/models.py`
- `api/v2/routes.py`
- `frontend/src/lib/semantic/protocol.ts`
- `frontend/src/app/workbench/page.tsx`
