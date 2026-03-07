# Motherlabs API Reference

This document covers the public API surface that matters for the current Motherlabs product.

If you are trying to understand the current system, start with:

1. [README.md](../README.md)
2. [docs/ARCHITECTURE.md](ARCHITECTURE.md)
3. this file

Base URLs:

- local API: `http://127.0.0.1:8001`
- docs: `/docs`
- health: `/v2/health`

## 1. Authentication and BYO Model Keys

### API Authentication

API auth is optional and controlled by:

```text
MOTHERLABS_REQUIRE_AUTH=1
```

When enabled:

```bash
curl -H "X-API-Key: your-key" http://127.0.0.1:8001/v2/health
```

### BYO LLM Key

Compile requests can provide a model key per request.
That is the intended local and MVP flow.

Headers:

```text
X-LLM-API-Key: sk-ant-...
X-LLM-Provider: claude
```

Supported providers:

- `claude`
- `openai`
- `grok`
- `gemini`

## 2. Main Product Flow

The primary API loop is:

```text
POST /v2/compile/async
  -> receive task_id
GET  /v2/tasks/{task_id}
  -> running | awaiting_decision | complete | error
POST /v2/tasks/{task_id}/decisions
  -> continue compile or return termination condition
```

This is the governed compile loop used by the Motherlabs workbench.

## 3. Core Response Surfaces

Important fields returned by compile/task endpoints:

- `blueprint`
- `semantic_nodes`
- `governance_report`
- `termination_condition`
- `trust`
- `verification`
- `structured_insights`
- `stage_results`

### `termination_condition`

This field exists so the compiler can stop honestly instead of looping.

Current statuses:

- `awaiting_human`
- `stalled`
- `halted`
- `complete`

### `semantic_nodes`

Canonical postcode-native node surface used by the workbench.

Each node is addressed by:

```text
postcode/name
```

### `governance_report`

User-facing audit surface including:

- coverage
- promoted/quarantined nodes
- escalations
- human decisions
- anti-goal coverage
- compilation depth
- cost report

## 4. Compile Endpoints

### `POST /v2/compile`

Run a synchronous compile.

Request body:

```json
{
  "description": "Build a booking system for a tattoo studio",
  "domain": "software",
  "provider": "claude",
  "enrich": true
}
```

Main request fields:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `description` | string | yes | seed intent |
| `domain` | string | no | defaults to `software` |
| `provider` | string | no | provider override |
| `enrich` | boolean | no | expands sparse seed before compile |
| `canonical_components` | string[] | no | optional anchor names |
| `canonical_relationships` | string[][] | no | optional relationship anchors |

This route returns the full compile payload directly.

### `POST /v2/compile/async`

Queue a compile for async execution.

Response:

```json
{
  "task_id": "abc123",
  "status": "queued",
  "poll_url": "/v2/tasks/abc123"
}
```

Use this route for the webapp and any client that needs governed pause/resume behavior.

## 5. Task Endpoints

### `GET /v2/tasks/{task_id}`

Poll a compile task.

Task statuses:

- `pending`
- `running`
- `awaiting_decision`
- `complete`
- `error`
- `cancelled`

Important behavior:

- if a semantic gate blocks progress, status becomes `awaiting_decision`
- if a stable termination condition has been written to the task ledger, task returns `complete` with `termination_condition`
- progress includes structured insights, escalations, human decisions, and termination data

Example:

```json
{
  "task_id": "abc123",
  "status": "awaiting_decision",
  "result": {
    "success": false,
    "termination_condition": {
      "status": "awaiting_human",
      "reason": "human_decision_required"
    }
  },
  "progress": {
    "current_stage": "awaiting_decision",
    "escalations": [
      {
        "postcode": "STR.ENT.APP.WHAT.SFT",
        "question": "Which direction should Motherlabs lock for AuthService: storage strategy?",
        "options": [
          "Persist sessions in PostgreSQL",
          "Keep sessions stateless with JWT"
        ]
      }
    ]
  }
}
```

### `POST /v2/tasks/{task_id}/decisions`

Record a human decision against a paused compile.

Request body:

```json
{
  "postcode": "STR.ENT.APP.WHAT.SFT",
  "question": "Which direction should Motherlabs lock for AuthService: storage strategy?",
  "answer": "Persist sessions in PostgreSQL"
}
```

Response cases:

1. continuation task created:

```json
{
  "task_id": "abc123",
  "saved": true,
  "next_task_id": "abc124"
}
```

2. continuation blocked by guard:

```json
{
  "task_id": "abc123",
  "saved": true,
  "next_task_id": null,
  "termination_condition": {
    "status": "stalled",
    "reason": "continuation_cycle_detected"
  }
}
```

This second case is important.
Motherlabs will stop instead of blindly resuming if the same semantic pause keeps recurring or the continuation chain becomes unbounded.

### `DELETE /v2/tasks/{task_id}`

Cancel a pending compile task.

## 6. Swarm Endpoints

The swarm surface wraps compile into a broader multi-agent build path.

### `POST /v2/swarm/execute`

Queue an async swarm workflow.

### `POST /v2/swarm/execute/sync`

Run swarm synchronously.

### `GET /v2/swarm/status/{task_id}`

Poll swarm execution status.

### `GET /v2/swarm/result/{task_id}`

Get finished swarm result.

Use swarm when you want compile plus broader workflow orchestration.
Use `/v2/compile/async` when you want the direct Motherlabs compile loop.

## 7. Secondary V2 Endpoints

These routes still exist and are useful, but they are not the main public product loop:

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v2/compile-tree` | multi-subsystem compilation |
| `POST` | `/v2/materialize` | materialize a blueprint |
| `POST` | `/v2/recompile` | evolve an existing blueprint |
| `POST` | `/v2/validate` | trust-only compile |
| `GET` | `/v2/domains` | list domain adapters |
| `GET` | `/v2/domains/{name}` | domain adapter details |
| `GET` | `/v2/metrics` | platform metrics |
| `GET` | `/v2/corpus/benchmarks` | benchmark comparison |
| `POST` | `/v2/instance/peers` | register a known peer instance |
| `GET` | `/v2/instance/peers` | list known peer instances |
| `GET` | `/v2/instance/peers/status` | check peer liveness |

## 8. Health and Domain Discovery

### `GET /v2/health`

Basic platform health and domain availability.

### `GET /v2/domains`

Returns installed domain adapters.

Current public default domains include:

- `software`
- `process`
- `api`
- `agent_system`

## 9. Rate Limiting

Compile routes are rate-limited per IP when:

```text
MOTHERLABS_RATE_LIMIT_RPM > 0
```

Default in many local/dev setups is disabled or low-volume.

## 10. Practical Local Example

Start async compile:

```bash
curl -X POST http://127.0.0.1:8001/v2/compile/async \
  -H "Content-Type: application/json" \
  -H "X-LLM-API-Key: sk-ant-your-key" \
  -d '{
    "description": "Build a booking system for a tattoo studio with artist scheduling and deposits",
    "domain": "software"
  }'
```

Poll:

```bash
curl http://127.0.0.1:8001/v2/tasks/<task_id>
```

If paused:

```bash
curl -X POST http://127.0.0.1:8001/v2/tasks/<task_id>/decisions \
  -H "Content-Type: application/json" \
  -d '{
    "postcode": "STR.ENT.APP.WHAT.SFT",
    "question": "Which direction should Motherlabs lock for AuthService: storage strategy?",
    "answer": "Persist sessions in PostgreSQL"
  }'
```

## 11. Canonical Contract Files

When the API docs and implementation need to be checked against source, use:

- `api/v2/models.py`
- `api/v2/routes.py`
- `core/blueprint_protocol.py`

## 12. Legacy V1 Note

Legacy V1 routes still exist in `api/main.py` for compatibility and older tests.
They are not the primary public product surface anymore.

If you need them, inspect:

- `api/main.py`
- `api/models.py`

## 13. Error Shape

Common errors are returned as structured JSON.

Typical fields:

```json
{
  "error": "Compilation failed",
  "reason": "No LLM API key configured",
  "suggestion": "Set ANTHROPIC_API_KEY or pass via X-LLM-API-Key header"
}
```

Typical status codes:

- `200` success
- `400` malformed request or invalid domain
- `404` resource not found
- `422` compilation or validation failure
- `429` rate limit exceeded
- `503` engine unavailable
