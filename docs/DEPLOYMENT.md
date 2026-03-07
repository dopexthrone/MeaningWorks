# Motherlabs Deployment Guide

This document covers the deployment shape that actually exists in this repo.
It is not a generic cloud architecture writeup.

Current production boundary:

- static Motherlabs webapp
- FastAPI API surface
- Huey worker for async compile tasks
- Caddy reverse proxy
- optional Litestream backup for SQLite data

## 1. Stack

```text
server
  -> Caddy
  -> static frontend export (frontend/out)
  -> FastAPI api container
  -> Huey worker container
  -> shared /data volume
  -> optional Litestream replication
```

The important point is simple:

- the workbench is static
- compile is API + worker
- state lives in the shared data volume

## 2. Prerequisites

- Linux host with Docker and Docker Compose
- DNS pointed at the server
- at least one provider API key
- a domain for the public Motherlabs surface

## 3. Clone And Configure

```bash
git clone https://github.com/alexrozex/Motherlabs-Semantic-Compiler.git
cd Motherlabs-Semantic-Compiler
cp .env.example .env
```

Minimum production `.env`:

```text
ANTHROPIC_API_KEY=sk-ant-...
CADDY_SITE_ADDRESS=compile.yourdomain.com
MOTHERLABS_REQUIRE_AUTH=1
MOTHERLABS_CORS_ORIGINS=https://compile.yourdomain.com
MOTHERLABS_TASK_SECRET=change-me-to-a-long-random-secret
MOTHERLABS_RATE_LIMIT_RPM=10
```

Optional provider keys:

```text
OPENAI_API_KEY=...
XAI_API_KEY=...
SENTRY_DSN=...
```

## 4. Build The Frontend Export

Caddy serves `frontend/out`, so build that before starting the stack.

```bash
cd frontend
npm ci
NEXT_PUBLIC_API_URL=https://compile.yourdomain.com npm run build
cd ..
```

The frontend uses static export mode in `frontend/next.config.mjs`, so the output is written to `frontend/out/`.

## 5. Start The Stack

```bash
docker compose up -d --build
```

This brings up:

- `api` on internal port `8000`
- `worker` consuming Huey tasks
- `caddy` on `80/443`

If you want backup replication too:

```bash
docker compose --profile backup up -d --build
```

## 6. Verify

Health check:

```bash
curl https://compile.yourdomain.com/v2/health
```

Useful checks:

```bash
docker compose ps
docker compose logs api
docker compose logs worker
docker compose logs caddy
```

## 7. Service Roles

### `api`

FastAPI application exposing:

- `/v2/compile`
- `/v2/compile/async`
- `/v2/tasks/{task_id}`
- `/v2/tasks/{task_id}/decisions`
- `/v2/health`

Important runtime facts:

- listens on `8000`
- stores runtime data in `/data`
- shares task secret and storage with worker

### `worker`

Huey consumer for async compile and swarm tasks.

Current command in `docker-compose.yml`:

```text
python -m huey.bin.huey_consumer worker.config.huey --workers=2 --worker-type=thread
```

If async compile is broken in production, check worker first.

### `caddy`

Public entrypoint:

- serves `frontend/out`
- proxies `/v1/*`, `/v2/*`, `/docs`, and `/openapi.json` to `api:8000`
- terminates TLS

Source of truth:

- `Caddyfile`

### `litestream` (optional)

Optional SQLite replication to Cloudflare R2.

Required vars:

```text
LITESTREAM_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
LITESTREAM_BUCKET=motherlabs-backup
LITESTREAM_ACCESS_KEY_ID=...
LITESTREAM_SECRET_ACCESS_KEY=...
```

## 8. Security And Ops Notes

- `MOTHERLABS_REQUIRE_AUTH=1` should be on in production.
- `MOTHERLABS_TASK_SECRET` must be changed from the default.
- lock `MOTHERLABS_CORS_ORIGINS` to your public domain.
- compile endpoints can be rate-limited with `MOTHERLABS_RATE_LIMIT_RPM`.
- user-passed provider keys are encrypted before async task handoff.

## 9. Updating

Typical update flow:

```bash
git pull
cd frontend && NEXT_PUBLIC_API_URL=https://compile.yourdomain.com npm run build && cd ..
docker compose up -d --build
```

## 10. Scaling

The first scaling lever is usually worker throughput, not the frontend.

To increase worker concurrency, edit the worker command in `docker-compose.yml`.
For example:

```yaml
worker:
  command: ["python", "-m", "huey.bin.huey_consumer", "worker.config.huey", "--workers=4", "--worker-type=thread"]
```

For more parallel workers:

```bash
docker compose up -d --scale worker=3
```

## 11. Backup And Recovery

If Litestream is enabled, restore from the configured bucket into `/data`.

If you are doing manual backup, copy the shared data volume contents out of the running container set.

At minimum, preserve:

- SQLite data files under `/data`
- deployment `.env`
- frontend build inputs if you need deterministic rebuilds

## 12. Troubleshooting

### Frontend is blank or stale

- rebuild `frontend/out`
- confirm `NEXT_PUBLIC_API_URL`
- check Caddy is serving the latest export

### Async compile never completes

- check `docker compose logs worker`
- confirm `MOTHERLABS_TASK_SECRET` matches across api and worker
- confirm the shared `/data` volume is mounted

### Health endpoint works but compile fails

- check provider API keys
- check `MOTHERLABS_REQUIRE_AUTH`
- inspect `api` logs for provider or validation errors
