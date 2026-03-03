# ---- Stage 1: Builder ----
FROM python:3.14-slim AS builder

WORKDIR /app

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip hatchling

# Copy dependency manifest first (layer caching)
COPY pyproject.toml README.md ./

# Install all dependencies (no source yet — just deps)
RUN pip install --no-cache-dir anthropic openai pyyaml requests fastapi uvicorn httpx websockets textual huey "discord.py>=2.3.0"

# Copy all source packages
COPY core/ core/
COPY agents/ agents/
COPY api/ api/
COPY persistence/ persistence/
COPY cli/ cli/
COPY codegen/ codegen/
COPY adapters/ adapters/
COPY kernel/ kernel/
COPY messaging/ messaging/
COPY motherlabs_platform/ motherlabs_platform/
COPY worker/ worker/
COPY bots/ bots/
COPY mother/ mother/

# Install the project itself
RUN pip install --no-cache-dir .

# ---- Stage 2: Runtime ----
FROM python:3.14-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY --from=builder /app/core core/
COPY --from=builder /app/agents agents/
COPY --from=builder /app/api api/
COPY --from=builder /app/persistence persistence/
COPY --from=builder /app/cli cli/
COPY --from=builder /app/codegen codegen/
COPY --from=builder /app/adapters adapters/
COPY --from=builder /app/kernel kernel/
COPY --from=builder /app/messaging messaging/
COPY --from=builder /app/motherlabs_platform motherlabs_platform/
COPY --from=builder /app/worker worker/
COPY --from=builder /app/bots bots/
COPY --from=builder /app/mother mother/

# Non-root user for security
RUN useradd -m -u 1000 motherlabs

# Consolidated data directory
RUN mkdir -p /data && chown motherlabs:motherlabs /data

ENV MOTHERLABS_DATA_DIR=/data

USER motherlabs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v2/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
