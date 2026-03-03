"""
Engine bridge — the single integration seam between TUI and core engine.

Only file in mother/ that imports from core/. All engine interaction
flows through this module.
"""

import asyncio
import queue
import threading
import time
import logging
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator

logger = logging.getLogger("mother.bridge")

# Per-million-token rates (USD) — conservative estimates
COST_RATES: Dict[str, Dict[str, float]] = {
    "claude":  {"input": 3.0,  "output": 15.0},
    "openai":  {"input": 2.0,  "output": 8.0},
    "grok":    {"input": 2.0,  "output": 20.0},
    "gemini":  {"input": 0.10, "output": 0.40},
    "local":   {"input": 0.0,  "output": 0.0},
}


class EngineBridge:
    """Async-to-sync bridge for the Motherlabs engine.

    Wraps synchronous engine calls in asyncio.to_thread() for use
    from Textual's async event loop.
    """

    def __init__(
        self,
        provider: str = "claude",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        file_access: bool = True,
        screen_capture_enabled: bool = False,
        microphone_enabled: bool = False,
        openai_api_key: Optional[str] = None,
        camera_enabled: bool = False,
        pipeline_mode: str = "staged",
        max_concurrent_appendages: int = 5,
        min_uses_to_solidify: int = 5,
    ):
        self._provider = provider
        self._model = model
        self._chat_model = None  # set via set_chat_model() for tiered routing
        self._api_key = api_key
        self._llm = None
        self._chat_llm = None  # separate client for conversation (cheaper model)
        self._engine = None
        self._filesystem = None
        self._screen_bridge = None
        self._microphone_bridge = None
        self._camera_bridge = None
        self._voice_bridge_ref = None
        self._file_access = file_access
        self._screen_capture_enabled = screen_capture_enabled
        self._microphone_enabled = microphone_enabled
        self._openai_api_key = openai_api_key
        self._camera_enabled = camera_enabled
        self._pipeline_mode = pipeline_mode
        self._max_concurrent_appendages = max_concurrent_appendages
        self._min_uses_to_solidify = min_uses_to_solidify
        self._session_cost: float = 0.0
        self._last_call_cost: float = 0.0
        self._engine_cost_baseline: float = 0.0  # Track engine cost delta between compiles
        self._compile_done: asyncio.Event = asyncio.Event()
        self._compile_done.set()  # Not compiling initially
        self._insight_queue: queue.Queue = queue.Queue()  # Thread-safe for cross-thread insight pushing
        self._build_done: asyncio.Event = asyncio.Event()
        self._build_done.set()  # Not building initially
        self._build_phase_queue: queue.Queue = queue.Queue()  # Thread-safe build phase events
        # Streaming chat state
        self._chat_token_queue: queue.Queue = queue.Queue()
        self._chat_stream_done = threading.Event()
        self._chat_stream_done.set()  # Not streaming initially
        self._chat_stream_result: Optional[str] = None
        self._chat_stream_usage: Dict[str, Any] = {}
        # Self-build streaming state
        self._self_build_event_queue: queue.Queue = queue.Queue()
        self._self_build_done: asyncio.Event = asyncio.Event()
        self._self_build_done.set()  # Not building initially
        self._self_build_events: list = []  # Raw events for log writing
        # Appendage processes (appendage_id -> AppendageProcess)
        self._appendage_processes: Dict[int, Any] = {}
        # Tool health tracking (tool_name -> {runs, failures})
        self._tool_health: Dict[str, Dict[str, int]] = {}

    def _get_llm(self):
        """Lazy-load LLM client. Local provider uses FailoverClient with API fallback."""
        if self._llm is None:
            from core.llm import create_client
            if self._provider in ("local", "ollama"):
                from core.llm import OllamaClient, FailoverClient
                import os
                ollama_url = os.environ.get("LOCAL_MODEL_URL", "http://localhost:11434")
                primary = OllamaClient(
                    base_url=ollama_url,
                    model=self._model or "llama3:8b",
                    deterministic=False,
                )
                # Build fallback chain from available API keys
                fallbacks = []
                for prov, env in [("claude", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY"),
                                  ("gemini", "GOOGLE_API_KEY"), ("grok", "XAI_API_KEY")]:
                    if os.environ.get(env):
                        try:
                            fallbacks.append(create_client(provider=prov, deterministic=False))
                        except Exception:
                            pass
                if fallbacks:
                    self._llm = FailoverClient([primary] + fallbacks)
                else:
                    self._llm = primary
            else:
                self._llm = create_client(
                    provider=self._provider,
                    model=self._model,
                    api_key=self._api_key,
                    deterministic=False,
                )
        return self._llm

    def set_chat_model(self, model: str) -> None:
        """Set a separate (cheaper) model for conversation.

        When set, chat() and stream_chat() use this model instead of the
        main model. The main model is still used for compilation and
        code engine. This enables tiered routing: Haiku for chat,
        Sonnet for builds.
        """
        if model and model != self._model:
            self._chat_model = model
            self._chat_llm = None  # force re-creation on next call
            logger.info(f"Chat model set to {model} (main: {self._model})")

    def _get_chat_llm(self):
        """Get LLM client for conversation. Falls back to main LLM if no chat model set."""
        if not self._chat_model:
            return self._get_llm()
        if self._chat_llm is None:
            from core.llm import create_client
            self._chat_llm = create_client(
                provider=self._provider,
                model=self._chat_model,
                api_key=self._api_key,
                deterministic=False,
            )
        return self._chat_llm

    def _get_engine(self):
        """Lazy-load engine with insight callback wired to queue.

        When a chat_model is set (tiered routing), the engine uses the cheap
        model for dialogue agents and the main model for synthesis/verification.
        """
        if self._engine is None:
            from core.engine import MotherlabsEngine

            def _on_insight(msg: str) -> None:
                try:
                    self._insight_queue.put_nowait(msg)
                except queue.Full:
                    pass

            # Tiered compilation: cheap model for dialogue, quality model for synthesis
            quality_llm = None
            engine_model = self._model
            if self._chat_model and self._chat_model != self._model:
                # chat_model is cheap (e.g. Haiku) — use it for dialogue agents
                # main model (e.g. Sonnet) becomes the quality LLM for synthesis/verify
                from core.llm import create_client
                engine_model = self._chat_model
                try:
                    quality_llm = create_client(
                        provider=self._provider,
                        model=self._model,
                        api_key=self._api_key,
                    )
                    logger.info(
                        f"Tiered compilation: dialogue={self._chat_model}, "
                        f"synthesis={self._model}"
                    )
                except Exception as e:
                    logger.warning(f"Quality LLM creation failed, using single model: {e}")
                    engine_model = self._model

            self._engine = MotherlabsEngine(
                provider=self._provider,
                model=engine_model,
                api_key=self._api_key,
                on_insight=_on_insight,
                pipeline_mode=self._pipeline_mode,
                quality_llm=quality_llm,
            )
        return self._engine

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        images: Optional[List[str]] = None,
    ) -> str:
        """Send messages to LLM and get response.

        Uses the chat LLM (cheaper model if configured) with temperature=0.7.
        Runs in thread to avoid blocking the event loop.

        images: optional list of base64-encoded PNG strings to attach
        to the last user message as vision content.
        """
        llm = self._get_chat_llm()

        # Inject images into last user message if provided
        if images:
            messages = self._inject_images(messages, images)

        def _call():
            return llm.complete(
                messages=messages,
                system=system_prompt,
                temperature=0.7,
            )

        start = time.monotonic()
        response = await asyncio.to_thread(_call)
        elapsed = time.monotonic() - start

        # Track cost
        usage = getattr(llm, "last_usage", {})
        self._track_cost(usage)

        logger.debug(f"Chat response in {elapsed:.1f}s, tokens: {usage}")
        return response

    # --- Streaming chat ---

    def begin_chat_stream(self) -> None:
        """Prepare for streaming. Call before creating stream task."""
        while not self._chat_token_queue.empty():
            try:
                self._chat_token_queue.get_nowait()
            except queue.Empty:
                break
        self._chat_stream_done.clear()
        self._chat_stream_result = None
        self._chat_stream_usage = {}

    def _stream_chat_sync(self, messages, system_prompt, images=None, max_tokens=4096):
        """Blocking: stream tokens from LLM, push to queue. Runs in thread."""
        llm = self._get_chat_llm()
        if images:
            messages = self._inject_images(messages, images)
        collected = []
        try:
            for token in llm.stream(messages, system=system_prompt, temperature=0.7, max_tokens=max_tokens):
                collected.append(token)
                self._chat_token_queue.put_nowait(token)
        finally:
            self._chat_stream_result = "".join(collected)
            self._chat_stream_usage = getattr(llm, "last_usage", {})
            self._track_cost(self._chat_stream_usage)
            self._chat_stream_done.set()

    async def stream_chat(self, messages, system_prompt, images=None, max_tokens=4096):
        """Async wrapper for streaming chat. Runs LLM in thread."""
        await asyncio.to_thread(
            self._stream_chat_sync, messages, system_prompt, images, max_tokens
        )

    async def stream_chat_tokens(self) -> AsyncGenerator[str, None]:
        """Yield tokens as they arrive from the LLM thread."""
        while True:
            if self._chat_stream_done.is_set() and self._chat_token_queue.empty():
                return
            try:
                token = self._chat_token_queue.get_nowait()
                if token is None:
                    return
                yield token
            except queue.Empty:
                if self._chat_stream_done.is_set():
                    return
                await asyncio.sleep(0.05)

    def get_stream_result(self) -> Optional[str]:
        """Full assembled text from completed stream."""
        return self._chat_stream_result

    def cancel_chat_stream(self) -> None:
        """Signal consumer to stop reading tokens.

        Drains token queue and sets done flag. Background LLM thread
        continues until API call completes (tokens discarded).
        """
        while not self._chat_token_queue.empty():
            try:
                self._chat_token_queue.get_nowait()
            except queue.Empty:
                break
        self._chat_stream_done.set()

    # ── Duplex voice support ──────────────────────────────────────────

    _duplex_system_prompt: str = ""

    def get_duplex_system_prompt(self) -> str:
        """Return the current system prompt for duplex voice sessions."""
        return self._duplex_system_prompt

    def set_duplex_system_prompt(self, prompt: str) -> None:
        """Set the system prompt used by the local LLM server for duplex voice."""
        self._duplex_system_prompt = prompt

    async def stream_chat_for_duplex(
        self, messages: List[Dict], system_prompt: str
    ) -> AsyncGenerator[str, None]:
        """Async generator yielding LLM tokens for duplex voice.

        Unlike stream_chat() which uses token queue + thread, this yields
        directly from a threaded LLM call via an asyncio queue bridge.
        Used by LocalLLMServer to serve ElevenLabs chat completions.
        """
        token_q: queue.Queue = queue.Queue()  # Thread-safe stdlib queue
        done = threading.Event()

        def _run_llm():
            llm = self._get_chat_llm()
            try:
                for token in llm.stream(
                    messages, system=system_prompt, temperature=0.7, max_tokens=300
                ):
                    token_q.put_nowait(token)
            except Exception as e:
                logger.warning(f"Duplex LLM stream error: {e}")
            finally:
                done.set()

        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, _run_llm)

        while True:
            if done.is_set() and token_q.empty():
                return
            try:
                token = token_q.get_nowait()
                yield token
            except queue.Empty:
                if done.is_set():
                    return
                await asyncio.sleep(0.02)

    # Models known to support vision (image input)
    _VISION_MODELS = frozenset({
        # xAI
        "grok-2-vision", "grok-2-vision-latest",
        "grok-4-1-fast-reasoning", "grok-4",
        # OpenAI
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-vision-preview",
        "o1", "o1-mini", "o3", "o3-mini", "o4-mini",
        # Anthropic (all Claude 3+ support vision)
        "claude-sonnet-4-20250514", "claude-opus-4-20250514",
        "claude-haiku-4-5-20251001", "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022", "claude-3-opus-20240229",
        # Gemini
        "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro",
    })

    def _chat_model_supports_vision(self) -> bool:
        """Check if the current chat model supports image inputs."""
        model = self._chat_model or self._model or ""
        # Exact match
        if model in self._VISION_MODELS:
            return True
        # Prefix match for versioned models
        for vm in self._VISION_MODELS:
            if model.startswith(vm):
                return True
        # Claude provider: all models support vision
        if self._provider == "claude":
            return True
        # Gemini provider: all models support vision
        if self._provider == "gemini":
            return True
        # Local (Ollama): allow — Ollama handles unsupported formats gracefully
        if self._provider == "local":
            return True
        return False

    def _inject_images(
        self,
        messages: List[Dict[str, Any]],
        images: List,
    ) -> List[Dict[str, Any]]:
        """Convert last user message to multimodal format with images.

        images: list of base64 strings OR (base64, media_type) tuples.
        Provider-specific formatting:
        - Claude: content as list with text + image source blocks
        - OpenAI/Grok: content as list with text + image_url blocks
        - Gemini: content as list with text + image_url blocks (OpenAI-compatible)

        Returns messages unchanged if the chat model doesn't support vision.
        """
        if not messages or not images:
            return messages

        # Gate: skip image injection for non-vision models
        if not self._chat_model_supports_vision():
            logger.debug(
                f"Skipping image injection: {self._chat_model or self._model} "
                "does not support vision"
            )
            return messages

        # Normalize to (b64, media_type) tuples
        normalized = []
        for item in images:
            if isinstance(item, tuple):
                normalized.append(item)
            else:
                normalized.append((item, "image/png"))

        # Make a shallow copy so we don't mutate the original
        messages = list(messages)

        # Find last user message
        last_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_idx = i
                break

        if last_idx is None:
            return messages

        msg = messages[last_idx]
        text = msg.get("content", "")

        # Build multimodal content
        if self._provider in ("local", "ollama"):
            # Ollama format: images field on the message (list of base64 strings)
            messages[last_idx] = {
                "role": "user",
                "content": text,
                "images": [img_b64 for img_b64, _ in normalized],
            }
            return messages
        elif self._provider == "claude":
            # Anthropic format
            content = [{"type": "text", "text": text}]
            for img_b64, media_type in normalized:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_b64,
                    },
                })
        else:
            # OpenAI / Grok / Gemini format
            content = [{"type": "text", "text": text}]
            for img_b64, media_type in normalized:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{img_b64}",
                    },
                })

        messages[last_idx] = {"role": "user", "content": content}
        return messages

    async def compile(
        self,
        description: str,
        mode: str = "build",
    ) -> Any:
        """Compile a description into a blueprint (or context/exploration map).

        Returns CompileResult. Pushes insights to the queue.
        Sets _compile_done event for stream_insights sentinel.

        Args:
            description: What to compile.
            mode: Compilation mode — "build", "context", "explore", or "self".
        """
        engine = self._get_engine()
        self._compile_done.clear()

        def _call():
            return engine.compile(description, pipeline_mode=mode)

        try:
            result = await asyncio.to_thread(_call)
            # Extract compile cost from engine's internal tracking
            engine_total = getattr(engine, "_session_cost_usd", 0.0)
            compile_cost = engine_total - self._engine_cost_baseline
            self._engine_cost_baseline = engine_total
            if compile_cost > 0:
                self._last_call_cost = compile_cost
                self._session_cost += compile_cost

            # CONTEXT mode post-processing: persist to memory
            context_map = getattr(result, "context_map", None)
            if context_map and mode == "context":
                try:
                    from mother.memory_indexer import index_context_map
                    stats = index_context_map(context_map, description)
                    logger.info(
                        f"Context indexed: {stats.get('facts_saved', 0)} facts "
                        f"({stats.get('concepts_indexed', 0)} concepts, "
                        f"{stats.get('vocabulary_indexed', 0)} vocab)"
                    )
                except Exception as e:
                    logger.debug(f"Context memory persistence skipped: {e}")

            return result
        finally:
            self._compile_done.set()

    def get_semantic_grid(self, result: Any) -> dict | None:
        """Extract semantic grid data from a CompileResult.

        Returns dict with 'nav' (text), 'manifest' (dict), 'cells', 'layers'
        or None if kernel compilation didn't run.
        """
        return getattr(result, "semantic_grid", None)

    def get_semantic_nav(self, result: Any) -> str:
        """Extract the navigation-layer text from a CompileResult.

        Returns the lightweight grid serialization (~30 tokens/node)
        or empty string if kernel compilation didn't run.
        """
        grid_data = self.get_semantic_grid(result)
        return grid_data.get("nav", "") if grid_data else ""

    def get_depth_chains(self, result: Any) -> list[dict]:
        """Extract depth chains from a CompileResult for intent chaining.

        Returns list of chain dicts with chain_type, intent_text, source_postcodes,
        priority, layer, concern — or empty list if no chains.
        """
        return getattr(result, "depth_chains", []) or []

    @staticmethod
    def get_context_map(result: Any) -> dict | None:
        """Extract context map from a CONTEXT mode CompileResult.

        Returns dict with concepts, relationships, assumptions, unknowns,
        vocabulary, memory_connections, confidence — or None if not a CONTEXT result.
        """
        return getattr(result, "context_map", None)

    @staticmethod
    def get_exploration_map(result: Any) -> dict | None:
        """Extract exploration map from an EXPLORE mode CompileResult.

        Returns dict with insights, frontier_questions, adjacent_domains,
        alternative_framings, depth_chains — or None if not an EXPLORE result.
        """
        return getattr(result, "exploration_map", None)

    # --- World Grid Methods (Phase 6) ---

    def get_world_grid(self) -> Any:
        """Load the persistent world grid from maps.db.

        Returns Grid or None if not found.
        """
        try:
            from kernel.store import load_grid
            return load_grid("world")
        except Exception as e:
            logger.debug(f"World grid load failed: {e}")
            return None

    def bootstrap_world_grid(self) -> Any:
        """Create or extend the world grid. Returns the Grid."""
        try:
            from kernel.store import load_grid, save_grid
            from kernel.world_grid import bootstrap_world_grid as _bootstrap

            existing = load_grid("world")
            grid = _bootstrap(existing)
            save_grid(grid, "world", name="Mother World Model")
            return grid
        except Exception as e:
            logger.debug(f"World grid bootstrap failed: {e}")
            # Return an in-memory grid as fallback
            from kernel.world_grid import bootstrap_world_grid as _bootstrap
            return _bootstrap()

    def save_world_grid(self, grid: Any) -> None:
        """Persist the world grid to maps.db."""
        try:
            from kernel.store import save_grid
            save_grid(grid, "world", name="Mother World Model")
        except Exception as e:
            logger.debug(f"World grid save failed: {e}")

    def fill_world_cell(self, grid: Any, **fill_kwargs) -> Any:
        """Fill a cell in the world grid. Returns FillResult or None."""
        try:
            from kernel.ops import fill as grid_fill
            return grid_fill(grid, **fill_kwargs)
        except Exception as e:
            logger.debug(f"World grid fill failed: {e}")
            return None

    def score_world_candidates(self, grid: Any, **kwargs) -> list:
        """Score world grid candidates using world-model navigator.

        Returns list of ScoredCell or empty list on error.
        """
        try:
            from kernel.navigator import score_world_candidates as _score
            return _score(grid, **kwargs)
        except Exception as e:
            logger.debug(f"World grid scoring failed: {e}")
            return []

    def dispatch_from_cell(self, postcode_key: str, score: float, primitive: str = "") -> Any:
        """Map a cell to an action dispatch. Returns ActionDispatch or None."""
        try:
            from kernel.dispatch import dispatch_from_cell as _dispatch
            return _dispatch(postcode_key, score, primitive)
        except Exception as e:
            logger.debug(f"World grid dispatch failed: {e}")
            return None

    def merge_compilation_into_world(self, world_grid: Any, result: Any, project_id: str = "session") -> int:
        """Merge compilation results into the world grid. Returns count merged."""
        try:
            from kernel.world_grid import merge_compilation_into_world as _merge

            # Extract the kernel grid from CompileResult
            grid_data = self.get_semantic_grid(result)
            if grid_data is None:
                return 0

            # Load compilation grid from store
            from kernel.store import load_grid
            comp_grid = load_grid("session")
            if comp_grid is None:
                return 0

            return _merge(world_grid, comp_grid, project_id)
        except Exception as e:
            logger.debug(f"Compilation merge failed: {e}")
            return 0

    def apply_staleness_decay(self, grid: Any) -> int:
        """Decay stale observation cells. Returns count decayed."""
        try:
            from kernel.world_grid import apply_staleness_decay as _decay
            return _decay(grid)
        except Exception as e:
            logger.debug(f"Staleness decay failed: {e}")
            return 0

    def world_grid_health(self, grid: Any) -> dict:
        """Get world grid health diagnostics."""
        try:
            from kernel.world_grid import world_grid_health as _health
            return _health(grid)
        except Exception as e:
            logger.debug(f"World grid health failed: {e}")
            return {}

    def get_feedback_report(self) -> dict | None:
        """Get governor feedback report from compilation history.

        Returns dict with outcomes_analyzed, rejection_rate, weaknesses,
        strengths, compiler_hints, trend — or None if no engine.
        """
        try:
            from mother.governor_feedback import analyze_outcomes
            outcomes = getattr(self._engine, "_compilation_outcomes", [])
            if not outcomes:
                return None
            report = analyze_outcomes(outcomes)
            return {
                "outcomes_analyzed": report.outcomes_analyzed,
                "rejection_rate": report.rejection_rate,
                "weaknesses": [
                    {"dimension": w.dimension, "severity": w.severity,
                     "mean_score": w.mean_score, "pattern": w.pattern}
                    for w in report.weaknesses
                ],
                "strengths": list(report.strengths),
                "compiler_hints": list(report.compiler_hints),
                "trend": report.trend,
            }
        except Exception:
            return None

    def get_improvement_goals(self) -> dict | None:
        """Generate improvement goals from grid + feedback + observations.

        Returns dict with goals list, summary, counts — or None on failure.
        """
        try:
            from mother.goal_generator import (
                goals_from_grid,
                goals_from_feedback,
                goals_from_anomalies,
                generate_goal_set,
            )
            from mother.governor_feedback import analyze_outcomes

            # Grid goals
            grid_goals = []
            try:
                from kernel.grid import Grid
                grid = getattr(self._engine, "_kernel_grid", None)
                if grid and hasattr(grid, "cells"):
                    cell_data = [
                        (pk, c.confidence, c.fill.name, c.primitive)
                        for pk, c in grid.cells.items()
                    ]
                    activated = grid.activated_layers if hasattr(grid, "activated_layers") else set()
                    grid_goals = goals_from_grid(cell_data, activated, len(grid.cells))
            except Exception:
                pass

            # Feedback goals
            feedback_goals = []
            try:
                outcomes = getattr(self._engine, "_compilation_outcomes", [])
                if outcomes:
                    report = analyze_outcomes(outcomes)
                    weaknesses = [
                        (w.dimension, w.severity, w.mean_score)
                        for w in report.weaknesses
                    ]
                    feedback_goals = goals_from_feedback(
                        weaknesses, report.rejection_rate, report.trend
                    )
            except Exception:
                pass

            # Anomaly goals (placeholder — wired when observer runs)
            anomaly_goals = goals_from_anomalies(0, [])

            goal_set = generate_goal_set(grid_goals, feedback_goals, anomaly_goals)
            return {
                "goals": [
                    {"id": g.goal_id, "priority": g.priority,
                     "category": g.category, "description": g.description,
                     "source": g.source}
                    for g in goal_set.goals
                ],
                "summary": goal_set.analysis_summary,
                "total": goal_set.total_goals,
                "critical": goal_set.critical_count,
            }
        except Exception:
            return None

    def sync_goals_to_store(self, db_path) -> int:
        """Sync self-generated improvement goals into the persistent GoalStore.

        Generates ImprovementGoal objects from grid + feedback, deduplicates
        against existing active goals, and adds new ones.
        Returns count of new goals added. Never raises.
        """
        try:
            from mother.goals import GoalStore
            from pathlib import Path

            goal_data = self.get_improvement_goals()
            if not goal_data or not goal_data.get("goals"):
                return 0

            _priority_map = {
                "critical": "urgent",
                "high": "high",
                "medium": "normal",
                "low": "low",
            }

            gs = GoalStore(Path(db_path) if not isinstance(db_path, Path) else db_path)
            existing = gs.active(limit=50)

            from mother.goals import goal_dedup_key
            existing_keys = [goal_dedup_key(g.description) for g in existing]

            added = 0
            for goal in goal_data["goals"]:
                desc = goal.get("description", "")
                if not desc:
                    continue
                # Dedup: strip numbers/percentages so "5 cells" matches "71 cells"
                key = goal_dedup_key(desc)
                if any(key in ek or ek in key for ek in existing_keys):
                    continue
                mapped_priority = _priority_map.get(goal.get("priority", "medium"), "normal")
                gs.add(
                    description=desc,
                    source="system",
                    priority=mapped_priority,
                )
                existing_keys.append(key)  # prevent intra-batch dupes
                added += 1

            gs.close()
            return added
        except Exception:
            return 0

    def record_task_failure(
        self,
        db_path,
        task_type: str,
        description: str,
        error: str,
    ) -> bool:
        """Record a task failure as an improvement goal for self-repair.

        Creates a goal in GoalStore with enough signal words for the daemon
        to route it through the self_build pipeline. Returns True if goal added.
        """
        try:
            from mother.goals import GoalStore
            from pathlib import Path

            # Build a goal description that:
            # 1. Describes the failure clearly
            # 2. Contains signal words for _is_self_build_goal (needs 2+)
            # 3. Is actionable — tells the self-build what to fix
            error_short = error[:200].strip()
            goal_desc = (
                f"[SELF-REPAIR] Fix {task_type} capability failure. "
                f"Error: {error_short}. "
                f"Strengthen resilience of {task_type} error handling in Mother's codebase. "
                f"Original task: {description[:150]}"
            )

            gs = GoalStore(Path(db_path) if not isinstance(db_path, Path) else db_path)

            # Dedup: don't add if a similar active goal exists
            existing = gs.active(limit=30)
            goal_lower = goal_desc.lower()
            for g in existing:
                if "[self-repair]" in g.description.lower() and error_short[:60].lower() in g.description.lower():
                    gs.close()
                    return False  # Already tracking this failure

            gs.add(
                description=goal_desc,
                source=f"self-repair:{task_type}",
                priority="high",
            )
            gs.close()
            logger.info(f"Self-repair goal created for {task_type} failure: {error_short[:80]}")
            return True
        except Exception as e:
            logger.debug(f"record_task_failure skipped: {e}")
            return False

    # --- Weekly build governance ---

    def get_weekly_briefing(self) -> Optional[dict]:
        """Return current week's briefing from ScheduleStore.

        Returns dict with 'briefing' (formatted markdown), 'items' (raw list),
        'status', 'week_start'. None if no briefing exists.
        """
        try:
            from mother.scheduler import ScheduleStore, format_briefing_markdown
            maps_path = Path.home() / ".motherlabs" / "maps.db"
            if not maps_path.exists():
                return None
            sched = ScheduleStore(maps_path)
            week = sched.current_week()
            sched.close()
            if not week or not week.get("briefing"):
                return None
            briefing = week["briefing"]
            return {
                "briefing_markdown": format_briefing_markdown(briefing),
                "items": [
                    {
                        "goal_id": item.goal_id,
                        "description": item.description,
                        "priority": item.priority,
                        "risk": item.risk,
                        "estimated_cost": item.estimated_cost,
                        "rationale": item.rationale,
                    }
                    for item in briefing.items
                ],
                "total_goals": briefing.total_goals,
                "summary": briefing.summary,
                "status": week["status"],
                "week_start": week["week_start"],
            }
        except Exception as e:
            logger.debug(f"get_weekly_briefing skipped: {e}")
            return None

    def approve_weekly_builds(self, goal_ids: list) -> bool:
        """Approve goals for the weekly build window.

        Sets goals to 'approved' in GoalStore and records approval in ScheduleStore.
        Returns True on success.
        """
        try:
            from mother.scheduler import ScheduleStore
            from mother.goals import GoalStore

            # Approve in GoalStore
            db_path = Path.home() / ".motherlabs" / "history.db"
            gs = GoalStore(db_path)
            int_ids = [int(gid) for gid in goal_ids]
            gs.approve_goals(int_ids)
            gs.close()

            # Record in ScheduleStore
            maps_path = Path.home() / ".motherlabs" / "maps.db"
            sched = ScheduleStore(maps_path)
            week = sched.current_week()
            if week:
                str_ids = [str(gid) for gid in goal_ids]
                sched.save_approval(week["week_start"], str_ids)
            sched.close()

            logger.info(f"Weekly builds approved: {goal_ids}")
            return True
        except Exception as e:
            logger.debug(f"approve_weekly_builds failed: {e}")
            return False

    def get_build_report(self) -> Optional[dict]:
        """Return the last execution report.

        Returns dict with 'report_markdown', 'total_cost', 'success_count',
        'failure_count'. None if no report exists.
        """
        try:
            from mother.scheduler import ScheduleStore, format_report_markdown
            maps_path = Path.home() / ".motherlabs" / "maps.db"
            if not maps_path.exists():
                return None
            sched = ScheduleStore(maps_path)
            report = sched.last_report()
            sched.close()
            if not report:
                return None
            return {
                "report_markdown": format_report_markdown(report),
                "total_cost": report.total_cost,
                "success_count": report.success_count,
                "failure_count": report.failure_count,
                "skipped_count": report.skipped_count,
                "window_date": report.window_date,
            }
        except Exception as e:
            logger.debug(f"get_build_report skipped: {e}")
            return None

    def is_build_window_active(self) -> bool:
        """Check if we're currently inside the build window."""
        try:
            from datetime import datetime as _dt
            from mother.scheduler import BuildWindow, is_in_build_window
            from mother.config import load_config

            cfg = load_config()
            if not cfg.weekly_build_enabled:
                return False
            window = BuildWindow(
                start_hour=cfg.build_window_start_hour,
                end_hour=cfg.build_window_end_hour,
                day_of_week=cfg.build_window_day,
            )
            return is_in_build_window(_dt.now(), window)
        except Exception:
            return False

    def begin_build(self) -> None:
        """Signal that a build is about to start.

        Call BEFORE asyncio.create_task(bridge.build(...)) to avoid
        a race where stream_build_phases() sees _build_done=True and exits
        before build() has a chance to clear it.
        """
        self._build_done.clear()

    async def build(
        self,
        description: str,
        output_dir: str = "./output",
    ) -> Any:
        """Compile and emit as a project.

        Returns AgentResult with project manifest.
        Pushes phase events to _build_phase_queue for live streaming.
        """
        from core.agent_orchestrator import AgentOrchestrator, AgentConfig

        config = AgentConfig(
            output_dir=output_dir,
            write_project=True,
            build=True,
        )
        engine = self._get_engine()
        phase_q = self._build_phase_queue
        self._build_done.clear()

        def _on_progress(phase: str, detail: str) -> None:
            try:
                phase_q.put_nowait((phase, detail))
            except queue.Full:
                pass

        def _call():
            orchestrator = AgentOrchestrator(
                engine=engine,
                config=config,
            )
            return orchestrator.run(description, on_progress=_on_progress)

        try:
            result = await asyncio.to_thread(_call)
            # Track cost delta from engine
            engine_total = getattr(engine, "_session_cost_usd", 0.0)
            build_cost = engine_total - self._engine_cost_baseline
            self._engine_cost_baseline = engine_total
            if build_cost > 0:
                self._last_call_cost = build_cost
                self._session_cost += build_cost
            return result
        finally:
            self._build_done.set()

    async def stream_build_phases(self) -> AsyncGenerator[tuple, None]:
        """Yield (phase, detail) tuples as build progresses.

        Polls thread-safe queue. Terminates when _build_done is set
        and queue is empty.
        """
        while True:
            if self._build_done.is_set() and self._build_phase_queue.empty():
                return
            try:
                event = self._build_phase_queue.get_nowait()
                yield event
            except queue.Empty:
                if self._build_done.is_set():
                    return
                await asyncio.sleep(0.2)

    # Sentinel pushed to _self_build_event_queue to signal stream end.
    _SELF_BUILD_SENTINEL = None

    def begin_self_build(self) -> None:
        """Signal that a self-build is about to start.

        Call BEFORE asyncio.create_task(bridge.self_build(...)) to avoid
        a race where stream_self_build_events() exits before events arrive.
        Drains any leftover events from a previous build.
        """
        self._self_build_done.clear()
        self._self_build_events = []
        # Drain stale events from previous build
        while not self._self_build_event_queue.empty():
            try:
                self._self_build_event_queue.get_nowait()
            except queue.Empty:
                break

    async def stream_self_build_events(self) -> AsyncGenerator[Dict, None]:
        """Yield event dicts as the self-build progresses.

        Terminates when a sentinel (None) is received from the queue,
        guaranteeing all events pushed by self_build() are consumed
        before the generator exits.
        """
        while True:
            try:
                event = self._self_build_event_queue.get_nowait()
                if event is self._SELF_BUILD_SENTINEL:
                    return
                yield event
            except queue.Empty:
                # Check done flag as fallback (in case sentinel was lost)
                if self._self_build_done.is_set():
                    # One final drain to catch stragglers
                    while not self._self_build_event_queue.empty():
                        try:
                            event = self._self_build_event_queue.get_nowait()
                            if event is self._SELF_BUILD_SENTINEL:
                                return
                            yield event
                        except queue.Empty:
                            break
                    return
                await asyncio.sleep(0.15)

    async def drain_insights(self) -> List[str]:
        """Drain all pending insights from the queue."""
        insights = []
        while not self._insight_queue.empty():
            try:
                insights.append(self._insight_queue.get_nowait())
            except queue.Empty:
                break
        return insights

    def _track_cost(self, usage: Dict[str, Any]) -> None:
        """Estimate and accumulate session cost using provider rate table."""
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        rates = COST_RATES.get(self._provider, COST_RATES["claude"])
        cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
        self._last_call_cost = cost
        self._session_cost += cost

    def get_session_cost(self) -> float:
        return self._session_cost

    def get_last_call_cost(self) -> float:
        """Cost of the most recent LLM call."""
        return self._last_call_cost

    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Detailed cost breakdown for the current session."""
        rates = COST_RATES.get(self._provider, COST_RATES["claude"])
        return {
            "session_total": self._session_cost,
            "last_call": self._last_call_cost,
            "provider": self._provider,
            "model": self._model or "default",
            "rates": rates,
        }

    def get_provider_info(self) -> Dict[str, str]:
        return {
            "provider": self._provider,
            "model": self._model or "default",
        }

    async def run_self_test(self) -> Dict[str, Any]:
        """Run pytest suite as boot-time self-test. Returns pass/fail/summary."""
        import time as _time

        start = _time.monotonic()
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    ".venv/bin/pytest", "tests/", "-x", "-q", "--tb=no", "-q",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                ),
                timeout=10,  # subprocess creation timeout
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return {
                    "passed": False,
                    "summary": "timeout",
                    "duration_seconds": round(_time.monotonic() - start, 1),
                }
            output = stdout.decode(errors="replace") if stdout else ""
            passed = proc.returncode == 0
            # Extract last non-empty line as summary
            lines = [l.strip() for l in output.strip().splitlines() if l.strip()]
            summary = lines[-1] if lines else ("passed" if passed else "failed")
            return {
                "passed": passed,
                "summary": summary,
                "duration_seconds": round(_time.monotonic() - start, 1),
            }
        except Exception as e:
            return {
                "passed": False,
                "summary": f"error: {e}",
                "duration_seconds": round(_time.monotonic() - start, 1),
            }

    def get_rejection_hints(self) -> List[str]:
        """Get remediation hints from governor rejection log. Feeds anti-fragile loop."""
        try:
            from core.rejection_log import RejectionLog
            log = RejectionLog()
            summary = log.get_summary()
            if summary.total_rejections > 0 and summary.remediation_hints:
                return list(summary.remediation_hints)
        except Exception:
            pass
        return []

    def get_workspace_info(self) -> dict:
        """Analyze workspace environment. Returns disk, project stats, warnings."""
        import shutil
        import os
        warnings = []
        info = {"disk_free_gb": 0.0, "project_files": 0, "warnings": warnings}
        try:
            usage = shutil.disk_usage(os.path.expanduser("~"))
            free_gb = usage.free / (1024 ** 3)
            info["disk_free_gb"] = round(free_gb, 1)
            if free_gb < 1.0:
                warnings.append(f"Low disk space: {free_gb:.1f} GB free")
        except Exception:
            pass
        try:
            home = os.path.expanduser("~/.motherlabs")
            if os.path.isdir(home):
                total = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, fns in os.walk(home)
                    for f in fns
                )
                mb = total / (1024 ** 2)
                info["data_dir_mb"] = round(mb, 1)
                if mb > 500:
                    warnings.append(f"Data directory large: {mb:.0f} MB")
        except Exception:
            pass
        return info

    def get_learning_context(self, db_path) -> dict:
        """Get learning context from journal patterns. Returns trends, failures, chronic weaknesses."""
        try:
            from mother.journal import BuildJournal
            from mother.journal_patterns import extract_patterns
            from dataclasses import asdict
            journal = BuildJournal(db_path)
            entries = journal.recent(limit=20)
            if not entries:
                return {"trends_line": "", "failure_line": "", "chronic_weak": []}
            entry_dicts = [asdict(e) for e in entries]
            patterns = extract_patterns(entry_dicts)
            return {
                "trends_line": patterns.trends_line,
                "failure_line": patterns.failure_line,
                "chronic_weak": list(patterns.chronic_weak),
                "dimension_averages": dict(patterns.dimension_averages),
            }
        except Exception:
            return {"trends_line": "", "failure_line": "", "chronic_weak": []}

    def begin_compile(self) -> None:
        """Signal that a compile is about to start.

        Call this BEFORE asyncio.create_task(bridge.compile(...)) to avoid
        a race where stream_insights() sees _compile_done=True and exits
        before compile() has a chance to clear it.
        """
        self._compile_done.clear()

    async def stream_insights(self) -> AsyncGenerator[str, None]:
        """Yield insights as they arrive during compilation.

        Polls thread-safe queue.Queue (not asyncio.Queue) since insights
        are pushed from engine thread via asyncio.to_thread().
        Terminates when _compile_done is set and queue is empty.

        Usage:
            bridge.begin_compile()
            task = asyncio.create_task(bridge.compile(desc))
            async for insight in bridge.stream_insights():
                ...
            result = await task
        """
        while True:
            if self._compile_done.is_set() and self._insight_queue.empty():
                return
            try:
                insight = self._insight_queue.get_nowait()
                yield insight
            except queue.Empty:
                if self._compile_done.is_set():
                    return
                await asyncio.sleep(0.2)

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List tools from the tool registry."""
        def _call():
            from motherlabs_platform.tool_registry import get_tool_registry
            registry = get_tool_registry()
            return registry.list_tools()

        return await asyncio.to_thread(_call)

    def get_status(self) -> Dict[str, Any]:
        """Aggregated status: cost, provider, model, rates."""
        rates = COST_RATES.get(self._provider, COST_RATES["claude"])
        return {
            "provider": self._provider,
            "model": self._model or "default",
            "session_cost": self._session_cost,
            "rates": {
                "input": f"${rates['input']:.2f}/MTok",
                "output": f"${rates['output']:.2f}/MTok",
            },
        }

    # --- Filesystem operations ---

    def _get_filesystem(self):
        """Lazy-load FileSystemBridge."""
        if self._filesystem is None:
            from mother.filesystem import FileSystemBridge
            self._filesystem = FileSystemBridge(file_access=self._file_access)
        return self._filesystem

    async def search_files(self, query: str, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for files using Spotlight or glob fallback."""
        fs = self._get_filesystem()
        return await asyncio.to_thread(fs.search, query, path)

    async def read_file(self, path: str) -> Dict[str, Any]:
        """Read file contents."""
        fs = self._get_filesystem()
        return await asyncio.to_thread(fs.read_file, path)

    async def write_file(self, path: str, content: str, overwrite: bool = False) -> Dict[str, Any]:
        """Write text to file."""
        fs = self._get_filesystem()
        return await asyncio.to_thread(fs.write_file, path, content, overwrite)

    async def edit_file(self, path: str, old_text: str, new_text: str) -> Dict[str, Any]:
        """Replace text in a file."""
        fs = self._get_filesystem()
        return await asyncio.to_thread(fs.edit_file, path, old_text, new_text)

    async def append_file(self, path: str, content: str) -> Dict[str, Any]:
        """Append text to a file."""
        fs = self._get_filesystem()
        return await asyncio.to_thread(fs.append_file, path, content)

    async def move_file(self, src: str, dst: str) -> Dict[str, Any]:
        """Move or rename a file."""
        fs = self._get_filesystem()
        return await asyncio.to_thread(fs.move_file, src, dst)

    async def copy_file(self, src: str, dst: str) -> Dict[str, Any]:
        """Copy a file."""
        fs = self._get_filesystem()
        return await asyncio.to_thread(fs.copy_file, src, dst)

    async def delete_file(self, path: str) -> Dict[str, Any]:
        """Delete a file (Trash on macOS)."""
        fs = self._get_filesystem()
        return await asyncio.to_thread(fs.delete_file, path)

    async def list_dir(self, path: str, pattern: str = "*") -> List[Dict[str, Any]]:
        """List directory contents."""
        fs = self._get_filesystem()
        return await asyncio.to_thread(fs.list_dir, path, pattern)

    async def file_info(self, path: str) -> Dict[str, Any]:
        """Get file metadata."""
        fs = self._get_filesystem()
        return await asyncio.to_thread(fs.file_info, path)

    # --- Screen capture ---

    def _get_screen_bridge(self):
        """Lazy-load ScreenCaptureBridge."""
        if self._screen_bridge is None:
            from mother.screen import ScreenCaptureBridge
            self._screen_bridge = ScreenCaptureBridge(
                enabled=self._screen_capture_enabled,
            )
        return self._screen_bridge

    async def capture_screen(
        self, display: int = 1, region=None
    ) -> Optional[str]:
        """Capture screen and return base64 PNG."""
        bridge = self._get_screen_bridge()
        return await bridge.capture_screen(display=display, region=region)

    # --- Microphone ---

    def _get_microphone_bridge(self):
        """Lazy-load MicrophoneBridge."""
        if self._microphone_bridge is None:
            from mother.microphone import MicrophoneBridge
            self._microphone_bridge = MicrophoneBridge(
                openai_api_key=self._openai_api_key or "",
                enabled=self._microphone_enabled,
            )
        return self._microphone_bridge

    async def record_and_transcribe(
        self, duration_seconds: float = 5.0
    ) -> Optional[str]:
        """Record audio and transcribe via Whisper."""
        bridge = self._get_microphone_bridge()
        return await bridge.record_and_transcribe(duration_seconds)

    # --- Camera ---

    def _get_camera_bridge(self):
        """Lazy-load CameraBridge."""
        if not hasattr(self, "_camera_bridge") or self._camera_bridge is None:
            from mother.camera import CameraBridge
            self._camera_bridge = CameraBridge(
                enabled=getattr(self, "_camera_enabled", False),
            )
        return self._camera_bridge

    async def capture_camera(self, device_index: int = 0) -> Optional[str]:
        """Capture webcam frame and return base64 JPEG."""
        bridge = self._get_camera_bridge()
        return await bridge.capture_frame(device_index=device_index)

    @property
    def is_voice_playing(self) -> bool:
        """True if voice bridge is currently playing audio."""
        if hasattr(self, "_voice_bridge_ref") and self._voice_bridge_ref:
            return getattr(self._voice_bridge_ref, "is_playing", False)
        return False

    def set_voice_bridge_ref(self, voice_bridge) -> None:
        """Set reference to voice bridge for playback detection."""
        self._voice_bridge_ref = voice_bridge

    def interpret_trust(self, verification: Dict[str, Any]) -> str:
        """Map verification scores to natural-language interpretation."""
        overall = verification.get("overall_score", 0.0)

        # Find weakest dimension — handle both flat (score: 72) and
        # nested ({"score": 72, "gaps": [...]}) verification formats
        dimensions: Dict[str, float] = {}
        for k, v in verification.items():
            if k == "overall_score":
                continue
            if isinstance(v, (int, float)):
                dimensions[k] = float(v)
            elif isinstance(v, dict) and "score" in v:
                dimensions[k] = float(v["score"])

        weakest = None
        weakest_score = 101.0
        for dim, score in dimensions.items():
            if score < weakest_score:
                weakest = dim
                weakest_score = score

        if overall >= 80:
            msg = "High confidence."
            if weakest and weakest_score < 70:
                msg += f" Weakest area: {weakest} ({weakest_score:.0f}%)."
            return msg
        elif overall >= 40:
            msg = "Moderate confidence — review the gaps before building."
            if weakest:
                msg += f" Weakest: {weakest} ({weakest_score:.0f}%)."
            return msg
        else:
            msg = "Low confidence. I'd recommend refining the input."
            if weakest:
                msg += f" {weakest} scored {weakest_score:.0f}%."
            return msg

    # --- Project launcher ---

    async def launch(self, project_dir: str, entry_point: str = "main.py") -> dict:
        """Launch a compiled project as subprocess."""
        from mother.launcher import ProjectLauncher
        if hasattr(self, "_launcher") and self._launcher and self._launcher.is_running():
            await asyncio.to_thread(self._launcher.stop)
        self._launcher = ProjectLauncher(project_dir, entry_point)
        result = await asyncio.to_thread(self._launcher.start)
        # Wire health probe if launch succeeded
        if result.success and result.pid:
            from mother.health import HealthProbe
            self._health_probe = HealthProbe(
                pid=result.pid,
                port=result.port or 0,
            )
        return {
            "success": result.success,
            "pid": result.pid,
            "port": result.port,
            "error": result.error,
            "entry_point": result.entry_point,
            "project_dir": result.project_dir,
        }

    async def stop_project(self) -> None:
        """Stop the running project subprocess."""
        launcher = getattr(self, "_launcher", None)
        if launcher:
            await asyncio.to_thread(launcher.stop)
            self._launcher = None
        self._health_probe = None

    def get_project_output(self) -> list:
        """Non-blocking: return queued output lines from running project."""
        launcher = getattr(self, "_launcher", None)
        if launcher:
            return launcher.drain_output()
        return []

    def is_project_running(self) -> bool:
        """True if a launched project is currently running."""
        launcher = getattr(self, "_launcher", None)
        return launcher is not None and launcher.is_running()

    # --- Rejection log ---

    async def get_rejection_summary(self) -> Dict[str, Any]:
        """Get governor rejection summary. Returns empty dict on failure."""
        def _call():
            try:
                from core.rejection_log import RejectionLog
                log = RejectionLog()
                summary = log.get_summary()
                return {
                    "total": summary.total_rejections,
                    "by_check": dict(summary.by_check),
                    "hints": list(summary.remediation_hints),
                }
            except Exception:
                return {}

        return await asyncio.to_thread(_call)

    # --- Context data queries (for context synthesis) ---

    async def get_corpus_summary(self) -> Dict[str, Any]:
        """Get compilation corpus stats. Returns empty dict on failure."""
        def _call():
            try:
                from persistence.sqlite_corpus import SQLiteCorpus
                corpus = SQLiteCorpus()
                stats = corpus.get_stats()
                # Also compute avg trust from recent compilations
                records = corpus.list_all(per_page=50)
                trust_scores = [
                    r.verification_score for r in records
                    if r.verification_score is not None and r.verification_score > 0
                ]
                avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0.0
                stats["avg_trust"] = avg_trust

                # Corpus depth: anti-patterns, constraints, pattern health
                try:
                    from persistence.corpus_analysis import CorpusAnalyzer
                    analyzer = CorpusAnalyzer(corpus)
                    anti_patterns = analyzer.detect_anti_patterns()
                    stats["anti_pattern_count"] = len(anti_patterns)
                    # Constraint templates require a domain — aggregate across known domains
                    constraint_count = 0
                    for domain in stats.get("domains", {}):
                        ct = analyzer.extract_constraint_templates(domain)
                        constraint_count += len(ct)
                    stats["constraint_count"] = constraint_count
                    # L2 feedback: pattern health for dominant domain
                    domains = stats.get("domains", {})
                    if domains:
                        top_domain = max(domains, key=domains.get)
                        stats["pattern_health"] = analyzer.summarize_pattern_health(top_domain)
                    else:
                        stats["pattern_health"] = ""
                except Exception:
                    stats["anti_pattern_count"] = 0
                    stats["constraint_count"] = 0
                    stats["pattern_health"] = ""

                return stats
            except Exception:
                return {}

        return await asyncio.to_thread(_call)

    async def get_tool_summary(self) -> Dict[str, Any]:
        """Get tool portfolio summary. Returns empty dict on failure."""
        def _call():
            try:
                from motherlabs_platform.tool_registry import get_tool_registry
                registry = get_tool_registry()
                tools = registry.list_tools()
                if not tools:
                    return {"count": 0, "verified": 0, "domains": {}}
                verified = sum(1 for t in tools if t.verification_badge == "verified")
                domains: Dict[str, int] = {}
                for t in tools:
                    domains[t.domain] = domains.get(t.domain, 0) + 1
                return {
                    "count": len(tools),
                    "verified": verified,
                    "domains": domains,
                }
            except Exception:
                return {}

        return await asyncio.to_thread(_call)

    # --- Operational awareness: journal, health ---

    def _get_journal(self, db_path):
        """Lazy-load BuildJournal."""
        if not hasattr(self, "_journal") or self._journal is None:
            from mother.journal import BuildJournal
            self._journal = BuildJournal(db_path)
        return self._journal

    async def record_journal_entry(self, db_path, **kwargs) -> None:
        """Record a journal entry. kwargs match JournalEntry fields."""
        journal = self._get_journal(db_path)

        def _call():
            from mother.journal import JournalEntry
            entry = JournalEntry(**kwargs)
            journal.record(entry)

        await asyncio.to_thread(_call)

    async def get_journal_summary(self, db_path) -> dict:
        """Get journal summary as dict. Returns empty dict on failure."""
        journal = self._get_journal(db_path)

        def _call():
            try:
                s = journal.get_summary()
                return {
                    "total_compiles": s.total_compiles,
                    "total_builds": s.total_builds,
                    "success_rate": s.success_rate,
                    "avg_trust": s.avg_trust,
                    "total_cost": s.total_cost,
                    "domains": dict(s.domains),
                    "streak": s.streak,
                }
            except Exception:
                return {}

        return await asyncio.to_thread(_call)

    async def check_health(self) -> tuple:
        """Check health of launched project.

        Returns (status_dict, event_dict_or_None).
        Returns ({}, None) if no health probe active.
        """
        probe = getattr(self, "_health_probe", None)
        if probe is None:
            return {}, None

        def _call():
            from dataclasses import asdict
            status, event = probe.check()
            return (
                asdict(status),
                asdict(event) if event is not None else None,
            )

        return await asyncio.to_thread(_call)

    # --- Neurologis Automatica: temporal, attention, recall ---

    def _get_temporal_engine(self):
        """Lazy-load TemporalEngine."""
        if not hasattr(self, "_temporal_engine") or self._temporal_engine is None:
            from mother.temporal import TemporalEngine
            self._temporal_engine = TemporalEngine()
        return self._temporal_engine

    def _get_attention_filter(self):
        """Lazy-load AttentionFilter."""
        if not hasattr(self, "_attention_filter") or self._attention_filter is None:
            from mother.attention import AttentionFilter
            self._attention_filter = AttentionFilter()
        return self._attention_filter

    def _get_recall_engine(self, db_path):
        """Lazy-load RecallEngine."""
        if not hasattr(self, "_recall_engine") or self._recall_engine is None:
            from mother.recall import RecallEngine
            self._recall_engine = RecallEngine(db_path)
        return self._recall_engine

    async def recall_for_context(self, query: str, db_path, max_tokens: int = 1000) -> str:
        """Search conversation history for relevant context."""
        engine = self._get_recall_engine(db_path)

        def _call():
            return engine.recall_for_context(query, max_tokens=max_tokens)

        return await asyncio.to_thread(_call)

    async def index_message_for_recall(self, db_path, message_id: int, content: str) -> None:
        """Index a message for recall search."""
        engine = self._get_recall_engine(db_path)

        def _call():
            engine.index_message(message_id, content)

        await asyncio.to_thread(_call)

    # --- Memory bank: unified retrieval across all memory systems ---

    def _get_memory_indexer(self, db_path):
        """Lazy-load MemoryIndexer."""
        if not hasattr(self, "_memory_indexer") or self._memory_indexer is None:
            from mother.memory_indexer import MemoryIndexer
            self._memory_indexer = MemoryIndexer(db_path=db_path)
        return self._memory_indexer

    def memory_query(self, query: str, db_path, max_tokens: int = 500) -> str:
        """Unified memory query across all stores.

        Returns a formatted [MEMORY] block for context injection.
        Replaces the old [Recalled] block with richer cross-store retrieval.
        """
        from mother.memory_bank import query as mb_query, format_memory_context
        recall = None
        try:
            recall = self._get_recall_engine(db_path)
        except Exception:
            pass
        results = mb_query(query, recall_engine=recall, db_path=db_path)
        return format_memory_context(results, max_tokens=max_tokens)

    def index_message_for_memory(self, content: str, role: str, session_id: str, db_path=None):
        """Index a message for both recall FTS and knowledge extraction.

        Call after every message save. Extracts facts in the background.
        """
        indexer = self._get_memory_indexer(db_path)
        indexer.index_message(content, role, session_id=session_id)

    def close_session_memory(self, session_id: str, db_path=None):
        """Close a session: compress buffered messages into an episode.

        Call when the session ends (app shutdown, timeout, etc.).
        """
        indexer = self._get_memory_indexer(db_path)
        return indexer.close_session(session_id)

    @staticmethod
    def memory_stats(db_path=None):
        """Return memory bank statistics."""
        from mother.memory_bank import memory_stats
        from pathlib import Path
        return memory_stats(db_path=db_path or Path.home() / ".motherlabs" / "history.db")

    # --- Self-extension: auto-register, project memory, tool invocation ---

    async def register_build_as_tool(self, result, description: str) -> Optional[Dict[str, Any]]:
        """Package a successful build result and register it in ToolRegistry.

        Args:
            result: AgentResult from orchestrator.run()
            description: Original user description

        Returns:
            {package_id, name, trust_score, badge} or None on failure/duplicate
        """
        def _call():
            import hashlib

            # Validate required data
            blueprint = getattr(result, "blueprint", None)
            generated_code = getattr(result, "generated_code", None)
            trust = getattr(result, "trust", None)

            if not blueprint or not generated_code:
                return None

            # Compute hashes
            input_hash = hashlib.sha256(description.encode()).hexdigest()[:16]
            code_keys = ",".join(sorted(generated_code.keys()))
            compilation_id = hashlib.sha256(
                (description + code_keys).encode()
            ).hexdigest()[:16]

            # Structural fingerprint
            from core.determinism import compute_structural_fingerprint
            fp = compute_structural_fingerprint(blueprint)

            # Instance identity
            from motherlabs_platform.instance_identity import InstanceIdentityStore
            store = InstanceIdentityStore()
            record = store.get_or_create_self()
            instance_id = record.instance_id if record else "local"

            # Trust data
            trust_score = trust.overall_score if trust else 0.0
            badge = trust.verification_badge if trust else "unverified"
            fidelity = {}
            if trust:
                for attr in ("structural", "semantic", "behavioral", "interface",
                             "dependency", "consistency", "completeness"):
                    val = getattr(trust, attr, None)
                    if val is not None:
                        fidelity[attr] = int(val) if isinstance(val, (int, float)) else 0

            # Package
            from core.tool_package import package_tool
            # Derive tool name: blueprint core_need > user description > fallback
            tool_name = blueprint.get("core_need", "") or description[:80] if description else ""
            pkg = package_tool(
                compilation_id=compilation_id,
                blueprint=blueprint,
                generated_code=generated_code,
                trust_score=trust_score,
                verification_badge=badge,
                fidelity_scores=fidelity,
                fingerprint_hash=fp.hash_digest,
                instance_id=instance_id,
                domain=blueprint.get("domain", "software"),
                provider=self._provider,
                input_hash=input_hash,
                name=tool_name or None,
            )

            # Register (skip duplicates)
            from motherlabs_platform.tool_registry import get_tool_registry
            registry = get_tool_registry()

            # Check fingerprint duplicate
            existing = registry.find_by_fingerprint(fp.hash_digest)
            if existing:
                return None

            try:
                registry.register_tool(pkg, is_local=True)
            except Exception:
                return None  # IntegrityError = duplicate package_id

            return {
                "package_id": pkg.package_id,
                "name": pkg.name,
                "trust_score": pkg.trust_score,
                "badge": pkg.verification_badge,
            }

        try:
            return await asyncio.to_thread(_call)
        except Exception as e:
            logger.debug(f"register_build_as_tool error: {e}")
            return None

    async def get_recent_projects(self, db_path) -> List[Dict[str, Any]]:
        """Get recent successful builds with project paths from journal.

        Returns deduplicated list of [{name, path, description, trust, timestamp}].
        """
        def _call():
            from mother.journal import BuildJournal
            journal = BuildJournal(db_path)
            entries = journal.recent(limit=20)
            seen_paths: set = set()
            projects: list = []
            for entry in entries:
                if not entry.success or not entry.project_path:
                    continue
                if entry.project_path in seen_paths:
                    continue
                seen_paths.add(entry.project_path)
                import os
                name = os.path.basename(entry.project_path)
                projects.append({
                    "name": name,
                    "path": entry.project_path,
                    "description": entry.description,
                    "trust": entry.trust_score,
                    "timestamp": entry.timestamp,
                })
            return projects[:10]

        try:
            return await asyncio.to_thread(_call)
        except Exception as e:
            logger.debug(f"get_recent_projects error: {e}")
            return []

    async def get_tool_details(self) -> List[Dict[str, str]]:
        """Get tool names and domains for system prompt awareness.

        Returns [{name, domain, package_id, trust}] capped at 20.
        """
        def _call():
            from motherlabs_platform.tool_registry import get_tool_registry
            registry = get_tool_registry()
            tools = registry.list_tools()
            return [
                {
                    "name": t.name,
                    "domain": t.domain,
                    "package_id": t.package_id,
                    "trust": str(t.trust_score),
                }
                for t in tools[:20]
            ]

        try:
            return await asyncio.to_thread(_call)
        except Exception as e:
            logger.debug(f"get_tool_details error: {e}")
            return []

    async def run_tool(self, tool_name: str, input_text: str = "") -> Dict[str, Any]:
        """Find and run a registered tool by name.

        Searches registry, finds project directory, runs as subprocess,
        records usage.

        Returns {success, output, error, tool_name}.
        """
        def _call():
            from motherlabs_platform.tool_registry import get_tool_registry
            from mother.tool_runner import find_tool_project, run_tool as _run_tool

            registry = get_tool_registry()
            matches = registry.search_tools(tool_name)
            if not matches:
                return {
                    "success": False,
                    "output": "",
                    "error": f"No tool found matching '{tool_name}'",
                    "tool_name": tool_name,
                }

            matched_tool = matches[0]

            # Find project directory
            project_dir = find_tool_project(tool_name)
            if not project_dir:
                # Try matching by the registry name
                project_dir = find_tool_project(matched_tool.name)
            if not project_dir:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Project directory not found for '{matched_tool.name}'",
                    "tool_name": matched_tool.name,
                }

            # Run it
            result = _run_tool(project_dir, input_text=input_text)

            # Record usage + track health (#77 tool-chain-managing)
            try:
                action = "success" if result.success else "failure"
                registry.record_usage(matched_tool.package_id, action)
                registry.increment_usage_count(matched_tool.package_id)
            except Exception:
                pass

            # Track tool health in bridge-level registry
            tool_key = result.tool_name or tool_name
            if tool_key not in self._tool_health:
                self._tool_health[tool_key] = {"runs": 0, "failures": 0}
            self._tool_health[tool_key]["runs"] += 1
            if not result.success:
                self._tool_health[tool_key]["failures"] += 1

            return {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "tool_name": result.tool_name,
            }

        try:
            return await asyncio.to_thread(_call)
        except Exception as e:
            logger.debug(f"run_tool error: {e}")
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "tool_name": tool_name,
            }

    def get_tool_health(self) -> Dict[str, Dict[str, Any]]:
        """Return tool health stats: {tool_name: {runs, failures, health_pct}}."""
        result = {}
        for name, stats in self._tool_health.items():
            runs = stats["runs"]
            failures = stats["failures"]
            health_pct = ((runs - failures) / runs * 100) if runs > 0 else 100.0
            result[name] = {"runs": runs, "failures": failures, "health_pct": health_pct}
        return result

    async def get_instance_age_days(self) -> int:
        """Get instance age in days. Returns 0 on failure."""
        def _call():
            try:
                from motherlabs_platform.instance_identity import InstanceIdentityStore
                store = InstanceIdentityStore()
                record = store.get_or_create_self()
                if record and record.created_at:
                    from datetime import datetime
                    created = datetime.fromisoformat(record.created_at)
                    now = datetime.now()
                    return max(0, (now - created).days)
            except Exception:
                pass
            return 0

        return await asyncio.to_thread(_call)

    # --- Claude Code CLI / self-build ---

    async def invoke_claude_code(
        self,
        prompt: str,
        cwd: str,
        max_turns: int = 15,
        max_budget_usd: float = 3.0,
        claude_path: str = "",
    ) -> dict:
        """Invoke coding agent with failover. Returns dict with result fields."""
        def _call():
            from mother.coding_agent import invoke_coding_agent, CodingAgentProvider, default_providers
            providers = default_providers()
            if claude_path:
                providers = [
                    CodingAgentProvider(
                        name=p.name, binary=claude_path if p.name == "claude" else p.binary,
                        env_var=p.env_var, priority=p.priority,
                        max_turns=p.max_turns, max_budget_usd=p.max_budget_usd,
                        timeout=p.timeout,
                    )
                    for p in providers
                ]
            result = invoke_coding_agent(
                prompt=prompt,
                cwd=cwd,
                providers=providers,
                preferred="claude",
                max_turns=max_turns,
                max_budget_usd=max_budget_usd,
            )
            return {
                "success": result.success,
                "result_text": result.result_text,
                "session_id": result.session_id,
                "cost_usd": result.cost_usd,
                "duration_seconds": result.duration_seconds,
                "num_turns": result.num_turns,
                "error": result.error,
                "is_error": result.is_error,
                "provider": result.provider,
            }

        return await asyncio.to_thread(_call)

    def _try_native_code_engine(
        self,
        prompt: str,
        repo_dir: str,
        max_turns: int = 30,
        max_budget_usd: float = 3.0,
        on_event: Optional[Callable] = None,
        sandbox_profile=None,
        system_prompt: Optional[str] = None,
    ) -> Optional[dict]:
        """Try to run the native code engine. Returns result dict or None if unavailable.

        Attempts to create an LLM client and adapter from available API keys.
        If no provider is available or the adapter fails, returns None
        (caller should fall back to CLI coding agents).
        """
        try:
            from mother.code_engine import run_code_engine, CodeEngineConfig, to_coding_agent_result
            from mother.code_engine_adapters import select_adapter
            from core.llm import (
                ClaudeClient, OpenAIClient, GrokClient, GeminiClient,
            )
            import os as _os

            # Try providers in order: Claude, OpenAI, Grok, Gemini
            client = None
            # Map config key names to env var names for fallback lookup
            _config_key_map = {
                "claude": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "grok": "XAI_API_KEY",
                "gemini": "GEMINI_API_KEY",
            }
            all_provider_configs = [
                ("claude", "ANTHROPIC_API_KEY", ClaudeClient, {"model": "claude-sonnet-4-20250514"}),
                ("openai", "OPENAI_API_KEY", OpenAIClient, {"model": "gpt-5.1"}),
                ("grok", "XAI_API_KEY", GrokClient, {"model": "grok-4-1-fast-reasoning"}),
                ("gemini", "GEMINI_API_KEY", GeminiClient, {"model": "gemini-2.5-pro"}),
            ]
            # Put configured provider first so it's tried before burning credits elsewhere
            provider_configs = sorted(
                all_provider_configs,
                key=lambda x: (0 if x[0] == self._provider else 1),
            )
            # Build list of (client, adapter) candidates — all providers with keys
            candidates = []
            for name, env_var, cls, kwargs in provider_configs:
                key = _os.environ.get(env_var)
                if not key and name == "gemini":
                    key = _os.environ.get("GOOGLE_API_KEY")
                # Fallback: check instance API keys from config
                if not key and name == self._provider and self._api_key:
                    key = self._api_key
                if not key and name == "openai" and self._openai_api_key:
                    key = self._openai_api_key
                if key:
                    try:
                        c = cls(api_key=key, **kwargs)
                        candidates.append((name, c))
                    except Exception as e:
                        logger.debug(f"Native engine: {name} client creation failed: {e}")
                        continue

            if not candidates:
                logger.debug("Native code engine: no LLM provider available")
                return None

            # Build config (shared across attempts)
            config = CodeEngineConfig(
                working_dir=repo_dir,
                allowed_paths=[repo_dir],
                max_turns=max_turns,
                cost_cap_usd=max_budget_usd,
                on_event=on_event,
                sandbox_profile=sandbox_profile,
            )

            # Wire in code safety checker — self_build=True exempts subprocess/os
            try:
                from core.governor_validation import check_code_safety
                from functools import partial
                config.safety_checker = partial(check_code_safety, self_build=True)
            except ImportError:
                pass

            # Try each provider — failover on API/auth/credit errors
            for prov_name, client in candidates:
                adapter = select_adapter(client)
                try:
                    result = run_code_engine(prompt=prompt, adapter=adapter, config=config, system_prompt=system_prompt)
                except Exception as e:
                    logger.debug(f"Native engine: {prov_name} run failed: {e}")
                    continue

                # If the engine returned a provider-level error (auth, credits, rate limit)
                # on the very first turn, try the next provider instead
                if (not result.success
                        and result.turns_used <= 1
                        and result.error
                        and any(s in result.error.lower() for s in
                                ("credit", "balance", "billing", "unauthorized",
                                 "401", "403", "429", "rate limit", "quota"))):
                    logger.info(f"Native engine: {prov_name} provider error, trying next: {result.error[:80]}")
                    continue

                return {
                    "success": result.success,
                    "result_text": result.final_text,
                    "cost_usd": result.total_cost_usd,
                    "rolled_back": False,
                    "error": result.error,
                    "provider": result.provider_name,
                    "files_modified": list(result.files_modified),
                    "turns_used": result.turns_used,
                }

            # All providers exhausted with provider-level errors — signal unavailable
            logger.debug("Native code engine: all providers failed with API errors")
            return None
        except Exception as e:
            logger.debug(f"Native code engine failed: {e}")
            return None

    async def self_build(
        self,
        prompt: str,
        repo_dir: str,
        max_turns: int = 15,
        max_budget_usd: float = 3.0,
        claude_path: str = "",
    ) -> dict:
        """Orchestrate a self-build: snapshot → invoke (streaming) → test → commit or rollback.

        Tries native code engine first, falls back to CLI coding agents.
        Pushes streaming events to _self_build_event_queue for live display.
        Returns dict with keys: success, result_text, cost_usd, rolled_back, error.
        """
        event_q = self._self_build_event_queue
        # Use a local list for event accumulation — immune to begin_self_build() clearing
        log_events: list = []

        def _on_event(event: dict) -> None:
            """Push each stream event to the queue + accumulate for logging.

            Runs in the worker thread. Queue is thread-safe.
            """
            log_events.append(event)
            try:
                event_q.put_nowait(event)
            except queue.Full:
                logger.warning("Self-build event queue full, dropping event")

        def _call():
            from mother.coding_agent import (
                invoke_coding_agent_streaming,
                CodingAgentProvider,
                default_providers,
            )
            from mother.claude_code import (
                save_build_log,
                git_snapshot,
                git_rollback,
                run_tests,
            )
            import subprocess as _subprocess

            # Signal phases to streaming consumer
            def _push_phase(phase: str, detail: str = "") -> None:
                try:
                    event_q.put_nowait({"_type": "phase", "phase": phase, "detail": detail})
                except queue.Full:
                    pass

            _build_start_mono = time.monotonic()

            # --- Worktree + sandbox setup ---
            _worktree = None
            _sandbox_prof = None
            _build_dir = repo_dir  # default: direct modification (fallback)

            try:
                from mother.worktree import (
                    worktree_create,
                    worktree_merge,
                    worktree_remove,
                    worktree_has_protected_changes,
                    worktree_diff_summary,
                )
                _use_worktree = True
            except ImportError:
                _use_worktree = False
                logger.warning("mother.worktree not available — running without isolation")

            try:
                # 1. Snapshot on main repo (safety net)
                _push_phase("snapshot", "Creating git snapshot...")
                snapshot_hash = git_snapshot(repo_dir)

                # 2. Create worktree for isolated build
                if _use_worktree:
                    try:
                        _push_phase("worktree", "Creating isolated build environment...")
                        _worktree = worktree_create(repo_dir)
                        _build_dir = _worktree.path
                        logger.info(f"Build isolation: worktree at {_build_dir}")

                        # Create sandbox profile
                        try:
                            from mother.sandbox import create_build_profile, is_sandbox_available
                            if is_sandbox_available():
                                _venv_dir = str(__import__("pathlib").Path(repo_dir) / ".venv")
                                _sandbox_prof = create_build_profile(_build_dir, _venv_dir)
                                logger.info("Build isolation: sandbox-exec enabled")
                            else:
                                logger.warning("sandbox-exec not available — worktree only (no sandbox)")
                        except ImportError:
                            logger.warning("mother.sandbox not available — worktree only (no sandbox)")
                    except Exception as e:
                        logger.warning(f"Worktree creation failed, falling back to direct: {e}")
                        _worktree = None
                        _build_dir = repo_dir

                # 3a. Try native code engine first
                _push_phase("invoke", "Running native code engine...")
                native_result = self._try_native_code_engine(
                    prompt=prompt,
                    repo_dir=_build_dir,
                    max_turns=max_turns * 2,  # native engine uses more granular turns
                    max_budget_usd=max_budget_usd,
                    on_event=_on_event,
                    sandbox_profile=_sandbox_prof,
                )
                if native_result is not None:
                    # Native engine ran (success or failure)
                    from mother.claude_code import save_build_log as _save_log
                    from types import SimpleNamespace as _NS
                    _save_log(log_events, prompt, repo_dir, _NS(
                        success=native_result["success"],
                        cost_usd=native_result["cost_usd"],
                        duration_seconds=0,
                        error=native_result["error"],
                        num_turns=native_result.get("turns_used", 0),
                    ), prompt[:40].split("\n")[0])
                    self._self_build_events = list(log_events)

                    if not native_result["success"]:
                        # Worktree is disposed in finally block — no rollback needed
                        if not _worktree and snapshot_hash:
                            git_rollback(repo_dir, snapshot_hash)
                        # Close feedback loop on failure
                        _fail_pcs = self._extract_postcodes_from_text(prompt)
                        try:
                            self._close_feedback_loop(
                                target_postcodes=_fail_pcs,
                                success=False,
                                failure_reason=native_result.get("error", "native build failed"),
                            )
                        except Exception:
                            pass
                        return {
                            "success": False,
                            "result_text": native_result["result_text"],
                            "cost_usd": native_result["cost_usd"],
                            "rolled_back": not _worktree and bool(snapshot_hash),
                            "error": native_result["error"],
                            "provider": native_result.get("provider", "native"),
                        }

                    # Native succeeded — run tests in worktree (sandboxed)
                    _push_phase("testing", "Running tests...")
                    tests_pass = run_tests(_build_dir, sandbox_profile=_sandbox_prof)
                    if not tests_pass:
                        # Worktree is disposed in finally block — no rollback needed
                        if not _worktree and snapshot_hash:
                            git_rollback(repo_dir, snapshot_hash)
                        # Close feedback loop on test failure
                        _test_fail_pcs = self._extract_postcodes_from_text(prompt)
                        try:
                            self._close_feedback_loop(
                                target_postcodes=_test_fail_pcs,
                                success=False,
                                failure_reason="Tests failed after modification",
                            )
                        except Exception:
                            pass
                        return {
                            "success": False,
                            "result_text": native_result["result_text"],
                            "cost_usd": native_result["cost_usd"],
                            "rolled_back": not _worktree and bool(snapshot_hash),
                            "error": "Tests failed after modification",
                            "provider": native_result.get("provider", "native"),
                        }

                    # Check for protected file changes in worktree
                    if _worktree:
                        try:
                            _wt_protected = worktree_has_protected_changes(_worktree)
                            if _wt_protected:
                                logger.warning(f"Protected files modified in worktree: {_wt_protected}")
                                return {
                                    "success": False,
                                    "result_text": native_result["result_text"],
                                    "cost_usd": native_result["cost_usd"],
                                    "rolled_back": False,
                                    "error": f"Protected files modified: {', '.join(_wt_protected)}",
                                    "provider": native_result.get("provider", "native"),
                                }
                        except Exception as e:
                            logger.debug(f"Protected file check skipped: {e}")

                    # Merge worktree back to main (or direct commit if no worktree)
                    if _worktree:
                        _push_phase("merge", "Merging changes to main...")
                        # Commit changes in worktree first
                        _wt_env = __import__("os").environ.copy()
                        _wt_env.pop("CLAUDECODE", None)
                        _wt_env.pop("ANTHROPIC_API_KEY", None)
                        try:
                            _subprocess.run(
                                ["git", "add", "-A"],
                                cwd=_worktree.path, capture_output=True, timeout=30, env=_wt_env,
                            )
                            _subprocess.run(
                                ["git", "commit", "-m", f"build: {prompt[:80]}", "--allow-empty"],
                                cwd=_worktree.path, capture_output=True, timeout=30, env=_wt_env,
                            )
                        except Exception as e:
                            logger.debug(f"Worktree commit failed: {e}")

                        _merge_ok, _merge_err = worktree_merge(repo_dir, _worktree)
                        if not _merge_ok:
                            return {
                                "success": False,
                                "result_text": native_result["result_text"],
                                "cost_usd": native_result["cost_usd"],
                                "rolled_back": False,
                                "error": f"Merge failed: {_merge_err}",
                                "provider": native_result.get("provider", "native"),
                            }
                    else:
                        # No worktree — safety tag if protected files changed
                        try:
                            from mother.build_ops import (
                                detect_protected_changes,
                                tag_before_protected_change,
                            )
                            _protected = detect_protected_changes(repo_dir, snapshot_hash)
                            if _protected:
                                tag_before_protected_change(repo_dir, _protected)
                        except Exception as e:
                            logger.debug(f"Safety tag skipped: {e}")

                    # Commit with meaningful message (on main repo)
                    _push_phase("commit", "Committing changes...")
                    _commit_hash = ""
                    try:
                        from mother.build_ops import commit_with_message
                        _commit_ok, _commit_hash = commit_with_message(
                            repo_dir, snapshot_hash, prompt,
                        )
                        if not _commit_ok and _commit_hash:
                            logger.warning(f"Commit failed: {_commit_hash}")
                    except Exception as e:
                        logger.debug(f"Smart commit failed, falling back: {e}")
                        from mother.coding_agent import _clean_env as _ce2
                        env = _ce2()
                        try:
                            _subprocess.run(
                                ["git", "add", "-A"],
                                cwd=repo_dir, capture_output=True, timeout=30, env=env,
                            )
                            _subprocess.run(
                                ["git", "commit", "-m", f"self-build: {prompt[:80]}"],
                                cwd=repo_dir, capture_output=True, timeout=30, env=env,
                            )
                        except Exception:
                            pass

                    # Auto-push if enabled
                    _push_ok = False
                    try:
                        _auto_push = getattr(self, "_config", None)
                        _push_enabled = getattr(_auto_push, "auto_push_after_build", False) if _auto_push else False
                        if not _push_enabled:
                            _push_enabled = getattr(_auto_push, "auto_push_enabled", False) if _auto_push else False
                        if _push_enabled:
                            _push_phase("push", "Pushing to remote...")
                            from mother.build_ops import push_after_build
                            _push_ok, _push_out = push_after_build(repo_dir)
                            if not _push_ok:
                                logger.warning(f"Auto-push failed: {_push_out}")
                    except Exception as e:
                        logger.debug(f"Auto-push skipped: {e}")

                    # Build journal entry
                    try:
                        from mother.build_ops import (
                            build_journal_entry_from_result,
                            append_to_build_journal,
                        )
                        _journal_result = {
                            "success": True,
                            "cost_usd": native_result["cost_usd"],
                            "duration_seconds": time.monotonic() - _build_start_mono,
                            "provider": native_result.get("provider", "native"),
                        }
                        _entry = build_journal_entry_from_result(
                            _journal_result, prompt, snapshot_hash, repo_dir,
                        )
                        append_to_build_journal(repo_dir, _entry)
                    except Exception as e:
                        logger.debug(f"Build journal skipped: {e}")

                    # Close feedback loop: observe + re-analyze + generate goals
                    _extracted_pcs = self._extract_postcodes_from_text(prompt)
                    _loop_pcs = _extracted_pcs or ()
                    try:
                        loop_result = self._close_feedback_loop(
                            target_postcodes=_loop_pcs,
                            success=True,
                            build_description=prompt[:200],
                        )
                        if loop_result.get("new_goals_added", 0):
                            logger.info(
                                f"Feedback loop: {loop_result['new_goals_added']} new goals"
                            )
                    except Exception:
                        pass

                    return {
                        "success": True,
                        "result_text": native_result["result_text"],
                        "cost_usd": native_result["cost_usd"],
                        "rolled_back": False,
                        "error": "",
                        "provider": native_result.get("provider", "native"),
                        "pushed": _push_ok,
                        "commit_hash": _commit_hash or "",
                    }

                # 3b. Native engine unavailable — fall back to CLI coding agents
                _push_phase("invoke", "Running CLI coding agent...")

                # Build provider list — override claude binary if specified
                providers = default_providers()
                if claude_path:
                    providers = [
                        CodingAgentProvider(
                            name=p.name, binary=claude_path if p.name == "claude" else p.binary,
                            env_var=p.env_var, priority=p.priority,
                            max_turns=p.max_turns, max_budget_usd=p.max_budget_usd,
                            timeout=p.timeout,
                        )
                        for p in providers
                    ]

                result = invoke_coding_agent_streaming(
                    prompt=prompt,
                    cwd=_build_dir,
                    on_event=_on_event,
                    providers=providers,
                    preferred="claude",
                    max_turns=max_turns,
                    max_budget_usd=max_budget_usd,
                )

                # Save build log regardless of outcome
                slug = prompt[:40].split("\n")[0]
                save_build_log(log_events, prompt, repo_dir, result, slug)
                # Also store on instance for external access
                self._self_build_events = list(log_events)

                if not result.success:
                    # Worktree is disposed in finally — no rollback needed
                    if not _worktree and snapshot_hash:
                        git_rollback(repo_dir, snapshot_hash)
                    # Close feedback loop on CLI failure
                    _cli_fail_pcs = self._extract_postcodes_from_text(prompt)
                    try:
                        self._close_feedback_loop(
                            target_postcodes=_cli_fail_pcs,
                            success=False,
                            failure_reason=result.error or "CLI build failed",
                        )
                    except Exception:
                        pass
                    return {
                        "success": False,
                        "result_text": result.result_text,
                        "cost_usd": result.cost_usd,
                        "rolled_back": not _worktree and bool(snapshot_hash),
                        "error": result.error,
                        "provider": getattr(result, "provider", ""),
                    }

                # Run tests in worktree (sandboxed)
                _push_phase("testing", "Running tests...")
                tests_pass = run_tests(_build_dir, sandbox_profile=_sandbox_prof)

                if not tests_pass:
                    if not _worktree and snapshot_hash:
                        git_rollback(repo_dir, snapshot_hash)
                    # Close feedback loop on test failure
                    _cli_test_fail_pcs = self._extract_postcodes_from_text(prompt)
                    try:
                        self._close_feedback_loop(
                            target_postcodes=_cli_test_fail_pcs,
                            success=False,
                            failure_reason="Tests failed after modification",
                        )
                    except Exception:
                        pass
                    return {
                        "success": False,
                        "result_text": result.result_text,
                        "cost_usd": result.cost_usd,
                        "rolled_back": not _worktree and bool(snapshot_hash),
                        "error": "Tests failed after modification",
                        "provider": getattr(result, "provider", ""),
                    }

                # Check for protected file changes
                if _worktree:
                    try:
                        _cli_wt_protected = worktree_has_protected_changes(_worktree)
                        if _cli_wt_protected:
                            logger.warning(f"Protected files modified in worktree: {_cli_wt_protected}")
                            return {
                                "success": False,
                                "result_text": result.result_text,
                                "cost_usd": result.cost_usd,
                                "rolled_back": False,
                                "error": f"Protected files modified: {', '.join(_cli_wt_protected)}",
                                "provider": getattr(result, "provider", ""),
                            }
                    except Exception as e:
                        logger.debug(f"Protected file check (CLI) skipped: {e}")

                # Merge worktree back or direct commit
                if _worktree:
                    _push_phase("merge", "Merging changes to main...")
                    _wt_env2 = __import__("os").environ.copy()
                    _wt_env2.pop("CLAUDECODE", None)
                    _wt_env2.pop("ANTHROPIC_API_KEY", None)
                    try:
                        _subprocess.run(
                            ["git", "add", "-A"],
                            cwd=_worktree.path, capture_output=True, timeout=30, env=_wt_env2,
                        )
                        _subprocess.run(
                            ["git", "commit", "-m", f"build: {prompt[:80]}", "--allow-empty"],
                            cwd=_worktree.path, capture_output=True, timeout=30, env=_wt_env2,
                        )
                    except Exception:
                        pass

                    _cli_merge_ok, _cli_merge_err = worktree_merge(repo_dir, _worktree)
                    if not _cli_merge_ok:
                        return {
                            "success": False,
                            "result_text": result.result_text,
                            "cost_usd": result.cost_usd,
                            "rolled_back": False,
                            "error": f"Merge failed: {_cli_merge_err}",
                            "provider": getattr(result, "provider", ""),
                        }
                else:
                    # Safety tag if protected files changed
                    try:
                        from mother.build_ops import (
                            detect_protected_changes,
                            tag_before_protected_change,
                        )
                        _cli_protected = detect_protected_changes(repo_dir, snapshot_hash)
                        if _cli_protected:
                            tag_before_protected_change(repo_dir, _cli_protected)
                    except Exception as e:
                        logger.debug(f"Safety tag (CLI) skipped: {e}")

                # Commit with meaningful message
                _push_phase("commit", "Committing changes...")
                _cli_commit_hash = ""
                try:
                    from mother.build_ops import commit_with_message
                    _cli_commit_ok, _cli_commit_hash = commit_with_message(
                        repo_dir, snapshot_hash, prompt,
                    )
                    if not _cli_commit_ok and _cli_commit_hash:
                        logger.warning(f"Commit (CLI) failed: {_cli_commit_hash}")
                except Exception as e:
                    logger.debug(f"Smart commit (CLI) failed, falling back: {e}")
                    from mother.coding_agent import _clean_env as _ce
                    env = _ce()
                    try:
                        _subprocess.run(
                            ["git", "add", "-A"],
                            cwd=repo_dir, capture_output=True, timeout=30, env=env,
                        )
                        _subprocess.run(
                            ["git", "commit", "-m", f"self-build: {prompt[:80]}"],
                            cwd=repo_dir, capture_output=True, timeout=30, env=env,
                        )
                    except Exception:
                        pass

                # Compute diff stats for body map
                diff_stats = {}
                try:
                    from mother.introspection import git_diff_stats as _git_diff
                    if snapshot_hash:
                        diff = _git_diff(repo_dir, snapshot_hash)
                        diff_stats = {
                            "files_modified": diff.files_modified,
                            "files_added": diff.files_added,
                            "lines_added": diff.lines_added,
                            "lines_removed": diff.lines_removed,
                            "modules_touched": diff.modules_touched,
                        }
                except Exception:
                    pass

                # Auto-push if enabled
                _cli_push_ok = False
                try:
                    _auto_push_cli = getattr(self, "_config", None)
                    _push_en = getattr(_auto_push_cli, "auto_push_after_build", False) if _auto_push_cli else False
                    if not _push_en:
                        _push_en = getattr(_auto_push_cli, "auto_push_enabled", False) if _auto_push_cli else False
                    if _push_en:
                        _push_phase("push", "Pushing to remote...")
                        from mother.build_ops import push_after_build
                        _cli_push_ok, _cli_push_out = push_after_build(repo_dir)
                        if not _cli_push_ok:
                            logger.warning(f"Auto-push (CLI) failed: {_cli_push_out}")
                except Exception as e:
                    logger.debug(f"Auto-push (CLI) skipped: {e}")

                # Build journal entry
                try:
                    from mother.build_ops import (
                        build_journal_entry_from_result,
                        append_to_build_journal,
                    )
                    _cli_journal_result = {
                        "success": True,
                        "cost_usd": result.cost_usd,
                        "duration_seconds": result.duration_seconds,
                        "provider": getattr(result, "provider", "claude"),
                        "diff_stats": diff_stats,
                    }
                    _cli_entry = build_journal_entry_from_result(
                        _cli_journal_result, prompt, snapshot_hash, repo_dir,
                    )
                    append_to_build_journal(repo_dir, _cli_entry)
                except Exception as e:
                    logger.debug(f"Build journal (CLI) skipped: {e}")

                # Close feedback loop: observe + re-analyze + generate goals
                _extracted_pcs_cli = self._extract_postcodes_from_text(prompt)
                _loop_pcs_cli = _extracted_pcs_cli or ()
                try:
                    loop_result_cli = self._close_feedback_loop(
                        target_postcodes=_loop_pcs_cli,
                        success=True,
                        build_description=prompt[:200],
                    )
                    if loop_result_cli.get("new_goals_added", 0):
                        logger.info(
                            f"Feedback loop (CLI): {loop_result_cli['new_goals_added']} new goals"
                        )
                except Exception:
                    pass

                return {
                    "success": True,
                    "result_text": result.result_text,
                    "cost_usd": result.cost_usd,
                    "rolled_back": False,
                    "error": "",
                    "diff_stats": diff_stats,
                    "provider": getattr(result, "provider", "claude"),
                    "pushed": _cli_push_ok,
                    "commit_hash": _cli_commit_hash or "",
                }
            finally:
                # Clean up worktree (always — success or failure)
                if _worktree is not None:
                    try:
                        worktree_remove(repo_dir, _worktree)
                    except Exception as e:
                        logger.debug(f"Worktree cleanup failed: {e}")

                # Push sentinel so consumer knows stream is complete.
                # This MUST be the last thing pushed to the queue.
                try:
                    event_q.put_nowait(self._SELF_BUILD_SENTINEL)
                except queue.Full:
                    pass

        self._self_build_done.clear()
        try:
            return await asyncio.to_thread(_call)
        finally:
            self._self_build_done.set()

    async def code_task(
        self,
        prompt: str,
        target_dir: str,
        allowed_tools: str = "Read,Edit,Write,Glob,Grep,Bash",
        max_turns: int = 20,
        max_budget_usd: float = 3.0,
        claude_path: str = "",
        run_tests: bool = False,
    ) -> dict:
        """Write code in a user's project via Claude Code CLI.

        Like self_build() but targets user projects, not Mother's own repo.
        Git snapshot/rollback only if target_dir is a git repo.
        Returns dict with keys: success, result_text, cost_usd, rolled_back, error, diff_stats.
        """
        def _call():
            from pathlib import Path as _Path
            from mother.coding_agent import invoke_coding_agent, CodingAgentProvider, default_providers
            from mother.claude_code import (
                git_snapshot,
                git_rollback,
                run_tests as _run_tests,
            )
            from mother.coding_agent import _clean_env
            import subprocess as _subprocess

            # Ensure target directory exists
            _Path(target_dir).mkdir(parents=True, exist_ok=True)

            # Check if target is a git repo
            is_git = (_Path(target_dir) / ".git").is_dir()

            # 1. Snapshot (only if git repo)
            snapshot_hash = git_snapshot(target_dir) if is_git else ""

            # 2a. Try native code engine first (same pattern as self_build)
            native_result = self._try_native_code_engine(
                prompt=prompt,
                repo_dir=target_dir,
                max_turns=max_turns * 2,
                max_budget_usd=max_budget_usd,
            )
            if native_result is not None:
                if not native_result["success"]:
                    if snapshot_hash:
                        git_rollback(target_dir, snapshot_hash)
                    return {
                        "success": False,
                        "result_text": native_result.get("result_text", ""),
                        "cost_usd": native_result.get("cost_usd", 0.0),
                        "rolled_back": bool(snapshot_hash),
                        "error": native_result.get("error", "native engine failed"),
                        "diff_stats": {},
                    }
                # Native engine succeeded — compute diff stats and return
                diff_stats = {}
                if is_git and snapshot_hash:
                    try:
                        from mother.introspection import git_diff_stats as _git_diff
                        diff = _git_diff(target_dir, snapshot_hash)
                        diff_stats = {
                            "files_modified": diff.files_modified,
                            "files_added": diff.files_added,
                            "lines_added": diff.lines_added,
                            "lines_removed": diff.lines_removed,
                            "modules_touched": diff.modules_touched,
                        }
                    except Exception:
                        pass
                return {
                    "success": True,
                    "result_text": native_result.get("result_text", ""),
                    "cost_usd": native_result.get("cost_usd", 0.0),
                    "rolled_back": False,
                    "error": "",
                    "diff_stats": diff_stats,
                }

            # 2b. Fallback: invoke CLI coding agent
            providers = default_providers()
            if claude_path:
                providers = [
                    CodingAgentProvider(
                        name=p.name, binary=claude_path if p.name == "claude" else p.binary,
                        env_var=p.env_var, priority=p.priority,
                        max_turns=p.max_turns, max_budget_usd=p.max_budget_usd,
                        timeout=p.timeout,
                    )
                    for p in providers
                ]
            result = invoke_coding_agent(
                prompt=prompt,
                cwd=target_dir,
                providers=providers,
                preferred="claude",
                max_turns=max_turns,
                max_budget_usd=max_budget_usd,
                allowed_tools=allowed_tools,
            )

            if not result.success:
                if snapshot_hash:
                    git_rollback(target_dir, snapshot_hash)
                return {
                    "success": False,
                    "result_text": result.result_text,
                    "cost_usd": result.cost_usd,
                    "rolled_back": bool(snapshot_hash),
                    "error": result.error,
                    "diff_stats": {},
                }

            # 3. Optionally run tests
            if run_tests:
                tests_pass = _run_tests(target_dir)
                if not tests_pass:
                    if snapshot_hash:
                        git_rollback(target_dir, snapshot_hash)
                    return {
                        "success": False,
                        "result_text": result.result_text,
                        "cost_usd": result.cost_usd,
                        "rolled_back": bool(snapshot_hash),
                        "error": "Tests failed after code changes",
                        "diff_stats": {},
                    }

            # 4. Commit if git repo
            if is_git:
                env = _clean_env()
                short_desc = prompt[:80].replace('"', "'")
                try:
                    _subprocess.run(
                        ["git", "add", "-A"],
                        cwd=target_dir, capture_output=True, timeout=30, env=env,
                    )
                    _subprocess.run(
                        ["git", "commit", "-m", f"mother: {short_desc}"],
                        cwd=target_dir, capture_output=True, timeout=30, env=env,
                    )
                except Exception:
                    pass  # Commit failure is non-fatal

            # 5. Compute diff stats
            diff_stats = {}
            if is_git and snapshot_hash:
                try:
                    from mother.introspection import git_diff_stats as _git_diff
                    diff = _git_diff(target_dir, snapshot_hash)
                    diff_stats = {
                        "files_modified": diff.files_modified,
                        "files_added": diff.files_added,
                        "lines_added": diff.lines_added,
                        "lines_removed": diff.lines_removed,
                        "modules_touched": diff.modules_touched,
                    }
                except Exception:
                    pass

            return {
                "success": True,
                "result_text": result.result_text,
                "cost_usd": result.cost_usd,
                "rolled_back": False,
                "error": "",
                "diff_stats": diff_stats,
            }

        return await asyncio.to_thread(_call)

    async def generate_project(
        self,
        blueprint: dict,
        verification: dict,
        output_dir: str,
        project_name: str = "",
        language: str = "",
        max_turns: int = 40,
        max_budget_usd: float = 5.0,
        on_event: Optional[Callable] = None,
    ) -> dict:
        """Generate a runnable project from a compiled blueprint.

        Translates blueprint into a code engine prompt via project_planner,
        then dispatches to native code engine (or CLI fallback).
        Returns dict with keys: success, project_dir, files_created, cost_usd, error.
        """
        def _call():
            from pathlib import Path as _Path
            from mother.project_planner import assemble_project_build_spec

            # Assemble the prompt spec
            spec = assemble_project_build_spec(
                blueprint=blueprint,
                verification=verification,
                output_dir=output_dir,
                project_name=project_name,
                language=language,
            )

            # Ensure project directory exists
            _Path(spec.project_dir).mkdir(parents=True, exist_ok=True)

            # Try native code engine first
            native_result = self._try_native_code_engine(
                prompt=spec.prompt,
                repo_dir=spec.project_dir,
                max_turns=max_turns,
                max_budget_usd=max_budget_usd,
                on_event=on_event,
                system_prompt=spec.system_prompt,
            )
            if native_result is not None:
                # Count files created
                files_created = []
                try:
                    for root, _dirs, files in os.walk(spec.project_dir):
                        for f in files:
                            rel = os.path.relpath(os.path.join(root, f), spec.project_dir)
                            files_created.append(rel)
                except Exception:
                    pass
                return {
                    "success": native_result.get("success", False),
                    "project_dir": spec.project_dir,
                    "project_name": spec.project_name,
                    "files_created": files_created,
                    "cost_usd": native_result.get("cost_usd", 0.0),
                    "error": native_result.get("error", ""),
                    "result_text": native_result.get("result_text", ""),
                    "component_count": spec.component_count,
                    "trust_score": spec.trust_score,
                }

            # Fallback: CLI coding agent
            try:
                from mother.coding_agent import invoke_coding_agent, default_providers
                providers = default_providers()
                result = invoke_coding_agent(
                    prompt=spec.prompt,
                    cwd=spec.project_dir,
                    providers=providers,
                    preferred="claude",
                    max_turns=max_turns,
                    max_budget_usd=max_budget_usd,
                )
                files_created = []
                try:
                    for root, _dirs, files in os.walk(spec.project_dir):
                        for f in files:
                            rel = os.path.relpath(os.path.join(root, f), spec.project_dir)
                            files_created.append(rel)
                except Exception:
                    pass
                return {
                    "success": result.success,
                    "project_dir": spec.project_dir,
                    "project_name": spec.project_name,
                    "files_created": files_created,
                    "cost_usd": result.cost_usd,
                    "error": result.error,
                    "result_text": result.result_text,
                    "component_count": spec.component_count,
                    "trust_score": spec.trust_score,
                }
            except Exception as e:
                return {
                    "success": False,
                    "project_dir": spec.project_dir,
                    "project_name": spec.project_name,
                    "files_created": [],
                    "cost_usd": 0.0,
                    "error": f"No code engine available: {e}",
                    "result_text": "",
                    "component_count": spec.component_count,
                    "trust_score": spec.trust_score,
                }

        return await asyncio.to_thread(_call)

    async def update_grid_after_build(
        self,
        target_postcodes: tuple[str, ...],
        success: bool,
        build_description: str = "",
    ) -> int:
        """Update kernel grids after self-build.

        Boosts cell confidence on success (+0.15), penalizes on failure (-0.10).
        Operates on both 'compiler-self-desc' and 'session' map_ids.
        Returns total cells updated across all maps.
        """
        def _call():
            from kernel.store import boost_cell_confidence, record_build_outcome

            delta = 0.15 if success else -0.10
            total_updated = 0

            for map_id in ("compiler-self-desc", "session"):
                updated = boost_cell_confidence(
                    map_id=map_id,
                    postcodes=target_postcodes,
                    delta=delta,
                )
                total_updated += updated

                record_build_outcome(
                    map_id=map_id,
                    postcodes=target_postcodes,
                    success=success,
                    build_description=build_description,
                )

            return total_updated

        try:
            return await asyncio.to_thread(_call)
        except Exception as e:
            logger.debug(f"update_grid_after_build error: {e}")
            return 0

    # --- Autonomy loop closure ---

    @staticmethod
    def _extract_postcodes_from_text(text: str) -> tuple[str, ...]:
        """Extract 5-axis postcodes from freeform text.

        Matches LAYER.CONCERN.SCOPE.DIMENSION.DOMAIN patterns where each
        axis is 2-4 uppercase letters (or DIMENSION like HOW_MUCH).
        Returns deduplicated tuple in order of first appearance.
        Falls back to keyword-based inference when no explicit postcodes found.
        """
        import re
        # Match postcodes: 5 dot-separated uppercase tokens (2-8 chars each for HOW_MUCH)
        pattern = r'\b([A-Z]{2,4}\.[A-Z]{2,4}\.[A-Z]{2,4}\.[A-Z_]{2,8}\.[A-Z]{2,4})\b'
        seen: dict[str, None] = {}  # ordered dedup
        for match in re.finditer(pattern, text):
            pc = match.group(1)
            if pc not in seen:
                seen[pc] = None
        if seen:
            return tuple(seen.keys())
        # Fallback: infer from keywords
        return EngineBridge._infer_postcodes_from_keywords(text)

    @staticmethod
    def _infer_postcodes_from_keywords(text: str) -> tuple[str, ...]:
        """Deterministic keyword→postcode mapping for feedback loop closure.

        When self-build prompts don't contain explicit postcodes (the common case),
        this maps goal/prompt keywords to likely affected postcodes so the observer
        can record observations on the right cells.
        """
        lower = text.lower()
        _KEYWORD_MAP = {
            "coherence":    "VER.FNC.APP.HOW.MTH",
            "completeness": "VER.FNC.APP.WHAT.MTH",
            "traceability": "VER.FNC.APP.WHY.MTH",
            "consistency":  "VER.FNC.APP.HOW.MTH",
            "confidence":   "MET.MEM.DOM.WHAT.MTH",
            "test":         "BLD.FNC.APP.HOW.MTH",
            "synthesis":    "SYN.FNC.APP.HOW.MTH",
            "verification": "VER.FNC.APP.WHAT.MTH",
            "dialogue":     "DIA.FNC.APP.HOW.MTH",
            "pattern":      "MET.MEM.DOM.WHY.MTH",
            "feedback":     "MET.GOL.DOM.WHY.MTH",
            "governor":     "GOV.FNC.APP.HOW.MTH",
            "kernel":       "KRN.ENT.CMP.WHAT.MTH",
            "grid":         "KRN.ENT.CMP.WHAT.MTH",
            "perception":   "OBS.ENV.EXT.WHAT.MTH",
            "memory":       "MET.MEM.DOM.WHAT.MTH",
            "goal":         "MET.GOL.DOM.WHAT.MTH",
            "build":        "BLD.FNC.APP.HOW.MTH",
            "compile":      "SYN.FNC.APP.HOW.MTH",
            "fix":          "BLD.FNC.APP.HOW.MTH",
        }
        matched: dict[str, None] = {}
        for keyword, postcode in _KEYWORD_MAP.items():
            if keyword in lower and postcode not in matched:
                matched[postcode] = None
        return tuple(matched.keys())

    def _close_feedback_loop(
        self,
        target_postcodes: tuple[str, ...],
        success: bool,
        build_description: str = "",
        failure_reason: str = "",
    ) -> dict:
        """Observe build outcome on grid, re-analyze, generate new goals.

        Synchronous. Called within self_build's _call() after update_grid_after_build.
        Returns {cells_observed, cells_improved, cells_degraded, transitions, new_goals_added}.
        """
        from kernel.store import load_grid, save_grid
        from kernel.observer import record_observation, apply_batch
        from mother.goal_generator import goals_from_grid, goals_from_feedback, generate_goal_set
        from mother.goals import GoalStore
        from pathlib import Path

        summary = {
            "cells_observed": 0,
            "cells_improved": 0,
            "cells_degraded": 0,
            "transitions": [],
            "new_goals_added": 0,
        }

        if not target_postcodes:
            return summary

        # 1. Load grid — try compiler-self-desc first, fall back to session
        grid = None
        active_map_id = None
        for map_id in ("compiler-self-desc", "session"):
            grid = load_grid(map_id)
            if grid is not None:
                active_map_id = map_id
                break

        if grid is None:
            logger.debug("_close_feedback_loop: no grid found")
            return summary

        # 2. Record observations for each target postcode
        event_type = "self_build"
        expected = "build success" if success else "build attempt"
        actual = "build succeeded" if success else f"build failed: {failure_reason[:100]}"

        deltas = []
        for pc in target_postcodes:
            delta = record_observation(
                grid=grid,
                postcode=pc,
                event_type=event_type,
                expected="build success",
                actual=actual,
                confirmed=success,
                anomaly=not success,
                anomaly_detail=failure_reason[:200] if not success else "",
            )
            deltas.append(delta)

        # 3. Apply batch to grid
        if deltas:
            batch = apply_batch(grid, deltas)
            summary["cells_observed"] = batch.cells_touched
            summary["cells_improved"] = batch.cells_improved
            summary["cells_degraded"] = batch.cells_degraded
            summary["transitions"] = [
                {"postcode": t[0], "from": t[1], "to": t[2]}
                for t in batch.transitions
            ]

        # 4. Save grid back
        try:
            save_grid(grid, active_map_id)
        except Exception as e:
            logger.debug(f"_close_feedback_loop grid save failed: {e}")

        # 5. Re-run goals_from_grid with updated cell data
        try:
            cell_data = [
                (pk, c.confidence, c.fill.name, c.primitive)
                for pk, c in grid.cells.items()
                if c.fill.name in ("F", "P", "Q")
            ]
            activated_layers = set(grid.activated_layers) if grid.activated_layers else set()
            grid_goals = goals_from_grid(cell_data, activated_layers, len(grid.cells))

            # 6. Merge with feedback goals if we have in-memory outcomes
            feedback_goals = []
            if hasattr(self, "_engine") and self._engine:
                try:
                    from mother.governor_feedback import analyze_outcomes
                    outcomes = getattr(self._engine, "_compilation_outcomes", [])
                    if outcomes:
                        report = analyze_outcomes(outcomes)
                        weaknesses = [
                            (w.dimension, w.severity, w.mean_score)
                            for w in report.weaknesses
                        ]
                        feedback_goals = goals_from_feedback(
                            weaknesses, report.rejection_rate, report.trend
                        )
                except Exception as e:
                    logger.debug(f"Feedback goals in loop closure skipped: {e}")

            goal_set = generate_goal_set(grid_goals, feedback_goals, [])

            # 7. Persist new goals (deduplicated)
            if goal_set.goals:
                db_path = Path.home() / ".motherlabs" / "history.db"
                gs = GoalStore(db_path)
                existing = gs.active(limit=50)

                from mother.goals import goal_dedup_key as _dedup_key
                existing_keys = [_dedup_key(g.description) for g in existing]

                _priority_map = {
                    "critical": "urgent",
                    "high": "high",
                    "medium": "normal",
                    "low": "low",
                }

                added = 0
                for goal in goal_set.goals:
                    key = _dedup_key(goal.description)
                    if any(key in ek or ek in key for ek in existing_keys):
                        continue
                    mapped = _priority_map.get(goal.priority, "normal")
                    gs.add(description=goal.description, source="system", priority=mapped)
                    existing_keys.append(key)
                    added += 1

                gs.close()
                summary["new_goals_added"] = added

        except Exception as e:
            logger.debug(f"_close_feedback_loop goal generation failed: {e}")

        return summary

    async def observe_and_reanalyze(
        self,
        target_postcodes: tuple[str, ...],
        success: bool,
        build_description: str = "",
        failure_reason: str = "",
    ) -> dict:
        """Observe build outcome on grid, re-analyze, generate new goals.

        Async wrapper around _close_feedback_loop for external callers.
        Returns {cells_observed, cells_improved, cells_degraded, transitions, new_goals_added}.
        """
        def _call():
            return self._close_feedback_loop(
                target_postcodes=target_postcodes,
                success=success,
                build_description=build_description,
                failure_reason=failure_reason,
            )

        try:
            return await asyncio.to_thread(_call)
        except Exception as e:
            logger.debug(f"observe_and_reanalyze error: {e}")
            return {
                "cells_observed": 0,
                "cells_improved": 0,
                "cells_degraded": 0,
                "transitions": [],
                "new_goals_added": 0,
            }

    def get_recent_outcomes(self, limit: int = 50) -> list:
        """Load recent compilation outcomes from persistent store.

        Bridge-safe access to core/outcome_store — daemon and other mother/
        modules should call this instead of importing from core/ directly.
        Returns list of dicts with outcome fields, or empty list on error.
        """
        try:
            from core.outcome_store import OutcomeStore
            store = OutcomeStore()
            records = store.recent(limit=limit)
            store.close()
            return [
                {
                    "compile_id": r.compile_id,
                    "input_summary": r.input_summary,
                    "trust_score": r.trust_score,
                    "completeness": r.completeness,
                    "consistency": r.consistency,
                    "coherence": r.coherence,
                    "traceability": r.traceability,
                    "component_count": r.component_count,
                    "rejected": r.rejected,
                    "rejection_reason": r.rejection_reason,
                    "domain": r.domain,
                    "compression_loss_categories": getattr(
                        r, "compression_loss_categories", ""
                    ),
                }
                for r in records
            ]
        except Exception as e:
            logger.debug(f"get_recent_outcomes error: {e}")
            return []

    # --- Appendage lifecycle ---

    async def build_appendage(
        self,
        db_path,
        name: str,
        description: str,
        capability_gap: str,
        capabilities: List[str],
        repo_dir: str,
        max_budget_usd: float = 3.0,
        claude_path: str = "",
    ) -> Dict[str, Any]:
        """Build a new appendage: scaffold -> Claude Code -> validate -> register.

        Returns {success, appendage_id, project_dir, cost_usd, error}.
        """
        def _call():
            from mother.appendage import AppendageStore
            from mother.appendage_builder import (
                BuildSpec, scaffold_project_dir, generate_build_prompt,
                validate_agent_script,
            )
            import os

            store = AppendageStore(db_path)

            # Determine base directory
            base_dir = os.path.join(
                os.path.expanduser("~"), "motherlabs", "appendages"
            )

            # Scaffold
            project_dir = scaffold_project_dir(base_dir, name)

            # Generate build prompt with parent context (#17 successor-spawning)
            parent_context = ""
            try:
                sys_summary = self.generate_system_summary()
                if sys_summary:
                    parent_context += f"PARENT SYSTEM:\n{sys_summary}\n\n"
                changelog = self.generate_changelog(db_path, limit=5)
                if changelog:
                    parent_context += f"RECENT BUILD HISTORY:\n{changelog}\n\n"
            except Exception:
                pass
            spec = BuildSpec(
                name=name,
                description=description,
                capability_gap=capability_gap,
                capabilities=tuple(capabilities),
                constraints=parent_context,
            )
            prompt = generate_build_prompt(spec)

            # Register as 'building'
            appendage_id = store.register(
                name=name,
                description=description,
                capability_gap=capability_gap,
                build_prompt=prompt,
                project_dir=project_dir,
                capabilities=capabilities,
            )
            store.update_status(appendage_id, "building")

            # Invoke coding agent with failover
            cost_usd = 0.0
            try:
                from mother.coding_agent import invoke_coding_agent, CodingAgentProvider, default_providers
                providers = default_providers()
                if claude_path:
                    providers = [
                        CodingAgentProvider(
                            name=p.name, binary=claude_path if p.name == "claude" else p.binary,
                            env_var=p.env_var, priority=p.priority,
                            max_turns=p.max_turns, max_budget_usd=p.max_budget_usd,
                            timeout=p.timeout,
                        )
                        for p in providers
                    ]
                result = invoke_coding_agent(
                    prompt=prompt,
                    cwd=project_dir,
                    providers=providers,
                    preferred="claude",
                    max_turns=15,
                    max_budget_usd=max_budget_usd,
                    timeout=300,
                )
                cost_usd = result.cost_usd

                if not result.success:
                    store.update_status(
                        appendage_id, "failed",
                        error=result.error or "Claude Code build failed",
                    )
                    store.update_cost(appendage_id, cost_usd)
                    return {
                        "success": False,
                        "appendage_id": appendage_id,
                        "project_dir": project_dir,
                        "cost_usd": cost_usd,
                        "error": result.error or "Build failed",
                    }
            except Exception as e:
                store.update_status(appendage_id, "failed", error=str(e))
                return {
                    "success": False,
                    "appendage_id": appendage_id,
                    "project_dir": project_dir,
                    "cost_usd": cost_usd,
                    "error": str(e),
                }

            # Validate the built script
            valid, msg = validate_agent_script(project_dir)
            if not valid:
                store.update_status(appendage_id, "failed", error=msg)
                store.update_cost(appendage_id, cost_usd)
                return {
                    "success": False,
                    "appendage_id": appendage_id,
                    "project_dir": project_dir,
                    "cost_usd": cost_usd,
                    "error": f"Validation failed: {msg}",
                }

            store.update_status(appendage_id, "built")
            store.update_cost(appendage_id, cost_usd)
            return {
                "success": True,
                "appendage_id": appendage_id,
                "project_dir": project_dir,
                "cost_usd": cost_usd,
                "error": "",
            }

        return await asyncio.to_thread(_call)

    async def spawn_appendage(self, db_path, appendage_id: int) -> Dict[str, Any]:
        """Spawn a built appendage as subprocess.

        Returns {success, pid, capabilities, error}.
        """
        if len(self._appendage_processes) >= self._max_concurrent_appendages:
            return {
                "success": False,
                "pid": 0,
                "capabilities": [],
                "error": f"Max concurrent appendages ({self._max_concurrent_appendages}) reached",
            }

        from mother.appendage import AppendageStore
        from mother.appendage_runtime import AppendageProcess

        store = AppendageStore(db_path)
        spec = store.get(appendage_id)
        if spec is None:
            return {"success": False, "pid": 0, "capabilities": [], "error": "Not found"}

        if spec.status not in ("built", "solidified", "failed"):
            # Already running or not ready
            if spec.status in ("spawned", "active"):
                return {
                    "success": True,
                    "pid": spec.pid,
                    "capabilities": [],
                    "error": "",
                }
            return {
                "success": False,
                "pid": 0,
                "capabilities": [],
                "error": f"Cannot spawn from status '{spec.status}'",
            }

        proc = AppendageProcess(spec.project_dir, spec.entry_point)
        result = proc.spawn(timeout=15.0)

        if result.success:
            self._appendage_processes[appendage_id] = proc
            store.update_status(
                appendage_id, "spawned",
                pid=result.pid,
            )
            return {
                "success": True,
                "pid": result.pid,
                "capabilities": list(result.capabilities),
                "error": "",
            }
        else:
            store.update_status(appendage_id, "failed", error=result.error)
            return {
                "success": False,
                "pid": 0,
                "capabilities": [],
                "error": result.error,
            }

    async def invoke_appendage(
        self,
        db_path,
        appendage_id_or_name,
        params: dict,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Invoke a running appendage. Auto-spawns if built but not running.

        Returns {success, output, error, duration}.
        """
        from mother.appendage import AppendageStore

        store = AppendageStore(db_path)

        # Resolve by name or ID
        if isinstance(appendage_id_or_name, str):
            spec = store.get_by_name(appendage_id_or_name)
        else:
            spec = store.get(appendage_id_or_name)

        if spec is None:
            return {
                "success": False,
                "output": None,
                "error": "Appendage not found",
                "duration": 0.0,
            }

        aid = spec.appendage_id

        # Auto-spawn if built but not running
        if aid not in self._appendage_processes or not self._appendage_processes[aid].is_alive():
            if spec.status in ("built", "solidified", "failed"):
                spawn_result = await self.spawn_appendage(db_path, aid)
                if not spawn_result["success"]:
                    return {
                        "success": False,
                        "output": None,
                        "error": f"Auto-spawn failed: {spawn_result['error']}",
                        "duration": 0.0,
                    }
            elif spec.status not in ("spawned", "active"):
                return {
                    "success": False,
                    "output": None,
                    "error": f"Appendage not in runnable state (status={spec.status})",
                    "duration": 0.0,
                }

        proc = self._appendage_processes.get(aid)
        if proc is None or not proc.is_alive():
            return {
                "success": False,
                "output": None,
                "error": "Process not available",
                "duration": 0.0,
            }

        result = proc.invoke(params, timeout=timeout)

        if result.success:
            store.record_use(aid)
            if spec.status == "spawned":
                store.update_status(aid, "active", pid=spec.pid)
            # Auto-solidify well-used appendages
            refreshed = store.get(aid)
            if (refreshed and refreshed.status in ("spawned", "active")
                    and refreshed.use_count >= self._min_uses_to_solidify):
                store.update_status(aid, "solidified", pid=refreshed.pid)
                logger.info(f"Auto-solidified appendage {aid} after {refreshed.use_count} uses")

        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "duration": result.duration_seconds,
        }

    async def stop_appendage(self, db_path, appendage_id: int) -> None:
        """Stop a running appendage process."""
        proc = self._appendage_processes.pop(appendage_id, None)
        if proc:
            proc.stop()

        from mother.appendage import AppendageStore
        store = AppendageStore(db_path)
        spec = store.get(appendage_id)
        if spec and spec.status in ("spawned", "active"):
            store.update_status(appendage_id, "built")

    async def dissolve_appendage(self, db_path, appendage_id: int) -> None:
        """Stop and mark an appendage as dissolved."""
        await self.stop_appendage(db_path, appendage_id)

        from mother.appendage import AppendageStore
        store = AppendageStore(db_path)
        store.update_status(appendage_id, "dissolved")

    async def solidify_appendage(self, db_path, appendage_id: int) -> None:
        """Mark an appendage as solidified (permanent, auto-spawn on start)."""
        from mother.appendage import AppendageStore
        store = AppendageStore(db_path)
        spec = store.get(appendage_id)
        if spec and spec.status in ("spawned", "active"):
            store.update_status(appendage_id, "solidified", pid=spec.pid)

    async def get_active_appendages(self, db_path) -> list:
        """Get list of active appendage dicts."""
        def _call():
            from mother.appendage import AppendageStore
            store = AppendageStore(db_path)
            appendages = store.active()
            return [
                {
                    "appendage_id": a.appendage_id,
                    "name": a.name,
                    "description": a.description,
                    "status": a.status,
                    "use_count": a.use_count,
                    "pid": a.pid,
                }
                for a in appendages
            ]

        return await asyncio.to_thread(_call)

    async def stop_all_appendages(self) -> None:
        """Stop all running appendage processes. For shutdown cleanup."""
        for aid, proc in list(self._appendage_processes.items()):
            try:
                proc.stop(timeout=3.0)
            except Exception:
                pass
        self._appendage_processes.clear()

    # --- Idea journal ---

    async def add_idea(
        self,
        db_path,
        description: str,
        source_context: str = "",
        priority: str = "normal",
    ) -> int:
        """Record a new idea. Returns idea_id."""
        def _call():
            from mother.idea_journal import IdeaJournal
            journal = IdeaJournal(db_path)
            return journal.add(description, source_context, priority)

        return await asyncio.to_thread(_call)

    async def get_pending_ideas(self, db_path, limit: int = 10) -> list:
        """Get pending ideas as list of dicts."""
        def _call():
            from mother.idea_journal import IdeaJournal
            journal = IdeaJournal(db_path)
            ideas = journal.pending(limit=limit)
            return [
                {
                    "idea_id": i.idea_id,
                    "description": i.description,
                    "priority": i.priority,
                    "timestamp": i.timestamp,
                }
                for i in ideas
            ]

        return await asyncio.to_thread(_call)

    async def count_pending_ideas(self, db_path) -> int:
        """Count pending ideas."""
        def _call():
            from mother.idea_journal import IdeaJournal
            journal = IdeaJournal(db_path)
            return journal.count_pending()

        return await asyncio.to_thread(_call)

    async def update_idea_status(
        self,
        db_path,
        idea_id: int,
        status: str,
        outcome: str = "",
    ) -> None:
        """Update idea status (pending → in_progress → done/dismissed)."""
        def _call():
            from mother.idea_journal import IdeaJournal
            journal = IdeaJournal(db_path)
            journal.update_status(idea_id, status, outcome)

        await asyncio.to_thread(_call)

    async def get_top_pending_idea(self, db_path) -> dict | None:
        """Get the highest-priority pending idea, or None."""
        def _call():
            from mother.idea_journal import IdeaJournal
            journal = IdeaJournal(db_path)
            ideas = journal.pending(limit=1)
            if not ideas:
                return None
            i = ideas[0]
            return {
                "idea_id": i.idea_id,
                "description": i.description,
                "priority": i.priority,
                "timestamp": i.timestamp,
            }

        return await asyncio.to_thread(_call)

    def generate_changelog(self, db_path, limit: int = 20) -> str:
        """Generate a formatted changelog from recent journal entries."""
        from mother.journal import BuildJournal
        journal = BuildJournal(db_path)
        entries = journal.recent(limit=limit)
        if not entries:
            return ""
        lines = []
        for e in entries:
            status = "ok" if e.success else "FAIL"
            trust = f" (trust: {e.trust_score:.0f}%)" if e.trust_score > 0 else ""
            cost = f" ${e.cost_usd:.2f}" if e.cost_usd > 0 else ""
            lines.append(f"[{status}]{trust}{cost} {e.description[:80]}")
        return "\n".join(lines)

    def generate_system_summary(self) -> str:
        """Generate a system summary from introspection topology."""
        try:
            from mother.introspection import scan_topology
            import os
            repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            topo = scan_topology(repo_dir)
            lines = [
                f"Codebase: {topo.total_files} files, {topo.total_lines:,} lines, {topo.total_tests} tests",
                f"Modules: {', '.join(f'{k} ({v})' for k, v in sorted(topo.modules.items(), key=lambda x: -x[1])[:5])}",
                f"Protected: {', '.join(topo.protected_files[:3])}",
                f"Boundary: {topo.boundary_rule}",
            ]
            return "\n".join(lines)
        except Exception:
            return ""

    def generate_handoff(self, db_path=None, snapshot: dict = None) -> str:
        """Generate a Markdown handoff document for peer delegation or session export.

        Combines introspection snapshot + active goals + recent builds into
        a formatted document suitable for handoff to another operator or instance.
        """
        lines = ["# Mother — Session Handoff\n"]

        # System summary
        sys_summary = self.generate_system_summary()
        if sys_summary:
            lines.append("## System\n")
            lines.append(sys_summary + "\n")

        # Introspection snapshot
        if snapshot:
            lines.append("## Identity\n")
            identity = snapshot.get("identity", {})
            lines.append(f"- Name: {identity.get('name', 'Mother')}")
            lines.append(f"- Personality: {identity.get('personality', 'composed')}")
            lines.append(f"- Provider: {identity.get('provider', 'unknown')}")
            session = snapshot.get("session", {})
            lines.append(f"- Messages this session: {session.get('messages', 0)}")
            lines.append(f"- Session cost: ${session.get('cost_usd', 0):.2f}\n")

        # Active goals
        if db_path:
            try:
                from mother.goals import GoalStore, compute_goal_health
                import time as _time
                store = GoalStore(db_path)
                goals = store.active(limit=10)
                now = _time.time()
                if goals:
                    lines.append("## Active Goals\n")
                    for g in goals:
                        health = compute_goal_health(g, now)
                        deadline = ""
                        if g.due_timestamp > 0:
                            days_left = (g.due_timestamp - now) / 86400
                            deadline = f" (due in {days_left:.0f}d)" if days_left > 0 else " (OVERDUE)"
                        lines.append(f"- [{g.priority}] {g.description} — health {health:.0%}{deadline}")
                    lines.append("")
                store.close()
            except Exception:
                pass

            # Recent builds
            changelog = self.generate_changelog(db_path, limit=5)
            if changelog:
                lines.append("## Recent Builds\n")
                lines.append(changelog + "\n")

        return "\n".join(lines)

    def broadcast_corpus_sync(self, summary: str, trust_score: float = 0.0, domain: str = "") -> int:
        """Broadcast a corpus_sync message to all connected peers. Returns count sent."""
        wormhole = getattr(self, '_wormhole', None)
        if not wormhole or not wormhole.connections:
            return 0
        import time as _time
        import asyncio as _asyncio
        from mother.wormhole import WormholeMessage
        msg_payload = {
            "peer_id": getattr(self, '_instance_id', 'local'),
            "summary": summary[:300],
            "trust_score": trust_score,
            "domain": domain,
        }
        count = 0
        for peer_id in list(wormhole.connections.keys()):
            try:
                wm = WormholeMessage(
                    message_id=f"sync-{int(_time.time()*1000)}",
                    message_type="corpus_sync",
                    payload=msg_payload,
                    timestamp=_time.time(),
                )
                loop = _asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(wormhole.send_message(peer_id, wm))
                    count += 1
            except Exception:
                pass
        return count

    # --- Goal store ---

    async def get_active_goals(self, db_path, limit: int = 10) -> list:
        """Get active goals as list of dicts."""
        def _call():
            from mother.goals import GoalStore
            store = GoalStore(db_path)
            goals = store.active(limit=limit)
            result = [
                {
                    "goal_id": g.goal_id,
                    "description": g.description,
                    "source": g.source,
                    "priority": g.priority,
                    "status": g.status,
                    "progress_note": g.progress_note,
                    "timestamp": g.timestamp,
                }
                for g in goals
            ]
            store.close()
            return result

        return await asyncio.to_thread(_call)

    async def add_goal(
        self,
        db_path,
        description: str,
        source: str = "user",
        priority: str = "normal",
    ) -> int:
        """Record a new goal. Returns goal_id."""
        def _call():
            from mother.goals import GoalStore
            store = GoalStore(db_path)
            gid = store.add(description, source, priority)
            store.close()
            return gid

        return await asyncio.to_thread(_call)

    async def complete_goal(
        self,
        db_path,
        goal_id: int,
        note: str = "",
    ) -> None:
        """Mark a goal as done."""
        def _call():
            from mother.goals import GoalStore
            store = GoalStore(db_path)
            store.update_status(goal_id, "done", completion_note=note)
            store.close()

        await asyncio.to_thread(_call)

    async def get_next_goal(self, db_path) -> Optional[Dict]:
        """Get highest-priority actionable goal. Returns None if empty."""
        def _call():
            from mother.goals import GoalStore
            store = GoalStore(db_path)
            goal = store.next_actionable()
            store.close()
            if goal is None:
                return None
            return {
                "goal_id": goal.goal_id,
                "description": goal.description,
                "source": goal.source,
                "priority": goal.priority,
                "status": goal.status,
                "progress_note": goal.progress_note,
                "stall_count": goal.stall_count,
                "attempt_count": goal.attempt_count,
            }

        return await asyncio.to_thread(_call)

    async def score_and_prune_goals(self, db_path, threshold: float = 0.1) -> int:
        """Auto-dismiss stale goals. Returns count dismissed."""
        def _call():
            from mother.goals import GoalStore
            store = GoalStore(db_path)
            dismissed = store.score_and_prune(threshold)
            store.close()
            return dismissed

        return await asyncio.to_thread(_call)

    async def increment_goal_engagement(self, db_path, goal_id: int) -> None:
        """Record user engagement with a goal."""
        def _call():
            from mother.goals import GoalStore
            store = GoalStore(db_path)
            store.increment_engagement(goal_id)
            store.close()

        await asyncio.to_thread(_call)

    async def increment_goal_redirect(self, db_path, goal_id: int) -> None:
        """Record user redirecting away from a goal."""
        def _call():
            from mother.goals import GoalStore
            store = GoalStore(db_path)
            store.increment_redirect(goal_id)
            store.close()

        await asyncio.to_thread(_call)

    async def increment_goal_stall(self, db_path, goal_id: int) -> int:
        """Bump stall_count, return new count."""
        def _call():
            from mother.goals import GoalStore
            store = GoalStore(db_path)
            count = store.increment_stall(goal_id)
            store.close()
            return count

        return await asyncio.to_thread(_call)

    async def reset_goal_stall(self, db_path, goal_id: int) -> None:
        """Zero stall_count — progress was made."""
        def _call():
            from mother.goals import GoalStore
            store = GoalStore(db_path)
            store.reset_stall(goal_id)
            store.close()

        await asyncio.to_thread(_call)

    async def increment_goal_attempt(self, db_path, goal_id: int) -> int:
        """Bump attempt_count, return new count."""
        def _call():
            from mother.goals import GoalStore
            store = GoalStore(db_path)
            count = store.increment_attempt(goal_id)
            store.close()
            return count

        return await asyncio.to_thread(_call)

    async def mark_goal_stuck(self, db_path, goal_id: int, note: str = "") -> None:
        """Mark goal as stuck."""
        def _call():
            from mother.goals import GoalStore
            store = GoalStore(db_path)
            store.update_status(goal_id, "stuck", progress_note=note)
            store.close()

        await asyncio.to_thread(_call)

    async def reactivate_goal(self, db_path, goal_id: int) -> None:
        """Reactivate a stuck goal (user recovery)."""
        def _call():
            from mother.goals import GoalStore
            store = GoalStore(db_path)
            store.update_status(goal_id, "active", progress_note="Reactivated")
            store.reset_stall(goal_id)
            store.close()

        await asyncio.to_thread(_call)

    # --- Goal plan execution (Phase E: compiler compiles goals) ---

    async def compile_goal_to_plan(
        self,
        db_path,
        goal_id: int,
        goal_description: str,
    ) -> Optional[Dict]:
        """Compile a goal description and create an execution plan.

        The critical moment: the compiler pointed inward.
        Returns dict with plan_id, trust_score, total_steps, or None.
        """
        engine = self._get_engine()

        def _call():
            import json
            from mother.executive import PlanStore, extract_steps_from_blueprint
            from pathlib import Path as _Path

            # 1. Compile
            result = engine.compile(goal_description)
            if not result.success or not result.blueprint:
                return None

            blueprint = result.blueprint

            # 2. Extract trust score
            trust = 0.0
            v = result.verification
            if v and isinstance(v, dict):
                trust = v.get("overall_score", 0.0)

            # 3. Extract steps from blueprint
            steps = extract_steps_from_blueprint(blueprint, goal_description=goal_description)
            if not steps:
                return None

            # 4. Enrich self_build steps with rich prompts
            target_postcodes = ()
            has_self_build = any(s.get("action_type") == "self_build" for s in steps)
            if has_self_build:
                try:
                    from mother.self_build_planner import (
                        blueprint_to_build_context,
                        goal_to_build_intent,
                        assemble_self_build_prompt,
                    )
                    repo_dir = _Path(__file__).resolve().parent.parent

                    bp_context = blueprint_to_build_context(blueprint)
                    build_intent = goal_to_build_intent(
                        {"description": goal_description, "category": "confidence"},
                        blueprint=blueprint,
                    )
                    spec = assemble_self_build_prompt(
                        build_intent=build_intent,
                        repo_dir=repo_dir,
                        blueprint_context=bp_context,
                    )
                    target_postcodes = spec.target_postcodes

                    # Override self_build step's action_arg with the rich prompt
                    for s in steps:
                        if s.get("action_type") == "self_build":
                            s["action_arg"] = spec.prompt
                except Exception as e:
                    logger.debug(f"Self-build prompt enrichment skipped: {e}")

            # 5. Persist plan
            store = PlanStore(db_path)
            plan_id = store.create_plan(
                goal_id=goal_id,
                blueprint_json=json.dumps(blueprint),
                trust_score=trust,
                steps=steps,
            )
            store.close()

            result_dict = {
                "plan_id": plan_id,
                "trust_score": trust,
                "total_steps": len(steps),
                "step_names": [s["name"] for s in steps],
            }
            if target_postcodes:
                result_dict["target_postcodes"] = target_postcodes
            return result_dict

        try:
            result = await asyncio.to_thread(_call)
            # Track cost delta
            engine_total = getattr(engine, "_session_cost_usd", 0.0)
            compile_cost = engine_total - self._engine_cost_baseline
            self._engine_cost_baseline = engine_total
            if compile_cost > 0:
                self._last_call_cost = compile_cost
                self._session_cost += compile_cost
            return result
        except Exception as e:
            logger.debug(f"compile_goal_to_plan error: {e}")
            return None

    async def get_goal_plan(self, db_path, goal_id: int) -> Optional[Dict]:
        """Get existing plan for a goal."""
        def _call():
            from mother.executive import PlanStore
            store = PlanStore(db_path)
            plan = store.get_plan_for_goal(goal_id)
            store.close()
            if plan is None:
                return None
            return {
                "plan_id": plan.plan_id,
                "goal_id": plan.goal_id,
                "trust_score": plan.trust_score,
                "status": plan.status,
                "total_steps": plan.total_steps,
                "completed_steps": plan.completed_steps,
                "steps": [
                    {
                        "step_id": s.step_id,
                        "position": s.position,
                        "name": s.name,
                        "description": s.description,
                        "action_type": s.action_type,
                        "action_arg": s.action_arg,
                        "status": s.status,
                    }
                    for s in plan.steps
                ],
            }

        return await asyncio.to_thread(_call)

    async def get_next_plan_step(self, db_path, plan_id: int) -> Optional[Dict]:
        """Get next pending step from a plan."""
        def _call():
            from mother.executive import PlanStore
            store = PlanStore(db_path)
            step = store.next_step(plan_id)
            store.close()
            if step is None:
                return None
            return {
                "step_id": step.step_id,
                "plan_id": step.plan_id,
                "position": step.position,
                "name": step.name,
                "description": step.description,
                "action_type": step.action_type,
                "action_arg": step.action_arg,
                "status": step.status,
            }

        return await asyncio.to_thread(_call)

    async def update_plan_step(
        self,
        db_path,
        step_id: int,
        status: str,
        result_note: str = "",
    ) -> None:
        """Update step status after execution. Recomputes plan progress."""
        def _call():
            from mother.executive import PlanStore
            store = PlanStore(db_path)
            store.update_step(step_id, status, result_note)
            # Get plan_id for this step to update progress
            row = store._conn.execute(
                "SELECT plan_id FROM plan_steps WHERE step_id = ?",
                (step_id,),
            ).fetchone()
            if row:
                store.update_plan_progress(row[0])
            store.close()

        await asyncio.to_thread(_call)

    # --- Project integration ---

    async def integrate_project(
        self,
        project_path: str,
        description: str,
    ) -> Dict[str, Any]:
        """Integrate existing project as registered tool."""
        def _call():
            import json
            import hashlib
            from pathlib import Path
            from mother.integrator import (
                validate_project_structure,
                extract_project_metadata,
                extract_code_files,
            )
            from core.tool_package import package_tool
            from core.determinism import compute_structural_fingerprint
            from motherlabs_platform.tool_registry import get_tool_registry
            from motherlabs_platform.instance_identity import InstanceIdentityStore

            # Validate
            p = Path(project_path).expanduser().resolve()
            valid, error = validate_project_structure(p)
            if not valid:
                return {"success": False, "error": error}

            # Extract metadata
            metadata = extract_project_metadata(p)
            code_files = extract_code_files(p)

            if not code_files:
                return {"success": False, "error": "No Python files found in project"}

            # Build minimal blueprint for fingerprinting
            blueprint = {
                "name": metadata.get("name", "imported-project"),
                "description": description or metadata.get("description", ""),
                "domain": metadata.get("domain", "software"),
                "components": metadata.get("components", []),
            }

            # Compute fingerprint
            code_hash = hashlib.sha256(
                json.dumps(sorted(code_files.keys())).encode()
            ).hexdigest()[:16]

            # Package
            instance = InstanceIdentityStore()
            record = instance.get_or_create_self()
            instance_id = record.instance_id if record else "local"

            pkg = package_tool(
                compilation_id=code_hash,
                blueprint=blueprint,
                generated_code=code_files,
                trust_score=50.0,  # External/unverified projects: 50%
                verification_badge="imported",
                fingerprint_hash=code_hash,
                instance_id=instance_id,
                domain=blueprint["domain"],
            )

            # Register
            registry = get_tool_registry()

            # Check fingerprint duplicate
            existing = registry.find_by_fingerprint(code_hash)
            if existing:
                return {
                    "success": False,
                    "error": f"Project already registered as '{existing.name}'",
                }

            try:
                pkg_id = registry.register_tool(pkg, is_local=True)
            except Exception as e:
                return {"success": False, "error": f"Registration failed: {e}"}

            # Record project path for later execution
            try:
                registry.update_metadata(pkg_id, {"project_path": str(p)})
            except Exception:
                pass  # Non-fatal if metadata update fails

            return {
                "success": True,
                "package_id": pkg_id,
                "name": pkg.metadata.name,
                "trust_score": pkg.trust_score,
                "components": len(blueprint["components"]) if blueprint["components"] else len(code_files),
                "project_path": str(p),
            }

        return await asyncio.to_thread(_call)

    # --- GitHub integration ---

    async def github_push(
        self,
        repo_dir: str = "",
        branch: str = "",
    ) -> Dict[str, any]:
        """Push commits to GitHub."""
        def _call():
            from mother.github import push_to_github
            repo = repo_dir or str(Path(__file__).resolve().parent.parent)
            result = push_to_github(repo, branch=branch)
            return {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "duration": result.duration_seconds,
            }

        return await asyncio.to_thread(_call)

    async def github_create_release(
        self,
        tag: str,
        title: str = "",
        notes: str = "",
        repo_dir: str = "",
    ) -> Dict[str, any]:
        """Create a GitHub release."""
        def _call():
            from mother.github import create_release
            repo = repo_dir or str(Path(__file__).resolve().parent.parent)
            result = create_release(tag, title=title, notes=notes, repo_dir=repo)
            return {
                "success": result.success,
                "output": result.output,
                "url": result.url,
                "error": result.error,
                "duration": result.duration_seconds,
            }

        return await asyncio.to_thread(_call)

    async def github_get_repo_info(self, repo_dir: str = "") -> Dict[str, str]:
        """Get current repository info."""
        def _call():
            from mother.github import get_repo_info
            repo = repo_dir or str(Path(__file__).resolve().parent.parent)
            return get_repo_info(repo)

        return await asyncio.to_thread(_call)

    # --- Twitter integration ---

    async def tweet(self, text: str) -> Dict[str, any]:
        """Post a tweet."""
        def _call():
            from mother.twitter import TwitterClient
            try:
                client = TwitterClient()
                result = client.post_tweet(text)
                client.close()
                return {
                    "success": result.success,
                    "tweet_id": result.tweet_id,
                    "tweet_url": result.tweet_url,
                    "error": result.error,
                    "duration": result.duration_seconds,
                }
            except (ImportError, ValueError) as e:
                return {
                    "success": False,
                    "tweet_id": "",
                    "tweet_url": "",
                    "error": str(e),
                    "duration": 0.0,
                }

        return await asyncio.to_thread(_call)

    async def get_mentions(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent Twitter mentions."""
        def _call():
            from mother.twitter import TwitterClient
            try:
                client = TwitterClient()
                mentions = client.get_mentions(limit=limit)
                client.close()
                return mentions
            except (ImportError, ValueError):
                return []

        return await asyncio.to_thread(_call)

    # --- Project publishing + social ---

    async def publish_project(
        self,
        project_dir: str,
        name: str,
        description: str = "",
        public: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Publish an emitted project to GitHub (create repo, push code)."""
        def _call():
            from mother.project_publisher import publish_project, GitIdentity
            from mother.config import load_config
            cfg = load_config()
            is_public = public if public is not None else cfg.publish_projects_public
            identity = GitIdentity(
                name=cfg.mother_git_name,
                email=cfg.mother_git_email,
            )
            result = publish_project(
                project_dir=project_dir,
                name=name,
                description=description,
                public=is_public,
                identity=identity,
            )
            return {
                "success": result.success,
                "repo_url": result.repo_url,
                "repo_name": result.repo_name,
                "commit_hash": result.commit_hash,
                "files_pushed": result.files_pushed,
                "error": result.error,
                "duration": result.duration_seconds,
            }

        return await asyncio.to_thread(_call)

    async def discord_post(
        self,
        content: str = "",
        embed: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Post a message to Discord via webhook."""
        def _call():
            from mother.discord import post_webhook
            from mother.config import load_config
            cfg = load_config()
            if not cfg.discord_webhook_url:
                return {"success": False, "error": "Discord webhook URL not configured"}
            result = post_webhook(
                webhook_url=cfg.discord_webhook_url,
                content=content,
                embed=embed,
            )
            return {
                "success": result.success,
                "message_id": result.message_id,
                "error": result.error,
                "duration": result.duration_seconds,
            }

        return await asyncio.to_thread(_call)

    async def bluesky_post(self, text: str) -> Dict[str, Any]:
        """Post to Bluesky."""
        def _call():
            from mother.bluesky import BlueskyClient
            from mother.config import load_config
            cfg = load_config()
            try:
                client = BlueskyClient(
                    handle=cfg.bluesky_handle,
                    app_password=cfg.bluesky_app_password,
                )
                result = client.post(text)
                client.close()
                return {
                    "success": result.success,
                    "post_url": result.post_url,
                    "post_uri": result.post_uri,
                    "error": result.error,
                    "duration": result.duration_seconds,
                }
            except Exception as e:
                return {
                    "success": False,
                    "post_url": "",
                    "post_uri": "",
                    "error": str(e),
                    "duration": 0.0,
                }

        return await asyncio.to_thread(_call)

    async def announce_build(
        self,
        name: str,
        description: str,
        repo_url: str = "",
        components: int = 0,
        trust: float = 0.0,
    ) -> Dict[str, Any]:
        """Enqueue build announcements to all enabled social platforms."""
        def _call():
            from mother.social_queue import SocialQueue
            from mother.config import load_config
            cfg = load_config()

            platforms = []
            if cfg.auto_tweet_enabled:
                platforms.append("twitter")
            if cfg.auto_discord_enabled and cfg.discord_webhook_url:
                platforms.append("discord")
            if cfg.auto_bluesky_enabled and cfg.bluesky_handle:
                platforms.append("bluesky")

            if not platforms:
                return {"enqueued": 0, "platforms": []}

            sq = SocialQueue()
            row_ids = sq.enqueue_announcement(
                name=name,
                description=description,
                repo_url=repo_url,
                components=components,
                trust=trust,
                platforms=platforms,
            )
            return {"enqueued": len(row_ids), "platforms": platforms}

        return await asyncio.to_thread(_call)

    async def process_social_queue(self) -> List[Dict[str, Any]]:
        """Process pending social posts. Returns list of results."""
        def _call():
            from mother.social_queue import SocialQueue
            from mother.config import load_config
            cfg = load_config()

            sq = SocialQueue()

            if sq.pending_count() == 0:
                return []

            dispatchers = {}

            # Twitter dispatcher
            if cfg.auto_tweet_enabled:
                def _twitter(content, embed):
                    from mother.twitter import TwitterClient
                    try:
                        client = TwitterClient()
                        result = client.post_tweet(content)
                        client.close()
                        return {
                            "success": result.success,
                            "url": result.tweet_url,
                            "error": result.error or "",
                        }
                    except Exception as e:
                        return {"success": False, "url": "", "error": str(e)}
                dispatchers["twitter"] = _twitter

            # Discord dispatcher
            if cfg.auto_discord_enabled and cfg.discord_webhook_url:
                def _discord(content, embed):
                    from mother.discord import post_webhook
                    result = post_webhook(
                        webhook_url=cfg.discord_webhook_url,
                        content=content,
                        embed=embed,
                    )
                    return {
                        "success": result.success,
                        "url": "",
                        "error": result.error or "",
                    }
                dispatchers["discord"] = _discord

            # Bluesky dispatcher
            if cfg.auto_bluesky_enabled and cfg.bluesky_handle:
                def _bluesky(content, embed):
                    from mother.bluesky import BlueskyClient
                    try:
                        client = BlueskyClient(
                            handle=cfg.bluesky_handle,
                            app_password=cfg.bluesky_app_password,
                        )
                        result = client.post(content)
                        client.close()
                        return {
                            "success": result.success,
                            "url": result.post_url,
                            "error": result.error or "",
                        }
                    except Exception as e:
                        return {"success": False, "url": "", "error": str(e)}
                dispatchers["bluesky"] = _bluesky

            results = sq.process_pending(dispatchers)
            return [
                {"platform": r.platform, "success": r.success, "url": r.url, "error": r.error}
                for r in results
            ]

        return await asyncio.to_thread(_call)

    async def get_social_queue_status(self) -> Dict[str, Any]:
        """Get social queue status — pending counts and recent posts."""
        def _call():
            from mother.social_queue import SocialQueue
            sq = SocialQueue()
            return {
                "pending": sq.pending_count(),
                "recent": sq.recent_posts(limit=10),
                "failed": sq.failed_posts(),
            }

        return await asyncio.to_thread(_call)

    # --- Peer networking ---

    async def discover_peers(self, timeout: float = 5.0) -> int:
        """Discover Mother instances via mDNS. Returns number found."""
        def _call():
            from mother.peer_discovery import PeerRegistry, start_discovery
            try:
                registry = PeerRegistry()
                count = start_discovery(registry, timeout=timeout)
                return count
            except ImportError:
                return 0

        return await asyncio.to_thread(_call)

    async def list_peers(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """List known peers."""
        def _call():
            from mother.peer_discovery import PeerRegistry
            try:
                registry = PeerRegistry()
                peers = registry.list_peers(active_only=active_only)
                return [
                    {
                        "instance_id": p.instance_id,
                        "name": p.name,
                        "host": p.host,
                        "port": p.port,
                        "last_seen": p.last_seen,
                        "capabilities": p.capabilities,
                    }
                    for p in peers
                ]
            except ImportError:
                return []

        return await asyncio.to_thread(_call)

    async def connect_to_peer(self, host: str, port: int) -> Dict[str, Any]:
        """Connect to a peer via wormhole. Returns connection status."""
        # This would be managed by a persistent Wormhole instance
        # For now, return stub
        return {
            "success": False,
            "error": "Wormhole not yet active (needs persistent background task)",
        }

    async def delegate_compile(
        self,
        peer_id: str,
        description: str,
        domain: str = "",
    ) -> Dict[str, Any]:
        """Delegate a compile task to a peer Mother instance."""
        # Stub for now — requires active wormhole + delegation router
        return {
            "success": False,
            "error": "Delegation not yet active (requires wormhole running)",
            "peer_id": peer_id,
        }

    async def delegate_build(
        self,
        peer_id: str,
        description: str,
    ) -> Dict[str, Any]:
        """Delegate a build task to a peer Mother instance."""
        # Stub for now — requires active wormhole + delegation router
        return {
            "success": False,
            "error": "Delegation not yet active (requires wormhole running)",
            "peer_id": peer_id,
        }

    # --- WhatsApp integration ---

    async def send_whatsapp(self, to: str, body: str) -> Dict[str, any]:
        """Send a WhatsApp message via Twilio."""
        def _call():
            from mother.whatsapp import WhatsAppClient
            try:
                # Get config values - would need to be passed from chat
                import json
                from pathlib import Path
                config_path = Path.home() / ".motherlabs" / "mother.json"
                config = json.loads(config_path.read_text())

                client = WhatsAppClient(
                    account_sid=config.get("twilio_account_sid", ""),
                    auth_token=config.get("twilio_auth_token", ""),
                    from_number=config.get("twilio_whatsapp_number", ""),
                )
                result = client.send_message(to, body)
                client.close()
                return {
                    "success": result.success,
                    "message_sid": result.message_sid,
                    "to": result.to,
                    "error": result.error,
                    "duration": result.duration_seconds,
                }
            except (ImportError, ValueError) as e:
                return {
                    "success": False,
                    "error": str(e),
                }

        return await asyncio.to_thread(_call)

    async def get_connected_peers(self) -> List[Dict[str, str]]:
        """Get list of currently connected peers from wormhole."""
        # Stub for now — will be real when wormhole is active in app lifecycle
        # For now, return peers from registry that were seen recently
        def _call():
            from mother.peer_discovery import PeerRegistry
            try:
                registry = PeerRegistry()
                peers = registry.list_peers(active_only=True, timeout=600.0)  # 10 min
                return [
                    {
                        "instance_id": p.instance_id,
                        "name": p.name,
                        "host": p.host,
                        "port": str(p.port),
                    }
                    for p in peers
                ]
            except Exception:
                return []

        return await asyncio.to_thread(_call)

    async def get_provider_balances(self, api_keys: Dict[str, str]) -> Dict[str, float]:
        """Get API balances for all configured providers."""
        def _call():
            from mother.budget_monitor import BudgetMonitor
            try:
                monitor = BudgetMonitor()
                balances = monitor.get_all_balances(api_keys)
                return {p: b.balance_usd for p, b in balances.items() if b.available}
            except ImportError:
                return {}

        return await asyncio.to_thread(_call)

    async def choose_best_provider(
        self,
        api_keys: Dict[str, str],
        session_cost: float,
        session_limit: float,
    ) -> Optional[str]:
        """Choose best provider based on balances and session state."""
        def _call():
            from mother.budget_monitor import BudgetMonitor
            try:
                monitor = BudgetMonitor()
                return monitor.choose_provider(api_keys, session_cost, session_limit)
            except ImportError:
                return None

        return await asyncio.to_thread(_call)

    # --- Multimodal cognitive infrastructure accessors ---

    @staticmethod
    def get_environment_snapshot(env_model):
        """Get current environment snapshot from model. Returns None on error."""
        try:
            from mother.environment_model import format_environment_context
            snap = env_model.snapshot()
            return snap
        except Exception:
            return None

    @staticmethod
    def get_fusion_signals(fusion):
        """Get current fusion signals. Returns empty list on error."""
        try:
            return fusion.detect()
        except Exception:
            return []

    @staticmethod
    def get_actuator_summary(store):
        """Get actuator log summary. Returns None on error."""
        try:
            return store.summary()
        except Exception:
            return None

    @staticmethod
    def record_actuator_receipt(store, action_type, success, started_at, completed_at, **kwargs):
        """Record an actuator receipt. Silent on error."""
        try:
            from mother.actuator_receipt import create_receipt
            receipt = create_receipt(action_type, success, started_at, completed_at, **kwargs)
            store.record(receipt)
        except Exception:
            pass

    @staticmethod
    def get_experience_narration(chamber_input, memory=None):
        """Compile experience and return narration. Returns empty string on error."""
        try:
            from mother.sentience import compile_experience
            result = compile_experience(chamber_input, memory)
            return result.narration if result.gate_passed else ""
        except Exception:
            return ""

    # --- Post-compile artifact accessors ---

    @staticmethod
    def render_diagram(blueprint, target_format="tree"):
        """Render a blueprint as a diagram. Returns empty string on error."""
        try:
            from mother.diagrams import translate_blueprint
            return translate_blueprint(blueprint, target_format=target_format)
        except Exception:
            return ""

    @staticmethod
    def generate_client_brief(blueprint, verification=None):
        """Generate a client-facing brief from a blueprint. Returns None on error."""
        try:
            from mother.client_docs import generate_client_brief
            return generate_client_brief(blueprint, verification=verification)
        except Exception:
            return None

    @staticmethod
    def generate_meeting_briefing(description, components=None, risks=None):
        """Generate a meeting briefing. Returns None on error."""
        try:
            from mother.meeting_prep import generate_briefing
            return generate_briefing(description, components=components or [], risks=risks or [])
        except Exception:
            return None

    @staticmethod
    def source_summary(max_words: int = 5000) -> str:
        """Read Mother's own source code via AST and return compressed summary.

        Returns empty string on any error.
        """
        try:
            from mother.source_reader import read_codebase, format_source_summary
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            snapshot = read_codebase(project_root)
            return format_source_summary(snapshot, max_words=max_words)
        except Exception as e:
            logger.debug(f"Source summary unavailable: {e}")
            return ""

    async def compile_self_context(self):
        """Read source code, feed into CONTEXT compilation, persist structural facts.

        Returns CompileResult or None on error.
        """
        try:
            from mother.source_reader import read_codebase, format_source_summary, source_snapshot_to_facts
            import os

            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            snapshot = read_codebase(project_root)
            summary = format_source_summary(snapshot, max_words=5000)

            intent = f"Understand Mother's architecture from source code:\n\n{summary}"
            result = await self.compile(intent, mode="context")

            # Also persist raw structural facts directly
            try:
                from mother.knowledge_base import KnowledgeFact, save_facts
                raw_facts = source_snapshot_to_facts(snapshot)
                kf_list = [KnowledgeFact(**f) for f in raw_facts]
                saved = save_facts(kf_list)
                logger.info(f"Source facts persisted: {saved} structural facts")
            except Exception as e:
                logger.debug(f"Source fact persistence skipped: {e}")

            return result
        except Exception as e:
            logger.debug(f"compile_self_context failed: {e}")
            return None

    @staticmethod
    def extract_brand_profile(messages):
        """Extract brand signals from user messages. Returns None on error."""
        try:
            from mother.brand_identity import extract_brand_signals
            return extract_brand_signals(messages)
        except Exception:
            return None

    @staticmethod
    def generate_negotiation_brief(goal, context="", parties=None):
        """Generate a negotiation brief. Returns None on error."""
        try:
            from mother.brand_identity import generate_negotiation_brief
            return generate_negotiation_brief(goal, context=context, parties=parties)
        except Exception:
            return None
