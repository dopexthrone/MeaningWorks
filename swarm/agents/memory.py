"""MemoryAgent — episodic memory + learned patterns.

Reads:  state.user_id, state.session_id, state.domain
Writes: state.memory_context

No LLM calls. Wraps kernel/memory.py functions.
Output includes "relevant_patterns" key for CompileAgent._enrich_description().
"""

import logging
from typing import Any, Dict, List

from swarm.agents.base import SwarmAgent
from swarm.state import SwarmState

logger = logging.getLogger("motherlabs.swarm.memory")


class MemoryAgent(SwarmAgent):
    """Episodic memory retrieval and pattern extraction."""

    name = "memory"
    criticality = "low"

    @property
    def input_keys(self) -> List[str]:
        return ["user_id", "session_id", "domain"]

    @property
    def output_keys(self) -> List[str]:
        return ["memory_context"]

    def execute(self, state: SwarmState, config: Dict[str, Any]) -> SwarmState:
        """Load memories and extract patterns.

        Steps:
        1. Load recent episodic memories (domain-filtered)
        2. Detect session patterns from memories
        3. Load persisted L2 patterns
        4. Merge + deduplicate by pattern_id
        5. Format for CompileAgent consumption
        """
        from kernel.memory import (
            load_recent_memories,
            detect_patterns,
            load_patterns,
            format_pattern_context,
        )

        # 1. Load recent memories
        try:
            memories = load_recent_memories(limit=50, domain=state.domain)
        except Exception as e:
            logger.warning("Failed to load memories: %s", e)
            memories = []

        # 2. Detect session patterns
        session_patterns = []
        if memories:
            try:
                session_patterns = detect_patterns(memories, min_occurrences=3)
            except Exception as e:
                logger.warning("Pattern detection failed: %s", e)

        # 3. Load persisted L2 patterns
        try:
            persisted_patterns = load_patterns(min_confidence=0.5)
        except Exception as e:
            logger.warning("Failed to load persisted patterns: %s", e)
            persisted_patterns = []

        # 4. Merge + deduplicate by pattern_id
        seen_ids = set()
        all_patterns = []
        for p in list(session_patterns) + list(persisted_patterns):
            pid = getattr(p, "pattern_id", id(p))
            if pid not in seen_ids:
                seen_ids.add(pid)
                all_patterns.append(p)

        # 5. Format for CompileAgent
        try:
            formatted = format_pattern_context(all_patterns)
        except Exception as e:
            logger.warning("Pattern formatting failed: %s", e)
            formatted = ""

        memory_context = {
            "relevant_patterns": formatted,
            "memory_count": len(memories),
            "session_pattern_count": len(session_patterns),
            "persisted_pattern_count": len(persisted_patterns),
            "total_unique_patterns": len(all_patterns),
        }

        logger.info(
            "MemoryAgent completed: memories=%d session_patterns=%d persisted=%d unique=%d",
            len(memories),
            len(session_patterns),
            len(persisted_patterns),
            len(all_patterns),
        )

        return state.with_updates(memory_context=memory_context)
