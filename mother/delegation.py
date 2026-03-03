"""
Task delegation — route compile/build tasks to peer Mother instances or humans.

LEAF module. Orchestrates wormhole communication for peer delegation
and structured task handoff for human delegation.

Enables Mother to:
- Delegate compile tasks to peers (route to instance with better model/cost)
- Delegate build tasks to peers (route to instance with more resources)
- Prepare structured tasks for human delegation (Genome #137)
- Wait for results and integrate them locally
- Track delegation performance (latency, success rate)

Protocol messages (via Wormhole):
- compile_request: {description, domain, constraints}
- compile_response: {blueprint, trust_score, insights, cost_usd}
- build_request: {description, emit_format}
- build_response: {project_path, files, trust_score, cost_usd}
- capability_query: {capability_name}
- capability_response: {has_capability, metadata}
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mother.wormhole import Wormhole, WormholeMessage


@dataclass(frozen=True)
class DelegationResult:
    """Outcome of a delegated task."""

    success: bool
    peer_id: str = ""
    peer_name: str = ""
    task_type: str = ""  # compile, build
    result_data: Dict[str, Any] = None
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    error: Optional[str] = None

    def __post_init__(self):
        # Ensure result_data is never None
        if self.result_data is None:
            object.__setattr__(self, 'result_data', {})


class DelegationRouter:
    """
    Routes tasks to the best peer based on capabilities and load.

    Uses the wormhole for communication. Tracks peer performance
    for intelligent routing decisions.
    """

    def __init__(self, wormhole: Wormhole):
        self.wormhole = wormhole
        self.peer_stats: Dict[str, Dict[str, Any]] = {}

    async def delegate_compile(
        self,
        peer_id: str,
        description: str,
        domain: str = "",
        timeout: float = 120.0,
    ) -> DelegationResult:
        """
        Delegate a compilation to a peer Mother instance.

        Sends compile_request, waits for compile_response.
        Returns the peer's blueprint, trust score, and insights.
        """
        start = time.time()

        peer_conn = self.wormhole.connections.get(peer_id)
        if not peer_conn:
            return DelegationResult(
                success=False,
                peer_id=peer_id,
                task_type="compile",
                error="Peer not connected",
            )

        try:
            response = await self.wormhole.request(
                peer_id=peer_id,
                message_type="compile_request",
                payload={
                    "description": description,
                    "domain": domain,
                },
                timeout=timeout,
            )

            elapsed = time.time() - start

            if not response:
                return DelegationResult(
                    success=False,
                    peer_id=peer_id,
                    peer_name=peer_conn.peer_name,
                    task_type="compile",
                    error="Timeout waiting for response",
                    duration_seconds=elapsed,
                )

            # Track stats
            self._record_delegation(peer_id, "compile", success=True, duration=elapsed)

            # Trust federation: reward successful delegation
            try:
                from mother.peer_discovery import PeerRegistry
                PeerRegistry().update_trust_score(peer_id, delta=0.05)
            except Exception:
                pass

            return DelegationResult(
                success=True,
                peer_id=peer_id,
                peer_name=peer_conn.peer_name,
                task_type="compile",
                result_data=response,
                cost_usd=response.get("cost_usd", 0.0),
                duration_seconds=elapsed,
            )

        except Exception as e:
            self._record_delegation(peer_id, "compile", success=False, duration=time.time() - start)
            return DelegationResult(
                success=False,
                peer_id=peer_id,
                peer_name=peer_conn.peer_name if peer_conn else "",
                task_type="compile",
                error=str(e),
                duration_seconds=time.time() - start,
            )

    async def delegate_build(
        self,
        peer_id: str,
        description: str,
        timeout: float = 180.0,
    ) -> DelegationResult:
        """Delegate a build to a peer Mother instance."""
        start = time.time()

        peer_conn = self.wormhole.connections.get(peer_id)
        if not peer_conn:
            return DelegationResult(
                success=False,
                peer_id=peer_id,
                task_type="build",
                error="Peer not connected",
            )

        try:
            response = await self.wormhole.request(
                peer_id=peer_id,
                message_type="build_request",
                payload={"description": description},
                timeout=timeout,
            )

            elapsed = time.time() - start

            if not response:
                return DelegationResult(
                    success=False,
                    peer_id=peer_id,
                    peer_name=peer_conn.peer_name,
                    task_type="build",
                    error="Timeout waiting for response",
                    duration_seconds=elapsed,
                )

            self._record_delegation(peer_id, "build", success=True, duration=elapsed)

            # Trust federation: reward successful delegation
            try:
                from mother.peer_discovery import PeerRegistry
                PeerRegistry().update_trust_score(peer_id, delta=0.05)
            except Exception:
                pass

            return DelegationResult(
                success=True,
                peer_id=peer_id,
                peer_name=peer_conn.peer_name,
                task_type="build",
                result_data=response,
                cost_usd=response.get("cost_usd", 0.0),
                duration_seconds=elapsed,
            )

        except Exception as e:
            self._record_delegation(peer_id, "build", success=False, duration=time.time() - start)
            return DelegationResult(
                success=False,
                peer_id=peer_id,
                peer_name=peer_conn.peer_name if peer_conn else "",
                task_type="build",
                error=str(e),
                duration_seconds=time.time() - start,
            )

    def choose_peer_for_task(
        self,
        task_type: str,
        prefer_local: bool = False,
        required_domain: str = "",
    ) -> Optional[str]:
        """
        Select the best peer for a given task type.

        Algorithm:
        - If prefer_local and local instance can handle it → None (do it locally)
        - Else: pick peer with lowest avg latency and highest success rate
        - Tie-break: most recent connection

        Returns peer_id or None (None means do it locally).
        """
        if prefer_local:
            return None

        # Get connected peers with the capability
        candidates = []
        for peer_id, conn in self.wormhole.connections.items():
            if task_type in conn.capabilities or "compile" in conn.capabilities:
                candidates.append(peer_id)

        if not candidates:
            return None

        # Sort by performance stats
        def score(peer_id):
            conn = self.wormhole.connections.get(peer_id)
            trust_bonus = 0.2 if (conn and conn.trust_verified) else 0.0
            stats = self.peer_stats.get(peer_id, {})
            successes = stats.get("successes", 0)
            failures = stats.get("failures", 0)
            total = successes + failures
            if total == 0:
                return 0.5 + trust_bonus
            success_rate = successes / total
            avg_duration = stats.get("total_duration", 0) / total if total > 0 else 999
            return success_rate / (avg_duration + 1) + trust_bonus

        candidates.sort(key=score, reverse=True)
        return candidates[0]

    def _record_delegation(self, peer_id: str, task_type: str, success: bool, duration: float):
        """Track delegation performance."""
        if peer_id not in self.peer_stats:
            self.peer_stats[peer_id] = {
                "successes": 0,
                "failures": 0,
                "total_duration": 0.0,
                "by_task": {},
            }

        stats = self.peer_stats[peer_id]
        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
        stats["total_duration"] += duration

        if task_type not in stats["by_task"]:
            stats["by_task"][task_type] = {"count": 0, "successes": 0}

        stats["by_task"][task_type]["count"] += 1
        if success:
            stats["by_task"][task_type]["successes"] += 1

    def get_peer_stats(self, peer_id: str) -> Dict[str, Any]:
        """Get performance stats for a peer."""
        return self.peer_stats.get(peer_id, {})


# =============================================================================
# Multi-instance capability profiling — Genome #81: multi-instance
# =============================================================================

@dataclass(frozen=True)
class CapabilityProfile:
    """Capability profile for a peer Mother instance."""

    instance_id: str
    capabilities: tuple  # ("compile", "build", "voice", ...)
    domain_strengths: tuple  # ("software", "process", ...)
    current_load: float  # 0.0 (idle) to 1.0 (saturated)
    trust_score: float  # 0.0 to 1.0
    model_tier: str  # "opus", "sonnet", "haiku", "unknown"


def build_capability_profile(
    instance_id: str,
    capabilities: Optional[List[str]] = None,
    domain_strengths: Optional[List[str]] = None,
    current_load: float = 0.0,
    trust_score: float = 0.5,
    model_tier: str = "unknown",
) -> CapabilityProfile:
    """Build a capability profile for a peer instance."""
    return CapabilityProfile(
        instance_id=instance_id,
        capabilities=tuple(capabilities or []),
        domain_strengths=tuple(domain_strengths or []),
        current_load=max(0.0, min(1.0, current_load)),
        trust_score=max(0.0, min(1.0, trust_score)),
        model_tier=model_tier,
    )


_MODEL_TIER_WEIGHTS: Dict[str, float] = {
    "opus": 1.0,
    "sonnet": 0.8,
    "haiku": 0.6,
    "unknown": 0.5,
}


def select_best_peer(
    profiles: List[CapabilityProfile],
    task_type: str = "compile",
    required_domain: str = "",
) -> Optional[str]:
    """Select the best peer for a task based on capability profiles.

    Scoring: trust * tier_weight * (1 - load) * domain_bonus(1.5x).
    Filters by task_type capability match.

    Returns instance_id of best peer, or None if no suitable peer.
    """
    if not profiles:
        return None

    candidates: List[tuple] = []  # (score, instance_id)

    for profile in profiles:
        # Filter: must have the required capability
        if task_type and task_type not in profile.capabilities:
            continue

        tier_weight = _MODEL_TIER_WEIGHTS.get(profile.model_tier, 0.5)
        load_factor = 1.0 - profile.current_load

        # Domain bonus
        domain_bonus = 1.0
        if required_domain and required_domain in profile.domain_strengths:
            domain_bonus = 1.5

        score = profile.trust_score * tier_weight * load_factor * domain_bonus
        candidates.append((score, profile.instance_id))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# =============================================================================
# Human Delegation — Genome #137: delegation-preparing
# =============================================================================

@dataclass
class HumanDelegationTask:
    """A structured task for delegation to a human collaborator."""

    title: str
    description: str
    context: str                            # background info the person needs
    acceptance_criteria: List[str]          # testable success conditions
    priority: str = "normal"               # "urgent", "high", "normal", "low"
    deadline: str = ""                     # human-readable deadline
    estimated_effort: str = ""             # e.g. "2-3 hours", "1 day"
    related_files: List[str] = field(default_factory=list)
    prior_attempts: List[str] = field(default_factory=list)
    escalation_path: str = ""              # when to loop Mother back in


def generate_delegation_task(
    goal_description: str,
    domain: str = "software",
    priority: str = "normal",
    deadline: str = "",
    context: str = "",
    prior_attempts: Optional[List[str]] = None,
    related_files: Optional[List[str]] = None,
) -> HumanDelegationTask:
    """Generate a structured task for human delegation from a goal.

    Transforms an internal goal into an externally-actionable task
    with enough context for a human to complete it independently.
    """
    # Build acceptance criteria from goal description
    criteria = _extract_acceptance_criteria(goal_description, domain)

    # Estimate effort from description complexity
    effort = _estimate_effort(goal_description, domain)

    return HumanDelegationTask(
        title=_extract_title(goal_description),
        description=goal_description,
        context=context or f"This task is in the {domain} domain.",
        acceptance_criteria=criteria,
        priority=priority,
        deadline=deadline,
        estimated_effort=effort,
        related_files=related_files or [],
        prior_attempts=prior_attempts or [],
        escalation_path="Loop Mother back in if blocked for more than 2 hours or if requirements are unclear.",
    )


def format_delegation_markdown(task: HumanDelegationTask) -> str:
    """Format a delegation task as a Markdown document for handoff."""
    lines = [f"# Task: {task.title}\n"]

    # Priority + deadline
    meta = [f"**Priority:** {task.priority}"]
    if task.deadline:
        meta.append(f"**Deadline:** {task.deadline}")
    if task.estimated_effort:
        meta.append(f"**Estimated effort:** {task.estimated_effort}")
    lines.append(" | ".join(meta) + "\n")

    # Description
    lines.append("## Description\n")
    lines.append(task.description + "\n")

    # Context
    if task.context:
        lines.append("## Context\n")
        lines.append(task.context + "\n")

    # Acceptance criteria
    if task.acceptance_criteria:
        lines.append("## Acceptance Criteria\n")
        for criterion in task.acceptance_criteria:
            lines.append(f"- [ ] {criterion}")
        lines.append("")

    # Related files
    if task.related_files:
        lines.append("## Related Files\n")
        for f in task.related_files:
            lines.append(f"- `{f}`")
        lines.append("")

    # Prior attempts
    if task.prior_attempts:
        lines.append("## Prior Attempts\n")
        for attempt in task.prior_attempts:
            lines.append(f"- {attempt}")
        lines.append("")

    # Escalation
    if task.escalation_path:
        lines.append("## Escalation\n")
        lines.append(task.escalation_path + "\n")

    return "\n".join(lines)


def format_delegation_json(task: HumanDelegationTask) -> Dict[str, Any]:
    """Format a delegation task as a JSON-serializable dict."""
    return {
        "title": task.title,
        "description": task.description,
        "context": task.context,
        "acceptance_criteria": task.acceptance_criteria,
        "priority": task.priority,
        "deadline": task.deadline,
        "estimated_effort": task.estimated_effort,
        "related_files": task.related_files,
        "prior_attempts": task.prior_attempts,
        "escalation_path": task.escalation_path,
    }


# --- Internal helpers ---

# =============================================================================
# Instance Specialization — Genome #162: instance-specializing
# =============================================================================

_SPECIALIZATION_THRESHOLDS = {
    "specialist_min_tasks": 10,
    "specialist_min_success": 0.75,
}

_MODEL_TASK_AFFINITY: Dict[str, tuple] = {
    "opus": ("compile", "review", "architect"),
    "sonnet": ("compile", "build", "review"),
    "haiku": ("build", "query", "format"),
}


@dataclass(frozen=True)
class SpecializationProfile:
    """Instance specialization profile for task routing."""

    instance_id: str
    specialization: str           # "specialist" or "generalist"
    domain_specialty: str         # primary domain or "general"
    success_rate: float
    avg_latency_seconds: float
    tasks_completed: int
    recommended_tasks: tuple      # task types this instance is best for


def infer_specialization(
    profile: CapabilityProfile,
    task_history: Optional[List[Dict[str, Any]]] = None,
) -> SpecializationProfile:
    """Infer specialization from capability profile and task history.

    Single domain_strength → domain_specialty.
    Model tier → base recommended_tasks from _MODEL_TASK_AFFINITY.
    Task history with ≥10 tasks + ≥0.75 success → specialist.
    Else → generalist.

    Returns frozen SpecializationProfile.
    """
    # Domain specialty from profile
    domain_specialty = "general"
    if profile.domain_strengths:
        domain_specialty = profile.domain_strengths[0]

    # Base recommended tasks from model tier
    base_tasks = _MODEL_TASK_AFFINITY.get(profile.model_tier, ("compile", "build"))

    # Analyze task history
    tasks_completed = 0
    successes = 0
    total_latency = 0.0

    if task_history:
        tasks_completed = len(task_history)
        successes = sum(1 for t in task_history if t.get("success", False))
        total_latency = sum(t.get("duration", 0.0) for t in task_history)

    success_rate = successes / tasks_completed if tasks_completed > 0 else 0.0
    avg_latency = total_latency / tasks_completed if tasks_completed > 0 else 0.0

    # Determine specialization
    min_tasks = _SPECIALIZATION_THRESHOLDS["specialist_min_tasks"]
    min_success = _SPECIALIZATION_THRESHOLDS["specialist_min_success"]

    if tasks_completed >= min_tasks and success_rate >= min_success:
        specialization = "specialist"
    else:
        specialization = "generalist"

    return SpecializationProfile(
        instance_id=profile.instance_id,
        specialization=specialization,
        domain_specialty=domain_specialty,
        success_rate=round(success_rate, 4),
        avg_latency_seconds=round(avg_latency, 2),
        tasks_completed=tasks_completed,
        recommended_tasks=base_tasks,
    )


def recommend_instance_roles(
    profiles: List[SpecializationProfile],
    task_types: List[str],
) -> List[tuple]:
    """Recommend instance-to-task assignments.

    For each task_type, find best-matching profile via specialization
    and recommended_tasks.

    Returns list of (instance_id, task_type) pairs.
    """
    if not profiles or not task_types:
        return []

    assignments: List[tuple] = []

    for task_type in task_types:
        best_profile: Optional[SpecializationProfile] = None
        best_score = -1.0

        for profile in profiles:
            score = 0.0

            # Bonus for having this task in recommended_tasks
            if task_type in profile.recommended_tasks:
                score += 2.0

            # Bonus for specialist status
            if profile.specialization == "specialist":
                score += 1.0

            # Bonus for higher success rate
            score += profile.success_rate

            # Penalty for high latency
            if profile.avg_latency_seconds > 60:
                score -= 0.5

            if score > best_score:
                best_score = score
                best_profile = profile

        if best_profile:
            assignments.append((best_profile.instance_id, task_type))

    return assignments


def _extract_title(description: str) -> str:
    """Extract a concise title from a goal description."""
    # First sentence or first 80 chars
    first_sentence = description.split(".")[0].strip()
    if len(first_sentence) <= 80:
        return first_sentence
    return first_sentence[:77] + "..."


def _extract_acceptance_criteria(description: str, domain: str) -> List[str]:
    """Generate testable acceptance criteria from a goal description."""
    criteria = []

    # Always include "task completed as described"
    criteria.append("Task completed as described")

    # Domain-specific criteria
    if domain == "software":
        criteria.append("Code compiles/runs without errors")
        criteria.append("Changes tested and verified")
    elif domain == "process":
        criteria.append("Process documented and validated")
    elif domain == "api":
        criteria.append("API endpoints respond correctly")
        criteria.append("Request/response formats verified")

    # Add "results communicated back" for delegation loop closure
    criteria.append("Results communicated back to Mother")

    return criteria


def _estimate_effort(description: str, domain: str) -> str:
    """Heuristic effort estimation from description complexity."""
    word_count = len(description.split())

    if word_count < 20:
        return "30 minutes - 1 hour"
    elif word_count < 50:
        return "1-2 hours"
    elif word_count < 100:
        return "2-4 hours"
    else:
        return "half day or more"
