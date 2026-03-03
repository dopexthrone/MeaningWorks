# Contributing to MeaningWorks

Hey, thanks for being interested. This project is still early — one developer, lots of moving parts, very clear direction. Here's how you can help without us stepping on each other's toes.

---

## The short version

1. Read this document
2. Open an issue before writing code
3. Follow the principles below
4. Submit a PR with tests

---

## Core principles

These aren't style preferences. They're structural constraints that the entire architecture depends on. If a contribution violates any of these, it gets rejected — not because it's bad code, but because it breaks the system's guarantees.

### Excavation, not generation
Every output must trace back to input. If you're adding a feature, it should be derivable from existing requirements, not invented because it "seems useful." Ask: *can I point to the input that implies this?*

### Provenance is non-negotiable
Every component needs a `derived_from` link. Every decision needs a trace. If you can't explain why something exists by pointing to a chain of reasoning, it doesn't belong.

### Asymmetry is a feature
Agents have complementary blind spots by design. Entity sees structure, not behavior. Process sees behavior, not structure. Don't "fix" this by making agents more capable — the limitation is the mechanism.

### Bounded, not unbounded
Everything must converge in finite steps at known cost. If your change introduces an unbounded loop, an open-ended process, or an unpredictable cost — rethink the approach.

### Determinism
Same input + same seed = same output. If your change introduces non-determinism without very good reason and explicit documentation, it won't merge.

---

## How to contribute

### Found a bug?
Open an issue. Include:
- What you expected
- What actually happened
- Minimal reproduction steps
- Which domain adapter (software, process, api, agent_system) if relevant

### Have an idea?
Open an issue first. Describe:
- What problem this solves
- How it traces to existing requirements (excavation, not invention)
- Whether it affects provenance chains
- Whether it breaks any of the 8 constraints (C001-C008)

### Writing code?

1. **Fork and branch** from `main`
2. **Write tests first** — we have 7,000+ and the bar is zero regressions
3. **Follow existing patterns** — look at how existing agents, gates, and protocols work
4. **Keep it bounded** — if your function doesn't have a clear termination condition, add one
5. **Trace everything** — `derived_from` fields, insight patterns, provenance links

### PR checklist

- [ ] All existing tests pass
- [ ] New functionality has tests
- [ ] No new hardcoded domain assumptions (use adapter protocol)
- [ ] Provenance chains intact (every component traces to input)
- [ ] Asymmetry preserved (no agent is superset of another)
- [ ] Bounded execution (finite steps, known cost)
- [ ] Unknowns are explicit, not hidden

---

## What I'm looking for help with

- **Domain adapters** — extending to new verticals beyond software/process/api/agent_system
- **Testing** — edge cases, adversarial inputs, scale testing
- **Documentation** — making the architecture accessible without dumbing it down
- **Integration** — connecting to external tools, APIs, platforms

---

## What I'm NOT looking for

- "Improvements" that make agents see everything (breaks asymmetry)
- Features that don't trace to existing requirements (breaks excavation principle)
- Unbounded processes (breaks convergence guarantee)
- Removing provenance tracking for "performance" (provenance IS the product)

---

## Code style

- Python 3.10+
- Type hints everywhere
- Frozen dataclasses for data structures
- Pure functions where possible
- `pytest` for testing
- No magic globals — everything flows through SharedState or explicit parameters

---

## Questions?

Open an issue or email [motherlabsai@gmail.com](mailto:motherlabsai@gmail.com).

This is a solo project with a clear vision. I appreciate contributions that strengthen the architecture, not ones that dilute it. If you're not sure whether something fits, just ask — I'd rather have a conversation than reject a PR.
