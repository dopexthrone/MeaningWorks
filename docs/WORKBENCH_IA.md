# Motherlabs Workbench IA
## Status: Draft derived from `docs/BLUEPRINTS_SSOT.md`

This is the workbench rebuild contract.
It translates the blueprint SSOT into UI surfaces and interaction rules.

---

## 1. Product Truth

Motherlabs is a semantic compiler with a workbench shell.
It is not a generic code IDE.
It is not a graph canvas.
It is not chat-first.

The workbench exists to make compiled context:
- readable
- navigable
- branchable
- governable
- renderable into code

Primary interaction:

`read -> navigate -> branch -> resolve -> recompile -> render`

Not:

`prompt -> wait -> skim -> reprompt`

---

## 2. Core Objects

The workbench revolves around these objects:

1. `Blueprint`
2. `Node`
3. `Postcode`
4. `Branch`
5. `Gap`
6. `Silence`
7. `Governance Report`
8. `Renderer Activity`
9. `Node Reference`

Files are secondary.
Files are one representation of resolved semantic context.
Code is not shown in the workbench.

---

## 3. Entry Flows

Supported entry paths:

1. Landing page -> compile intent -> first-pass map -> workbench
2. Landing page -> login -> compile -> workbench
3. Login -> workbench -> manual compile
4. Existing blueprint -> reopen workbench

The first compile should produce a broad map in one pass.
That broad map is enough to start reading, navigating, and branching.
Deeper compilation happens afterward.

---

## 4. Shell Layout

The shell should feel native and tool-like, but not copied from VS Code.
Use IDE density and clarity, not IDE mimicry.

### Left Rail

- Activity bar
- Workspace / blueprint switcher
- Semantic tree
- Open tabs / recent nodes

### Left Panel

Sections:
- `OPEN ITEMS`
- `WORKSPACE`
- `SEMANTIC MAP`
- `GAPS`
- `SILENCES`
- `FAILED PATHS`

`SEMANTIC MAP` is a hierarchical namespace, not a graph poster.
Primary unit is postcode + primitive + state.

### Center

The center is the main reading and building surface.
It should open one selected node or artifact as tabs.

Allowed tab types:
- `Context`
- `Why`
- `Schema`
- `Blueprint`
- `Dependencies`
- `Dependents`
- `Examples`
- `Report`

This is one semantic object shown in multiple representations.
Graph view is optional and secondary.
The export zip is where code generation starts.

### Right Panel

Grounded conversation and branching.

Contains:
- context-locked chat
- precompiled questions
- branch questions
- human-in-loop escalations

The panel must always know the current postcode or explicit branch target.
Chat must never bury the active semantic thread.
The in-workbench reading agent is `Dora`.

### Bottom Panel

Operational surfaces only:
- `PROVENANCE GATES`
- `PIPELINE`
- `OUTPUT`
- `PROBLEMS`
- `RENDERER`
- `EVENT BUS`

This is where the user sees what the compiler or coding agent is doing now.

---

## 5. Center-Pane Rules

Default open state after first compile:
- broad compiled context
- not empty editor
- not giant node graph

Visuals still matter.
They should serve fast orientation and entropy collapse.

A visual surface is valid only if it helps the user understand in milliseconds:
- what this node is
- where it sits
- what state it is in
- what opens next

Invalid visual behavior:
- decorative floating graph bubbles
- full-screen DAG theater
- diagrams that do not improve navigation

---

## 6. Semantic Map Rules

The semantic map is not mainly a diagram.
It is an addressable navigation substrate.

Canonical clickable identity:

`postcode/name`

Examples:
- `EXC.FNC.CMP.HOW.SFT/divide`
- `CTR.PLY.APP.HOW.SFT/division_guard`

Each node row must expose:
- fill state
- postcode
- primitive
- confidence
- warnings
- unresolved count

Each node opens as a document-like object, not a bubble.

The map must support:
- prefix navigation by postcode
- layer filtering
- domain filtering
- branch visibility
- reused-vs-new status

---

## 7. Build State Visibility

The user must always be able to answer:

- What are we building?
- Where are we in the pipeline?
- What is resolved?
- What is provisional?
- What is broken?
- What needs human input?
- What is the coding agent doing?
- What changed since the last compile?

This should be available at all times, but not visually noisy.

---

## 8. Chat Rules

Chat remains in the workbench, but it is not primary.
It is the deep reading layer.

Chat modes:
- `Explain current node`
- `Challenge current node`
- `Branch from current node`
- `Ask blueprint-wide question`
- `Teach current concept`
- `Audit current blueprint`

Rules:
- default to current postcode context
- allow explicit branch-out
- preserve branch history as first-class semantic objects
- never collapse important questions into an untraceable chat log

---

## 9. Renderer Integration

Claude Code / Codex should appear as renderers walking the blueprint surface.

The user should be able to see:
- current postcode
- current file
- current rendering phase
- last completed postcode
- blocked postcode
- trace back to source node

This is the bridge between compiled context and implementation.
Primary export action for MVP: `download zip`.

---

## 10. Visual Language

The workbench should feel:
- dense
- clear
- calm
- precise
- interactive

Avoid:
- oversized card UI
- soft dashboard spacing
- decorative graph hero treatment
- marketing-style chrome inside the workbench surface

Prefer:
- compact spacing
- strong active states
- stable tab rhythm
- crisp separators
- small but meaningful motion

Every visual must encode information:
- fill state
- confidence
- provenance
- human decisions
- gaps
- trust tier

---

## 11. Remove From Current Workbench

Delete or demote these patterns:
- graph-first center layout
- speculative assistant surfaces with no real backend behavior
- opened-file tabs inside the bottom tray
- duplicated compile input surfaces
- dashboard cards competing with semantic objects

---

## 12. Rebuild Order

1. Protocol-first data layer
2. Postcode-native left navigation
3. Node document center pane
4. Grounded right-side branching/chat
5. Bottom operational tray
6. Renderer activity bridge
7. Visual refinement and micro-interactions

---

## 13. Success Condition

The workbench is correct when a user can:
- enter vague intent
- get a broad compiled map
- read the project like a manual
- see where ambiguity still lives
- branch only where needed
- watch a coding agent render against the map
- trace code back to meaning

If the user still has to hold the project together in chat, the workbench failed.
If the workbench starts looking like a code editor, it also failed.
