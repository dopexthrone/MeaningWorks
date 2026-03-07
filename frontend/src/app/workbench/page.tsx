'use client';

import { useRouter, useSearchParams } from 'next/navigation';
import { Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { getTaskStatus, recordTaskDecision } from '@/lib/api/compile';
import { makeNodeRef, normalizeBlueprintNode } from '@/lib/api/types';
import { useCompile } from '@/lib/hooks/useCompile';
import { useInsightParser } from '@/lib/hooks/useInsightParser';
import type {
  BlueprintNode as ProtocolBlueprintNode,
  CompileResponse,
  CompilationProgress,
  FillStateCode,
  TaskStatus,
  TaskStatusResponse,
  TerminationCondition,
} from '@/lib/api/types';

const STAGE_STEPS = [
  { id: 'intent', label: 'Frame' },
  { id: 'persona', label: 'Context' },
  { id: 'entity_process', label: 'Map' },
  { id: 'synthesis', label: 'Compose' },
  { id: 'verify', label: 'Check' },
  { id: 'governor', label: 'Lock' },
] as const;

const STAGE_COPY: Record<string, string> = {
  intent: 'Framing the seed intent',
  personas: 'Building domain context',
  dialogue: 'Mapping structure and behavior',
  synthesis: 'Composing the blueprint',
  verification: 'Checking gaps and drift',
  governor: 'Locking what is buildable',
  enrichment: 'Expanding sparse regions',
  materialization: 'Packing the export surface',
  resynthesis: 'Reweaving the blueprint',
  kernel: 'Stabilizing semantic coordinates',
};

const LAYER_LABELS: Record<string, string> = {
  INT: 'Intent',
  SEM: 'Semantic',
  ORG: 'Organization',
  COG: 'Cognitive',
  AGN: 'Agency',
  STR: 'Structure',
  STA: 'State',
  IDN: 'Identity',
  TME: 'Time',
  EXC: 'Execution',
  DAT: 'Data',
  SFX: 'Side Effects',
  NET: 'Network',
  RES: 'Resource',
  OBS: 'Observability',
  SEC: 'Security',
  CTR: 'Control',
  EMG: 'Emergence',
  MET: 'Meta',
};

const COMPONENT_TO_COORDINATE = {
  entity: { layer: 'STR', concern: 'ENT', scope: 'APP', dimension: 'WHAT', domain: 'SFT' },
  process: { layer: 'EXC', concern: 'FNC', scope: 'APP', dimension: 'HOW', domain: 'SFT' },
  interface: { layer: 'NET', concern: 'SCH', scope: 'APP', dimension: 'WHERE', domain: 'SFT' },
  event: { layer: 'SFX', concern: 'EMT', scope: 'APP', dimension: 'WHEN', domain: 'SFT' },
  constraint: { layer: 'CTR', concern: 'PLY', scope: 'APP', dimension: 'HOW', domain: 'SFT' },
  subsystem: { layer: 'STR', concern: 'ENT', scope: 'DOM', dimension: 'WHAT', domain: 'SFT' },
} as const;

type WorkbenchView = 'node' | 'perspectives' | 'map' | 'live' | 'governance' | 'export';
type PerspectiveLens = 'structural' | 'behavioral' | 'business' | 'technical' | 'risk';

type BlueprintComponent = CompileResponse['blueprint']['components'][number];
type BlueprintRelationship = CompileResponse['blueprint']['relationships'][number];
type BlueprintConstraint = CompileResponse['blueprint']['constraints'][number];

interface SemanticLink {
  name: string;
  nodeRef: string;
  relation?: string;
}

interface SemanticNode {
  name: string;
  nodeRef: string;
  postcode: string;
  layer: string;
  concern: string;
  scope: string;
  dimension: string;
  domain: string;
  kind: string;
  description: string;
  derivedFrom: string;
  confidence: number;
  fillState: FillStateCode;
  notes: string[];
  attributes: Record<string, unknown>;
  methods: BlueprintComponent['methods'];
  validationRules: string[];
  stateMachine: BlueprintComponent['state_machine'];
  inbound: SemanticLink[];
  outbound: SemanticLink[];
  connections: SemanticLink[];
  readBefore: SemanticLink[];
  readAfter: SemanticLink[];
  seeAlso: SemanticLink[];
  warns: SemanticLink[];
  gaps: string[];
}

interface LayerSummary {
  layer: string;
  label: string;
  nodes: SemanticNode[];
  fillPercent: number;
  filled: number;
  partial: number;
  blocked: number;
  gaps: string[];
  silences: string[];
}

interface DoraMessage {
  id: string;
  role: 'user' | 'dora';
  text: string;
  refs: string[];
}

interface SemanticModel {
  nodes: SemanticNode[];
  nodeByRef: Map<string, SemanticNode>;
  layerSummaries: LayerSummary[];
  projectLabel: string;
  gapTexts: string[];
  silenceTexts: string[];
  weakestLayer: LayerSummary | null;
}

interface FillStateSummary {
  filled: number;
  partial: number;
  blocked: number;
  candidate: number;
  quarantined: number;
  empty: number;
}

interface TerminationViewModel {
  status: string;
  reason: string;
  title: string;
  message: string;
  nextAction: string;
  shortLabel: string;
  tone: string;
  badgeTone: string;
  semanticProgress: NonNullable<TerminationCondition['semantic_progress']> | null;
}

function relFrom(relationship: BlueprintRelationship): string {
  return relationship.from_component || relationship.from || '';
}

function relTo(relationship: BlueprintRelationship): string {
  return relationship.to_component || relationship.to || '';
}

function slugify(value: string): string {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '') || 'node';
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function roundConfidence(value: number): number {
  return Math.round(value * 100) / 100;
}

function stageIndex(progress: CompilationProgress | null): number {
  const index = progress?.stage_index ?? 0;
  return clamp(index, 0, STAGE_STEPS.length - 1);
}

function stageCopy(progress: CompilationProgress | null, taskStatus?: TaskStatus | null): string {
  if (taskStatus === 'awaiting_decision') {
    return 'Waiting for a human decision to continue this compile';
  }
  if (!progress?.current_stage) {
    return 'Waiting for compile input';
  }
  return STAGE_COPY[progress.current_stage] || 'Compiling semantic context';
}

function layerLabel(layer: string): string {
  return LAYER_LABELS[layer] || layer;
}

function extractLayerToken(text: string): string | null {
  const match = text.match(/\b(INT|SEM|ORG|COG|AGN|STR|STA|IDN|TME|EXC|DAT|SFX|NET|RES|OBS|SEC|CTR|EMG|MET)\b/);
  return match ? match[1] : null;
}

function inferCoordinate(component: BlueprintComponent) {
  const lower = `${component.name} ${component.description}`.toLowerCase();
  if (component.type === 'constraint') {
    if (/(auth|permission|token|secure|validation|sanitize|encryption|guard)/.test(lower)) {
      return { layer: 'SEC', concern: 'PLY', scope: 'APP', dimension: 'HOW', domain: 'SFT' };
    }
    if (/(limit|bound|threshold|rate)/.test(lower)) {
      return { layer: 'CTR', concern: 'LMT', scope: 'APP', dimension: 'IF', domain: 'SFT' };
    }
  }
  if (component.type === 'process' && component.state_machine) {
    return { layer: 'STA', concern: 'STA', scope: 'APP', dimension: 'WHEN', domain: 'SFT' };
  }
  return COMPONENT_TO_COORDINATE[component.type as keyof typeof COMPONENT_TO_COORDINATE]
    || { layer: 'SEM', concern: 'SEM', scope: 'APP', dimension: 'WHAT', domain: 'SFT' };
}

function inferConfidence(
  component: BlueprintComponent,
  relatedGaps: string[],
  trustScore: number,
): number {
  let score = 0.56;
  if (component.description) score += 0.12;
  if (component.derived_from) score += 0.12;
  if (Object.keys(component.attributes || {}).length > 0) score += 0.04;
  if ((component.methods || []).length > 0) score += 0.07;
  if ((component.validation_rules || []).length > 0) score += 0.05;
  if (component.state_machine) score += 0.08;
  score += ((trustScore || 0) / 100 - 0.5) * 0.08;
  score -= Math.min(0.18, relatedGaps.length * 0.08);
  return roundConfidence(clamp(score, 0.35, 0.98));
}

function inferFillState(
  component: BlueprintComponent,
  relatedGaps: string[],
  confidence: number,
): FillStateCode {
  if (relatedGaps.some((gap) => /blocked|needs|halt|missing dependency/i.test(gap))) {
    return 'B';
  }
  if (!component.derived_from || relatedGaps.length > 0 || confidence < 0.82) {
    return 'P';
  }
  return 'F';
}

function fillStateLabel(fillState: FillStateCode): string {
  const labels: Record<FillStateCode, string> = {
    F: 'Filled',
    P: 'Partial',
    E: 'Empty',
    B: 'Blocked',
    Q: 'Quarantined',
    C: 'Candidate',
  };
  return labels[fillState];
}

function getTerminationViewModel(condition?: TerminationCondition | null): TerminationViewModel | null {
  if (!condition || Object.keys(condition).length === 0) {
    return null;
  }

  const status = condition.status || 'complete';
  const reason = condition.reason || 'unknown';
  const semanticProgress = condition.semantic_progress || null;

  const defaults: Record<string, Omit<TerminationViewModel, 'status' | 'reason' | 'semanticProgress'>> = {
    awaiting_human: {
      title: 'Compilation paused for human input',
      message: 'The compiler found a semantic decision it cannot safely resolve without you.',
      nextAction: 'Review the blocking question and record a decision to continue.',
      shortLabel: 'Paused for input',
      tone: 'border-amber-500/20 bg-amber-500/8 text-amber-100',
      badgeTone: 'bg-amber-500/12 text-amber-300',
    },
    stalled: {
      title: 'Compilation stopped because semantic progress stalled',
      message: 'The compiler stopped because another loop would not materially improve the blueprint.',
      nextAction: 'Narrow the scope, answer an unresolved question, or deepen one postcode before recompiling.',
      shortLabel: 'Semantic stall',
      tone: 'border-orange-500/20 bg-orange-500/8 text-orange-100',
      badgeTone: 'bg-orange-500/12 text-orange-300',
    },
    halted: {
      title: 'Compilation halted',
      message: 'The compiler stopped because it hit a hard failure, timeout, or fidelity break.',
      nextAction: 'Review the reason, adjust the seed or constraints, then compile again.',
      shortLabel: 'Halted',
      tone: 'border-red-500/20 bg-red-500/8 text-red-100',
      badgeTone: 'bg-red-500/12 text-red-300',
    },
    complete: {
      title: 'Compilation completed',
      message: 'This compile reached the current quality floor and emitted a readable blueprint.',
      nextAction: 'Read the blueprint, inspect gaps, deepen a region, or export the bundle.',
      shortLabel: 'Complete',
      tone: 'border-emerald-500/20 bg-emerald-500/8 text-emerald-100',
      badgeTone: 'bg-emerald-500/12 text-emerald-300',
    },
  };

  const selected = defaults[status] || defaults.complete;
  const title = reason === 'quality_floor_reached'
    ? 'Compilation stopped at the current quality floor'
    : reason === 'verification_passed'
      ? 'Compilation completed with verification pass'
      : selected.title;

  return {
    status,
    reason,
    title,
    message: condition.message || selected.message,
    nextAction: condition.next_action || selected.nextAction,
    shortLabel: selected.shortLabel,
    tone: selected.tone,
    badgeTone: selected.badgeTone,
    semanticProgress,
  };
}

function fillStateTone(fillState: FillStateCode): string {
  const tones: Record<FillStateCode, string> = {
    F: 'bg-emerald-500/12 text-emerald-300 border-emerald-500/20',
    P: 'bg-amber-500/12 text-amber-300 border-amber-500/20',
    E: 'bg-slate-500/12 text-slate-300 border-slate-500/20',
    B: 'bg-orange-500/12 text-orange-300 border-orange-500/20',
    Q: 'bg-red-500/12 text-red-300 border-red-500/20',
    C: 'bg-sky-500/12 text-sky-300 border-sky-500/20',
  };
  return tones[fillState];
}

function inferNodeNotes(component: BlueprintComponent, relatedGaps: string[]): string[] {
  const notes: string[] = [];
  if ((component.methods || []).length > 0) {
    notes.push(`Carries ${(component.methods || []).length} implementation hint${(component.methods || []).length === 1 ? '' : 's'}.`);
  }
  if (component.state_machine) {
    notes.push(`Includes ${component.state_machine.states.length} state${component.state_machine.states.length === 1 ? '' : 's'}.`);
  }
  if ((component.validation_rules || []).length > 0) {
    notes.push(`Protected by ${(component.validation_rules || []).length} validation rule${(component.validation_rules || []).length === 1 ? '' : 's'}.`);
  }
  if (!component.derived_from) {
    notes.push('Direct provenance is still thin for this node.');
  }
  if (relatedGaps.length > 0) {
    notes.push(`${relatedGaps.length} open gap${relatedGaps.length === 1 ? '' : 's'} still touch this node.`);
  }
  return notes;
}

function inferKindFromProtocolNode(node: ProtocolBlueprintNode): string {
  if (node.concern === 'ENT') return 'entity';
  if (node.concern === 'FNC' || node.concern === 'STA') return 'process';
  if (node.concern === 'PLY' || node.concern === 'LMT') return 'constraint';
  if (node.concern === 'SCH') return 'interface';
  if (node.concern === 'EMT') return 'event';
  return 'semantic';
}

function buildSemanticLink(
  ref: string,
  nodesByRef: Map<string, ProtocolBlueprintNode>,
  nodesByPostcode: Map<string, ProtocolBlueprintNode>,
): SemanticLink {
  const resolved = nodesByRef.get(ref) || nodesByPostcode.get(ref);
  if (resolved) {
    return {
      name: resolved.primitive,
      nodeRef: makeNodeRef(resolved.postcode, resolved.primitive),
    };
  }

  const fallbackName = ref.includes('/') ? ref.split('/').slice(-1)[0] : ref;
  return {
    name: fallbackName,
    nodeRef: ref,
  };
}

function buildSemanticModelFromProtocolNodes(
  result: CompileResponse,
  gapTexts: string[],
  silenceTexts: string[],
): SemanticModel {
  const protocolNodes = (result.semantic_nodes || []).map((node) => normalizeBlueprintNode(node));
  const nodesByRef = new Map(protocolNodes.map((node) => [makeNodeRef(node.postcode, node.primitive), node]));
  const nodesByPostcode = new Map(protocolNodes.map((node) => [node.postcode, node]));
  const components = result.blueprint?.components || [];
  const componentByName = new Map(components.map((component) => [component.name, component]));

  const nodes = protocolNodes.map((node) => {
    const nodeRef = makeNodeRef(node.postcode, node.primitive);
    const component = componentByName.get(node.primitive);
    const relatedGaps = gapTexts.filter((gap) => {
      const lower = gap.toLowerCase();
      return lower.includes(node.primitive.toLowerCase())
        || lower.includes(nodeRef.toLowerCase())
        || lower.includes(node.postcode.toLowerCase());
    });
    const inbound = node.references.read_before.map((ref) => buildSemanticLink(ref, nodesByRef, nodesByPostcode));
    const outbound = node.references.read_after.map((ref) => buildSemanticLink(ref, nodesByRef, nodesByPostcode));
    const connections = (node.connections.length > 0 ? node.connections : [...node.references.read_before, ...node.references.read_after])
      .map((ref) => buildSemanticLink(ref, nodesByRef, nodesByPostcode));
    const seeAlso = node.references.see_also.map((ref) => buildSemanticLink(ref, nodesByRef, nodesByPostcode));
    const warns = node.references.warns.map((ref) => buildSemanticLink(ref, nodesByRef, nodesByPostcode));

    return {
      name: node.primitive,
      nodeRef,
      postcode: node.postcode,
      layer: node.layer,
      concern: node.concern,
      scope: node.scope,
      dimension: node.dimension,
      domain: node.domain,
      kind: component?.type || inferKindFromProtocolNode(node),
      description: node.description || component?.description || 'No semantic description yet.',
      derivedFrom: node.provenance.source_ref[0] || component?.derived_from || 'No direct provenance recorded.',
      confidence: node.confidence,
      fillState: node.fill_state,
      notes: node.notes,
      attributes: component?.attributes || {},
      methods: component?.methods || [],
      validationRules: Array.from(new Set([
        ...node.constraints.map((constraint) => constraint.description),
        ...(component?.validation_rules || []),
      ])),
      stateMachine: component?.state_machine || null,
      inbound,
      outbound,
      connections,
      readBefore: inbound,
      readAfter: outbound,
      seeAlso,
      warns,
      gaps: relatedGaps,
    } satisfies SemanticNode;
  }).sort((left, right) => left.nodeRef.localeCompare(right.nodeRef));

  const layerSummaries = Array.from(
    new Set([
      ...protocolNodes.map((node) => node.layer),
      ...gapTexts.map((gap) => extractLayerToken(gap)).filter((value): value is string => Boolean(value)),
      ...silenceTexts.map((silence) => extractLayerToken(silence)).filter((value): value is string => Boolean(value)),
    ]),
  )
    .sort((left, right) => left.localeCompare(right))
    .map((layer) => {
      const layerNodes = nodes.filter((node) => node.layer === layer);
      const filled = layerNodes.filter((node) => node.fillState === 'F').length;
      const partial = layerNodes.filter((node) => node.fillState === 'P').length;
      const blocked = layerNodes.filter((node) => node.fillState === 'B').length;
      const weight = layerNodes.reduce((total, node) => {
        if (node.fillState === 'F') return total + 1;
        if (node.fillState === 'P') return total + 0.65;
        if (node.fillState === 'B') return total + 0.35;
        if (node.fillState === 'C') return total + 0.45;
        if (node.fillState === 'Q') return total + 0.1;
        return total;
      }, 0);

      return {
        layer,
        label: layerLabel(layer),
        nodes: layerNodes,
        fillPercent: layerNodes.length > 0 ? Math.round((weight / layerNodes.length) * 100) : 0,
        filled,
        partial,
        blocked,
        gaps: gapTexts.filter((gap) => extractLayerToken(gap) === layer),
        silences: silenceTexts.filter((silence) => extractLayerToken(silence) === layer),
      } satisfies LayerSummary;
    });

  const weakestLayer = layerSummaries
    .filter((summary) => summary.nodes.length > 0 || summary.gaps.length > 0)
    .sort((left, right) => left.fillPercent - right.fillPercent)[0] || null;

  return {
    nodes,
    nodeByRef: new Map(nodes.map((node) => [node.nodeRef, node])),
    layerSummaries,
    projectLabel: result.project_name
      || result.blueprint?.core_need
      || result.input_text?.slice(0, 80)
      || 'Motherlabs Workspace',
    gapTexts,
    silenceTexts,
    weakestLayer,
  };
}

function buildSemanticModel(result: CompileResponse | null): SemanticModel {
  if (!result) {
    return {
      nodes: [],
      nodeByRef: new Map<string, SemanticNode>(),
      layerSummaries: [],
      projectLabel: 'Motherlabs Workspace',
      gapTexts: [],
      silenceTexts: [],
      weakestLayer: null,
    };
  }

  const components = result.blueprint?.components || [];
  const relationships = result.blueprint?.relationships || [];
  const constraints = result.blueprint?.constraints || [];
  const gapTexts = [
    ...(result.trust?.gap_report || []).map((gap) => String(gap)),
    ...(result.blueprint?.unresolved || []).map((gap) => `Unresolved: ${gap}`),
  ];
  const silenceTexts = (result.trust?.silence_zones || []).map((silence) => String(silence));

  if ((result.semantic_nodes || []).length > 0) {
    return buildSemanticModelFromProtocolNodes(result, gapTexts, silenceTexts);
  }

  const trustScore = result.trust?.overall_score || 0;

  const baseNodes = components.map((component) => {
    const coordinate = inferCoordinate(component);
    const postcode = `${coordinate.layer}.${coordinate.concern}.${coordinate.scope}.${coordinate.dimension}.${coordinate.domain}`;
    const nodeRef = makeNodeRef(postcode, component.name);
    const relatedGaps = gapTexts.filter((gap) => gap.toLowerCase().includes(component.name.toLowerCase()));
    const confidence = inferConfidence(component, relatedGaps, trustScore);
    const fillState = inferFillState(component, relatedGaps, confidence);

    return {
      component,
      postcode,
      nodeRef,
      coordinate,
      relatedGaps,
      confidence,
      fillState,
    };
  });

  const baseByName = new Map(baseNodes.map((entry) => [entry.component.name, entry]));

  const nodes = baseNodes.map((entry) => {
    const component = entry.component;
    const inbound = relationships
      .filter((relationship) => relTo(relationship) === component.name)
      .flatMap((relationship) => {
        const source = baseByName.get(relFrom(relationship));
        return source ? [{
          name: source.component.name,
          nodeRef: source.nodeRef,
          relation: relationship.type,
        }] : [];
      });

    const outbound = relationships
      .filter((relationship) => relFrom(relationship) === component.name)
      .flatMap((relationship) => {
        const target = baseByName.get(relTo(relationship));
        return target ? [{
          name: target.component.name,
          nodeRef: target.nodeRef,
          relation: relationship.type,
        }] : [];
      });

    const seeAlso = baseNodes
      .filter((candidate) => candidate.nodeRef !== entry.nodeRef && candidate.coordinate.layer === entry.coordinate.layer)
      .slice(0, 4)
      .map((candidate) => ({
        name: candidate.component.name,
        nodeRef: candidate.nodeRef,
      }));

    const relatedConstraints = constraints.filter((constraint) => constraint.applies_to?.includes(component.name));
    const warnings = outbound.slice(0, 3);
    const notes = inferNodeNotes(component, entry.relatedGaps);

    return {
      name: component.name,
      nodeRef: entry.nodeRef,
      postcode: entry.postcode,
      layer: entry.coordinate.layer,
      concern: entry.coordinate.concern,
      scope: entry.coordinate.scope,
      dimension: entry.coordinate.dimension,
      domain: entry.coordinate.domain,
      kind: component.type,
      description: component.description || 'No semantic description yet.',
      derivedFrom: component.derived_from || 'No direct provenance recorded.',
      confidence: entry.confidence,
      fillState: entry.fillState,
      notes,
      attributes: component.attributes || {},
      methods: component.methods || [],
      validationRules: component.validation_rules || [],
      stateMachine: component.state_machine || null,
      inbound,
      outbound,
      connections: [...inbound, ...outbound],
      readBefore: inbound.slice(0, 3),
      readAfter: outbound.slice(0, 3),
      seeAlso,
      warns: warnings,
      gaps: [
        ...entry.relatedGaps,
        ...relatedConstraints.map((constraint) => constraint.description),
      ],
    } satisfies SemanticNode;
  }).sort((left, right) => left.nodeRef.localeCompare(right.nodeRef));

  const layers = Array.from(
    new Set([
      ...nodes.map((node) => node.layer),
      ...gapTexts.map((gap) => extractLayerToken(gap)).filter((value): value is string => Boolean(value)),
      ...silenceTexts.map((silence) => extractLayerToken(silence)).filter((value): value is string => Boolean(value)),
    ]),
  ).sort((left, right) => left.localeCompare(right));

  const layerSummaries = layers.map((layer) => {
    const layerNodes = nodes.filter((node) => node.layer === layer);
    const filled = layerNodes.filter((node) => node.fillState === 'F').length;
    const partial = layerNodes.filter((node) => node.fillState === 'P').length;
    const blocked = layerNodes.filter((node) => node.fillState === 'B').length;
    const weight = layerNodes.reduce((total, node) => {
      if (node.fillState === 'F') return total + 1;
      if (node.fillState === 'P') return total + 0.65;
      if (node.fillState === 'B') return total + 0.35;
      if (node.fillState === 'C') return total + 0.45;
      if (node.fillState === 'Q') return total + 0.1;
      return total;
    }, 0);

    return {
      layer,
      label: layerLabel(layer),
      nodes: layerNodes,
      fillPercent: layerNodes.length > 0 ? Math.round((weight / layerNodes.length) * 100) : 0,
      filled,
      partial,
      blocked,
      gaps: gapTexts.filter((gap) => extractLayerToken(gap) === layer),
      silences: silenceTexts.filter((silence) => extractLayerToken(silence) === layer),
    } satisfies LayerSummary;
  });

  const weakestLayer = layerSummaries
    .filter((summary) => summary.nodes.length > 0 || summary.gaps.length > 0)
    .sort((left, right) => left.fillPercent - right.fillPercent)[0] || null;

  const nodeByRef = new Map(nodes.map((node) => [node.nodeRef, node]));
  const projectLabel = result.project_name
    || result.blueprint?.core_need
    || result.input_text?.slice(0, 80)
    || 'Motherlabs Workspace';

  return {
    nodes,
    nodeByRef,
    layerSummaries,
    projectLabel,
    gapTexts,
    silenceTexts,
    weakestLayer,
  };
}

function buildQuestionSet(node: SemanticNode | null, model: SemanticModel, result: CompileResponse | null): string[] {
  const questions: string[] = [];

  if (node) {
    questions.push(`Why does ${node.name} exist?`);
    questions.push(`What breaks if I change ${node.name}?`);
    if (node.stateMachine) {
      questions.push(`What does the state machine around ${node.name} look like?`);
    }
  }

  questions.push('What are the biggest gaps?');
  questions.push('What should I compile next?');

  if (result?.usage) {
    questions.push('What would this cost to build?');
  }

  if (questions.length < 6 && model.weakestLayer) {
    questions.push(`Why is the ${model.weakestLayer.label} layer thin?`);
  }

  return Array.from(new Set(questions)).slice(0, 6);
}

function formatRefs(refs: string[]): string[] {
  return Array.from(new Set(refs)).slice(0, 4);
}

function buildDoraAnswer(
  question: string,
  node: SemanticNode | null,
  model: SemanticModel,
  result: CompileResponse | null,
): { text: string; refs: string[] } {
  const lower = question.toLowerCase();

  if (lower.includes('gap') || lower.includes('broken') || lower.includes('missing')) {
    const topGaps = model.gapTexts.slice(0, 3);
    const weakestLayer = model.weakestLayer;
    return {
      text: topGaps.length > 0
        ? `The blueprint is weakest where gaps are still unresolved. ${weakestLayer ? `Right now the thinnest layer is ${weakestLayer.label} at ${weakestLayer.fillPercent}% fill.` : ''} The biggest open items are: ${topGaps.join(' | ')}`
        : 'There are no explicit verification gaps in the current blueprint. The next move is to deepen partial nodes rather than repair breakage.',
      refs: formatRefs([
        ...(node ? [node.nodeRef] : []),
        ...((weakestLayer?.nodes || []).slice(0, 2).map((candidate) => candidate.nodeRef)),
      ]),
    };
  }

  if ((lower.includes('why') || lower.includes('exist')) && node) {
    return {
      text: `${node.name} exists because the seed intent needs this semantic role. ${node.description} Its current provenance traces back to: ${node.derivedFrom}`,
      refs: formatRefs([
        node.nodeRef,
        ...node.readBefore.map((link) => link.nodeRef),
      ]),
    };
  }

  if ((lower.includes('change') || lower.includes('remove') || lower.includes('break')) && node) {
    const impacted = node.outbound.map((link) => link.name);
    return {
      text: impacted.length > 0
        ? `Changing ${node.name} will ripple into these downstream nodes: ${impacted.join(', ')}. Those are the parts most likely to drift first.`
        : `${node.name} does not currently expose downstream semantic dependents. The safer assumption is that it is locally scoped for now.`,
      refs: formatRefs([
        node.nodeRef,
        ...node.outbound.map((link) => link.nodeRef),
      ]),
    };
  }

  if (lower.includes('state machine')) {
    if (node?.stateMachine) {
      const transitions = node.stateMachine.transitions.map((transition) => `${transition.from_state} -> ${transition.to_state}`).join(', ');
      return {
        text: `${node.name} carries ${node.stateMachine.states.length} state nodes. Valid transitions currently include: ${transitions || 'No explicit transitions recorded yet.'}`,
        refs: formatRefs([node.nodeRef]),
      };
    }

    const machineOwner = model.nodes.find((candidate) => candidate.stateMachine);
    return machineOwner
      ? {
          text: `The clearest state surface right now lives on ${machineOwner.name}. Open that node if you want the temporal flow rather than the structural view.`,
          refs: formatRefs([machineOwner.nodeRef]),
        }
      : {
          text: 'No explicit state machine is compiled yet. The next compile should deepen temporal behavior if state transitions matter to this project.',
          refs: [],
        };
  }

  if (lower.includes('compile next') || lower.includes('next') || lower.includes('deepen')) {
    if (node && node.fillState !== 'F') {
      return {
        text: `${node.name} is still ${fillStateLabel(node.fillState).toLowerCase()}. The highest-leverage move is to compile deeper here before widening the map.`,
        refs: formatRefs([node.nodeRef]),
      };
    }
    if (model.weakestLayer) {
      return {
        text: `Compile the ${model.weakestLayer.label} layer next. It has the lowest fill score and the highest chance of hiding unknowns.`,
        refs: formatRefs(model.weakestLayer.nodes.slice(0, 3).map((candidate) => candidate.nodeRef)),
      };
    }
  }

  if (lower.includes('cost') || lower.includes('money') || lower.includes('price')) {
    return {
      text: result?.usage
        ? `The latest compile consumed ${result.usage.tokens.toLocaleString()} tokens and cost $${result.usage.cost_usd.toFixed(4)}. To keep spending efficient, deepen the weakest layer instead of recompiling the whole map.`
        : 'Cost telemetry is only available after a completed compile. During the run, treat node growth and gap reduction as the signal of whether the spend is earning depth.',
      refs: formatRefs(node ? [node.nodeRef] : []),
    };
  }

  if (node) {
    return {
      text: `${node.name} is currently the best semantic anchor in view. Read its node card first, then switch lenses if you want structure, behavior, business rationale, technical shape, or risk.`,
      refs: formatRefs([
        node.nodeRef,
        ...node.seeAlso.slice(0, 2).map((link) => link.nodeRef),
      ]),
    };
  }

  return {
    text: `This blueprint has ${model.nodes.length} compiled nodes across ${model.layerSummaries.length} active layers. Browse the map, pick a node, then use Dora to read deeper without losing your place.`,
    refs: formatRefs(model.nodes.slice(0, 3).map((candidate) => candidate.nodeRef)),
  };
}

function buildDeepCompilePrompt(node: SemanticNode): string {
  return `Go deeper on ${node.name}. Keep the original intent, but deepen the ${layerLabel(node.layer)} layer around ${node.nodeRef}. Expand open edge cases, constraints, and decision points without widening the whole blueprint.`;
}

function summarizeFillStates(model: SemanticModel): FillStateSummary {
  return model.nodes.reduce<FillStateSummary>((summary, node) => {
    if (node.fillState === 'F') summary.filled += 1;
    else if (node.fillState === 'P') summary.partial += 1;
    else if (node.fillState === 'B') summary.blocked += 1;
    else if (node.fillState === 'C') summary.candidate += 1;
    else if (node.fillState === 'Q') summary.quarantined += 1;
    else summary.empty += 1;
    return summary;
  }, {
    filled: 0,
    partial: 0,
    blocked: 0,
    candidate: 0,
    quarantined: 0,
    empty: 0,
  });
}

function inferDepthSnapshot(
  result: CompileResponse | null,
  model: SemanticModel,
): { label: string; score: number } {
  const counts = summarizeFillStates(model);
  const total = Math.max(model.nodes.length, 1);
  const weightedFill = (
    counts.filled
    + counts.partial * 0.7
    + counts.blocked * 0.4
    + counts.candidate * 0.35
    + counts.quarantined * 0.15
  ) / total;

  const signals = [weightedFill];
  if (typeof result?.trust?.overall_score === 'number') {
    signals.push(result.trust.overall_score / 100);
  }
  if (typeof result?.benchmark?.composite_pct === 'number') {
    signals.push(result.benchmark.composite_pct / 100);
  }

  const average = signals.reduce((sum, value) => sum + value, 0) / signals.length;
  const gapPenalty = Math.min(0.22, model.gapTexts.length * 0.03);
  const score = clamp(average - gapPenalty, 0.1, 0.97);

  if (score < 0.35) {
    return { label: 'sketch', score };
  }
  if (score < 0.55) {
    return { label: 'demo', score };
  }
  if (score < 0.78) {
    return { label: 'standard', score };
  }
  return { label: 'production', score };
}

function flattenVerificationEntries(
  verification: Record<string, unknown> | undefined,
): { label: string; detail: string }[] {
  if (!verification) {
    return [];
  }

  return Object.entries(verification).flatMap(([label, value]) => {
    if (Array.isArray(value)) {
      return value.length > 0
        ? [{ label, detail: value.map(String).slice(0, 3).join(' | ') }]
        : [];
    }

    if (value && typeof value === 'object') {
      const record = value as Record<string, unknown>;
      const parts: string[] = [];

      if (record.status != null) {
        parts.push(`status: ${String(record.status)}`);
      }
      if (record.score != null) {
        parts.push(`score: ${String(record.score)}`);
      }
      if (Array.isArray(record.gaps) && record.gaps.length > 0) {
        parts.push(`gaps: ${record.gaps.map(String).slice(0, 2).join(' | ')}`);
      }
      if (Array.isArray(record.warnings) && record.warnings.length > 0) {
        parts.push(`warnings: ${record.warnings.map(String).slice(0, 2).join(' | ')}`);
      }

      if (parts.length > 0) {
        return [{ label, detail: parts.join(' | ') }];
      }

      return Object.entries(record)
        .slice(0, 2)
        .map(([innerLabel, innerValue]) => ({
          label: `${label}.${innerLabel}`,
          detail: String(innerValue),
        }));
    }

    if (value == null) {
      return [];
    }

    return [{ label, detail: String(value) }];
  }).slice(0, 8);
}

function findNodeRefForText(model: SemanticModel, text: string): string | null {
  const lower = text.toLowerCase();
  const match = model.nodes.find((node) => (
    lower.includes(node.nodeRef.toLowerCase())
    || lower.includes(node.name.toLowerCase())
  ));
  return match?.nodeRef || null;
}

function mergeLedgerItems<T extends Record<string, unknown>>(
  first: T[],
  second: T[],
): T[] {
  const merged: T[] = [];
  const seen = new Set<string>();

  for (const item of [...first, ...second]) {
    const key = JSON.stringify(item);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    merged.push(item);
  }

  return merged;
}

function buildExportFiles(result: CompileResponse | null, model: SemanticModel): Array<{ name: string; input: string }> {
  if (!result) {
    return [];
  }

  const root = slugify(model.projectLabel || result.project_name || 'motherlabs_blueprint');
  const counts = summarizeFillStates(model);
  const depth = inferDepthSnapshot(result, model);
  const files: Array<{ name: string; input: string }> = [
    {
      name: `${root}/blueprint.json`,
      input: JSON.stringify(result.blueprint, null, 2),
    },
    {
      name: `${root}/compile-response.json`,
      input: JSON.stringify(result, null, 2),
    },
    {
      name: `${root}/trust.json`,
      input: JSON.stringify(result.trust, null, 2),
    },
  ];

  if (result.verification && Object.keys(result.verification).length > 0) {
    files.push({
      name: `${root}/verification/report.json`,
      input: JSON.stringify(result.verification, null, 2),
    });
  }

  if (result.stage_results && result.stage_results.length > 0) {
    files.push({
      name: `${root}/verification/stage-results.json`,
      input: JSON.stringify(result.stage_results, null, 2),
    });
  }

  if (result.structured_insights && result.structured_insights.length > 0) {
    files.push({
      name: `${root}/insights/structured-insights.json`,
      input: JSON.stringify(result.structured_insights, null, 2),
    });
  }

  for (const [name, content] of Object.entries(result.yaml_output || {})) {
    files.push({
      name: `${root}/yaml/${name}`,
      input: content,
    });
  }

  for (const [name, content] of Object.entries(result.materialized_output || {})) {
    files.push({
      name: `${root}/materialized/${name}`,
      input: content,
    });
  }

  for (const [name, content] of Object.entries(result.project_files || {})) {
    files.push({
      name: `${root}/project/${name}`,
      input: content,
    });
  }

  const manifest = {
    version: '1.0.0',
    exported_at: new Date().toISOString(),
    bundle_role: 'semantic_renderer_handoff',
    project: {
      label: model.projectLabel,
      domain: result.domain,
      project_name: result.project_name || null,
    },
    summary: {
      node_count: model.nodes.length,
      promoted: counts.filled,
      partial: counts.partial,
      blocked: counts.blocked,
      gaps: model.gapTexts.length,
      silences: model.silenceTexts.length,
      depth: depth.label,
      trust_score: result.trust?.overall_score ?? null,
      verification_badge: result.trust?.verification_badge ?? null,
    },
    termination_condition: result.termination_condition || {},
    governance: {
      anti_goals_checked: result.governance_report?.anti_goals_checked ?? 0,
      human_decisions: result.governance_report?.human_decisions?.length ?? 0,
      escalations: result.governance_report?.escalated?.length ?? 0,
      coverage: result.governance_report?.coverage ?? null,
    },
    renderer_laws: [
      'Blueprint supremacy: treat the blueprint as the source of truth.',
      'Use postcode/name citations when tracing work back to the semantic map.',
      'Respect open gaps and anti-goals; do not silently design around them.',
      'If a change needs new meaning, ask for recompilation instead of improvising architecture.',
    ],
    files: files.map((file) => file.name.replace(`${root}/`, '')),
  };

  files.push({
    name: `${root}/motherlabs-manifest.json`,
    input: JSON.stringify(manifest, null, 2),
  });
  files.push({
    name: `${root}/README_RENDERER.md`,
    input: [
      '# Motherlabs Renderer Handoff',
      '',
      'This bundle is a semantic renderer contract for a coding agent.',
      '',
      'Rules:',
      '- Treat `blueprint.json` and `motherlabs-manifest.json` as source of truth.',
      '- Use `postcode/name` citations when planning and implementing.',
      '- Do not redesign around open gaps; surface them or request recompilation.',
      '- Respect anti-goals, governance decisions, and termination conditions.',
      '',
      'Primary files:',
      '- `blueprint.json`',
      '- `compile-response.json`',
      '- `motherlabs-manifest.json`',
      '- `trust.json`',
      '',
      `Termination reason: ${(result.termination_condition || {}).reason || 'complete'}`,
      `Next action: ${(result.termination_condition || {}).next_action || 'Read the blueprint and render against it.'}`,
    ].join('\n'),
  });

  return files;
}

function Composer({ onCompile }: { onCompile: (description: string) => void }) {
  const [text, setText] = useState('');
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    ref.current?.focus();
  }, []);

  const submit = () => {
    if (text.trim().length >= 12) {
      onCompile(text.trim());
    }
  };

  return (
    <div className="flex h-full items-center justify-center px-6 py-10">
      <div className="w-full max-w-3xl rounded-2xl border border-[var(--border)] bg-[var(--bg-secondary)] shadow-[var(--shadow-2)]">
        <div className="border-b border-[var(--border-subtle)] px-6 py-5">
          <div className="text-[11px] uppercase tracking-[0.24em] text-[var(--accent)]">Motherlabs</div>
          <h1 className="mt-2 text-2xl font-semibold text-[var(--text-primary)]">
            Compile intent into buildable software
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-[var(--text-secondary)]">
            This is not a code editor. Start with intent. The compiler maps context, surfaces gaps,
            and turns the project into something you can actually read and build from.
          </p>
        </div>
        <div className="px-6 py-5">
          <textarea
            ref={ref}
            value={text}
            onChange={(event) => setText(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
                event.preventDefault();
                submit();
              }
            }}
            placeholder="Example: Build a semantic compiler workbench for domain experts who need AI to preserve intent, expose gaps, and export deterministic build packs."
            rows={7}
            className="w-full resize-none rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] px-4 py-4 text-sm leading-relaxed text-[var(--text-primary)] outline-none transition-colors placeholder:text-[var(--text-muted)] focus:border-[var(--accent-dim)]"
          />
          <div className="mt-4 flex items-center justify-between gap-4">
            <div className="text-[11px] text-[var(--text-muted)]">
              Read, explore, compile. Chat is the depth layer, not the starting point.
            </div>
            <button
              onClick={submit}
              disabled={text.trim().length < 12}
              className="rounded-md bg-[var(--accent)] px-4 py-2 text-[12px] font-semibold text-[var(--bg-primary)] transition-opacity disabled:cursor-not-allowed disabled:opacity-35"
            >
              Compile
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function RefButton({
  link,
  onOpen,
}: {
  link: SemanticLink;
  onOpen: (nodeRef: string) => void;
}) {
  return (
    <button
      onClick={() => onOpen(link.nodeRef)}
      className="rounded-md border border-[var(--border)] bg-[var(--bg-primary)] px-2 py-1 text-left text-[10px] text-[var(--text-secondary)] transition-colors hover:border-[var(--accent-dim)] hover:text-[var(--text-primary)]"
    >
      <div className="font-mono text-[10px] text-[var(--accent)]">{link.nodeRef}</div>
      {link.relation && <div className="mt-0.5 text-[9px] uppercase tracking-[0.18em]">{link.relation}</div>}
    </button>
  );
}

function FillBadge({ fillState }: { fillState: FillStateCode }) {
  return (
    <span className={`inline-flex items-center rounded-md border px-2 py-1 text-[10px] uppercase tracking-[0.18em] ${fillStateTone(fillState)}`}>
      {fillStateLabel(fillState)}
    </span>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  return (
    <div className="flex items-center gap-3">
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--bg-tertiary)]">
        <div
          className="h-full rounded-full bg-[var(--accent)]"
          style={{ width: `${Math.round(value * 100)}%` }}
        />
      </div>
      <span className="w-12 text-right font-mono text-[11px] text-[var(--text-secondary)]">
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  );
}

function TerrainSidebar({
  model,
  selectedLayer,
  selectedNodeRef,
  onSelectLayer,
  onSelectNode,
  onOpenMap,
}: {
  model: SemanticModel;
  selectedLayer: string | null;
  selectedNodeRef: string | null;
  onSelectLayer: (layer: string) => void;
  onSelectNode: (nodeRef: string) => void;
  onOpenMap: () => void;
}) {
  const visibleLayer = selectedLayer || model.layerSummaries[0]?.layer || null;
  const visibleNodes = visibleLayer
    ? model.nodes.filter((node) => node.layer === visibleLayer)
    : model.nodes;

  return (
    <aside className="border-b border-[var(--border-subtle)] bg-[var(--bg-secondary)] lg:border-b-0 lg:border-r">
      <div className="border-b border-[var(--border-subtle)] px-4 py-4">
        <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Workspace</div>
        <div className="mt-2 text-sm font-semibold text-[var(--text-primary)]">{model.projectLabel}</div>
        <button
          onClick={onOpenMap}
          className="mt-3 rounded-md border border-[var(--border)] px-3 py-1.5 text-[11px] text-[var(--text-secondary)] transition-colors hover:border-[var(--accent-dim)] hover:text-[var(--text-primary)]"
        >
          Open map view
        </button>
      </div>

      <div className="max-h-[28vh] overflow-auto border-b border-[var(--border-subtle)] px-3 py-3 lg:max-h-none">
        <div className="mb-2 text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Terrain</div>
        <div className="space-y-2">
          {model.layerSummaries.map((summary) => (
            <button
              key={summary.layer}
              onClick={() => onSelectLayer(summary.layer)}
              className={`w-full rounded-lg border px-3 py-2 text-left transition-colors ${
                summary.layer === visibleLayer
                  ? 'border-[var(--accent-dim)] bg-[var(--accent)]/8'
                  : 'border-[var(--border)] bg-[var(--bg-primary)] hover:border-[var(--border-strong)]'
              }`}
            >
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="font-mono text-[11px] text-[var(--text-primary)]">{summary.layer}</div>
                  <div className="text-[10px] text-[var(--text-muted)]">{summary.label}</div>
                </div>
                <div className="text-right">
                  <div className="text-[11px] font-semibold text-[var(--text-primary)]">{summary.fillPercent}%</div>
                  <div className="text-[10px] text-[var(--text-muted)]">{summary.nodes.length} nodes</div>
                </div>
              </div>
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-[var(--bg-tertiary)]">
                <div
                  className="h-full rounded-full bg-[var(--accent)]"
                  style={{ width: `${summary.fillPercent}%` }}
                />
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="max-h-[30vh] overflow-auto border-b border-[var(--border-subtle)] px-3 py-3 lg:max-h-none">
        <div className="mb-2 text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">
          Nodes{visibleLayer ? ` in ${visibleLayer}` : ''}
        </div>
        <div className="space-y-1.5">
          {visibleNodes.length === 0 && (
            <div className="rounded-lg border border-dashed border-[var(--border)] px-3 py-3 text-[11px] text-[var(--text-muted)]">
              No nodes in this layer yet.
            </div>
          )}
          {visibleNodes.map((node) => (
            <button
              key={node.nodeRef}
              onClick={() => onSelectNode(node.nodeRef)}
              className={`w-full rounded-lg border px-3 py-2 text-left transition-colors ${
                node.nodeRef === selectedNodeRef
                  ? 'border-[var(--accent-dim)] bg-[var(--accent)]/8'
                  : 'border-[var(--border)] bg-[var(--bg-primary)] hover:border-[var(--border-strong)]'
              }`}
            >
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="truncate text-[12px] font-medium text-[var(--text-primary)]">{node.name}</div>
                  <div className="truncate font-mono text-[10px] text-[var(--text-muted)]">{node.nodeRef}</div>
                </div>
                <FillBadge fillState={node.fillState} />
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="max-h-[24vh] overflow-auto px-3 py-3 lg:max-h-none">
        <div className="mb-2 text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Open gaps</div>
        <div className="space-y-1.5">
          {model.gapTexts.length === 0 && (
            <div className="rounded-lg border border-dashed border-[var(--border)] px-3 py-3 text-[11px] text-[var(--text-muted)]">
              No gaps detected in this compile.
            </div>
          )}
          {model.gapTexts.slice(0, 5).map((gap) => (
            <div key={gap} className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2 text-[11px] leading-relaxed text-amber-300">
              {gap}
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}

function NodeCardView({
  node,
  inputText,
  onOpenNode,
  onDeepCompile,
  onAskDora,
}: {
  node: SemanticNode | null;
  inputText: string | undefined;
  onOpenNode: (nodeRef: string) => void;
  onDeepCompile: (node: SemanticNode) => void;
  onAskDora: (question: string) => void;
}) {
  if (!node) {
    return (
      <div className="flex h-full items-center justify-center px-6">
        <div className="max-w-md text-center">
          <div className="text-sm font-semibold text-[var(--text-primary)]">Select a node</div>
          <p className="mt-2 text-sm leading-relaxed text-[var(--text-secondary)]">
            The node card is the atomic reading surface. Pick any semantic node from the terrain to inspect what it is, why it exists, and what connects to it.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto">
      <div className="border-b border-[var(--border-subtle)] px-6 py-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="font-mono text-[11px] text-[var(--accent)]">{node.nodeRef}</div>
            <div className="mt-2 flex items-center gap-3">
              <h2 className="text-2xl font-semibold text-[var(--text-primary)]">{node.name}</h2>
              <FillBadge fillState={node.fillState} />
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-3 text-[11px] text-[var(--text-muted)]">
              <span>{layerLabel(node.layer)}</span>
              <span>{node.kind}</span>
              <span>{node.connections.length} connections</span>
            </div>
          </div>
          <div className="w-full max-w-xs space-y-2">
            <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Confidence</div>
            <ConfidenceBar value={node.confidence} />
          </div>
        </div>
      </div>

      <div className="grid gap-5 px-6 py-5 xl:grid-cols-[minmax(0,1.4fr),minmax(300px,0.9fr)]">
        <div className="space-y-5">
          <section>
            <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Description</div>
            <p className="mt-2 text-sm leading-7 text-[var(--text-secondary)]">{node.description}</p>
          </section>

          {node.notes.length > 0 && (
            <section>
              <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Notes</div>
              <div className="mt-2 space-y-2">
                {node.notes.map((note) => (
                  <div key={note} className="rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)] px-3 py-2 text-[12px] leading-relaxed text-[var(--text-secondary)]">
                    {note}
                  </div>
                ))}
              </div>
            </section>
          )}

          {node.connections.length > 0 && (
            <section>
              <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Connections</div>
              <div className="mt-2 grid gap-2 md:grid-cols-2">
                {node.connections.map((link) => (
                  <RefButton key={`${node.nodeRef}:${link.nodeRef}`} link={link} onOpen={onOpenNode} />
                ))}
              </div>
            </section>
          )}

          <section className="grid gap-4 md:grid-cols-2">
            <div>
              <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Read before</div>
              <div className="mt-2 space-y-2">
                {node.readBefore.length === 0 && (
                  <div className="rounded-lg border border-dashed border-[var(--border)] px-3 py-3 text-[11px] text-[var(--text-muted)]">
                    No upstream context is required before this node.
                  </div>
                )}
                {node.readBefore.map((link) => (
                  <RefButton key={`before:${link.nodeRef}`} link={link} onOpen={onOpenNode} />
                ))}
              </div>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Warns</div>
              <div className="mt-2 space-y-2">
                {node.warns.length === 0 && (
                  <div className="rounded-lg border border-dashed border-[var(--border)] px-3 py-3 text-[11px] text-[var(--text-muted)]">
                    No downstream warning markers are attached yet.
                  </div>
                )}
                {node.warns.map((link) => (
                  <RefButton key={`warn:${link.nodeRef}`} link={link} onOpen={onOpenNode} />
                ))}
              </div>
            </div>
          </section>
        </div>

        <div className="space-y-5">
          <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4">
            <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Provenance trace</div>
            <div className="mt-3 space-y-3 text-[12px] leading-relaxed text-[var(--text-secondary)]">
              <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2">
                <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">Seed</div>
                <div className="mt-1">{inputText || 'Seed intent not recorded.'}</div>
              </div>
              <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2">
                <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">Derived from</div>
                <div className="mt-1 text-[var(--accent)]">{node.derivedFrom}</div>
              </div>
            </div>
          </section>

          <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4">
            <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Read next</div>
            <div className="mt-2 space-y-2">
              {node.readAfter.length === 0 && (
                <div className="rounded-lg border border-dashed border-[var(--border)] px-3 py-3 text-[11px] text-[var(--text-muted)]">
                  This node does not open a deeper branch yet.
                </div>
              )}
              {node.readAfter.map((link) => (
                <RefButton key={`after:${link.nodeRef}`} link={link} onOpen={onOpenNode} />
              ))}
            </div>
          </section>

          {node.gaps.length > 0 && (
            <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4">
              <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Open pressure</div>
              <div className="mt-2 space-y-2">
                {node.gaps.map((gap) => (
                  <div key={gap} className="rounded-lg border border-amber-500/20 bg-amber-500/8 px-3 py-2 text-[11px] leading-relaxed text-amber-300">
                    {gap}
                  </div>
                ))}
              </div>
            </section>
          )}

          <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4">
            <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Actions</div>
            <div className="mt-3 flex flex-wrap gap-2">
              <button
                onClick={() => onAskDora(`Why does ${node.name} exist?`)}
                className="rounded-md border border-[var(--border)] px-3 py-1.5 text-[11px] text-[var(--text-secondary)] transition-colors hover:border-[var(--accent-dim)] hover:text-[var(--text-primary)]"
              >
                Ask Dora about this
              </button>
              <button
                onClick={() => onDeepCompile(node)}
                className="rounded-md bg-[var(--accent)] px-3 py-1.5 text-[11px] font-semibold text-[var(--bg-primary)]"
              >
                Compile deeper here
              </button>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

function PerspectiveView({
  node,
  lens,
  onLensChange,
  onOpenNode,
}: {
  node: SemanticNode | null;
  lens: PerspectiveLens;
  onLensChange: (lens: PerspectiveLens) => void;
  onOpenNode: (nodeRef: string) => void;
}) {
  const lenses: { id: PerspectiveLens; label: string }[] = [
    { id: 'structural', label: 'Structural' },
    { id: 'behavioral', label: 'Behavioral' },
    { id: 'business', label: 'Business' },
    { id: 'technical', label: 'Technical' },
    { id: 'risk', label: 'Risk' },
  ];

  if (!node) {
    return (
      <div className="flex h-full items-center justify-center px-6">
        <div className="max-w-md text-center">
          <div className="text-sm font-semibold text-[var(--text-primary)]">No node selected</div>
          <p className="mt-2 text-sm leading-relaxed text-[var(--text-secondary)]">
            Perspective views work on one semantic address at a time. Pick a node first, then switch lenses instead of forcing more chat onto the same context.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="border-b border-[var(--border-subtle)] px-6 py-4">
        <div className="font-mono text-[11px] text-[var(--accent)]">{node.nodeRef}</div>
        <div className="mt-3 flex flex-wrap gap-2">
          {lenses.map((option) => (
            <button
              key={option.id}
              onClick={() => onLensChange(option.id)}
              className={`rounded-md border px-3 py-1.5 text-[11px] transition-colors ${
                lens === option.id
                  ? 'border-[var(--accent-dim)] bg-[var(--accent)]/8 text-[var(--text-primary)]'
                  : 'border-[var(--border)] text-[var(--text-secondary)] hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]'
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-auto px-6 py-5">
        {lens === 'structural' && (
          <div className="grid gap-5 xl:grid-cols-2">
            <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4">
              <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">What it is</div>
              <div className="mt-3 space-y-2 text-[13px] leading-relaxed text-[var(--text-secondary)]">
                <div>{node.description}</div>
                <div>Kind: <span className="text-[var(--text-primary)]">{node.kind}</span></div>
                <div>Layer: <span className="text-[var(--text-primary)]">{layerLabel(node.layer)}</span></div>
              </div>
            </section>
            <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4">
              <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Relationships</div>
              <div className="mt-3 space-y-2">
                {[...node.inbound, ...node.outbound].length === 0 && (
                  <div className="text-[12px] text-[var(--text-muted)]">No explicit structural links are compiled yet.</div>
                )}
                {[...node.inbound, ...node.outbound].map((link) => (
                  <RefButton key={`${lens}:${link.nodeRef}`} link={link} onOpen={onOpenNode} />
                ))}
              </div>
            </section>
          </div>
        )}

        {lens === 'behavioral' && (
          <div className="grid gap-5 xl:grid-cols-2">
            <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4">
              <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">What it does</div>
              <p className="mt-3 text-[13px] leading-relaxed text-[var(--text-secondary)]">{node.description}</p>
              {node.methods.length > 0 && (
                <div className="mt-4 space-y-2">
                  {node.methods.map((method) => (
                    <div key={method.name} className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2">
                      <div className="font-mono text-[11px] text-[var(--text-primary)]">
                        {method.name}({(method.parameters || []).map((parameter) => parameter.name).join(', ')})
                      </div>
                      {method.description && (
                        <div className="mt-1 text-[11px] text-[var(--text-muted)]">{method.description}</div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </section>
            <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4">
              <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">State and flow</div>
              {node.stateMachine ? (
                <div className="mt-3 space-y-2">
                  <div className="text-[12px] text-[var(--text-secondary)]">
                    States: <span className="text-[var(--text-primary)]">{node.stateMachine.states.join(', ')}</span>
                  </div>
                  {node.stateMachine.transitions.map((transition, index) => (
                    <div key={`${transition.from_state}-${transition.to_state}-${index}`} className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2 text-[11px] text-[var(--text-secondary)]">
                      {transition.from_state}
                      {' -> '}
                      {transition.to_state}
                      {' on '}
                      {transition.trigger}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="mt-3 text-[12px] text-[var(--text-muted)]">
                  No explicit state machine is compiled for this node yet.
                </div>
              )}
            </section>
          </div>
        )}

        {lens === 'business' && (
          <div className="grid gap-5 xl:grid-cols-2">
            <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4">
              <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Why it exists</div>
              <div className="mt-3 text-[13px] leading-relaxed text-[var(--text-secondary)]">
                {node.derivedFrom}
              </div>
            </section>
            <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4">
              <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">What it unlocks</div>
              <div className="mt-3 space-y-2">
                {node.readAfter.length === 0 && (
                  <div className="text-[12px] text-[var(--text-muted)]">
                    This node does not yet unlock a wider branch.
                  </div>
                )}
                {node.readAfter.map((link) => (
                  <RefButton key={`business:${link.nodeRef}`} link={link} onOpen={onOpenNode} />
                ))}
              </div>
            </section>
          </div>
        )}

        {lens === 'technical' && (
          <div className="grid gap-5 xl:grid-cols-2">
            <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4">
              <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Implementation hints</div>
              <div className="mt-3 space-y-2 text-[12px] leading-relaxed text-[var(--text-secondary)]">
                <div>Semantic shape: {node.kind}</div>
                <div>Coordinate: {node.postcode}</div>
                <div>Inputs/outputs should stay traceable to this semantic address.</div>
                {Object.keys(node.attributes).length > 0 && (
                  <div>Attributes: {Object.keys(node.attributes).join(', ')}</div>
                )}
              </div>
            </section>
            <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4">
              <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Rules and guards</div>
              <div className="mt-3 space-y-2">
                {node.validationRules.length === 0 && node.gaps.length === 0 && (
                  <div className="text-[12px] text-[var(--text-muted)]">No explicit rules are attached yet.</div>
                )}
                {node.validationRules.map((rule) => (
                  <div key={rule} className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2 text-[11px] text-[var(--text-secondary)]">
                    {rule}
                  </div>
                ))}
                {node.gaps.map((gap) => (
                  <div key={`tech-gap:${gap}`} className="rounded-lg border border-amber-500/20 bg-amber-500/8 px-3 py-2 text-[11px] text-amber-300">
                    {gap}
                  </div>
                ))}
              </div>
            </section>
          </div>
        )}

        {lens === 'risk' && (
          <div className="grid gap-5 xl:grid-cols-2">
            <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4">
              <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Risk surface</div>
              <div className="mt-3 space-y-2">
                {node.gaps.length === 0 && (
                  <div className="text-[12px] text-[var(--text-muted)]">No explicit node-local risks are recorded yet.</div>
                )}
                {node.gaps.map((gap) => (
                  <div key={`risk-gap:${gap}`} className="rounded-lg border border-amber-500/20 bg-amber-500/8 px-3 py-2 text-[11px] text-amber-300">
                    {gap}
                  </div>
                ))}
              </div>
            </section>
            <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4">
              <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Impact radius</div>
              <div className="mt-3 space-y-2">
                {node.outbound.length === 0 && (
                  <div className="text-[12px] text-[var(--text-muted)]">No downstream impact is compiled yet.</div>
                )}
                {node.outbound.map((link) => (
                  <RefButton key={`risk:${link.nodeRef}`} link={link} onOpen={onOpenNode} />
                ))}
              </div>
            </section>
          </div>
        )}
      </div>
    </div>
  );
}

function MapView({
  model,
  selectedLayer,
  onSelectLayer,
  onSelectNode,
}: {
  model: SemanticModel;
  selectedLayer: string | null;
  onSelectLayer: (layer: string) => void;
  onSelectNode: (nodeRef: string) => void;
}) {
  const layer = model.layerSummaries.find((summary) => summary.layer === selectedLayer)
    || model.layerSummaries[0]
    || null;

  return (
    <div className="h-full overflow-auto px-6 py-5">
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr),minmax(320px,0.9fr)]">
        <section className="rounded-2xl border border-[var(--border)] bg-[var(--bg-secondary)] p-5">
          <div className="flex items-center justify-between gap-4">
            <div>
              <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Map view</div>
              <h2 className="mt-2 text-xl font-semibold text-[var(--text-primary)]">Semantic terrain</h2>
            </div>
            <div className="text-right text-[11px] text-[var(--text-muted)]">
              <div>{model.nodes.length} nodes</div>
              <div>{model.gapTexts.length} gaps</div>
            </div>
          </div>
          <div className="mt-5 space-y-3">
            {model.layerSummaries.map((summary) => (
              <button
                key={summary.layer}
                onClick={() => onSelectLayer(summary.layer)}
                className={`w-full rounded-xl border px-4 py-3 text-left transition-colors ${
                  summary.layer === layer?.layer
                    ? 'border-[var(--accent-dim)] bg-[var(--accent)]/8'
                    : 'border-[var(--border)] bg-[var(--bg-primary)] hover:border-[var(--border-strong)]'
                }`}
              >
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <div className="font-mono text-[12px] text-[var(--text-primary)]">{summary.layer}</div>
                    <div className="text-[11px] text-[var(--text-muted)]">{summary.label}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-[12px] font-semibold text-[var(--text-primary)]">{summary.fillPercent}%</div>
                    <div className="text-[10px] text-[var(--text-muted)]">{summary.nodes.length} nodes</div>
                  </div>
                </div>
                <div className="mt-3 h-2 overflow-hidden rounded-full bg-[var(--bg-tertiary)]">
                  <div className="h-full rounded-full bg-[var(--accent)]" style={{ width: `${summary.fillPercent}%` }} />
                </div>
                <div className="mt-2 flex flex-wrap gap-3 text-[10px] text-[var(--text-muted)]">
                  <span>{summary.filled} filled</span>
                  <span>{summary.partial} partial</span>
                  <span>{summary.blocked} blocked</span>
                </div>
              </button>
            ))}
          </div>
        </section>

        <section className="rounded-2xl border border-[var(--border)] bg-[var(--bg-secondary)] p-5">
          <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Layer detail</div>
          {layer ? (
            <>
              <div className="mt-2 flex items-center justify-between gap-4">
                <div>
                  <div className="font-mono text-[14px] text-[var(--accent)]">{layer.layer}</div>
                  <div className="text-lg font-semibold text-[var(--text-primary)]">{layer.label}</div>
                </div>
                <div className="text-right text-[11px] text-[var(--text-muted)]">
                  <div>{layer.fillPercent}% filled</div>
                  <div>{layer.nodes.length} nodes</div>
                </div>
              </div>

              <div className="mt-4 space-y-2">
                {layer.nodes.map((node) => (
                  <button
                    key={node.nodeRef}
                    onClick={() => onSelectNode(node.nodeRef)}
                    className="w-full rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2 text-left transition-colors hover:border-[var(--accent-dim)]"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="min-w-0">
                        <div className="truncate text-[12px] font-medium text-[var(--text-primary)]">{node.name}</div>
                        <div className="truncate font-mono text-[10px] text-[var(--text-muted)]">{node.nodeRef}</div>
                      </div>
                      <FillBadge fillState={node.fillState} />
                    </div>
                  </button>
                ))}
              </div>

              {(layer.gaps.length > 0 || layer.silences.length > 0) && (
                <div className="mt-5 space-y-3">
                  {layer.gaps.length > 0 && (
                    <div>
                      <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Gaps</div>
                      <div className="mt-2 space-y-2">
                        {layer.gaps.map((gap) => (
                          <div key={gap} className="rounded-lg border border-amber-500/20 bg-amber-500/8 px-3 py-2 text-[11px] text-amber-300">
                            {gap}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  {layer.silences.length > 0 && (
                    <div>
                      <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Silences</div>
                      <div className="mt-2 space-y-2">
                        {layer.silences.map((silence) => (
                          <div key={silence} className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2 text-[11px] text-[var(--text-secondary)]">
                            {silence}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          ) : (
            <div className="mt-4 text-[12px] text-[var(--text-muted)]">No layer data yet.</div>
          )}
        </section>
      </div>
    </div>
  );
}

function TerminationPanel({
  termination,
}: {
  termination: TerminationViewModel;
}) {
  return (
    <div className={`rounded-xl border p-4 ${termination.tone}`}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-[10px] uppercase tracking-[0.2em] opacity-80">Termination condition</div>
          <div className="mt-2 text-base font-semibold text-[var(--text-primary)]">{termination.title}</div>
        </div>
        <span className={`rounded-full px-2 py-1 text-[10px] uppercase tracking-[0.18em] ${termination.badgeTone}`}>
          {termination.shortLabel}
        </span>
      </div>

      <div className="mt-3 space-y-2 text-[12px] leading-relaxed">
        <div>{termination.message}</div>
        <div>
          Next: <span className="text-[var(--text-primary)]">{termination.nextAction}</span>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-3 text-[10px] text-[var(--text-muted)]">
        <span>Status: <span className="font-mono text-[var(--text-primary)]">{termination.status}</span></span>
        <span>Reason: <span className="font-mono text-[var(--text-primary)]">{termination.reason}</span></span>
      </div>

      {termination.semanticProgress && (
        <div className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
          <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2">
            <div className="text-[10px] uppercase tracking-[0.16em] text-[var(--text-muted)]">Fingerprint</div>
            <div className="mt-1 text-[11px] text-[var(--text-primary)]">
              {termination.semanticProgress.fingerprint_changed ? 'changed' : 'unchanged'}
            </div>
          </div>
          <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2">
            <div className="text-[10px] uppercase tracking-[0.16em] text-[var(--text-muted)]">Verification delta</div>
            <div className="mt-1 text-[11px] text-[var(--text-primary)]">
              {typeof termination.semanticProgress.verification_score_delta === 'number'
                ? termination.semanticProgress.verification_score_delta.toFixed(3)
                : '--'}
            </div>
          </div>
          <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2">
            <div className="text-[10px] uppercase tracking-[0.16em] text-[var(--text-muted)]">Component delta</div>
            <div className="mt-1 text-[11px] text-[var(--text-primary)]">
              {typeof termination.semanticProgress.components_delta === 'number'
                ? termination.semanticProgress.components_delta
                : '--'}
            </div>
          </div>
          <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2">
            <div className="text-[10px] uppercase tracking-[0.16em] text-[var(--text-muted)]">Gate signature</div>
            <div className="mt-1 text-[11px] text-[var(--text-primary)]">
              {termination.semanticProgress.gates_changed ? 'changed' : 'unchanged'}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function LiveView({
  isCompiling,
  taskStatus,
  progress,
  result,
  parsedEntities,
  recentSignals,
}: {
  isCompiling: boolean;
  taskStatus: TaskStatus | null;
  progress: CompilationProgress | null;
  result: CompileResponse | null;
  parsedEntities: string[];
  recentSignals: string[];
}) {
  const termination = getTerminationViewModel(result?.termination_condition);
  const activeIndex = stageIndex(progress);
  const passedStages = result?.stage_results?.filter((stage) => stage.success).length
    ?? (isCompiling ? activeIndex : 0);
  const failedStages = result?.stage_results?.filter((stage) => !stage.success).length || 0;
  const liveCost = result?.usage?.cost_usd;
  const nodeCount = result?.blueprint?.components?.length || parsedEntities.length;

  return (
    <div className="h-full overflow-auto px-6 py-5">
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.1fr),minmax(340px,0.9fr)]">
        <section className="rounded-2xl border border-[var(--border)] bg-[var(--bg-secondary)] p-5">
          <div className="flex items-center justify-between gap-4">
            <div>
              <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Compilation live view</div>
              <h2 className="mt-2 text-xl font-semibold text-[var(--text-primary)]">
                {taskStatus === 'awaiting_decision'
                  ? 'Waiting for a human decision'
                  : isCompiling
                    ? 'Compiling semantic context'
                    : 'Last compilation run'}
              </h2>
              <p className="mt-2 max-w-2xl text-sm leading-relaxed text-[var(--text-secondary)]">
                {stageCopy(progress, taskStatus)}
              </p>
            </div>
            <div className="rounded-full border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-1.5 text-[11px] text-[var(--text-secondary)]">
              {isCompiling ? 'Live' : 'Snapshot'}
            </div>
          </div>

          {!isCompiling && termination && (
            <div className="mt-5">
              <TerminationPanel termination={termination} />
            </div>
          )}

          <div className="mt-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] px-4 py-3">
              <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">Nodes</div>
              <div className="mt-2 text-2xl font-semibold text-[var(--text-primary)]">{nodeCount}</div>
            </div>
            <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] px-4 py-3">
              <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">Gates passed</div>
              <div className="mt-2 text-2xl font-semibold text-[var(--text-primary)]">{passedStages}</div>
            </div>
            <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] px-4 py-3">
              <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">Gates failed</div>
              <div className="mt-2 text-2xl font-semibold text-[var(--text-primary)]">{failedStages}</div>
            </div>
            <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] px-4 py-3">
              <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">Cost</div>
              <div className="mt-2 text-2xl font-semibold text-[var(--text-primary)]">
                {typeof liveCost === 'number' ? `$${liveCost.toFixed(2)}` : '--'}
              </div>
            </div>
          </div>

          <div className="mt-6 grid gap-2 sm:grid-cols-2 xl:grid-cols-6">
            {STAGE_STEPS.map((step, index) => {
              const isActive = isCompiling && index === activeIndex;
              const isDone = !isCompiling ? index < passedStages : index < activeIndex;
              return (
                <div
                  key={step.id}
                  className={`rounded-xl border px-3 py-3 transition-colors ${
                    isActive
                      ? 'border-[var(--accent-dim)] bg-[var(--accent)]/8'
                      : isDone
                        ? 'border-emerald-500/20 bg-emerald-500/8'
                        : 'border-[var(--border)] bg-[var(--bg-primary)]'
                  }`}
                >
                  <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">{step.label}</div>
                  <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-[var(--bg-tertiary)]">
                    <div
                      className={`h-full rounded-full ${isActive ? 'bg-[var(--accent)]' : isDone ? 'bg-emerald-400' : 'bg-[var(--border)]'}`}
                      style={{ width: `${isDone ? 100 : isActive ? 65 : 20}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        <section className="rounded-2xl border border-[var(--border)] bg-[var(--bg-secondary)] p-5">
          <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Recent signals</div>
          <div className="mt-3 space-y-2">
            {recentSignals.length === 0 && (
              <div className="rounded-lg border border-dashed border-[var(--border)] px-3 py-3 text-[11px] text-[var(--text-muted)]">
                Signals will appear here as the compile progresses.
              </div>
            )}
            {recentSignals.map((signal, index) => (
              <div key={`${signal}-${index}`} className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2 text-[11px] leading-relaxed text-[var(--text-secondary)]">
                {signal}
              </div>
            ))}
          </div>

          {parsedEntities.length > 0 && (
            <div className="mt-5">
              <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Nodes appearing</div>
              <div className="mt-3 flex flex-wrap gap-2">
                {parsedEntities.map((entity) => (
                  <span key={entity} className="rounded-full border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-1.5 text-[10px] text-[var(--text-secondary)]">
                    {entity}
                  </span>
                ))}
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function GovernanceView({
  result,
  model,
  progress,
  taskStatus,
  taskId,
  selectedNode,
  onOpenNode,
  onContinueTask,
}: {
  result: CompileResponse | null;
  model: SemanticModel;
  progress: CompilationProgress | null;
  taskStatus: TaskStatus | null;
  taskId: string | null;
  selectedNode: SemanticNode | null;
  onOpenNode: (nodeRef: string) => void;
  onContinueTask: (taskId: string) => void;
}) {
  const [ledgerProgress, setLedgerProgress] = useState<CompilationProgress | null>(progress);
  const [decisionQuestion, setDecisionQuestion] = useState('');
  const [decisionAnswer, setDecisionAnswer] = useState('');
  const [decisionStatus, setDecisionStatus] = useState<string | null>(null);
  const [savingDecision, setSavingDecision] = useState(false);
  const [decisionTermination, setDecisionTermination] = useState<TerminationViewModel | null>(null);

  useEffect(() => {
    setLedgerProgress(progress);
  }, [progress]);

  const report = result?.governance_report;
  const termination = getTerminationViewModel(result?.termination_condition);
  const effectiveTermination = decisionTermination || termination;
  const counts = summarizeFillStates(model);
  const depth = inferDepthSnapshot(result, model);
  const coverage = report?.coverage ?? (model.nodes.length > 0
    ? Math.round(((counts.filled + counts.partial * 0.6) / model.nodes.length) * 100)
    : 0);
  const verificationEntries = flattenVerificationEntries(result?.verification);
  const stageResults = result?.stage_results || [];
  const humanDecisions = mergeLedgerItems(
    report?.human_decisions || [],
    ledgerProgress?.human_decisions || [],
  );
  const escalations = mergeLedgerItems(
    report?.escalated || [],
    ledgerProgress?.escalations || [],
  );
  const depthLabel = report?.compilation_depth?.label || depth.label;
  const antiGoalsChecked = report?.anti_goals_checked ?? 0;
  const primaryEscalation = escalations[0] || null;
  const awaitingDecision = taskStatus === 'awaiting_decision' && !decisionTermination;
  const activeDecisionPostcode = selectedNode?.postcode || primaryEscalation?.postcode || null;
  const activeDecisionQuestion = decisionQuestion.trim() || primaryEscalation?.question || '';

  useEffect(() => {
    if (awaitingDecision && primaryEscalation && !decisionQuestion.trim()) {
      setDecisionQuestion(primaryEscalation.question);
    }
  }, [awaitingDecision, decisionQuestion, primaryEscalation]);

  useEffect(() => {
    if (taskStatus !== 'awaiting_decision') {
      setDecisionTermination(null);
    }
  }, [taskStatus]);

  const handleRecordDecision = async () => {
    if (!taskId || !activeDecisionPostcode || !activeDecisionQuestion || !decisionAnswer.trim()) {
      return;
    }

    setSavingDecision(true);
    setDecisionStatus(null);
    try {
      const response = await recordTaskDecision(taskId, {
        postcode: activeDecisionPostcode,
        question: activeDecisionQuestion,
        answer: decisionAnswer.trim(),
      });
      setLedgerProgress(response.progress ?? ledgerProgress);
      setDecisionQuestion('');
      setDecisionAnswer('');
      if (response.next_task_id) {
        setDecisionStatus('Decision saved. Continuing compile...');
        onContinueTask(response.next_task_id);
        return;
      }
      if (response.termination_condition) {
        const guardedTermination = getTerminationViewModel(response.termination_condition);
        setDecisionTermination(guardedTermination);
        setDecisionStatus(guardedTermination?.message || 'Decision saved. Motherlabs stopped this chain to prevent a continuation loop.');
        return;
      }
      setDecisionStatus('Decision saved to task ledger.');
    } catch {
      setDecisionStatus('Decision save failed.');
    } finally {
      setSavingDecision(false);
    }
  };

  return (
    <div className="h-full overflow-auto px-6 py-5">
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.1fr),minmax(340px,0.9fr)]">
        <section className="space-y-6">
          <div className="rounded-2xl border border-[var(--border)] bg-[var(--bg-secondary)] p-5">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Governance report</div>
                <h2 className="mt-2 text-xl font-semibold text-[var(--text-primary)]">What was checked and what is still open</h2>
              </div>
              <div className="rounded-full border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-1.5 text-[11px] text-[var(--text-secondary)]">
                {depthLabel} depth
              </div>
            </div>

            {effectiveTermination && (
              <div className="mt-5">
                <TerminationPanel termination={effectiveTermination} />
              </div>
            )}

            <div className="mt-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] px-4 py-3">
                <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">Coverage</div>
                <div className="mt-2 text-2xl font-semibold text-[var(--text-primary)]">{coverage}%</div>
              </div>
              <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] px-4 py-3">
                <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">Promoted</div>
                <div className="mt-2 text-2xl font-semibold text-[var(--text-primary)]">{counts.filled}</div>
              </div>
              <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] px-4 py-3">
                <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">Partial</div>
                <div className="mt-2 text-2xl font-semibold text-[var(--text-primary)]">{counts.partial}</div>
              </div>
              <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] px-4 py-3">
                <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">Blocked</div>
                <div className="mt-2 text-2xl font-semibold text-[var(--text-primary)]">{counts.blocked}</div>
              </div>
            </div>

            <div className="mt-5 flex flex-wrap gap-3 text-[11px] text-[var(--text-muted)]">
              {result?.trust?.overall_score != null && <span>{result.trust.overall_score.toFixed(0)}% trust</span>}
              {result?.trust?.verification_badge && <span>{result.trust.verification_badge}</span>}
              {result?.usage && <span>${result.usage.cost_usd.toFixed(4)} spent</span>}
              <span>{Math.round(depth.score * 100)} depth score</span>
              {antiGoalsChecked > 0 && <span>{antiGoalsChecked} anti-goals checked</span>}
            </div>
          </div>

          <div className="rounded-2xl border border-[var(--border)] bg-[var(--bg-secondary)] p-5">
            <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Stage checks</div>
            <div className="mt-4 space-y-2">
              {stageResults.length === 0 && (
                <div className="rounded-lg border border-dashed border-[var(--border)] px-3 py-3 text-[11px] text-[var(--text-muted)]">
                  Stage-level governance is not attached to this run yet.
                </div>
              )}
              {stageResults.map((stage) => (
                <div key={stage.stage} className="rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] px-4 py-3">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--text-primary)]">{stage.stage}</div>
                      <div className="mt-1 text-[11px] text-[var(--text-muted)]">
                        {stage.success ? 'Passed' : 'Needs work'}{stage.retries > 0 ? ` • ${stage.retries} retr${stage.retries === 1 ? 'y' : 'ies'}` : ''}
                      </div>
                    </div>
                    <span className={`rounded-full px-2 py-1 text-[10px] uppercase tracking-[0.18em] ${
                      stage.success ? 'bg-emerald-500/12 text-emerald-300' : 'bg-amber-500/12 text-amber-300'
                    }`}>
                      {stage.success ? 'pass' : 'open'}
                    </span>
                  </div>
                  {(stage.errors.length > 0 || stage.warnings.length > 0) && (
                    <div className="mt-3 space-y-2 text-[11px] text-[var(--text-secondary)]">
                      {stage.errors.map((error) => (
                        <div key={`${stage.stage}:error:${error}`} className="rounded-lg border border-red-500/20 bg-red-500/8 px-3 py-2 text-red-200">
                          {error}
                        </div>
                      ))}
                      {stage.warnings.map((warning) => (
                        <div key={`${stage.stage}:warning:${warning}`} className="rounded-lg border border-amber-500/20 bg-amber-500/8 px-3 py-2 text-amber-300">
                          {warning}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="space-y-6">
          {awaitingDecision && primaryEscalation && (
            <div className="rounded-2xl border border-amber-500/20 bg-amber-500/8 p-5">
              <div className="text-[10px] uppercase tracking-[0.2em] text-amber-200">Human in the loop</div>
              <h3 className="mt-2 text-lg font-semibold text-[var(--text-primary)]">A decision is blocking this compile</h3>
              <div className="mt-3 text-[12px] leading-relaxed text-amber-100">{primaryEscalation.question}</div>
              {Array.isArray(primaryEscalation.options) && primaryEscalation.options.length > 0 && (
                <div className="mt-4 flex flex-wrap gap-2">
                  {primaryEscalation.options.map((option) => (
                    <button
                      key={option}
                      onClick={() => setDecisionAnswer(option)}
                      className={`rounded-full border px-3 py-1.5 text-[11px] transition-colors ${
                        decisionAnswer === option
                          ? 'border-[var(--accent-dim)] bg-[var(--accent)]/10 text-[var(--text-primary)]'
                          : 'border-amber-400/20 bg-[var(--bg-primary)] text-amber-100 hover:border-amber-300/40'
                      }`}
                    >
                      {option}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          <div className="rounded-2xl border border-[var(--border)] bg-[var(--bg-secondary)] p-5">
            <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Verification surface</div>
            <div className="mt-4 space-y-2">
              {verificationEntries.length === 0 && (
                <div className="rounded-lg border border-dashed border-[var(--border)] px-3 py-3 text-[11px] text-[var(--text-muted)]">
                  No structured verification entries were attached to this compile.
                </div>
              )}
              {verificationEntries.map((entry) => (
                <div key={entry.label} className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2">
                  <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-[var(--accent)]">{entry.label}</div>
                  <div className="mt-1 text-[11px] leading-relaxed text-[var(--text-secondary)]">{entry.detail}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-2xl border border-[var(--border)] bg-[var(--bg-secondary)] p-5">
            <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Escalations</div>
            <div className="mt-4 space-y-2">
              {escalations.length === 0 && model.gapTexts.length === 0 && (
                <div className="rounded-lg border border-dashed border-[var(--border)] px-3 py-3 text-[11px] text-[var(--text-muted)]">
                  No escalations are blocking this blueprint right now.
                </div>
              )}
              {(escalations.length > 0 ? escalations.map((entry) => entry.question) : model.gapTexts).map((gap) => {
                const sourceEscalation = escalations.find((entry) => entry.question === gap);
                const sourcePostcode = sourceEscalation?.postcode;
                const nodeRef = sourceEscalation?.node_ref
                  || (sourcePostcode
                    ? model.nodes.find((node) => node.postcode === sourcePostcode)?.nodeRef || findNodeRefForText(model, gap)
                    : findNodeRefForText(model, gap));
                if (nodeRef) {
                  return (
                    <button
                      key={gap}
                      onClick={() => onOpenNode(nodeRef)}
                      className="w-full rounded-lg border border-amber-500/20 bg-amber-500/8 px-3 py-2 text-left text-[11px] leading-relaxed text-amber-300 transition-colors hover:border-amber-400/40"
                    >
                      {gap}
                    </button>
                  );
                }

                return (
                  <div key={gap} className="rounded-lg border border-amber-500/20 bg-amber-500/8 px-3 py-2 text-[11px] leading-relaxed text-amber-300">
                    {gap}
                  </div>
                );
              })}
            </div>
          </div>

          <div className="rounded-2xl border border-[var(--border)] bg-[var(--bg-secondary)] p-5">
            <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Human decisions</div>
            <div className="mt-4 space-y-2">
              {humanDecisions.length === 0 && (
                <div className="rounded-lg border border-dashed border-[var(--border)] px-3 py-3 text-[11px] leading-relaxed text-[var(--text-muted)]">
                  No human decisions were recorded in this compile payload.
                </div>
              )}
              {humanDecisions.map((decision) => {
                const nodeRef = model.nodes.find((node) => node.postcode === decision.postcode)?.nodeRef || null;
                const content = (
                  <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2 text-left">
                    <div className="font-mono text-[10px] text-[var(--accent)]">{decision.postcode}</div>
                    <div className="mt-1 text-[11px] leading-relaxed text-[var(--text-secondary)]">{decision.question}</div>
                    <div className="mt-2 text-[11px] text-[var(--text-primary)]">Answer: {decision.answer}</div>
                    {decision.timestamp && (
                      <div className="mt-1 text-[10px] text-[var(--text-muted)]">{decision.timestamp}</div>
                    )}
                  </div>
                );

                if (!nodeRef) {
                  return <div key={`${decision.postcode}:${decision.timestamp}`}>{content}</div>;
                }

                return (
                  <button
                    key={`${decision.postcode}:${decision.timestamp}`}
                    onClick={() => onOpenNode(nodeRef)}
                    className="w-full transition-opacity hover:opacity-100"
                  >
                    {content}
                  </button>
                );
              })}
            </div>

            {taskId && activeDecisionPostcode && (
              <div className="mt-4 rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] p-4">
                <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">Record decision</div>
                <div className="mt-2 font-mono text-[10px] text-[var(--accent)]">{activeDecisionPostcode}</div>
                <div className="mt-3 space-y-3">
                  <input
                    value={decisionQuestion}
                    onChange={(event) => setDecisionQuestion(event.target.value)}
                    placeholder="Decision question"
                    className="w-full rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)] px-3 py-2 text-[12px] text-[var(--text-primary)] outline-none transition-colors placeholder:text-[var(--text-muted)] focus:border-[var(--accent-dim)]"
                  />
                  <textarea
                    value={decisionAnswer}
                    onChange={(event) => setDecisionAnswer(event.target.value)}
                    placeholder="Decision answer"
                    rows={3}
                    className="w-full resize-none rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)] px-3 py-2 text-[12px] leading-relaxed text-[var(--text-primary)] outline-none transition-colors placeholder:text-[var(--text-muted)] focus:border-[var(--accent-dim)]"
                  />
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-[10px] text-[var(--text-muted)]">
                      {awaitingDecision
                        ? 'This answer will continue the paused compile.'
                        : 'Attach the decision to the selected semantic node.'}
                    </div>
                    <button
                      onClick={handleRecordDecision}
                      disabled={savingDecision || !activeDecisionQuestion || !decisionAnswer.trim()}
                      className="rounded-md bg-[var(--accent)] px-3 py-1.5 text-[11px] font-semibold text-[var(--bg-primary)] disabled:cursor-not-allowed disabled:opacity-35"
                    >
                      {savingDecision ? 'Saving…' : awaitingDecision ? 'Continue compile' : 'Save decision'}
                    </button>
                  </div>
                  {decisionStatus && (
                    <div className="text-[11px] text-[var(--text-muted)]">{decisionStatus}</div>
                  )}
                </div>
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

function ExportView({
  result,
  model,
  taskId,
}: {
  result: CompileResponse | null;
  model: SemanticModel;
  taskId: string | null;
}) {
  const counts = summarizeFillStates(model);
  const depth = inferDepthSnapshot(result, model);
  const [status, setStatus] = useState<string | null>(null);
  const [zipping, setZipping] = useState(false);

  const files = useMemo(() => buildExportFiles(result, model), [result, model]);
  const includeStateMachines = model.nodes.filter((node) => node.stateMachine).length;
  const includeProcesses = model.nodes.filter((node) => node.kind === 'process').length;
  const endpoint = taskId ? `/api/v2/tasks/${taskId}` : null;

  const handleDownloadZip = async () => {
    if (files.length === 0) {
      return;
    }

    setZipping(true);
    try {
      const { downloadZip: createZip } = await import('client-zip');
      const blob = await createZip(files).blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${slugify(model.projectLabel || 'motherlabs_blueprint')}.zip`;
      link.click();
      URL.revokeObjectURL(url);
      setStatus('Blueprint zip downloaded.');
    } catch {
      setStatus('Zip export failed.');
    } finally {
      setZipping(false);
    }
  };

  const handleCopyJson = async () => {
    if (!result) {
      return;
    }
    try {
      await navigator.clipboard.writeText(JSON.stringify(result, null, 2));
      setStatus('Blueprint JSON copied.');
    } catch {
      setStatus('Copy failed.');
    }
  };

  const handleCopyEndpoint = async () => {
    if (!endpoint) {
      return;
    }
    try {
      await navigator.clipboard.writeText(endpoint);
      setStatus('Task endpoint copied.');
    } catch {
      setStatus('Copy failed.');
    }
  };

  return (
    <div className="h-full overflow-auto px-6 py-5">
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.05fr),minmax(340px,0.95fr)]">
        <section className="rounded-2xl border border-[var(--border)] bg-[var(--bg-secondary)] p-5">
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Export preview</div>
              <h2 className="mt-2 text-xl font-semibold text-[var(--text-primary)]">Ready to hand off to a coding agent</h2>
              <p className="mt-2 max-w-2xl text-sm leading-relaxed text-[var(--text-secondary)]">
                Motherlabs stays no-code. Export is the renderer handoff: compiled context, provenance, verification, and output packs in one bundle.
              </p>
            </div>
            <div className="rounded-full border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-1.5 text-[11px] text-[var(--text-secondary)]">
              {files.length} files
            </div>
          </div>

          <div className="mt-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] px-4 py-3">
              <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">Nodes</div>
              <div className="mt-2 text-2xl font-semibold text-[var(--text-primary)]">{model.nodes.length}</div>
            </div>
            <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] px-4 py-3">
              <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">Promoted</div>
              <div className="mt-2 text-2xl font-semibold text-[var(--text-primary)]">{counts.filled}</div>
            </div>
            <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] px-4 py-3">
              <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">Depth</div>
              <div className="mt-2 text-2xl font-semibold text-[var(--text-primary)]">{depth.label}</div>
            </div>
            <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] px-4 py-3">
              <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">Cost</div>
              <div className="mt-2 text-2xl font-semibold text-[var(--text-primary)]">
                {result?.usage ? `$${result.usage.cost_usd.toFixed(2)}` : '--'}
              </div>
            </div>
          </div>

          <div className="mt-5 rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] p-4">
            <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">Includes</div>
            <div className="mt-3 grid gap-2 text-[12px] text-[var(--text-secondary)]">
              <div>✓ {model.nodes.length} semantic nodes with postcode/name addresses</div>
              <div>✓ {result?.blueprint?.relationships?.length || 0} relationships and {result?.blueprint?.constraints?.length || 0} constraints</div>
              <div>✓ {includeProcesses} process surfaces and {includeStateMachines} state machine{includeStateMachines === 1 ? '' : 's'}</div>
              <div>✓ Governance + trust reports{result?.verification ? ' + verification payload' : ''}</div>
              <div>✓ Stage results and structured compile signals</div>
              <div>✓ Canonical renderer manifest and handoff instructions</div>
              {model.gapTexts.length > 0 && (
                <div>✓ {model.gapTexts.length} open gap{model.gapTexts.length === 1 ? '' : 's'} preserved in the bundle</div>
              )}
            </div>
          </div>

          <div className="mt-5 flex flex-wrap gap-2">
            <button
              onClick={handleDownloadZip}
              disabled={files.length === 0 || zipping}
              className="rounded-md bg-[var(--accent)] px-3 py-1.5 text-[11px] font-semibold text-[var(--bg-primary)] disabled:cursor-not-allowed disabled:opacity-35"
            >
              {zipping ? 'Building zip…' : 'Download zip'}
            </button>
            <button
              onClick={handleCopyJson}
              disabled={!result}
              className="rounded-md border border-[var(--border)] px-3 py-1.5 text-[11px] text-[var(--text-secondary)] transition-colors hover:border-[var(--accent-dim)] hover:text-[var(--text-primary)] disabled:cursor-not-allowed disabled:opacity-35"
            >
              Copy JSON
            </button>
            <button
              onClick={handleCopyEndpoint}
              disabled={!endpoint}
              className="rounded-md border border-[var(--border)] px-3 py-1.5 text-[11px] text-[var(--text-secondary)] transition-colors hover:border-[var(--accent-dim)] hover:text-[var(--text-primary)] disabled:cursor-not-allowed disabled:opacity-35"
            >
              Copy API link
            </button>
          </div>

          {status && (
            <div className="mt-3 text-[11px] text-[var(--text-muted)]">{status}</div>
          )}
        </section>

        <section className="rounded-2xl border border-[var(--border)] bg-[var(--bg-secondary)] p-5">
          <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Open pressure before export</div>
          <div className="mt-4 space-y-2">
            {model.gapTexts.length === 0 && (
              <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2 text-[11px] text-emerald-300">
                No explicit gaps remain. The bundle is ready for renderer handoff.
              </div>
            )}
            {model.gapTexts.map((gap) => (
              <div key={gap} className="rounded-lg border border-amber-500/20 bg-amber-500/8 px-3 py-2 text-[11px] leading-relaxed text-amber-300">
                {gap}
              </div>
            ))}
          </div>

          <div className="mt-5 rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] p-4">
            <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">Handoff notes</div>
            <div className="mt-3 space-y-2 text-[12px] leading-relaxed text-[var(--text-secondary)]">
              <div>The coding agent is a renderer. It should follow the bundle, not redesign the blueprint.</div>
              <div>Use postcode/name citations inside prompts or task notes so the builder stays on semantic coordinates.</div>
              {endpoint && <div>API endpoint: <span className="font-mono text-[var(--accent)]">{endpoint}</span></div>}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

function DoraPanel({
  node,
  model,
  result,
  messages,
  input,
  onInputChange,
  onSubmit,
  onSelectNode,
}: {
  node: SemanticNode | null;
  model: SemanticModel;
  result: CompileResponse | null;
  messages: DoraMessage[];
  input: string;
  onInputChange: (value: string) => void;
  onSubmit: (question: string) => void;
  onSelectNode: (nodeRef: string) => void;
}) {
  const questions = buildQuestionSet(node, model, result);

  return (
    <aside className="border-t border-[var(--border-subtle)] bg-[var(--bg-secondary)] lg:border-l lg:border-t-0">
      <div className="border-b border-[var(--border-subtle)] px-4 py-4">
        <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Dora</div>
        <div className="mt-2 text-lg font-semibold text-[var(--text-primary)]">Deep reading layer</div>
        <p className="mt-2 text-[12px] leading-relaxed text-[var(--text-secondary)]">
          Browse first. Stop here when you need to think, compare, or understand why the blueprint is shaped this way.
        </p>
      </div>

      <div className="border-b border-[var(--border-subtle)] px-4 py-4">
        <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Precompiled questions</div>
        <div className="mt-3 flex flex-wrap gap-2">
          {questions.map((question) => (
            <button
              key={question}
              onClick={() => onSubmit(question)}
              className="rounded-full border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-1.5 text-[11px] text-[var(--text-secondary)] transition-colors hover:border-[var(--accent-dim)] hover:text-[var(--text-primary)]"
            >
              {question}
            </button>
          ))}
        </div>
      </div>

      <div className="max-h-[38vh] overflow-auto px-4 py-4 lg:max-h-none lg:flex-1">
        <div className="space-y-3">
          {messages.length === 0 && (
            <div className="rounded-xl border border-dashed border-[var(--border)] px-4 py-4 text-[12px] leading-relaxed text-[var(--text-muted)]">
              Select a node, then ask Dora why it exists, what depends on it, or where the blueprint is still thin.
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className="space-y-2">
              <div className={`rounded-xl px-4 py-3 text-[12px] leading-relaxed ${
                message.role === 'user'
                  ? 'bg-[var(--bg-primary)] text-[var(--text-primary)]'
                  : 'border border-[var(--border)] bg-[var(--bg-primary)] text-[var(--text-secondary)]'
              }`}>
                {message.text}
              </div>
              {message.refs.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {message.refs.map((ref) => (
                    <button
                      key={`${message.id}:${ref}`}
                      onClick={() => onSelectNode(ref)}
                      className="rounded-md border border-[var(--border)] bg-[var(--bg-primary)] px-2 py-1 font-mono text-[10px] text-[var(--accent)] transition-colors hover:border-[var(--accent-dim)]"
                    >
                      {ref}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="border-t border-[var(--border-subtle)] px-4 py-4">
        <textarea
          value={input}
          onChange={(event) => onInputChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault();
              if (input.trim()) {
                onSubmit(input.trim());
              }
            }
          }}
          placeholder={node ? `Ask Dora about ${node.name}...` : 'Ask Dora about the blueprint...'}
          rows={3}
          className="w-full resize-none rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-3 text-[12px] leading-relaxed text-[var(--text-primary)] outline-none transition-colors placeholder:text-[var(--text-muted)] focus:border-[var(--accent-dim)]"
        />
        <div className="mt-3 flex items-center justify-between gap-4">
          <div className="text-[10px] text-[var(--text-muted)]">Position-aware reading companion</div>
          <button
            onClick={() => input.trim() && onSubmit(input.trim())}
            disabled={!input.trim()}
            className="rounded-md bg-[var(--accent)] px-3 py-1.5 text-[11px] font-semibold text-[var(--bg-primary)] disabled:cursor-not-allowed disabled:opacity-35"
          >
            Ask Dora
          </button>
        </div>
      </div>
    </aside>
  );
}

function ViewTabs({
  activeView,
  onChange,
}: {
  activeView: WorkbenchView;
  onChange: (view: WorkbenchView) => void;
}) {
  const tabs: { id: WorkbenchView; label: string }[] = [
    { id: 'node', label: 'Node Card' },
    { id: 'perspectives', label: 'Perspectives' },
    { id: 'map', label: 'Map' },
    { id: 'live', label: 'Live' },
    { id: 'governance', label: 'Governance' },
    { id: 'export', label: 'Export' },
  ];

  return (
    <div className="flex flex-wrap gap-2">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={`rounded-md border px-3 py-1.5 text-[11px] transition-colors ${
            activeView === tab.id
              ? 'border-[var(--accent-dim)] bg-[var(--accent)]/8 text-[var(--text-primary)]'
              : 'border-[var(--border)] text-[var(--text-secondary)] hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]'
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

function StatusStrip({
  result,
  model,
  isCompiling,
  progress,
  taskStatus,
}: {
  result: CompileResponse | null;
  model: SemanticModel;
  isCompiling: boolean;
  progress: CompilationProgress | null;
  taskStatus: TaskStatus | null;
}) {
  const termination = getTerminationViewModel(result?.termination_condition);

  return (
    <div className="border-t border-[var(--border-subtle)] bg-[var(--bg-secondary)] px-4 py-2">
      <div className="flex flex-wrap items-center justify-between gap-3 text-[10px] text-[var(--text-muted)]">
        <div className="flex flex-wrap items-center gap-3">
          <span>{isCompiling || taskStatus === 'awaiting_decision' ? stageCopy(progress, taskStatus) : 'Ready to read and deepen'}</span>
          {!isCompiling && termination && <span>{termination.shortLabel}</span>}
          <span>{model.nodes.length} nodes</span>
          <span>{model.gapTexts.length} gaps</span>
          <span>{model.silenceTexts.length} silences</span>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {result?.trust?.overall_score != null && <span>{result.trust.overall_score.toFixed(0)}% trust</span>}
          {result?.usage && <span>${result.usage.cost_usd.toFixed(4)}</span>}
          {result?.domain && <span className="uppercase">{result.domain}</span>}
        </div>
      </div>
    </div>
  );
}

function WorkbenchShell() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const initialTaskId = searchParams.get('id');

  const { phase, taskId, result: compileResult, pollData, submit, cancel, reset } = useCompile();
  const [loadedResult, setLoadedResult] = useState<CompileResponse | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loadStatus, setLoadStatus] = useState<TaskStatus | null>(null);
  const [loadingTask, setLoadingTask] = useState(false);
  const [loadProgress, setLoadProgress] = useState<CompilationProgress | null>(null);
  const [recentSignals, setRecentSignals] = useState<string[]>([]);

  const parsedProgress = useInsightParser(pollData);
  const progress = pollData?.progress ?? loadProgress ?? null;
  const isCompiling = phase === 'submitting' || phase === 'queued' || phase === 'polling' || loadingTask;
  const result = compileResult || loadedResult;
  const model = useMemo(() => buildSemanticModel(result), [result]);
  const currentTaskId = taskId || initialTaskId;
  const taskStatus = pollData?.status ?? loadStatus ?? null;
  const termination = useMemo(() => getTerminationViewModel(result?.termination_condition), [result]);

  const [activeView, setActiveView] = useState<WorkbenchView>('map');
  const [activeLens, setActiveLens] = useState<PerspectiveLens>('structural');
  const [selectedNodeRef, setSelectedNodeRef] = useState<string | null>(null);
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null);
  const [doraInput, setDoraInput] = useState('');
  const [doraMessages, setDoraMessages] = useState<DoraMessage[]>([]);
  const doraSeeded = useRef(false);

  const selectedNode = selectedNodeRef ? model.nodeByRef.get(selectedNodeRef) || null : null;

  useEffect(() => {
    if (isCompiling) {
      setActiveView('live');
    }
  }, [isCompiling]);

  useEffect(() => {
    if (taskStatus === 'awaiting_decision') {
      setActiveView('governance');
    }
  }, [taskStatus]);

  useEffect(() => {
    if (termination && termination.status !== 'complete') {
      setActiveView('governance');
    }
  }, [termination]);

  useEffect(() => {
    if (!selectedNodeRef && model.nodes.length > 0) {
      setSelectedNodeRef(model.nodes[0].nodeRef);
    }
  }, [model.nodes, selectedNodeRef]);

  useEffect(() => {
    if (!selectedLayer && model.layerSummaries.length > 0) {
      setSelectedLayer(model.layerSummaries[0].layer);
    }
  }, [model.layerSummaries, selectedLayer]);

  useEffect(() => {
    if (selectedNode && selectedNode.layer !== selectedLayer) {
      setSelectedLayer(selectedNode.layer);
    }
  }, [selectedNode, selectedLayer]);

  useEffect(() => {
    if (!result || doraSeeded.current) {
      return;
    }
    doraSeeded.current = true;
    const greeting = buildDoraAnswer('Summarize this blueprint.', model.nodes[0] || null, model, result);
    setDoraMessages([{
      id: 'dora-greeting',
      role: 'dora',
      text: greeting.text,
      refs: greeting.refs,
    }]);
  }, [model, result]);

  useEffect(() => {
    if (!isCompiling) {
      return;
    }
    const latest = parsedProgress.latestInsight;
    if (!latest) {
      return;
    }
    setRecentSignals((previous) => {
      if (previous[previous.length - 1] === latest) {
        return previous;
      }
      return [...previous, latest].slice(-8);
    });
  }, [isCompiling, parsedProgress.latestInsight]);

  useEffect(() => {
    if (phase === 'complete' && taskId) {
      router.replace(`/workbench?id=${taskId}`, { scroll: false });
    }
  }, [phase, router, taskId]);

  useEffect(() => {
    if (!initialTaskId || compileResult) {
      return;
    }

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    setLoadingTask(true);

    const poll = () => {
      getTaskStatus(initialTaskId)
        .then((task: TaskStatusResponse) => {
          if (cancelled) {
            return;
          }
          setLoadStatus(task.status);
          if (task.status === 'complete' && task.result) {
            setLoadedResult(task.result);
            setLoadingTask(false);
            setLoadProgress(task.progress ?? null);
            return;
          }
          if (task.status === 'awaiting_decision') {
            setLoadedResult(task.result ?? null);
            setLoadingTask(false);
            setLoadProgress(task.progress ?? null);
            return;
          }
          if (task.status === 'error') {
            setLoadError(task.error || 'Failed to load task.');
            setLoadingTask(false);
            return;
          }
          if (task.progress) {
            setLoadProgress(task.progress);
            const progressSignals = task.progress.insights || [];
            setRecentSignals((previous) => {
              const merged = [...previous];
              for (const signal of progressSignals) {
                if (!merged.includes(signal)) {
                  merged.push(signal);
                }
              }
              return merged.slice(-8);
            });
          }
          timer = setTimeout(poll, 2000);
        })
        .catch((error: Error) => {
          if (!cancelled) {
            setLoadError(error.message);
            setLoadingTask(false);
          }
        });
    };

    poll();

    return () => {
      cancelled = true;
      if (timer) {
        clearTimeout(timer);
      }
    };
  }, [compileResult, initialTaskId]);

  const handleCompile = (description: string) => {
    sessionStorage.setItem('last_description', description);
    setRecentSignals([]);
    setLoadStatus(null);
    doraSeeded.current = false;
    submit({ description, domain: 'software' });
  };

  const handleSelectNode = (nodeRef: string) => {
    setSelectedNodeRef(nodeRef);
    setActiveView('node');
  };

  const handleContinueTask = (nextTaskId: string) => {
    reset();
    setLoadedResult(null);
    setLoadProgress(null);
    setLoadStatus(null);
    setRecentSignals([]);
    router.replace(`/workbench?id=${nextTaskId}`, { scroll: false });
  };

  const handleAskDora = (question: string) => {
    const trimmed = question.trim();
    if (!trimmed) {
      return;
    }

    if (!result && !isCompiling) {
      handleCompile(trimmed);
      return;
    }

    const answer = buildDoraAnswer(trimmed, selectedNode, model, result);
    setDoraMessages((previous) => [
      ...previous,
      { id: `user-${previous.length}`, role: 'user', text: trimmed, refs: [] },
      { id: `dora-${previous.length}`, role: 'dora', text: answer.text, refs: answer.refs },
    ]);
    setDoraInput('');
  };

  const handleDeepCompile = (node: SemanticNode) => {
    handleCompile(buildDeepCompilePrompt(node));
  };

  if (loadError && !result) {
    return (
      <div className="flex h-full items-center justify-center bg-[var(--bg-primary)] px-6">
        <div className="max-w-md rounded-2xl border border-red-500/20 bg-red-500/8 px-6 py-5 text-center">
          <div className="text-sm font-semibold text-red-300">Failed to load workbench</div>
          <div className="mt-2 text-sm leading-relaxed text-red-200/80">{loadError}</div>
          <a
            href="/workbench"
            className="mt-4 inline-flex rounded-md border border-red-400/20 px-3 py-1.5 text-[12px] text-red-200 transition-colors hover:border-red-300/40"
          >
            Open empty workspace
          </a>
        </div>
      </div>
    );
  }

  if (!result && !isCompiling && !initialTaskId) {
    return <Composer onCompile={handleCompile} />;
  }

  return (
    <div className="flex h-full flex-col overflow-hidden bg-[var(--bg-primary)]">
      <header className="border-b border-[var(--border-subtle)] bg-[var(--bg-secondary)] px-4 py-4">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--accent)]">Motherlabs Workbench</div>
            <div className="mt-2 text-xl font-semibold text-[var(--text-primary)]">{model.projectLabel}</div>
            <div className="mt-2 max-w-3xl text-sm leading-relaxed text-[var(--text-secondary)]">
              Browse the semantic terrain, open a node, switch lenses, then use Dora when you need deeper reasoning.
            </div>
          </div>
          <ViewTabs activeView={activeView} onChange={setActiveView} />
        </div>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 overflow-hidden lg:grid-cols-[300px,minmax(0,1fr),380px]">
        <TerrainSidebar
          model={model}
          selectedLayer={selectedLayer}
          selectedNodeRef={selectedNodeRef}
          onSelectLayer={setSelectedLayer}
          onSelectNode={handleSelectNode}
          onOpenMap={() => setActiveView('map')}
        />

        <main className="min-h-0 overflow-hidden border-b border-[var(--border-subtle)] lg:border-b-0">
          {activeView === 'node' && (
            <NodeCardView
              node={selectedNode}
              inputText={result?.input_text}
              onOpenNode={handleSelectNode}
              onDeepCompile={handleDeepCompile}
              onAskDora={handleAskDora}
            />
          )}
          {activeView === 'perspectives' && (
            <PerspectiveView
              node={selectedNode}
              lens={activeLens}
              onLensChange={setActiveLens}
              onOpenNode={handleSelectNode}
            />
          )}
          {activeView === 'map' && (
            <MapView
              model={model}
              selectedLayer={selectedLayer}
              onSelectLayer={setSelectedLayer}
              onSelectNode={handleSelectNode}
            />
          )}
          {activeView === 'live' && (
            <LiveView
              isCompiling={isCompiling}
              taskStatus={taskStatus}
              progress={progress}
              result={result}
              parsedEntities={parsedProgress.entities}
              recentSignals={recentSignals}
            />
          )}
          {activeView === 'governance' && (
            <GovernanceView
              result={result}
              model={model}
              progress={progress}
              taskStatus={taskStatus}
              taskId={currentTaskId}
              selectedNode={selectedNode}
              onOpenNode={handleSelectNode}
              onContinueTask={handleContinueTask}
            />
          )}
          {activeView === 'export' && (
            <ExportView
              result={result}
              model={model}
              taskId={currentTaskId}
            />
          )}
        </main>

        <DoraPanel
          node={selectedNode}
          model={model}
          result={result}
          messages={doraMessages}
          input={doraInput}
          onInputChange={setDoraInput}
          onSubmit={handleAskDora}
          onSelectNode={handleSelectNode}
        />
      </div>

      <StatusStrip result={result} model={model} isCompiling={isCompiling} progress={progress} taskStatus={taskStatus} />

      {isCompiling && (
        <div className="border-t border-[var(--border-subtle)] bg-[var(--bg-secondary)] px-4 py-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="text-[11px] text-[var(--text-secondary)]">
              {stageCopy(progress, taskStatus)}
            </div>
            <button
              onClick={cancel}
              className="rounded-md border border-[var(--border)] px-3 py-1.5 text-[11px] text-[var(--text-secondary)] transition-colors hover:border-red-400/30 hover:text-red-300"
            >
              Cancel compile
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default function WorkbenchPage() {
  return (
    <Suspense
      fallback={
        <div className="flex h-full items-center justify-center bg-[var(--bg-primary)]">
          <div className="text-sm text-[var(--text-muted)]">Loading workbench...</div>
        </div>
      }
    >
      <WorkbenchShell />
    </Suspense>
  );
}
