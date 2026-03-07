export const BLUEPRINTS_SSOT_VERSION = '1.0.0';
export const BLUEPRINTS_SSOT_DATE = '2026-03-07';

export const LAYER_CODES = [
  'INT', 'SEM', 'ORG', 'COG', 'AGN', 'STR', 'STA', 'IDN', 'TME',
  'EXC', 'DAT', 'SFX', 'NET', 'RES', 'OBS', 'SEC', 'CTR', 'EMG', 'MET',
] as const;

export const CONCERN_CODES = [
  'SEM', 'ENT', 'REL', 'SCH', 'ENM', 'COL', 'BHV', 'FNC', 'TRG', 'STP', 'LGC',
  'STA', 'TRN', 'DAT', 'TRF', 'FLW', 'ACT', 'PRM', 'AUT', 'SCO', 'MEM', 'PLN',
  'DLG', 'NGT', 'CNS', 'HND', 'ORC', 'OBS', 'CTR', 'CFG', 'LMT', 'PLY', 'LOG',
  'MET', 'TRC', 'ALT', 'CND', 'PRV', 'VRS', 'SIM', 'RPT', 'WRT', 'EMT', 'RED',
] as const;

export const SCOPE_CODES = [
  'ECO', 'APP', 'DOM', 'FET', 'CMP', 'FNC', 'STP', 'OPR', 'EXP', 'VAL',
] as const;

export const DIMENSION_CODES = [
  'WHY', 'WHO', 'WHAT', 'WHEN', 'WHERE', 'HOW', 'HOW_MUCH', 'IF',
] as const;

export const DOMAIN_CODES = [
  'SFT', 'ORG', 'COG', 'ECN', 'PHY', 'SOC', 'NET', 'EDU', 'CRE', 'LGL',
] as const;

export const FILL_STATE_CODES = ['F', 'P', 'E', 'B', 'Q', 'C'] as const;
export const NODE_STATUS_CODES = ['authored', 'candidate', 'quarantined', 'promoted'] as const;
export const RULE_TYPES = ['constraint', 'policy', 'trigger', 'condition'] as const;
export const TEST_TYPES = ['rule', 'anti_goal', 'state_transition', 'invariant'] as const;

export type LayerCode = typeof LAYER_CODES[number];
export type ConcernCode = typeof CONCERN_CODES[number];
export type ScopeCode = typeof SCOPE_CODES[number];
export type DimensionCode = typeof DIMENSION_CODES[number];
export type DomainCode = typeof DOMAIN_CODES[number];
export type FillStateCode = typeof FILL_STATE_CODES[number];
export type NodeStatusCode = typeof NODE_STATUS_CODES[number];
export type RuleType = typeof RULE_TYPES[number];
export type TestType = typeof TEST_TYPES[number];

export const SCOPE_DEPTH: Record<ScopeCode, number> = {
  ECO: 0,
  APP: 1,
  DOM: 2,
  FET: 3,
  CMP: 4,
  FNC: 5,
  STP: 6,
  OPR: 7,
  EXP: 8,
  VAL: 9,
};

const LAYER_SET = new Set<string>(LAYER_CODES);
const CONCERN_SET = new Set<string>(CONCERN_CODES);
const SCOPE_SET = new Set<string>(SCOPE_CODES);
const DIMENSION_SET = new Set<string>(DIMENSION_CODES);
const DOMAIN_SET = new Set<string>(DOMAIN_CODES);
const FILL_STATE_SET = new Set<string>(FILL_STATE_CODES);
const NODE_STATUS_SET = new Set<string>(NODE_STATUS_CODES);
const RULE_TYPE_SET = new Set<string>(RULE_TYPES);
const TEST_TYPE_SET = new Set<string>(TEST_TYPES);
const NODE_REF_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

export interface Postcode {
  layer: LayerCode;
  concern: ConcernCode;
  scope: ScopeCode;
  dimension: DimensionCode;
  domain: DomainCode;
  key: string;
  depth: number;
}

export interface NodeRef {
  postcode: string;
  name: string;
  key: string;
}

export function parsePostcode(raw: string): Postcode {
  const parts = raw.trim().split('.');
  if (parts.length !== 5) {
    throw new Error(`Postcode must have exactly 5 axes: ${raw}`);
  }

  const [layer, concern, scope, dimension, domain] = parts;
  if (!LAYER_SET.has(layer)) {
    throw new Error(`Unknown layer code: ${layer}`);
  }
  if (!CONCERN_SET.has(concern)) {
    throw new Error(`Unknown concern code: ${concern}`);
  }
  if (!SCOPE_SET.has(scope)) {
    throw new Error(`Unknown scope code: ${scope}`);
  }
  if (!DIMENSION_SET.has(dimension)) {
    throw new Error(`Unknown dimension code: ${dimension}`);
  }
  if (!DOMAIN_SET.has(domain)) {
    throw new Error(`Unknown domain code: ${domain}`);
  }

  return {
    layer: layer as LayerCode,
    concern: concern as ConcernCode,
    scope: scope as ScopeCode,
    dimension: dimension as DimensionCode,
    domain: domain as DomainCode,
    key: [layer, concern, scope, dimension, domain].join('.'),
    depth: SCOPE_DEPTH[scope as ScopeCode],
  };
}

export function isPostcode(raw: string): boolean {
  try {
    parsePostcode(raw);
    return true;
  } catch {
    return false;
  }
}

export function parseNodeRef(raw: string): NodeRef {
  const trimmed = raw.trim();
  const slashIndex = trimmed.indexOf('/');
  if (slashIndex <= 0 || slashIndex === trimmed.length - 1) {
    throw new Error(`Node ref must use POSTCODE/name format: ${raw}`);
  }

  const postcode = trimmed.slice(0, slashIndex);
  const name = trimmed.slice(slashIndex + 1);
  parsePostcode(postcode);
  if (!NODE_REF_NAME_RE.test(name)) {
    throw new Error(`Invalid node ref name: ${name}`);
  }

  return { postcode, name, key: `${postcode}/${name}` };
}

export function isNodeRef(raw: string): boolean {
  try {
    parseNodeRef(raw);
    return true;
  } catch {
    return false;
  }
}

export function parseSemanticRef(raw: string): string {
  if (isNodeRef(raw)) {
    return raw;
  }
  parsePostcode(raw);
  return raw;
}

export function normalizeNodeName(raw: string): string {
  const normalized = raw
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^[._-]+|[._-]+$/g, '');

  return normalized || 'node';
}

export function makeNodeRef(postcode: string, primitive: string): string {
  return `${parsePostcode(postcode).key}/${normalizeNodeName(primitive)}`;
}

function validatePostcodeList(postcodes: string[], field: string): void {
  for (const postcode of postcodes) {
    try {
      parseSemanticRef(postcode);
    } catch {
      throw new Error(`Invalid semantic ref in ${field}: ${postcode}`);
    }
  }
}

export interface ConstraintSpec {
  description: string;
  expression?: string;
}

export interface FreshnessPolicy {
  decay_rate: number;
  floor: number;
  stale_after: number;
}

export interface NodeReferences {
  read_before: string[];
  read_after: string[];
  see_also: string[];
  deep_dive: string[];
  warns: string[];
}

export interface NodeProvenance {
  source_ref: string[];
  agent_id: string;
  run_id: string;
  timestamp: string;
  human_input: boolean;
}

export interface BlueprintNode {
  id: string;
  postcode: string;
  primitive: string;
  description: string;
  notes: string[];
  fill_state: FillStateCode;
  confidence: number;
  status: NodeStatusCode;
  version: number;
  created_at: string;
  updated_at: string;
  last_verified: string;
  freshness: FreshnessPolicy;
  parent: string | null;
  children: string[];
  connections: string[];
  depth: number;
  references: NodeReferences;
  provenance: NodeProvenance;
  token_cost: number;
  constraints: ConstraintSpec[];
  constraint_source: string[];
  layer: LayerCode;
  concern: ConcernCode;
  scope: ScopeCode;
  dimension: DimensionCode;
  domain: DomainCode;
}

export function normalizeBlueprintNode(
  node: Omit<BlueprintNode, 'layer' | 'concern' | 'scope' | 'dimension' | 'domain' | 'depth'> &
    Partial<Pick<BlueprintNode, 'layer' | 'concern' | 'scope' | 'dimension' | 'domain' | 'depth'>>
): BlueprintNode {
  const parsed = parsePostcode(node.postcode);

  if (!FILL_STATE_SET.has(node.fill_state)) {
    throw new Error(`Unknown fill state: ${node.fill_state}`);
  }
  if (!NODE_STATUS_SET.has(node.status)) {
    throw new Error(`Unknown node status: ${node.status}`);
  }

  validatePostcodeList(node.connections, 'connections');
  validatePostcodeList(node.constraint_source, 'constraint_source');
  validatePostcodeList(node.references.read_before, 'references.read_before');
  validatePostcodeList(node.references.read_after, 'references.read_after');
  validatePostcodeList(node.references.see_also, 'references.see_also');
  validatePostcodeList(node.references.warns, 'references.warns');

  if (node.layer && node.layer !== parsed.layer) {
    throw new Error(`layer conflicts with postcode: ${node.postcode}`);
  }
  if (node.concern && node.concern !== parsed.concern) {
    throw new Error(`concern conflicts with postcode: ${node.postcode}`);
  }
  if (node.scope && node.scope !== parsed.scope) {
    throw new Error(`scope conflicts with postcode: ${node.postcode}`);
  }
  if (node.dimension && node.dimension !== parsed.dimension) {
    throw new Error(`dimension conflicts with postcode: ${node.postcode}`);
  }
  if (node.domain && node.domain !== parsed.domain) {
    throw new Error(`domain conflicts with postcode: ${node.postcode}`);
  }
  if (node.depth !== undefined && node.depth !== parsed.depth) {
    throw new Error(`depth conflicts with postcode: ${node.postcode}`);
  }

  return {
    ...node,
    layer: parsed.layer,
    concern: parsed.concern,
    scope: parsed.scope,
    dimension: parsed.dimension,
    domain: parsed.domain,
    depth: parsed.depth,
  };
}

export function effectiveConfidence(node: BlueprintNode, daysSinceVerified = 0): number {
  return Math.max(
    node.confidence - (daysSinceVerified * node.freshness.decay_rate),
    node.freshness.floor,
  );
}

export interface AntiGoal {
  description: string;
  derived_from: string;
  severity: 'critical' | 'high' | 'medium';
  detection: string;
}

export interface RuntimeAnchor {
  enabled: boolean;
  invariants: string[];
  check_mode: 'startup' | 'periodic' | 'continuous';
}

export interface ContextBudget {
  total: number;
  reserved: number;
  available: number;
  per_agent: number;
  compression_trigger: number;
}

export interface IntentContract {
  seed_text: string;
  goals: string[];
  constraints: string[];
  layers_in_scope: LayerCode[];
  domains_in_scope: DomainCode[];
  known_unknowns: string[];
  budget_limit: number;
  anti_goals: AntiGoal[];
  runtime_anchor: RuntimeAnchor;
  context_budget: ContextBudget;
  seed_hash: string;
  contract_hash: string;
}

export interface Silence {
  layer: LayerCode;
  reason: string;
  type: 'intentional' | 'deferred' | 'out_of_scope';
  decided_by: 'agent' | 'human' | 'intent_contract';
}

export interface DepthReport {
  label?: string;
  average_scope_depth?: number;
  filled_ratio?: number;
  partial_ratio?: number;
  empty_ratio?: number;
  activated_layer_ratio?: number;
  anti_goals_identified?: number;
  dialogue_rounds_completed?: number;
  gaps_remaining?: number;
  [key: string]: unknown;
}

export interface CostReport {
  estimated_usd?: number;
  actual_usd?: number;
  by_agent?: Record<string, number>;
  budget_limit_usd?: number;
  halted?: boolean;
  [key: string]: unknown;
}

export interface GovernanceDecision {
  postcode: string;
  question: string;
  answer?: string;
  options?: string[];
  node_ref?: string | null;
  kind?: string;
  stage?: string;
}

export interface AxiomViolation {
  axiom: string;
  node: string;
  detail: string;
}

export interface HumanDecision {
  postcode: string;
  question: string;
  answer: string;
  timestamp: string;
}

export interface GovernanceReport {
  total_nodes: number;
  promoted: number;
  quarantined: GovernanceDecision[];
  escalated: GovernanceDecision[];
  axiom_violations: AxiomViolation[];
  human_decisions: HumanDecision[];
  coverage: number;
  anti_goals_checked: number;
  compilation_depth: DepthReport;
  cost_report: CostReport;
}

export interface BlueprintMetadata {
  id: string;
  seed: string;
  seed_hash: string;
  blueprint_hash: string;
  created_at: string;
  version: string;
  compilation_depth: DepthReport;
}

export interface LayerCoverage {
  layer: LayerCode;
  nodeCount: number;
  coverage: string[];
}

export interface CompiledEntity {
  name: string;
  postcode: string;
  description: string;
  attributes: string[];
  confidence: number;
}

export interface CompiledFunction {
  name: string;
  postcode: string;
  description: string;
  inputs: string[];
  outputs: string[];
  rules: string[];
  confidence: number;
}

export interface CompiledRule {
  name: string;
  postcode: string;
  description: string;
  type: RuleType;
  confidence: number;
}

export interface CompiledRelationship {
  from: string;
  to: string;
  relation: string;
  postcode: string;
  confidence: number;
}

export interface StateTransition {
  from: string;
  to: string;
  trigger: string;
}

export interface CompiledStateMachine {
  name: string;
  postcode: string;
  states: string[];
  transitions: StateTransition[];
}

export interface Gap {
  layer: LayerCode;
  concern: ConcernCode;
  scope: ScopeCode;
  reason: string;
  priority: 'high' | 'medium' | 'low';
}

export interface RejectedPath {
  postcode: string;
  reason: string;
}

export interface FailedPathConflict {
  nodes: string[];
  resolution: string;
}

export interface FailedPaths {
  rejected: RejectedPath[];
  deferred: RejectedPath[];
  conflicts: FailedPathConflict[];
}

export interface CompiledTest {
  source_postcode: string;
  test_type: TestType;
  description: string;
  assertion: string;
}

export interface CompiledBlueprint {
  metadata: BlueprintMetadata;
  intent_contract: IntentContract;
  layers: LayerCoverage[];
  entities: CompiledEntity[];
  functions: CompiledFunction[];
  rules: CompiledRule[];
  relationships: CompiledRelationship[];
  state_machines: CompiledStateMachine[];
  silences: Silence[];
  gaps: Gap[];
  failed_paths: FailedPaths;
  governance_report: GovernanceReport;
  tests: CompiledTest[];
}

export function validateRuleType(value: string): value is RuleType {
  return RULE_TYPE_SET.has(value);
}

export function validateTestType(value: string): value is TestType {
  return TEST_TYPE_SET.has(value);
}
