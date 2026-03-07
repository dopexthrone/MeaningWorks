// =============================================================================
// API Types — mirrors api/v2/models.py + core/schema.py
// =============================================================================

import type {
  BlueprintNode as SemanticBlueprintNode,
  GovernanceReport as SemanticGovernanceReport,
} from '../semantic/protocol';

// --- Enums ---

export type ComponentType = 'entity' | 'process' | 'interface' | 'event' | 'constraint' | 'subsystem';

export type RelationshipType =
  | 'contains' | 'triggers' | 'accesses' | 'depends_on' | 'flows_to'
  | 'generates' | 'snapshots' | 'propagates' | 'constrained_by' | 'bidirectional';

export type VerificationBadge = 'verified' | 'partial' | 'unverified';

export type TaskStatus = 'pending' | 'running' | 'awaiting_decision' | 'complete' | 'error' | 'cancelled';

export type TrustLevel = 'fast' | 'standard' | 'thorough';

// Glass box: structured compilation emissions
export type InsightCategory = 'discovery' | 'decision' | 'warning' | 'metric' | 'resolution' | 'status';

export interface StructuredInsight {
  text: string;
  category: InsightCategory;
  stage: string;
  metrics: Record<string, number>;
  reasoning: string;
  timestamp: number;
}

export interface DifficultySignal {
  input_quality: number;
  unknown_count: number;
  conflict_count: number;
  ambiguity_count: number;
  component_complexity: number;
  irritation_depth: number;
}

export interface StageResultSummary {
  stage: string;
  success: boolean;
  errors: string[];
  warnings: string[];
  retries: number;
}

// --- Blueprint Schema ---

export interface Parameter {
  name: string;
  type_hint: string;
  default?: string | null;
  derived_from: string;
}

export interface MethodSpec {
  name: string;
  parameters: Parameter[];
  return_type: string;
  description: string;
  derived_from: string;
}

export interface Transition {
  from_state: string;
  to_state: string;
  trigger: string;
  derived_from: string;
}

export interface StateSpec {
  states: string[];
  initial_state: string;
  transitions: Transition[];
  derived_from: string;
}

export interface BlueprintComponent {
  name: string;
  type: ComponentType;
  description: string;
  derived_from: string;
  attributes: Record<string, unknown>;
  methods: MethodSpec[];
  state_machine?: StateSpec | null;
  validation_rules: string[];
}

export type Cardinality = '1:1' | '1:N' | 'N:1' | 'N:M' | '';

export interface BlueprintRelationship {
  // API may return either `from`/`to` or `from_component`/`to_component`
  from_component?: string;
  to_component?: string;
  from?: string;
  to?: string;
  type: RelationshipType;
  description: string;
  derived_from: string;
  cardinality?: Cardinality;
}

export interface BlueprintConstraint {
  description: string;
  applies_to: string[];
  derived_from: string;
}

export interface AcknowledgedUnknown {
  original: string;
  reason: string;
  default_applied: string | null;
}

export interface Resolution {
  original: string;
  resolution: string;
  evidence: string[];
  type: 'context_match' | 'structural' | 'default_applied';
}

export interface Blueprint {
  components: BlueprintComponent[];
  relationships: BlueprintRelationship[];
  constraints: BlueprintConstraint[];
  unresolved: string[];
  core_need?: string;
  acknowledged_unknowns?: AcknowledgedUnknown[];
  resolutions?: Resolution[];
}

// --- Trust ---

export interface TrustResponse {
  overall_score: number;
  provenance_depth: number;
  fidelity_scores: Record<string, number>;
  gap_report: string[];
  verification_badge: VerificationBadge;
  silence_zones: string[];
  confidence_trajectory: number[];
  derivation_chain_length: number;
  dimensional_coverage: Record<string, number>;
}

// --- Request Models ---

export interface CompileRequest {
  description: string;
  domain?: string;
  trust_level?: TrustLevel;
  materialize?: boolean;
  output_format?: string | null;
  provider?: string | null;
  enrich?: boolean;
  canonical_components?: string[] | null;
  canonical_relationships?: string[][] | null;
}

// --- Response Models ---

export interface UsageResponse {
  tokens: number;
  cost_usd: number;
  domain: string;
  adapter_version: string;
}

export interface TerminationSemanticProgress {
  fingerprint_changed?: boolean;
  verification_score_delta?: number;
  components_delta?: number;
  gates_changed?: boolean;
}

export interface TerminationCondition {
  status?: 'awaiting_human' | 'stalled' | 'halted' | 'complete' | string;
  reason?: string;
  message?: string;
  next_action?: string;
  semantic_progress?: TerminationSemanticProgress;
}

export interface CompileResponse {
  success: boolean;
  blueprint: Blueprint;
  semantic_nodes?: SemanticBlueprintNode[];
  termination_condition?: TerminationCondition;
  governance_report?: SemanticGovernanceReport;
  materialized_output?: Record<string, string>;
  trust: TrustResponse;
  dimensional_metadata?: Record<string, unknown>;
  interface_map?: Record<string, unknown>;
  verification?: Record<string, unknown>;
  context_graph?: Record<string, unknown>;
  domain: string;
  adapter_version?: string;
  duration_seconds?: number;
  usage?: UsageResponse;
  error?: string | null;
  // Feature 1: Visible Provenance
  input_text?: string;
  // Feature 3: Runnable Output
  project_files?: Record<string, string>;
  entry_point?: string;
  project_name?: string;
  // YAML output tree (fractal blueprint)
  yaml_output?: Record<string, string>;
  // Glass box: structured compilation emissions
  structured_insights?: StructuredInsight[];
  difficulty?: DifficultySignal;
  stage_results?: StageResultSummary[];
  stage_timings?: Record<string, number>;
  retry_counts?: Record<string, number>;
  fracture?: {
    stage: string;
    competing_configs?: string[];
    collapsing_constraint: string;
    agent?: string;
    context?: string;
  } | null;
  interrogation?: Record<string, unknown>;
  // Compilation quality benchmark
  benchmark?: BenchmarkResponse;
}

export interface AsyncCompileResponse {
  task_id: string;
  status: string;
  poll_url: string;
}

export interface CompilationProgress {
  current_stage: string;
  stage_index: number;
  total_stages: number;
  insights: string[];
  structured_insights?: StructuredInsight[];
  difficulty?: DifficultySignal;
  termination_condition?: TerminationCondition;
  escalations?: Array<{
    postcode: string;
    question: string;
    answer?: string;
    options?: string[];
    node_ref?: string | null;
    kind?: string;
    stage?: string;
  }>;
  human_decisions?: Array<{ postcode: string; question: string; answer: string; timestamp: string }>;
}

export interface TaskStatusResponse {
  task_id: string;
  status: TaskStatus;
  result?: CompileResponse | null;
  error?: string | null;
  progress?: CompilationProgress | null;
}

export interface TaskDecisionRequest {
  postcode: string;
  question: string;
  answer: string;
  timestamp?: string | null;
}

export interface TaskDecisionResponse {
  task_id: string;
  saved: boolean;
  progress?: CompilationProgress | null;
  next_task_id?: string | null;
  termination_condition?: TerminationCondition | null;
}

// Feature 5: Corpus Benchmarking
export interface CorpusBenchmarkResponse {
  domain: string;
  total_compilations: number;
  avg_component_count: number;
  avg_trust_score: number;
  avg_gap_count: number;
}

// --- Benchmark (Compilation Quality) ---

export interface BenchmarkDimensionScore {
  dimension: string;
  weight: number;
  score: number;
  target: number;
  gap: number;
  met: boolean;
}

export interface BenchmarkResponse {
  dimensions: Record<string, number>;
  composite_score: number;
  composite_pct: number;
  gen_label: string;
  scorecard: BenchmarkDimensionScore[];
  details: Record<string, unknown>;
}

// --- Health ---

export interface HealthResponse {
  status: string;
  version: string;
  domains_available: string[];
  corpus_size: number;
  uptime_seconds: number;
  worker_queue_depth: number;
  disk_free_mb: number;
  disk_total_mb: number;
}

export interface MetricsResponse {
  per_domain: Record<string, Record<string, unknown>>;
  total_compilations: number;
  total_cost_usd: number;
}

// --- Domains ---

export interface DomainInfo {
  name: string;
  version: string;
  output_format: string;
  file_extension: string;
  vocabulary_types: string[];
  relationship_types: string[];
  actionability_checks: string[];
}

export interface DomainListResponse {
  domains: DomainInfo[];
}

// --- Corpus ---

// Summary record returned by GET /v1/corpus (list endpoint)
export interface CompilationRecordSummary {
  id: string;
  input_text: string;
  domain: string;
  timestamp: string;
  components_count: number;
  insights_count: number;
  success: boolean;
  provider: string;
  model: string;
}

// Full record assembled from GET /v1/corpus/{id} (detail endpoint)
export interface CompilationRecordDetail {
  record: CompilationRecordSummary;
  blueprint: Blueprint | null;
  context_graph: Record<string, unknown> | null;
}

export interface CorpusListResponse {
  records: CompilationRecordSummary[];
  total: number;
  page: number;
  per_page: number;
}

export type {
  BlueprintMetadata,
  BlueprintNode,
  CompiledBlueprint,
  ConcernCode,
  ContextBudget,
  CostReport,
  DepthReport,
  DimensionCode,
  DomainCode,
  FillStateCode,
  GovernanceReport,
  IntentContract,
  LayerCode,
  LayerCoverage,
  NodeRef,
  NodeReferences,
  NodeStatusCode,
  Postcode,
  ScopeCode,
  Silence,
} from '../semantic/protocol';

export {
  BLUEPRINTS_SSOT_DATE,
  BLUEPRINTS_SSOT_VERSION,
  effectiveConfidence,
  isNodeRef,
  isPostcode,
  makeNodeRef,
  normalizeNodeName,
  normalizeBlueprintNode,
  parseNodeRef,
  parsePostcode,
  parseSemanticRef,
} from '../semantic/protocol';
