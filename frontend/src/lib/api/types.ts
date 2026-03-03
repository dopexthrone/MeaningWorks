// =============================================================================
// API Types — mirrors api/v2/models.py + core/schema.py
// =============================================================================

// --- Enums ---

export type ComponentType = 'entity' | 'process' | 'interface' | 'event' | 'constraint' | 'subsystem';

export type RelationshipType =
  | 'contains' | 'triggers' | 'accesses' | 'depends_on' | 'flows_to'
  | 'generates' | 'snapshots' | 'propagates' | 'constrained_by' | 'bidirectional';

export type VerificationBadge = 'verified' | 'partial' | 'unverified';

export type TaskStatus = 'pending' | 'running' | 'complete' | 'error' | 'cancelled';

export type TrustLevel = 'fast' | 'standard' | 'thorough';

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

export interface BlueprintRelationship {
  from_component: string;
  to_component: string;
  type: RelationshipType;
  description: string;
  derived_from: string;
}

export interface BlueprintConstraint {
  description: string;
  applies_to: string[];
  derived_from: string;
}

export interface Blueprint {
  components: BlueprintComponent[];
  relationships: BlueprintRelationship[];
  constraints: BlueprintConstraint[];
  unresolved: string[];
  core_need?: string;
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

export interface CompileResponse {
  success: boolean;
  blueprint: Blueprint;
  materialized_output: Record<string, string>;
  trust: TrustResponse;
  dimensional_metadata: Record<string, unknown>;
  interface_map: Record<string, unknown>;
  verification: Record<string, unknown>;
  context_graph: Record<string, unknown>;
  domain: string;
  adapter_version: string;
  usage: UsageResponse;
  error?: string | null;
}

export interface AsyncCompileResponse {
  task_id: string;
  status: string;
  poll_url: string;
}

export interface TaskStatusResponse {
  task_id: string;
  status: TaskStatus;
  result?: CompileResponse | null;
  error?: string | null;
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

export interface CompilationRecord {
  id: string;
  description: string;
  domain: string;
  timestamp: string;
  blueprint: Blueprint;
  trust: TrustResponse;
  materialized_output: Record<string, string>;
  component_count?: number;
  relationship_count?: number;
}

export interface CorpusListResponse {
  compilations: CompilationRecord[];
  total: number;
  page: number;
  page_size: number;
}
