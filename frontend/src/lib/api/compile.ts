import { apiFetch } from './client';
import type {
  CompileRequest,
  AsyncCompileResponse,
  TaskDecisionRequest,
  TaskDecisionResponse,
  TaskStatusResponse,
} from './types';

export async function compileAsync(req: CompileRequest): Promise<AsyncCompileResponse> {
  return apiFetch<AsyncCompileResponse>('/v2/compile/async', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export async function getTaskStatus(taskId: string): Promise<TaskStatusResponse> {
  return apiFetch<TaskStatusResponse>(`/v2/tasks/${taskId}`);
}

export async function recordTaskDecision(taskId: string, req: TaskDecisionRequest): Promise<TaskDecisionResponse> {
  return apiFetch<TaskDecisionResponse>(`/v2/tasks/${taskId}/decisions`, {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export async function cancelTask(taskId: string): Promise<{ task_id: string; cancelled: boolean }> {
  return apiFetch(`/v2/tasks/${taskId}`, { method: 'DELETE' });
}

export interface SwarmExecuteRequest {
  description: string;
  domain?: string;
  request_type: string;
  cost_cap_usd?: number;
  provider?: string | null;
  previous_task_id?: string | null;
}

export interface SwarmExecuteResponse {
  task_id: string;
  estimated_cost_usd: number;
}

export async function swarmExecute(req: SwarmExecuteRequest): Promise<SwarmExecuteResponse> {
  return apiFetch<SwarmExecuteResponse>('/v2/swarm/execute', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}
