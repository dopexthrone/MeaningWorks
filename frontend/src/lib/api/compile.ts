import { apiFetch } from './client';
import type { CompileRequest, AsyncCompileResponse, TaskStatusResponse } from './types';

export async function compileAsync(req: CompileRequest): Promise<AsyncCompileResponse> {
  return apiFetch<AsyncCompileResponse>('/v2/compile/async', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export async function getTaskStatus(taskId: string): Promise<TaskStatusResponse> {
  return apiFetch<TaskStatusResponse>(`/v2/tasks/${taskId}`);
}

export async function cancelTask(taskId: string): Promise<{ task_id: string; cancelled: boolean }> {
  return apiFetch(`/v2/tasks/${taskId}`, { method: 'DELETE' });
}
