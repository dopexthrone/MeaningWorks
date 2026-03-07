'use client';

import { useState, useCallback } from 'react';
import { compileAsync, getTaskStatus, cancelTask } from '@/lib/api/compile';
import { useTaskPolling } from './useTaskPolling';
import type { CompileRequest, CompileResponse, TaskStatusResponse } from '@/lib/api/types';

export type CompilePhase = 'idle' | 'submitting' | 'queued' | 'polling' | 'awaiting_decision' | 'complete' | 'error' | 'cancelled';

interface UseCompileResult {
  phase: CompilePhase;
  taskId: string | null;
  result: CompileResponse | null;
  error: string | null;
  pollData: TaskStatusResponse | null;
  submit: (req: CompileRequest) => Promise<void>;
  cancel: () => Promise<void>;
  reset: () => void;
}

export function useCompile(): UseCompileResult {
  const [phase, setPhase] = useState<CompilePhase>('idle');
  const [taskId, setTaskId] = useState<string | null>(null);
  const [result, setResult] = useState<CompileResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const { data: pollData } = useTaskPolling<TaskStatusResponse>({
    fetcher: useCallback(() => getTaskStatus(taskId!), [taskId]),
    enabled: phase === 'polling' && !!taskId,
    interval: 2000,
    isComplete: useCallback((d: TaskStatusResponse) => {
      if (d.status === 'complete') {
        setPhase('complete');
        setResult(d.result ?? null);
        return true;
      }
      if (d.status === 'awaiting_decision') {
        setPhase('awaiting_decision');
        setResult(d.result ?? null);
        return true;
      }
      if (d.status === 'error') {
        setPhase('error');
        setError(d.error ?? 'Compilation failed');
        return true;
      }
      if (d.status === 'cancelled') {
        setPhase('cancelled');
        return true;
      }
      return false;
    }, []),
  });

  const submit = useCallback(async (req: CompileRequest) => {
    setPhase('submitting');
    setError(null);
    setResult(null);
    try {
      const res = await compileAsync(req);
      setTaskId(res.task_id);
      setPhase('polling');
    } catch (err) {
      setPhase('error');
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  const cancelFn = useCallback(async () => {
    if (!taskId) return;
    try {
      await cancelTask(taskId);
      setPhase('cancelled');
    } catch {
      // Already cancelled or completed
    }
  }, [taskId]);

  const reset = useCallback(() => {
    setPhase('idle');
    setTaskId(null);
    setResult(null);
    setError(null);
  }, []);

  return { phase, taskId, result, error, pollData, submit, cancel: cancelFn, reset };
}
