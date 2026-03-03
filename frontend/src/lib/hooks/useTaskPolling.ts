'use client';

import { useState, useEffect, useRef, useCallback } from 'react';

interface UseTaskPollingOptions<T> {
  fetcher: () => Promise<T>;
  interval?: number;
  enabled?: boolean;
  isComplete?: (data: T) => boolean;
}

interface UseTaskPollingResult<T> {
  data: T | null;
  error: Error | null;
  isPolling: boolean;
}

export function useTaskPolling<T>({
  fetcher,
  interval = 2000,
  enabled = true,
  isComplete,
}: UseTaskPollingOptions<T>): UseTaskPollingResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const visibleRef = useRef(true);

  const poll = useCallback(async () => {
    if (!visibleRef.current) return;
    try {
      const result = await fetcher();
      setData(result);
      setError(null);
      if (isComplete?.(result)) {
        setIsPolling(false);
        return;
      }
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      setIsPolling(false);
      return;
    }
    timerRef.current = setTimeout(poll, interval);
  }, [fetcher, interval, isComplete]);

  useEffect(() => {
    if (!enabled) {
      setIsPolling(false);
      return;
    }

    setIsPolling(true);
    poll();

    const onVisibility = () => {
      visibleRef.current = document.visibilityState === 'visible';
      if (visibleRef.current && isPolling) poll();
    };
    document.addEventListener('visibilitychange', onVisibility);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [enabled, poll, isPolling]);

  return { data, error, isPolling };
}
