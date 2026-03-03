'use client';

import { useState, useCallback } from 'react';
import { getHealth, getMetrics } from '@/lib/api/health';
import { useTaskPolling } from './useTaskPolling';
import type { HealthResponse, MetricsResponse } from '@/lib/api/types';

export function useHealth(pollInterval = 30000) {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);

  const { data: health, error } = useTaskPolling<HealthResponse>({
    fetcher: useCallback(() => getHealth(), []),
    interval: pollInterval,
    enabled: true,
  });

  // Fetch metrics once
  useState(() => {
    getMetrics().then(setMetrics).catch(() => {});
  });

  return { health, metrics, error };
}
