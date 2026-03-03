import { apiFetch } from './client';
import type { HealthResponse, MetricsResponse } from './types';

export async function getHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>('/v2/health');
}

export async function getMetrics(): Promise<MetricsResponse> {
  return apiFetch<MetricsResponse>('/v2/metrics');
}
