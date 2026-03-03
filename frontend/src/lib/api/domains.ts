import { apiFetch } from './client';
import type { DomainListResponse } from './types';

export async function listDomains(): Promise<DomainListResponse> {
  return apiFetch<DomainListResponse>('/v2/domains');
}
