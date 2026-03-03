import { apiFetch } from './client';
import type { CorpusListResponse, CompilationRecord } from './types';

export async function listCorpus(
  page = 1,
  pageSize = 20,
  domain?: string
): Promise<CorpusListResponse> {
  const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
  if (domain) params.set('domain', domain);
  return apiFetch<CorpusListResponse>(`/v1/corpus?${params}`);
}

export async function getCompilation(id: string): Promise<CompilationRecord> {
  return apiFetch<CompilationRecord>(`/v1/corpus/${id}`);
}
