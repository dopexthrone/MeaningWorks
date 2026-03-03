'use client';

import { useState, useEffect, useCallback } from 'react';
import { listCorpus } from '@/lib/api/corpus';
import type { CompilationRecord } from '@/lib/api/types';

export function useCorpus(domain?: string) {
  const [records, setRecords] = useState<CompilationRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetch = useCallback(async () => {
    setLoading(true);
    try {
      const res = await listCorpus(page, 20, domain);
      setRecords(res.compilations);
      setTotal(res.total);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [page, domain]);

  useEffect(() => { fetch(); }, [fetch]);

  return { records, total, page, setPage, loading, error, refetch: fetch };
}
