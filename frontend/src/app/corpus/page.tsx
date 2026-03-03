'use client';

import { useState } from 'react';
import { CorpusList } from '@/components/corpus/CorpusList';
import { DomainSelector } from '@/components/compile/DomainSelector';
import { useCorpus } from '@/lib/hooks/useCorpus';
import { LoadingSkeleton } from '@/components/shared/LoadingSkeleton';

export default function CorpusPage() {
  const [domain, setDomain] = useState<string>('');
  const { records, total, page, setPage, loading, error } = useCorpus(domain || undefined);

  return (
    <div className="max-w-4xl mx-auto space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Compilation History</h2>
        <div className="w-48">
          <select
            value={domain}
            onChange={(e) => { setDomain(e.target.value); setPage(1); }}
            className="input w-full text-sm"
          >
            <option value="">All domains</option>
            <option value="software">software</option>
            <option value="process">process</option>
            <option value="api">api</option>
            <option value="agent_system">agent_system</option>
          </select>
        </div>
      </div>

      {error && <div className="card text-red-400 text-sm">{error}</div>}

      {loading ? (
        <LoadingSkeleton lines={5} />
      ) : (
        <CorpusList records={records} total={total} page={page} onPageChange={setPage} />
      )}
    </div>
  );
}
