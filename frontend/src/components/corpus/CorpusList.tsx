'use client';

import Link from 'next/link';
import type { CompilationRecord } from '@/lib/api/types';
import { CorpusCard } from './CorpusCard';
import { EmptyState } from '@/components/shared/EmptyState';

interface CorpusListProps {
  records: CompilationRecord[];
  total: number;
  page: number;
  onPageChange: (page: number) => void;
}

export function CorpusList({ records, total, page, onPageChange }: CorpusListProps) {
  if (!records.length) {
    return (
      <EmptyState
        title="No compilations yet"
        description="Compile your first intent to see it here"
        action={<Link href="/" className="btn-primary text-sm">Start compiling</Link>}
      />
    );
  }

  const pageSize = 20;
  const totalPages = Math.ceil(total / pageSize);

  return (
    <div className="space-y-3">
      {records.map((record) => (
        <Link key={record.id} href={`/corpus/detail?id=${record.id}`}>
          <CorpusCard record={record} />
        </Link>
      ))}

      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 pt-4">
          <button
            onClick={() => onPageChange(page - 1)}
            disabled={page <= 1}
            className="btn-secondary text-xs disabled:opacity-30"
          >
            Previous
          </button>
          <span className="text-xs text-[var(--text-muted)]">
            Page {page} of {totalPages}
          </span>
          <button
            onClick={() => onPageChange(page + 1)}
            disabled={page >= totalPages}
            className="btn-secondary text-xs disabled:opacity-30"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
