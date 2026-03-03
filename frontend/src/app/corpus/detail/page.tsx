'use client';

import { useSearchParams } from 'next/navigation';
import { Suspense } from 'react';
import { CorpusDetail } from '@/components/corpus/CorpusDetail';
import { LoadingSkeleton } from '@/components/shared/LoadingSkeleton';

function CorpusDetailContent() {
  const searchParams = useSearchParams();
  const id = searchParams.get('id');

  if (!id) {
    return (
      <div className="card text-center py-8">
        <p className="text-[var(--text-muted)]">No compilation ID provided</p>
        <a href="/corpus" className="btn-primary text-sm mt-4 inline-block">Back to Corpus</a>
      </div>
    );
  }

  return <CorpusDetail id={id} />;
}

export default function CorpusDetailPage() {
  return (
    <Suspense fallback={<LoadingSkeleton lines={8} />}>
      <CorpusDetailContent />
    </Suspense>
  );
}
