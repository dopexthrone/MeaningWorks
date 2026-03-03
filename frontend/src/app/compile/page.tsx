'use client';

import { useSearchParams } from 'next/navigation';
import { Suspense } from 'react';
import { CompilationResult } from '@/components/compile/CompilationResult';
import { LoadingSkeleton } from '@/components/shared/LoadingSkeleton';

function CompileContent() {
  const searchParams = useSearchParams();
  const taskId = searchParams.get('id');

  if (!taskId) {
    return (
      <div className="card text-center py-8">
        <p className="text-[var(--text-muted)]">No task ID provided</p>
        <a href="/" className="btn-primary text-sm mt-4 inline-block">Back to Compile</a>
      </div>
    );
  }

  return <CompilationResult taskId={taskId} />;
}

export default function CompilePage() {
  return (
    <Suspense fallback={<LoadingSkeleton lines={8} />}>
      <CompileContent />
    </Suspense>
  );
}
