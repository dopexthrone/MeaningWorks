import Link from 'next/link';

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
      <div className="w-16 h-16 rounded-full bg-[var(--bg-tertiary)] flex items-center justify-center mb-6">
        <span className="text-2xl font-bold text-[var(--text-muted)]">404</span>
      </div>
      <h2 className="text-xl font-semibold text-[var(--text-primary)] mb-2">Page not found</h2>
      <p className="text-sm text-[var(--text-muted)] mb-6 max-w-sm">
        The page you&apos;re looking for doesn&apos;t exist or has been moved.
      </p>
      <Link href="/" className="btn-primary text-sm">
        Back to Compile
      </Link>
    </div>
  );
}
