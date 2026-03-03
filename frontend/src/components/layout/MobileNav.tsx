'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

interface MobileNavProps {
  open: boolean;
  onClose: () => void;
}

const NAV_ITEMS = [
  { href: '/', label: 'Compile' },
  { href: '/corpus', label: 'Corpus' },
  { href: '/health', label: 'Health' },
  { href: '/settings', label: 'Settings' },
];

export function MobileNav({ open, onClose }: MobileNavProps) {
  const pathname = usePathname();

  if (!open) return null;

  return (
    <>
      <div className="fixed inset-0 bg-black/50 z-40 md:hidden" onClick={onClose} />
      <div className="fixed inset-y-0 left-0 w-64 bg-[var(--bg-secondary)] border-r border-[var(--border)] z-50 md:hidden">
        <div className="flex items-center justify-between px-5 py-4 border-b border-[var(--border)]">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-brand-500 flex items-center justify-center">
              <span className="text-white font-bold text-sm">M</span>
            </div>
            <span className="font-semibold text-sm">Motherlabs</span>
          </div>
          <button onClick={onClose} className="p-1">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <nav className="py-3 px-2 space-y-0.5">
          {NAV_ITEMS.map(({ href, label }) => {
            const active = href === '/' ? pathname === '/' : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                onClick={onClose}
                className={`sidebar-link ${active ? 'sidebar-link-active' : ''}`}
              >
                {label}
              </Link>
            );
          })}
        </nav>
      </div>
    </>
  );
}
