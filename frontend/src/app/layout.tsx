'use client';

import { useState, useEffect } from 'react';
import '@/styles/globals.css';
import { Sidebar } from '@/components/layout/Sidebar';
import { MobileNav } from '@/components/layout/MobileNav';
import { Header } from '@/components/layout/Header';
import { usePathname } from 'next/navigation';

const TITLES: Record<string, string> = {
  '/': 'Compile',
  '/corpus': 'Corpus',
  '/health': 'Health',
  '/settings': 'Settings',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [healthStatus, setHealthStatus] = useState<'ok' | 'error' | 'unknown'>('unknown');
  const pathname = usePathname();

  const title = TITLES[pathname] || (pathname.startsWith('/compile') ? 'Compilation' : pathname.startsWith('/corpus') ? 'Corpus Detail' : 'Motherlabs');

  useEffect(() => {
    const check = () => {
      const base = typeof window !== 'undefined'
        ? localStorage.getItem('motherlabs_api_url') || process.env.NEXT_PUBLIC_API_URL || ''
        : '';
      fetch(`${base}/v2/health`)
        .then((r) => setHealthStatus(r.ok ? 'ok' : 'error'))
        .catch(() => setHealthStatus('error'));
    };
    check();
    const interval = setInterval(check, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>{title} - Motherlabs</title>
        <meta name="description" content="Semantic compiler that transforms natural language intent into verified, production-ready blueprints and code." />
        <meta property="og:title" content={`${title} - Motherlabs`} />
        <meta property="og:description" content="Transform ideas into working software. Semantic compilation with trust verification." />
        <meta property="og:type" content="website" />
        <meta property="og:image" content="/logo.svg" />
        <meta name="robots" content="index, follow" />
        <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
      </head>
      <body className="flex h-screen overflow-hidden">
        <Sidebar />
        <MobileNav open={mobileOpen} onClose={() => setMobileOpen(false)} />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Header
            title={title}
            healthStatus={healthStatus}
            onMenuToggle={() => setMobileOpen(true)}
          />
          <main className="flex-1 overflow-y-auto p-6">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
