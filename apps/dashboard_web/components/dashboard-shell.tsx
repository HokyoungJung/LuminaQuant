'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import type { ReactNode } from 'react';

import { navigationItems } from '@/lib/python-bridge';

export function DashboardShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand-block">
          <p className="eyebrow">LuminaQuant</p>
          <h1>Operations Console</h1>
          <p className="lede">Research &amp; live operations console.</p>
        </div>
        <nav aria-label="Dashboard sections">
          <ul className="nav-list">
            {navigationItems.map((item) => {
              const active = item.href === '/'
                ? pathname === '/'
                : pathname === item.href || pathname.startsWith(`${item.href}/`);
              return (
                <li key={item.id} className="nav-item">
                  <Link
                    href={item.href}
                    title={item.summary}
                    aria-current={active ? 'page' : undefined}
                  >
                    {item.label}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>
      </aside>
      <main className="content">{children}</main>
    </div>
  );
}
