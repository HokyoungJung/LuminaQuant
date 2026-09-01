import type { Metadata } from 'next';
import type { ReactNode } from 'react';

import { DashboardShell } from '@/components/dashboard-shell';

import './globals.css';

export const metadata: Metadata = {
  title: 'LuminaQuant Dashboard Web',
  description: 'Next.js migration foundation for the LuminaQuant dashboard.',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <DashboardShell>{children}</DashboardShell>
      </body>
    </html>
  );
}
