import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { Star } from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';

import { ThemeToggle } from '@/components/theme-toggle';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <div className="flex items-center gap-2">
          <Image
            src="/logo.svg"
            alt="Loclean Logo"
            width={30}
            height={30}
            className="logo-light"
          />
          <Image
            src="/logo-dark.svg"
            alt="Loclean Logo"
            width={30}
            height={30}
            className="logo-dark"
          />
          <span className="font-bold text-lg">Loclean</span>
        </div>
      ),
    },
    links: [
      {
        type: 'custom',
        on: 'nav',
        children: <ThemeToggle />,
      },
      {
        type: 'custom',
        on: 'nav',
        children: (
          <Link
            href="https://github.com/nxank4/loclean"
            target="_blank"
            className="inline-flex items-center gap-1.5 rounded-full border border-fd-border bg-fd-background/90 px-3 py-1 text-xs font-medium text-fd-muted-foreground hover:bg-fd-accent/80 hover:text-fd-accent-foreground transition-colors"
          >
            <span>Star</span>
            <Star className="h-3.5 w-3.5" />
          </Link>
        ),
      },
    ],
  };
}
