import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import Image from 'next/image';

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
        text: 'Getting Started',
        url: '/docs/getting-started',
        active: 'nested-url',
      },
      {
        text: 'Guides',
        url: '/docs/guides',
        active: 'nested-url',
      },
      {
        text: 'Documentation',
        url: '/docs',
        active: 'nested-url',
      },
    ],
  };
}
