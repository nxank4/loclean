import type { Metadata } from 'next';
import { Provider } from '@/components/provider';
import { Inter } from 'next/font/google';
import './global.css';

const inter = Inter({
  subsets: ['latin'],
});

const appUrl =
  process.env.NEXT_PUBLIC_APP_URL ?? 'http://localhost:3000';

export const metadata: Metadata = {
  title: 'Loclean Documentation',
  description: 'Local-first Data Cleaning & Extraction using LLMs',
  metadataBase: new URL(appUrl),
  icons: {
    icon: [
      {
        url: '/logo-dark.svg',
        type: 'image/svg+xml',
      },
    ],
  },
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="en" className={inter.className} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}
