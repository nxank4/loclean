import { DocsThemeConfig } from 'nextra-theme-docs';

const config: DocsThemeConfig = {
  logo: (
    <span>
      <img
        src="/loclean/loclean-logo-for-light.svg"
        alt="Loclean logo"
        style={{ height: '24px', marginRight: '8px', verticalAlign: 'middle' }}
      />
      <span style={{ fontWeight: 600 }}>Loclean</span>
    </span>
  ),
  project: {
    link: 'https://github.com/nxank4/loclean',
  },
  docsRepositoryBase: 'https://github.com/nxank4/loclean/tree/main/website',
  footer: {
    content: 'Loclean Documentation © 2024',
  },
  sidebar: {
    defaultMenuCollapseLevel: 1,
  },
  search: {
    placeholder: 'Search documentation...',
  },
  darkMode: true,
  nextThemes: {
    defaultTheme: 'system',
  },
  toc: {
    backToTop: true,
  },
  gitTimestamp: ({ timestamp }) => (
    <>Last updated on {timestamp.toLocaleDateString()}</>
  ),
  head: (
    <>
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <meta property="og:title" content="Loclean Documentation" />
      <meta property="og:description" content="Local-first Semantic Data Cleaning & Extraction library for Python" />
    </>
  ),
  navigation: {
    prev: true,
    next: true,
  },
};

export default config;
