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
};

export default config;
