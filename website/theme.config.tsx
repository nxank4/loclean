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
    text: 'Loclean Documentation © 2024',
  },
  useNextSeoProps() {
    return {
      titleTemplate: '%s – Loclean',
    };
  },
  sidebar: {
    defaultMenuCollapseLevel: 1,
    titleComponent: ({ title, type }) => {
      if (type === 'separator') {
        return <span className="cursor-default">{title}</span>;
      }
      return <>{title}</>;
    },
  },
  search: {
    placeholder: 'Search documentation...',
  },
  primaryHue: 217, // Blue color
  primarySaturation: 100,
};

export default config;
