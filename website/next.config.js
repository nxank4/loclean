const withNextra = require('nextra')({
  theme: 'nextra-theme-docs',
  themeConfig: './theme.config.tsx',
  // Custom CSS
  css: ['./styles/custom.css'],
});

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Base path for GitHub Pages
  basePath: '/loclean',
  // Static export for GitHub Pages
  output: 'export',
  // Disable image optimization for static export
  images: {
    unoptimized: true,
  },
  // Trailing slash for GitHub Pages compatibility
  trailingSlash: true,
};

module.exports = withNextra(nextConfig);
