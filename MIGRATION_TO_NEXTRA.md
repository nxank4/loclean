# Migration Plan: Astro/Starlight → Nextra

## Overview
Migrate documentation from Astro/Starlight to Nextra framework.

## Current Structure
- **Framework**: Astro with Starlight theme
- **Content**: MDX files in `website/src/content/docs/`
- **Assets**: Images/logos in `website/src/assets/`
- **Config**: `website/astro.config.mjs`
- **Base URL**: `/loclean` (for GitHub Pages)

## Migration Steps

### 1. Initialize Nextra Project
- [ ] Create new Next.js project structure
- [ ] Install Nextra dependencies:
  ```bash
  npm install nextra nextra-theme-docs next react react-dom
  ```
- [ ] Set up TypeScript configuration
- [ ] Create `next.config.js` with Nextra configuration

### 2. Project Structure Setup
```
website/
├── pages/
│   ├── _app.tsx          # Nextra app wrapper
│   ├── _meta.json        # Root metadata
│   └── [...slug].tsx      # Dynamic routes
├── public/               # Static assets
├── theme.config.tsx      # Nextra theme config
├── next.config.js        # Next.js + Nextra config
└── package.json
```

### 3. Content Migration
- [ ] Migrate content from `src/content/docs/` to `pages/`
- [ ] Convert Astro frontmatter to Nextra metadata
- [ ] Update internal links (Astro → Next.js routing)
- [ ] Preserve all existing content structure:
  - Getting Started
  - Guides
  - Concepts
  - API Reference

### 4. Configuration
- [ ] Set up `theme.config.tsx` with:
  - Logo and branding
  - Navigation structure
  - GitHub link
  - Base path for GitHub Pages (`/loclean`)
- [ ] Configure `next.config.js`:
  - Base path: `/loclean`
  - Output: `export` (static export for GitHub Pages)
  - Image optimization settings

### 5. Assets Migration
- [ ] Move assets from `src/assets/` to `public/`
- [ ] Update image references in content
- [ ] Preserve favicon and logos

### 6. Styling
- [ ] Migrate custom CSS from `src/styles/custom.css`
- [ ] Adapt styles for Nextra theme
- [ ] Test dark/light mode compatibility

### 7. Build & Deploy
- [ ] Update build scripts in `package.json`
- [ ] Test static export: `npm run build`
- [ ] Update CI/CD workflow (`.github/workflows/`)
- [ ] Verify GitHub Pages deployment

### 8. Testing
- [ ] Test all pages load correctly
- [ ] Verify navigation works
- [ ] Check internal/external links
- [ ] Test search functionality
- [ ] Verify responsive design

## Key Differences: Astro → Nextra

| Feature | Astro/Starlight | Nextra |
|---------|----------------|--------|
| Framework | Astro | Next.js |
| Content Location | `src/content/docs/` | `pages/` |
| Config File | `astro.config.mjs` | `theme.config.tsx` + `next.config.js` |
| Routing | File-based (Astro) | File-based (Next.js) |
| Static Export | Built-in | `output: 'export'` in next.config.js |

## Notes
- Nextra uses Next.js routing, so file structure in `pages/` determines URLs
- Frontmatter format is similar but may need adjustments
- Nextra has built-in search, similar to Starlight
- Theme customization is done via `theme.config.tsx`

## Branch
Current branch: `docs/migrate-to-nextra`
