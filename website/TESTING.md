# Testing Nextra Documentation Locally

## Quick Start

### 1. Install Dependencies

```bash
cd website
npm install
```

### 2. Run Development Server

```bash
npm run dev
```

This will start the Nextra dev server at `http://localhost:3000/loclean`

> **Note**: The base path is `/loclean` for GitHub Pages compatibility. Make sure to access `http://localhost:3000/loclean` (not just `http://localhost:3000`)

### 3. Test Production Build

```bash
npm run build
```

This will:
- Build the static site
- Output to `website/out/` directory
- Generate all 19 pages

### 4. Preview Production Build (Optional)

After building, you can preview the production build:

```bash
npm run start
```

Then visit `http://localhost:3000/loclean`

## What to Check

### ✅ Navigation
- [ ] All sections appear in sidebar (Getting Started, Guides, Concepts, Reference)
- [ ] All pages are accessible
- [ ] Links work correctly

### ✅ Content
- [ ] All pages render correctly
- [ ] Code blocks display properly
- [ ] Images load (logos, favicon)
- [ ] Tables format correctly

### ✅ Styling
- [ ] Dark/light mode works
- [ ] Logo displays correctly
- [ ] Custom CSS applies
- [ ] Responsive design works

### ✅ Build
- [ ] `npm run build` succeeds without errors
- [ ] All 19 pages generate in `out/` directory
- [ ] No console errors in browser

## Troubleshooting

### Port Already in Use
If port 3000 is busy:
```bash
npm run dev -- -p 3001
```

### Build Errors
1. Clear `.next` cache:
   ```bash
   rm -rf .next
   npm run build
   ```

2. Reinstall dependencies:
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   ```

### Missing Assets
- Check that `public/` contains all logos and images
- Verify paths in `theme.config.tsx` use `/loclean/` prefix

## Before PR

1. ✅ Run `npm run dev` and test all pages
2. ✅ Run `npm run build` and verify no errors
3. ✅ Check that `out/` directory contains all pages
4. ✅ Test navigation and links
5. ✅ Verify responsive design
