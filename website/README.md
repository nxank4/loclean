# Loclean Documentation

Built with [Nextra](https://nextra.site/) - a Next.js-based documentation framework.

## 🚀 Project Structure

```
.
├── pages/              # Documentation pages (MDX files)
│   ├── _meta.ts       # Navigation metadata
│   ├── getting-started/
│   ├── guides/
│   ├── concepts/
│   └── reference/
├── public/            # Static assets (images, logos, favicon)
├── styles/            # Custom CSS
├── theme.config.tsx   # Nextra theme configuration
├── next.config.mjs    # Next.js + Nextra configuration
└── package.json
```

## 🧞 Commands

All commands are run from the `website/` directory:

| Command                   | Action                                           |
| :------------------------ | :----------------------------------------------- |
| `npm install`             | Installs dependencies                            |
| `npm run dev`             | Starts local dev server at `localhost:3000`      |
| `npm run build`           | Build your production site to `./out/`          |
| `npm run start`           | Start production server (after build)            |

## 📝 Content

Documentation pages are written in MDX format and located in the `pages/` directory. The file structure determines the URL structure.

Navigation is configured via `_meta.ts` files in each directory.

## 🎨 Customization

- **Theme**: Edit `theme.config.tsx` for logo, colors, and navigation
- **Styles**: Custom CSS in `styles/custom.css`
- **Configuration**: Next.js config in `next.config.mjs`

## 📦 Deployment

The site is configured for static export to GitHub Pages:
- Base path: `/loclean`
- Output directory: `out/`
- Static export enabled

## 👀 Want to learn more?

Check out [Nextra documentation](https://nextra.site/) or [Next.js documentation](https://nextjs.org/docs).
