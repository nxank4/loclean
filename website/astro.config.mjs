// @ts-check
import starlight from '@astrojs/starlight';
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
	// QUAN TRỌNG: Cấu hình cho GitHub Pages
	site: 'https://nxank4.github.io',
	base: '/loclean',
	integrations: [
		starlight({
			title: 'Loclean',
			logo: {
				src: './src/assets/loclean-logo-for-light.svg',
				alt: 'Loclean logo',
			},
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/nxank4/loclean',
				},
			],
			sidebar: [
				{
					label: '🚀 Getting Started',
					autogenerate: { directory: 'getting-started' },
				},
				{
					label: '📘 User Guide',
					autogenerate: { directory: 'guides' },
				},
				{
					label: '🧠 Concepts',
					autogenerate: { directory: 'concepts' },
				},
				{
					label: '🔬 API Reference',
					autogenerate: { directory: 'reference' },
				},
			],
			customCss: ['./src/styles/custom.css'],
		}),
	],
});
