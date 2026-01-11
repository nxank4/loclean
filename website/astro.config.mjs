// @ts-check
import starlight from '@astrojs/starlight';
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
	// QUAN TRỌNG: Cấu hình cho GitHub Pages
	site: 'https://nxank4.github.io',
	base: '/loclean',
	integrations: [
		expressiveCode(),
		starlight({
			title: 'Loclean',
			logo: {
				src: './src/assets/loclean-logo-for-light.svg',
				alt: 'Loclean logo',
				replacesTitle: true,
			},
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/nxank4/loclean',
				},
			],
			editLink: {
				baseUrl: 'https://github.com/nxank4/loclean/edit/main/website/src/content/docs',
			},
			lastUpdated: true,
			pagination: true,
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
			head: [
				{
					tag: 'script',
					content: `
						function updateLogos() {
							const theme = document.documentElement.getAttribute('data-theme') || 
								(window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
							const logos = document.querySelectorAll('.starlight-logo img, .hero-image img, [class*="hero"] img');
							logos.forEach((img) => {
								const src = img.getAttribute('src') || img.src;
								if (theme === 'dark' && src.includes('for-light')) {
									img.src = src.replace('for-light', 'for-dark');
								} else if (theme === 'light' && src.includes('for-dark')) {
									img.src = src.replace('for-dark', 'for-light');
								}
							});
						}
						const observer = new MutationObserver(updateLogos);
						observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
						if (document.readyState === 'loading') {
							document.addEventListener('DOMContentLoaded', updateLogos);
						} else {
							updateLogos();
						}
						document.addEventListener('click', (e) => {
							if (e.target.closest('[data-theme-toggle]') || e.target.closest('button[aria-label*="theme"]')) {
								setTimeout(updateLogos, 100);
							}
						});
					`,
				},
			],
		}),
	],
});
