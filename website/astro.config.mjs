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
							
							// Update all logo images - use more specific selectors
							const logoSelectors = [
								'.starlight-logo img',
								'.hero-image img',
								'header img[alt*="logo" i]',
								'header img[alt*="Loclean" i]',
								'a[href="/loclean/"] img',
								'.title-wrapper img',
								'img[src*="loclean-logo"]'
							];
							
							logoSelectors.forEach((selector) => {
								const logos = document.querySelectorAll(selector);
								logos.forEach((img) => {
									let src = img.getAttribute('src') || img.src;
									// Handle Astro's asset processing with query params
									src = src.split('?')[0];
									
									if (theme === 'dark' && src.includes('for-light')) {
										const newSrc = src.replace('for-light', 'for-dark');
										// Preserve query params if any
										const query = (img.getAttribute('src') || img.src).split('?')[1];
										img.src = query ? newSrc + '?' + query : newSrc;
										img.setAttribute('src', img.src);
									} else if (theme === 'light' && src.includes('for-dark')) {
										const newSrc = src.replace('for-dark', 'for-light');
										const query = (img.getAttribute('src') || img.src).split('?')[1];
										img.src = query ? newSrc + '?' + query : newSrc;
										img.setAttribute('src', img.src);
									}
								});
							});
						}
						
						// Run immediately
						updateLogos();
						
						// Watch for theme changes
						const observer = new MutationObserver(() => {
							setTimeout(updateLogos, 50);
						});
						observer.observe(document.documentElement, { 
							attributes: true, 
							attributeFilter: ['data-theme'] 
						});
						
						// Also watch for theme toggle clicks
						document.addEventListener('click', (e) => {
							if (e.target.closest('[data-theme-toggle]') || 
								e.target.closest('button[aria-label*="theme" i]') ||
								e.target.closest('[aria-label*="theme" i]')) {
								setTimeout(updateLogos, 150);
							}
						});
						
						// Run on DOM ready
						if (document.readyState === 'loading') {
							document.addEventListener('DOMContentLoaded', updateLogos);
						}
						
						// Run after a short delay to catch dynamically loaded content
						setTimeout(updateLogos, 500);
					`,
				},
			],
		}),
	],
});
