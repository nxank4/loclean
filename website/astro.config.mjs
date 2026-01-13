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
						function updateSidebarLogo() {
							const theme = document.documentElement.getAttribute('data-theme') || 
								(window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
							
							// Only update sidebar logo (header logo), not hero images (handled by Starlight)
							const sidebarLogos = document.querySelectorAll('header img[alt*="logo" i], .starlight-logo img, .title-wrapper img, a[href="/loclean/"] img');
							
							sidebarLogos.forEach((img) => {
								let src = img.getAttribute('src') || img.src || '';
								
								// Skip Astro image optimization URLs (they cause errors)
								if (src.includes('_image?href=') || src.includes('/@fs/')) {
									// For dev server URLs with /@fs/, try to update the path
									if (src.includes('/@fs/')) {
										const basePath = src.split('?')[0];
										const queryString = src.includes('?') ? '?' + src.split('?').slice(1).join('?') : '';
										
										if (theme === 'dark' && basePath.includes('for-light')) {
											const newPath = basePath.replace('for-light', 'for-dark');
											img.src = newPath + queryString;
											img.setAttribute('src', img.src);
										} else if (theme === 'light' && basePath.includes('for-dark')) {
											const newPath = basePath.replace('for-dark', 'for-light');
											img.src = newPath + queryString;
											img.setAttribute('src', img.src);
										}
									}
									return;
								}
								
								// Only process if it's a logo file
								if (!src.includes('loclean-logo')) {
									return;
								}
								
								// Extract base path (before query params)
								const baseSrc = src.split('?')[0];
								const queryString = src.includes('?') ? '?' + src.split('?').slice(1).join('?') : '';
								
								if (theme === 'dark' && baseSrc.includes('for-light')) {
									const newSrc = baseSrc.replace('for-light', 'for-dark');
									img.src = newSrc + queryString;
									img.setAttribute('src', img.src);
								} else if (theme === 'light' && baseSrc.includes('for-dark')) {
									const newSrc = baseSrc.replace('for-dark', 'for-light');
									img.src = newSrc + queryString;
									img.setAttribute('src', img.src);
								}
							});
						}
						
						function markOutputBlocks() {
							// Find all paragraphs containing "Output:" text
							const paragraphs = Array.from(document.querySelectorAll('.sl-markdown-content p'));
							
							paragraphs.forEach((p) => {
								const text = p.textContent || '';
								if (text.includes('Output:') || text.includes('**Output:**')) {
									// Find the next sibling that is a code block (Expressive Code or regular pre)
									let nextSibling = p.nextElementSibling;
									
									// Check for Expressive Code blocks (figure.expressive-code)
									if (nextSibling && nextSibling.classList.contains('expressive-code')) {
										nextSibling.setAttribute('data-output', 'true');
										nextSibling.classList.add('output-block');
									}
									// Check for regular pre blocks
									else if (nextSibling && nextSibling.tagName === 'PRE') {
										nextSibling.setAttribute('data-output', 'true');
										nextSibling.classList.add('output-block');
									}
								}
							});
						}
						
						// Run immediately
						updateSidebarLogo();
						markOutputBlocks();
						
						// Watch for theme changes
						const observer = new MutationObserver(() => {
							setTimeout(updateSidebarLogo, 50);
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
								setTimeout(updateSidebarLogo, 150);
							}
						});
						
						// Run on DOM ready
						if (document.readyState === 'loading') {
							document.addEventListener('DOMContentLoaded', () => {
								updateSidebarLogo();
								markOutputBlocks();
							});
						}
						
						// Run after a short delay to catch dynamically loaded content
						setTimeout(() => {
							updateSidebarLogo();
							markOutputBlocks();
						}, 500);
						
						// Watch for new content being added (for SPA navigation)
						const contentObserver = new MutationObserver(() => {
							markOutputBlocks();
						});
						contentObserver.observe(document.body, {
							childList: true,
							subtree: true
						});
					`,
				},
			],
		}),
	],
});
