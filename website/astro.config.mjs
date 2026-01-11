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
							
							// Find all images that might be logos
							const allImages = document.querySelectorAll('img');
							
							allImages.forEach((img) => {
								let src = img.getAttribute('src') || img.src || '';
								const alt = (img.getAttribute('alt') || '').toLowerCase();
								
								// Check if this is a logo image
								const isLogo = src.includes('loclean-logo') || 
								               alt.includes('logo') || 
								               alt.includes('loclean');
								
								if (!isLogo) return;
								
								// Handle Astro image optimization URLs (_image?href=...)
								if (src.includes('_image?href=')) {
									// Extract the actual file path from the href parameter
									const hrefMatch = src.match(/href=([^&]+)/);
									if (hrefMatch) {
										const decodedHref = decodeURIComponent(hrefMatch[1]);
										// Check if it contains logo path
										if (decodedHref.includes('loclean-logo')) {
											let newHref = decodedHref;
											if (theme === 'dark' && decodedHref.includes('for-light')) {
												newHref = decodedHref.replace('for-light', 'for-dark');
											} else if (theme === 'light' && decodedHref.includes('for-dark')) {
												newHref = decodedHref.replace('for-dark', 'for-light');
											}
											
											if (newHref !== decodedHref) {
												// Reconstruct the URL with new href
												const url = new URL(src, window.location.origin);
												url.searchParams.set('href', encodeURIComponent(newHref));
												img.src = url.toString();
												img.setAttribute('src', img.src);
											}
										}
									}
								} else {
									// Handle regular image URLs
									// Extract base path (before query params)
									const baseSrc = src.split('?')[0];
									const queryString = src.includes('?') ? src.split('?').slice(1).join('?') : '';
									
									if (theme === 'dark' && baseSrc.includes('for-light')) {
										const newSrc = baseSrc.replace('for-light', 'for-dark');
										img.src = queryString ? newSrc + '?' + queryString : newSrc;
										img.setAttribute('src', img.src);
									} else if (theme === 'light' && baseSrc.includes('for-dark')) {
										const newSrc = baseSrc.replace('for-dark', 'for-light');
										img.src = queryString ? newSrc + '?' + queryString : newSrc;
										img.setAttribute('src', img.src);
									}
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
						updateLogos();
						markOutputBlocks();
						
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
							document.addEventListener('DOMContentLoaded', () => {
								updateLogos();
								markOutputBlocks();
							});
						}
						
						// Run after a short delay to catch dynamically loaded content
						setTimeout(() => {
							updateLogos();
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
