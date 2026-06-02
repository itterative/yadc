<script>
	import { browser } from '$app/environment';
	import { onMount } from 'svelte';
	import { on } from 'svelte/events';

	let { once = false, top = 0, bottom = 0, left = 0, right = 0, onintersect = () => {} } = $props();

	let fired = false;
	/** @type {HTMLElement} */
	let container;

	/**
	 * @param cb {() => void}
	 */
	function intersect(cb) {
		const shouldCleanup = !fired && once;

		if (!once || (!fired && once)) {
			onintersect();
		}

		fired = true;

		if (shouldCleanup) {
			cb();
		}
	}

	onMount(() => {
		if (!browser) {
			return;
		}

		if (typeof IntersectionObserver !== 'undefined') {
			const rootMargin = `${bottom}px ${left}px ${top}px ${right}px`;

			const observer = new IntersectionObserver(
				(entries) => {
					const intersecting = entries[0].isIntersecting;

					if (!intersecting) {
						return;
					}

					intersect(() => {
						observer.unobserve(container);
					});
				},
				{
					rootMargin
				}
			);

			observer.observe(container);
			return () => observer.unobserve(container);
		}

		let outOfView = false;

		// Fallback for browsers without IntersectionObserver
		function handler() {
			const bcr = container.getBoundingClientRect();
			const intersecting =
				bcr.bottom + bottom > 0 &&
				bcr.right + right > 0 &&
				bcr.top - top < window.innerHeight &&
				bcr.left - left < window.innerWidth;

			if (!intersecting) {
				outOfView = true;
				return;
			}

			if (outOfView) {
				outOfView = false;
				intersect(() => {
					window.removeEventListener('scroll', handler);
				});
			}
		}

		window.requestAnimationFrame(() => handler());
		return on(window, 'scroll', handler);
	});
</script>

<div class="clear-none h-0 w-0" bind:this={container}></div>
