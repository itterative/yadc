<script lang="ts">
  import type { Snippet } from "svelte";
  import "./layout.css";

  interface Props {
    children: Snippet;
  }

  let { children }: Props = $props();
</script>

<div class="app-shell">
  <nav class="app-nav">
    <div class="nav-brand">
      <a href="/">yadc</a>
    </div>
    <div class="nav-links">
      <a href="/">Datasets</a>
    </div>
  </nav>

  <main class="app-main">
    {@render children()}
  </main>
</div>

<style>
  .app-shell {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
  }

  .app-nav {
    display: flex;
    align-items: center;
    gap: 2rem;
    padding: 0.75rem 1.5rem;
    background: var(--color-surface);
    border-bottom: 1px solid var(--color-border);
  }

  .nav-brand a {
    font-weight: 700;
    font-size: 1.2rem;
    color: var(--color-accent);
    text-decoration: none;
  }

  .nav-links {
    display: flex;
    gap: 1rem;
  }

  .nav-links a {
    color: var(--color-text-dim);
    text-decoration: none;
    transition: color 0.2s;
  }

  .nav-links a:hover {
    color: var(--color-text);
  }

  .app-main {
    flex: 1;
    padding: 1.5rem;
  }
</style>
