<script lang="ts">
    import Topbar from '$lib/components/ui/Topbar.svelte';
    import { PromptGenerator } from '$lib/components/prompts';
</script>

<Topbar>
    <div class="min-w-0 flex-1">
        <h1 class="text-xl font-bold text-white">Prompt Generator</h1>
        <p class="mt-0.5 text-sm text-gray-400">
            Generate Jinja2 prompt templates from an intent and optional few-shot examples.
        </p>
    </div>
</Topbar>

<div class="h-[calc(100dvh-7.5rem)]">
    <PromptGenerator />
</div>
