<script lang="ts">
    import { generation, promptSettings, startGeneration } from '$lib/stores/prompts';
    import type {
        ExamplePair,
        PromptGenFocus,
        PromptGenMode,
        PromptImageQuality
    } from '$lib/stores/prompts';
    import { EditTemplateDialog } from '$lib/components/templates';
    import GenerationPreview from './GenerationPreview.svelte';
    import PromptSidePanel from './PromptSidePanel.svelte';

    // Form state ownership stays here so the persistence `$effect`
    // (mirrors every field into `promptSettings` on change) lives
    // alongside `EditTemplateDialog` — the Save-as-template flow,
    // which is triggered from the preview. The side panel below
    // binds these fields through itself to `PromptForm` and
    // `PromptSettings`; mutations flow back here as the single
    // source of truth.
    let examples = $state<ExamplePair[]>([]);

    const _initial = $promptSettings;
    let mode = $state<PromptGenMode>(_initial.mode);
    let env = $state(_initial.env);
    let apiUrl = $state(_initial.apiUrl);
    let apiToken = $state(_initial.apiToken);
    let apiModelName = $state(_initial.apiModelName);
    let intent = $state(_initial.intent);
    let focus = $state<PromptGenFocus>(_initial.focus);
    let maxTokens = $state(_initial.maxTokens);
    let imageQuality = $state<PromptImageQuality>(_initial.imageQuality);

    // Refine-mode-only. NOT persisted — on reload the user re-selects
    // a template (or pastes their own). The picker selection is also
    // ephemeral and lives inside the form.
    let templateContent = $state('');

    // Save-as-template dialog (triggered from the preview's header
    // button, not from the side panel — so this state lives here).
    let saveDialogOpen = $state(false);
    let saveBody = $state('');

    $effect(() => {
        promptSettings.update((s) => ({
            ...s,
            mode,
            env,
            apiUrl,
            apiToken,
            apiModelName,
            intent,
            focus,
            maxTokens,
            imageQuality
        }));
    });

    function handleChange(field: string, value: unknown) {
        // Hook for future side-effects. Persistence is handled by the
        // ``$effect`` above.
        void field;
        void value;
    }

    // Mirror the panel's button-disabled derivation as a defense-in-depth
    // guard here (the side panel derives its own copy to drive the
    // Generate button). Keeps `handleGenerate` safe to call directly
    // without going through the button.
    let isStreaming = $derived(generation.status === 'streaming');
    let canGenerate = $derived.by(() => {
        if (isStreaming || env.trim().length === 0 || intent.trim().length === 0) {
            return false;
        }
        return true;
    });

    async function handleGenerate() {
        if (!canGenerate) {
            return;
        }
        // Snapshot form values so the streaming run uses them
        // even if the user edits them mid-stream.
        await startGeneration({
            env,
            intent,
            examples: examples.map((e) => ({ ...e })),
            focus,
            apiModelName: apiModelName || null,
            maxTokens,
            imageQuality,
            templateContent: mode === 'refine' ? templateContent : null
        });
    }

    function handleSaveTemplate(body: string) {
        saveBody = body;
        saveDialogOpen = true;
    }

    function handleTemplateSaved(name: string) {
        void name;
        saveDialogOpen = false;
    }
</script>

<!-- Layout: full-page GenerationPreview on the left, side panel
     (form / settings / history tabs + Generate / Save-to-history
     actions) on the right. The side panel handles its own width
     via the generic `ui/SidePanel` primitive — desktop inline,
     mobile slide-over drawer. ``min-w-0`` on the preview wrapper
     is required — without it, the flex item's min-width is
     content-based, so unbreakable content (long reasoning line,
     wide table, …) would push the side panel rightward off the
     allotted space. Mirrors the ``min-w-0`` on
     ``routes/datasets/[name]/DropUploadZone.svelte``. -->
<div class="flex h-full min-h-0 gap-4">
    <div class="min-h-0 min-w-0 flex-1">
        <GenerationPreview onsavetemplate={handleSaveTemplate} />
    </div>
    <PromptSidePanel
        bind:mode
        bind:intent
        bind:focus
        bind:examples
        bind:templateContent
        bind:env
        bind:apiUrl
        bind:apiToken
        bind:apiModelName
        bind:maxTokens
        bind:imageQuality
        onchange={handleChange}
        ongenerate={handleGenerate}
    />
</div>

<EditTemplateDialog
    open={saveDialogOpen}
    templateName={null}
    initialContent={saveBody}
    onclose={() => (saveDialogOpen = false)}
    onsaved={handleTemplateSaved}
/>
