<script lang="ts">
    import Checkbox from '$lib/components/ui/Checkbox.svelte';

    // --- Types ---

    type Quality = 'auto' | 'high' | 'low';
    type Effort = 'low' | 'medium' | 'high';

    export interface CaptionOptionsDiffDefaults {
        maxTokens: number | null;
        imageQuality: Quality | null;
        rounds: number | null;
        overwrite: boolean;
        reasoningEnabled: boolean;
        reasoningEffort: Effort;
        draftName: string;
        storeConversation: boolean;
        reasoningExcludeOutput: boolean;
    }

    // --- Display config (read-only, grouped into a single prop) ---

    export interface CaptionOptionsDisplay {
        /** Prefix for form element IDs. Default: "caption" */
        idPrefix?: string;
        /** When true, empty number inputs set value to null ("not in TOML" semantics). Default: false */
        nullable?: boolean;
        /** When true, show help text under fields. Default: false */
        helpText?: boolean;
        /** Show the draft name field. Default: false */
        showDraftName?: boolean;
        /** Show the store conversation checkbox. Default: false */
        showStoreConversation?: boolean;
        /** Show the reasoning exclude-from-output checkbox. Default: false */
        showReasoningExcludeOutput?: boolean;
        /** If provided, show diff dots when values differ from these defaults. */
        diffDefaults?: Partial<CaptionOptionsDiffDefaults>;
    }

    // --- Props ---

    interface Props {
        /** Max tokens the model can output. Nullable in config-editing mode. */
        maxTokens?: number | null;
        /** Image upload quality. Nullable in config-editing mode. */
        imageQuality?: Quality | null;
        /** Number of captioning rounds. Nullable in config-editing mode. */
        rounds?: number | null;
        /** Whether to overwrite existing captions. */
        overwrite?: boolean;
        /** Enable chain-of-thought reasoning. */
        reasoningEnabled?: boolean;
        /** Reasoning thinking effort level. */
        reasoningEffort?: Effort;
        /** Draft name for caption drafts. Shown when display.showDraftName is true. */
        draftName?: string;
        /** Store conversation history. Shown when display.showStoreConversation is true. */
        storeConversation?: boolean;
        /** Exclude reasoning from output. Shown when display.showReasoningExcludeOutput is true. */
        reasoningExcludeOutput?: boolean;
        /** Display configuration — diff dots, help text, field visibility, etc. */
        display?: CaptionOptionsDisplay;
    }

    let {
        maxTokens = $bindable(512),
        imageQuality = $bindable('auto'),
        rounds = $bindable(1),
        overwrite = $bindable(false),
        reasoningEnabled = $bindable(false),
        reasoningEffort = $bindable('low'),
        draftName = $bindable(''),
        storeConversation = $bindable(false),
        reasoningExcludeOutput = $bindable(true),
        display = {}
    }: Props = $props();

    // --- Derived display config ---

    const pfx = $derived(display.idPrefix ?? 'caption');
    const nullable = $derived(display.nullable ?? false);
    const help = $derived(display.helpText ?? false);
    const diffs = $derived(display.diffDefaults);

    // --- Diff dot helpers ---

    /** Check if a field's current value differs from its diff default. */
    function isOverridden(field: keyof CaptionOptionsDiffDefaults): boolean {
        if (!diffs) {
            return false;
        }

        // Special case: effort only counts as overridden when reasoning is enabled
        if (field === 'reasoningEffort') {
            return reasoningEnabled && reasoningEffort !== diffs.reasoningEffort;
        }

        const current: Record<string, unknown> = {
            maxTokens,
            imageQuality,
            rounds,
            overwrite,
            reasoningEnabled,
            reasoningEffort,
            draftName,
            storeConversation,
            reasoningExcludeOutput
        };
        return current[field] !== diffs[field];
    }

    // --- Input handlers ---

    function handleNullableNumber(e: Event, setter: (v: number | null) => void) {
        const raw = (e.target as HTMLInputElement).value;
        setter(raw === '' ? (nullable ? null : 0) : Number(raw));
    }

    function handleQualityChange(e: Event) {
        const raw = (e.target as HTMLSelectElement).value;
        imageQuality = (raw === '' ? null : raw) as Quality | null;
    }
</script>

<!-- ═══ Section: Options ═══ -->
<section class="space-y-3">
    <h3 class="section-heading">Options</h3>

    <div class="grid grid-cols-2 gap-3">
        <!-- Max Tokens -->
        <div>
            <div class="mb-1 flex items-center gap-1.5">
                <label class="label mb-0" for="{pfx}-tokens">Max Tokens</label>
                {#if isOverridden('maxTokens')}
                    <span class="diff-dot" title="Differs from dataset config"></span>
                {/if}
            </div>
            <input
                id="{pfx}-tokens"
                type="number"
                value={maxTokens ?? ''}
                min={100}
                max={16384}
                class="input"
                placeholder={nullable ? '512 (default)' : undefined}
                oninput={(e) => handleNullableNumber(e, (v) => (maxTokens = v))}
            />
            {#if help}
                <p class="help-text">
                    Maximum tokens the model can output.{nullable
                        ? ' Clear to use default (512).'
                        : ''}
                </p>
            {/if}
        </div>

        <!-- Image Quality -->
        <div>
            <div class="mb-1 flex items-center gap-1.5">
                <label class="label mb-0" for="{pfx}-quality">Image Quality</label>
                {#if isOverridden('imageQuality')}
                    <span class="diff-dot" title="Differs from dataset config"></span>
                {/if}
            </div>
            <select
                id="{pfx}-quality"
                class="input cursor-pointer"
                value={imageQuality ?? ''}
                onchange={handleQualityChange}
            >
                {#if nullable}
                    <option value="">Auto (default)</option>
                {/if}
                <option value="auto">Auto</option>
                <option value="high">High</option>
                <option value="low">Low</option>
            </select>
            {#if help}
                <p class="help-text">
                    Quality of images sent to the model. Higher quality uses more tokens.
                </p>
            {/if}
        </div>

        <!-- Draft Name (CaptionSettings only) -->
        {#if display.showDraftName}
            <div>
                <div class="mb-1 flex items-center gap-1.5">
                    <label class="label mb-0" for="{pfx}-draft">Draft Name</label>
                    {#if isOverridden('draftName')}
                        <span class="diff-dot" title="Differs from dataset config"></span>
                    {/if}
                </div>
                <input
                    id="{pfx}-draft"
                    type="text"
                    bind:value={draftName}
                    class="input"
                    placeholder="(none — write caption directly)"
                />
            </div>
        {/if}

        <!-- Rounds -->
        <div>
            <div class="mb-1 flex items-center gap-1.5">
                <label class="label mb-0" for="{pfx}-rounds">Rounds</label>
                {#if isOverridden('rounds')}
                    <span class="diff-dot" title="Differs from dataset config"></span>
                {/if}
            </div>
            <input
                id="{pfx}-rounds"
                type="number"
                value={rounds ?? ''}
                min={1}
                max={10}
                class="input"
                placeholder={nullable ? '1 (default)' : undefined}
                oninput={(e) => handleNullableNumber(e, (v) => (rounds = v))}
            />
            {#if help}
                <p class="help-text">
                    Multiple rounds generate captions, then a final round refines them.{nullable
                        ? ' Clear to use default (1).'
                        : ''}
                </p>
            {/if}
        </div>
    </div>

    <!-- Checkboxes -->
    <div class="space-y-2">
        <div>
            <div class="flex items-center gap-2">
                <Checkbox id="{pfx}-overwrite" bind:checked={overwrite} />
                <label class="cursor-pointer text-sm text-gray-300" for="{pfx}-overwrite">
                    Overwrite existing captions
                </label>
                {#if isOverridden('overwrite')}
                    <span class="diff-dot" title="Differs from dataset config"></span>
                {/if}
            </div>
            {#if help}
                <p class="help-text ml-6">
                    When disabled, images with existing captions are skipped.
                </p>
            {/if}
        </div>

        {#if display.showStoreConversation}
            <div>
                <div class="flex items-center gap-2">
                    <Checkbox id="{pfx}-store-conversation" bind:checked={storeConversation} />
                    <label
                        class="cursor-pointer text-sm text-gray-300"
                        for="{pfx}-store-conversation"
                    >
                        Store conversation history
                    </label>
                </div>
                {#if help}
                    <p class="help-text ml-6">
                        Let the API provider store conversations for later review. Not recommended
                        for large datasets.
                    </p>
                {/if}
            </div>
        {/if}
    </div>
</section>

<!-- ═══ Section: Reasoning ═══ -->
<section class="space-y-3">
    <div class="flex items-center gap-2">
        <Checkbox id="{pfx}-reasoning" bind:checked={reasoningEnabled} />
        <label class="section-heading cursor-pointer" for="{pfx}-reasoning"> Reasoning </label>
        {#if isOverridden('reasoningEnabled')}
            <span class="diff-dot" title="Differs from dataset config"></span>
        {/if}
    </div>
    {#if help}
        <p class="help-text">
            Enables chain-of-thought reasoning for models that support it. Improves output quality
            at the cost of inference time.
        </p>
    {/if}

    {#if reasoningEnabled}
        <div class={display.showReasoningExcludeOutput ? 'grid grid-cols-2 gap-3' : ''}>
            <div>
                <div class="mb-1 flex items-center gap-1.5">
                    <label class="label mb-0" for="{pfx}-effort">Thinking Effort</label>
                    {#if isOverridden('reasoningEffort')}
                        <span class="diff-dot" title="Differs from dataset config"></span>
                    {/if}
                </div>
                <select id="{pfx}-effort" class="input cursor-pointer" bind:value={reasoningEffort}>
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                </select>
                {#if help}
                    <p class="help-text">
                        How much thinking the model does. Start with low and increase if needed.
                    </p>
                {/if}
            </div>

            {#if display.showReasoningExcludeOutput}
                <div class="flex items-end pb-1">
                    <div>
                        <div class="flex items-center gap-2">
                            <Checkbox
                                id="{pfx}-exclude-output"
                                bind:checked={reasoningExcludeOutput}
                            />
                            <label
                                class="cursor-pointer text-sm text-gray-300"
                                for="{pfx}-exclude-output"
                            >
                                Exclude from output
                            </label>
                        </div>
                        {#if help}
                            <p class="help-text ml-6">
                                Hide the reasoning section from the caption output.
                            </p>
                        {/if}
                    </div>
                </div>
            {/if}
        </div>
    {/if}
</section>
