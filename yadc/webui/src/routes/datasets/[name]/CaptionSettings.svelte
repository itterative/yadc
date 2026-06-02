<script lang="ts">
  import EnvManager from "$lib/components/dialogs/EnvManager.svelte";
  import EnvSelector from "$lib/components/settings/EnvSelector.svelte";
  import JinjaEditor from "$lib/components/ui/JinjaEditor.svelte";
  import Checkbox from "$lib/components/ui/Checkbox.svelte";
  import SvgPlus from "$lib/icons/SvgPlus.svelte";
  import SpinnerBlock from "$lib/components/ui/SpinnerBlock.svelte";
  import {
    fetchTemplates,
    fetchTemplate,
    saveTemplate,
    type TemplateListItem,
  } from "$lib/stores/templates";
  import type { CaptionOptions } from "$lib/stores/captionOptions";

  // --- Props ---

  interface Props {
    /** Which dataset this is for. */
    datasetName: string;
    /** Called when the user clicks "Start Captioning". Receives the assembled options. */
    onstart?: (options: CaptionOptions) => void;
    /** Called when the panel wants to close (e.g. after starting). */
    onclose?: () => void;
  }

  let { datasetName, onstart, onclose }: Props = $props();

  // --- State: Environment (managed by EnvSelector via bindings) ---

  let selectedEnv = $state("default");
  let envUrl = $state("");
  let envToken = $state("");
  let envModelName = $state("");
  let envReloadCounter = $state(0);
  let showEnvManager = $state(false);

  // --- State: Templates ---

  let templateList: TemplateListItem[] = $state([]);
  let selectedTemplate = $state("");
  let templateContent = $state("");
  let templateSource: "user" | "builtin" | "" = $state("");
  let isLoadingTemplate = $state(false);

  let isNewTemplate = $state(false);
  let newTemplateName = $state("");
  let isSavingTemplate = $state(false);
  let templateSaveError: string | null = $state(null);

  // --- State: Options ---

  let maxTokens = $state(512);
  let imageQuality: "auto" | "high" | "low" = $state("auto");
  let draftName = $state("");
  let overwrite = $state(false);
  let rounds = $state(1);

  // --- State: Reasoning ---

  let reasoningEnabled = $state(false);
  let reasoningEffort: "low" | "medium" | "high" = $state("low");

  // --- State: General ---

  let isLoadingTemplates = $state(false);
  let templatesError: string | null = $state(null);

  // --- Computed ---

  /** The model name to send — either the selected dropdown value or the typed value. */
  let effectiveModelName = $derived(envModelName.trim());

  /** Whether the template has unsaved edits (content differs from what was loaded). */
  let templateDirty = $state(false);

  /** The effective template name for the job. */
  let effectiveTemplateName = $derived(
    isNewTemplate ? newTemplateName.trim() : selectedTemplate,
  );

  // --- Load data on mount ---

  let dataLoaded = $state(false);

  $effect(() => {
    if (!dataLoaded) {
      dataLoaded = true;
      loadTemplateList();
    }
  });

  // --- Template loading & selection ---

  async function loadTemplateList() {
    isLoadingTemplates = true;
    templatesError = null;
    try {
      templateList = await fetchTemplates();
      // Auto-select "default" if available
      const hasDefault = templateList.some((t) => t.name === "default");
      if (hasDefault && !selectedTemplate) {
        selectedTemplate = "default";
      }
    } catch (e) {
      templatesError = e instanceof Error ? e.message : "Failed to load templates";
    } finally {
      isLoadingTemplates = false;
    }
  }

  /** Load template content when selection changes. */
  $effect(() => {
    const name = selectedTemplate;
    if (!name || isNewTemplate) return;

    let cancelled = false;
    isLoadingTemplate = true;
    (async () => {
      try {
        const info = await fetchTemplate(name);
        if (cancelled) return;
        templateContent = info.content;
        templateSource = info.source;
        templateDirty = false;
      } catch (e) {
        if (cancelled) return;
        templateContent = "";
        templateSource = "";
      } finally {
        if (!cancelled) isLoadingTemplate = false;
      }
    })();

    return () => { cancelled = true; };
  });

  function handleTemplateChange() {
    templateDirty = false;
    isNewTemplate = false;
    newTemplateName = "";
    templateSaveError = null;
  }

  function startNewTemplate() {
    isNewTemplate = true;
    newTemplateName = "";
    templateContent = "";
    templateSource = "";
    templateDirty = false;
    templateSaveError = null;
  }

  function cancelNewTemplate() {
    isNewTemplate = false;
    newTemplateName = "";
    templateSaveError = null;
    // Re-load current selection
    if (selectedTemplate) {
      const name = selectedTemplate;
      selectedTemplate = "";
      selectedTemplate = name;
    }
  }

  function handleTemplateContentChange(content: string) {
    templateContent = content;
    templateDirty = true;
  }

  async function handleSaveTemplate() {
    const name = newTemplateName.trim();
    if (!name) {
      templateSaveError = "Template name is required";
      return;
    }

    isSavingTemplate = true;
    templateSaveError = null;
    try {
      const info = await saveTemplate(name, templateContent);
      // Switch from "new" to the saved template
      isNewTemplate = false;
      selectedTemplate = info.name;
      templateSource = info.source;
      templateDirty = false;
      await loadTemplateList();
    } catch (e) {
      templateSaveError = e instanceof Error ? e.message : "Failed to save template";
    } finally {
      isSavingTemplate = false;
    }
  }

  // --- Assemble and submit ---

  function handleStart() {
    const options: CaptionOptions = {
      env: selectedEnv,
      api_url: envUrl.trim() || undefined,
      api_token: envToken || undefined,
      api_model_name: effectiveModelName || undefined,
      prompt_template: templateDirty ? templateContent : undefined,
      prompt_name: templateDirty ? undefined : (effectiveTemplateName || undefined),
      max_tokens: maxTokens,
      image_quality: imageQuality,
      draft: draftName.trim() || undefined,
      overwrite: overwrite || undefined,
      reasoning: reasoningEnabled || undefined,
      reasoning_effort: reasoningEnabled ? reasoningEffort : undefined,
    };

    onstart?.(options);
  }
</script>

<!-- Sub-dialogs -->
<EnvManager open={showEnvManager} onclose={() => { showEnvManager = false; envReloadCounter++; }} />

<div class="flex flex-col h-full">
  <!-- Scrollable content -->
  <div class="flex-1 overflow-y-auto p-4 space-y-5">
    <!-- ═══ Section: Environment ═══ -->
    <EnvSelector
      bind:env={selectedEnv}
      bind:apiUrl={envUrl}
      bind:apiToken={envToken}
      bind:apiModelName={envModelName}
      reload={envReloadCounter}
      onmanagerequest={() => (showEnvManager = true)}
    />

    <!-- ═══ Section: Template ═══ -->
    <section class="space-y-3">
      <h3 class="section-heading">Template</h3>

      {#if templatesError}
        <div class="alert-error">{templatesError}</div>
      {/if}

      <div class="flex gap-2 items-end">
        <div class="flex-1">
          {#if isNewTemplate}
            <label class="label" for="new-tpl-name">New Template Name</label>
            <input
              id="new-tpl-name"
              type="text"
              bind:value={newTemplateName}
              class="input"
              placeholder="my-template"
            />
          {:else}
            <label class="label" for="caption-template">Template</label>
            <select
              id="caption-template"
              class="input cursor-pointer"
              bind:value={selectedTemplate}
              onchange={handleTemplateChange}
              disabled={isLoadingTemplates}
            >
              {#each templateList as t (t.name)}
                <option value={t.name}>
                  {t.name}{#if t.source === "builtin"} (built-in){/if}
                </option>
              {/each}
            </select>
          {/if}
        </div>

        <div class="flex gap-1">
          {#if isNewTemplate}
            <button
              class="btn-secondary px-3 py-2"
              onclick={cancelNewTemplate}
            >
              Cancel
            </button>
            <button
              class="btn-primary px-3 py-2"
              onclick={handleSaveTemplate}
              disabled={isSavingTemplate || !newTemplateName.trim()}
            >
              {isSavingTemplate ? "Saving…" : "Save"}
            </button>
          {:else}
            <button
              class="btn-secondary px-3 py-2"
              onclick={startNewTemplate}
              title="Create new template"
            >
              <SvgPlus class="h-4 w-4" />
            </button>
            {#if templateDirty && selectedTemplate}
              <button
                class="btn-primary px-3 py-2"
                onclick={handleSaveTemplate}
                disabled={isSavingTemplate}
                title="Save changes to template"
              >
                {isSavingTemplate ? "…" : "Save"}
              </button>
            {/if}
          {/if}
        </div>
      </div>

      {#if templateSaveError}
        <p class="text-xs text-error">{templateSaveError}</p>
      {/if}

      <!-- Editor -->
      <div class="relative">
        {#if isLoadingTemplate}
          <SpinnerBlock class="py-8" size="h-4 w-4" label="Loading template…" />
        {:else}
          <JinjaEditor
            value={templateContent}
            onchange={handleTemplateContentChange}
          />
        {/if}
      </div>

      {#if templateSource === "builtin" && !templateDirty}
        <p class="text-xs text-gray-500">
          Built-in template. Edit above and save to create a user override.
        </p>
      {/if}
    </section>

    <!-- ═══ Section: Options ═══ -->
    <section class="space-y-3">
      <h3 class="section-heading">Options</h3>

      <div class="grid grid-cols-2 gap-3">
        <!-- Max Tokens -->
        <div>
          <label class="label" for="caption-tokens">Max Tokens</label>
          <input
            id="caption-tokens"
            type="number"
            bind:value={maxTokens}
            min={100}
            max={16384}
            class="input"
          />
        </div>

        <!-- Image Quality -->
        <div>
          <label class="label" for="caption-quality">Image Quality</label>
          <select
            id="caption-quality"
            class="input cursor-pointer"
            bind:value={imageQuality}
          >
            <option value="auto">Auto</option>
            <option value="high">High</option>
            <option value="low">Low</option>
          </select>
        </div>

        <!-- Draft -->
        <div>
          <label class="label" for="caption-draft">Draft Name</label>
          <input
            id="caption-draft"
            type="text"
            bind:value={draftName}
            class="input"
            placeholder="(none — write caption directly)"
          />
        </div>

        <!-- Rounds -->
        <div>
          <label class="label" for="caption-rounds">Rounds</label>
          <input
            id="caption-rounds"
            type="number"
            bind:value={rounds}
            min={1}
            max={10}
            class="input"
          />
        </div>
      </div>

      <!-- Overwrite -->
      <div class="flex items-center gap-2">
        <Checkbox id="caption-overwrite" bind:checked={overwrite} />
        <label class="text-sm text-gray-300 cursor-pointer" for="caption-overwrite">
          Overwrite existing captions
        </label>
      </div>
    </section>

    <!-- ═══ Section: Reasoning ═══ -->
    <section class="space-y-3">
      <div class="flex items-center gap-2">
        <Checkbox id="caption-reasoning" bind:checked={reasoningEnabled} />
        <label class="section-heading cursor-pointer" for="caption-reasoning">
          Reasoning
        </label>
      </div>

      {#if reasoningEnabled}
        <div>
          <label class="label" for="caption-effort">Thinking Effort</label>
          <select
            id="caption-effort"
            class="input cursor-pointer"
            bind:value={reasoningEffort}
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </div>
      {/if}
    </section>
  </div>

  <!-- Footer: sticky start button -->
  <div class="flex-shrink-0 p-4 border-t border-border">
    <button
      class="btn-primary w-full"
      onclick={handleStart}
    >
      Start Captioning
    </button>
  </div>
</div>
