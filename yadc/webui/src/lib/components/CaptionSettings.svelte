<script lang="ts">
  import Dialog from "$lib/components/Dialog.svelte";
  import EnvManager from "$lib/components/EnvManager.svelte";
  import JinjaEditor from "$lib/components/JinjaEditor.svelte";
  import Checkbox from "$lib/components/Checkbox.svelte";
  import SvgClose from "$lib/icons/SvgClose.svelte";
  import SvgRefresh from "$lib/icons/SvgRefresh.svelte";
  import SvgPlus from "$lib/icons/SvgPlus.svelte";
  import SvgSpinner from "$lib/icons/SvgSpinner.svelte";
  import {
    fetchEnvs,
    fetchEnv,
    fetchModels,
    type EnvInfo,
  } from "$lib/stores/envs";
  import {
    fetchTemplates,
    fetchTemplate,
    saveTemplate,
    type TemplateListItem,
  } from "$lib/stores/templates";
  import type { CaptionOptions } from "$lib/stores/captionOptions";

  // --- Props ---

  interface Props {
    /** Which dataset this dialog is for. */
    datasetName: string;
    open: boolean;
    onclose: () => void;
    /** Called when the user clicks "Start Captioning". Receives the assembled options. */
    onstart?: (options: CaptionOptions) => void;
  }

  let { datasetName, open, onclose, onstart }: Props = $props();

  // --- State: Environments ---

  let envNames: string[] = $state([]);
  let selectedEnv = $state("default");
  let envInfo: EnvInfo | null = $state(null);

  let envUrl = $state("");
  let envToken = $state("");
  let envModelName = $state("");

  let showEnvManager = $state(false);

  // --- State: Models ---

  let models: string[] = $state([]);
  let isLoadingModels = $state(false);
  let modelsError: string | null = $state(null);
  let modelFetchDone = $state(false);

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

  let isLoadingEnvs = $state(false);
  let isLoadingTemplates = $state(false);
  let envsError: string | null = $state(null);
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

  // --- Load data when dialog opens ---

  $effect(() => {
    if (open) {
      resetForm();
      loadEnvs();
      loadTemplateList();
    }
  });

  function resetForm() {
    maxTokens = 512;
    imageQuality = "auto";
    draftName = "";
    overwrite = false;
    rounds = 1;
    reasoningEnabled = false;
    reasoningEffort = "low";
    templateDirty = false;
    isNewTemplate = false;
    newTemplateName = "";
    templateSaveError = null;
    modelsError = null;
    modelFetchDone = false;
  }

  // --- Env loading & selection ---

  async function loadEnvs() {
    isLoadingEnvs = true;
    envsError = null;
    try {
      envNames = await fetchEnvs();
    } catch (e) {
      envsError = e instanceof Error ? e.message : "Failed to load environments";
    } finally {
      isLoadingEnvs = false;
    }
  }

  /** When env selection changes, load its details and auto-fill URL/token/model. */
  $effect(() => {
    const env = selectedEnv;
    if (!env) return;

    let cancelled = false;
    (async () => {
      try {
        const info = await fetchEnv(env);
        if (cancelled) return;
        envInfo = info;
        envUrl = info.api_url || "";
        envToken = ""; // Don't pre-fill token (masked as [REDACTED] in API)
        envModelName = info.api_model_name || "";

        // Reset model list when env changes
        models = [];
        modelFetchDone = false;
        modelsError = null;
      } catch {
        if (cancelled) return;
        envInfo = null;
      }
    })();

    return () => { cancelled = true; };
  });

  // --- Model fetching ---

  async function loadModels() {
    if (!selectedEnv) return;
    isLoadingModels = true;
    modelsError = null;
    try {
      const result = await fetchModels(selectedEnv);
      models = result.models;
      modelFetchDone = true;
    } catch (e) {
      modelsError = e instanceof Error ? e.message : "Failed to fetch models";
    } finally {
      isLoadingModels = false;
    }
  }

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
      // Trigger re-load by touching the selection
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

  // --- Env manager closed → refresh env list ---

  $effect(() => {
    // When EnvManager closes, reload env list
    if (!showEnvManager && open) {
      // Only reload if we've been open (not the initial false)
      loadEnvs();
    }
  });

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
    onclose();
  }

  // --- Close handling ---

  function handleClose() {
    onclose();
  }
</script>

<!-- Sub-dialogs -->
<EnvManager open={showEnvManager} onclose={() => (showEnvManager = false)} />

<Dialog
  class="w-full max-w-3xl max-h-[85vh] overflow-y-auto bg-surface rounded-xl shadow-2xl border border-border m-auto"
  open={open}
  onclose={handleClose}
>
  <div class="p-6 space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <h2 class="text-lg font-semibold text-white">Caption Settings</h2>
      <button
        class="p-1 text-gray-400 hover:text-white transition-colors cursor-pointer"
        onclick={handleClose}
      >
        <SvgClose class="h-5 w-5" />
      </button>
    </div>

    <!-- Errors -->
    {#if envsError}
      <div class="p-3 rounded-lg bg-error/10 border border-error/20 text-error text-sm">
        {envsError}
      </div>
    {/if}

    <!-- ═══ Section: Environment ═══ -->
    <section class="space-y-3">
      <div class="flex items-center justify-between">
        <h3 class="text-sm font-medium text-gray-300 uppercase tracking-wide">Environment</h3>
        <button
          class="text-xs text-accent hover:text-accent-hover cursor-pointer"
          onclick={() => (showEnvManager = true)}
        >
          Manage…
        </button>
      </div>

      <div>
        <label class="block text-sm text-gray-400 mb-1" for="caption-env">Environment</label>
        <select
          id="caption-env"
          class="w-full rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none cursor-pointer"
          bind:value={selectedEnv}
          disabled={isLoadingEnvs}
        >
          {#each envNames as name (name)}
            <option value={name}>{name}</option>
          {/each}
          {#if envNames.length === 0}
            <option value="default" disabled>default</option>
          {/if}
        </select>
      </div>

      <div class="grid grid-cols-1 gap-3">
        <!-- API URL -->
        <div>
          <label class="block text-sm text-gray-400 mb-1" for="caption-url">API URL</label>
          <input
            id="caption-url"
            type="text"
            bind:value={envUrl}
            class="w-full rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none"
            placeholder="https://api.openai.com"
          />
        </div>

        <!-- API Token -->
        <div>
          <label class="block text-sm text-gray-400 mb-1" for="caption-token">
            API Token
            {#if envInfo?.api_token}
              <span class="text-gray-500 ml-1">(leave blank to use saved)</span>
            {/if}
          </label>
          <input
            id="caption-token"
            type="password"
            bind:value={envToken}
            class="w-full rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none"
            placeholder={envInfo?.api_token ? "•••••••• (saved)" : "sk-…"}
          />
        </div>

        <!-- Model -->
        <div>
          <label class="block text-sm text-gray-400 mb-1" for="caption-model">Model</label>
          <div class="flex gap-2">
            {#if modelFetchDone && models.length > 0}
              <select
                id="caption-model"
                class="flex-1 rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none cursor-pointer"
                bind:value={envModelName}
              >
                {#each models as m (m)}
                  <option value={m}>{m}</option>
                {/each}
                {#if !models.includes(envModelName) && envModelName}
                  <option value={envModelName}>{envModelName}</option>
                {/if}
              </select>
            {:else}
              <input
                id="caption-model"
                type="text"
                bind:value={envModelName}
                class="flex-1 rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none"
                placeholder="gpt-4o-mini"
              />
            {/if}
            <button
              class="px-3 py-2 rounded-lg bg-bg border border-border text-gray-400 hover:text-white hover:border-gray-500 transition-colors cursor-pointer disabled:opacity-50"
              onclick={loadModels}
              disabled={isLoadingModels || !selectedEnv}
              title="Fetch available models"
            >
              {#if isLoadingModels}
                <SvgSpinner class="h-4 w-4 animate-spin" />
              {:else}
                <SvgRefresh class="h-4 w-4" />
              {/if}
            </button>
          </div>
          {#if modelsError}
            <p class="text-xs text-error mt-1">{modelsError}</p>
          {/if}
        </div>
      </div>
    </section>

    <!-- ═══ Section: Template ═══ -->
    <section class="space-y-3">
      <h3 class="text-sm font-medium text-gray-300 uppercase tracking-wide">Template</h3>

      {#if templatesError}
        <div class="p-3 rounded-lg bg-error/10 border border-error/20 text-error text-sm">
          {templatesError}
        </div>
      {/if}

      <div class="flex gap-2 items-end">
        <div class="flex-1">
          {#if isNewTemplate}
            <label class="block text-sm text-gray-400 mb-1" for="new-tpl-name">New Template Name</label>
            <input
              id="new-tpl-name"
              type="text"
              bind:value={newTemplateName}
              class="w-full rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none"
              placeholder="my-template"
            />
          {:else}
            <label class="block text-sm text-gray-400 mb-1" for="caption-template">Template</label>
            <select
              id="caption-template"
              class="w-full rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none cursor-pointer"
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
              class="px-3 py-2 text-sm rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 cursor-pointer"
              onclick={cancelNewTemplate}
            >
              Cancel
            </button>
            <button
              class="px-3 py-2 text-sm rounded-lg bg-accent hover:bg-accent-hover text-black font-medium cursor-pointer disabled:opacity-50"
              onclick={handleSaveTemplate}
              disabled={isSavingTemplate || !newTemplateName.trim()}
            >
              {isSavingTemplate ? "Saving…" : "Save"}
            </button>
          {:else}
            <button
              class="px-3 py-2 text-sm rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 cursor-pointer"
              onclick={startNewTemplate}
              title="Create new template"
            >
              <SvgPlus class="h-4 w-4" />
            </button>
            {#if templateDirty && selectedTemplate}
              <button
                class="px-3 py-2 text-sm rounded-lg bg-accent hover:bg-accent-hover text-black font-medium cursor-pointer disabled:opacity-50"
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
          <div class="flex items-center gap-2 text-gray-400 py-8 justify-center">
            <SvgSpinner class="h-4 w-4 animate-spin" />
            <span class="text-sm">Loading template…</span>
          </div>
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
      <h3 class="text-sm font-medium text-gray-300 uppercase tracking-wide">Options</h3>

      <div class="grid grid-cols-2 gap-3">
        <!-- Max Tokens -->
        <div>
          <label class="block text-sm text-gray-400 mb-1" for="caption-tokens">Max Tokens</label>
          <input
            id="caption-tokens"
            type="number"
            bind:value={maxTokens}
            min={100}
            max={16384}
            class="w-full rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none"
          />
        </div>

        <!-- Image Quality -->
        <div>
          <label class="block text-sm text-gray-400 mb-1" for="caption-quality">Image Quality</label>
          <select
            id="caption-quality"
            class="w-full rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none cursor-pointer"
            bind:value={imageQuality}
          >
            <option value="auto">Auto</option>
            <option value="high">High</option>
            <option value="low">Low</option>
          </select>
        </div>

        <!-- Draft -->
        <div>
          <label class="block text-sm text-gray-400 mb-1" for="caption-draft">Draft Name</label>
          <input
            id="caption-draft"
            type="text"
            bind:value={draftName}
            class="w-full rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none"
            placeholder="(none — write caption directly)"
          />
        </div>

        <!-- Rounds -->
        <div>
          <label class="block text-sm text-gray-400 mb-1" for="caption-rounds">Rounds</label>
          <input
            id="caption-rounds"
            type="number"
            bind:value={rounds}
            min={1}
            max={10}
            class="w-full rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none"
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
        <label class="text-sm font-medium text-gray-300 uppercase tracking-wide cursor-pointer" for="caption-reasoning">
          Reasoning
        </label>
      </div>

      {#if reasoningEnabled}
        <div>
          <label class="block text-sm text-gray-400 mb-1" for="caption-effort">Thinking Effort</label>
          <select
            id="caption-effort"
            class="w-full rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none cursor-pointer"
            bind:value={reasoningEffort}
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </div>
      {/if}
    </section>

    <!-- ═══ Footer ═══ -->
    <div class="flex gap-2 justify-end pt-2 border-t border-border">
      <button
        class="px-4 py-2 text-sm rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 cursor-pointer"
        onclick={handleClose}
      >
        Cancel
      </button>
      <button
        class="px-4 py-2 text-sm rounded-lg bg-accent hover:bg-accent-hover text-black font-medium cursor-pointer"
        onclick={handleStart}
      >
        Start Captioning
      </button>
    </div>
  </div>
</Dialog>
