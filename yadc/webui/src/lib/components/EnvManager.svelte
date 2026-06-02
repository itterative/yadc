<script lang="ts">
  import Dialog from "$lib/components/Dialog.svelte";
  import SvgClose from "$lib/icons/SvgClose.svelte";
  import SvgDelete from "$lib/icons/SvgDelete.svelte";
  import SvgEdit from "$lib/icons/SvgEdit.svelte";
  import SvgPlus from "$lib/icons/SvgPlus.svelte";
  import SvgSpinner from "$lib/icons/SvgSpinner.svelte";
  import { deleteEnv, fetchEnv, fetchEnvs, saveEnv, type EnvInfo } from "$lib/stores/envs";

  interface Props {
    open: boolean;
    onclose: () => void;
  }

  let { open, onclose }: Props = $props();

  let envNames: string[] = $state([]);
  let isLoading = $state(false);
  let error: string | null = $state(null);

  // Edit state — null means "list view", set means "editing this env"
  let editingEnv: EnvInfo | null = $state(null);
  let isNew = $state(false);
  let editName = $state("");
  let editUrl = $state("");
  let editToken = $state("");
  let editModelName = $state("");
  let isSaving = $state(false);
  let saveError: string | null = $state(null);

  // Confirm delete
  let confirmDelete: string | null = $state(null);

  $effect(() => {
    if (open) {
      loadEnvs();
      editingEnv = null;
      confirmDelete = null;
    }
  });

  async function loadEnvs() {
    isLoading = true;
    error = null;
    try {
      envNames = await fetchEnvs();
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load environments";
    } finally {
      isLoading = false;
    }
  }

  function startCreate() {
    isNew = true;
    editingEnv = null;
    editName = "";
    editUrl = "";
    editToken = "";
    editModelName = "";
    saveError = null;
  }

  async function startEdit(name: string) {
    saveError = null;
    try {
      const info = await fetchEnv(name);
      isNew = false;
      editingEnv = info;
      editName = info.name;
      editUrl = info.api_url || "";
      editToken = ""; // Don't pre-fill token (it's masked as [REDACTED])
      editModelName = info.api_model_name || "";
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load environment";
    }
  }

  function cancelEdit() {
    editingEnv = null;
    isNew = false;
    saveError = null;
  }

  async function handleSave() {
    const name = editName.trim();
    if (!name) {
      saveError = "Name is required";
      return;
    }

    isSaving = true;
    saveError = null;
    try {
      const data: Record<string, string> = { api_url: editUrl.trim() };
      if (editToken) data.api_token = editToken;
      if (editModelName.trim()) data.api_model_name = editModelName.trim();

      await saveEnv(name, data);
      editingEnv = null;
      isNew = false;
      await loadEnvs();
    } catch (e) {
      saveError = e instanceof Error ? e.message : "Failed to save environment";
    } finally {
      isSaving = false;
    }
  }

  async function handleDelete(name: string) {
    try {
      await deleteEnv(name);
      confirmDelete = null;
      await loadEnvs();
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to delete environment";
    }
  }

  function handleDialogClose() {
    if (editingEnv || isNew) {
      cancelEdit();
      return;
    }
    onclose();
  }
</script>

<Dialog
  class="w-full max-w-lg max-h-[80vh] overflow-y-auto bg-surface rounded-xl shadow-2xl border border-border m-auto"
  open={open}
  onclose={handleDialogClose}
>
  <div class="p-5">
    <!-- Header -->
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-lg font-semibold text-white">Manage Environments</h2>
      <button
        class="p-1 text-gray-400 hover:text-white transition-colors cursor-pointer"
        onclick={handleDialogClose}
      >
        <SvgClose class="h-5 w-5" />
      </button>
    </div>

    {#if error}
      <div class="mb-4 p-3 rounded-lg bg-error/10 border border-error/20 text-error text-sm">
        {error}
      </div>
    {/if}

    <!-- List view -->
    {#if !editingEnv && !isNew}
      <div class="space-y-2 mb-4">
        {#if isLoading}
          <div class="flex items-center justify-center py-8">
            <SvgSpinner class="h-6 w-6 animate-spin text-gray-400" />
          </div>
        {:else if envNames.length === 0}
          <p class="text-gray-500 text-sm text-center py-4">No environments yet.</p>
        {:else}
          {#each envNames as name (name)}
            <div class="flex items-center gap-3 p-3 rounded-lg bg-bg border border-border group">
              <div class="flex-1 min-w-0">
                <span class="text-sm font-medium text-white">{name}</span>
              </div>
              <div class="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  class="p-1.5 text-gray-400 hover:text-accent transition-colors cursor-pointer rounded"
                  title="Edit"
                  onclick={() => startEdit(name)}
                >
                  <SvgEdit class="h-4 w-4" />
                </button>
                {#if name !== "default"}
                  <button
                    class="p-1.5 text-gray-400 hover:text-error transition-colors cursor-pointer rounded"
                    title="Delete"
                    onclick={() => (confirmDelete = name)}
                  >
                    <SvgDelete class="h-4 w-4" />
                  </button>
                {/if}
              </div>
            </div>
          {/each}
        {/if}
      </div>

      <button
        class="flex items-center gap-2 w-full justify-center px-4 py-2.5 text-sm rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-200 transition-colors cursor-pointer"
        onclick={startCreate}
      >
        <SvgPlus class="h-4 w-4" />
        New Environment
      </button>

      <!-- Delete confirmation -->
      {#if confirmDelete}
        <div class="mt-4 p-4 rounded-lg bg-error/10 border border-error/20">
          <p class="text-sm text-gray-200 mb-3">Delete environment <strong>{confirmDelete}</strong>?</p>
          <div class="flex gap-2 justify-end">
            <button
              class="px-3 py-1.5 text-sm rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 cursor-pointer"
              onclick={() => (confirmDelete = null)}
            >
              Cancel
            </button>
            <button
              class="px-3 py-1.5 text-sm rounded-lg bg-error hover:bg-error/80 text-white cursor-pointer"
              onclick={() => handleDelete(confirmDelete!)}
            >
              Delete
            </button>
          </div>
        </div>
      {/if}

    <!-- Edit / Create form -->
    {:else}
      <div class="space-y-4">
        {#if saveError}
          <div class="p-3 rounded-lg bg-error/10 border border-error/20 text-error text-sm">
            {saveError}
          </div>
        {/if}

        <!-- Name -->
        <div>
          <label class="block text-sm text-gray-300 mb-1" for="env-name">Name</label>
          <input
            id="env-name"
            type="text"
            bind:value={editName}
            disabled={!isNew}
            class="w-full rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none disabled:opacity-50"
            placeholder="my-environment"
          />
        </div>

        <!-- API URL -->
        <div>
          <label class="block text-sm text-gray-300 mb-1" for="env-url">API URL</label>
          <input
            id="env-url"
            type="text"
            bind:value={editUrl}
            class="w-full rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none"
            placeholder="https://api.openai.com"
          />
        </div>

        <!-- API Token -->
        <div>
          <label class="block text-sm text-gray-300 mb-1" for="env-token">
            API Token
            {#if editingEnv?.api_token}
              <span class="text-gray-500 ml-1">(leave blank to keep current)</span>
            {/if}
          </label>
          <input
            id="env-token"
            type="password"
            bind:value={editToken}
            class="w-full rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none"
            placeholder={editingEnv?.api_token ? "••••••••" : "sk-..."}
          />
        </div>

        <!-- Default Model -->
        <div>
          <label class="block text-sm text-gray-300 mb-1" for="env-model">Default Model</label>
          <input
            id="env-model"
            type="text"
            bind:value={editModelName}
            class="w-full rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none"
            placeholder="gpt-4o-mini"
          />
        </div>

        <!-- Actions -->
        <div class="flex gap-2 justify-end pt-2">
          <button
            class="px-4 py-2 text-sm rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 cursor-pointer"
            onclick={cancelEdit}
            disabled={isSaving}
          >
            Cancel
          </button>
          <button
            class="px-4 py-2 text-sm rounded-lg bg-accent hover:bg-accent-hover text-black font-medium cursor-pointer disabled:opacity-50"
            onclick={handleSave}
            disabled={isSaving}
          >
            {isSaving ? "Saving..." : isNew ? "Create" : "Save"}
          </button>
        </div>
      </div>
    {/if}
  </div>
</Dialog>
