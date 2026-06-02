<script lang="ts">
  import Dialog from "$lib/components/ui/Dialog.svelte";
  import SvgClose from "$lib/icons/SvgClose.svelte";
  import TabBar from "$lib/components/ui/TabBar.svelte";
  import SpinnerBlock from "$lib/components/ui/SpinnerBlock.svelte";
  import { createDataset, importDataset, type DatasetInfo } from "$lib/stores/datasetImages";

  interface Props {
    open: boolean;
    onclose: () => void;
    oncreated: (dataset: DatasetInfo) => void;
  }

  let { open, onclose, oncreated }: Props = $props();

  type Mode = "import" | "create";
  let mode: Mode = $state("import");

  // Import state
  let importName = $state("");
  let importTomlPath = $state("");

  // Create state
  let createName = $state("");
  let createPaths = $state("");

  // Shared
  let isSubmitting = $state(false);
  let error: string | null = $state(null);

  $effect(() => {
    if (open) {
      error = null;
      isSubmitting = false;
      importName = "";
      importTomlPath = "";
      createName = "";
      createPaths = "";
    }
  });

  function switchMode(m: Mode) {
    mode = m;
    error = null;
  }

  async function handleImport() {
    const name = importName.trim();
    const tomlPath = importTomlPath.trim();

    if (!name) {
      error = "Dataset name is required";
      return;
    }
    if (!tomlPath) {
      error = "TOML file path is required";
      return;
    }

    isSubmitting = true;
    error = null;

    try {
      const dataset = await importDataset(name, tomlPath);
      oncreated(dataset);
      onclose();
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to import dataset";
    } finally {
      isSubmitting = false;
    }
  }

  async function handleCreate() {
    const name = createName.trim();
    const paths = createPaths
      .split("\n")
      .map((p) => p.trim())
      .filter((p) => p.length > 0);

    if (!name) {
      error = "Dataset name is required";
      return;
    }
    if (paths.length === 0) {
      error = "At least one image directory path is required";
      return;
    }

    isSubmitting = true;
    error = null;

    try {
      const dataset = await createDataset(name, paths);
      oncreated(dataset);
      onclose();
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to create dataset";
    } finally {
      isSubmitting = false;
    }
  }


</script>

<Dialog
  class="dialog-panel max-w-lg max-h-[80vh] overflow-y-auto"
  {open}
  {onclose}
>
  <div class="p-5">
    <!-- Header -->
    <div class="dialog-header">
      <h2 class="dialog-title">Add Dataset</h2>
      <button class="btn-close" onclick={onclose}>
        <SvgClose class="h-5 w-5" />
      </button>
    </div>

    <!-- Mode tabs -->
    <TabBar
      class="mb-4"
      tabs={[
        { value: "import", label: "Import TOML" },
        { value: "create", label: "Create New" },
      ]}
      selected={mode}
      onchange={(v) => switchMode(v as Mode)}
    />

    {#if error}
      <div class="alert-error mb-4">{error}</div>
    {/if}

    <!-- Import mode -->
    {#if mode === "import"}
      <div class="space-y-4">
        <div>
          <label class="label" for="import-name">Dataset Name</label>
          <input
            id="import-name"
            type="text"
            bind:value={importName}
            class="input"
            placeholder="my-dataset"
          />
        </div>

        <div>
          <label class="label" for="import-toml">TOML Config Path</label>
          <input
            id="import-toml"
            type="text"
            bind:value={importTomlPath}
            class="input"
            placeholder="/path/to/dataset.toml"
          />
          <p class="help-text">
            Absolute path to an existing yadc dataset config TOML file.
          </p>
        </div>

        <div class="btn-bar">
          <button
            class="btn-primary"
            onclick={handleImport}
            disabled={isSubmitting}
          >
            {isSubmitting ? "Importing…" : "Import"}
          </button>
        </div>
      </div>

    <!-- Create mode -->
    {:else if mode === "create"}
      <div class="space-y-4">
        <div>
          <label class="label" for="create-name">Dataset Name</label>
          <input
            id="create-name"
            type="text"
            bind:value={createName}
            class="input"
            placeholder="my-dataset"
          />
        </div>

        <div>
          <label class="label" for="create-paths">Image Directories</label>
          <textarea
            id="create-paths"
            bind:value={createPaths}
            rows={4}
            class="input resize-y"
            placeholder="/path/to/images&#10;/another/image/dir"></textarea>
          <p class="help-text">
            One directory path per line. Each directory will be scanned for images.
          </p>
        </div>

        <div class="btn-bar">
          <button
            class="btn-primary"
            onclick={handleCreate}
            disabled={isSubmitting}
          >
            {isSubmitting ? "Creating…" : "Create"}
          </button>
        </div>
      </div>
    {/if}

    {#if isSubmitting}
      <SpinnerBlock class="py-4" size="h-5 w-5" />
    {/if}
  </div>
</Dialog>
