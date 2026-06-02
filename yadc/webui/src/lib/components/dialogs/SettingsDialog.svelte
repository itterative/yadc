<script lang="ts">
  import Dialog from "$lib/components/ui/Dialog.svelte";
  import TabBar from "$lib/components/ui/TabBar.svelte";
  import ConfigEditor from "$lib/components/settings/ConfigEditor.svelte";
  import EnvManager from "$lib/components/dialogs/EnvManager.svelte";
  import SvgClose from "$lib/icons/SvgClose.svelte";

  interface Props {
    open: boolean;
    onclose: () => void;
  }

  let { open, onclose }: Props = $props();

  // --- Tabs ---
  type Tab = "configs" | "environments";
  let activeTab: Tab = $state("configs");

  // --- Env state ---
  let showEnvManager = $state(false);
</script>

<!-- Sub-dialogs -->
<EnvManager open={showEnvManager} onclose={() => (showEnvManager = false)} />

<Dialog
  class="dialog-panel max-w-3xl max-h-[85vh] overflow-hidden flex flex-col"
  {open}
  onclose={onclose}
>
  <div class="flex flex-col h-full max-h-[85vh]">
    <!-- Header -->
    <div class="flex items-center justify-between p-5 pb-0">
      <h2 class="dialog-title">Settings</h2>
      <button class="btn-close" onclick={onclose}>
        <SvgClose class="h-5 w-5" />
      </button>
    </div>

    <!-- Tabs -->
    <TabBar
      class="px-5 pt-3"
      tabs={[
        { value: "configs", label: "Configs" },
        { value: "environments", label: "Environments" },
      ]}
      selected={activeTab}
      onchange={(v) => (activeTab = v as Tab)}
    />

    <!-- Tab content -->
    <div class="flex-1 overflow-y-auto p-5 space-y-4">
      {#if activeTab === "configs"}
        <ConfigEditor {open} />
      {:else if activeTab === "environments"}
        <div class="py-8 text-center">
          <p class="text-gray-400 text-sm mb-4">
            Environments store API connection settings (URL, token, default model).
          </p>
          <button
            class="btn-primary"
            onclick={() => (showEnvManager = true)}
          >
            Open Environment Manager
          </button>
        </div>
      {/if}
    </div>
  </div>
</Dialog>
