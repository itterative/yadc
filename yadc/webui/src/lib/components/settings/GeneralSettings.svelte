<script lang="ts">
    import Checkbox from '$lib/components/ui/Checkbox.svelte';
    import { settings } from '$lib/stores/settings';
    import {
        fetchActiveTagger,
        fetchSuggestionVariant,
        listTaggerModels,
        setSuggestionVariantAction,
        swapActiveModelAction
    } from '$lib/stores/tagging';
    import { taggingStatuses, taggerStatus } from '$lib/stores/tagging';
    import { toast } from '$lib/stores/toasts';
    import type {
        ActiveTaggerSelection,
        SuggestionVariantResponse,
        SwapTaggerBody,
        TaggerModelSummary
    } from '$lib/stores/tagging';
    import { LOCAL_FILE_ID } from '$lib/stores/tagging';
    import {
        notificationsSupported,
        notificationPermission,
        requestNotificationPermission
    } from '$lib/notifications';
    import SvgSpinner from '$lib/icons/SvgSpinner.svelte';
    import { examplePath, isWindows } from '$lib/format';

    let permStatus = $state<NotificationPermission | 'unsupported'>('default');
    let notificationsOn = $state(false);
    let previousStoreValue = $state<'unset' | 'enabled' | 'disabled'>('unset');

    $effect(() => {
        const storeVal = $settings.notifications;
        if (storeVal !== previousStoreValue) {
            notificationsOn = storeVal === 'enabled';
            previousStoreValue = storeVal;
        }
        permStatus = notificationPermission();
    });

    $effect(() => {
        const enabled = notificationsOn;
        const currentStore = $settings.notifications;
        const wantsEnable = enabled && currentStore !== 'enabled';
        const wantsDisable = !enabled && currentStore === 'enabled';

        if (!wantsEnable && !wantsDisable) {
            return;
        }

        if (wantsEnable) {
            requestNotificationPermission().then((perm) => {
                permStatus = perm;
                if (perm === 'granted') {
                    settings.update((s) => ({ ...s, notifications: 'enabled' }));
                } else {
                    notificationsOn = false;
                }
            });
        } else {
            settings.update((s) => ({ ...s, notifications: 'disabled' }));
        }
    });

    // --- Tagger model picker ---
    //
    // The picker is a small composition of three things: the curated catalog
    // (dropdown options + the local-file sentinel), the active selection
    // (seeded from GET /api/tagger/active and pushed back on swap), and the
    // subprocess liveness indicator (from ``taggerStatus`` SSE). On submit
    // we call :func:`swapActiveModelAction` which toasts the outcome and
    // returns the persisted selection so we can re-seed the form without
    // a refetch.

    let pickerLoaded = $state(false);
    let catalog: TaggerModelSummary[] = $state([]);
    let profiles: string[] = $state([]);
    let active: ActiveTaggerSelection | null = $state(null);
    let pickerSelectedId = $state('');
    let localModelPath = $state('');
    let localLabelPath = $state('');
    let pickerProfile = $state('wd-tagger');
    let pickerSize = $state(0);
    let pickerError = $state<string | null>(null);
    let isSwapping = $state(false);
    // The swap POST returns 202 immediately; the drain+download+respawn runs
    // in the background and is reported over the ``tagger_status`` SSE stream.
    // These track that in-flight swap so the spinner stays up and the
    // success/failure toast fires when the SSE state actually settles.
    let swapInFlight = $state(false);
    // Guards against the initial ``ready`` value (the old subprocess) racing
    // ahead of the swap's own transition: we only resolve once we've seen a
    // swap-driven transition (stopping/stopped/starting) followed by ready/failed.
    let sawSwapTransition = $state(false);

    // Build the dropdown options from the curated catalog + the local sentinel.
    // The local sentinel inherits its profile / size from the active
    // selection at swap time (see ``buildSelectionBody``), so the dropdown
    // values here are placeholders only — they're never read.
    let pickerOptions: TaggerModelSummary[] = $derived(
        pickerLoaded
            ? [
                  ...catalog,
                  {
                      id: LOCAL_FILE_ID,
                      display: 'Local file…',
                      params: '',
                      default_preproc_profile: 'wd-tagger',
                      default_size: 0
                  }
              ]
            : []
    );

    // Any running batch across all datasets disables the swap button — the
    // backend would refuse with 409 anyway, but a disabled button makes the
    // cause obvious before the round-trip.
    let anyBatchRunning = $derived(
        Array.from($taggingStatuses.values()).some(
            (s) => s.status === 'running' || s.status === 'stopping'
        )
    );

    $effect(() => {
        if (pickerLoaded) {
            return;
        }
        const controller = new AbortController();
        void (async () => {
            try {
                const [a, m] = await Promise.all([
                    fetchActiveTagger(controller.signal),
                    listTaggerModels(controller.signal)
                ]);
                active = a.active;
                catalog = m.models;
                profiles = m.profiles;
                // Seed the dropdown to the active selection, if any.
                if (a.active) {
                    pickerSelectedId = a.active.kind === 'hf' ? a.active.repo_id : LOCAL_FILE_ID;
                    localModelPath = a.active.model_path;
                    localLabelPath = a.active.label_path;
                    // Preserve the active selection's profile / size so a
                    // swap-with-different-model doesn't silently drop the
                    // user's ``timm`` profile. Falls back to wd-tagger for
                    // a fresh install.
                    pickerProfile = a.active.preproc_profile || 'wd-tagger';
                    pickerSize = a.active.default_size || 0;
                }
                pickerLoaded = true;
            } catch (e) {
                if ((e as Error).name !== 'AbortError') {
                    pickerError = (e as Error).message;
                    pickerLoaded = true;
                }
            }
        })();
        return () => controller.abort();
    });

    // The catalog row for the currently-selected option. Derived so the
    // helper text + body builder + button gate all read the same value
    // without each re-running the .find() lookup.
    let selectedCatalog: TaggerModelSummary | null = $derived(
        pickerOptions.find((m) => m.id === pickerSelectedId) ?? null
    );

    let isLocalSelected = $derived(pickerSelectedId === LOCAL_FILE_ID);

    // Disable the Swap button when the form's effective body matches the
    // active selection — the backend would treat that as a no-op (same
    // identity), and the UI surfaces it as a disabled button so the user
    // knows there's nothing to change. Compares every persisted field,
    // not just the model path, so the user can freely edit Local fields
    // (profile, size, label path) and the button stays disabled until
    // any of them actually differ from the active selection.
    let canSwap: boolean = $derived.by(() => {
        if (!pickerLoaded || isSwapping || anyBatchRunning) {
            return false;
        }
        const body = buildSelectionBody();
        if (!body) {
            return false;
        }
        if (!active) {
            return true;
        }
        return body.kind === active.kind &&
            body.repo_id === active.repo_id &&
            body.model_path === active.model_path &&
            body.label_path === active.label_path &&
            body.preproc_profile === active.preproc_profile &&
            body.default_size === active.default_size
            ? false
            : true;
    });

    function buildSelectionBody(): SwapTaggerBody | null {
        if (!selectedCatalog) {
            return null;
        }
        if (selectedCatalog.id === LOCAL_FILE_ID) {
            if (!localModelPath.trim()) {
                return null;
            }
            return {
                kind: 'local',
                repo_id: '',
                repo_model_filename: 'model.onnx',
                repo_label_filename: 'selected_tags.csv',
                model_path: localModelPath.trim(),
                label_path: localLabelPath.trim(),
                preproc_profile: pickerProfile,
                default_size: pickerSize
            };
        }
        // Curated HF rows carry their own profile / size — the picker
        // hides the controls for these so the user can't pick a
        // mismatched combination. Local uses the editable controls
        // (above).
        return {
            kind: 'hf',
            repo_id: selectedCatalog.id,
            repo_model_filename: 'model.onnx',
            repo_label_filename: 'selected_tags.csv',
            model_path: '',
            label_path: '',
            preproc_profile: selectedCatalog.default_preproc_profile,
            default_size: selectedCatalog.default_size
        };
    }

    async function handleSwap() {
        const body = buildSelectionBody();
        if (!body) {
            return;
        }
        isSwapping = true;
        swapInFlight = true;
        sawSwapTransition = false;
        try {
            const next = await swapActiveModelAction(body);
            if (next) {
                // 202 Accepted — the backend accepted the swap and is running
                // the drain+download+respawn in the background. Show the
                // intended selection immediately; keep the spinner up until
                // the ``tagger_status`` SSE stream settles on ready/failed.
                active = {
                    ...next,
                    source: next.kind === 'hf' ? `hf:${next.repo_id}` : `local:${next.model_path}`
                };
            } else {
                // Refused (busy / in-progress / error) — the action already
                // toasted. No background work, so drop the spinner.
                isSwapping = false;
                swapInFlight = false;
            }
        } catch (e) {
            isSwapping = false;
            swapInFlight = false;
            toast.error('Failed to swap tagger model', {
                details: [(e as Error).message]
            });
        }
    }

    // Resolve the in-flight swap from the ``tagger_status`` SSE stream. The
    // swap always ends in ``ready`` (success) or ``failed`` (rolled back);
    // ``sawSwapTransition`` ignores the stale pre-swap status so an already-
    // ready subprocess doesn't falsely short-circuit the wait.
    $effect(() => {
        const status = $taggerStatus;
        if (!swapInFlight) {
            return;
        }
        if (
            status.state === 'stopping' ||
            status.state === 'stopped' ||
            status.state === 'starting'
        ) {
            sawSwapTransition = true;
            return;
        }
        if (!sawSwapTransition) {
            return;
        }
        // Terminal state reached.
        swapInFlight = false;
        isSwapping = false;
        if (status.state === 'ready') {
            toast.info(`Tagger swapped to ${status.source || 'the new model'}`);
        } else if (status.state === 'failed') {
            toast.error('Tagger swap failed — reverted to the previous model', {
                details: status.error ? [status.error] : undefined
            });
            // Rollback happened server-side; re-seed from the persisted truth.
            void fetchActiveTagger()
                .then((resp) => {
                    if (resp.active) {
                        active = {
                            ...resp.active,
                            source:
                                resp.active.kind === 'hf'
                                    ? `hf:${resp.active.repo_id}`
                                    : `local:${resp.active.model_path}`
                        };
                    }
                })
                .catch(() => {
                    /* best-effort re-sync */
                });
        }
    });

    // --- Tag autocomplete variant picker ---
    //
    // A single dropdown listing the suggestion catalog variants
    // (Anima / Illustrious / NoobAIXL). Applied immediately on change —
    // unlike the tagger model, this is a lightweight preference: the
    // backend persists and returns at once, reloading the new catalog in
    // the background. On failure the select reverts to the last
    // successfully-applied value and a toast explains why.

    let variantLoaded = $state(false);
    let variantError = $state<string | null>(null);
    let variantResponse = $state<SuggestionVariantResponse | null>(null);
    // Bound to the <select>; ``committedVariant`` tracks the last value the
    // backend accepted, so a failed switch can revert the dropdown.
    let variantValue = $state('');
    let committedVariant = $state('');
    let isVariantSwitching = $state(false);

    $effect(() => {
        if (variantLoaded) {
            return;
        }
        const controller = new AbortController();
        void (async () => {
            try {
                const resp = await fetchSuggestionVariant(controller.signal);
                variantResponse = resp;
                variantValue = resp.variant;
                committedVariant = resp.variant;
                variantLoaded = true;
            } catch (e) {
                if ((e as Error).name !== 'AbortError') {
                    variantError = (e as Error).message;
                    variantLoaded = true;
                }
            }
        })();
        return () => controller.abort();
    });

    function labelFor(value: string): string {
        return variantResponse?.variants.find((v) => v.value === value)?.label ?? value;
    }

    async function handleVariantChange(event: Event) {
        const next = (event.currentTarget as HTMLSelectElement).value;
        if (next === committedVariant || isVariantSwitching) {
            return;
        }
        isVariantSwitching = true;
        try {
            const resp = await setSuggestionVariantAction(next, labelFor(next));
            if (resp) {
                variantResponse = resp;
                variantValue = resp.variant;
                committedVariant = resp.variant;
            } else {
                // Revert — the backend rejected the switch (toast already shown).
                variantValue = committedVariant;
            }
        } finally {
            isVariantSwitching = false;
        }
    }
</script>

<div class="space-y-4 p-5">
    <section class="space-y-4">
        <h3 class="section-heading">Notifications</h3>

        {#if !notificationsSupported()}
            <p class="text-sm text-gray-500">
                Browser notifications are not supported in this environment.
            </p>
        {:else}
            <div class="flex items-start gap-3">
                <Checkbox id="settings-notifications" bind:checked={notificationsOn} />
                <div>
                    <label
                        class="cursor-pointer text-sm text-gray-300"
                        for="settings-notifications"
                    >
                        Browser notifications
                    </label>
                    <p class="mt-0.5 text-xs text-gray-500">
                        Get notified when captioning finishes or encounters an error.
                    </p>
                    {#if permStatus === 'denied'}
                        <p class="mt-1 text-xs text-yellow-400">
                            Notification permission is blocked. Enable it in your browser's site
                            settings.
                        </p>
                    {:else if permStatus === 'unsupported'}
                        <p class="mt-1 text-xs text-gray-500">
                            Notifications are not available in this browser.
                        </p>
                    {/if}
                </div>
            </div>
        {/if}
    </section>

    <section class="space-y-3">
        <h3 class="section-heading">Tagger model</h3>
        <p class="text-xs text-gray-500">
            Tags each image with general categories, characters, and a safety rating. Pick the model
            to use here.
        </p>

        {#if pickerError}
            <p class="text-sm text-yellow-400">Failed to load tagger catalog: {pickerError}</p>
        {:else if !pickerLoaded}
            <p class="text-xs text-gray-500">Loading…</p>
        {:else}
            <p class="text-xs text-gray-500">
                {#if !active}
                    No model selected.
                {:else}
                    Active:
                    <code class="text-gray-300">
                        {active.kind === 'hf' ? active.repo_id : active.model_path}
                    </code>
                    {#if active.preproc_profile}
                        <span class="text-gray-500">
                            (type <code class="text-gray-300">{active.preproc_profile}</code>)
                        </span>
                    {/if}
                {/if}
            </p>

            {#if anyBatchRunning}
                <div class="alert-warning">
                    A batch tagging job is running — stop it from the dataset view before swapping
                    the model.
                </div>
            {/if}

            <div>
                <label class="label mb-1 block" for="settings-tagger-model">Model</label>
                <div class="flex items-start gap-2">
                    <div class="min-w-0 flex-1">
                        <select
                            id="settings-tagger-model"
                            class="input"
                            bind:value={pickerSelectedId}
                            disabled={isSwapping || anyBatchRunning}
                        >
                            {#each pickerOptions as option (option.id)}
                                <option value={option.id}
                                    >{option.id === LOCAL_FILE_ID
                                        ? option.display
                                        : option.id}</option
                                >
                            {/each}
                        </select>
                        {#if isLocalSelected}
                            <p class="mx-2 mt-2 text-xs text-yellow-400">
                                Local files are expect to be in ONNX format. Match the type to how
                                the model was exported — wd-tagger for SmilingWolf-style, timm for
                                animetimm models.
                            </p>
                        {:else if selectedCatalog}
                            <p class="mx-2 mt-2 text-xs text-gray-500">
                                {selectedCatalog.display}

                                {#if selectedCatalog.params}
                                    ({selectedCatalog.params})
                                {/if}
                            </p>
                        {/if}
                    </div>
                    <button
                        class="btn-primary shrink-0"
                        onclick={handleSwap}
                        disabled={!canSwap}
                        title={anyBatchRunning
                            ? 'A batch tagging job is running — stop it before swapping the model.'
                            : undefined}
                    >
                        {#if !isSwapping}
                            Swap
                        {:else}
                            <SvgSpinner class="animate-spin"></SvgSpinner>
                        {/if}
                    </button>
                </div>
            </div>

            {#if isLocalSelected}
                <div class="space-y-2">
                    <div>
                        <label class="label mb-1 block" for="settings-tagger-local-model"
                            >Model path</label
                        >
                        <input
                            id="settings-tagger-local-model"
                            class="input"
                            placeholder={examplePath(
                                '/path/to/model.onnx',
                                'D:\\path\\to\\model.onnx',
                                $isWindows
                            )}
                            bind:value={localModelPath}
                            disabled={isSwapping}
                        />
                    </div>
                    <div>
                        <label class="label mb-1 block" for="settings-tagger-local-label"
                            >Label path (optional)</label
                        >
                        <input
                            id="settings-tagger-local-label"
                            class="input"
                            placeholder={examplePath(
                                '/path/to/selected_tags.csv',
                                'D:\\path\\to\\selected_tags.csv',
                                $isWindows
                            )}
                            bind:value={localLabelPath}
                            disabled={isSwapping}
                        />
                    </div>

                    <div class="grid grid-cols-2 gap-3">
                        <div>
                            <label class="label mb-1 block" for="settings-tagger-profile"
                                >Type</label
                            >
                            <select
                                id="settings-tagger-profile"
                                class="input"
                                bind:value={pickerProfile}
                                disabled={isSwapping}
                            >
                                {#each profiles as profile (profile)}
                                    <option value={profile}>{profile}</option>
                                {/each}
                            </select>
                            <p class="mt-1 text-xs text-gray-500">
                                {#if pickerProfile === 'wd-tagger'}
                                    Use <code class="text-gray-300">wd-tagger</code> when using a SmilingWolf
                                    (or derived) model.
                                {:else if pickerProfile === 'timm'}
                                    Use <code class="text-gray-300">timm</code> when using a animetimm
                                    (or derived) model.
                                {:else}
                                    Preprocessing pipeline must match how the model was exported.
                                    <code>wd-tagger</code> for SmilingWolf exports,
                                    <code>timm</code> for PyTorch-style exports.
                                {/if}
                            </p>
                        </div>
                        <div>
                            <label class="label mb-1 block" for="settings-tagger-size"
                                >Input size</label
                            >
                            <input
                                id="settings-tagger-size"
                                class="input"
                                type="number"
                                min="0"
                                step="1"
                                placeholder="auto"
                                bind:value={pickerSize}
                                disabled={isSwapping}
                            />
                            <p class="mt-1 text-xs text-gray-500">
                                Set to the model's expected image size. Leave at 0 (auto) when
                                unsure.
                            </p>
                        </div>
                    </div>
                </div>
            {/if}
        {/if}
    </section>

    <section class="space-y-3">
        <h3 class="section-heading">Tag autocomplete</h3>
        <p class="text-xs text-gray-500">
            Picks the tag catalog used for autocomplete in the Tags tab. Each variant is tuned for a
            different base model.
        </p>

        {#if variantError}
            <p class="text-sm text-yellow-400">Failed to load variants: {variantError}</p>
        {:else if !variantLoaded}
            <p class="text-xs text-gray-500">Loading…</p>
        {:else if variantResponse}
            <div>
                <label class="label mb-1 block" for="settings-suggestion-variant">Catalog</label>
                <div class="flex items-center gap-2">
                    <select
                        id="settings-suggestion-variant"
                        class="input"
                        bind:value={variantValue}
                        onchange={handleVariantChange}
                        disabled={isVariantSwitching}
                    >
                        {#each variantResponse.variants as option (option.value)}
                            <option value={option.value}>{option.label}</option>
                        {/each}
                    </select>
                    {#if isVariantSwitching}
                        <SvgSpinner class="animate-spin"></SvgSpinner>
                    {/if}
                </div>
                {#if variantResponse.variant !== variantResponse.default}
                    <p class="mx-2 mt-2 text-xs text-gray-500">
                        Default is {labelFor(variantResponse.default)}.
                    </p>
                {/if}
            </div>
        {/if}
    </section>
</div>
