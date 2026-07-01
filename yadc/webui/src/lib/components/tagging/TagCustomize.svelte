<script lang="ts">
    import { TAG_TIERS, type TagTier } from '$lib/stores/tagging';
    import TierList from './TierList.svelte';

    interface TierConfig {
        tier: TagTier;
        label: string;
        description: string;
    }

    // Order matters only for display; starred leads because it's the
    // strongest emphasis tier.
    const TIERS: readonly TierConfig[] = [
        {
            tier: TAG_TIERS.starred,
            label: 'Starred',
            description:
                'Stared tags will always appear at the top of the list of tags. Use this for common tags your may frequently need to add to your images.'
        },
        {
            tier: TAG_TIERS.desired,
            label: 'Desired',
            description: 'Highlighted with a green color wherever the tag appears.'
        },
        {
            tier: TAG_TIERS.undesired,
            label: 'Undesired',
            description: 'Highlighted with a red color wherever the tag appears.'
        }
    ];
</script>

<div class="space-y-5">
    <div>
        <h3 class="mb-1 text-sm font-medium text-gray-300">Customise highlights</h3>
        <p class="text-xs text-gray-500">
            Curate tags into tiers to emphasise them in the per-image prune grid. Lists are global
            (saved in your browser) and matched by name across every dataset and image. A tag can
            only belong to one tier.
        </p>
    </div>

    <div class="flex flex-col gap-6">
        {#each TIERS as cfg (cfg.tier)}
            <div class="space-y-1">
                <TierList
                    tier={cfg.tier}
                    label={cfg.label}
                    description={cfg.description}
                    allowCategoryOverride={cfg.tier === TAG_TIERS.starred}
                />
            </div>
        {/each}
    </div>
</div>
