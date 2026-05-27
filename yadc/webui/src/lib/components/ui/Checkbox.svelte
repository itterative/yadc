<script lang="ts">
    interface Props {
        id: string;
        checked: boolean;
    }

    let { id, checked = $bindable(false) }: Props = $props();
</script>

<label class="relative inline-flex cursor-pointer items-start" for={id}>
    <input
        type="checkbox"
        bind:checked
        class="peer h-5 w-5 cursor-pointer appearance-none rounded border border-slate-300 shadow transition-all checked:border-slate-800 checked:bg-slate-800 hover:shadow-md"
        {id}
    />
    <span
        class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 transform text-white opacity-0 peer-checked:opacity-100"
    >
        <svg
            xmlns="http://www.w3.org/2000/svg"
            class="h-3.5 w-3.5"
            viewBox="0 0 20 20"
            fill="currentColor"
            stroke="currentColor"
            stroke-width="1"
        >
            <path
                fill-rule="evenodd"
                d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                clip-rule="evenodd"
            ></path>
        </svg>
    </span>
</label>
