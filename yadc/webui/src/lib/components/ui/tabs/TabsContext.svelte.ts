import { getContext, setContext, type Component } from 'svelte';

const KEY = Symbol('tabs');

export interface TabItem {
    id: string;
    label: string;
    icon?: Component<{ class?: string }>;
}

export interface TabsState {
    readonly tabs: readonly TabItem[];
    readonly activeIndex: number;
    registerTab(id: string, label: string, icon?: Component<{ class?: string }>): number;
    setActiveIndex(index: number): void;
}

export function createTabsState(): TabsState {
    const tabs = $state<TabItem[]>([]);
    let activeIndex = $state(0);

    return {
        get tabs() {
            return tabs;
        },
        get activeIndex() {
            return activeIndex;
        },
        registerTab(id: string, label: string, icon?: Component<{ class?: string }>) {
            const index = tabs.length;
            tabs.push({ id, label, icon });
            return index;
        },
        setActiveIndex(index: number) {
            activeIndex = index;
        }
    };
}

export function setTabsContext(state: TabsState) {
    setContext(KEY, state);
}

export function getTabsContext(): TabsState {
    const ctx = getContext<TabsState | undefined>(KEY);
    if (!ctx) {
        throw new Error('Tab must be used inside a Tabs component');
    }
    return ctx;
}
