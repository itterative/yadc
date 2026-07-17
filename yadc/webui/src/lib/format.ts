export function formatBytes(bytes: number): string {
    if (bytes === 0) {
        return '0 B';
    }
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

/** Format a remaining-time estimate from seconds into a human-readable string. */
export function formatEta(seconds: number): string {
    if (seconds < 60) {
        return `${seconds}s`;
    }
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    if (m < 60) {
        return `${m}m ${s}s`;
    }
    const h = Math.floor(m / 60);
    const rm = m % 60;
    return `${h}h ${rm}m`;
}

/** Format an epoch timestamp (seconds) as a relative "ago" string. */
export function formatRelativeTime(epochSeconds: number): string {
    const now = Date.now() / 1000;
    const diff = now - epochSeconds;
    if (diff < 60) {
        return 'just now';
    }
    if (diff < 3600) {
        return `${Math.floor(diff / 60)}m ago`;
    }
    if (diff < 86400) {
        return `${Math.floor(diff / 3600)}h ago`;
    }
    return `${Math.floor(diff / 86400)}d ago`;
}

/** Format an epoch timestamp (seconds) as a locale date-time string. */
export function formatDateTime(epochSeconds: number): string {
    return new Date(epochSeconds * 1000).toLocaleString();
}

/** Detect if the user is on Windows based on the platform/user-agent. */
export function isWindows(): boolean {
    if (typeof navigator === 'undefined') {
        return false;
    }
    return (
        navigator.platform.toLowerCase().includes('win') ||
        navigator.userAgent.toLowerCase().includes('win')
    );
}

/** Return a platform-appropriate example path. */
export function examplePath(unix: string, windows: string): string {
    return isWindows() ? windows : unix;
}
