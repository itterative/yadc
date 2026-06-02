import { browser } from "$app/environment";

/**
 * Debounce a callback — only fires after `delay` ms of inactivity.
 */
export function deferred<T extends (...args: any[]) => void>(
  cb: T,
  delay: number = 10,
): (...args: Parameters<T>) => void {
  let cbTimeout: number | null = null;

  return (...args: Parameters<T>) => {
    if (cbTimeout) {
      window.clearTimeout(cbTimeout);
    }

    cbTimeout = window.setTimeout(() => cb(...args), delay);
  };
}

/**
 * Wraps an async function so calls are serialized — each invocation
 * waits for the previous one to complete before starting.
 */
export function synchronized<R, T extends (...args: any[]) => Promise<R>>(cb: T) {
  const promises: Promise<R>[] = [];

  return async (...args: Parameters<T>) => {
    if (promises.length) {
      await Promise.all(promises);
    }

    const promise = cb(...args);
    promises.push(promise);

    try {
      return await promise;
    } finally {
      const promiseIndex = promises.indexOf(promise);
      if (promiseIndex >= 0) {
        promises.splice(promiseIndex, 1);
      }
    }
  };
}

/** Sleep for the given number of seconds. */
export function sleep(delay: number): Promise<void> {
  if (!browser) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => window.setTimeout(resolve, delay * 1000));
}

/**
 * Delays execution of an async callback by `delay` seconds.
 */
export function delayed<R, T extends (...args: any[]) => Promise<R>>(cb: T, delay: number) {
  return async (...args: Parameters<T>) => {
    await sleep(delay);
    await cb();
  };
}
