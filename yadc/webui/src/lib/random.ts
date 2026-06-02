/**
 * A seeded pseudo-random number generator using sin-based hashing.
 * Returns values in [0, 1).
 */
export function random(seed: number | undefined = undefined) {
  let current = seed !== undefined ? seed : Math.random();

  return () => {
    current = Math.sin(current) * 99430;
    return current - Math.floor(current);
  };
}
