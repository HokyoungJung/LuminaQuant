/**
 * Auth utilities that are safe to use in both the Edge and Node runtimes.
 * No node:crypto — only Web Crypto / TextEncoder which are available everywhere.
 */

/**
 * Constant-time string comparison — avoids leaking the token via early-exit timing.
 *
 * Encodes both strings as UTF-8, XORs every byte pair, and ORs the results so
 * the comparison takes the same number of operations regardless of where strings
 * first differ.  Returns false for strings of different lengths without leaking
 * length via timing (the length check itself reveals length, which is acceptable
 * for fixed-size tokens).
 */
export function timingSafeEqualStr(a: string, b: string): boolean {
  const encoder = new TextEncoder();
  const aBytes = encoder.encode(a);
  const bBytes = encoder.encode(b);
  if (aBytes.length !== bBytes.length) {
    return false;
  }
  let diff = 0;
  for (let i = 0; i < aBytes.length; i += 1) {
    diff |= aBytes[i] ^ bBytes[i];
  }
  return diff === 0;
}
