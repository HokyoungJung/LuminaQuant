/**
 * Shared validation for user-controlled query params that end up in the
 * Python bridge subprocess argv.
 *
 * The pattern forbids a leading dash so argparse can never mistake a value
 * for a flag (e.g. run_id=--help), and restricts the charset to the shapes
 * run ids and symbols actually take (BTC/USDT, run_20260710_120000, ...).
 * Invalid values yield null; callers omit the flag and fall back to the
 * latest run instead of surfacing a subprocess failure.
 */
const SAFE_ARG_PATTERN = /^[A-Za-z0-9][A-Za-z0-9/:._-]*$/;

function sanitizeArg(value: string | null | undefined): string | null {
  const trimmed = value?.trim() ?? '';
  if (trimmed === '' || !SAFE_ARG_PATTERN.test(trimmed)) {
    return null;
  }
  return trimmed;
}

/** Validate a run_id query param; null when absent or unsafe for argv. */
export function sanitizeRunId(value: string | null | undefined): string | null {
  return sanitizeArg(value);
}

/** Validate a symbol query param; null when absent or unsafe for argv. */
export function sanitizeSymbol(value: string | null | undefined): string | null {
  return sanitizeArg(value);
}
