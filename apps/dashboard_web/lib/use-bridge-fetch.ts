'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

import { readJsonOrThrow } from '@/lib/bridge-fetch';

export interface BridgeFetchResult<T> {
  payload: T | null;
  error: string;
  loading: boolean;
  refetch: () => void;
  lastFetchedAt: string | null;
}

/**
 * Fetch hook for dashboard runtime components.
 *
 * Fires a GET request on mount (and every `pollMs` when polling is enabled),
 * stores the parsed JSON response, and exposes any error message plus a
 * manual `refetch`.  The previous payload is retained during refetches so
 * consumers can render stale-but-present data instead of a skeleton flash.
 *
 * Concurrency model: latest wins.  Every load takes a fresh generation from
 * `genRef`; state only commits when that generation is still current, so a
 * response for an outdated url (or an older poll tick) can never clobber a
 * newer one, and a url change always starts a fresh load immediately.  The
 * effect cleanup bumps the generation, which also invalidates in-flight
 * requests on unmount.
 */
export function useBridgeFetch<T>(
  url: string,
  errorLabel: string,
  opts?: { pollMs?: number },
): BridgeFetchResult<T> {
  const [payload, setPayload] = useState<T | null>(null);
  const [error, setError] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);
  const [lastFetchedAt, setLastFetchedAt] = useState<string | null>(null);
  const genRef = useRef(0);
  const pollMs = opts?.pollMs ?? 0;

  const load = useCallback(async () => {
    const gen = ++genRef.current;
    setLoading(true);
    try {
      const response = await fetch(url, { cache: 'no-store' });
      const body = await readJsonOrThrow<T>(response, errorLabel);
      if (gen === genRef.current) {
        setPayload(body);
        setError('');
      }
    } catch (fetchError: unknown) {
      if (gen === genRef.current) {
        setError(fetchError instanceof Error ? fetchError.message : String(fetchError));
      }
    } finally {
      if (gen === genRef.current) {
        setLoading(false);
        setLastFetchedAt(new Date().toISOString());
      }
    }
  }, [url, errorLabel]);

  useEffect(() => {
    void load();
    const timer = pollMs > 0 ? setInterval(() => void load(), pollMs) : undefined;
    return () => {
      // Invalidate any request this effect started: a stale response for the
      // previous url (or a request in flight at unmount) must never commit.
      genRef.current += 1;
      if (timer !== undefined) {
        clearInterval(timer);
      }
    };
  }, [load, pollMs]);

  const refetch = useCallback(() => {
    void load();
  }, [load]);

  return { payload, error, loading, refetch, lastFetchedAt };
}
