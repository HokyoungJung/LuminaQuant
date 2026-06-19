'use client';

import { useEffect, useState } from 'react';

import { readJsonOrThrow } from '@/lib/bridge-fetch';

/**
 * One-shot fetch hook for dashboard runtime components.
 *
 * Fires a single GET request on mount, stores the parsed JSON response, and
 * exposes any error message.  The active-flag cancel guard prevents
 * setState-after-unmount races: state setters are only called when the
 * component is still mounted.
 */
export function useBridgeFetch<T>(url: string, errorLabel: string): { payload: T | null; error: string } {
  const [payload, setPayload] = useState<T | null>(null);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    let active = true;
    fetch(url, { cache: 'no-store' })
      .then(async (response) => {
        const body = await readJsonOrThrow<T>(response, errorLabel);
        if (active) {
          setPayload(body);
        }
      })
      .catch((fetchError: unknown) => {
        if (active) {
          setError(fetchError instanceof Error ? fetchError.message : String(fetchError));
        }
      });
    return () => {
      active = false;
    };
  }, [url, errorLabel]);

  return { payload, error };
}
