import type { BridgeErrorBody } from '@/lib/dashboard-contracts';

export async function readJsonOrThrow<T>(response: Response, fallbackMessage: string): Promise<T> {
  if (!response.ok) {
    // Error bodies may be non-JSON (gateway HTML, empty 502s); never let a
    // SyntaxError mask the HTTP failure.
    let detail: string | undefined;
    try {
      const errorBody = (await response.json()) as BridgeErrorBody;
      detail = errorBody.detail ?? errorBody.error;
    } catch {
      detail = undefined;
    }
    throw new Error(detail ?? `${fallbackMessage}: HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}
