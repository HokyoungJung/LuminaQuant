export interface ExactWindowEmptyState {
  message: string;
  detail: string | null;
}

export function buildExactWindowEmptyState(status: string, error?: string | null): ExactWindowEmptyState {
  switch (status) {
    case 'missing_bundle':
      return {
        message:
          'No exact-window artifact bundle is available yet. Run the existing exact-window workflow first, then refresh this page.',
        detail: error ?? null,
      };
    case 'missing_summary':
      return {
        message: 'The exact-window bundle exists, but the latest summary artifact is missing.',
        detail: error ?? null,
      };
    case 'load_failed':
      return {
        message: 'The exact-window bundle could not be loaded cleanly.',
        detail: error ?? 'Check the latest artifact files and the Python compatibility bridge logs.',
      };
    default:
      return {
        message: 'No exact-window payload is available yet.',
        detail: error ?? null,
      };
  }
}
