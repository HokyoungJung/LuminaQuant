import { describe, expect, it } from 'vitest';

import { buildExactWindowEmptyState } from '../lib/exact-window-status';

describe('buildExactWindowEmptyState', () => {
  it('keeps missing bundle guidance explicit', () => {
    expect(buildExactWindowEmptyState('missing_bundle')).toEqual({
      message:
        'No exact-window artifact bundle is available yet. Run the existing exact-window workflow first, then refresh this page.',
      detail: null,
    });
  });

  it('preserves explicit load failure details', () => {
    expect(buildExactWindowEmptyState('load_failed', 'artifact parse failed')).toEqual({
      message: 'The exact-window bundle could not be loaded cleanly.',
      detail: 'artifact parse failed',
    });
  });

  it('provides a default troubleshooting detail for opaque load failures', () => {
    expect(buildExactWindowEmptyState('load_failed')).toEqual({
      message: 'The exact-window bundle could not be loaded cleanly.',
      detail: 'Check the latest artifact files and the Python compatibility bridge logs.',
    });
  });
});
