/**
 * Unit tests for the workflow-jobs control route auth logic and the dashboard middleware.
 *
 * These tests exercise the auth/IP-check logic directly without spinning up Next.js.
 * They import helper functions extracted below via re-export; since the route and middleware
 * are not directly importable in this vitest environment (they reference Next.js edge
 * globals), we replicate the pure logic under test here to provide deterministic coverage.
 *
 * MANUAL VERIFICATION CHECKLIST (for environments where the Next.js server can run):
 *   1. With LQ_DASHBOARD_CONTROL_TOKEN unset:
 *      curl -X POST http://localhost:3000/api/python/dashboard/workflow-jobs/control \
 *        -H 'Content-Type: application/json' -d '{"job_id":"x","action":"kill"}'
 *      -> Expected: 401, error: "control_token_not_configured"
 *
 *   2. With LQ_DASHBOARD_CONTROL_TOKEN=secret, wrong token:
 *      curl -X POST http://localhost:3000/api/python/dashboard/workflow-jobs/control \
 *        -H 'Content-Type: application/json' -H 'x-lq-control-token: wrong' \
 *        -d '{"job_id":"x","action":"kill"}'
 *      -> Expected: 401, error: "unauthorized"
 *
 *   3. With LQ_DASHBOARD_CONTROL_TOKEN=secret, correct token:
 *      curl -X POST http://localhost:3000/api/python/dashboard/workflow-jobs/control \
 *        -H 'Content-Type: application/json' -H 'x-lq-control-token: secret' \
 *        -d '{"job_id":"x","action":"kill"}'
 *      -> Expected: 200 or 400 from backend (auth passed)
 *
 *   4. Middleware: from a non-localhost IP without token:
 *      curl -X GET http://<host>:3000/api/python/dashboard/workflow-jobs \
 *        -H 'x-forwarded-for: 203.0.113.1'
 *      -> Expected: 403, error: "forbidden"
 *
 *   5. Middleware: GET from localhost (no token needed):
 *      curl -X GET http://localhost:3000/api/python/dashboard/workflow-jobs
 *      -> Expected: 200 (read-only, no auth required from localhost)
 *
 *   6. Middleware (O6): remote peer spoofing a loopback x-forwarded-for:
 *      curl -X GET http://<beyond-loopback-host>:3000/api/python/dashboard/overview \
 *        -H 'x-forwarded-for: 127.0.0.1'
 *      -> Expected: 403 — the direct socket peer is remote, so the forged XFF is ignored.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';

// ---------------------------------------------------------------------------
// Replicated pure logic (mirrors route.ts and middleware.ts exactly)
// ---------------------------------------------------------------------------

const STATE_CHANGING_ACTIONS = new Set(['kill', 'stop', 'cancel']);

function checkControlToken(
  headers: Record<string, string>,
  action: string,
  envToken: string | undefined,
): { status: number; body: Record<string, unknown> } | null {
  if (!STATE_CHANGING_ACTIONS.has(action)) {
    return null; // read-only actions pass through
  }
  if (!envToken) {
    return {
      status: 401,
      body: { ok: false, error: 'control_token_not_configured' },
    };
  }
  const provided = headers['x-lq-control-token'] ?? '';
  if (provided !== envToken) {
    return { status: 401, body: { ok: false, error: 'unauthorized' } };
  }
  return null; // authenticated
}

const LOOPBACK = new Set(['127.0.0.1', '::1', '::ffff:127.0.0.1']);

// Mirrors middleware.ts isLocalhost (O6 trust model): x-forwarded-for is only trusted to
// prove locality when the DIRECT socket peer is itself loopback (a trusted local proxy).
function isLocalhost(headers: Record<string, string>, directIp?: string): boolean {
  const xff = headers['x-forwarded-for'];
  const forwardedClient = xff ? xff.split(',')[0].trim() : null;

  if (directIp) {
    if (!LOOPBACK.has(directIp)) {
      // Genuine remote peer — never trust its x-forwarded-for claim of loopback.
      return false;
    }
    return forwardedClient === null ? true : LOOPBACK.has(forwardedClient);
  }

  // No direct peer IP: x-forwarded-for cannot GRANT access, but a self-declared
  // non-loopback client is honored as a fail-closed reject.
  if (forwardedClient !== null && !LOOPBACK.has(forwardedClient)) {
    return false;
  }
  return true;
}

function middlewareDecision(
  headers: Record<string, string>,
  pathname: string,
  envToken: string | undefined,
  directIp?: string,
): { allowed: boolean; status?: number; error?: string } {
  if (!pathname.startsWith('/api/python/dashboard')) {
    return { allowed: true };
  }
  const local = isLocalhost(headers, directIp);
  const hasToken = !!envToken && (headers['x-lq-control-token'] ?? '') === envToken;
  if (local || hasToken) {
    return { allowed: true };
  }
  return { allowed: false, status: 403, error: 'forbidden' };
}

// ---------------------------------------------------------------------------
// checkControlToken — route-level auth
// ---------------------------------------------------------------------------

describe('checkControlToken (route.ts logic)', () => {
  it('returns null (passes) for non-state-changing actions regardless of token', () => {
    expect(checkControlToken({}, 'status', undefined)).toBeNull();
    expect(checkControlToken({}, 'list', 'secret')).toBeNull();
    expect(checkControlToken({}, '', undefined)).toBeNull();
  });

  it('returns 401 control_token_not_configured when env token is unset and action is kill', () => {
    const result = checkControlToken({}, 'kill', undefined);
    expect(result).not.toBeNull();
    expect(result!.status).toBe(401);
    expect(result!.body.error).toBe('control_token_not_configured');
  });

  it('returns 401 control_token_not_configured for stop when env token is unset', () => {
    const result = checkControlToken({}, 'stop', undefined);
    expect(result!.status).toBe(401);
    expect(result!.body.error).toBe('control_token_not_configured');
  });

  it('returns 401 control_token_not_configured for cancel when env token is unset', () => {
    const result = checkControlToken({}, 'cancel', undefined);
    expect(result!.status).toBe(401);
    expect(result!.body.error).toBe('control_token_not_configured');
  });

  it('returns 401 unauthorized when token is set but header is missing', () => {
    const result = checkControlToken({}, 'kill', 'mysecret');
    expect(result!.status).toBe(401);
    expect(result!.body.error).toBe('unauthorized');
  });

  it('returns 401 unauthorized when token is set but header value is wrong', () => {
    const result = checkControlToken({ 'x-lq-control-token': 'wrongtoken' }, 'kill', 'mysecret');
    expect(result!.status).toBe(401);
    expect(result!.body.error).toBe('unauthorized');
  });

  it('returns null (passes) when token matches for kill', () => {
    const result = checkControlToken({ 'x-lq-control-token': 'mysecret' }, 'kill', 'mysecret');
    expect(result).toBeNull();
  });

  it('returns null (passes) when token matches for stop', () => {
    const result = checkControlToken({ 'x-lq-control-token': 'tok' }, 'stop', 'tok');
    expect(result).toBeNull();
  });

  it('returns null (passes) when token matches for cancel', () => {
    const result = checkControlToken({ 'x-lq-control-token': 'tok' }, 'cancel', 'tok');
    expect(result).toBeNull();
  });

  it('rejects empty-string token even if both sides are empty (env unset treated as unset)', () => {
    // An empty LQ_DASHBOARD_CONTROL_TOKEN must still fail closed.
    const result = checkControlToken({ 'x-lq-control-token': '' }, 'kill', '');
    // Empty string is falsy — treated as unset, so fails with not_configured.
    // (The real code uses `if (!expectedToken)` which is true for ''.)
    expect(result!.status).toBe(401);
  });
});

// ---------------------------------------------------------------------------
// isLocalhost — middleware IP detection
// ---------------------------------------------------------------------------

describe('isLocalhost (middleware.ts logic)', () => {
  it('returns true when no x-forwarded-for header (direct connection, no proxy)', () => {
    // No xff = request arrived directly to the Node process = localhost context.
    expect(isLocalhost({})).toBe(true);
  });

  it('recognises 127.0.0.1 via x-forwarded-for', () => {
    expect(isLocalhost({ 'x-forwarded-for': '127.0.0.1' })).toBe(true);
  });

  it('recognises ::1 via x-forwarded-for', () => {
    expect(isLocalhost({ 'x-forwarded-for': '::1' })).toBe(true);
  });

  it('recognises ::ffff:127.0.0.1 (IPv4-mapped) via x-forwarded-for', () => {
    expect(isLocalhost({ 'x-forwarded-for': '::ffff:127.0.0.1' })).toBe(true);
  });

  it('rejects a public IP in x-forwarded-for', () => {
    expect(isLocalhost({ 'x-forwarded-for': '203.0.113.1' })).toBe(false);
  });

  it('uses the first (leftmost) value of x-forwarded-for', () => {
    expect(isLocalhost({ 'x-forwarded-for': '127.0.0.1, 10.0.0.1' })).toBe(true);
    expect(isLocalhost({ 'x-forwarded-for': '203.0.113.1, 127.0.0.1' })).toBe(false);
  });

  it('trims whitespace around x-forwarded-for entries', () => {
    expect(isLocalhost({ 'x-forwarded-for': '  127.0.0.1  , 10.0.0.1' })).toBe(true);
  });

  // ---- O6: x-forwarded-for spoof prevention (direct peer IP dimension) ----

  it('O6: rejects a remote peer that spoofs x-forwarded-for: 127.0.0.1', () => {
    // A genuine remote client reaches a beyond-loopback-bound server and forges XFF.
    // The direct peer IP (203.0.113.1) is authoritative — the spoof must NOT grant local.
    expect(isLocalhost({ 'x-forwarded-for': '127.0.0.1' }, '203.0.113.1')).toBe(false);
    expect(isLocalhost({ 'x-forwarded-for': '::1' }, '203.0.113.1')).toBe(false);
    expect(isLocalhost({ 'x-forwarded-for': '::ffff:127.0.0.1' }, '198.51.100.7')).toBe(false);
  });

  it('O6: rejects a remote peer regardless of x-forwarded-for value', () => {
    expect(isLocalhost({}, '203.0.113.1')).toBe(false);
    expect(isLocalhost({ 'x-forwarded-for': '203.0.113.1' }, '203.0.113.1')).toBe(false);
    expect(isLocalhost({ 'x-forwarded-for': '127.0.0.1, 203.0.113.1' }, '10.8.0.2')).toBe(false);
  });

  it('O6: honors x-forwarded-for only when the direct peer is a loopback proxy', () => {
    // Trusted local reverse proxy (peer is loopback) forwards the true client.
    expect(isLocalhost({ 'x-forwarded-for': '127.0.0.1' }, '127.0.0.1')).toBe(true);
    expect(isLocalhost({ 'x-forwarded-for': '::1' }, '::1')).toBe(true);
    // Loopback proxy forwarding a REMOTE client => not localhost.
    expect(isLocalhost({ 'x-forwarded-for': '203.0.113.1' }, '127.0.0.1')).toBe(false);
    expect(isLocalhost({ 'x-forwarded-for': '203.0.113.1, 127.0.0.1' }, '::1')).toBe(false);
    // Loopback peer with no forwarded header => the peer itself is the client.
    expect(isLocalhost({}, '127.0.0.1')).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// middlewareDecision — end-to-end middleware path decisions
// ---------------------------------------------------------------------------

describe('middlewareDecision (middleware.ts logic)', () => {
  it('allows non-dashboard routes unconditionally', () => {
    const r = middlewareDecision(
      { 'x-forwarded-for': '203.0.113.1' },
      '/api/other/route',
      undefined,
    );
    expect(r.allowed).toBe(true);
  });

  it('allows direct (no proxy) requests without a token — no xff means local', () => {
    const r = middlewareDecision({}, '/api/python/dashboard/workflow-jobs', undefined);
    expect(r.allowed).toBe(true);
  });

  it('allows 127.0.0.1 in xff without a token (read-only GET path)', () => {
    const r = middlewareDecision(
      { 'x-forwarded-for': '127.0.0.1' },
      '/api/python/dashboard/workflow-jobs',
      undefined,
    );
    expect(r.allowed).toBe(true);
  });

  it('allows ::1 in xff without a token', () => {
    const r = middlewareDecision(
      { 'x-forwarded-for': '::1' },
      '/api/python/dashboard/overview',
      undefined,
    );
    expect(r.allowed).toBe(true);
  });

  it('rejects non-localhost xff without token (env unset)', () => {
    const r = middlewareDecision(
      { 'x-forwarded-for': '203.0.113.1' },
      '/api/python/dashboard/workflow-jobs/control',
      undefined,
    );
    expect(r.allowed).toBe(false);
    expect(r.status).toBe(403);
    expect(r.error).toBe('forbidden');
  });

  it('rejects non-localhost xff with wrong token', () => {
    const r = middlewareDecision(
      { 'x-forwarded-for': '203.0.113.1', 'x-lq-control-token': 'wrong' },
      '/api/python/dashboard/workflow-jobs/control',
      'correct',
    );
    expect(r.allowed).toBe(false);
    expect(r.status).toBe(403);
  });

  it('allows non-localhost xff with correct token', () => {
    const r = middlewareDecision(
      { 'x-forwarded-for': '203.0.113.1', 'x-lq-control-token': 'correct' },
      '/api/python/dashboard/workflow-jobs/control',
      'correct',
    );
    expect(r.allowed).toBe(true);
  });

  it('regression: leftmost xff entry is used — spoofed loopback at end does not help', () => {
    // A reverse proxy setup: real client IP is first, proxy appends its own address.
    // xff = "203.0.113.1, 127.0.0.1" — leftmost (203.0.113.1) is the real client.
    const r = middlewareDecision(
      { 'x-forwarded-for': '203.0.113.1, 127.0.0.1' },
      '/api/python/dashboard/workflow-jobs/control',
      undefined,
    );
    expect(r.allowed).toBe(false);
    expect(r.status).toBe(403);
  });

  it('O6: rejects a remote peer that spoofs x-forwarded-for: 127.0.0.1 (read path, no token)', () => {
    // GET/read on a beyond-loopback server: direct peer is remote, XFF forges loopback.
    const r = middlewareDecision(
      { 'x-forwarded-for': '127.0.0.1' },
      '/api/python/dashboard/overview',
      undefined,
      '203.0.113.1',
    );
    expect(r.allowed).toBe(false);
    expect(r.status).toBe(403);
    expect(r.error).toBe('forbidden');
  });

  it('O6: spoofed loopback XFF still needs a valid token to reach control route', () => {
    const spoofHeaders = { 'x-forwarded-for': '::1', 'x-lq-control-token': 'wrong' };
    const denied = middlewareDecision(
      spoofHeaders,
      '/api/python/dashboard/workflow-jobs/control',
      'correct',
      '203.0.113.1',
    );
    expect(denied.allowed).toBe(false);
    expect(denied.status).toBe(403);
    // The same remote peer WITH the correct token is allowed (token, not the spoof).
    const allowed = middlewareDecision(
      { 'x-forwarded-for': '::1', 'x-lq-control-token': 'correct' },
      '/api/python/dashboard/workflow-jobs/control',
      'correct',
      '203.0.113.1',
    );
    expect(allowed.allowed).toBe(true);
  });

  it('O6: a trusted loopback proxy forwarding a localhost client is allowed', () => {
    const r = middlewareDecision(
      { 'x-forwarded-for': '127.0.0.1' },
      '/api/python/dashboard/overview',
      undefined,
      '127.0.0.1',
    );
    expect(r.allowed).toBe(true);
  });
});
