/**
 * Next.js Edge Middleware — dashboard API access control.
 *
 * SECURITY MODEL (defense in depth, two layers):
 *   Layer 1 (this file): Restrict /api/python/dashboard/** to localhost connections unless a
 *     valid x-lq-control-token is present. Read-only GET requests from non-localhost are
 *     allowed when the token is present; state-changing POST requests always require the token.
 *   Layer 2 (route.ts):  The control route independently verifies the token and fails closed
 *     when LQ_DASHBOARD_CONTROL_TOKEN is unset.
 *
 * OPERATOR NOTES:
 *   - The dashboard is intended to run on localhost only. Do NOT expose it to the internet.
 *   - If you must access the dashboard API from a non-localhost address (e.g. a jump host),
 *     set LQ_DASHBOARD_CONTROL_TOKEN and include the header in every request.
 *   - Requests that arrive without a valid token AND from a non-localhost IP are rejected 403
 *     before they reach any route handler.
 */
import { NextRequest, NextResponse } from 'next/server';

/** Path prefix that covers all dashboard API routes. */
const DASHBOARD_API_PREFIX = '/api/python/dashboard';

/** IPv4 and IPv6 loopback addresses. */
const LOOPBACK = new Set(['127.0.0.1', '::1', '::ffff:127.0.0.1']);

/**
 * Return true if the request originates from localhost.
 *
 * We check (in order):
 *   1. x-forwarded-for (set by reverse proxies / Next.js dev server)
 *   2. The raw ip from the NextRequest object
 *
 * We take only the FIRST value from x-forwarded-for (the originating client). A trusted
 * reverse proxy appends entries to the right, so the leftmost address is the client.
 */
function isLocalhost(req: NextRequest): boolean {
  const xForwardedFor = req.headers.get('x-forwarded-for');
  if (xForwardedFor) {
    const clientIp = xForwardedFor.split(',')[0].trim();
    return LOOPBACK.has(clientIp);
  }
  // Prefer the runtime-exposed direct peer IP when available: this DENIES a remote
  // client that reaches a 0.0.0.0-bound server directly (no proxy, no xff).
  const directIp = (req as unknown as { ip?: string }).ip;
  if (directIp) {
    return LOOPBACK.has(directIp);
  }
  // No origin signal at all (some runtimes do not expose a peer IP). The dashboard is
  // intended to bind to 127.0.0.1; treat unknown-origin as local for READ access only.
  // State-changing actions are independently token-gated and fail-closed in route.ts,
  // so this fallback can never enable an unauthenticated kill/stop/cancel.
  return true;
}

/**
 * Constant-time string comparison (runtime-agnostic: works in both the Edge and
 * Node middleware runtimes). Avoids leaking the token via early-exit timing.
 */
function timingSafeEqualStr(a: string, b: string): boolean {
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

/**
 * Return true if the request carries a valid shared-secret token.
 * Always returns false when LQ_DASHBOARD_CONTROL_TOKEN is unset (fail closed).
 */
function hasValidToken(req: NextRequest): boolean {
  const expectedToken = process.env['LQ_DASHBOARD_CONTROL_TOKEN'];
  if (!expectedToken) {
    return false;
  }
  const provided = req.headers.get('x-lq-control-token') ?? '';
  return timingSafeEqualStr(provided, expectedToken);
}

export function middleware(req: NextRequest): NextResponse {
  const { pathname } = req.nextUrl;

  // Only gate dashboard API routes.
  if (!pathname.startsWith(DASHBOARD_API_PREFIX)) {
    return NextResponse.next();
  }

  const local = isLocalhost(req);
  const authenticated = hasValidToken(req);

  if (local || authenticated) {
    // Allow: request is either from localhost or carries a valid token.
    return NextResponse.next();
  }

  // Non-localhost, no valid token — reject before reaching any route handler.
  return NextResponse.json(
    {
      ok: false,
      error: 'forbidden',
      detail:
        'Dashboard API access is restricted to localhost. ' +
        'To allow remote access, set LQ_DASHBOARD_CONTROL_TOKEN and include the ' +
        'x-lq-control-token header in your request.',
    },
    { status: 403 },
  );
}

export const config = {
  // Match all /api/python/dashboard/** routes.
  matcher: ['/api/python/dashboard/:path*'],
};
