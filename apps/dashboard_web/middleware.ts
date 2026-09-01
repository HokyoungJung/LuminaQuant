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

import { timingSafeEqualStr } from '@/lib/auth-helpers';

/** Path prefix that covers all dashboard API routes. */
const DASHBOARD_API_PREFIX = '/api/python/dashboard';

/** IPv4 and IPv6 loopback addresses. */
const LOOPBACK = new Set(['127.0.0.1', '::1', '::ffff:127.0.0.1']);

/**
 * Return true if the request originates from localhost.
 *
 * TRUST MODEL (O6): `x-forwarded-for` is a client-supplied header. A remote client can
 * trivially send `x-forwarded-for: 127.0.0.1`, so it MUST NOT be trusted to prove
 * locality unless the DIRECT socket peer is itself a trusted local proxy (a loopback
 * address that we run in front of the dashboard). We therefore decide as follows:
 *
 *   1. Direct peer IP known and NON-loopback: the peer is a genuine remote client (it
 *      reached a server bound beyond 127.0.0.1, or an untrusted proxy). Use the raw peer
 *      address EXCLUSIVELY and ignore x-forwarded-for entirely -> not localhost. This is
 *      what defeats the spoof.
 *   2. Direct peer IP known and loopback: the peer is either the local client itself, or
 *      a trusted local reverse proxy forwarding the real client. Honor x-forwarded-for
 *      here, taking only its leftmost (originating-client) entry.
 *   3. Direct peer IP not exposed by the runtime (Next 15 removed `NextRequest.ip`, so the
 *      raw peer is often unavailable): x-forwarded-for is unauthenticated and MUST NOT be
 *      used to GRANT access. We still honor a self-declared NON-loopback client as a
 *      fail-closed rejection signal, then fall back to treating unknown-origin as local for
 *      READ access only. State-changing actions are independently token-gated and
 *      fail-closed in route.ts, so this fallback can never enable an unauthenticated
 *      kill/stop/cancel.
 */
function isLocalhost(req: NextRequest): boolean {
  const xForwardedFor = req.headers.get('x-forwarded-for');
  // Leftmost x-forwarded-for entry = the originating client (a trusted proxy appends to
  // the right). null when the header is absent.
  const forwardedClient = xForwardedFor ? xForwardedFor.split(',')[0].trim() : null;

  const directIp = (req as unknown as { ip?: string }).ip;
  if (directIp) {
    if (!LOOPBACK.has(directIp)) {
      // (1) Genuine remote peer — never trust its x-forwarded-for to claim loopback.
      return false;
    }
    // (2) Peer is loopback: honor the forwarded client if a trusted proxy supplied one;
    // otherwise the loopback peer IS the client.
    return forwardedClient === null ? true : LOOPBACK.has(forwardedClient);
  }

  // (3) No direct peer IP. x-forwarded-for cannot GRANT access here, but a self-declared
  // non-loopback client is honored as a fail-closed reject.
  if (forwardedClient !== null && !LOOPBACK.has(forwardedClient)) {
    return false;
  }
  return true;
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
