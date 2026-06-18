/**
 * WARNING: This endpoint executes state-changing actions (kill/stop/cancel) on live workflow
 * jobs, including sending SIGTERM to live trading processes. It MUST be protected by a shared
 * secret and MUST NOT be exposed beyond localhost without the LQ_DASHBOARD_CONTROL_TOKEN env
 * var set. The companion middleware.ts enforces the localhost restriction; this route adds a
 * second layer (defense in depth) by checking the x-lq-control-token header on every
 * state-changing request.
 *
 * OPERATOR SETUP:
 *   1. Generate a random token:  node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
 *   2. Set it in your environment:  LQ_DASHBOARD_CONTROL_TOKEN=<token>
 *   3. Pass it in every control request:  -H "x-lq-control-token: <token>"
 *
 * If LQ_DASHBOARD_CONTROL_TOKEN is unset, all state-changing actions (kill/stop/cancel) are
 * REJECTED with 401. This is intentional fail-closed behavior — an unprotected kill endpoint
 * is worse than a temporarily unavailable one.
 */
import { NextResponse } from 'next/server';
import { execFileSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const REPO_ROOT = resolve(__dirname, '../../../../../../../');

export const dynamic = 'force-dynamic';

/** Actions that modify live-trading process state and require authentication. */
const STATE_CHANGING_ACTIONS = new Set(['kill', 'stop', 'cancel']);

/**
 * Constant-time string comparison — avoids leaking the token via early-exit timing.
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
 * Verify the shared-secret token for state-changing actions.
 *
 * Returns null on success, or a NextResponse with an appropriate error status/body on failure.
 */
function checkControlToken(request: Request, action: string): NextResponse | null {
  if (!STATE_CHANGING_ACTIONS.has(action)) {
    // Read-only or unknown actions are not gated here; the backend handles them.
    return null;
  }

  const expectedToken = process.env['LQ_DASHBOARD_CONTROL_TOKEN'];
  if (!expectedToken) {
    // Fail CLOSED: if the operator has not configured a token, refuse all kill/stop/cancel
    // actions rather than allowing unauthenticated process termination.
    return NextResponse.json(
      {
        ok: false,
        error: 'control_token_not_configured',
        detail:
          'LQ_DASHBOARD_CONTROL_TOKEN is not set. State-changing control actions (kill/stop/cancel) ' +
          'are disabled until an operator configures this environment variable. ' +
          'Run: node -e "console.log(require(\'crypto\').randomBytes(32).toString(\'hex\'))" ' +
          'to generate a token, then set LQ_DASHBOARD_CONTROL_TOKEN=<token> and restart the dashboard.',
      },
      { status: 401 },
    );
  }

  const providedToken = request.headers.get('x-lq-control-token') ?? '';
  if (!timingSafeEqualStr(providedToken, expectedToken)) {
    return NextResponse.json(
      {
        ok: false,
        error: 'unauthorized',
        detail:
          'Missing or invalid x-lq-control-token header. ' +
          'Set the header to the value of LQ_DASHBOARD_CONTROL_TOKEN.',
      },
      { status: 401 },
    );
  }

  return null; // authenticated
}

export async function POST(request: Request) {
  try {
    const payload = (await request.json()) as { job_id?: string; action?: string };
    const action = String(payload.action || '');

    const authError = checkControlToken(request, action);
    if (authError) {
      return authError;
    }

    const stdout = execFileSync(
      'uv',
      ['run', 'python', '-c', `
from lumina_quant.dashboard.workflow_jobs_service import control_workflow_job
import json
payload = control_workflow_job(dsn=None, job_id=${JSON.stringify(String(payload.job_id || ''))}, action=${JSON.stringify(action)})
print(json.dumps(payload, sort_keys=True))
`],
      {
        cwd: REPO_ROOT,
        encoding: 'utf-8',
      },
    );
    const result = JSON.parse(stdout.trim()) as Record<string, unknown>;
    return NextResponse.json(result, { status: result.ok ? 200 : 400 });
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { ok: false, error: 'workflow_job_control_failed', detail },
      { status: 500 },
    );
  }
}
