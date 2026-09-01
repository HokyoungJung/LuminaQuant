# Security Policy

## Supported Versions

Security fixes are applied to the active default branch.

## Reporting a Vulnerability

Please do **not** open public issues for security-sensitive reports.

Use one of:

1. GitHub Security Advisories (preferred)
2. Private maintainer contact path used by this repository owner

Include:

- impact summary
- reproduction steps
- affected files/paths
- suggested mitigation (if available)

We will acknowledge receipt and triage as quickly as possible.

## Secrets and Credentials

- Never commit API keys, tokens, or private DSNs.
- Use `.env` / environment variables for credentials.
- Rotate credentials immediately if leakage is suspected.
