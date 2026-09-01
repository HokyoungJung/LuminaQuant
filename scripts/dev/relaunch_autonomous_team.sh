#!/usr/bin/env bash
set -euo pipefail

cd /home/hoky/Quants-agent/LuminaQuant
omx team ralph 1:executor "Continue the autonomous portfolio research loop from docs/autonomous-research-resume-2026-03-15.md and the latest follow-up artifacts under var/reports/exact_window_backtests/followup_status/. Keep one heavy lane at a time, stay under 8 GiB RSS, compare new challengers against the current incumbent baseline from portfolio_one_shot_current_opt/portfolio_optimization_latest.json, keep winners, discard losers, and keep experiments.tsv/research_state/probe/decision artifacts up to date without pausing for user confirmation."
