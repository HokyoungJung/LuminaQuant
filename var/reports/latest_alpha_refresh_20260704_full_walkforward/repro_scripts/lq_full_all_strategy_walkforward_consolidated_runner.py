#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path('/home/hoky/Quants-agent/LuminaQuant')
PYTHON = REPO / '.venv/bin/python'
RUNNER = REPO / 'scripts/run_research_candidates.py'
BASE = REPO / 'var/reports/latest_alpha_refresh_20260704_full_walkforward'
MANIFEST_ROOT = REPO / 'var/reports/ultragoal_full_pool_strategy'
SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'TRX/USDT',
    'XRP/USDT', 'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT',
]
FOLDS = [
    {'fold_id': 'WF202604', 'train_start': '2025-01-01', 'train_end': '2026-01-31T23:59:59', 'validation_start': '2026-02-01', 'validation_end': '2026-03-31T23:59:59', 'oos_start': '2026-04-01', 'oos_end': '2026-04-30T23:59:59'},
    {'fold_id': 'WF202605', 'train_start': '2025-01-01', 'train_end': '2026-02-28T23:59:59', 'validation_start': '2026-03-01', 'validation_end': '2026-04-30T23:59:59', 'oos_start': '2026-05-01', 'oos_end': '2026-05-31T23:59:59'},
    {'fold_id': 'WF202606', 'train_start': '2025-01-01', 'train_end': '2026-03-31T23:59:59', 'validation_start': '2026-04-01', 'validation_end': '2026-05-31T23:59:59', 'oos_start': '2026-06-01', 'oos_end': '2026-06-30T23:59:59'},
    {'fold_id': 'WF202607_PARTIAL', 'train_start': '2025-01-01', 'train_end': '2026-04-30T23:59:59', 'validation_start': '2026-05-01', 'validation_end': '2026-06-30T23:59:59', 'oos_start': '2026-07-01', 'oos_end': '2026-07-04T00:00:00'},
]
SHARDS = [
    {'shard_id': '30m', 'timeframe': '30m', 'manifest': MANIFEST_ROOT / 'g005_walkforward_candidate_manifest_30m.json'},
    {'shard_id': '4h', 'timeframe': '4h', 'manifest': MANIFEST_ROOT / 'g005_walkforward_candidate_manifest_4h.json'},
    {'shard_id': '1d', 'timeframe': '1d', 'manifest': MANIFEST_ROOT / 'g005_walkforward_candidate_manifest_1d.json'},
    {'shard_id': '1h_all', 'timeframe': '1h', 'manifest': BASE / 'g005_walkforward_candidate_manifest_1h_no_alpha101_timeout_filtered.json'},
]
MAX_PARALLEL = 4

def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

def emit(event: str, **payload) -> None:
    print(json.dumps({'event': event, 'ts': now(), **payload}, sort_keys=True), flush=True)

def manifest_count(path: Path) -> int:
    try:
        payload = json.loads(path.read_text('utf-8'))
        return len(list(payload.get('candidates') or []))
    except Exception:
        return -1

def latest_summary(path: Path) -> dict:
    report_path = path / 'candidate_research_latest.json'
    if not report_path.exists():
        return {'report_exists': False}
    payload = json.loads(report_path.read_text('utf-8'))
    return {
        'report_exists': True,
        'candidate_count': len(list(payload.get('candidates') or [])),
        'stage1': payload.get('stage1') or {},
        'split': payload.get('split') or {},
    }

def run_one(fold: dict, shard: dict) -> dict:
    fold_id = fold['fold_id']
    shard_id = shard['shard_id']
    outdir = BASE / fold_id / shard_id
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / 'runner_stdout_stderr.log'
    latest_path = outdir / 'candidate_research_latest.json'
    if latest_path.exists():
        result = {'fold_id': fold_id, 'shard_id': shard_id, 'timeframe': shard['timeframe'], 'returncode': 0, 'elapsed_seconds': 0.0, 'output_dir': str(outdir), 'log_path': str(log_path), **latest_summary(outdir), 'skipped_existing': True}
        emit('shard_skipped_existing', **result)
        return result
    cmd = [
        str(PYTHON), str(RUNNER),
        '--manifest', str(shard['manifest']),
        '--output-dir', str(outdir),
        '--symbols', *SYMBOLS,
        '--timeframes', str(shard['timeframe']),
        '--base-timeframe', '30m',
        '--skip-coverage-rebuild',
        '--stage1-keep-ratio', '1.0',
        '--max-candidates', '9999',
        '--top-k', '80',
        '--train-start', fold['train_start'],
        '--train-end', fold['train_end'],
        '--validation-start', fold['validation_start'],
        '--validation-end', fold['validation_end'],
        '--oos-start', fold['oos_start'],
        '--oos-end', fold['oos_end'],
    ]
    emit('shard_start', fold_id=fold_id, shard_id=shard_id, timeframe=shard['timeframe'], manifest_count=manifest_count(shard['manifest']), output_dir=str(outdir))
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    log_path.write_text(proc.stdout or '', 'utf-8')
    result = {'fold_id': fold_id, 'shard_id': shard_id, 'timeframe': shard['timeframe'], 'returncode': proc.returncode, 'elapsed_seconds': round(elapsed, 3), 'output_dir': str(outdir), 'log_path': str(log_path), **latest_summary(outdir)}
    if proc.returncode == 0:
        emit('shard_done', **result)
    else:
        emit('shard_failed', **result, output_tail='\n'.join((proc.stdout or '').splitlines()[-30:]))
    return result

def main() -> int:
    BASE.mkdir(parents=True, exist_ok=True)
    tasks = [(fold, shard) for fold in FOLDS for shard in SHARDS]
    emit('full_walkforward_consolidated_start', output_root=str(BASE), fold_count=len(FOLDS), shard_count=len(SHARDS), task_count=len(tasks), manifest_candidate_count_per_fold=sum(max(0, manifest_count(s['manifest'])) for s in SHARDS), max_parallel=MAX_PARALLEL)
    results=[]; failures=[]
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        future_map={pool.submit(run_one, fold, shard):(fold, shard) for fold, shard in tasks}
        for future in as_completed(future_map):
            fold, shard = future_map[future]
            try:
                result=future.result()
            except Exception as exc:
                result={'fold_id': fold['fold_id'], 'shard_id': shard['shard_id'], 'returncode': -999, 'error': repr(exc)}
                emit('shard_exception', **result)
            results.append(result)
            if int(result.get('returncode') or 0) != 0:
                failures.append(result)
            emit('full_walkforward_consolidated_progress', completed=len(results), total=len(tasks), failures=len(failures))
    summary={'artifact_kind':'full_all_strategy_walkforward_consolidated_run_summary','generated_at_utc':now(),'output_root':str(BASE),'folds':FOLDS,'shards':[{**s,'manifest':str(s['manifest']),'manifest_count':manifest_count(s['manifest'])} for s in SHARDS],'max_parallel':MAX_PARALLEL,'results':results,'failure_count':len(failures)}
    summary_path=BASE/'full_walkforward_consolidated_run_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), 'utf-8')
    emit('full_walkforward_consolidated_complete', output_root=str(BASE), summary_path=str(summary_path), failure_count=len(failures), completed=len(results), total=len(tasks))
    return 1 if failures else 0

if __name__ == '__main__':
    raise SystemExit(main())
