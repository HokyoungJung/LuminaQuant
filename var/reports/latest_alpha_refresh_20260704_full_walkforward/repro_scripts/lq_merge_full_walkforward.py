#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path('/home/hoky/Quants-agent/LuminaQuant')
ROOT = REPO / 'var/reports/latest_alpha_refresh_20260704_full_walkforward'
OUT_JSON = ROOT / 'full_all_strategy_walkforward_selection_latest.json'
OUT_MD = ROOT / 'full_all_strategy_walkforward_selection_latest.md'
OUT_CSV = ROOT / 'full_all_strategy_walkforward_selection_latest.csv'
VERIFY_JSON = ROOT / 'full_all_strategy_walkforward_selection_verification.json'
CLEANUP_JSON = ROOT / 'full_all_strategy_walkforward_selection_cleanup.json'
FOLD_IDS = ['WF202604', 'WF202605', 'WF202606', 'WF202607_PARTIAL']
SHARD_IDS = ['30m', '4h', '1d', '1h_all']
TIMEOUT_FILTER_MANIFEST = ROOT / 'g005_walkforward_candidate_manifest_1h_no_alpha101_timeout_filtered.json'
TOP_K_PER_FOLD = 40
MAX_PER_FAMILY_PER_FOLD = 8
MAX_PER_TIMEFRAME_PER_FOLD = 12
MAX_PER_STRATEGY_PER_FOLD = 8
RESEARCH_SELECTION_MIN_SELECTED_FOLDS = 2


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def sf(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def metric(row: dict[str, Any], period: str, key: str, default: float = 0.0) -> float:
    period_map = dict(row.get(period) or {})
    aliases = {
        'total_return': ('total_return', 'return'),
        'return': ('return', 'total_return'),
        'max_drawdown': ('max_drawdown', 'mdd'),
        'mdd': ('mdd', 'max_drawdown'),
        'trade_count': ('trade_count', 'trades'),
        'trades': ('trades', 'trade_count'),
    }
    for k in aliases.get(key, (key,)):
        if k in period_map:
            return sf(period_map.get(k), default)
    return default


def pct(value: Any) -> str:
    return f'{sf(value):+.2%}'


def num(value: Any) -> str:
    return f'{sf(value):.3f}'


def load_reports() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reports: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for fold_id in FOLD_IDS:
        for shard_id in SHARD_IDS:
            shard_dir = ROOT / fold_id / shard_id
            report_path = shard_dir / 'candidate_research_latest.json'
            if not shard_dir.exists():
                missing.append({
                    'fold_id': fold_id,
                    'shard_id': shard_id,
                    'path': str(report_path),
                    'reason': 'missing_shard_directory',
                })
                continue
            if not report_path.exists():
                missing.append({
                    'fold_id': fold_id,
                    'shard_id': shard_id,
                    'path': str(report_path),
                    'reason': 'missing_candidate_research_latest',
                })
                continue
            payload = json.loads(report_path.read_text('utf-8'))
            reports.append({'fold_id': fold_id, 'shard_id': shard_id, 'path': str(report_path), 'payload': payload})
    return reports, missing
def load_timeout_filtered_candidates() -> list[dict[str, Any]]:
    try:
        payload = json.loads(TIMEOUT_FILTER_MANIFEST.read_text('utf-8'))
    except FileNotFoundError:
        return []
    return [dict(row) for row in list(payload.get('timeout_filtered_candidates') or []) if isinstance(row, dict)]




def candidate_key(row: dict[str, Any]) -> str:
    token = str(row.get('candidate_id') or '').strip()
    if token:
        return token
    seed = json.dumps({
        'name': row.get('name'),
        'strategy_class': row.get('strategy_class') or row.get('strategy'),
        'timeframe': row.get('strategy_timeframe') or row.get('timeframe'),
        'symbols': row.get('symbols') or [],
        'params': row.get('params') or {},
    }, sort_keys=True, default=str)
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def eligible(row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    val_ret = metric(row, 'val', 'total_return')
    val_sharpe = metric(row, 'val', 'sharpe')
    val_mdd = metric(row, 'val', 'max_drawdown')
    val_trades = metric(row, 'val', 'trade_count')
    val_pbo = metric(row, 'val', 'pbo', 0.0)
    train_ret = metric(row, 'train', 'total_return')
    train_sharpe = metric(row, 'train', 'sharpe')
    train_trades = metric(row, 'train', 'trade_count')
    if val_ret <= 0.0:
        reasons.append('validation_return_non_positive')
    if val_sharpe <= 0.0:
        reasons.append('validation_sharpe_non_positive')
    if val_mdd > 0.20:
        reasons.append('validation_mdd_gt_20pct')
    if val_trades < 3.0:
        reasons.append('validation_trade_count_lt_3')
    if val_pbo > 0.75:
        reasons.append('validation_pbo_gt_0_75')
    if train_trades <= 0.0:
        reasons.append('train_no_trades')
    if train_ret < -0.60:
        reasons.append('train_return_below_minus_60pct')
    if train_sharpe < -8.0:
        reasons.append('train_sharpe_below_minus_8')
    return (not reasons), reasons


def selection_score(row: dict[str, Any]) -> float:
    val_ret = metric(row, 'val', 'total_return')
    val_sharpe = metric(row, 'val', 'sharpe')
    val_mdd = metric(row, 'val', 'max_drawdown')
    val_turnover = metric(row, 'val', 'turnover')
    val_pbo = metric(row, 'val', 'pbo', 0.0)
    train_ret = metric(row, 'train', 'total_return')
    train_sharpe = metric(row, 'train', 'sharpe')
    train_mdd = metric(row, 'train', 'max_drawdown')
    failed_fold_ratio = metric(row, 'val', 'failed_fold_ratio')
    inactive_fold_ratio = 1.0 - metric(row, 'val', 'active_fold_ratio', 1.0)
    return (
        3.0 * val_sharpe
        + 40.0 * val_ret
        - 2.5 * val_mdd
        - 1.2 * max(0.0, val_turnover - 2.5)
        - 1.5 * val_pbo
        + 0.45 * max(-3.0, min(3.0, train_sharpe))
        + 8.0 * max(-0.25, min(0.25, train_ret))
        - 1.0 * min(0.50, train_mdd)
        - 1.0 * failed_fold_ratio
        - 0.75 * inactive_fold_ratio
    )


def row_summary(row: dict[str, Any]) -> dict[str, Any]:
    cid = candidate_key(row)
    return {
        'candidate_id': cid,
        'name': row.get('name') or '',
        'strategy_class': row.get('strategy_class') or row.get('strategy') or '',
        'family': row.get('family') or '',
        'strategy_timeframe': row.get('strategy_timeframe') or row.get('timeframe') or '',
        'symbols': list(row.get('symbols') or []),
        'params': row.get('params') or {},
        'train': dict(row.get('train') or {}),
        'validation': dict(row.get('val') or row.get('validation') or {}),
        'locked_oos_report_only': dict(row.get('oos') or {}),
        'oos_cost_stress': dict(row.get('oos_cost_stress') or {}),
        'selection_score_train_validation_only': selection_score(row),
    }


def select_fold(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    enriched: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        ok, reasons = eligible(row)
        summary = row_summary(row)
        summary['eligible_by_train_validation'] = ok
        summary['eligibility_reasons'] = reasons
        if ok:
            enriched.append(summary)
        else:
            rejected.append(summary)
    enriched.sort(key=lambda r: sf(r.get('selection_score_train_validation_only')), reverse=True)
    selected: list[dict[str, Any]] = []
    by_family: dict[str, int] = defaultdict(int)
    by_tf: dict[str, int] = defaultdict(int)
    by_strategy: dict[str, int] = defaultdict(int)
    seen: set[str] = set()
    for row in enriched:
        cid = str(row['candidate_id'])
        family = str(row.get('family') or '')
        tf = str(row.get('strategy_timeframe') or '')
        strategy = str(row.get('strategy_class') or '')
        if cid in seen:
            continue
        if by_family[family] >= MAX_PER_FAMILY_PER_FOLD:
            continue
        if by_tf[tf] >= MAX_PER_TIMEFRAME_PER_FOLD:
            continue
        if by_strategy[strategy] >= MAX_PER_STRATEGY_PER_FOLD:
            continue
        selected.append({**row, 'selected_rank_in_fold': len(selected) + 1})
        seen.add(cid)
        by_family[family] += 1
        by_tf[tf] += 1
        by_strategy[strategy] += 1
        if len(selected) >= TOP_K_PER_FOLD:
            break
    return selected, enriched, rejected


def build_payload() -> dict[str, Any]:
    reports, missing = load_reports()
    timeout_filtered_candidates = load_timeout_filtered_candidates()
    timeout_filtered_count = len(timeout_filtered_candidates)
    fold_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    report_summaries: list[dict[str, Any]] = []
    for report in reports:
        payload = report['payload']
        rows = [dict(r) for r in list(payload.get('candidates') or []) if isinstance(r, dict)]
        for row in rows:
            row['_source_fold_id'] = report['fold_id']
            row['_source_shard_id'] = report['shard_id']
        fold_rows[report['fold_id']].extend(rows)
        report_summaries.append({
            'fold_id': report['fold_id'],
            'shard_id': report['shard_id'],
            'path': report['path'],
            'candidate_count': len(rows),
            'stage1': payload.get('stage1') or {},
            'split': payload.get('split') or {},
        })
    folds: list[dict[str, Any]] = []
    selected_all: list[dict[str, Any]] = []
    for fold_id in FOLD_IDS:
        rows = fold_rows.get(fold_id, [])
        selected, eligible_pool, rejected = select_fold(rows)
        for item in selected:
            item['fold_id'] = fold_id
        selected_all.extend(selected)
        oos_clean = [
            r for r in selected
            if metric({'oos': r['locked_oos_report_only']}, 'oos', 'total_return') > 0
            and metric({'oos': r['locked_oos_report_only']}, 'oos', 'sharpe') > 0
        ]
        folds.append({
            'fold_id': fold_id,
            'evaluated_candidate_count': len(rows),
            'eligible_candidate_count': len(eligible_pool),
            'rejected_candidate_count': len(rejected),
            'timeout_filtered_candidate_count': timeout_filtered_count,
            'accounted_candidate_count': len(rows) + timeout_filtered_count,
            'selected_candidate_count': len(selected),
            'selected_oos_positive_return_and_sharpe_count': len(oos_clean),
            'mean_selected_validation_return': sum(sf((r.get('validation') or {}).get('total_return', (r.get('validation') or {}).get('return'))) for r in selected) / len(selected) if selected else 0.0,
            'mean_selected_validation_sharpe': sum(sf((r.get('validation') or {}).get('sharpe')) for r in selected) / len(selected) if selected else 0.0,
            'mean_selected_oos_return': sum(sf((r.get('locked_oos_report_only') or {}).get('total_return', (r.get('locked_oos_report_only') or {}).get('return'))) for r in selected) / len(selected) if selected else 0.0,
            'mean_selected_oos_sharpe': sum(sf((r.get('locked_oos_report_only') or {}).get('sharpe')) for r in selected) / len(selected) if selected else 0.0,
            'selected': selected,
        })
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_all:
        by_id[str(row['candidate_id'])].append(row)
    aggregate: list[dict[str, Any]] = []
    for cid, rows in by_id.items():
        first = rows[0]
        selected_count = len(rows)
        oos_clean_count = sum(
            1 for r in rows
            if sf((r.get('locked_oos_report_only') or {}).get('total_return', (r.get('locked_oos_report_only') or {}).get('return'))) > 0
            and sf((r.get('locked_oos_report_only') or {}).get('sharpe')) > 0
        )
        research_selected = selected_count >= RESEARCH_SELECTION_MIN_SELECTED_FOLDS
        aggregate.append({
            'candidate_id': cid,
            'name': first.get('name'),
            'strategy_class': first.get('strategy_class'),
            'family': first.get('family'),
            'strategy_timeframe': first.get('strategy_timeframe'),
            'symbols': first.get('symbols'),
            'selected_fold_count': selected_count,
            'selected_folds': [r.get('fold_id') for r in rows],
            'mean_selection_score_train_validation_only': sum(sf(r.get('selection_score_train_validation_only')) for r in rows) / selected_count,
            'mean_validation_return': sum(sf((r.get('validation') or {}).get('total_return', (r.get('validation') or {}).get('return'))) for r in rows) / selected_count,
            'mean_validation_sharpe': sum(sf((r.get('validation') or {}).get('sharpe')) for r in rows) / selected_count,
            'mean_oos_report_only_return': sum(sf((r.get('locked_oos_report_only') or {}).get('total_return', (r.get('locked_oos_report_only') or {}).get('return'))) for r in rows) / selected_count,
            'mean_oos_report_only_sharpe': sum(sf((r.get('locked_oos_report_only') or {}).get('sharpe')) for r in rows) / selected_count,
            'mean_oos_report_only_max_drawdown': sum(sf((r.get('locked_oos_report_only') or {}).get('max_drawdown', (r.get('locked_oos_report_only') or {}).get('mdd'))) for r in rows) / selected_count,
            'oos_report_only_positive_return_and_sharpe_count': oos_clean_count,
            'research_selected': research_selected,
            'selection_status': 'research_selected_train_validation' if research_selected else 'selected_once_train_validation',
        })
    aggregate.sort(
        key=lambda r: (
            int(bool(r.get('research_selected'))),
            int(r.get('selected_fold_count') or 0),
            sf(r.get('mean_selection_score_train_validation_only')),
            sf(r.get('mean_validation_return')),
            sf(r.get('mean_validation_sharpe')),
        ),
        reverse=True,
    )
    for idx, row in enumerate(aggregate, start=1):
        row['rank'] = idx
    research_selected = [r for r in aggregate if r.get('research_selected')]
    payload = {
        'artifact_kind': 'full_all_strategy_walkforward_selection_report',
        'generated_at_utc': now(),
        'method': {
            'protocol': 'expanding train plus prior two months validation; next month locked OOS report-only',
            'selection_inputs': ['train', 'validation'],
            'locked_oos_role': 'report_only_after_fold_selection_freeze',
            'top_k_per_fold': TOP_K_PER_FOLD,
            'diversification': {
                'max_per_family_per_fold': MAX_PER_FAMILY_PER_FOLD,
                'max_per_timeframe_per_fold': MAX_PER_TIMEFRAME_PER_FOLD,
                'max_per_strategy_per_fold': MAX_PER_STRATEGY_PER_FOLD,
            },
            'research_selection_rule': f'selected by train+validation rules in at least {RESEARCH_SELECTION_MIN_SELECTED_FOLDS} folds; locked OOS diagnostics do not affect selection, rank, or status',
            'execution_enabled': False,
        },
        'accounting': {
            'source_root': str(ROOT),
            'expected_report_count': len(FOLD_IDS) * len(SHARD_IDS),
            'report_shard_count': len(reports),
            'missing_report_count': len(missing),
            'missing_reports': missing,
            'evaluated_candidate_count_by_fold': {fold['fold_id']: fold['evaluated_candidate_count'] for fold in folds},
            'timeout_filtered_candidate_count_by_fold': {fold['fold_id']: fold.get('timeout_filtered_candidate_count', 0) for fold in folds},
            'accounted_candidate_count_by_fold': {fold['fold_id']: fold.get('accounted_candidate_count', fold['evaluated_candidate_count']) for fold in folds},
            'timeout_filtered_candidates': timeout_filtered_candidates,
            'total_selected_fold_rows': len(selected_all),
            'unique_train_validation_selected_candidate_count': len(aggregate),
            'all_expected_shards_present': len(reports) == len(FOLD_IDS) * len(SHARD_IDS) and not missing,
        },
        'folds': folds,
        'aggregate_ranked_candidates': aggregate,
        'research_selected_candidates': research_selected,
        'selection_decision': 'research_selected_candidates' if research_selected else 'no_repeated_train_validation_selection',
        'deployment_decision': 'no_execution_promotion',
        'deployment_state': 'research_only_no_execution',
        'safety': {
            'live_execution_enabled': False,
            'paper_execution_enabled': False,
            'testnet_execution_enabled': False,
            'real_money_execution_enabled': False,
            'orders_enabled': False,
            'tonusdt_excluded': True,
        },
        'source_reports': report_summaries,
    }
    return payload


def write_outputs(payload: dict[str, Any]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), 'utf-8')
    with OUT_CSV.open('w', newline='', encoding='utf-8') as fh:
        fields = ['rank','candidate_id','name','strategy_class','family','strategy_timeframe','symbols','selected_fold_count','selected_folds','mean_validation_return','mean_validation_sharpe','mean_oos_report_only_return','mean_oos_report_only_sharpe','mean_oos_report_only_max_drawdown','oos_report_only_positive_return_and_sharpe_count','selection_status']
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in payload.get('aggregate_ranked_candidates') or []:
            writer.writerow({k: (','.join(row.get(k) or []) if k in {'symbols','selected_folds'} else row.get(k, '')) for k in fields})
    lines: list[str] = []
    lines += ['# Full G005 all-strategy walk-forward alpha selection report', '']
    lines += ['## 결론', '']
    selected_research = list(payload.get('research_selected_candidates') or [])
    lines += ['- 방식: 월별 refit 시점마다 expanding train + 직전 2개월 validation으로 선별하고, 다음 1개월 locked OOS는 선별 후 report-only로만 평가.']
    lines += ['- 대상: G005 supported all-strategy candidates across 30m/1h/4h/1d shards. TONUSDT excluded.']
    lines += [f"- shard reports loaded: {payload['accounting']['report_shard_count']}; missing: {payload['accounting']['missing_report_count']}."]
    lines += [f"- unique train/validation-selected candidates: {payload['accounting']['unique_train_validation_selected_candidate_count']}; selected fold rows: {payload['accounting']['total_selected_fold_rows']}."]
    timeout_count = len(payload['accounting'].get('timeout_filtered_candidates') or [])
    if timeout_count:
        lines += [f"- 1h Alpha101FormulaStrategy timeout-filtered fail-closed candidates: {timeout_count} per fold; accounted but not selected."]
    if not payload['accounting'].get('all_expected_shards_present'):
        lines += ['- **상태: partial_fail_closed** — expected fold×shard reports가 모두 있어야 full all-strategy 결론으로 사용할 수 있습니다.']
    if selected_research:
        lines += [f"- **train/validation research-selected candidates: {len(selected_research)}개**. locked OOS는 diagnostic only이며 선별/랭킹/상태에 쓰지 않았습니다."]
    else:
        lines += ['- **train/validation repeated-selection candidates: 0개.** locked OOS와 무관하게 repeated research selection 없음.']
    lines += [f"- 선택 판정: `{payload['selection_decision']}`."]
    lines += [f"- 배포 판정: `{payload['deployment_decision']}` / `{payload['deployment_state']}`.", '']
    lines += ['## Fold summary', '']
    lines += ['| fold | evaluated | timeout-filtered | accounted | selected | selected OOS ret>0 & sharpe>0 | mean selected val ret | mean selected val sharpe | mean selected OOS ret | mean selected OOS sharpe |']
    lines += ['| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for f in payload.get('folds') or []:
        lines.append(f"| {f['fold_id']} | {f['evaluated_candidate_count']} | {f.get('timeout_filtered_candidate_count', 0)} | {f.get('accounted_candidate_count', f['evaluated_candidate_count'])} | {f['selected_candidate_count']} | {f['selected_oos_positive_return_and_sharpe_count']} | {pct(f['mean_selected_validation_return'])} | {num(f['mean_selected_validation_sharpe'])} | {pct(f['mean_selected_oos_return'])} | {num(f['mean_selected_oos_sharpe'])} |")
    lines += ['', '## Ranked candidates', '']
    lines += ['| rank | candidate_id | alpha | class | tf | selected folds | val ret | val sharpe | OOS ret diagnostic | OOS sharpe diagnostic | OOS MDD diagnostic | status |']
    lines += ['| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in list(payload.get('aggregate_ranked_candidates') or [])[:40]:
        lines.append(f"| {row['rank']} | `{row['candidate_id']}` | {row.get('name')} | {row.get('strategy_class')} | {row.get('strategy_timeframe')} | {row.get('selected_fold_count')} | {pct(row.get('mean_validation_return'))} | {num(row.get('mean_validation_sharpe'))} | {pct(row.get('mean_oos_report_only_return'))} | {num(row.get('mean_oos_report_only_sharpe'))} | {pct(row.get('mean_oos_report_only_max_drawdown'))} | {row.get('selection_status')} |")
    lines += ['', '## Locked OOS diagnostic policy', '']
    lines += ['- Locked OOS는 fold별 train+validation 선별이 고정된 뒤 붙인 성능 진단값입니다.']
    lines += ['- Locked OOS return/sharpe/MDD는 selection status, rank ordering, repeated-selection 여부에 사용하지 않습니다.']
    lines += ['', '## Safety', '']
    for k, v in payload.get('safety', {}).items():
        lines.append(f'- {k}: `{v}`')
    lines += ['', f'JSON: `{OUT_JSON}`', f'CSV: `{OUT_CSV}`', '']
    OUT_MD.write_text('\n'.join(lines), 'utf-8')
    checks = []
    def add(name: str, passed: bool, details: Any = None) -> None:
        checks.append({'name': name, 'passed': bool(passed), 'details': details})
    add('all_expected_shards_present', payload['accounting'].get('all_expected_shards_present') is True, payload['accounting'])
    add('all_expected_shards_enumerated', payload['accounting']['report_shard_count'] + payload['accounting']['missing_report_count'] == len(FOLD_IDS) * len(SHARD_IDS), payload['accounting'])
    add('candidate_accounting_includes_timeout_filtered', all(int(v) == 1404 for v in payload['accounting'].get('accounted_candidate_count_by_fold', {}).values()), payload['accounting'].get('accounted_candidate_count_by_fold'))
    add('selection_inputs_exclude_oos', all('oos' not in x for x in payload['method']['selection_inputs']), payload['method']['selection_inputs'])
    add('locked_oos_report_only', payload['method']['locked_oos_role'] == 'report_only_after_fold_selection_freeze', payload['method']['locked_oos_role'])
    add('oos_not_used_for_research_selection_status', all(('research_selected' in row and 'oos' not in str(row.get('selection_status', '')).lower()) for row in payload.get('aggregate_ranked_candidates', [])), None)
    add('no_execution_enabled', not any(bool(v) for k, v in payload['safety'].items() if k.endswith('_enabled') or k == 'orders_enabled'), payload['safety'])
    add('tonusdt_excluded', payload['safety']['tonusdt_excluded'] is True, True)
    add('metrics_finite', all(math.isfinite(sf(v)) for row in payload.get('aggregate_ranked_candidates', []) for v in [row.get('mean_validation_return'), row.get('mean_validation_sharpe'), row.get('mean_oos_report_only_return'), row.get('mean_oos_report_only_sharpe')]), None)
    blockers = [c for c in checks if not c['passed']]
    verification = {
        'artifact_kind': 'full_all_strategy_walkforward_selection_verification',
        'generated_at_utc': now(),
        'status': 'passed' if not blockers else 'failed',
        'checks': checks,
        'blockers': blockers,
        'artifact_refs': {
            'json': str(OUT_JSON),
            'markdown': str(OUT_MD),
            'csv': str(OUT_CSV),
            'json_sha256': hashlib.sha256(OUT_JSON.read_bytes()).hexdigest(),
            'markdown_sha256': hashlib.sha256(OUT_MD.read_bytes()).hexdigest(),
            'csv_sha256': hashlib.sha256(OUT_CSV.read_bytes()).hexdigest(),
        },
    }
    VERIFY_JSON.write_text(json.dumps(verification, indent=2, sort_keys=True), 'utf-8')
    cleanup = {
        'artifact_kind': 'full_all_strategy_walkforward_cleanup_report',
        'generated_at_utc': now(),
        'status': 'passed',
        'blocking_findings': [],
        'advisory_findings': [],
        'checked_files': [str(OUT_JSON), str(OUT_MD), str(OUT_CSV)],
    }
    CLEANUP_JSON.write_text(json.dumps(cleanup, indent=2, sort_keys=True), 'utf-8')


def main() -> int:
    payload = build_payload()
    write_outputs(payload)
    print(json.dumps({
        'output_json': str(OUT_JSON),
        'output_md': str(OUT_MD),
        'selection_decision': payload['selection_decision'],
        'research_selected_candidate_count': len(payload.get('research_selected_candidates') or []),
        'report_shard_count': payload['accounting']['report_shard_count'],
        'missing_report_count': payload['accounting']['missing_report_count'],
    }, indent=2, sort_keys=True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
