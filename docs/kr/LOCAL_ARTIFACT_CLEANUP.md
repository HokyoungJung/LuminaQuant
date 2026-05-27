# 로컬 산출물 정리

leader workspace에 cache/build/runtime log가 쌓였지만 연구 증거는 보존해야 할 때 사용합니다.

## 연구노트와 증거 위치

- Alpha Zoo 기본 연구노트: `docs/research_note_profit_moonshot_alpha_zoo_real_data_20260512.md`
- State-distilled 연구노트: `docs/research_note_profit_moonshot_state_distilled_20260511.md`
- 전체 연구 이력/source ledger: `docs/profit_moonshot_research_history_20260510.md`
- 세션 handoff: `docs/session_handoff_*.md`
- live/paper 운영 런북: `docs/live-readiness/`
- 세션 checkpoint memory: `.omx/notepad.md`
- 전체 생성 연구 결과/artifact: `var/reports/`
- 시장 데이터: `data/`

## 안전 정리 명령

먼저 dry-run으로 확인합니다:

```bash
uv run python scripts/dev/cleanup_local_artifacts.py --json
```

기본 안전 정리를 적용합니다:

```bash
uv run python scripts/dev/cleanup_local_artifacts.py --apply
```

기본 정리는 로컬 생성물만 삭제합니다: Python/Ruff/test cache, 루트 runtime `logs/`, dashboard `.next`/`node_modules`, 무시되는 dashboard incremental 파일, Python build/egg 출력, Rust `target/`, root `reports/quality`, OMX runtime `cache/tmp/logs`.

## 기본 보존 대상

정리 스크립트는 아래를 기본 보존합니다:

- `data/`
- `var/reports/`
- `var/logs/` (일부 untracked log는 research/run evidence일 수 있음)
- `docs/` 연구노트와 handoff
- `.omx/notepad.md`, `.omx/context/`, `.omx/plans/`, `.omx/project-memory.json`
- `.env`
- `.venv` (재설치 없이 verification을 돌리기 위해 보존)
- `best_optimized_parameters/`
- `.gitnexus/` (source 변경 후 `npx gitnexus analyze`로 refresh)

`.gitnexus/parse-cache` 또는 `.venv`까지 지우는 optional flag가 있지만 기본값은 꺼져 있습니다.
