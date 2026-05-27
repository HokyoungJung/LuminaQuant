# Local Artifact Cleanup

Use this when the leader workspace accumulates local caches, build outputs, and runtime logs but research evidence must stay intact.

## Research-note and evidence locations

- Primary Alpha Zoo research note: `docs/research_note_profit_moonshot_alpha_zoo_real_data_20260512.md`
- State-distilled research note: `docs/research_note_profit_moonshot_state_distilled_20260511.md`
- Global research history/source ledger: `docs/profit_moonshot_research_history_20260510.md`
- Session handoffs: `docs/session_handoff_*.md`
- Operator live/paper runbooks: `docs/live-readiness/`
- Session checkpoint memory: `.omx/notepad.md`
- Full generated research artifacts/results: `var/reports/`
- Market data: `data/`

## Safe cleanup command

Dry-run first:

```bash
uv run python scripts/dev/cleanup_local_artifacts.py --json
```

Apply the default safe cleanup:

```bash
uv run python scripts/dev/cleanup_local_artifacts.py --apply
```

Default cleanup removes local generated artifacts only: Python/Ruff/test caches, root runtime `logs/`, dashboard `.next`/`node_modules`, ignored dashboard incremental files, Python build/egg outputs, root `reports/quality`, and OMX runtime `cache/tmp/logs`.

Native Rust `target/` directories are preserved by default so Python wrappers can keep loading proven release shared libraries such as `native/rust_rawfirst/target/release/liblumina_rawfirst.so`. Use `--include-native-targets` only when you intentionally want to force a rebuild.

## Preserved by default

The cleanup script deliberately preserves:

- `data/`
- `var/reports/`
- `var/logs/` (some untracked logs are research/run evidence)
- `docs/` research notes and handoffs
- `.omx/notepad.md`, `.omx/context/`, `.omx/plans/`, `.omx/project-memory.json`
- `.env`
- `.venv` (kept so verification can run without reinstalling)
- `best_optimized_parameters/`
- `.gitnexus/` (refresh with `npx gitnexus analyze` after source changes)
- native Rust release build outputs under `native/rust_*/target/` (optional accelerators loaded by Python wrappers)

Optional flags exist for explicit local-only cleanup of `.gitnexus/parse-cache`, `.venv`, or native Rust `target/` directories, but those are off by default.
