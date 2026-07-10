# Dashboard Web UI/UX Audit — 2026-07-10

- **Scope**: `apps/dashboard_web` (Next.js 15, 11 routes) + `src/lumina_quant/dashboard/` payload services
- **Method**: 4 parallel review lanes (IA/nav, strategy-performance visualization, market comprehension, UX states/a11y) -> 40 raw findings -> per-finding adversarial verification (independent refutation agents). **34 confirmed / 0 refuted / 6 P3 unverified nits.** Visual evidence: 25 Playwright captures on live dev server (`:3100`) — real empty state (missing_dsn) + contract-conformant mock payloads injected at fetch layer.
- **Verdict**: 골격(계약 타입, localhost+토큰 미들웨어, 빈 상태 일부, WCAG 대비)은 양호. 그러나 화면은 아직 마이그레이션 검증 뷰에 머물러 있어 (1) 후보 전략 성적의 시각적 판단, (2) 대상 마켓 판단이 사실상 불가하고, 표시 수치를 왜곡하는 데이터-신뢰 결함 3건(꼬리구간 지표 산출, Price Change % 100x, 252 고정 연율화)이 있다.

## Lane verdicts

### ia-nav

The information architecture is inverted for its audience: the home page spends 5 of 6 sections on migration metadata (memory budget, route contracts, cutover evidence) and buries the only live trading section, while the sidebar functions as a migration tracker — 11 identical 'available' pills, migration-prose summaries, and an aria-current bug that marks every link as the current page with no actual active-route indication. There is no global context (run id, as-of time, live-vs-backtest), which is the most important missing capability given fetch-once data and a currently unreachable DB. On the positive side, there are no dead or 'coming soon' routes: all 11 nav items resolve to real pages, and the flat route inventory itself is reasonable — it mainly needs regrouping (operations vs research), operator-language renames (notably 'Exact-window'), and de-duplication of the capability inventory.</lane_verdict>
</invoke>

### perf-viz

Today the user can only very weakly judge candidate-strategy performance visually: the entire "charting" layer is four 420x120 min-max-normalized sparkline paths with no axes, scales, tooltips, legends, or overlays, and everything else — optimization candidates, exact-window candidates, factor IC "heatmap", trades — is unformatted HTML tables of raw floats for the single latest run only. Worse, the headline Sharpe/CAGR/MDD tiles are computed from a silently truncated 120-240-point equity tail with a hardcoded 252-period annualization, so the numbers shown can be materially wrong, not merely hard to read. Top 3 gaps: (1) real charts — axes/scale on equity+drawdown, benchmark rebased and overlaid, trade markers plotted; (2) a run/strategy selector plus a visual candidate-comparison view (the payloads already carry 8-12 candidates with Sharpe/CAGR/MDD/robustness, currently table-dumped with best_candidate as raw JSON); (3) trustworthy, scannable headline metrics — full-run windows, cadence-aware annualization, percent formatting with directional color.

### market-info

No — the user cannot easily judge the target market from this dashboard. The market page identifies symbol/timeframe/exchange but renders price action only as a 24-row OHLCV table (no chart, despite loading 240 bars and having an existing sparkline helper), its "Indicator parity" section contains no actual indicators (no funding rate, volatility, or regime — the exact inputs a perp go/no-go needs), and the headline "Price Change %" metric displays a raw fraction that misreads 100x low. The traded universe is invisible: a multi-symbol system is collapsed to whichever symbol filled last, with no symbol switching or per-symbol exposure anywhere, and the page is an island with no connection to strategy/performance context. Top 3 gaps: (1) no price/trend visualization, (2) no market-state indicators (funding/vol/regime) with contextualized values, (3) no universe/per-symbol exposure view — plus the P0 percent-formatting bug that actively misleads.

### ux-states

The visual foundation is decent — semantic headings, labeled SVGs, WCAG-passing contrast throughout, and overflow-wrapped tables — but the app behaves as a write-once snapshot viewer with no freshness signal, no refresh path, and empty states that are DSN-blind on 8 of 10 surfaces. The two P0s are on the only truly interactive surface: workflow Stop/Kill fails silently (unhandled rejection, no feedback), and a missing DSN renders as "No managed workflow jobs recorded yet," a false factual claim. Fixing the shared seams (status-aware empty-state helper, as_of rendering plus refetch in useBridgeFetch, error handling in triggerAction) would resolve most findings with small, localized changes.


## Confirmed findings (34, adversarially verified)


### P0

#### 1. [P0/ia-nav] Home page is dominated by migration/engineering metadata; live trading state is buried in section 5 of 6

- **Where**: `apps/dashboard_web/app/page.tsx:8`
- **Evidence**: Home renders, in order: hero 'Overview parity slice' (line 8), 'Memory budget' / '8GB guardrail' with host RAM and RSS targets (line 17), 'Legacy-to-web route contract' 11-row table of Python source modules vs Next routes (line 56), 'Cutover gate evidence' with 7 evidence bullets and launcher status (line 90), then finally <OverviewRuntime /> (line 118) which is the only section containing trading data (equity/drawdown sparklines, performance metrics, recent runs/jobs), followed by 'Foundation scope / Available now' (line 120) which re-lists the same 11 capabilities. Quantified: 5 of 6 sections are migration meta; the meta sections (4-metric grid + 3 guidance bullets + 11-row table + cutover section) occupy roughly 2-3 viewport-heights before any P&L, run, or position information appears. Section titles include 'Overview placeholder' (python-bridge.ts:107) rendered verbatim into the UI cards.
- **Why it matters**: The operator's first questions are 'what is my equity/drawdown, which run is active, are my candidates healthy'. Instead the first screens answer 'what is the dashboard app's memory budget and route contract' — internal engineering state that never changes at decision time. The one live section is invisible without scrolling, so the home page fails as an overview.
- **Recommendation**: Invert the hierarchy: OverviewRuntime (equity, drawdown, performance metrics, recent runs/jobs, risk flags) first and full-width. Move memory budget, route contract, and cutover-gate evidence to a /system or /about-migration page (or delete them now that cutover is declared complete by dashboardCutoverGate.remainingGate).
- **Verifier correction**: One partial mitigation the finding did not mention: DashboardShell (components/dashboard-shell.tsx lines 17-29) renders a persistent sidebar nav from navigationItems, so the operator can jump directly to dedicated data routes (performance-price, risk-health, execution-analytics, etc.) without scrolling the home page. This softens the practical impact slightly (arguably P0 -> P1 for a single operator who learns the nav) but does not refute the core claim that the home/overview route itself buries live trading state under 4 static migration-metadata sections. Also, 'Overview placeholder' appears as a capability title inside the table and feature cards, not as a section heading — the finding's phrasing 'section titles' is slightly imprecise.

#### 2. [P0/market-info] "Price Change %" displays a raw fraction under a percent label — off by 100x

- **Where**: `src/lumina_quant/dashboard/cutover_surfaces_service.py:680`
- **Evidence**: Builder: `overview_metric("Price Change %", None if price_change_pct is None else round(float(price_change_pct), 6), ...)` where `price_change_pct = (latest - first)/first` (cutover_surfaces_service.py:662-684). Renderer passes it through `formatMetricValue` which is just `String(value)` (apps/dashboard_web/lib/format.ts:37-42; apps/dashboard_web/components/market-data-runtime.tsx:26-31). A +1.2% move renders as "0.012" under the label "Price Change %".
- **Why it matters**: The user reads "Price Change %: 0.012" as 0.012% when the actual move is 1.2%. For a crypto-perp trader gauging recent market movement before a go/no-go call, a 100x misstatement of the headline market-move metric actively misleads.
- **Recommendation**: Multiply by 100 and format with a % suffix and sign (e.g. "+1.20%"), or introduce a typed metric (kind: 'percent') so formatMetricValue can render fractions correctly across all surfaces.
- **Verifier correction**: The mechanics and evidence are accurate, but two details need correction: (1) severity — P0 is overstated for a localhost single-operator parity-preview panel (the payload labels itself "price-only parity preview"); the value is a correct fraction with a wrong unit label, not corrupted data, so P1 is the right rating. (2) The metric is the change from the first to the last bar of the entire loaded market window, not a short-horizon "recent market movement" gauge, which further weakens the go/no-go-decision framing. Fix is trivial: multiply by 100 in the builder or format by key in the renderer.

#### 3. [P0/perf-viz] All four SVG charts are unlabeled min-max-normalized sparklines — no axes, ticks, y-scale, tooltips, legend, or time labels, so magnitude and timing of performance are unreadable

- **Where**: `apps/dashboard_web/lib/format.ts:23`
- **Evidence**: buildSparklinePath: `const min = Math.min(...values); const max = Math.max(...values); const range = max - min || 1;` then maps to a bare 420x120 path. Every consumer renders it as `<svg viewBox="0 0 420 120"><path d={...} stroke="currentColor"/></svg>` with nothing else (overview-runtime.tsx:57-59, 66-68; performance-price-runtime.tsx:46-48, 57-59, 68-70, 79-81). No axis, gridline, tick, hover, or numeric annotation exists anywhere in the app.
- **Why it matters**: Min-max normalization stretches any series to full card height: an equity curve that gained +0.1% renders pixel-identical to one that gained +50%; a -3% drawdown looks identical to -40%. The user literally cannot answer 'how good was this strategy' from the visuals — the charts convey shape only, and the shape's scale actively misleads. This is the core of question #1 and it fails.
- **Recommendation**: Promote the equity/drawdown charts to real charts: y-axis with 3-5 tick labels (equity in $/%, drawdown in %), x-axis with start/end timestamps, a zero/peak baseline on the drawdown chart, and hover tooltip showing (timestamp, value). Keep sparkline styling for the funding trace if desired, but annotate min/max/last values on every card at minimum.
- **Verifier correction**: Six sparkline charts, not four (2 in overview-runtime.tsx + 4 in performance-price-runtime.tsx), all min-max-normalized and unlabeled exactly as described. Mitigating context the finder omitted: both pages display exact numeric performance data alongside the charts (metric tiles for CAGR/Sharpe/Max Drawdown etc., a recent-equity table, and a full performance-metrics table), so magnitude is readable from the page though not from the charts themselves. Suggested severity P1 rather than P0: the charts convey shape only and their implied scale can mislead, but they are not the sole or authoritative performance readout in this single-operator localhost dashboard.

#### 4. [P0/perf-viz] Headline CAGR/Sharpe/Sortino/MDD are silently computed from only the last 120-240 equity points of the run, then presented as run-level performance

- **Where**: `src/lumina_quant/dashboard/overview_service.py:194`
- **Evidence**: load_overview_payload defaults `limit: int = 120` and queries `SELECT ... FROM equity WHERE run_id = %s ORDER BY id DESC LIMIT %s` (lines 234-245); build_overview_payload_from_frames then derives cagr/sharpe/sortino/max_drawdown from that truncated frame (lines 114-141). load_performance_price_payload does the same with `point_limit: int = 240` (cutover_surfaces_service.py:917). The UI labels these 'Derived metrics' / summary metrics with no window disclosure (overview-runtime.tsx:134-154).
- **Why it matters**: For a multi-month backtest with per-bar equity rows, 120 points is a tiny tail slice. 'Total Return', 'Max Drawdown', and Sharpe shown on the dashboard describe the last ~120 bars, not the run — a strategy that blew up early and recovered lately will look excellent. This actively misleads a go-live evaluation.
- **Recommendation**: Either compute metrics server-side over the full equity series (downsample only for the chart payload), or label the tiles explicitly ('last 120 points') and surface the run's stored full-period metrics. Never let the same truncated frame feed both the sparkline and the headline stats without disclosure.
- **Verifier correction**: Finding is accurate; refinements: (a) the report-export/snapshot path (cutover_surfaces_service.py ~880-900) makes it worse than stated — it labels tail-derived performance_metrics with the full run's period_start; (b) 'silently' is almost exact — the only hint is an undisclosed 'Equity Points' count metric; (c) additionally, annualization hardcodes periods=252 and CAGR uses point-count as days regardless of bar frequency, further distorting intraday runs.

#### 5. [P0/ux-states] Stop/Kill job actions fail completely silently (unhandled promise rejection, no feedback)

- **Where**: `apps/dashboard_web/components/workflow-jobs-runtime.tsx:77`
- **Evidence**: onClick={() => void triggerAction(job.job_id, 'stop')} (lines 77, 80). triggerAction throws on failure (line 26: `throw new Error(body.detail ?? body.error ?? 'workflow job action failed')`) but the click handlers void the promise with no .catch and no state update. The Python control endpoint currently returns `{"ok": False, "error": "missing_dsn"}` (src/lumina_quant/dashboard/workflow_jobs_service.py:140-143), so every Stop/Kill click throws into an unhandled rejection.
- **Why it matters**: A quant clicking Kill on a runaway backtest/walkforward job gets zero indication the kill did not happen — no error text, no toast, no row change. The user reasonably assumes the job was terminated while it keeps running and consuming the machine. This actively misleads on the single most safety-critical interaction in the dashboard.
- **Recommendation**: Catch triggerAction errors into a visible per-row or page-level error state; show success feedback (row status change or inline message) after refresh; disable the clicked button while the request is in flight.
- **Verifier correction**: The evidence's claim that the Python endpoint "currently returns {ok:false, error:'missing_dsn'}" is imprecise: missing_dsn is conditional on DSN resolution and is unreachable from the UI anyway, because the Next.js control route rejects stop/kill with 401 first (control_token_not_configured, or unauthorized when the token is set) since the frontend never sends the required x-lq-control-token header. The corrected statement: every UI Stop/Kill click is guaranteed to fail (401 from the bridge route's token gate), triggerAction throws, and the voided promise swallows it with no UI feedback — the buttons are non-functional in all configurations, not just when the DSN is missing.

#### 6. [P0/ux-states] missing_dsn renders as 'No managed workflow jobs recorded yet' — silent-empty indistinguishable from real empty history

- **Where**: `apps/dashboard_web/components/workflow-jobs-runtime.tsx:51`
- **Evidence**: `if (payload.jobs.length === 0) { return <p>No managed workflow jobs recorded yet.</p>; }` — the payload status field is never inspected. The service returns `{"jobs": [], "status": "missing_dsn"}` (src/lumina_quant/dashboard/workflow_jobs_service.py:118-121). Same pattern in overview-runtime.tsx:130 (`No managed workflow jobs have been recorded yet.`) even though that file has a DSN-aware helper it uses for other sections.
- **Why it matters**: 'No jobs recorded yet' is a factual claim about job history; when the DB is simply unreachable it is false and misleading. The user may conclude a launched job never registered, or that history was lost, instead of realizing LQ_POSTGRES_DSN is unset. This is exactly the actively-misleading silent-empty class.
- **Recommendation**: Branch on payload.status === 'missing_dsn' (and other non-ok statuses) before the empty-array check and render the DSN remediation message, like buildOverviewEmptyStateMessage already does for overview equity/runs sections.
- **Verifier correction**: Mechanism and evidence are exactly as stated; only the severity is overstated. On the overview page the misleading jobs message is partially mitigated: the top section header renders overview.source.status in a metric-badge (so 'missing_dsn' is visible on-screen, though unexplained), and adjacent equity/performance empty states show the explicit 'Set LQ_POSTGRES_DSN...' message when DSN is missing. The dedicated workflow-jobs page has no indicator at all. For a localhost single-operator dashboard with no data loss or unsafe action, P1 (actively-misleading empty state on the workflow-jobs page, inconsistent empty state on overview) is more accurate than P0.


### P1

#### 7. [P1/ia-nav] aria-current="page" is set on every nav link and there is no active-route indication anywhere

- **Where**: `apps/dashboard_web/components/dashboard-shell.tsx:21`
- **Evidence**: `<Link href={item.href} aria-current={item.status === 'available' ? 'page' : undefined}>` — aria-current is keyed to migration status, not the current route. All 11 navigationItems have status 'available' (lib/python-bridge.ts:197-275), so all 11 links simultaneously assert they are the current page. `grep -rn usePathname app components lib` returns nothing, and globals.css has no active/current nav style (nav rules at lines 70-99), so sighted users also get zero indication of which section they are on.
- **Why it matters**: Screen readers announce every sidebar link as 'current page' (actively misleading), and all users lose wayfinding: with 11 similar telemetry pages, the only way to know where you are is to read the page content. DashboardShell is a server component, so this cannot be fixed inline without a client boundary.
- **Recommendation**: Make the nav (or a small NavLink child) a client component using usePathname(); set aria-current='page' only when pathname === item.href, and add a visible active style (accent border/background) on .nav-item.
- **Verifier correction**: Finding is accurate as written. Minor severity note: for a localhost single-operator dashboard the screen-reader harm is mostly hypothetical, so P2 would also be defensible, but the universally-wrong ARIA plus total loss of visual wayfinding across 11 similar telemetry pages makes P1 reasonable.

#### 8. [P1/ia-nav] No global context header: no as-of/fetched-at timestamp, active run id, or live-vs-backtest environment indicator on any page

- **Where**: `apps/dashboard_web/components/dashboard-shell.tsx:6`
- **Evidence**: DashboardShell renders only brand block + nav + <main>{children}</main> (lines 6-33); layout.tsx adds nothing else. No component renders a fetch timestamp, data-freshness age, environment badge, or the active run_id at shell level — run_id exists in nearly every payload (lib/dashboard-contracts.ts:31,54,80,190,218,259,285,320,341) but surfaces only inside per-page tables (e.g. report-export-runtime.tsx:60). Combined with the established fetch-once/no-refresh behavior of use-bridge-fetch.ts and the current missing_dsn empty payloads, nothing tells the user when data was captured or whether it reflects a live session or an old backtest.
- **Why it matters**: For someone gating a real-money go-live, 'which run am I looking at, how stale is it, and is this live or backtest' is the single most important context, and it must be answerable from every page. A tab left open shows frozen data indistinguishable from current data; different pages can even reflect different runs after a new run starts, with no cross-page anchor to detect the mismatch.
- **Recommendation**: Add a shell-level context bar: active run_id + mode (live/backtest/optimize), payload generated_at or fetched-at with relative age, and connection status (e.g. surface source.status like 'missing_dsn' globally instead of per-page empty states).
- **Verifier correction**: Finding is accurate with two refinements: (a) one page (exact-window-runtime.tsx:150,163) does render payload.generated_at, so 'no page shows a data timestamp' should be scoped to 'no shell-level/cross-page indicator; only exact-window shows generated_at locally'; (b) the hardcoded 'live' status pills in overview-runtime.tsx and report-export-runtime.tsx are static className badges, not environment indicators — worth citing as they falsely suggest live status.

#### 9. [P1/ia-nav] Sidebar status pills are migration artifacts: 11 identical 'available' pills with zero decision-time information

- **Where**: `apps/dashboard_web/components/dashboard-shell.tsx:23`
- **Evidence**: `<span className={`status-pill status-${item.status}`}>{item.status}</span>` renders item.status for every nav item; the NavigationStatus type is 'available' | 'planned' (lib/python-bridge.ts:1) and every one of the 11 items is 'available', so the sidebar shows 11 identical green 'available' badges. The 'planned' branch is dead — grep shows no item and no CSS consumer uses it besides the type definition. Nav summaries are likewise migration prose, e.g. 'First parity slice backed by the Python compatibility contract' (python-bridge.ts:202).
- **Why it matters**: The pills occupy the highest-value pixel real estate in the app (right edge of every nav row) and communicate a constant. A trading operator needs pills that vary: data freshness per surface, error/empty state (missing_dsn), active alerts on Risk & Health, pending candidate count on Factor Insights. As-is the sidebar reads like a migration tracker, not navigation.
- **Recommendation**: Remove the status pills (they now encode a tautology) or repurpose them as live badges (e.g. risk-event count, stale-data warning, candidate-queue count). Rewrite nav summaries to state the operator question each page answers.
- **Verifier correction**: Minor imprecision only: the claim that no CSS consumer of 'planned' exists is wrong — app/globals.css:137 defines a .status-planned rule (with --planned: #ffbe5c at line 11). That CSS rule is itself dead code since no item ever has status 'planned', so the finding's substance is unchanged. Also worth adding: dashboard-shell.tsx:21 misuses the same constant status for aria-current, marking all 11 links as the current page permanently.

#### 10. [P1/ia-nav] Nav labels and grouping do not map to operator questions; 'Exact-window' is opaque jargon and candidate evaluation is scattered across four routes

- **Where**: `apps/dashboard_web/lib/python-bridge.ts:248`
- **Evidence**: Nav item 'Exact-window' with summary 'Latest exact-window artifact summary from the Python research bundle' (lines 248-253) names an internal artifact format, not a user task. 'How are my candidate strategies performing?' has no single home — it is split across 'Optimization Insights' (candidate quality, line 227), 'Factor Insights' (pending candidate review queue, line 269), 'Workflow Jobs' (backtest/optimize runs, line 234), and 'Exact-window', with nothing in the labels indicating which to open. Nav order (lines 197-275) is migration order: Risk & Health sits 7th of 11, below Market Data and Optimization Insights, and research surfaces are interleaved with live-operations surfaces.
- **Why it matters**: The stated core workflow is evaluating candidate strategies before go-live; a user cannot route that question from labels alone and must open several pages to assemble the answer. Ungrouped flat nav also mixes 'operate the live run' pages with 'research/evaluate' pages, the two distinct modes of this user's day.
- **Recommendation**: Group nav into 2-3 sections (e.g. Live Operations: Overview, Performance & Price, Execution, Risk & Health, Market Data; Research: Optimization, Factor Insights, Exact-window, Workflows; Data/Export: Raw Data, Report Export). Rename 'Exact-window' to something task-oriented (e.g. 'Walkforward Windows' or 'Research Artifacts') and put risk higher in the order.
- **Verifier correction**: Finding is accurate with one refinement: the sidebar (components/dashboard-shell.tsx line 25) displays each nav item's summary text under its label, so users see more than labels alone. This does not resolve the issue — the Exact-window summary is still internal artifact jargon, no summary designates a single home for candidate evaluation, and the nav remains a flat migration-ordered list mixing research and live-operations surfaces.

#### 11. [P1/market-info] recent_bars is a 24-row raw OHLCV table — no price chart, trend/regime unjudgeable at a glance

- **Where**: `apps/dashboard_web/components/market-data-runtime.tsx:87`
- **Evidence**: The only rendering of recent_bars is `<table>` with Timestamp/Open/High/Low/Close/Volume rows (market-data-runtime.tsx:87-118). The Python side loads up to 240 bars (`--point-limit 240`, lib/market-data-server.ts:8) but truncates the payload to `market_frame.tail(24)` (cutover_surfaces_service.py:713). A working sparkline helper `buildSparklinePath` already exists in lib/format.ts:12-33 and is used by overview/performance pages, but not here.
- **Why it matters**: Core question #2 is "can I easily judge the target market?". Trend direction, recent volatility, and regime are impossible to read from 24 rows of six-decimal numbers; 90% of the loaded data (240→24 bars) is discarded before it reaches the UI. This page is the market page and it has no visual price representation at all.
- **Recommendation**: Render the full 240-bar close series as at least a sparkline/line chart (helpers already exist), ideally candlesticks with volume; keep the table as a secondary detail view of the last N bars.
- **Verifier correction**: Finding is accurate as written. One nuance: the dashboard does show a price-line sparkline elsewhere ("Benchmark price preview" in performance-price-runtime.tsx, from metrics_frame.benchmark_price), but it is run telemetry on a different page, not the market page's OHLCV, so it does not mitigate this finding.

#### 12. [P1/market-info] "Indicator parity" section contains zero indicators — no RSI/ATR/vol/funding/regime, only labels

- **Where**: `src/lumina_quant/dashboard/cutover_surfaces_service.py:639`
- **Evidence**: indicator_summary is exactly four items: strategy name, the literal string "price-only parity preview", a min-max "Price Range" string, and "Timeframe Clamped: yes/no" (cutover_surfaces_service.py:639-703). Grep for rsi/atr/volatility/regime/funding in the service finds nothing except the performance-price funding_curve. The UI headlines this as "Indicator parity / Guarded market-view summary" (market-data-runtime.tsx:65-66).
- **Why it matters**: For a USDT-perp strategist, the market-state inputs that drive go/no-go (funding rate level, realized-vol percentile, trend/regime classification) are entirely absent, while the section title promises indicators. Bare, uncontextualized values (no overbought flag, no percentile) would already be weak — here there are no values at all, so the section is decorative.
- **Recommendation**: Compute a small deterministic indicator set from the already-loaded 240-bar frame (e.g. return vol vs trailing percentile, ATR, simple trend slope/regime tag, latest funding rate from the metrics table) and render each with contextual state (badge: high/normal/low), or rename/remove the section until it does.
- **Verifier correction**: Code claims are fully accurate: indicator_summary is exactly 4 items (cutover_surfaces_service.py:639-646 seed with Strategy + literal "price-only parity preview"; :686-703 extend with Price Range and Timeframe Clamped), no RSI/ATR/vol/regime/funding anywhere in the market-data payload or any other dashboard component, and market-data-runtime.tsx:65-66 headlines it "Indicator parity / Guarded market-view summary". However, the section self-discloses via its first row ("Indicator Mode: price-only parity preview"), so it is an honest placeholder rather than a silently misleading surface; combined with the localhost single-operator context, severity is better rated P2 (title/scope mismatch, missing decision-relevant market-state indicators) than P1.

#### 13. [P1/market-info] Traded universe invisible — page shows one symbol chosen by the most recent fill

- **Where**: `src/lumina_quant/dashboard/cutover_surfaces_service.py:150`
- **Evidence**: `_resolve_market_context` picks `fills_frame["symbol"].dropna().astype(str).iloc[-1]` (last fill wins), falling back to metadata symbols[0] then configured symbols[0] (cutover_surfaces_service.py:150-159). Config is multi-symbol (`trading.symbols: [BTC/USDT, ETH/USDT]`, configs/config.example.yaml:17-19). The UI renders that single symbol with no indication others exist (market-data-runtime.tsx:43-46); no page shows per-symbol exposure/allocation — only exact-window lists `requested_symbols` as a joined string (exact-window-runtime.tsx:110-111).
- **Why it matters**: In a multi-symbol run the market page silently flips between BTCUSDT and ETHUSDT depending on which traded last, and the user has no way to see the full universe, switch symbols, or compare per-symbol exposure. "What markets am I in and how much of each?" is unanswerable anywhere in the dashboard.
- **Recommendation**: Surface the full symbols list in market_context, add a symbol selector (query param into _load_market), and add a per-symbol exposure/position breakdown table sourced from fills/positions.
- **Verifier correction**: The finding is accurate with one softening: the universe is not fully invisible — execution-analytics-runtime.tsx:106 shows a symbol column per trade row, and exact-window-runtime.tsx:110-111 lists requested_symbols, so traded symbols can be inferred by scanning trades. However there is no per-symbol exposure/allocation view, no symbol switcher (no route accepts a symbol param), and the market context (used on multiple surfaces via _resolve_market_context at lines 633/1113/1191, not just one) is chosen by the most recent fill exactly as described.

#### 14. [P1/perf-viz] No run/strategy selection anywhere — every surface is hardwired to the single latest run

- **Where**: `src/lumina_quant/dashboard/cutover_surfaces_service.py:960`
- **Evidence**: `run_id = str(runs.iloc[0]["run_id"] or "")` (also overview_service.py:231, and analogous iloc[0] in execution/market/raw-data loaders). API routes accept no parameters: `export async function GET() { ... loadOverviewPayloadFromPython() }` (app/api/python/dashboard/overview/route.ts:6) — no request arg, no searchParams in any route (grep over app/api returns zero hits for searchParams). The overview section titled 'Run selection parity' (overview-runtime.tsx:159) is a read-only table with no interaction; no <select>/onClick exists in any runtime component except workflow stop/kill buttons.
- **Why it matters**: Evaluating candidate strategies means looking at more than the most recent run. The user cannot pull up strategy A's backtest next to strategy B's, nor revisit yesterday's walkforward once a new run starts — the newest run silently replaces everything on every page.
- **Recommendation**: Add a run_id query parameter through route.ts → python-bridge → service loaders (they already take run_id internally), and turn the 'Recent runs' table rows into links/selectors that re-fetch all surfaces for the chosen run.

#### 15. [P1/perf-viz] No visual cross-candidate comparison: candidate pools are plain tables, best candidate is a raw JSON dump, promoted/rejected status is not visually distinguished

- **Where**: `apps/dashboard_web/components/optimization-insights-runtime.tsx:40`
- **Evidence**: Best candidate renders as `<pre className="code-block">{JSON.stringify(payload.best_candidate ?? {}, null, 2)}</pre>`; top_candidates (12 rows with sharpe/train_sharpe/robustness/cagr/mdd) and stage_breakdown are undifferentiated tables (lines 47-107) — no bars, no rank highlighting, no sharpe-vs-train_sharpe overfit cue despite both columns being present. exact-window-runtime.tsx renders top_candidates (lines 230-256) and timeframes the same way, dropping the payload's `promoted` boolean entirely from the top-candidates table (contract dashboard-contracts.ts:169). factor-insights-runtime.tsx calls its table a heatmap but applies zero color encoding to cells (lines 45-70).
- **Why it matters**: Comparing candidates is exactly what the data supports (Sharpe/CAGR/MDD/robustness across up to 12 optimization candidates and 8 exact-window candidates), but the user must eyeball columns of unformatted floats. Which candidate won, which were promoted vs rejected, and where train Sharpe diverges from OOS Sharpe (overfit signal) are all invisible.
- **Recommendation**: Render best_candidate as a stat-tile card (Sharpe, CAGR, MDD, robustness, params summary); add inline bar cells or sorting to the candidate tables; highlight promoted rows and flag large train-vs-OOS Sharpe gaps; give the IC 'heatmap' an actual diverging background color per cell.
- **Verifier correction**: Two minor imprecisions: (1) in optimization-insights-runtime.tsx, lines 47-68 are the stage_breakdown table and 72-107 the top_candidates table (the finding's "lines 47-107" conflates them); (2) exact-window-runtime.tsx does surface `promoted` textually in the timeframes table (line 211: reject_reasons.join(', ') || (row.promoted ? 'promoted' : 'n/a')) — it is only the top-candidates table (230-256) that drops it entirely, and even the timeframes rendering is plain text with no visual distinction.

#### 16. [P1/perf-viz] benchmark_curve and trade_markers are never composed with the equity/price chart — benchmark is an independently normalized separate sparkline, trades are a 12-row table

- **Where**: `apps/dashboard_web/components/performance-price-runtime.tsx:62`
- **Evidence**: Benchmark renders as its own card (lines 62-72) whose path comes from the same min-max buildSparklinePath, so its shape shares no scale with the equity card above it. trade_markers render only as a table (lines 105-138); the producer even caps them at `trade_analytics.tail(12)` (cutover_surfaces_service.py:487). No overlay, no buy/sell glyphs, no PnL coloring.
- **Why it matters**: The two questions a benchmark and trade markers exist to answer — 'did the strategy beat BTC over the same window?' and 'where did it enter/exit relative to price?' — are unanswerable. Two independently normalized sparklines cannot be visually compared even in shape-relative terms, and 12 table rows of trades give no spatial context.
- **Recommendation**: Overlay a rebased (=100 at window start) benchmark line on the equity chart with a 2-entry legend, and plot trade markers as up/down triangles on the benchmark-price chart colored by realized PnL sign; keep the table as a detail view.

#### 17. [P1/perf-viz] Annualization hardcoded to periods=252 regardless of equity-row cadence — Sharpe/CAGR/vol are on the wrong scale for intraday crypto-perp equity curves

- **Where**: `src/lumina_quant/dashboard/overview_service.py:132`
- **Evidence**: `periods = 252` then `create_cagr(latest_equity, initial_equity, len(totals), periods)`, `create_sharpe_ratio(returns, periods=periods)` etc. (lines 132-141). The equity table for Binance USDT-perp runs is populated per bar/snapshot (state_store equity rows), not daily; timestamps are available in the frame but the cadence is never inferred.
- **Why it matters**: For hourly equity rows the Sharpe shown is understated by roughly sqrt(24/1)≈4.9x relative to a true annualized figure computed with the right period count, and CAGR treats 120 hourly points as 120 trading days. The headline tiles the user reads to rank candidates are numerically wrong, not just unlabeled.
- **Recommendation**: Infer periods-per-year from median timestamp spacing in the equity frame (already datetime-coerced) or from run metadata timeframe; at minimum display the assumed cadence next to the metrics.
- **Verifier correction**: Two refinements: (1) for 24/7 crypto-perp hourly bars the correct annual period count is ~8760, so Sharpe (and annualized vol) are understated by sqrt(8760/252)≈5.9x, not 4.9x (the 4.9x figure assumes 252 trading days × 24h); (2) an additional aggravator: load_overview_payload defaults to limit=120, so the metrics are computed over only the trailing ~120 equity rows of the latest run, further distorting the headline tiles. The repo's canonical fix already exists at src/lumina_quant/indicators/annualization.py (median_bar_spacing_seconds / bars_per_year_from_spacing) and should be wired into overview_service.py.

#### 18. [P1/perf-viz] No number formatting or directional color anywhere on the two main performance surfaces: ratios like max_drawdown/CAGR display as raw 6-decimal floats via String(value)

- **Where**: `apps/dashboard_web/lib/format.ts:38`
- **Evidence**: formatMetricValue is `return String(value);` — no percent conversion, no fixed decimals, no thousands separators, no sign. overview-runtime.tsx:147 uses `<strong>{String(value)}</strong>` for CAGR/Sharpe/Max Drawdown tiles fed by round(x, 6) (overview_service.py:157-162), so Max Drawdown shows as e.g. '0.235461' and Total Return as '0.048712'. No green/red or positive/negative CSS class exists in globals.css (only status-pill states). Only exact-window-runtime.tsx has its own formatPercent (lines 14-19), so the same MDD concept is '23.55%' on one page and '0.235461' on another.
- **Why it matters**: Percent-vs-ratio ambiguity on drawdown and returns is a classic misread ('0.23' — 0.23% or 23%?), and without sign coloring the stat tiles are not scannable. Inconsistency between pages forces mental re-normalization when comparing candidates across surfaces.
- **Recommendation**: Extend format.ts with formatPercent/formatRatio/formatSigned used by all runtimes; key by metric semantics (cagr/max_drawdown/total_return → percent, sharpe/sortino/calmar → 2-decimal ratio); add .positive/.negative classes for signed values.
- **Verifier correction**: Substantively accurate. One minor imprecision: exact-window-runtime.tsx is not the only component with local number formatting — factor-insights-runtime.tsx:7 also has its own toFixed(3) helper. Neither touches the two main performance surfaces cited, so the finding stands as written otherwise.

#### 19. [P1/ux-states] 8 of 10 data surfaces hide the missing_dsn root cause behind generic 'No X payload available yet'

- **Where**: `apps/dashboard_web/components/market-data-runtime.tsx:20`
- **Evidence**: market-data-runtime.tsx:19-20, performance-price-runtime.tsx:19-21, risk-health-runtime.tsx:18-20, raw-data-runtime.tsx:18-19, factor-insights-runtime.tsx:22-23, execution-analytics-runtime.tsx:18-19, optimization-insights-runtime.tsx:19-20, report-export-runtime.tsx:28-29 all collapse every non-ok status (including missing_dsn) into `return <p>No … payload available yet.</p>;`. Only overview-runtime (via lib/overview-status.ts:3-4, which says 'Set LQ_POSTGRES_DSN…') and exact-window (file-bundle statuses, lib/exact-window-status.ts) give actionable status-specific messages. The raw status string is not even echoed on these 8 pages.
- **Why it matters**: On this machine every page except overview reads as 'data not produced yet', when the actual fix is a one-line env var. The user must know to visit overview (or curl the API) to learn why 9 surfaces are blank. Distinguishing 'infrastructure not wired' from 'no results yet' is a core comprehension need for someone evaluating strategy runs.
- **Recommendation**: Extract the overview-status.ts pattern into a shared buildEmptyStateMessage(status) used by every runtime; at minimum render the raw payload.status in the fallback message (e.g., 'bridge returned status: missing_dsn').
- **Verifier correction**: Two count imprecisions. (a) factor-insights-runtime.tsx:22-23 does collapse non-ok statuses generically, but factor_insights_service.py never returns 'missing_dsn' — its only statuses are 'ok' and 'empty' (file-based, no DSN), so it hides 'empty', not the DSN root cause. (b) The finding misses workflow-jobs-runtime.tsx:52, which shows 'No managed workflow jobs recorded yet.' even when workflow_jobs_service.py:118-121 returns status='missing_dsn' (the runtime never inspects status at all). Net: 7 DSN-backed surfaces + workflow-jobs = 8 surfaces hide missing_dsn; factor-insights is generic-collapse but not a missing_dsn case.

#### 20. [P1/ux-states] No staleness indication and no refresh path: as_of exists in every contract but is never rendered, and all views freeze at page load

- **Where**: `apps/dashboard_web/lib/use-bridge-fetch.ts:19`
- **Evidence**: useBridgeFetch fires exactly one fetch in useEffect (lines 19-36) with no polling, no retry, no exposed refetch function. `grep -rn as_of components/ app/` returns zero hits although lib/dashboard-contracts.ts declares `as_of: string` on every payload (lines 20, 79, 94, 115, 189, 217, 258, 284, 319, 340, 374) and the Python side populates it (cutover_surfaces_service.py:178-183). workflow-jobs-runtime.tsx only calls refresh() on mount and after a successful control action (lines 28, 33) — a running job's status never updates on screen.
- **Why it matters**: Jobs progress over minutes/hours; equity and risk telemetry change during live runs. A tab left open shows arbitrarily stale data with no timestamp and no refresh button, so the user cannot tell whether 'queued' means still queued or the view is 20 minutes old. The data (as_of) to fix this is already delivered and thrown away.
- **Recommendation**: Render `as of {payload.as_of}` on every surface header; expose a refetch() from useBridgeFetch wired to a Refresh button; add a modest poll interval (5-15s) on the workflow jobs page specifically.
- **Verifier correction**: Minor overstatement only: a manual browser reload does fetch fresh data because every request uses cache: 'no-store', so a refresh path technically exists outside the app. The core defect stands — no in-app refresh control, no polling, and no rendered as_of timestamp, so the operator cannot tell the view is stale or know a reload is needed.

#### 21. [P1/ux-states] Kill/Stop are one-click destructive actions with no confirmation, no in-flight/disabled state

- **Where**: `apps/dashboard_web/components/workflow-jobs-runtime.tsx:80`
- **Evidence**: Both buttons fire immediately on click (lines 77-82) — no confirm dialog, no disabled attribute, no pending indicator. They are also plain `<button>` elements with no className, so they render as tiny default browser buttons adjacent to each other in the table cell; Kill sits directly next to Stop with identical styling. Nothing prevents double-clicks issuing duplicate control POSTs.
- **Why it matters**: Kill presumably SIGKILLs a workflow process; an accidental click (or hitting Kill when Stop was intended — they are visually identical and 2px apart) irreversibly terminates a long-running walkforward/optimization with no undo and, per the finding above, no feedback either way.
- **Recommendation**: Add window.confirm (or an inline two-step confirm) for kill at minimum; visually differentiate Kill (danger styling) from Stop; disable both buttons for a row while its action is in flight.
- **Verifier correction**: One imprecision: Kill does not SIGKILL. terminate_process (workflow_jobs_service.py line 38) sends SIGTERM on POSIX and `taskkill /PID <pid> /T /F` on Windows; the DB then records status='KILLED', exit_code=-9, which merely mimics SIGKILL semantics. It is still an immediate non-graceful termination (whole process tree on Windows), so the irreversible-termination consequence stands. Additionally, failures are silent in a stronger sense than claimed: the thrown error from triggerAction is an unhandled promise rejection, so failed actions produce no UI feedback at all.

#### 22. [P1/ux-states] Sparklines are min/max-normalized with no axes, scale, or endpoint labels — magnitude and sign are unreadable

- **Where**: `apps/dashboard_web/lib/format.ts:23`
- **Evidence**: buildSparklinePath rescales every series to fill the full 0-120px height: `const range = max - min || 1; … height - ((value - min) / range) * height` (lines 23-29). Consumers (overview-runtime.tsx:57-67, performance-price-runtime.tsx:46-81) render only `<path>` inside the SVG — no axis, no min/max tick, no first/last value label, no zero line on the drawdown chart.
- **Why it matters**: A -0.2% drawdown wiggle and a -45% crash produce pixel-identical charts. For the core go/no-go question ('how bad is this strategy's drawdown?') the equity/drawdown visuals convey shape only, and the user must fall back to the metric tables. Also affects a11y: the aria-label says 'Drawdown curve preview' but conveys no values to anyone.
- **Recommendation**: Overlay at least min/max/last text labels (SVG <text>) and a zero/baseline reference line on drawdown; alternatively print start→end values beside each chart.
- **Verifier correction**: The technical claim is fully accurate, but severity is mildly overstated for context: both pages display the quantitative answer (Max Drawdown metric tile, summary metric tiles, full performance-metrics table) adjacent to the sparklines, and the charts self-describe as 'preview' in their aria-labels with point-count pills — they are shape previews by design, not the sole source of the go/no-go number. For a localhost single-operator dashboard this is closer to P2 (charts convey shape only; add min/max/last-value endpoint labels and a zero baseline on the drawdown/funding charts) than P1. The a11y sub-point stands as written.


### P2

#### 23. [P2/ia-nav] Triple redundancy on home page: sidebar nav, route-contract table, and 'Foundation scope' cards all list the same 11 surfaces

- **Where**: `apps/dashboard_web/app/page.tsx:120`
- **Evidence**: buildOverviewCards() (python-bridge.ts:277-283) maps dashboardBridgeContract.capabilities 1:1 into the 'Available now' card grid (page.tsx:120-138); the same capabilities array is already rendered as the 11-row 'Legacy-to-web route contract' table on the same page (page.tsx:64-88); and the sidebar (dashboard-shell.tsx:19-27) lists the same surfaces a third time. All three carry identical 'available' status pills. The cards are not even links — they are static <article> elements.
- **Why it matters**: One screen presents the same static inventory three times, consuming most of the home page's space budget while providing no navigation affordance (cards aren't clickable) and no state that the sidebar doesn't already show.
- **Recommendation**: Delete the 'Foundation scope' section and the route-contract table from the home page (the sidebar is the canonical inventory); if the contract table has audit value, move it to a /system page.
- **Verifier correction**: The triple listing is real, but the sidebar is driven by a separate navigationItems array (python-bridge.ts:197-275), not the same capabilities array, and the two lists agree on only 10 of 11 surfaces: navigationItems includes 'factor-insights' but not 'python-compatibility', while capabilities (feeding both the route-contract table and the overview cards) include 'python-compatibility' but not 'factor-insights'. Also, the route-contract table is not fully redundant — it uniquely shows sourceModule and nextRoute migration-mapping columns; the truly redundant element is the 'Foundation scope' card grid, whose non-clickable cards duplicate the sidebar's titles, statuses, and near-identical descriptions while discarding the nextRoute needed to make them links.

#### 24. [P2/ia-nav] Every page hero and the shell brand block speak in migration/engineering language ('parity slice', '8GB-safe', 'Python-backed')

- **Where**: `apps/dashboard_web/components/dashboard-shell.tsx:13`
- **Evidence**: Shell lede: 'Next.js dashboard foundation, kept intentionally lean for the 8GB baseline.' (dashboard-shell.tsx:13-15). Route heroes repeat the pattern: 'This route covers the highest-priority performance slice while staying Python-backed and 8GB-safe.' (app/performance-price/page.tsx:9-11), 'Performance & Price parity' / 'Workflow parity' / 'Risk & Health parity' eyebrows on every page, plus per-section eyebrows like 'Python-backed performance feed'. Home hero: 'Overview parity slice' (app/page.tsx:10).
- **Why it matters**: Permanent UI chrome addresses the implementer, not the operator. Hero space on every page is spent restating how the page is built instead of what it shows (symbols, run, period), and 'parity slice' phrasing implies a temporary scaffold, undermining trust in a tool meant to gate real-money decisions.
- **Recommendation**: Rewrite heroes to state content and context ('Equity, drawdown & benchmark — run <id>, as of <time>'); drop 8GB/parity/Python-backed phrasing from user-facing copy.
- **Verifier correction**: Minor precision fixes only: in performance-price/page.tsx the quoted lede is line 10 (eyebrow 'Performance & Price parity' is line 7). Scope is broader than stated: beyond page heroes, the migration language also appears in runtime components — overview-runtime.tsx lines 99/137/159 ('Workflow parity', 'Performance parity', 'Run selection parity'), exact-window-runtime.tsx lines 40/181/225/265, and loading messages like 'Loading market data parity payload…' (market-data-runtime.tsx:17, raw-data-runtime.tsx:16, optimization-insights-runtime.tsx:17), plus an '8GB guardrail' eyebrow on the home page (app/page.tsx:20).

#### 25. [P2/ia-nav] Static document title for all routes: no per-page metadata

- **Where**: `apps/dashboard_web/app/layout.tsx:9`
- **Evidence**: layout.tsx exports the only Metadata in the app: title 'LuminaQuant Dashboard Web' (lines 8-11); `grep -rn metadata app --include=page.tsx` matches no route page, so all 11 routes share one browser-tab title and history entry label.
- **Why it matters**: An operator working across multiple tabs (Risk & Health open next to Optimization Insights, a common pattern for this audience) cannot distinguish tabs, and browser history/bookmarks are unusable for returning to a specific surface.
- **Recommendation**: Export `metadata` (or use title.template in layout) per route, e.g. 'Risk & Health — LuminaQuant'.

#### 26. [P2/market-info] Non-ok statuses (missing_dsn, no_runs) collapse into a dead-end generic message

- **Where**: `apps/dashboard_web/components/market-data-runtime.tsx:19`
- **Evidence**: `if (payload.status !== 'ok' && payload.status !== 'no_market_data') { return <p>No market data payload available yet.</p>; }` (market-data-runtime.tsx:19-21); same pattern in raw-data-runtime.tsx:18-20. The Python side deliberately encodes the reason (`_empty_surface_payload(reason="missing_dsn"|"no_runs")`, cutover_surfaces_service.py:1082,1102) but the UI discards it.
- **Why it matters**: On this machine every surface returns missing_dsn, so the market page permanently reads "No market data payload available yet" — indistinguishable from "data is still loading" or "no runs exist", with no remediation hint (set LQ_POSTGRES_DSN). The user cannot tell a config problem from an empty database.
- **Recommendation**: Render payload.status verbatim with a per-reason hint (missing_dsn → "configure LQ_POSTGRES_DSN"; no_runs → "no runs recorded yet"), consistently across all runtimes.
- **Verifier correction**: Finding is accurate as stated, with one softening nuance: the remediation hint (Set LQ_POSTGRES_DSN) IS shown on the overview page via buildOverviewEmptyStateMessage (lib/overview-status.ts, used only by overview-runtime.tsx), so the operator is not entirely without a hint site-wide — only the market-data and raw-data pages are dead ends. Also raw-data-runtime.tsx is stricter than quoted: it treats ANY status !== 'ok' as the generic message. Fix is trivial: reuse the existing buildOverviewEmptyStateMessage helper in both components.

#### 27. [P2/market-info] Summary metrics computed over an unlabeled 240-bar window while the table shows 24 bars

- **Where**: `src/lumina_quant/dashboard/cutover_surfaces_service.py:660`
- **Evidence**: Latest Close / Price Change % / Price Range are computed over the whole loaded frame (up to 240 bars: close_series first vs last at lines 660-664, low_series.min()/high_series.max() at line 693), but recent_bars is tail(24) (line 713). No metric label or UI copy states the window (240 x 1m ≈ 4h). "Market Bars: 240" (line 667) is the only clue.
- **Why it matters**: The user cannot reconcile "Price Change %" or "Price Range" against the visible table — the range spans data the table doesn't show, and the change period is undisclosed, so the numbers can't be trusted for a timeframe-sensitive judgment.
- **Recommendation**: Label the window explicitly (e.g. "Price Change (last 240 x 1m bars / ~4h)") or compute summary metrics over the same window the UI displays.
- **Verifier correction**: Two minor imprecisions: (1) the window is up to 240 bars of the run's (possibly clamped) timeframe, not necessarily 1m, so "240 x 1m ≈ 4h" is illustrative only; (2) besides "Market Bars: 240", the Market context card also shows "Timeframe", so a diligent operator could derive the window — but nothing discloses that summary metrics span the full window while the table shows only tail(24).

#### 28. [P2/market-info] Raw-data previews rendered as JSON.stringify dumps despite column metadata being available

- **Where**: `apps/dashboard_web/components/raw-data-runtime.tsx:76`
- **Evidence**: `<pre className="code-block">{JSON.stringify(preview.rows, null, 2)}</pre>` (raw-data-runtime.tsx:76) while the payload carries `preview.columns` (dashboard-contracts.ts:332-336) that the component never uses. The "Market OHLCV" frame preview thus appears as a wall of JSON. context.market is a pre-concatenated string built server-side (cutover_surfaces_service.py:1216-1220).
- **Why it matters**: The raw-data page is the fallback for inspecting what the system actually ingested; a JSON blob for OHLCV rows is far harder to scan than the table the same app renders on the market page, and the preformatted market string prevents any structured reuse client-side.
- **Recommendation**: Render previews as tables using preview.columns (the table-wrap pattern already exists in this file), and pass symbol/timeframe/exchange as separate fields.
- **Verifier correction**: Finding is accurate as written; only nuance is that P2 is the upper bound — P3 would also be defensible since the page is a diagnostic fallback on a localhost single-operator dashboard and all data remains visible, merely unscannable.

#### 29. [P2/perf-viz] Overview equity/drawdown cards are always labeled 'live' with green status pills even for finished backtests or stale data

- **Where**: `apps/dashboard_web/components/overview-runtime.tsx:55`
- **Evidence**: `<span className="status-pill status-available">live</span>` is a hardcoded literal on both chart cards (lines 55 and 64) regardless of run.mode/status; performance-price cards similarly hardcode status-available showing only point counts (performance-price-runtime.tsx:44, 55, 66, 77).
- **Why it matters**: A quant deciding on real-money go-live must know whether numbers describe a live session or an old backtest. A permanent green 'live' badge on a completed backtest curve misstates data provenance.
- **Recommendation**: Derive the pill from payload.source.mode/status and the run's mode (backtest/paper/live) and show the as_of timestamp on the card.
- **Verifier correction**: The finding is accurate but overstates the isolation of the misleading label. Two partial mitigations exist on the same page: (1) the section header renders `overview.source.status` in a metric-badge (overview-runtime.tsx line 48), though its values are bridge-health states ("ok"/"missing_dsn"/"no_runs"/"no_equity"), not run provenance, so it does not distinguish live vs backtest; (2) the "Recent runs" table lower on the same page shows each run's actual mode and status columns, so an attentive operator can cross-check that the top run is a backtest. The pill therefore misstates provenance at a glance but the truth is visible on the same screen. For a localhost single-operator dashboard, P2 is at the high end; P2-P3 boundary. The fix is trivial since the payload already carries run mode/status (recent_runs[0].mode/.status or summary_metrics "Mode"/"Status") — the frontend just never uses them for the pill.

#### 30. [P2/perf-viz] factor_ranking (t-stat, IC positive ratio, turnover, quantile spread) and candidate sharpe/robustness/submitted_at are delivered in the payload but never rendered

- **Where**: `apps/dashboard_web/components/factor-insights-runtime.tsx:26`
- **Evidence**: Component destructures only `ic_heatmap` and `candidate_queue`; the contract's factor_ranking array (dashboard-contracts.ts:392-401, produced by _factor_ranking in factor_insights_service.py:119) is unused, and the candidate table (lines 72-93) shows only candidate_id/strategy/status/score, dropping sharpe, robustness_score, and submitted_at that _build_candidate_queue emits (factor_insights_service.py:166-167).
- **Why it matters**: The statistical-significance columns (t_stat, ic_positive_ratio) are precisely what distinguishes a real factor from noise, and candidate sharpe/robustness are the comparison axes the user cares about. The backend already pays for this data; the UI throws it away.
- **Recommendation**: Add a factor-ranking table (or bar chart of IC-IR with t-stat annotation) and extend the candidate queue table with sharpe/robustness/submitted_at columns.
- **Verifier correction**: Finding is accurate as stated. One minor nuance the finder could add: factor_ranking's ic_mean/ic_ir fields are partially redundant with the rendered ic_heatmap (IC-IR column at line 65 of factor-insights-runtime.tsx), so the genuinely lost columns are t_stat, ic_positive_ratio, turnover_mean, quantile_spread_mean, and n_periods, plus candidate sharpe/robustness_score/submitted_at.

#### 31. [P2/perf-viz] alpha-evidence API route has no page or component — the alpha classification / reality-gate / live-readiness surface is unreachable from the web UI

- **Where**: `apps/dashboard_web/app/api/python/dashboard/alpha-evidence/route.ts:1`
- **Evidence**: app/ contains pages for exact-window, execution-analytics, factor-insights, market-data, optimization-insights, performance-price, raw-data, report-export, risk-health, workflows — no alpha-evidence page, and grep for 'alpha-evidence|AlphaEvidence' across components/ and app/*.tsx returns nothing, while the API route and AlphaEvidencePayload contract (dashboard-contracts.ts:92-112, alpha_evidence_service.py:41) both exist with classifications, reality_gates, and live_readiness_action.
- **Why it matters**: Reality-gate pass/fail and live-readiness action are the final checklist before real-money go-live; the data pipeline exists end-to-end but the user has no page to see it.
- **Recommendation**: Add an alpha-evidence page rendering classifications as counts, reality_gates as pass/fail badges with observation counts, and live_readiness_action as a prominent callout.
- **Verifier correction**: Finding is accurate as scoped to the web UI. One nuance: the alpha-evidence data is not entirely unreachable to the operator — src/lumina_quant/dashboard/mcp_server.py exposes a read-only get_alpha_evidence MCP tool and bridge.py exposes the same payload, so the surface exists outside the browser. The web-UI gap itself is confirmed: no page directory, no runtime component fetches /api/python/dashboard/alpha-evidence, and navigationItems in apps/dashboard_web/lib/python-bridge.ts has no alpha-evidence entry.

#### 32. [P2/ux-states] Error state is a bare unstyled string; non-JSON responses surface raw parse exceptions; no retry

- **Where**: `apps/dashboard_web/lib/bridge-fetch.ts:4`
- **Evidence**: readJsonOrThrow calls `await response.json()` before checking response.ok (lines 4-5), so a 502/504 HTML error page from the bridge yields a SyntaxError like "Unexpected token '<'…" which every runtime renders verbatim as `return <p>{error}</p>;` (e.g. overview-runtime.tsx:14-16, market-data-runtime.tsx:13-15) — no error styling, no context, no retry button. Loading is likewise a bare `<p>Loading …</p>` with no skeleton, so section cards pop in with layout shift.
- **Why it matters**: A transient bridge hiccup leaves the page permanently showing a cryptic parse error until a manual full-page reload; the user cannot distinguish 'my Python service is down' from 'proxy returned HTML'.
- **Recommendation**: Wrap response.json() in try/catch and throw `${fallbackMessage} (HTTP ${response.status})` when parsing fails; render errors in a styled callout with a Retry button (needs the refetch from the staleness finding).
- **Verifier correction**: The json()-before-ok ordering in bridge-fetch.ts, the bare unstyled `<p>{error}</p>` / `<p>Loading …</p>` states, and the absence of any retry mechanism (one-shot fetch on mount in use-bridge-fetch.ts) are all confirmed. But the '502/504 HTML from the bridge yields SyntaxError' scenario is inaccurate: the Python bridge is a spawned subprocess (runUvPythonModuleJson), not a proxied HTTP upstream, and every route handler plus the middleware returns JSON on both success and failure (e.g., overview/route.ts catch returns {error, detail} JSON with status 500). In practice a bridge failure shows a readable Python error detail, not a parse exception; raw SyntaxError can only occur if the Next.js server itself emits a non-JSON error page. Corrected severity: P3 (unpolished error/loading UX, no retry, sticky error until manual reload) for a localhost-only single-operator dashboard.

#### 33. [P2/ux-states] aria-current="page" is set on every 'available' nav item regardless of the active route

- **Where**: `apps/dashboard_web/components/dashboard-shell.tsx:21`
- **Evidence**: `<Link href={item.href} aria-current={item.status === 'available' ? 'page' : undefined}>` — the condition is the item's availability status, not a comparison with the current pathname. All available nav links simultaneously announce themselves as the current page, and there is no visual active-route highlight either (no usePathname anywhere in the shell).
- **Why it matters**: Screen-reader users hear 'current page' on ~10 links at once, which is worse than no aria-current. Sighted users also get zero indication of which of the 11 sections they are on — the sidebar looks identical on every route.
- **Recommendation**: Use usePathname() (client) or a layout-level prop to set aria-current only on the matching href, and add an active style for that nav item.

#### 34. [P2/ux-states] Hardcoded 'live' status pills on a one-shot frozen snapshot

- **Where**: `apps/dashboard_web/components/overview-runtime.tsx:55`
- **Evidence**: `<span className="status-pill status-available">live</span>` is a string literal (overview-runtime.tsx:55, 64; report-export-runtime.tsx:95, 102) rendered whenever the payload exists, in green 'available' styling. The data behind it is fetched once at mount and never updates (use-bridge-fetch.ts).
- **Why it matters**: 'live' plus green implies streaming/real-time telemetry; combined with the missing as_of timestamp, the user is told the opposite of the truth about data freshness.
- **Recommendation**: Replace 'live' with the payload's as_of timestamp or a neutral 'snapshot' label; reserve green live styling for an actual polling mode.
- **Verifier correction**: Minor correction: the as_of timestamp is not missing from the data — OverviewPayload (lib/dashboard-contracts.ts:20) includes as_of: string, but overview-runtime.tsx never renders it (exact-window-runtime.tsx does render its generated_at, so the display pattern exists in the codebase). This makes the fix trivial and the finding slightly stronger: fresh-ness data is delivered and silently discarded while a hardcoded 'live' pill is shown instead.


## P3 nits (unverified)

- **[ia-nav]** Sidebar nav items are full cards with summary paragraphs, forcing the nav itself to scroll — `apps/dashboard_web/app/globals.css:78`. Collapse nav items to compact label rows (summary as title/tooltip), and make the sidebar sticky with its own overflow-y.
- **[perf-viz]** Overview 'recent equity' table shows the last 5 raw equity rows with unformatted floats and ISO timestamps — near-zero information content — `apps/dashboard_web/components/overview-runtime.tsx:21`. Replace with a compact summary row (start equity, end equity, period return %, max DD %) or drop it once the chart gains axis labels.
- **[market-info]** Bar table shows verbose ISO timestamps, newest row buried at the bottom — `apps/dashboard_web/components/market-data-runtime.tsx:103`. Sort newest-first (or highlight the latest row), and format timestamps compactly (HH:mm, with the date shown once).
- **[ux-states]** Shared flex rule contaminates .table-wrap and .metric-grid layout — `apps/dashboard_web/app/globals.css:85`. Remove .metric-grid and .table-wrap from the grouped flex rule; give each its own layout rule.
- **[ux-states]** Dark-only theme with no prefers-color-scheme support (contrast itself is good) — `apps/dashboard_web/app/globals.css:2`. No action required now; if theming is ever added, the existing CSS-variable structure makes a light override cheap.
- **[ux-states]** Table headers lack scope attributes; mobile layout forces scrolling past the full nav on every page — `apps/dashboard_web/app/globals.css:242`. Add scope="col" to header cells; on the <=960px breakpoint collapse nav descriptions (hide .nav-item p) or make the nav a horizontal scroller.


## What is already good

- Contract-typed API layer (`lib/dashboard-contracts.ts`) mirrored by Python services; typed bridge with 32MB buffer guard.
- Security middleware: localhost-only + timing-safe token, fail-closed control route.
- Home empty state names the root cause (missing_dsn badge + "Set LQ_POSTGRES_DSN...") — the pattern the other 8 surfaces should adopt.
- Exact-window page proves percent formatting and reject-reason surfacing are already achievable in this codebase.
- Contrast/heading semantics pass; tables overflow-wrap correctly.

## Suggested fix order

**Week 1 (data trust + silent failures)**
1. Full-run performance metrics + cadence-aware annualization (`overview_service.py:132,194`)
2. Price Change % x100 (`cutover_surfaces_service.py:680`)
3. `formatMetricValue`: percent/round/thousands/directional color — single seam used by all surfaces (`lib/format.ts:38`)
4. Status-aware empty-state helper shared by all runtimes (kills the missing_dsn masquerade, incl. workflows false copy)
5. Workflow control: confirm dialog, in-flight/disabled state, error surface (`workflow-jobs-runtime.tsx:77-84`)
6. Remove hardcoded 'live' pills; render `as_of` in shell header

**Next (answer the operator's questions)**
7. One real chart component (axes/ticks/hover): equity+rebased benchmark overlay, drawdown shading, trade markers
8. Market Data: price line/candles from the already-loaded 240 bars, funding/vol/regime tiles, symbol switcher (universe view)
9. Candidate comparison view: sharpe/robustness bars, train-vs-OOS scatter, best-candidate card instead of JSON dump
10. Home inversion: OverviewRuntime first; migration meta -> /system page
11. `useBridgeFetch` refetch + polling for jobs; active-route nav highlight (fix aria-current)

## Artifacts

- Interactive report with annotated screenshots: claude.ai artifact `dashboard-uiux-audit` (2026-07-10)
- Raw findings JSON + 25 captures: session scratchpad (`shots/`, `audit-result.json`) — regenerate via Playwright mock-injection harness if needed.
