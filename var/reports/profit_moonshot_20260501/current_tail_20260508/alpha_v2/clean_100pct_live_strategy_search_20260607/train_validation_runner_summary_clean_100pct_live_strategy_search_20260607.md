# Train/validation-only runner summary

- Generated UTC: `2026-06-07T07:11:42Z`
- Ultragoal: `G004-train-validation-only-search-or-runn`
- New manifest-bound heavy runner executed: **false**

## Existing train/validation-first artifact

- Artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607/clean_new_alpha_discovery_latest.json`
- Pre-registered search-space SHA256: `4a6fee0f540f5d9ce15158beaf6b7c91ad89600cb5d76e1c4bfa0e33008b81b7`
- Selection inputs: `['train', 'validation']`
- Locked-OOS selection/objective/pruning/fitting flags: `False` / `False` / `False` / `False`
- OOS report-only annualized approx: `3.01%`
- Compounded OOS: `2.51%`
- Max OOS MDD: `8.77%`
- Label blockers: `['continuous_position_state_across_split_boundaries', 'fresh_forward_required_before_promotion']`

## Feature-bounded diagnostic

- Annualized approx: `-0.57%`
- Compounded OOS: `-0.24%`
- Max OOS MDD: `8.32%`

## Decision

No current clean process produced a manifest-bound 100%+ candidate. Existing train/validation-first artifacts stay report-only because they predate this manifest and/or explicitly set `clean_promotion_eligible=false`.
