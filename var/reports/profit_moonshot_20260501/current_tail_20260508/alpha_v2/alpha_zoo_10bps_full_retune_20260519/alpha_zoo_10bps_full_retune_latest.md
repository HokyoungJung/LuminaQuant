# Alpha Zoo 10bps full retune

- real_money_execution: `False`
- primary_cost_bps: `10.0`
- candidate_model_count: `600`
- live_promotable_10bps_model_id: `None`
- memory_pass_under_8gb: `True`

Locked-OOS is gate/report-only after candidate freeze; prior OOS/top-bucket references remain shadow-only until regenerated through train+validation-only retune.
