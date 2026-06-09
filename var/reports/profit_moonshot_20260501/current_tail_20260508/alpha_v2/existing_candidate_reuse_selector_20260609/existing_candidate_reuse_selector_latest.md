# Existing candidate reuse selector research

- generated: `2026-06-09T12:20:35.380056Z`
- candidate rows: `100000`
- folds: `10`
- selection input: `train + validation only`
- locked-OOS: `report only after fold freeze`
- promotion: `blocked; fresh-forward required`

## Variants

| Variant | OOS comp | Annualized | MDD | Positive folds | PF |
| --- | ---: | ---: | ---: | ---: | ---: |
| `robust_top1` | 22.14% | 27.12% | 3.10% | 6/10 | 4.95 |
| `robust_top2_equal` | 20.02% | 24.48% | 3.67% | 7/10 | 4.00 |
| `robust_diverse3_equal` | -0.07% | -0.08% | 8.12% | 5/10 | 1.03 |
