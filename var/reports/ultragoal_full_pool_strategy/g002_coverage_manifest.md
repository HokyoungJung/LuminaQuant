# G002 Full-Pool Universe/Data Coverage Manifest

- Generated: `2026-07-02T14:05:11.618850Z`
- Manifest content SHA256 (excluding self-hash): `b88508af518a33b339112a1701d658cc8ab2c6ae3487155f6fc50615a420f3d6`
- Whole-file SHA256 is recorded in `g002_verification_test_report.json` and differs by design.
- ExchangeInfo payload SHA256: `e2a62af464a3bb12caacdbefa2e04d38fe3554f93ad8fdb13817c58e5f2bb01b`
- Symbols: 128 (10 core crypto, 118 TradFi perps)
- Latest archive OHLCV-ready symbols: 128/128
- Frozen historical candidate eligible: 10/128
- Feature-dependent eligible: 9/128
- Support refresh upserted rows: 95282
- Live shadow / paper / real-money execution: **false / blocked**

## Downstream enforcement contract

Full-pool latest OHLCV coverage is an observation-pool refresh, not blanket candidate eligibility. G004/G005 must honor `eligible_for_frozen_historical_candidate_search`, `eligible_for_feature_dependent_families`, and every quarantine reason. Quarantined symbols are excluded from historical promotion until a later manifest explicitly proves the missing coverage/status.

## Quarantine reason counts

- `feature_or_funding_coverage_insufficient_for_feature_dependent_families`: 1
- `insufficient_train_validation_locked_oos_history`: 118
- `latest_exchangeInfo_status_TRADING_HALT_not_currently_trading`: 1

## ExchangeInfo deltas

- Added vs static: ALABUSDT, BSPUSDT, CATUSDT, CIENUSDT, FLEXUSDT, KLACUSDT, KORUUSDT, KSTRUSDT, LRCXUSDT, MVLLUSDT, SMCIUSDT, SONYUSDT, SQQQUSDT, STRCUSDT, TERUSDT, TQQQUSDT, TTWOUSDT, TXNUSDT
- Static symbols absent from latest TRADING TradFi set: CRWDUSDT
- Static non-TRADING in exchangeInfo: CRWDUSDT

## Coverage table

| symbol | kind | OHLCV min | OHLCV max | latest archive | historical | feature | quarantine |
|---|---|---|---|---:|---:|---:|---|
| `BTCUSDT` | core_crypto | `2023-11-14T22:13:20Z` | `2026-07-02T12:53:30Z` | True | True | True | clean |
| `ETHUSDT` | core_crypto | `2025-01-01T00:00:05Z` | `2026-07-02T12:53:30Z` | True | True | True | clean |
| `SOLUSDT` | core_crypto | `2025-01-01T00:00:05Z` | `2026-07-02T12:53:30Z` | True | True | True | clean |
| `BNBUSDT` | core_crypto | `2025-01-01T00:00:05Z` | `2026-07-02T12:53:30Z` | True | True | True | clean |
| `TRXUSDT` | core_crypto | `2025-01-01T00:00:05Z` | `2026-07-02T12:53:30Z` | True | True | True | clean |
| `XRPUSDT` | core_crypto | `2025-01-01T00:00:05Z` | `2026-07-02T12:53:30Z` | True | True | True | clean |
| `DOGEUSDT` | core_crypto | `2025-01-01T00:00:05Z` | `2026-07-02T12:53:30Z` | True | True | True | clean |
| `ADAUSDT` | core_crypto | `2025-01-01T00:00:05Z` | `2026-07-02T12:53:30Z` | True | True | True | clean |
| `AVAXUSDT` | core_crypto | `2025-01-01T00:00:05Z` | `2026-07-02T12:53:30Z` | True | True | True | clean |
| `TONUSDT` | core_crypto | `2025-01-01T00:00:05Z` | `2026-07-02T13:31:54Z` | True | True | False | feature_or_funding_coverage_insufficient_for_feature_dependent_families |
| `XAUUSDT` | tradfi_perpetual | `2025-12-11T08:05:20Z` | `2026-07-02T12:53:30Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `XAGUSDT` | tradfi_perpetual | `2026-01-07T10:00:01Z` | `2026-07-02T12:53:30Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `TSLAUSDT` | tradfi_perpetual | `2026-01-28T14:30:02Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `XPTUSDT` | tradfi_perpetual | `2026-01-30T10:00:00Z` | `2026-07-02T12:53:30Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `XPDUSDT` | tradfi_perpetual | `2026-01-30T10:15:00Z` | `2026-07-02T12:53:30Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `INTCUSDT` | tradfi_perpetual | `2026-02-02T14:29:59Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `HOODUSDT` | tradfi_perpetual | `2026-02-02T14:45:00Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `MSTRUSDT` | tradfi_perpetual | `2026-02-09T14:30:00Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `AMZNUSDT` | tradfi_perpetual | `2026-02-09T14:40:01Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `CRCLUSDT` | tradfi_perpetual | `2026-06-21T11:01:00Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `COINUSDT` | tradfi_perpetual | `2026-06-21T11:01:22Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `PLTRUSDT` | tradfi_perpetual | `2026-06-21T11:01:23Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `COPPERUSDT` | tradfi_perpetual | `2026-03-06T09:00:01Z` | `2026-07-02T12:53:30Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `EWYUSDT` | tradfi_perpetual | `2026-03-16T13:30:01Z` | `2026-07-02T12:53:30Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `EWJUSDT` | tradfi_perpetual | `2026-03-19T13:30:01Z` | `2026-07-02T12:53:30Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `PAYPUSDT` | tradfi_perpetual | `2026-06-21T11:01:00Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `METAUSDT` | tradfi_perpetual | `2026-06-21T11:01:06Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `NVDAUSDT` | tradfi_perpetual | `2026-06-21T11:01:05Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `GOOGLUSDT` | tradfi_perpetual | `2026-06-21T11:01:25Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `CLUSDT` | tradfi_perpetual | `2026-04-01T09:00:00Z` | `2026-07-02T12:53:30Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `BZUSDT` | tradfi_perpetual | `2026-04-01T09:10:01Z` | `2026-07-02T12:53:30Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `NATGASUSDT` | tradfi_perpetual | `2026-04-01T09:20:01Z` | `2026-07-02T12:53:30Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `QQQUSDT` | tradfi_perpetual | `2026-04-06T13:30:02Z` | `2026-07-02T12:53:30Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `SPYUSDT` | tradfi_perpetual | `2026-04-06T13:40:03Z` | `2026-07-02T12:53:30Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `AAPLUSDT` | tradfi_perpetual | `2026-06-21T11:04:05Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `TSMUSDT` | tradfi_perpetual | `2026-06-21T11:01:06Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `MUUSDT` | tradfi_perpetual | `2026-06-21T11:01:00Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `SNDKUSDT` | tradfi_perpetual | `2026-06-21T11:01:00Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `MSFTUSDT` | tradfi_perpetual | `2026-06-21T11:01:21Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `AVGOUSDT` | tradfi_perpetual | `2026-06-21T11:01:50Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `BABAUSDT` | tradfi_perpetual | `2026-06-21T11:03:02Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `AMDUSDT` | tradfi_perpetual | `2026-06-21T11:01:11Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `QCOMUSDT` | tradfi_perpetual | `2026-06-21T11:01:00Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `USARUSDT` | tradfi_perpetual | `2026-06-21T11:02:38Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `LITEUSDT` | tradfi_perpetual | `2026-06-21T11:01:24Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `ORCLUSDT` | tradfi_perpetual | `2026-06-21T11:01:38Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `DISUSDT` | tradfi_perpetual | `2026-06-21T11:04:41Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `UBERUSDT` | tradfi_perpetual | `2026-06-21T11:03:51Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `CSCOUSDT` | tradfi_perpetual | `2026-06-21T11:17:28Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `HDUSDT` | tradfi_perpetual | `2026-06-21T11:02:33Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `SOXLUSDT` | tradfi_perpetual | `2026-05-15T14:00:09Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `MRVLUSDT` | tradfi_perpetual | `2026-06-21T11:01:06Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `CRWVUSDT` | tradfi_perpetual | `2026-06-21T11:01:32Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `WMTUSDT` | tradfi_perpetual | `2026-06-21T11:03:48Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `JPMUSDT` | tradfi_perpetual | `2026-06-21T11:02:00Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `VUSDT` | tradfi_perpetual | `2026-06-21T11:05:03Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `BRKBUSDT` | tradfi_perpetual | `2026-06-21T11:01:23Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `FLNCUSDT` | tradfi_perpetual | `2026-06-21T11:01:51Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `DRAMUSDT` | tradfi_perpetual | `2026-06-21T11:01:06Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `RKLBUSDT` | tradfi_perpetual | `2026-06-21T11:01:05Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `CBRSUSDT` | tradfi_perpetual | `2026-06-21T11:01:06Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `SPCXUSDT` | tradfi_perpetual | `2026-06-21T11:01:00Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `OPENAIUSDT` | tradfi_perpetual | `2026-06-21T11:01:06Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `NBISUSDT` | tradfi_perpetual | `2026-06-21T11:01:00Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `WDCUSDT` | tradfi_perpetual | `2026-06-21T11:01:24Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `ARMUSDT` | tradfi_perpetual | `2026-06-21T11:01:03Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `BEUSDT` | tradfi_perpetual | `2026-06-21T11:01:11Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `COHRUSDT` | tradfi_perpetual | `2026-06-21T11:01:54Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `QNTXUSDT` | tradfi_perpetual | `2026-06-21T11:01:31Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `LLYUSDT` | tradfi_perpetual | `2026-06-21T11:02:41Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `NVOUSDT` | tradfi_perpetual | `2026-06-21T11:02:33Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `BBXUSDT` | tradfi_perpetual | `2026-06-21T11:02:24Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `NOKUSDT` | tradfi_perpetual | `2026-06-21T11:01:06Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `EWTUSDT` | tradfi_perpetual | `2026-06-01T13:50:00Z` | `2026-07-02T12:53:30Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `ASTSUSDT` | tradfi_perpetual | `2026-06-21T11:01:59Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `SKHYNIXUSDT` | tradfi_perpetual | `2026-06-21T11:01:00Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `SAMSUNGUSDT` | tradfi_perpetual | `2026-06-21T11:01:06Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `HYUNDAIUSDT` | tradfi_perpetual | `2026-06-21T11:01:37Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `ANTHROPICUSDT` | tradfi_perpetual | `2026-06-21T11:01:25Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `DELLUSDT` | tradfi_perpetual | `2026-06-21T11:01:21Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `IBMUSDT` | tradfi_perpetual | `2026-06-21T11:01:14Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `NOWUSDT` | tradfi_perpetual | `2026-06-21T11:01:04Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `CRMUSDT` | tradfi_perpetual | `2026-06-21T11:11:12Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `IRENUSDT` | tradfi_perpetual | `2026-06-21T11:01:00Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `ONDSUSDT` | tradfi_perpetual | `2026-06-21T11:01:00Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `BXUSDT` | tradfi_perpetual | `2026-06-21T11:43:40Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `HPEUSDT` | tradfi_perpetual | `2026-06-21T11:54:50Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `AMATUSDT` | tradfi_perpetual | `2026-06-21T11:01:10Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `CRWDUSDT` | tradfi_perpetual | `2026-06-21T11:01:00Z` | `2026-07-01T23:59:58Z` | True | False | False | latest_exchangeInfo_status_TRADING_HALT_not_currently_trading, insufficient_train_validation_locked_oos_history |
| `CRDOUSDT` | tradfi_perpetual | `2026-06-21T11:17:00Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `AAOIUSDT` | tradfi_perpetual | `2026-06-21T11:01:00Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `IWMUSDT` | tradfi_perpetual | `2026-06-08T09:30:01Z` | `2026-07-02T12:53:30Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `AXTIUSDT` | tradfi_perpetual | `2026-06-21T11:01:00Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `NFLXUSDT` | tradfi_perpetual | `2026-06-21T11:17:39Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `COSTUSDT` | tradfi_perpetual | `2026-06-21T11:18:09Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `URNMUSDT` | tradfi_perpetual | `2026-06-09T09:10:01Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `HIMSUSDT` | tradfi_perpetual | `2026-06-21T11:05:45Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `EBAYUSDT` | tradfi_perpetual | `2026-06-21T11:17:24Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `ZMUSDT` | tradfi_perpetual | `2026-06-21T11:18:00Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `DKNGUSDT` | tradfi_perpetual | `2026-06-21T11:01:51Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `RIVNUSDT` | tradfi_perpetual | `2026-06-21T11:12:08Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `GMEUSDT` | tradfi_perpetual | `2026-06-21T11:01:06Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `XLEUSDT` | tradfi_perpetual | `2026-06-10T09:20:01Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `EWZUSDT` | tradfi_perpetual | `2026-06-10T09:25:01Z` | `2026-07-02T12:53:30Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `BMNRUSDT` | tradfi_perpetual | `2026-06-21T11:03:26Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `UVXYUSDT` | tradfi_perpetual | `2026-06-11T09:20:09Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `ADBEUSDT` | tradfi_perpetual | `2026-06-21T11:07:47Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `GLWUSDT` | tradfi_perpetual | `2026-06-21T11:01:16Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `STXXUSDT` | tradfi_perpetual | `2026-06-11T09:35:01Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `ASMLUSDT` | tradfi_perpetual | `2026-06-21T11:01:45Z` | `2026-07-01T23:59:58Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `LRCXUSDT` | tradfi_perpetual | `2026-06-27T00:00:10Z` | `2026-07-02T13:31:54Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `KLACUSDT` | tradfi_perpetual | `2026-06-27T00:00:09Z` | `2026-07-02T13:31:54Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `ALABUSDT` | tradfi_perpetual | `2026-06-27T00:05:53Z` | `2026-07-02T13:31:54Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `SMCIUSDT` | tradfi_perpetual | `2026-06-27T00:00:10Z` | `2026-07-02T13:31:54Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `CIENUSDT` | tradfi_perpetual | `2026-06-27T00:00:23Z` | `2026-07-02T13:31:54Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `KORUUSDT` | tradfi_perpetual | `2026-06-27T00:00:00Z` | `2026-07-02T13:31:54Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `SONYUSDT` | tradfi_perpetual | `2026-06-27T00:01:12Z` | `2026-07-02T13:31:54Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `MVLLUSDT` | tradfi_perpetual | `2026-06-29T13:35:03Z` | `2026-07-02T13:31:54Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `TQQQUSDT` | tradfi_perpetual | `2026-06-29T13:40:00Z` | `2026-07-02T13:31:54Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `SQQQUSDT` | tradfi_perpetual | `2026-06-29T13:45:02Z` | `2026-07-02T13:31:54Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `STRCUSDT` | tradfi_perpetual | `2026-07-02T09:15:01Z` | `2026-07-02T13:31:54Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `CATUSDT` | tradfi_perpetual | `2026-07-02T09:20:00Z` | `2026-07-02T13:31:54Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `TXNUSDT` | tradfi_perpetual | `2026-07-02T09:25:00Z` | `2026-07-02T13:31:54Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `FLEXUSDT` | tradfi_perpetual | `2026-07-02T09:30:10Z` | `2026-07-02T13:31:54Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `TERUSDT` | tradfi_perpetual | `2026-07-02T09:35:03Z` | `2026-07-02T13:31:54Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `TTWOUSDT` | tradfi_perpetual | `2026-07-02T09:40:00Z` | `2026-07-02T13:31:54Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `KSTRUSDT` | tradfi_perpetual | `2026-07-02T09:45:02Z` | `2026-07-02T13:31:54Z` | True | False | False | insufficient_train_validation_locked_oos_history |
| `BSPUSDT` | tradfi_perpetual | `2026-07-02T09:50:06Z` | `2026-07-02T13:31:54Z` | True | False | False | insufficient_train_validation_locked_oos_history |
