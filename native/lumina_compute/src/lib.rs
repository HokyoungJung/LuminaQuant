//! lumina_compute — clean-room pyo3 island for all native compute kernels.
//!
//! Replaces five ctypes loaders:
//!   optimization/native_backend.py      → evaluate_metrics
//!   alpha_zoo/native_alpha_fold_backend.py  → simulate_symbol_fold
//!   alpha_zoo/native_live_signal_backend.py → debounced_state_signal,
//!                                             trailing_state_signal
//!   alpha_zoo/native_hybrid_optuna_backend.py → evaluate_hybrid_optuna_portfolio
//!   data/native_raw_first_backend.py    → aggregate_raw_aggtrades_to_1s
//!   storage/wal/native_backend.py       → append_ohlcv_1s_wal
//!
//! Computation logic is bit-identical to the replaced ctypes crates.
//! Module name: lumina_quant._compute  (set via pyproject.toml module-name)

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use numpy::ndarray::{Array1, Array2};

// ============================================================================
// Metrics helpers  (logic from native/rust_metrics/src/lib.rs)
// ============================================================================

fn max_drawdown(series: &[f64]) -> f64 {
    if series.is_empty() {
        return 0.0;
    }
    let mut peak = series[0];
    let mut max_dd = 0.0;
    for &value in series {
        if value > peak {
            peak = value;
        }
        if peak > 0.0 {
            let dd = (peak - value) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }
    }
    max_dd
}

// ============================================================================
// Alpha-fold helpers  (logic from native/rust_alpha_fold/src/lib.rs)
// ============================================================================

fn np_maximum_price(value: f64) -> f64 {
    if value.is_nan() {
        f64::NAN
    } else if value > 1e-12 {
        value
    } else {
        1e-12
    }
}

// ============================================================================
// Live-signals helpers  (logic from native/rust_live_signals/src/lib.rs)
// ============================================================================

fn side_allows_long(side_mode: i32) -> bool {
    side_mode == 0 || side_mode == 2
}

fn side_allows_short(side_mode: i32) -> bool {
    side_mode == 1 || side_mode == 2
}

fn bool_at(values: &[u8], idx: usize) -> bool {
    values[idx] != 0
}

// ============================================================================
// Hybrid-optuna helpers  (logic from native/rust_hybrid_optuna/src/lib.rs)
// ============================================================================

fn row_value(returns: &[f64], cols: usize, row: usize, col: usize) -> f64 {
    returns[row * cols + col]
}

fn candidate_scores(
    returns: &[f64],
    rows: usize,
    cols: usize,
    end: usize,
    window: usize,
    priors: &[f64],
    prior_ratio: f64,
    out: &mut [f64],
) {
    let end_i = end.min(rows);
    let window_i = window.max(1);
    let start = end_i.saturating_sub(window_i);
    let count = end_i.saturating_sub(start);
    if count == 0 {
        for value in out.iter_mut().take(cols) {
            *value = 0.0;
        }
        return;
    }
    for col in 0..cols {
        let mut sum = 0.0;
        for row in start..end_i {
            sum += row_value(returns, cols, row, col);
        }
        let mean = sum / count as f64;
        let mut std = 0.0;
        if count > 1 {
            let mut var = 0.0;
            for row in start..end_i {
                let diff = row_value(returns, cols, row, col) - mean;
                var += diff * diff;
            }
            std = (var / (count - 1) as f64).sqrt();
        }
        let mut downside_sum = 0.0;
        for row in start..end_i {
            let value = row_value(returns, cols, row, col);
            if value < 0.0 {
                downside_sum += value * value;
            }
        }
        let downside = (downside_sum / count as f64).sqrt();
        let raw_score = mean / (std + downside + 1e-9);
        let score = if raw_score.is_finite() { raw_score } else { 0.0 };
        out[col] = (1.0 - prior_ratio) * score + prior_ratio * priors[col];
    }
}

fn softmax(scores: &[f64], weights: &mut [f64]) {
    if scores.is_empty() {
        return;
    }
    let mut max_value = f64::NEG_INFINITY;
    for &value in scores {
        let clean = if value.is_finite() { value } else { -1e9 };
        let clipped = clean.clamp(-20.0, 20.0);
        if clipped > max_value {
            max_value = clipped;
        }
    }
    let mut total = 0.0;
    for (idx, weight) in weights.iter_mut().enumerate() {
        let clean = if scores[idx].is_finite() { scores[idx] } else { -1e9 };
        let clipped = clean.clamp(-20.0, 20.0);
        let value = (clipped - max_value).exp();
        *weight = value;
        total += value;
    }
    if total <= 0.0 || !total.is_finite() {
        let equal = 1.0 / weights.len() as f64;
        for weight in weights.iter_mut() {
            *weight = equal;
        }
        return;
    }
    for weight in weights.iter_mut() {
        *weight /= total;
    }
}

fn argmax(values: &[f64]) -> usize {
    let mut best_idx = 0usize;
    let mut best_value = f64::NEG_INFINITY;
    for (idx, &value) in values.iter().enumerate() {
        let clean = if value.is_finite() { value } else { f64::NEG_INFINITY };
        if clean > best_value {
            best_value = clean;
            best_idx = idx;
        }
    }
    best_idx
}

fn recent_cross_section_vol(
    returns: &[f64],
    rows: usize,
    cols: usize,
    end: usize,
    window: usize,
) -> f64 {
    let end_i = end.min(rows);
    let start = end_i.saturating_sub(window.max(2));
    let count = end_i.saturating_sub(start);
    if count == 0 || cols == 0 {
        return 0.0;
    }
    let mut row_means = Vec::with_capacity(count);
    let mut mean_sum = 0.0;
    for row in start..end_i {
        let mut row_sum = 0.0;
        for col in 0..cols {
            row_sum += row_value(returns, cols, row, col);
        }
        let row_mean = row_sum / cols as f64;
        row_means.push(row_mean);
        mean_sum += row_mean;
    }
    let mean = mean_sum / count as f64;
    let mut var = 0.0;
    for value in row_means {
        let diff = value - mean;
        var += diff * diff;
    }
    (var / count as f64).sqrt()
}

fn mean_range_column(
    returns: &[f64],
    rows: usize,
    cols: usize,
    start: usize,
    end: usize,
    col: usize,
) -> f64 {
    let start_i = start.min(rows);
    let end_i = end.min(rows);
    if end_i <= start_i || col >= cols {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for row in start_i..end_i {
        sum += row_value(returns, cols, row, col);
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

fn mean_abs_tail(values: &[f64], end_count: usize, window: usize) -> f64 {
    if end_count == 0 || window == 0 {
        return 0.0;
    }
    let start = end_count.saturating_sub(window);
    let mut sum = 0.0;
    let mut count = 0usize;
    for &value in &values[start..end_count] {
        sum += value.abs();
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

fn mean_tail(values: &[f64], end_count: usize, window: usize) -> f64 {
    if end_count == 0 || window == 0 {
        return 0.0;
    }
    let start = end_count.saturating_sub(window);
    let mut sum = 0.0;
    let mut count = 0usize;
    for &value in &values[start..end_count] {
        sum += value;
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

// ============================================================================
// Rawfirst helpers  (logic from native/rust_rawfirst/src/lib.rs)
// ============================================================================

#[derive(Clone, Copy)]
struct Bucket {
    bucket_ms: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

fn latest_complete_bucket_start_ms(complete_through_ms: i64) -> Option<i64> {
    let bucket_ms = 1_000_i64;
    if complete_through_ms < bucket_ms - 1 {
        return None;
    }
    Some((((complete_through_ms + 1) / bucket_ms) * bucket_ms) - bucket_ms)
}

fn crc32(payload: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in payload {
        crc ^= byte as u32;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

fn encode_wal_record(
    ts_ms: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
) -> Option<[u8; 64]> {
    if ts_ms % 1_000 != 0 {
        return None;
    }
    let mut body = Vec::with_capacity(56);
    body.push(1u8);
    body.push(0u8);
    body.extend_from_slice(&0u16.to_le_bytes());
    body.extend_from_slice(&64u32.to_le_bytes());
    body.extend_from_slice(&ts_ms.to_le_bytes());
    body.extend_from_slice(&open.to_le_bytes());
    body.extend_from_slice(&high.to_le_bytes());
    body.extend_from_slice(&low.to_le_bytes());
    body.extend_from_slice(&close.to_le_bytes());
    body.extend_from_slice(&volume.to_le_bytes());
    let crc = crc32(&body);
    let mut record = [0u8; 64];
    record[0..4].copy_from_slice(b"LQWB");
    record[4..60].copy_from_slice(&body);
    record[60..64].copy_from_slice(&crc.to_le_bytes());
    Some(record)
}

fn aggregate_present_buckets(
    timestamps_ms: &[i64],
    prices: &[f64],
    quantities: &[f64],
    start_filter_ms: Option<i64>,
    end_filter_ms: Option<i64>,
    effective_complete_through_ms: i64,
) -> Vec<Bucket> {
    let mut buckets: Vec<Bucket> = Vec::new();
    for idx in 0..timestamps_ms.len() {
        let ts_ms = timestamps_ms[idx];
        if let Some(start_ms) = start_filter_ms {
            if ts_ms < start_ms {
                continue;
            }
        }
        if let Some(end_ms) = end_filter_ms {
            if ts_ms > end_ms {
                continue;
            }
        }
        if ts_ms > effective_complete_through_ms {
            continue;
        }
        let bucket_ms = (ts_ms / 1_000) * 1_000;
        let price = prices[idx];
        let quantity = quantities[idx];
        match buckets.last_mut() {
            Some(last) if last.bucket_ms == bucket_ms => {
                if price > last.high {
                    last.high = price;
                }
                if price < last.low {
                    last.low = price;
                }
                last.close = price;
                last.volume += quantity;
            }
            _ => buckets.push(Bucket {
                bucket_ms,
                open: price,
                high: price,
                low: price,
                close: price,
                volume: quantity,
            }),
        }
    }
    buckets
}

// ============================================================================
// PyO3 functions
// ============================================================================

/// Evaluate Sharpe/CAGR/MaxDD from an equity-curve series.
///
/// Args:
///     total_series: 1-D float64 equity curve
///     annual_periods: trading periods per year (252 for daily; <=0 → 252)
/// Returns:
///     (sharpe, cagr, max_dd)
#[pyfunction]
fn evaluate_metrics(
    total_series: PyReadonlyArray1<f64>,
    annual_periods: i32,
) -> PyResult<(f64, f64, f64)> {
    let bars = total_series
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("total_series must be contiguous: {e}")))?;

    if bars.len() < 2 {
        return Ok((-999.0, 0.0, 0.0));
    }

    let periods = if annual_periods <= 0 { 252 } else { annual_periods } as f64;

    let mut mean_r = 0.0;
    for pair in bars.windows(2) {
        let den = if pair[0] == 0.0 { 1.0 } else { pair[0] };
        mean_r += (pair[1] - pair[0]) / den;
    }
    mean_r /= (bars.len() - 1) as f64;

    let mut var_r = 0.0;
    for pair in bars.windows(2) {
        let den = if pair[0] == 0.0 { 1.0 } else { pair[0] };
        let ret = (pair[1] - pair[0]) / den;
        let d = ret - mean_r;
        var_r += d * d;
    }
    if bars.len() > 2 {
        var_r /= (bars.len() - 2) as f64;
    }
    let std_r = if var_r > 0.0 { var_r.sqrt() } else { 0.0 };
    let sharpe = if std_r > 0.0 {
        (mean_r / std_r) * periods.sqrt()
    } else {
        -999.0
    };

    let initial = bars[0];
    let final_val = bars[bars.len() - 1];
    let cagr = if initial <= 0.0 {
        0.0
    } else {
        let years = (bars.len() as f64) / periods;
        if years <= 0.0 {
            0.0
        } else {
            (final_val / initial).powf(1.0 / years) - 1.0
        }
    };

    let mdd = max_drawdown(bars);
    Ok((sharpe, cagr, mdd))
}

/// Simulate one symbol fold: returns, liquidation flags, wipeout flags.
///
/// All input arrays must be the same length and C-contiguous float64.
/// Returns (returns_f64, liquidation_u8, wipeout_u8).
#[pyfunction]
fn simulate_symbol_fold<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    signal: PyReadonlyArray1<'py, f64>,
    integer_leverage: i32,
    allocation_fraction: f64,
    round_trip_cost_bps: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u8>>,
)> {
    let close_s = close
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("close must be contiguous: {e}")))?;
    let high_s = high
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("high must be contiguous: {e}")))?;
    let low_s = low
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("low must be contiguous: {e}")))?;
    let signal_s = signal
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("signal must be contiguous: {e}")))?;

    let len = close_s.len();
    if high_s.len() != len || low_s.len() != len || signal_s.len() != len {
        return Err(PyValueError::new_err(
            "close, high, low, signal must all have the same length",
        ));
    }

    let mut returns = vec![0.0f64; len];
    let mut liquidation = vec![0u8; len];
    let mut account_wipeout = vec![0u8; len];

    if len == 0 {
        return Ok((
            Array1::from(returns).into_pyarray(py),
            Array1::from(liquidation).into_pyarray(py),
            Array1::from(account_wipeout).into_pyarray(py),
        ));
    }

    let leverage = integer_leverage as f64;
    let notional = leverage * allocation_fraction;
    let cost_scale = (round_trip_cost_bps / 10000.0) * notional / 2.0;
    let mut prev_signal = 0.0_f64;
    let mut equity = 1.0_f64;

    for idx in 0..len {
        let next_return = if idx + 1 < len {
            (close_s[idx + 1] - close_s[idx]) / np_maximum_price(close_s[idx])
        } else {
            0.0
        };
        let transition = (signal_s[idx] - prev_signal).abs();
        let value = signal_s[idx] * notional * next_return - cost_scale * transition;
        returns[idx] = value;
        prev_signal = signal_s[idx];

        let denom = np_maximum_price(close_s[idx]);
        let long_liq =
            signal_s[idx] > 0.0 && ((low_s[idx] / denom) - 1.0) * leverage <= -0.95;
        let short_liq =
            signal_s[idx] < 0.0 && ((high_s[idx] / denom) - 1.0) * leverage >= 0.95;
        liquidation[idx] = u8::from(long_liq || short_liq);

        equity *= 1.0 + value;
        account_wipeout[idx] = u8::from(equity <= 0.0);
    }

    Ok((
        Array1::from(returns).into_pyarray(py),
        Array1::from(liquidation).into_pyarray(py),
        Array1::from(account_wipeout).into_pyarray(py),
    ))
}

/// Debounced state-signal kernel.
///
/// signal arrays must be uint8 (0/1), length `array_len`.
/// side_mode: 0=long_only, 1=short_only, 2=long_short
/// Returns float64 signal array (-1.0, 0.0, 1.0).
#[pyfunction]
fn debounced_state_signal<'py>(
    py: Python<'py>,
    long_entry: PyReadonlyArray1<'py, u8>,
    long_exit: PyReadonlyArray1<'py, u8>,
    short_entry: PyReadonlyArray1<'py, u8>,
    short_exit: PyReadonlyArray1<'py, u8>,
    array_len: i32,
    side_mode: i32,
    min_hold_bars: i32,
    cooldown_bars: i32,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if array_len < 0 {
        return Err(PyValueError::new_err("array_len must be >= 0"));
    }
    let len_usize = array_len as usize;
    let le = long_entry
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("long_entry must be contiguous: {e}")))?;
    let lx = long_exit
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("long_exit must be contiguous: {e}")))?;
    let se = short_entry
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("short_entry must be contiguous: {e}")))?;
    let sx = short_exit
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("short_exit must be contiguous: {e}")))?;

    let mut out = vec![0.0f64; len_usize];
    if len_usize == 0 {
        return Ok(Array1::from(out).into_pyarray(py));
    }

    let min_hold = min_hold_bars;
    let cooldown_config = cooldown_bars;
    let mut state = 0.0_f64;
    let mut bars_held = 1_000_000_000_i32;
    let mut cooldown_remaining = 0_i32;

    for idx in 0..len_usize {
        let can_exit = bars_held >= min_hold;
        let mut exited = false;
        let long_exit_now = state > 0.0 && bool_at(lx, idx);
        let short_exit_now = state < 0.0 && bool_at(sx, idx);
        if can_exit && (long_exit_now || short_exit_now) {
            state = 0.0;
            bars_held = 0;
            cooldown_remaining = cooldown_config;
            exited = true;
        }
        if state == 0.0 {
            if cooldown_remaining > 0 {
                cooldown_remaining -= 1;
            } else if !exited {
                if side_allows_long(side_mode) && bool_at(le, idx) {
                    state = 1.0;
                    bars_held = 0;
                } else if side_allows_short(side_mode) && bool_at(se, idx) {
                    state = -1.0;
                    bars_held = 0;
                }
            }
        }
        out[idx] = state;
        if state != 0.0 {
            bars_held += 1;
        }
    }

    Ok(Array1::from(out).into_pyarray(py))
}

/// Trailing-stop state-signal kernel.
///
/// close, atr are float64; signal arrays are uint8 (0/1).
/// side_mode: 0=long_only, 1=short_only, 2=long_short
/// Returns float64 signal array (-1.0, 0.0, 1.0).
#[pyfunction]
fn trailing_state_signal<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    long_entry: PyReadonlyArray1<'py, u8>,
    short_entry: PyReadonlyArray1<'py, u8>,
    long_exit: PyReadonlyArray1<'py, u8>,
    short_exit: PyReadonlyArray1<'py, u8>,
    atr: PyReadonlyArray1<'py, f64>,
    array_len: i32,
    side_mode: i32,
    min_hold_bars: i32,
    cooldown_bars: i32,
    trail_atr_mult: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if array_len < 0 {
        return Err(PyValueError::new_err("array_len must be >= 0"));
    }
    let len_usize = array_len as usize;
    let close_s = close
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("close must be contiguous: {e}")))?;
    let le = long_entry
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("long_entry must be contiguous: {e}")))?;
    let se = short_entry
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("short_entry must be contiguous: {e}")))?;
    let lx = long_exit
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("long_exit must be contiguous: {e}")))?;
    let sx = short_exit
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("short_exit must be contiguous: {e}")))?;
    let atr_s = atr
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("atr must be contiguous: {e}")))?;

    let mut out = vec![0.0f64; len_usize];
    if len_usize == 0 {
        return Ok(Array1::from(out).into_pyarray(py));
    }

    let min_hold = min_hold_bars;
    let cooldown_config = cooldown_bars;
    let mut state = 0.0_f64;
    let mut stop = f64::NAN;
    let mut bars_held = 1_000_000_000_i32;
    let mut cooldown = 0_i32;

    for idx in 0..len_usize {
        let price = close_s[idx];
        let atr_value = atr_s[idx];
        if !price.is_finite() || !atr_value.is_finite() || atr_value <= 0.0 {
            out[idx] = state;
            if state != 0.0 {
                bars_held += 1;
            }
            continue;
        }

        let can_exit = bars_held >= min_hold;
        let mut exited = false;
        if state > 0.0 {
            let next_stop = price - trail_atr_mult * atr_value;
            stop = if stop.is_finite() { stop.max(next_stop) } else { next_stop };
            if can_exit && (bool_at(lx, idx) || price < stop) {
                state = 0.0;
                stop = f64::NAN;
                bars_held = 0;
                cooldown = cooldown_config;
                exited = true;
            }
        } else if state < 0.0 {
            let next_stop = price + trail_atr_mult * atr_value;
            stop = if stop.is_finite() { stop.min(next_stop) } else { next_stop };
            if can_exit && (bool_at(sx, idx) || price > stop) {
                state = 0.0;
                stop = f64::NAN;
                bars_held = 0;
                cooldown = cooldown_config;
                exited = true;
            }
        }

        if state == 0.0 {
            if cooldown > 0 {
                cooldown -= 1;
            } else if !exited {
                if side_allows_long(side_mode) && bool_at(le, idx) {
                    state = 1.0;
                    stop = price - trail_atr_mult * atr_value;
                    bars_held = 0;
                } else if side_allows_short(side_mode) && bool_at(se, idx) {
                    state = -1.0;
                    stop = price + trail_atr_mult * atr_value;
                    bars_held = 0;
                }
            }
        }

        out[idx] = state;
        if state != 0.0 {
            bars_held += 1;
        }
    }

    Ok(Array1::from(out).into_pyarray(py))
}

/// Hybrid Optuna portfolio evaluation.
///
/// returns_matrix: 2D float64 array (rows × cols), C-contiguous.
/// Returns (portfolio, exposed_weights, raw_weights, default_idx, high_vol_feature, exposure).
#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn evaluate_hybrid_optuna_portfolio<'py>(
    py: Python<'py>,
    returns_matrix: PyReadonlyArray2<'py, f64>,
    version_code: i32,
    start_idx: usize,
    mape_window: usize,
    bias_window: usize,
    short_vol_window: usize,
    bias_correction_alpha: f64,
    bias_combine_ratio: f64,
    max_single_weight: f64,
    initial_default_idx: usize,
    high_vol_idx: usize,
    default_weight_ratio: f64,
    high_vol_threshold: f64,
    high_vol_weight_boost: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let shape = returns_matrix.shape();
    let rows = shape[0];
    let cols = shape[1];

    if rows == 0 || cols == 0 {
        return Err(PyValueError::new_err("returns_matrix must be non-empty"));
    }
    if initial_default_idx >= cols {
        return Err(PyValueError::new_err(
            "initial_default_idx out of bounds for given cols",
        ));
    }
    let returns = returns_matrix
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("returns_matrix must be C-contiguous: {e}")))?;

    let mut out_returns = vec![0.0f64; rows];
    let mut exposed_weights = vec![0.0f64; rows * cols];
    let mut raw_weights = vec![0.0f64; rows * cols];
    let mut default_idx_out = vec![-1i64; rows];
    let mut high_vol_feature_out = vec![0.0f64; rows];
    let mut exposure_out = vec![0.0f64; rows];

    let mape_window_i = mape_window.max(2);
    let bias_window_i = bias_window.max(1);
    let short_vol_window_i = short_vol_window.max(2);
    let start_idx_i = start_idx.min(rows);

    let mut prior_scores = vec![0.0; cols];
    let zero_priors = vec![0.0; cols];
    candidate_scores(
        returns,
        rows,
        cols,
        start_idx_i.max(1),
        mape_window_i,
        &zero_priors,
        0.0,
        &mut prior_scores,
    );

    let mut default_idx = initial_default_idx;
    let high_vol_idx_i = high_vol_idx;
    let mut rolling_scores = vec![0.0; cols];
    let mut score_weights = vec![0.0; cols];
    let mut weights = vec![0.0; cols];
    let mut history: Vec<f64> = Vec::with_capacity(rows.saturating_sub(start_idx_i));

    for t in start_idx_i..rows {
        candidate_scores(
            returns,
            rows,
            cols,
            t,
            mape_window_i,
            &prior_scores,
            bias_combine_ratio.clamp(0.0, 1.0),
            &mut rolling_scores,
        );
        let enough = t >= mape_window_i;
        if version_code == 36 && enough {
            default_idx = argmax(&rolling_scores);
        }

        softmax(&rolling_scores, &mut score_weights);
        for col in 0..cols {
            let base = if col == default_idx { 1.0 } else { 0.0 };
            weights[col] =
                default_weight_ratio * base + (1.0 - default_weight_ratio) * score_weights[col];
        }

        let high_vol_feature =
            recent_cross_section_vol(returns, rows, cols, t, short_vol_window_i);
        if high_vol_feature > high_vol_threshold && high_vol_idx_i < cols {
            weights[high_vol_idx_i] += high_vol_weight_boost;
        }

        let mut weight_sum = 0.0;
        for weight in weights.iter_mut() {
            if *weight < 0.0 || !weight.is_finite() {
                *weight = 0.0;
            }
            weight_sum += *weight;
        }
        if weight_sum <= 0.0 {
            let equal = 1.0 / cols as f64;
            for weight in weights.iter_mut() {
                *weight = equal;
            }
        } else {
            for weight in weights.iter_mut() {
                *weight /= weight_sum;
            }
        }

        let cap = (1.0_f64 / cols as f64).max(max_single_weight.min(0.99));
        for _ in 0..3 {
            let mut over = vec![false; cols];
            let mut excess = 0.0;
            for col in 0..cols {
                if weights[col] > cap {
                    over[col] = true;
                    excess += weights[col] - cap;
                    weights[col] = cap;
                }
            }
            if !over.iter().any(|v| *v) {
                break;
            }
            if excess > 0.0 {
                let mut under_sum = 0.0;
                for col in 0..cols {
                    if !over[col] {
                        under_sum += weights[col];
                    }
                }
                if under_sum > 0.0 {
                    for col in 0..cols {
                        if !over[col] {
                            weights[col] += excess * weights[col] / under_sum;
                        }
                    }
                }
            }
        }
        let final_weight_sum: f64 = weights.iter().sum();
        let denom = final_weight_sum.max(1e-12);
        for weight in weights.iter_mut() {
            *weight /= denom;
        }

        let mut exposure = 1.0;
        let hist_count = history.len();
        if hist_count >= bias_window_i {
            let ens_bias = mean_tail(&history, hist_count, bias_window_i);
            let model_start = t.saturating_sub(bias_window_i);
            let model_bias =
                mean_range_column(returns, rows, cols, model_start, t, default_idx);
            let combined_bias =
                bias_combine_ratio * model_bias + (1.0 - bias_combine_ratio) * ens_bias;
            let denom_abs = mean_abs_tail(&history, hist_count, bias_window_i) + 1e-9;
            if combined_bias < 0.0 {
                let reduction =
                    bias_correction_alpha * (combined_bias.abs() / denom_abs).min(0.80);
                exposure = (1.0 - reduction).max(0.0);
            }
        }

        let mut ret = 0.0;
        for col in 0..cols {
            ret += weights[col] * row_value(returns, cols, t, col);
        }
        ret *= exposure;
        if !ret.is_finite() {
            ret = 0.0;
        }
        out_returns[t] = ret;
        history.push(ret);
        default_idx_out[t] = default_idx as i64;
        high_vol_feature_out[t] = high_vol_feature;
        exposure_out[t] = exposure;
        for col in 0..cols {
            raw_weights[t * cols + col] = weights[col];
            exposed_weights[t * cols + col] = weights[col] * exposure;
        }
    }

    let ew_arr = Array2::from_shape_vec((rows, cols), exposed_weights)
        .map_err(|e| PyValueError::new_err(format!("exposed_weights reshape failed: {e}")))?;
    let rw_arr = Array2::from_shape_vec((rows, cols), raw_weights)
        .map_err(|e| PyValueError::new_err(format!("raw_weights reshape failed: {e}")))?;

    Ok((
        Array1::from(out_returns).into_pyarray(py),
        ew_arr.into_pyarray(py),
        rw_arr.into_pyarray(py),
        Array1::from(default_idx_out).into_pyarray(py),
        Array1::from(high_vol_feature_out).into_pyarray(py),
        Array1::from(exposure_out).into_pyarray(py),
    ))
}

/// Aggregate raw aggTrades to 1-second OHLCV bars.
///
/// Returns (timestamps_ms, open, high, low, close, volume) as 1D arrays,
/// truncated to the actual output length.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn aggregate_raw_aggtrades_to_1s<'py>(
    py: Python<'py>,
    timestamps_ms: PyReadonlyArray1<'py, i64>,
    prices: PyReadonlyArray1<'py, f64>,
    quantities: PyReadonlyArray1<'py, f64>,
    range_start_ms: i64,
    has_range_start_ms: i32,
    range_end_ms: i64,
    has_range_end_ms: i32,
    previous_close: f64,
    has_previous_close: i32,
    complete_through_ms: i64,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let ts = timestamps_ms
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("timestamps_ms must be contiguous: {e}")))?;
    let pr = prices
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("prices must be contiguous: {e}")))?;
    let qt = quantities
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("quantities must be contiguous: {e}")))?;

    let empty = || {
        Ok((
            Array1::<i64>::from(vec![]).into_pyarray(py),
            Array1::<f64>::from(vec![]).into_pyarray(py),
            Array1::<f64>::from(vec![]).into_pyarray(py),
            Array1::<f64>::from(vec![]).into_pyarray(py),
            Array1::<f64>::from(vec![]).into_pyarray(py),
            Array1::<f64>::from(vec![]).into_pyarray(py),
        ))
    };

    if ts.is_empty() {
        return empty();
    }

    let last_complete_second = match latest_complete_bucket_start_ms(complete_through_ms) {
        Some(v) => v,
        None => return empty(),
    };

    let first_trade_second = (ts[0] / 1_000) * 1_000;
    let mut start_second = if has_range_start_ms != 0 {
        let aligned = (range_start_ms / 1_000) * 1_000;
        if has_previous_close == 0 {
            aligned.max(first_trade_second)
        } else {
            aligned
        }
    } else {
        first_trade_second
    };

    let mut end_second = last_complete_second;
    if has_range_end_ms != 0 {
        end_second = end_second.min((range_end_ms / 1_000) * 1_000);
    }
    if end_second < start_second {
        return empty();
    }

    let buckets = aggregate_present_buckets(
        ts,
        pr,
        qt,
        if has_range_start_ms != 0 { Some(range_start_ms) } else { None },
        if has_range_end_ms != 0 { Some(range_end_ms) } else { None },
        complete_through_ms,
    );
    if buckets.is_empty() {
        return empty();
    }

    if has_previous_close == 0 {
        start_second = start_second.max(buckets[0].bucket_ms);
    }
    if end_second < start_second {
        return empty();
    }

    let expected_len = ((end_second - start_second) / 1_000 + 1).max(0) as usize;
    let mut out_ts: Vec<i64> = Vec::with_capacity(expected_len);
    let mut out_open: Vec<f64> = Vec::with_capacity(expected_len);
    let mut out_high: Vec<f64> = Vec::with_capacity(expected_len);
    let mut out_low: Vec<f64> = Vec::with_capacity(expected_len);
    let mut out_close: Vec<f64> = Vec::with_capacity(expected_len);
    let mut out_volume: Vec<f64> = Vec::with_capacity(expected_len);

    let mut bucket_idx = 0usize;
    let mut carry_close: Option<f64> = if has_previous_close != 0 {
        Some(previous_close)
    } else {
        None
    };

    let mut second = start_second;
    while second <= end_second {
        if bucket_idx < buckets.len() && buckets[bucket_idx].bucket_ms == second {
            let b = buckets[bucket_idx];
            out_ts.push(second);
            out_open.push(b.open);
            out_high.push(b.high);
            out_low.push(b.low);
            out_close.push(b.close);
            out_volume.push(b.volume);
            carry_close = Some(b.close);
            bucket_idx += 1;
        } else if let Some(c) = carry_close {
            out_ts.push(second);
            out_open.push(c);
            out_high.push(c);
            out_low.push(c);
            out_close.push(c);
            out_volume.push(0.0);
        }
        second += 1_000;
    }

    Ok((
        Array1::from(out_ts).into_pyarray(py),
        Array1::from(out_open).into_pyarray(py),
        Array1::from(out_high).into_pyarray(py),
        Array1::from(out_low).into_pyarray(py),
        Array1::from(out_close).into_pyarray(py),
        Array1::from(out_volume).into_pyarray(py),
    ))
}

/// Append 1s OHLCV records to a WAL file.
///
/// Returns the number of records written, or raises an exception on I/O error.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn append_ohlcv_1s_wal<'py>(
    wal_path: &str,
    timestamps_ms: PyReadonlyArray1<'py, i64>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    fsync_after_write: bool,
) -> PyResult<i32> {
    use std::fs::OpenOptions;
    use std::io::Write;

    let ts = timestamps_ms
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("timestamps_ms must be contiguous: {e}")))?;
    let op = open
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("open must be contiguous: {e}")))?;
    let hi = high
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("high must be contiguous: {e}")))?;
    let lo = low
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("low must be contiguous: {e}")))?;
    let cl = close
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("close must be contiguous: {e}")))?;
    let vo = volume
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("volume must be contiguous: {e}")))?;

    let len = ts.len();
    if len == 0 {
        return Ok(0);
    }

    let mut encoded = Vec::with_capacity(len * 64);
    for idx in 0..len {
        let record = encode_wal_record(ts[idx], op[idx], hi[idx], lo[idx], cl[idx], vo[idx])
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "timestamp {} is not 1s-aligned (must be divisible by 1000ms)",
                    ts[idx]
                ))
            })?;
        encoded.extend_from_slice(&record);
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(wal_path)
        .map_err(|e| PyValueError::new_err(format!("failed to open WAL file {wal_path}: {e}")))?;

    file.write_all(&encoded)
        .map_err(|e| PyValueError::new_err(format!("WAL write failed: {e}")))?;
    file.flush()
        .map_err(|e| PyValueError::new_err(format!("WAL flush failed: {e}")))?;
    if fsync_after_write {
        file.sync_data()
            .map_err(|e| PyValueError::new_err(format!("WAL fsync failed: {e}")))?;
    }

    Ok(len as i32)
}

// ============================================================================
// Module registration
// ============================================================================

#[pymodule]
fn _compute(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_symbol_fold, m)?)?;
    m.add_function(wrap_pyfunction!(debounced_state_signal, m)?)?;
    m.add_function(wrap_pyfunction!(trailing_state_signal, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_hybrid_optuna_portfolio, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_raw_aggtrades_to_1s, m)?)?;
    m.add_function(wrap_pyfunction!(append_ohlcv_1s_wal, m)?)?;
    Ok(())
}
