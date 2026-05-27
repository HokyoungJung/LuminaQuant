use std::slice;

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
        let score = if raw_score.is_finite() {
            raw_score
        } else {
            0.0
        };
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
        let clean = if scores[idx].is_finite() {
            scores[idx]
        } else {
            -1e9
        };
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
        let clean = if value.is_finite() {
            value
        } else {
            f64::NEG_INFINITY
        };
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
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
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
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
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
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

#[no_mangle]
pub extern "C" fn evaluate_hybrid_optuna_portfolio(
    returns_ptr: *const f64,
    rows: usize,
    cols: usize,
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
    out_returns_ptr: *mut f64,
    exposed_weights_ptr: *mut f64,
    raw_weights_ptr: *mut f64,
    default_idx_ptr: *mut i64,
    high_vol_feature_ptr: *mut f64,
    exposure_ptr: *mut f64,
) -> i32 {
    if returns_ptr.is_null()
        || out_returns_ptr.is_null()
        || exposed_weights_ptr.is_null()
        || raw_weights_ptr.is_null()
        || default_idx_ptr.is_null()
        || high_vol_feature_ptr.is_null()
        || exposure_ptr.is_null()
        || rows == 0
        || cols == 0
        || initial_default_idx >= cols
    {
        return 2;
    }

    let len = match rows.checked_mul(cols) {
        Some(value) => value,
        None => return 2,
    };
    let returns = unsafe { slice::from_raw_parts(returns_ptr, len) };
    let out_returns = unsafe { slice::from_raw_parts_mut(out_returns_ptr, rows) };
    let exposed_weights = unsafe { slice::from_raw_parts_mut(exposed_weights_ptr, len) };
    let raw_weights = unsafe { slice::from_raw_parts_mut(raw_weights_ptr, len) };
    let default_idx_out = unsafe { slice::from_raw_parts_mut(default_idx_ptr, rows) };
    let high_vol_feature_out = unsafe { slice::from_raw_parts_mut(high_vol_feature_ptr, rows) };
    let exposure_out = unsafe { slice::from_raw_parts_mut(exposure_ptr, rows) };

    for value in out_returns.iter_mut() {
        *value = 0.0;
    }
    for value in exposed_weights.iter_mut() {
        *value = 0.0;
    }
    for value in raw_weights.iter_mut() {
        *value = 0.0;
    }
    for idx in 0..rows {
        default_idx_out[idx] = -1;
        high_vol_feature_out[idx] = 0.0;
        exposure_out[idx] = 0.0;
    }

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

        let high_vol_feature = recent_cross_section_vol(returns, rows, cols, t, short_vol_window_i);
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

        let cap = (1.0 / cols as f64).max(max_single_weight.min(0.99));
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
            if !over.iter().any(|value| *value) {
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
            let model_bias = mean_range_column(returns, rows, cols, model_start, t, default_idx);
            let combined_bias =
                bias_combine_ratio * model_bias + (1.0 - bias_combine_ratio) * ens_bias;
            let denom_abs = mean_abs_tail(&history, hist_count, bias_window_i) + 1e-9;
            if combined_bias < 0.0 {
                let reduction = bias_correction_alpha * (combined_bias.abs() / denom_abs).min(0.80);
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

    0
}
