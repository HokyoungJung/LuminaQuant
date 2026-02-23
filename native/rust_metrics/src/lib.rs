use std::slice;

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

#[no_mangle]
pub extern "C" fn evaluate_metrics(
    total_series: *const f64,
    n: i32,
    annual_periods: i32,
    out_sharpe: *mut f64,
    out_cagr: *mut f64,
    out_max_dd: *mut f64,
) -> i32 {
    if total_series.is_null()
        || out_sharpe.is_null()
        || out_cagr.is_null()
        || out_max_dd.is_null()
        || n <= 0
    {
        return 2;
    }

    let bars = unsafe { slice::from_raw_parts(total_series, n as usize) };
    if bars.len() < 2 {
        unsafe {
            *out_sharpe = -999.0;
            *out_cagr = 0.0;
            *out_max_dd = 0.0;
        }
        return 0;
    }

    let periods = if annual_periods <= 0 { 252 } else { annual_periods } as f64;

    let mut mean_r = 0.0;
    for pair in bars.windows(2) {
        let prev = pair[0];
        let next = pair[1];
        let den = if prev == 0.0 { 1.0 } else { prev };
        mean_r += (next - prev) / den;
    }
    mean_r /= (bars.len() - 1) as f64;

    let mut var_r = 0.0;
    for pair in bars.windows(2) {
        let prev = pair[0];
        let next = pair[1];
        let den = if prev == 0.0 { 1.0 } else { prev };
        let ret = (next - prev) / den;
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
    let final_value = bars[bars.len() - 1];
    let cagr = if initial <= 0.0 {
        0.0
    } else {
        let years = (bars.len() as f64) / periods;
        if years <= 0.0 {
            0.0
        } else {
            (final_value / initial).powf(1.0 / years) - 1.0
        }
    };

    let max_dd = max_drawdown(bars);
    unsafe {
        *out_sharpe = sharpe;
        *out_cagr = cagr;
        *out_max_dd = max_dd;
    }
    0
}
