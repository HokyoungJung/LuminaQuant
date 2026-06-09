use std::slice;

fn np_maximum_price(value: f64) -> f64 {
    if value.is_nan() {
        f64::NAN
    } else if value > 1e-12 {
        value
    } else {
        1e-12
    }
}

#[no_mangle]
pub extern "C" fn simulate_symbol_fold_native(
    close_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    signal_ptr: *const f64,
    len: usize,
    integer_leverage: i32,
    allocation_fraction: f64,
    round_trip_cost_bps: f64,
    returns_ptr: *mut f64,
    liquidation_ptr: *mut u8,
    account_wipeout_ptr: *mut u8,
) -> i32 {
    if close_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || signal_ptr.is_null()
        || returns_ptr.is_null()
        || liquidation_ptr.is_null()
        || account_wipeout_ptr.is_null()
    {
        return 2;
    }
    if len == 0 {
        return 0;
    }

    let close = unsafe { slice::from_raw_parts(close_ptr, len) };
    let high = unsafe { slice::from_raw_parts(high_ptr, len) };
    let low = unsafe { slice::from_raw_parts(low_ptr, len) };
    let signal = unsafe { slice::from_raw_parts(signal_ptr, len) };
    let returns = unsafe { slice::from_raw_parts_mut(returns_ptr, len) };
    let liquidation = unsafe { slice::from_raw_parts_mut(liquidation_ptr, len) };
    let account_wipeout = unsafe { slice::from_raw_parts_mut(account_wipeout_ptr, len) };

    let leverage = integer_leverage as f64;
    let notional = leverage * allocation_fraction;
    let cost_scale = (round_trip_cost_bps / 10000.0) * notional / 2.0;
    let mut prev_signal = 0.0_f64;
    let mut equity = 1.0_f64;

    for idx in 0..len {
        let next_return = if idx + 1 < len {
            (close[idx + 1] - close[idx]) / np_maximum_price(close[idx])
        } else {
            0.0
        };
        let transition = (signal[idx] - prev_signal).abs();
        let value = signal[idx] * notional * next_return - cost_scale * transition;
        returns[idx] = value;
        prev_signal = signal[idx];

        let denom = np_maximum_price(close[idx]);
        let long_liq = signal[idx] > 0.0 && ((low[idx] / denom) - 1.0) * leverage <= -0.95;
        let short_liq = signal[idx] < 0.0 && ((high[idx] / denom) - 1.0) * leverage >= 0.95;
        liquidation[idx] = u8::from(long_liq || short_liq);

        equity *= 1.0 + value;
        account_wipeout[idx] = u8::from(equity <= 0.0);
    }

    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulates_entry_cost_and_next_bar_return() {
        let close = [100.0_f64, 101.0, 100.0];
        let high = [101.0_f64, 102.0, 101.0];
        let low = [99.0_f64, 100.0, 99.0];
        let signal = [1.0_f64, 1.0, 0.0];
        let mut returns = [0.0_f64; 3];
        let mut liq = [0_u8; 3];
        let mut wipe = [0_u8; 3];
        let status = simulate_symbol_fold_native(
            close.as_ptr(),
            high.as_ptr(),
            low.as_ptr(),
            signal.as_ptr(),
            close.len(),
            2,
            0.1,
            10.0,
            returns.as_mut_ptr(),
            liq.as_mut_ptr(),
            wipe.as_mut_ptr(),
        );
        assert_eq!(status, 0);
        assert!(returns[0] > 0.0);
        assert!(returns[1] < 0.0);
        assert!(returns[2] < 0.0);
        assert_eq!(liq, [0, 0, 0]);
        assert_eq!(wipe, [0, 0, 0]);
    }
}
