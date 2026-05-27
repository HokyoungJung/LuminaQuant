use std::slice;

fn side_allows_long(side_mode: i32) -> bool {
    side_mode == 0 || side_mode == 2
}

fn side_allows_short(side_mode: i32) -> bool {
    side_mode == 1 || side_mode == 2
}

fn bool_at(values: &[u8], idx: usize) -> bool {
    values[idx] != 0
}

#[no_mangle]
pub extern "C" fn debounced_state_signal_native(
    long_entry: *const u8,
    long_exit: *const u8,
    short_entry: *const u8,
    short_exit: *const u8,
    len: i32,
    side_mode: i32,
    min_hold_bars: i32,
    cooldown_bars: i32,
    out_signal: *mut f64,
) -> i32 {
    if len < 0 || out_signal.is_null() {
        return 2;
    }
    let len_usize = len as usize;
    if len_usize == 0 {
        return 0;
    }
    if long_entry.is_null() || long_exit.is_null() || short_entry.is_null() || short_exit.is_null()
    {
        return 2;
    }

    let long_entry = unsafe { slice::from_raw_parts(long_entry, len_usize) };
    let long_exit = unsafe { slice::from_raw_parts(long_exit, len_usize) };
    let short_entry = unsafe { slice::from_raw_parts(short_entry, len_usize) };
    let short_exit = unsafe { slice::from_raw_parts(short_exit, len_usize) };
    let out = unsafe { slice::from_raw_parts_mut(out_signal, len_usize) };

    let min_hold = min_hold_bars;
    let cooldown_config = cooldown_bars;
    let mut state = 0.0_f64;
    let mut bars_held = 1_000_000_000_i32;
    let mut cooldown_remaining = 0_i32;

    for idx in 0..len_usize {
        let can_exit = bars_held >= min_hold;
        let mut exited = false;
        let long_exit_now = state > 0.0 && bool_at(long_exit, idx);
        let short_exit_now = state < 0.0 && bool_at(short_exit, idx);
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
                if side_allows_long(side_mode) && bool_at(long_entry, idx) {
                    state = 1.0;
                    bars_held = 0;
                } else if side_allows_short(side_mode) && bool_at(short_entry, idx) {
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

    0
}

#[no_mangle]
pub extern "C" fn trailing_state_signal_native(
    close: *const f64,
    long_entry: *const u8,
    short_entry: *const u8,
    long_exit: *const u8,
    short_exit: *const u8,
    atr: *const f64,
    len: i32,
    side_mode: i32,
    min_hold_bars: i32,
    cooldown_bars: i32,
    trail_atr_mult: f64,
    out_signal: *mut f64,
) -> i32 {
    if len < 0 || out_signal.is_null() {
        return 2;
    }
    let len_usize = len as usize;
    if len_usize == 0 {
        return 0;
    }
    if close.is_null()
        || long_entry.is_null()
        || short_entry.is_null()
        || long_exit.is_null()
        || short_exit.is_null()
        || atr.is_null()
    {
        return 2;
    }

    let close = unsafe { slice::from_raw_parts(close, len_usize) };
    let long_entry = unsafe { slice::from_raw_parts(long_entry, len_usize) };
    let short_entry = unsafe { slice::from_raw_parts(short_entry, len_usize) };
    let long_exit = unsafe { slice::from_raw_parts(long_exit, len_usize) };
    let short_exit = unsafe { slice::from_raw_parts(short_exit, len_usize) };
    let atr = unsafe { slice::from_raw_parts(atr, len_usize) };
    let out = unsafe { slice::from_raw_parts_mut(out_signal, len_usize) };

    let min_hold = min_hold_bars;
    let cooldown_config = cooldown_bars;
    let mut state = 0.0_f64;
    let mut stop = f64::NAN;
    let mut bars_held = 1_000_000_000_i32;
    let mut cooldown = 0_i32;

    for idx in 0..len_usize {
        let price = close[idx];
        let atr_value = atr[idx];
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
            stop = if stop.is_finite() {
                stop.max(next_stop)
            } else {
                next_stop
            };
            if can_exit && (bool_at(long_exit, idx) || price < stop) {
                state = 0.0;
                stop = f64::NAN;
                bars_held = 0;
                cooldown = cooldown_config;
                exited = true;
            }
        } else if state < 0.0 {
            let next_stop = price + trail_atr_mult * atr_value;
            stop = if stop.is_finite() {
                stop.min(next_stop)
            } else {
                next_stop
            };
            if can_exit && (bool_at(short_exit, idx) || price > stop) {
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
                if side_allows_long(side_mode) && bool_at(long_entry, idx) {
                    state = 1.0;
                    stop = price - trail_atr_mult * atr_value;
                    bars_held = 0;
                } else if side_allows_short(side_mode) && bool_at(short_entry, idx) {
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

    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debounced_respects_min_hold_and_cooldown() {
        let long_entry = [1_u8, 0, 1, 1, 0, 1, 0, 0];
        let long_exit = [0_u8, 1, 1, 0, 1, 0, 1, 0];
        let short_entry = [0_u8; 8];
        let short_exit = [0_u8; 8];
        let mut out = [0.0_f64; 8];
        let status = debounced_state_signal_native(
            long_entry.as_ptr(),
            long_exit.as_ptr(),
            short_entry.as_ptr(),
            short_exit.as_ptr(),
            8,
            0,
            2,
            1,
            out.as_mut_ptr(),
        );
        assert_eq!(status, 0);
        assert_eq!(out, [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn trailing_skips_invalid_atr_without_resetting_state() {
        let close = [100.0_f64, 101.0, 102.0, f64::NAN, 99.0];
        let atr = [1.0_f64, 1.0, 1.0, 1.0, 1.0];
        let long_entry = [1_u8, 0, 0, 0, 0];
        let short_entry = [0_u8; 5];
        let long_exit = [0_u8, 0, 0, 0, 1];
        let short_exit = [0_u8; 5];
        let mut out = [0.0_f64; 5];
        let status = trailing_state_signal_native(
            close.as_ptr(),
            long_entry.as_ptr(),
            short_entry.as_ptr(),
            long_exit.as_ptr(),
            short_exit.as_ptr(),
            atr.as_ptr(),
            5,
            0,
            1,
            0,
            2.0,
            out.as_mut_ptr(),
        );
        assert_eq!(status, 0);
        assert_eq!(out, [1.0, 1.0, 1.0, 1.0, 0.0]);
    }
}
