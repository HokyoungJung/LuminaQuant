use std::{env, fs};

use rust_backtest_kernel::{parse_csv, run_backtest};

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        return Err(
            "usage: rust_backtest_kernel <csv> <fast_window> <slow_window> <initial_cash> <fee_bps>"
                .to_string(),
        );
    }
    let text = fs::read_to_string(&args[1]).map_err(|err| err.to_string())?;
    let bars = parse_csv(&text)?;
    let result = run_backtest(
        &bars,
        args[2].parse::<usize>().map_err(|err| err.to_string())?,
        args[3].parse::<usize>().map_err(|err| err.to_string())?,
        args[4].parse::<f64>().map_err(|err| err.to_string())?,
        args[5].parse::<f64>().map_err(|err| err.to_string())?,
    )?;
    println!("{}", result.to_json());
    Ok(())
}
