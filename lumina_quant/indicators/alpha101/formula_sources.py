"""Generated Alpha101 code-native programs (no raw formula string table)."""

from __future__ import annotations
# ruff: noqa: I001,UP034,SIM300,RUF022

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True, slots=True)
class AlphaProgramDefinition:
    alpha_id: int
    program: Any
    tunable_constants: dict[str, float]

def alpha_001_program(env, const):
    return ((env["rank"](env["ts_argmax"](env["signed_power"](env["where"](((env["returns"]) < (0)), env["ts_stddev"](env["returns"], const("alpha101.1.const.001", 20.0)), env["close"]), 2.0), const("alpha101.1.const.002", 5.0)))) - (0.5))

def alpha_002_program(env, const):
    return (((-(1))) * (env["ts_corr"](env["rank"](env["delta"](env["log"](env["volume"]), 2)), env["rank"](((((env["close"]) - (env["open"]))) / (env["open"]))), const("alpha101.2.const.001", 6.0))))

def alpha_003_program(env, const):
    return (((-(1))) * (env["ts_corr"](env["rank"](env["open"]), env["rank"](env["volume"]), const("alpha101.3.const.001", 10.0))))

def alpha_004_program(env, const):
    return (((-(1))) * (env["ts_rank"](env["rank"](env["low"]), const("alpha101.4.const.001", 9.0))))

def alpha_005_program(env, const):
    return ((env["rank"](((env["open"]) - (((env["ts_sum"](env["vwap"], const("alpha101.5.const.001", 10.0))) / (const("alpha101.5.const.002", 10.0))))))) * ((((-(1))) * (env["abs"](env["rank"](((env["close"]) - (env["vwap"]))))))))

def alpha_006_program(env, const):
    return (((-(1))) * (env["ts_corr"](env["open"], env["volume"], const("alpha101.6.const.001", 10.0))))

def alpha_007_program(env, const):
    return env["where"](((env["adv20"]) < (env["volume"])), (((((-(1))) * (env["ts_rank"](env["abs"](env["delta"](env["close"], const("alpha101.7.const.001", 7.0))), const("alpha101.7.const.002", 60.0))))) * (env["sign"](env["delta"](env["close"], const("alpha101.7.const.003", 7.0))))), (((-(1))) * (1)))

def alpha_008_program(env, const):
    return (((-(1))) * (env["rank"](((((env["ts_sum"](env["open"], const("alpha101.8.const.001", 5.0))) * (env["ts_sum"](env["returns"], const("alpha101.8.const.002", 5.0))))) - (env["delay"](((env["ts_sum"](env["open"], const("alpha101.8.const.003", 5.0))) * (env["ts_sum"](env["returns"], const("alpha101.8.const.004", 5.0)))), const("alpha101.8.const.005", 10.0)))))))

def alpha_009_program(env, const):
    return env["where"](((0) < (env["ts_min"](env["delta"](env["close"], 1), const("alpha101.9.const.001", 5.0)))), env["delta"](env["close"], 1), env["where"](((env["ts_max"](env["delta"](env["close"], 1), const("alpha101.9.const.002", 5.0))) < (0)), env["delta"](env["close"], 1), (((-(1))) * (env["delta"](env["close"], 1)))))

def alpha_010_program(env, const):
    return env["rank"](env["where"](((0) < (env["ts_min"](env["delta"](env["close"], 1), 4))), env["delta"](env["close"], 1), env["where"](((env["ts_max"](env["delta"](env["close"], 1), 4)) < (0)), env["delta"](env["close"], 1), (((-(1))) * (env["delta"](env["close"], 1))))))

def alpha_011_program(env, const):
    return ((((env["rank"](env["ts_max"](((env["vwap"]) - (env["close"])), const("alpha101.11.const.001", 3.0)))) + (env["rank"](env["ts_min"](((env["vwap"]) - (env["close"])), const("alpha101.11.const.002", 3.0)))))) * (env["rank"](env["delta"](env["volume"], const("alpha101.11.const.003", 3.0)))))

def alpha_012_program(env, const):
    return ((env["sign"](env["delta"](env["volume"], 1))) * ((((-(1))) * (env["delta"](env["close"], 1)))))

def alpha_013_program(env, const):
    return (((-(1))) * (env["rank"](env["ts_cov"](env["rank"](env["close"]), env["rank"](env["volume"]), const("alpha101.13.const.001", 5.0)))))

def alpha_014_program(env, const):
    return (((((-(1))) * (env["rank"](env["delta"](env["returns"], const("alpha101.14.const.001", 3.0)))))) * (env["ts_corr"](env["open"], env["volume"], const("alpha101.14.const.002", 10.0))))

def alpha_015_program(env, const):
    return (((-(1))) * (env["ts_sum"](env["rank"](env["ts_corr"](env["rank"](env["high"]), env["rank"](env["volume"]), const("alpha101.15.const.001", 3.0))), const("alpha101.15.const.002", 3.0))))

def alpha_016_program(env, const):
    return (((-(1))) * (env["rank"](env["ts_cov"](env["rank"](env["high"]), env["rank"](env["volume"]), const("alpha101.16.const.001", 5.0)))))

def alpha_017_program(env, const):
    return (((((((-(1))) * (env["rank"](env["ts_rank"](env["close"], const("alpha101.17.const.001", 10.0)))))) * (env["rank"](env["delta"](env["delta"](env["close"], 1), 1))))) * (env["rank"](env["ts_rank"](((env["volume"]) / (env["adv20"])), const("alpha101.17.const.002", 5.0)))))

def alpha_018_program(env, const):
    return (((-(1))) * (env["rank"](((((env["ts_stddev"](env["abs"](((env["close"]) - (env["open"]))), const("alpha101.18.const.001", 5.0))) + (((env["close"]) - (env["open"]))))) + (env["ts_corr"](env["close"], env["open"], const("alpha101.18.const.002", 10.0)))))))

def alpha_019_program(env, const):
    return (((((-(1))) * (env["sign"](((((env["close"]) - (env["delay"](env["close"], const("alpha101.19.const.001", 7.0))))) + (env["delta"](env["close"], const("alpha101.19.const.002", 7.0)))))))) * (((1) + (env["rank"](((1) + (env["ts_sum"](env["returns"], const("alpha101.19.const.003", 250.0)))))))))

def alpha_020_program(env, const):
    return (((((((-(1))) * (env["rank"](((env["open"]) - (env["delay"](env["high"], 1))))))) * (env["rank"](((env["open"]) - (env["delay"](env["close"], 1))))))) * (env["rank"](((env["open"]) - (env["delay"](env["low"], 1))))))

def alpha_021_program(env, const):
    return env["where"](((((((env["ts_sum"](env["close"], 8)) / (8))) + (env["ts_stddev"](env["close"], 8)))) < (((env["ts_sum"](env["close"], 2)) / (2)))), (((-(1))) * (1)), env["where"](((((env["ts_sum"](env["close"], 2)) / (2))) < (((((env["ts_sum"](env["close"], 8)) / (8))) - (env["ts_stddev"](env["close"], 8))))), 1, env["where"](((((1) < (((env["volume"]) / (env["adv20"]))))) | (((((env["volume"]) / (env["adv20"]))) == (1)))), 1, (((-(1))) * (1)))))

def alpha_022_program(env, const):
    return (((-(1))) * (((env["delta"](env["ts_corr"](env["high"], env["volume"], const("alpha101.22.const.001", 5.0)), const("alpha101.22.const.002", 5.0))) * (env["rank"](env["ts_stddev"](env["close"], const("alpha101.22.const.003", 20.0)))))))

def alpha_023_program(env, const):
    return env["where"](((((env["ts_sum"](env["high"], const("alpha101.23.const.001", 20.0))) / (const("alpha101.23.const.002", 20.0)))) < (env["high"])), (((-(1))) * (env["delta"](env["high"], 2))), 0)

def alpha_024_program(env, const):
    return env["where"](((((((env["delta"](((env["ts_sum"](env["close"], const("alpha101.24.const.001", 100.0))) / (const("alpha101.24.const.002", 100.0))), const("alpha101.24.const.003", 100.0))) / (env["delay"](env["close"], const("alpha101.24.const.004", 100.0))))) < (const("alpha101.24.const.005", 0.05)))) | (((((env["delta"](((env["ts_sum"](env["close"], const("alpha101.24.const.006", 100.0))) / (const("alpha101.24.const.007", 100.0))), const("alpha101.24.const.008", 100.0))) / (env["delay"](env["close"], const("alpha101.24.const.009", 100.0))))) == (const("alpha101.24.const.010", 0.05))))), (((-(1))) * (((env["close"]) - (env["ts_min"](env["close"], const("alpha101.24.const.011", 100.0)))))), (((-(1))) * (env["delta"](env["close"], const("alpha101.24.const.012", 3.0)))))

def alpha_025_program(env, const):
    return env["rank"]((((((((((-(1))) * (env["returns"]))) * (env["adv20"]))) * (env["vwap"]))) * (((env["high"]) - (env["close"])))))

def alpha_026_program(env, const):
    return (((-(1))) * (env["ts_max"](env["ts_corr"](env["ts_rank"](env["volume"], const("alpha101.26.const.001", 5.0)), env["ts_rank"](env["high"], const("alpha101.26.const.002", 5.0)), const("alpha101.26.const.003", 5.0)), const("alpha101.26.const.004", 3.0))))

def alpha_027_program(env, const):
    return env["where"](((0.5) < (env["rank"](((env["ts_sum"](env["ts_corr"](env["rank"](env["volume"]), env["rank"](env["vwap"]), const("alpha101.27.const.001", 6.0)), 2)) / (2.0))))), (((-(1))) * (1)), 1)

def alpha_028_program(env, const):
    return env["scale"](((((env["ts_corr"](env["adv20"], env["low"], const("alpha101.28.const.001", 5.0))) + (((((env["high"]) + (env["low"]))) / (2))))) - (env["close"])))

def alpha_029_program(env, const):
    return ((env["min"](env["ts_product"](env["rank"](env["rank"](env["scale"](env["log"](env["ts_sum"](env["ts_min"](env["rank"](env["rank"]((((-(1))) * (env["rank"](env["delta"](((env["close"]) - (1)), const("alpha101.29.const.001", 5.0))))))), 2), 1))))), 1), const("alpha101.29.const.002", 5.0))) + (env["ts_rank"](env["delay"]((((-(1))) * (env["returns"])), const("alpha101.29.const.003", 6.0)), const("alpha101.29.const.004", 5.0))))

def alpha_030_program(env, const):
    return ((((((1.0) - (env["rank"](((((env["sign"](((env["close"]) - (env["delay"](env["close"], 1))))) + (env["sign"](((env["delay"](env["close"], 1)) - (env["delay"](env["close"], 2))))))) + (env["sign"](((env["delay"](env["close"], 2)) - (env["delay"](env["close"], const("alpha101.30.const.001", 3.0))))))))))) * (env["ts_sum"](env["volume"], const("alpha101.30.const.002", 5.0))))) / (env["ts_sum"](env["volume"], const("alpha101.30.const.003", 20.0))))

def alpha_031_program(env, const):
    return ((((env["rank"](env["rank"](env["rank"](env["decay_linear"]((((-(1))) * (env["rank"](env["rank"](env["delta"](env["close"], const("alpha101.31.const.001", 10.0)))))), const("alpha101.31.const.002", 10.0)))))) + (env["rank"]((((-(1))) * (env["delta"](env["close"], const("alpha101.31.const.003", 3.0)))))))) + (env["sign"](env["scale"](env["ts_corr"](env["adv20"], env["low"], const("alpha101.31.const.004", 12.0))))))

def alpha_032_program(env, const):
    return ((env["scale"](((((env["ts_sum"](env["close"], const("alpha101.32.const.001", 7.0))) / (const("alpha101.32.const.002", 7.0)))) - (env["close"])))) + (((const("alpha101.32.const.003", 20.0)) * (env["scale"](env["ts_corr"](env["vwap"], env["delay"](env["close"], const("alpha101.32.const.004", 5.0)), const("alpha101.32.const.005", 230.0)))))))

def alpha_033_program(env, const):
    return env["rank"]((((-(1))) * (((((1) - (((env["open"]) / (env["close"]))))) ** (1)))))

def alpha_034_program(env, const):
    return env["rank"](((((1) - (env["rank"](((env["ts_stddev"](env["returns"], 2)) / (env["ts_stddev"](env["returns"], const("alpha101.34.const.001", 5.0)))))))) + (((1) - (env["rank"](env["delta"](env["close"], 1)))))))

def alpha_035_program(env, const):
    return ((((env["ts_rank"](env["volume"], 32)) * (((1) - (env["ts_rank"](((((env["close"]) + (env["high"]))) - (env["low"])), 16)))))) * (((1) - (env["ts_rank"](env["returns"], 32)))))

def alpha_036_program(env, const):
    return ((((((((((const("alpha101.36.const.001", 2.21)) * (env["rank"](env["ts_corr"](((env["close"]) - (env["open"])), env["delay"](env["volume"], 1), const("alpha101.36.const.002", 15.0)))))) + (((const("alpha101.36.const.003", 0.7)) * (env["rank"](((env["open"]) - (env["close"])))))))) + (((const("alpha101.36.const.004", 0.73)) * (env["rank"](env["ts_rank"](env["delay"]((((-(1))) * (env["returns"])), const("alpha101.36.const.005", 6.0)), const("alpha101.36.const.006", 5.0)))))))) + (env["rank"](env["abs"](env["ts_corr"](env["vwap"], env["adv20"], const("alpha101.36.const.007", 6.0))))))) + (((const("alpha101.36.const.008", 0.6)) * (env["rank"](((((((env["ts_sum"](env["close"], const("alpha101.36.const.009", 200.0))) / (const("alpha101.36.const.010", 200.0)))) - (env["open"]))) * (((env["close"]) - (env["open"])))))))))

def alpha_037_program(env, const):
    return ((env["rank"](env["ts_corr"](env["delay"](((env["open"]) - (env["close"])), 1), env["close"], const("alpha101.37.const.001", 200.0)))) + (env["rank"](((env["open"]) - (env["close"])))))

def alpha_038_program(env, const):
    return (((((-(1))) * (env["rank"](env["ts_rank"](env["close"], const("alpha101.38.const.001", 10.0)))))) * (env["rank"](((env["close"]) / (env["open"])))))

def alpha_039_program(env, const):
    return (((((-(1))) * (env["rank"](((env["delta"](env["close"], const("alpha101.39.const.001", 7.0))) * (((1) - (env["rank"](env["decay_linear"](((env["volume"]) / (env["adv20"])), const("alpha101.39.const.002", 9.0))))))))))) * (((1) + (env["rank"](env["ts_sum"](env["returns"], const("alpha101.39.const.003", 250.0)))))))

def alpha_040_program(env, const):
    return (((((-(1))) * (env["rank"](env["ts_stddev"](env["high"], const("alpha101.40.const.001", 10.0)))))) * (env["ts_corr"](env["high"], env["volume"], const("alpha101.40.const.002", 10.0))))

def alpha_041_program(env, const):
    return ((((((env["high"]) * (env["low"]))) ** (0.5))) - (env["vwap"]))

def alpha_042_program(env, const):
    return ((env["rank"](((env["vwap"]) - (env["close"])))) / (env["rank"](((env["vwap"]) + (env["close"])))))

def alpha_043_program(env, const):
    return ((env["ts_rank"](((env["volume"]) / (env["adv20"])), const("alpha101.43.const.001", 20.0))) * (env["ts_rank"]((((-(1))) * (env["delta"](env["close"], const("alpha101.43.const.002", 7.0)))), 8)))

def alpha_044_program(env, const):
    return (((-(1))) * (env["ts_corr"](env["high"], env["rank"](env["volume"]), const("alpha101.44.const.001", 5.0))))

def alpha_045_program(env, const):
    return (((-(1))) * (((((env["rank"](((env["ts_sum"](env["delay"](env["close"], const("alpha101.45.const.001", 5.0)), const("alpha101.45.const.002", 20.0))) / (const("alpha101.45.const.003", 20.0))))) * (env["ts_corr"](env["close"], env["volume"], 2)))) * (env["rank"](env["ts_corr"](env["ts_sum"](env["close"], const("alpha101.45.const.004", 5.0)), env["ts_sum"](env["close"], const("alpha101.45.const.005", 20.0)), 2))))))

def alpha_046_program(env, const):
    return env["where"](((0.25) < (((((((env["delay"](env["close"], const("alpha101.46.const.001", 20.0))) - (env["delay"](env["close"], const("alpha101.46.const.002", 10.0))))) / (const("alpha101.46.const.003", 10.0)))) - (((((env["delay"](env["close"], const("alpha101.46.const.004", 10.0))) - (env["close"]))) / (const("alpha101.46.const.005", 10.0))))))), (((-(1))) * (1)), env["where"](((((((((env["delay"](env["close"], const("alpha101.46.const.006", 20.0))) - (env["delay"](env["close"], const("alpha101.46.const.007", 10.0))))) / (const("alpha101.46.const.008", 10.0)))) - (((((env["delay"](env["close"], const("alpha101.46.const.009", 10.0))) - (env["close"]))) / (const("alpha101.46.const.010", 10.0)))))) < (0)), 1, (((((-(1))) * (1))) * (((env["close"]) - (env["delay"](env["close"], 1)))))))

def alpha_047_program(env, const):
    return ((((((((env["rank"](((1) / (env["close"])))) * (env["volume"]))) / (env["adv20"]))) * (((((env["high"]) * (env["rank"](((env["high"]) - (env["close"])))))) / (((env["ts_sum"](env["high"], const("alpha101.47.const.001", 5.0))) / (const("alpha101.47.const.002", 5.0)))))))) - (env["rank"](((env["vwap"]) - (env["delay"](env["vwap"], const("alpha101.47.const.003", 5.0)))))))

def alpha_048_program(env, const):
    return ((env["indneutralize"](((((env["ts_corr"](env["delta"](env["close"], 1), env["delta"](env["delay"](env["close"], 1), 1), const("alpha101.48.const.001", 250.0))) * (env["delta"](env["close"], 1)))) / (env["close"])), env["subindustry"])) / (env["ts_sum"](((((env["delta"](env["close"], 1)) / (env["delay"](env["close"], 1)))) ** (2)), const("alpha101.48.const.002", 250.0))))

def alpha_049_program(env, const):
    return env["where"](((((((((env["delay"](env["close"], const("alpha101.49.const.001", 20.0))) - (env["delay"](env["close"], const("alpha101.49.const.002", 10.0))))) / (const("alpha101.49.const.003", 10.0)))) - (((((env["delay"](env["close"], const("alpha101.49.const.004", 10.0))) - (env["close"]))) / (const("alpha101.49.const.005", 10.0)))))) < ((((-(1))) * (const("alpha101.49.const.006", 0.1))))), 1, (((((-(1))) * (1))) * (((env["close"]) - (env["delay"](env["close"], 1))))))

def alpha_050_program(env, const):
    return (((-(1))) * (env["ts_max"](env["rank"](env["ts_corr"](env["rank"](env["volume"]), env["rank"](env["vwap"]), const("alpha101.50.const.001", 5.0))), const("alpha101.50.const.002", 5.0))))

def alpha_051_program(env, const):
    return env["where"](((((((((env["delay"](env["close"], const("alpha101.51.const.001", 20.0))) - (env["delay"](env["close"], const("alpha101.51.const.002", 10.0))))) / (const("alpha101.51.const.003", 10.0)))) - (((((env["delay"](env["close"], const("alpha101.51.const.004", 10.0))) - (env["close"]))) / (const("alpha101.51.const.005", 10.0)))))) < ((((-(1))) * (const("alpha101.51.const.006", 0.05))))), 1, (((((-(1))) * (1))) * (((env["close"]) - (env["delay"](env["close"], 1))))))

def alpha_052_program(env, const):
    return (((((((((-(1))) * (env["ts_min"](env["low"], const("alpha101.52.const.001", 5.0))))) + (env["delay"](env["ts_min"](env["low"], const("alpha101.52.const.002", 5.0)), const("alpha101.52.const.003", 5.0))))) * (env["rank"](((((env["ts_sum"](env["returns"], const("alpha101.52.const.004", 240.0))) - (env["ts_sum"](env["returns"], const("alpha101.52.const.005", 20.0))))) / (const("alpha101.52.const.006", 220.0))))))) * (env["ts_rank"](env["volume"], const("alpha101.52.const.007", 5.0))))

def alpha_053_program(env, const):
    return (((-(1))) * (env["delta"](((((((env["close"]) - (env["low"]))) - (((env["high"]) - (env["close"]))))) / (((env["close"]) - (env["low"])))), const("alpha101.53.const.001", 9.0))))

def alpha_054_program(env, const):
    return (((((-(1))) * (((((env["low"]) - (env["close"]))) * (((env["open"]) ** (const("alpha101.54.const.001", 5.0)))))))) / (((((env["low"]) - (env["high"]))) * (((env["close"]) ** (const("alpha101.54.const.002", 5.0)))))))

def alpha_055_program(env, const):
    return (((-(1))) * (env["ts_corr"](env["rank"](((((env["close"]) - (env["ts_min"](env["low"], const("alpha101.55.const.001", 12.0))))) / (((env["ts_max"](env["high"], const("alpha101.55.const.002", 12.0))) - (env["ts_min"](env["low"], const("alpha101.55.const.003", 12.0))))))), env["rank"](env["volume"]), const("alpha101.55.const.004", 6.0))))

def alpha_056_program(env, const):
    return ((0) - (((1) * (((env["rank"](((env["ts_sum"](env["returns"], const("alpha101.56.const.001", 10.0))) / (env["ts_sum"](env["ts_sum"](env["returns"], 2), const("alpha101.56.const.002", 3.0)))))) * (env["rank"](((env["returns"]) * (env["cap"])))))))))

def alpha_057_program(env, const):
    return ((0) - (((1) * (((((env["close"]) - (env["vwap"]))) / (env["decay_linear"](env["rank"](env["ts_argmax"](env["close"], const("alpha101.57.const.001", 30.0))), 2)))))))

def alpha_058_program(env, const):
    return (((-(1))) * (env["ts_rank"](env["decay_linear"](env["ts_corr"](env["indneutralize"](env["vwap"], env["sector"]), env["volume"], const("alpha101.58.const.001", 3.92795)), const("alpha101.58.const.002", 7.89291)), const("alpha101.58.const.003", 5.50322))))

def alpha_059_program(env, const):
    return (((-(1))) * (env["ts_rank"](env["decay_linear"](env["ts_corr"](env["indneutralize"](((((env["vwap"]) * (const("alpha101.59.const.001", 0.728317)))) + (((env["vwap"]) * (((1) - (const("alpha101.59.const.002", 0.728317))))))), env["industry"]), env["volume"], const("alpha101.59.const.003", 4.25197)), const("alpha101.59.const.004", 16.2289)), const("alpha101.59.const.005", 8.19648))))

def alpha_060_program(env, const):
    return ((0) - (((1) * (((((2) * (env["scale"](env["rank"](((((((((env["close"]) - (env["low"]))) - (((env["high"]) - (env["close"]))))) / (((env["high"]) - (env["low"]))))) * (env["volume"]))))))) - (env["scale"](env["rank"](env["ts_argmax"](env["close"], const("alpha101.60.const.001", 10.0))))))))))

def alpha_061_program(env, const):
    return ((env["rank"](((env["vwap"]) - (env["ts_min"](env["vwap"], const("alpha101.61.const.001", 16.1219)))))) < (env["rank"](env["ts_corr"](env["vwap"], env["adv180"], const("alpha101.61.const.002", 17.9282)))))

def alpha_062_program(env, const):
    return ((((env["rank"](env["ts_corr"](env["vwap"], env["ts_sum"](env["adv20"], const("alpha101.62.const.001", 22.4101)), const("alpha101.62.const.002", 9.91009)))) < (env["rank"](((((env["rank"](env["open"])) + (env["rank"](env["open"])))) < (((env["rank"](((((env["high"]) + (env["low"]))) / (2)))) + (env["rank"](env["high"]))))))))) * ((-(1))))

def alpha_063_program(env, const):
    return ((((env["rank"](env["decay_linear"](env["delta"](env["indneutralize"](env["close"], env["industry"]), const("alpha101.63.const.001", 2.25164)), const("alpha101.63.const.002", 8.22237)))) - (env["rank"](env["decay_linear"](env["ts_corr"](((((env["vwap"]) * (const("alpha101.63.const.003", 0.318108)))) + (((env["open"]) * (((1) - (const("alpha101.63.const.004", 0.318108))))))), env["ts_sum"](env["adv180"], const("alpha101.63.const.005", 37.2467)), const("alpha101.63.const.006", 13.557)), const("alpha101.63.const.007", 12.2883)))))) * ((-(1))))

def alpha_064_program(env, const):
    return ((((env["rank"](env["ts_corr"](env["ts_sum"](((((env["open"]) * (const("alpha101.64.const.001", 0.178404)))) + (((env["low"]) * (((1) - (const("alpha101.64.const.002", 0.178404))))))), const("alpha101.64.const.003", 12.7054)), env["ts_sum"](env["adv120"], const("alpha101.64.const.004", 12.7054)), const("alpha101.64.const.005", 16.6208)))) < (env["rank"](env["delta"](((((((((env["high"]) + (env["low"]))) / (2))) * (const("alpha101.64.const.006", 0.178404)))) + (((env["vwap"]) * (((1) - (const("alpha101.64.const.007", 0.178404))))))), const("alpha101.64.const.008", 3.69741)))))) * ((-(1))))

def alpha_065_program(env, const):
    return ((((env["rank"](env["ts_corr"](((((env["open"]) * (const("alpha101.65.const.001", 0.00817205)))) + (((env["vwap"]) * (((1) - (const("alpha101.65.const.002", 0.00817205))))))), env["ts_sum"](env["adv60"], const("alpha101.65.const.003", 8.6911)), const("alpha101.65.const.004", 6.40374)))) < (env["rank"](((env["open"]) - (env["ts_min"](env["open"], const("alpha101.65.const.005", 13.635)))))))) * ((-(1))))

def alpha_066_program(env, const):
    return ((((env["rank"](env["decay_linear"](env["delta"](env["vwap"], const("alpha101.66.const.001", 3.51013)), const("alpha101.66.const.002", 7.23052)))) + (env["ts_rank"](env["decay_linear"](((((((((env["low"]) * (const("alpha101.66.const.003", 0.96633)))) + (((env["low"]) * (((1) - (const("alpha101.66.const.004", 0.96633)))))))) - (env["vwap"]))) / (((env["open"]) - (((((env["high"]) + (env["low"]))) / (2)))))), const("alpha101.66.const.005", 11.4157)), const("alpha101.66.const.006", 6.72611))))) * ((-(1))))

def alpha_067_program(env, const):
    return ((((env["rank"](((env["high"]) - (env["ts_min"](env["high"], const("alpha101.67.const.001", 2.14593)))))) ** (env["rank"](env["ts_corr"](env["indneutralize"](env["vwap"], env["sector"]), env["indneutralize"](env["adv20"], env["subindustry"]), const("alpha101.67.const.002", 6.02936)))))) * ((-(1))))

def alpha_068_program(env, const):
    return ((((env["ts_rank"](env["ts_corr"](env["rank"](env["high"]), env["rank"](env["adv15"]), const("alpha101.68.const.001", 8.91644)), const("alpha101.68.const.002", 13.9333))) < (env["rank"](env["delta"](((((env["close"]) * (const("alpha101.68.const.003", 0.518371)))) + (((env["low"]) * (((1) - (const("alpha101.68.const.004", 0.518371))))))), const("alpha101.68.const.005", 1.06157)))))) * ((-(1))))

def alpha_069_program(env, const):
    return ((((env["rank"](env["ts_max"](env["delta"](env["indneutralize"](env["vwap"], env["industry"]), const("alpha101.69.const.001", 2.72412)), const("alpha101.69.const.002", 4.79344)))) ** (env["ts_rank"](env["ts_corr"](((((env["close"]) * (const("alpha101.69.const.003", 0.490655)))) + (((env["vwap"]) * (((1) - (const("alpha101.69.const.004", 0.490655))))))), env["adv20"], const("alpha101.69.const.005", 4.92416)), const("alpha101.69.const.006", 9.0615))))) * ((-(1))))

def alpha_070_program(env, const):
    return ((((env["rank"](env["delta"](env["vwap"], const("alpha101.70.const.001", 1.29456)))) ** (env["ts_rank"](env["ts_corr"](env["indneutralize"](env["close"], env["industry"]), env["adv50"], const("alpha101.70.const.002", 17.8256)), const("alpha101.70.const.003", 17.9171))))) * ((-(1))))

def alpha_071_program(env, const):
    return env["max"](env["ts_rank"](env["decay_linear"](env["ts_corr"](env["ts_rank"](env["close"], const("alpha101.71.const.001", 3.43976)), env["ts_rank"](env["adv180"], const("alpha101.71.const.002", 12.0647)), const("alpha101.71.const.003", 18.0175)), const("alpha101.71.const.004", 4.20501)), const("alpha101.71.const.005", 15.6948)), env["ts_rank"](env["decay_linear"](((env["rank"](((((env["low"]) + (env["open"]))) - (((env["vwap"]) + (env["vwap"])))))) ** (2)), const("alpha101.71.const.006", 16.4662)), const("alpha101.71.const.007", 4.4388)))

def alpha_072_program(env, const):
    return ((env["rank"](env["decay_linear"](env["ts_corr"](((((env["high"]) + (env["low"]))) / (2)), env["adv40"], const("alpha101.72.const.001", 8.93345)), const("alpha101.72.const.002", 10.1519)))) / (env["rank"](env["decay_linear"](env["ts_corr"](env["ts_rank"](env["vwap"], const("alpha101.72.const.003", 3.72469)), env["ts_rank"](env["volume"], const("alpha101.72.const.004", 18.5188)), const("alpha101.72.const.005", 6.86671)), const("alpha101.72.const.006", 2.95011)))))

def alpha_073_program(env, const):
    return ((env["max"](env["rank"](env["decay_linear"](env["delta"](env["vwap"], const("alpha101.73.const.001", 4.72775)), const("alpha101.73.const.002", 2.91864))), env["ts_rank"](env["decay_linear"](((((env["delta"](((((env["open"]) * (const("alpha101.73.const.003", 0.147155)))) + (((env["low"]) * (((1) - (const("alpha101.73.const.004", 0.147155))))))), const("alpha101.73.const.005", 2.03608))) / (((((env["open"]) * (const("alpha101.73.const.006", 0.147155)))) + (((env["low"]) * (((1) - (const("alpha101.73.const.007", 0.147155)))))))))) * ((-(1)))), const("alpha101.73.const.008", 3.33829)), const("alpha101.73.const.009", 16.7411)))) * ((-(1))))

def alpha_074_program(env, const):
    return ((((env["rank"](env["ts_corr"](env["close"], env["ts_sum"](env["adv30"], const("alpha101.74.const.001", 37.4843)), const("alpha101.74.const.002", 15.1365)))) < (env["rank"](env["ts_corr"](env["rank"](((((env["high"]) * (const("alpha101.74.const.003", 0.0261661)))) + (((env["vwap"]) * (((1) - (const("alpha101.74.const.004", 0.0261661)))))))), env["rank"](env["volume"]), const("alpha101.74.const.005", 11.4791)))))) * ((-(1))))

def alpha_075_program(env, const):
    return ((env["rank"](env["ts_corr"](env["vwap"], env["volume"], const("alpha101.75.const.001", 4.24304)))) < (env["rank"](env["ts_corr"](env["rank"](env["low"]), env["rank"](env["adv50"]), const("alpha101.75.const.002", 12.4413)))))

def alpha_076_program(env, const):
    return ((env["max"](env["rank"](env["decay_linear"](env["delta"](env["vwap"], const("alpha101.76.const.001", 1.24383)), const("alpha101.76.const.002", 11.8259))), env["ts_rank"](env["decay_linear"](env["ts_rank"](env["ts_corr"](env["indneutralize"](env["low"], env["sector"]), env["adv81"], const("alpha101.76.const.003", 8.14941)), const("alpha101.76.const.004", 19.569)), const("alpha101.76.const.005", 17.1543)), const("alpha101.76.const.006", 19.383)))) * ((-(1))))

def alpha_077_program(env, const):
    return env["min"](env["rank"](env["decay_linear"](((((((((env["high"]) + (env["low"]))) / (2))) + (env["high"]))) - (((env["vwap"]) + (env["high"])))), const("alpha101.77.const.001", 20.0451))), env["rank"](env["decay_linear"](env["ts_corr"](((((env["high"]) + (env["low"]))) / (2)), env["adv40"], const("alpha101.77.const.002", 3.1614)), const("alpha101.77.const.003", 5.64125))))

def alpha_078_program(env, const):
    return ((env["rank"](env["ts_corr"](env["ts_sum"](((((env["low"]) * (const("alpha101.78.const.001", 0.352233)))) + (((env["vwap"]) * (((1) - (const("alpha101.78.const.002", 0.352233))))))), const("alpha101.78.const.003", 19.7428)), env["ts_sum"](env["adv40"], const("alpha101.78.const.004", 19.7428)), const("alpha101.78.const.005", 6.83313)))) ** (env["rank"](env["ts_corr"](env["rank"](env["vwap"]), env["rank"](env["volume"]), const("alpha101.78.const.006", 5.77492)))))

def alpha_079_program(env, const):
    return ((env["rank"](env["delta"](env["indneutralize"](((((env["close"]) * (const("alpha101.79.const.001", 0.60733)))) + (((env["open"]) * (((1) - (const("alpha101.79.const.002", 0.60733))))))), env["sector"]), const("alpha101.79.const.003", 1.23438)))) < (env["rank"](env["ts_corr"](env["ts_rank"](env["vwap"], const("alpha101.79.const.004", 3.60973)), env["ts_rank"](env["adv150"], const("alpha101.79.const.005", 9.18637)), const("alpha101.79.const.006", 14.6644)))))

def alpha_080_program(env, const):
    return ((((env["rank"](env["sign"](env["delta"](env["indneutralize"](((((env["open"]) * (const("alpha101.80.const.001", 0.868128)))) + (((env["high"]) * (((1) - (const("alpha101.80.const.002", 0.868128))))))), env["industry"]), const("alpha101.80.const.003", 4.04545))))) ** (env["ts_rank"](env["ts_corr"](env["high"], env["adv10"], const("alpha101.80.const.004", 5.11456)), const("alpha101.80.const.005", 5.53756))))) * ((-(1))))

def alpha_081_program(env, const):
    return ((((env["rank"](env["log"](env["ts_product"](env["rank"](((env["rank"](env["ts_corr"](env["vwap"], env["ts_sum"](env["adv10"], const("alpha101.81.const.001", 49.6054)), const("alpha101.81.const.002", 8.47743)))) ** (4))), const("alpha101.81.const.003", 14.9655))))) < (env["rank"](env["ts_corr"](env["rank"](env["vwap"]), env["rank"](env["volume"]), const("alpha101.81.const.004", 5.07914)))))) * ((-(1))))

def alpha_082_program(env, const):
    return ((env["min"](env["rank"](env["decay_linear"](env["delta"](env["open"], const("alpha101.82.const.001", 1.46063)), const("alpha101.82.const.002", 14.8717))), env["ts_rank"](env["decay_linear"](env["ts_corr"](env["indneutralize"](env["volume"], env["sector"]), ((((env["open"]) * (const("alpha101.82.const.003", 0.634196)))) + (((env["open"]) * (((1) - (const("alpha101.82.const.004", 0.634196))))))), const("alpha101.82.const.005", 17.4842)), const("alpha101.82.const.006", 6.92131)), const("alpha101.82.const.007", 13.4283)))) * ((-(1))))

def alpha_083_program(env, const):
    return ((((env["rank"](env["delay"](((((env["high"]) - (env["low"]))) / (((env["ts_sum"](env["close"], const("alpha101.83.const.001", 5.0))) / (const("alpha101.83.const.002", 5.0))))), 2))) * (env["rank"](env["rank"](env["volume"]))))) / (((((((env["high"]) - (env["low"]))) / (((env["ts_sum"](env["close"], const("alpha101.83.const.003", 5.0))) / (const("alpha101.83.const.004", 5.0)))))) / (((env["vwap"]) - (env["close"]))))))

def alpha_084_program(env, const):
    return env["signed_power"](env["ts_rank"](((env["vwap"]) - (env["ts_max"](env["vwap"], const("alpha101.84.const.001", 15.3217)))), const("alpha101.84.const.002", 20.7127)), env["delta"](env["close"], const("alpha101.84.const.003", 4.96796)))

def alpha_085_program(env, const):
    return ((env["rank"](env["ts_corr"](((((env["high"]) * (const("alpha101.85.const.001", 0.876703)))) + (((env["close"]) * (((1) - (const("alpha101.85.const.002", 0.876703))))))), env["adv30"], const("alpha101.85.const.003", 9.61331)))) ** (env["rank"](env["ts_corr"](env["ts_rank"](((((env["high"]) + (env["low"]))) / (2)), const("alpha101.85.const.004", 3.70596)), env["ts_rank"](env["volume"], const("alpha101.85.const.005", 10.1595)), const("alpha101.85.const.006", 7.11408)))))

def alpha_086_program(env, const):
    return ((((env["ts_rank"](env["ts_corr"](env["close"], env["ts_sum"](env["adv20"], const("alpha101.86.const.001", 14.7444)), const("alpha101.86.const.002", 6.00049)), const("alpha101.86.const.003", 20.4195))) < (env["rank"](((((env["open"]) + (env["close"]))) - (((env["vwap"]) + (env["open"])))))))) * ((-(1))))

def alpha_087_program(env, const):
    return ((env["max"](env["rank"](env["decay_linear"](env["delta"](((((env["close"]) * (const("alpha101.87.const.001", 0.369701)))) + (((env["vwap"]) * (((1) - (const("alpha101.87.const.002", 0.369701))))))), const("alpha101.87.const.003", 1.91233)), const("alpha101.87.const.004", 2.65461))), env["ts_rank"](env["decay_linear"](env["abs"](env["ts_corr"](env["indneutralize"](env["adv81"], env["industry"]), env["close"], const("alpha101.87.const.005", 13.4132))), const("alpha101.87.const.006", 4.89768)), const("alpha101.87.const.007", 14.4535)))) * ((-(1))))

def alpha_088_program(env, const):
    return env["min"](env["rank"](env["decay_linear"](((((env["rank"](env["open"])) + (env["rank"](env["low"])))) - (((env["rank"](env["high"])) + (env["rank"](env["close"]))))), const("alpha101.88.const.001", 8.06882))), env["ts_rank"](env["decay_linear"](env["ts_corr"](env["ts_rank"](env["close"], const("alpha101.88.const.002", 8.44728)), env["ts_rank"](env["adv60"], const("alpha101.88.const.003", 20.6966)), const("alpha101.88.const.004", 8.01266)), const("alpha101.88.const.005", 6.65053)), const("alpha101.88.const.006", 2.61957)))

def alpha_089_program(env, const):
    return ((env["ts_rank"](env["decay_linear"](env["ts_corr"](((((env["low"]) * (const("alpha101.89.const.001", 0.967285)))) + (((env["low"]) * (((1) - (const("alpha101.89.const.002", 0.967285))))))), env["adv10"], const("alpha101.89.const.003", 6.94279)), const("alpha101.89.const.004", 5.51607)), const("alpha101.89.const.005", 3.79744))) - (env["ts_rank"](env["decay_linear"](env["delta"](env["indneutralize"](env["vwap"], env["industry"]), const("alpha101.89.const.006", 3.48158)), const("alpha101.89.const.007", 10.1466)), const("alpha101.89.const.008", 15.3012))))

def alpha_090_program(env, const):
    return ((((env["rank"](((env["close"]) - (env["ts_max"](env["close"], const("alpha101.90.const.001", 4.66719)))))) ** (env["ts_rank"](env["ts_corr"](env["indneutralize"](env["adv40"], env["subindustry"]), env["low"], const("alpha101.90.const.002", 5.38375)), const("alpha101.90.const.003", 3.21856))))) * ((-(1))))

def alpha_091_program(env, const):
    return ((((env["ts_rank"](env["decay_linear"](env["decay_linear"](env["ts_corr"](env["indneutralize"](env["close"], env["industry"]), env["volume"], const("alpha101.91.const.001", 9.74928)), const("alpha101.91.const.002", 16.398)), const("alpha101.91.const.003", 3.83219)), const("alpha101.91.const.004", 4.8667))) - (env["rank"](env["decay_linear"](env["ts_corr"](env["vwap"], env["adv30"], const("alpha101.91.const.005", 4.01303)), const("alpha101.91.const.006", 2.6809)))))) * ((-(1))))

def alpha_092_program(env, const):
    return env["min"](env["ts_rank"](env["decay_linear"](((((((((env["high"]) + (env["low"]))) / (2))) + (env["close"]))) < (((env["low"]) + (env["open"])))), const("alpha101.92.const.001", 14.7221)), const("alpha101.92.const.002", 18.8683)), env["ts_rank"](env["decay_linear"](env["ts_corr"](env["rank"](env["low"]), env["rank"](env["adv30"]), const("alpha101.92.const.003", 7.58555)), const("alpha101.92.const.004", 6.94024)), const("alpha101.92.const.005", 6.80584)))

def alpha_093_program(env, const):
    return ((env["ts_rank"](env["decay_linear"](env["ts_corr"](env["indneutralize"](env["vwap"], env["industry"]), env["adv81"], const("alpha101.93.const.001", 17.4193)), const("alpha101.93.const.002", 19.848)), const("alpha101.93.const.003", 7.54455))) / (env["rank"](env["decay_linear"](env["delta"](((((env["close"]) * (const("alpha101.93.const.004", 0.524434)))) + (((env["vwap"]) * (((1) - (const("alpha101.93.const.005", 0.524434))))))), const("alpha101.93.const.006", 2.77377)), const("alpha101.93.const.007", 16.2664)))))

def alpha_094_program(env, const):
    return ((((env["rank"](((env["vwap"]) - (env["ts_min"](env["vwap"], const("alpha101.94.const.001", 11.5783)))))) ** (env["ts_rank"](env["ts_corr"](env["ts_rank"](env["vwap"], const("alpha101.94.const.002", 19.6462)), env["ts_rank"](env["adv60"], const("alpha101.94.const.003", 4.02992)), const("alpha101.94.const.004", 18.0926)), const("alpha101.94.const.005", 2.70756))))) * ((-(1))))

def alpha_095_program(env, const):
    return ((env["rank"](((env["open"]) - (env["ts_min"](env["open"], const("alpha101.95.const.001", 12.4105)))))) < (env["ts_rank"](((env["rank"](env["ts_corr"](env["ts_sum"](((((env["high"]) + (env["low"]))) / (2)), const("alpha101.95.const.002", 19.1351)), env["ts_sum"](env["adv40"], const("alpha101.95.const.003", 19.1351)), const("alpha101.95.const.004", 12.8742)))) ** (const("alpha101.95.const.005", 5.0))), const("alpha101.95.const.006", 11.7584))))

def alpha_096_program(env, const):
    return ((env["max"](env["ts_rank"](env["decay_linear"](env["ts_corr"](env["rank"](env["vwap"]), env["rank"](env["volume"]), const("alpha101.96.const.001", 3.83878)), const("alpha101.96.const.002", 4.16783)), const("alpha101.96.const.003", 8.38151)), env["ts_rank"](env["decay_linear"](env["ts_argmax"](env["ts_corr"](env["ts_rank"](env["close"], const("alpha101.96.const.004", 7.45404)), env["ts_rank"](env["adv60"], const("alpha101.96.const.005", 4.13242)), const("alpha101.96.const.006", 3.65459)), const("alpha101.96.const.007", 12.6556)), const("alpha101.96.const.008", 14.0365)), const("alpha101.96.const.009", 13.4143)))) * ((-(1))))

def alpha_097_program(env, const):
    return ((((env["rank"](env["decay_linear"](env["delta"](env["indneutralize"](((((env["low"]) * (const("alpha101.97.const.001", 0.721001)))) + (((env["vwap"]) * (((1) - (const("alpha101.97.const.002", 0.721001))))))), env["industry"]), const("alpha101.97.const.003", 3.3705)), const("alpha101.97.const.004", 20.4523)))) - (env["ts_rank"](env["decay_linear"](env["ts_rank"](env["ts_corr"](env["ts_rank"](env["low"], const("alpha101.97.const.005", 7.87871)), env["ts_rank"](env["adv60"], const("alpha101.97.const.006", 17.255)), const("alpha101.97.const.007", 4.97547)), const("alpha101.97.const.008", 18.5925)), const("alpha101.97.const.009", 15.7152)), const("alpha101.97.const.010", 6.71659))))) * ((-(1))))

def alpha_098_program(env, const):
    return ((env["rank"](env["decay_linear"](env["ts_corr"](env["vwap"], env["ts_sum"](env["adv5"], const("alpha101.98.const.001", 26.4719)), const("alpha101.98.const.002", 4.58418)), const("alpha101.98.const.003", 7.18088)))) - (env["rank"](env["decay_linear"](env["ts_rank"](env["ts_argmin"](env["ts_corr"](env["rank"](env["open"]), env["rank"](env["adv15"]), const("alpha101.98.const.004", 20.8187)), const("alpha101.98.const.005", 8.62571)), const("alpha101.98.const.006", 6.95668)), const("alpha101.98.const.007", 8.07206)))))

def alpha_099_program(env, const):
    return ((((env["rank"](env["ts_corr"](env["ts_sum"](((((env["high"]) + (env["low"]))) / (2)), const("alpha101.99.const.001", 19.8975)), env["ts_sum"](env["adv60"], const("alpha101.99.const.002", 19.8975)), const("alpha101.99.const.003", 8.8136)))) < (env["rank"](env["ts_corr"](env["low"], env["volume"], const("alpha101.99.const.004", 6.28259)))))) * ((-(1))))

def alpha_100_program(env, const):
    return ((0) - (((1) * (((((((const("alpha101.100.const.001", 1.5)) * (env["scale"](env["indneutralize"](env["indneutralize"](env["rank"](((((((((env["close"]) - (env["low"]))) - (((env["high"]) - (env["close"]))))) / (((env["high"]) - (env["low"]))))) * (env["volume"]))), env["subindustry"]), env["subindustry"]))))) - (env["scale"](env["indneutralize"](((env["ts_corr"](env["close"], env["rank"](env["adv20"]), const("alpha101.100.const.002", 5.0))) - (env["rank"](env["ts_argmin"](env["close"], const("alpha101.100.const.003", 30.0))))), env["subindustry"]))))) * (((env["volume"]) / (env["adv20"]))))))))

def alpha_101_program(env, const):
    return ((((env["close"]) - (env["open"]))) / (((((env["high"]) - (env["low"]))) + (const("alpha101.101.const.001", 0.001)))))

ALPHA_PROGRAM_DEFINITIONS: dict[int, AlphaProgramDefinition] = {
    1: AlphaProgramDefinition(
        alpha_id=1,
        program=alpha_001_program,
        tunable_constants={
            "alpha101.1.const.001": 20.0,
            "alpha101.1.const.002": 5.0,
        },
    ),
    2: AlphaProgramDefinition(
        alpha_id=2,
        program=alpha_002_program,
        tunable_constants={
            "alpha101.2.const.001": 6.0,
        },
    ),
    3: AlphaProgramDefinition(
        alpha_id=3,
        program=alpha_003_program,
        tunable_constants={
            "alpha101.3.const.001": 10.0,
        },
    ),
    4: AlphaProgramDefinition(
        alpha_id=4,
        program=alpha_004_program,
        tunable_constants={
            "alpha101.4.const.001": 9.0,
        },
    ),
    5: AlphaProgramDefinition(
        alpha_id=5,
        program=alpha_005_program,
        tunable_constants={
            "alpha101.5.const.001": 10.0,
            "alpha101.5.const.002": 10.0,
        },
    ),
    6: AlphaProgramDefinition(
        alpha_id=6,
        program=alpha_006_program,
        tunable_constants={
            "alpha101.6.const.001": 10.0,
        },
    ),
    7: AlphaProgramDefinition(
        alpha_id=7,
        program=alpha_007_program,
        tunable_constants={
            "alpha101.7.const.001": 7.0,
            "alpha101.7.const.002": 60.0,
            "alpha101.7.const.003": 7.0,
        },
    ),
    8: AlphaProgramDefinition(
        alpha_id=8,
        program=alpha_008_program,
        tunable_constants={
            "alpha101.8.const.001": 5.0,
            "alpha101.8.const.002": 5.0,
            "alpha101.8.const.003": 5.0,
            "alpha101.8.const.004": 5.0,
            "alpha101.8.const.005": 10.0,
        },
    ),
    9: AlphaProgramDefinition(
        alpha_id=9,
        program=alpha_009_program,
        tunable_constants={
            "alpha101.9.const.001": 5.0,
            "alpha101.9.const.002": 5.0,
        },
    ),
    10: AlphaProgramDefinition(
        alpha_id=10,
        program=alpha_010_program,
        tunable_constants={},
    ),
    11: AlphaProgramDefinition(
        alpha_id=11,
        program=alpha_011_program,
        tunable_constants={
            "alpha101.11.const.001": 3.0,
            "alpha101.11.const.002": 3.0,
            "alpha101.11.const.003": 3.0,
        },
    ),
    12: AlphaProgramDefinition(
        alpha_id=12,
        program=alpha_012_program,
        tunable_constants={},
    ),
    13: AlphaProgramDefinition(
        alpha_id=13,
        program=alpha_013_program,
        tunable_constants={
            "alpha101.13.const.001": 5.0,
        },
    ),
    14: AlphaProgramDefinition(
        alpha_id=14,
        program=alpha_014_program,
        tunable_constants={
            "alpha101.14.const.001": 3.0,
            "alpha101.14.const.002": 10.0,
        },
    ),
    15: AlphaProgramDefinition(
        alpha_id=15,
        program=alpha_015_program,
        tunable_constants={
            "alpha101.15.const.001": 3.0,
            "alpha101.15.const.002": 3.0,
        },
    ),
    16: AlphaProgramDefinition(
        alpha_id=16,
        program=alpha_016_program,
        tunable_constants={
            "alpha101.16.const.001": 5.0,
        },
    ),
    17: AlphaProgramDefinition(
        alpha_id=17,
        program=alpha_017_program,
        tunable_constants={
            "alpha101.17.const.001": 10.0,
            "alpha101.17.const.002": 5.0,
        },
    ),
    18: AlphaProgramDefinition(
        alpha_id=18,
        program=alpha_018_program,
        tunable_constants={
            "alpha101.18.const.001": 5.0,
            "alpha101.18.const.002": 10.0,
        },
    ),
    19: AlphaProgramDefinition(
        alpha_id=19,
        program=alpha_019_program,
        tunable_constants={
            "alpha101.19.const.001": 7.0,
            "alpha101.19.const.002": 7.0,
            "alpha101.19.const.003": 250.0,
        },
    ),
    20: AlphaProgramDefinition(
        alpha_id=20,
        program=alpha_020_program,
        tunable_constants={},
    ),
    21: AlphaProgramDefinition(
        alpha_id=21,
        program=alpha_021_program,
        tunable_constants={},
    ),
    22: AlphaProgramDefinition(
        alpha_id=22,
        program=alpha_022_program,
        tunable_constants={
            "alpha101.22.const.001": 5.0,
            "alpha101.22.const.002": 5.0,
            "alpha101.22.const.003": 20.0,
        },
    ),
    23: AlphaProgramDefinition(
        alpha_id=23,
        program=alpha_023_program,
        tunable_constants={
            "alpha101.23.const.001": 20.0,
            "alpha101.23.const.002": 20.0,
        },
    ),
    24: AlphaProgramDefinition(
        alpha_id=24,
        program=alpha_024_program,
        tunable_constants={
            "alpha101.24.const.001": 100.0,
            "alpha101.24.const.002": 100.0,
            "alpha101.24.const.003": 100.0,
            "alpha101.24.const.004": 100.0,
            "alpha101.24.const.005": 0.05,
            "alpha101.24.const.006": 100.0,
            "alpha101.24.const.007": 100.0,
            "alpha101.24.const.008": 100.0,
            "alpha101.24.const.009": 100.0,
            "alpha101.24.const.010": 0.05,
            "alpha101.24.const.011": 100.0,
            "alpha101.24.const.012": 3.0,
        },
    ),
    25: AlphaProgramDefinition(
        alpha_id=25,
        program=alpha_025_program,
        tunable_constants={},
    ),
    26: AlphaProgramDefinition(
        alpha_id=26,
        program=alpha_026_program,
        tunable_constants={
            "alpha101.26.const.001": 5.0,
            "alpha101.26.const.002": 5.0,
            "alpha101.26.const.003": 5.0,
            "alpha101.26.const.004": 3.0,
        },
    ),
    27: AlphaProgramDefinition(
        alpha_id=27,
        program=alpha_027_program,
        tunable_constants={
            "alpha101.27.const.001": 6.0,
        },
    ),
    28: AlphaProgramDefinition(
        alpha_id=28,
        program=alpha_028_program,
        tunable_constants={
            "alpha101.28.const.001": 5.0,
        },
    ),
    29: AlphaProgramDefinition(
        alpha_id=29,
        program=alpha_029_program,
        tunable_constants={
            "alpha101.29.const.001": 5.0,
            "alpha101.29.const.002": 5.0,
            "alpha101.29.const.003": 6.0,
            "alpha101.29.const.004": 5.0,
        },
    ),
    30: AlphaProgramDefinition(
        alpha_id=30,
        program=alpha_030_program,
        tunable_constants={
            "alpha101.30.const.001": 3.0,
            "alpha101.30.const.002": 5.0,
            "alpha101.30.const.003": 20.0,
        },
    ),
    31: AlphaProgramDefinition(
        alpha_id=31,
        program=alpha_031_program,
        tunable_constants={
            "alpha101.31.const.001": 10.0,
            "alpha101.31.const.002": 10.0,
            "alpha101.31.const.003": 3.0,
            "alpha101.31.const.004": 12.0,
        },
    ),
    32: AlphaProgramDefinition(
        alpha_id=32,
        program=alpha_032_program,
        tunable_constants={
            "alpha101.32.const.001": 7.0,
            "alpha101.32.const.002": 7.0,
            "alpha101.32.const.003": 20.0,
            "alpha101.32.const.004": 5.0,
            "alpha101.32.const.005": 230.0,
        },
    ),
    33: AlphaProgramDefinition(
        alpha_id=33,
        program=alpha_033_program,
        tunable_constants={},
    ),
    34: AlphaProgramDefinition(
        alpha_id=34,
        program=alpha_034_program,
        tunable_constants={
            "alpha101.34.const.001": 5.0,
        },
    ),
    35: AlphaProgramDefinition(
        alpha_id=35,
        program=alpha_035_program,
        tunable_constants={},
    ),
    36: AlphaProgramDefinition(
        alpha_id=36,
        program=alpha_036_program,
        tunable_constants={
            "alpha101.36.const.001": 2.21,
            "alpha101.36.const.002": 15.0,
            "alpha101.36.const.003": 0.7,
            "alpha101.36.const.004": 0.73,
            "alpha101.36.const.005": 6.0,
            "alpha101.36.const.006": 5.0,
            "alpha101.36.const.007": 6.0,
            "alpha101.36.const.008": 0.6,
            "alpha101.36.const.009": 200.0,
            "alpha101.36.const.010": 200.0,
        },
    ),
    37: AlphaProgramDefinition(
        alpha_id=37,
        program=alpha_037_program,
        tunable_constants={
            "alpha101.37.const.001": 200.0,
        },
    ),
    38: AlphaProgramDefinition(
        alpha_id=38,
        program=alpha_038_program,
        tunable_constants={
            "alpha101.38.const.001": 10.0,
        },
    ),
    39: AlphaProgramDefinition(
        alpha_id=39,
        program=alpha_039_program,
        tunable_constants={
            "alpha101.39.const.001": 7.0,
            "alpha101.39.const.002": 9.0,
            "alpha101.39.const.003": 250.0,
        },
    ),
    40: AlphaProgramDefinition(
        alpha_id=40,
        program=alpha_040_program,
        tunable_constants={
            "alpha101.40.const.001": 10.0,
            "alpha101.40.const.002": 10.0,
        },
    ),
    41: AlphaProgramDefinition(
        alpha_id=41,
        program=alpha_041_program,
        tunable_constants={},
    ),
    42: AlphaProgramDefinition(
        alpha_id=42,
        program=alpha_042_program,
        tunable_constants={},
    ),
    43: AlphaProgramDefinition(
        alpha_id=43,
        program=alpha_043_program,
        tunable_constants={
            "alpha101.43.const.001": 20.0,
            "alpha101.43.const.002": 7.0,
        },
    ),
    44: AlphaProgramDefinition(
        alpha_id=44,
        program=alpha_044_program,
        tunable_constants={
            "alpha101.44.const.001": 5.0,
        },
    ),
    45: AlphaProgramDefinition(
        alpha_id=45,
        program=alpha_045_program,
        tunable_constants={
            "alpha101.45.const.001": 5.0,
            "alpha101.45.const.002": 20.0,
            "alpha101.45.const.003": 20.0,
            "alpha101.45.const.004": 5.0,
            "alpha101.45.const.005": 20.0,
        },
    ),
    46: AlphaProgramDefinition(
        alpha_id=46,
        program=alpha_046_program,
        tunable_constants={
            "alpha101.46.const.001": 20.0,
            "alpha101.46.const.002": 10.0,
            "alpha101.46.const.003": 10.0,
            "alpha101.46.const.004": 10.0,
            "alpha101.46.const.005": 10.0,
            "alpha101.46.const.006": 20.0,
            "alpha101.46.const.007": 10.0,
            "alpha101.46.const.008": 10.0,
            "alpha101.46.const.009": 10.0,
            "alpha101.46.const.010": 10.0,
        },
    ),
    47: AlphaProgramDefinition(
        alpha_id=47,
        program=alpha_047_program,
        tunable_constants={
            "alpha101.47.const.001": 5.0,
            "alpha101.47.const.002": 5.0,
            "alpha101.47.const.003": 5.0,
        },
    ),
    48: AlphaProgramDefinition(
        alpha_id=48,
        program=alpha_048_program,
        tunable_constants={
            "alpha101.48.const.001": 250.0,
            "alpha101.48.const.002": 250.0,
        },
    ),
    49: AlphaProgramDefinition(
        alpha_id=49,
        program=alpha_049_program,
        tunable_constants={
            "alpha101.49.const.001": 20.0,
            "alpha101.49.const.002": 10.0,
            "alpha101.49.const.003": 10.0,
            "alpha101.49.const.004": 10.0,
            "alpha101.49.const.005": 10.0,
            "alpha101.49.const.006": 0.1,
        },
    ),
    50: AlphaProgramDefinition(
        alpha_id=50,
        program=alpha_050_program,
        tunable_constants={
            "alpha101.50.const.001": 5.0,
            "alpha101.50.const.002": 5.0,
        },
    ),
    51: AlphaProgramDefinition(
        alpha_id=51,
        program=alpha_051_program,
        tunable_constants={
            "alpha101.51.const.001": 20.0,
            "alpha101.51.const.002": 10.0,
            "alpha101.51.const.003": 10.0,
            "alpha101.51.const.004": 10.0,
            "alpha101.51.const.005": 10.0,
            "alpha101.51.const.006": 0.05,
        },
    ),
    52: AlphaProgramDefinition(
        alpha_id=52,
        program=alpha_052_program,
        tunable_constants={
            "alpha101.52.const.001": 5.0,
            "alpha101.52.const.002": 5.0,
            "alpha101.52.const.003": 5.0,
            "alpha101.52.const.004": 240.0,
            "alpha101.52.const.005": 20.0,
            "alpha101.52.const.006": 220.0,
            "alpha101.52.const.007": 5.0,
        },
    ),
    53: AlphaProgramDefinition(
        alpha_id=53,
        program=alpha_053_program,
        tunable_constants={
            "alpha101.53.const.001": 9.0,
        },
    ),
    54: AlphaProgramDefinition(
        alpha_id=54,
        program=alpha_054_program,
        tunable_constants={
            "alpha101.54.const.001": 5.0,
            "alpha101.54.const.002": 5.0,
        },
    ),
    55: AlphaProgramDefinition(
        alpha_id=55,
        program=alpha_055_program,
        tunable_constants={
            "alpha101.55.const.001": 12.0,
            "alpha101.55.const.002": 12.0,
            "alpha101.55.const.003": 12.0,
            "alpha101.55.const.004": 6.0,
        },
    ),
    56: AlphaProgramDefinition(
        alpha_id=56,
        program=alpha_056_program,
        tunable_constants={
            "alpha101.56.const.001": 10.0,
            "alpha101.56.const.002": 3.0,
        },
    ),
    57: AlphaProgramDefinition(
        alpha_id=57,
        program=alpha_057_program,
        tunable_constants={
            "alpha101.57.const.001": 30.0,
        },
    ),
    58: AlphaProgramDefinition(
        alpha_id=58,
        program=alpha_058_program,
        tunable_constants={
            "alpha101.58.const.001": 3.92795,
            "alpha101.58.const.002": 7.89291,
            "alpha101.58.const.003": 5.50322,
        },
    ),
    59: AlphaProgramDefinition(
        alpha_id=59,
        program=alpha_059_program,
        tunable_constants={
            "alpha101.59.const.001": 0.728317,
            "alpha101.59.const.002": 0.728317,
            "alpha101.59.const.003": 4.25197,
            "alpha101.59.const.004": 16.2289,
            "alpha101.59.const.005": 8.19648,
        },
    ),
    60: AlphaProgramDefinition(
        alpha_id=60,
        program=alpha_060_program,
        tunable_constants={
            "alpha101.60.const.001": 10.0,
        },
    ),
    61: AlphaProgramDefinition(
        alpha_id=61,
        program=alpha_061_program,
        tunable_constants={
            "alpha101.61.const.001": 16.1219,
            "alpha101.61.const.002": 17.9282,
        },
    ),
    62: AlphaProgramDefinition(
        alpha_id=62,
        program=alpha_062_program,
        tunable_constants={
            "alpha101.62.const.001": 22.4101,
            "alpha101.62.const.002": 9.91009,
        },
    ),
    63: AlphaProgramDefinition(
        alpha_id=63,
        program=alpha_063_program,
        tunable_constants={
            "alpha101.63.const.001": 2.25164,
            "alpha101.63.const.002": 8.22237,
            "alpha101.63.const.003": 0.318108,
            "alpha101.63.const.004": 0.318108,
            "alpha101.63.const.005": 37.2467,
            "alpha101.63.const.006": 13.557,
            "alpha101.63.const.007": 12.2883,
        },
    ),
    64: AlphaProgramDefinition(
        alpha_id=64,
        program=alpha_064_program,
        tunable_constants={
            "alpha101.64.const.001": 0.178404,
            "alpha101.64.const.002": 0.178404,
            "alpha101.64.const.003": 12.7054,
            "alpha101.64.const.004": 12.7054,
            "alpha101.64.const.005": 16.6208,
            "alpha101.64.const.006": 0.178404,
            "alpha101.64.const.007": 0.178404,
            "alpha101.64.const.008": 3.69741,
        },
    ),
    65: AlphaProgramDefinition(
        alpha_id=65,
        program=alpha_065_program,
        tunable_constants={
            "alpha101.65.const.001": 0.00817205,
            "alpha101.65.const.002": 0.00817205,
            "alpha101.65.const.003": 8.6911,
            "alpha101.65.const.004": 6.40374,
            "alpha101.65.const.005": 13.635,
        },
    ),
    66: AlphaProgramDefinition(
        alpha_id=66,
        program=alpha_066_program,
        tunable_constants={
            "alpha101.66.const.001": 3.51013,
            "alpha101.66.const.002": 7.23052,
            "alpha101.66.const.003": 0.96633,
            "alpha101.66.const.004": 0.96633,
            "alpha101.66.const.005": 11.4157,
            "alpha101.66.const.006": 6.72611,
        },
    ),
    67: AlphaProgramDefinition(
        alpha_id=67,
        program=alpha_067_program,
        tunable_constants={
            "alpha101.67.const.001": 2.14593,
            "alpha101.67.const.002": 6.02936,
        },
    ),
    68: AlphaProgramDefinition(
        alpha_id=68,
        program=alpha_068_program,
        tunable_constants={
            "alpha101.68.const.001": 8.91644,
            "alpha101.68.const.002": 13.9333,
            "alpha101.68.const.003": 0.518371,
            "alpha101.68.const.004": 0.518371,
            "alpha101.68.const.005": 1.06157,
        },
    ),
    69: AlphaProgramDefinition(
        alpha_id=69,
        program=alpha_069_program,
        tunable_constants={
            "alpha101.69.const.001": 2.72412,
            "alpha101.69.const.002": 4.79344,
            "alpha101.69.const.003": 0.490655,
            "alpha101.69.const.004": 0.490655,
            "alpha101.69.const.005": 4.92416,
            "alpha101.69.const.006": 9.0615,
        },
    ),
    70: AlphaProgramDefinition(
        alpha_id=70,
        program=alpha_070_program,
        tunable_constants={
            "alpha101.70.const.001": 1.29456,
            "alpha101.70.const.002": 17.8256,
            "alpha101.70.const.003": 17.9171,
        },
    ),
    71: AlphaProgramDefinition(
        alpha_id=71,
        program=alpha_071_program,
        tunable_constants={
            "alpha101.71.const.001": 3.43976,
            "alpha101.71.const.002": 12.0647,
            "alpha101.71.const.003": 18.0175,
            "alpha101.71.const.004": 4.20501,
            "alpha101.71.const.005": 15.6948,
            "alpha101.71.const.006": 16.4662,
            "alpha101.71.const.007": 4.4388,
        },
    ),
    72: AlphaProgramDefinition(
        alpha_id=72,
        program=alpha_072_program,
        tunable_constants={
            "alpha101.72.const.001": 8.93345,
            "alpha101.72.const.002": 10.1519,
            "alpha101.72.const.003": 3.72469,
            "alpha101.72.const.004": 18.5188,
            "alpha101.72.const.005": 6.86671,
            "alpha101.72.const.006": 2.95011,
        },
    ),
    73: AlphaProgramDefinition(
        alpha_id=73,
        program=alpha_073_program,
        tunable_constants={
            "alpha101.73.const.001": 4.72775,
            "alpha101.73.const.002": 2.91864,
            "alpha101.73.const.003": 0.147155,
            "alpha101.73.const.004": 0.147155,
            "alpha101.73.const.005": 2.03608,
            "alpha101.73.const.006": 0.147155,
            "alpha101.73.const.007": 0.147155,
            "alpha101.73.const.008": 3.33829,
            "alpha101.73.const.009": 16.7411,
        },
    ),
    74: AlphaProgramDefinition(
        alpha_id=74,
        program=alpha_074_program,
        tunable_constants={
            "alpha101.74.const.001": 37.4843,
            "alpha101.74.const.002": 15.1365,
            "alpha101.74.const.003": 0.0261661,
            "alpha101.74.const.004": 0.0261661,
            "alpha101.74.const.005": 11.4791,
        },
    ),
    75: AlphaProgramDefinition(
        alpha_id=75,
        program=alpha_075_program,
        tunable_constants={
            "alpha101.75.const.001": 4.24304,
            "alpha101.75.const.002": 12.4413,
        },
    ),
    76: AlphaProgramDefinition(
        alpha_id=76,
        program=alpha_076_program,
        tunable_constants={
            "alpha101.76.const.001": 1.24383,
            "alpha101.76.const.002": 11.8259,
            "alpha101.76.const.003": 8.14941,
            "alpha101.76.const.004": 19.569,
            "alpha101.76.const.005": 17.1543,
            "alpha101.76.const.006": 19.383,
        },
    ),
    77: AlphaProgramDefinition(
        alpha_id=77,
        program=alpha_077_program,
        tunable_constants={
            "alpha101.77.const.001": 20.0451,
            "alpha101.77.const.002": 3.1614,
            "alpha101.77.const.003": 5.64125,
        },
    ),
    78: AlphaProgramDefinition(
        alpha_id=78,
        program=alpha_078_program,
        tunable_constants={
            "alpha101.78.const.001": 0.352233,
            "alpha101.78.const.002": 0.352233,
            "alpha101.78.const.003": 19.7428,
            "alpha101.78.const.004": 19.7428,
            "alpha101.78.const.005": 6.83313,
            "alpha101.78.const.006": 5.77492,
        },
    ),
    79: AlphaProgramDefinition(
        alpha_id=79,
        program=alpha_079_program,
        tunable_constants={
            "alpha101.79.const.001": 0.60733,
            "alpha101.79.const.002": 0.60733,
            "alpha101.79.const.003": 1.23438,
            "alpha101.79.const.004": 3.60973,
            "alpha101.79.const.005": 9.18637,
            "alpha101.79.const.006": 14.6644,
        },
    ),
    80: AlphaProgramDefinition(
        alpha_id=80,
        program=alpha_080_program,
        tunable_constants={
            "alpha101.80.const.001": 0.868128,
            "alpha101.80.const.002": 0.868128,
            "alpha101.80.const.003": 4.04545,
            "alpha101.80.const.004": 5.11456,
            "alpha101.80.const.005": 5.53756,
        },
    ),
    81: AlphaProgramDefinition(
        alpha_id=81,
        program=alpha_081_program,
        tunable_constants={
            "alpha101.81.const.001": 49.6054,
            "alpha101.81.const.002": 8.47743,
            "alpha101.81.const.003": 14.9655,
            "alpha101.81.const.004": 5.07914,
        },
    ),
    82: AlphaProgramDefinition(
        alpha_id=82,
        program=alpha_082_program,
        tunable_constants={
            "alpha101.82.const.001": 1.46063,
            "alpha101.82.const.002": 14.8717,
            "alpha101.82.const.003": 0.634196,
            "alpha101.82.const.004": 0.634196,
            "alpha101.82.const.005": 17.4842,
            "alpha101.82.const.006": 6.92131,
            "alpha101.82.const.007": 13.4283,
        },
    ),
    83: AlphaProgramDefinition(
        alpha_id=83,
        program=alpha_083_program,
        tunable_constants={
            "alpha101.83.const.001": 5.0,
            "alpha101.83.const.002": 5.0,
            "alpha101.83.const.003": 5.0,
            "alpha101.83.const.004": 5.0,
        },
    ),
    84: AlphaProgramDefinition(
        alpha_id=84,
        program=alpha_084_program,
        tunable_constants={
            "alpha101.84.const.001": 15.3217,
            "alpha101.84.const.002": 20.7127,
            "alpha101.84.const.003": 4.96796,
        },
    ),
    85: AlphaProgramDefinition(
        alpha_id=85,
        program=alpha_085_program,
        tunable_constants={
            "alpha101.85.const.001": 0.876703,
            "alpha101.85.const.002": 0.876703,
            "alpha101.85.const.003": 9.61331,
            "alpha101.85.const.004": 3.70596,
            "alpha101.85.const.005": 10.1595,
            "alpha101.85.const.006": 7.11408,
        },
    ),
    86: AlphaProgramDefinition(
        alpha_id=86,
        program=alpha_086_program,
        tunable_constants={
            "alpha101.86.const.001": 14.7444,
            "alpha101.86.const.002": 6.00049,
            "alpha101.86.const.003": 20.4195,
        },
    ),
    87: AlphaProgramDefinition(
        alpha_id=87,
        program=alpha_087_program,
        tunable_constants={
            "alpha101.87.const.001": 0.369701,
            "alpha101.87.const.002": 0.369701,
            "alpha101.87.const.003": 1.91233,
            "alpha101.87.const.004": 2.65461,
            "alpha101.87.const.005": 13.4132,
            "alpha101.87.const.006": 4.89768,
            "alpha101.87.const.007": 14.4535,
        },
    ),
    88: AlphaProgramDefinition(
        alpha_id=88,
        program=alpha_088_program,
        tunable_constants={
            "alpha101.88.const.001": 8.06882,
            "alpha101.88.const.002": 8.44728,
            "alpha101.88.const.003": 20.6966,
            "alpha101.88.const.004": 8.01266,
            "alpha101.88.const.005": 6.65053,
            "alpha101.88.const.006": 2.61957,
        },
    ),
    89: AlphaProgramDefinition(
        alpha_id=89,
        program=alpha_089_program,
        tunable_constants={
            "alpha101.89.const.001": 0.967285,
            "alpha101.89.const.002": 0.967285,
            "alpha101.89.const.003": 6.94279,
            "alpha101.89.const.004": 5.51607,
            "alpha101.89.const.005": 3.79744,
            "alpha101.89.const.006": 3.48158,
            "alpha101.89.const.007": 10.1466,
            "alpha101.89.const.008": 15.3012,
        },
    ),
    90: AlphaProgramDefinition(
        alpha_id=90,
        program=alpha_090_program,
        tunable_constants={
            "alpha101.90.const.001": 4.66719,
            "alpha101.90.const.002": 5.38375,
            "alpha101.90.const.003": 3.21856,
        },
    ),
    91: AlphaProgramDefinition(
        alpha_id=91,
        program=alpha_091_program,
        tunable_constants={
            "alpha101.91.const.001": 9.74928,
            "alpha101.91.const.002": 16.398,
            "alpha101.91.const.003": 3.83219,
            "alpha101.91.const.004": 4.8667,
            "alpha101.91.const.005": 4.01303,
            "alpha101.91.const.006": 2.6809,
        },
    ),
    92: AlphaProgramDefinition(
        alpha_id=92,
        program=alpha_092_program,
        tunable_constants={
            "alpha101.92.const.001": 14.7221,
            "alpha101.92.const.002": 18.8683,
            "alpha101.92.const.003": 7.58555,
            "alpha101.92.const.004": 6.94024,
            "alpha101.92.const.005": 6.80584,
        },
    ),
    93: AlphaProgramDefinition(
        alpha_id=93,
        program=alpha_093_program,
        tunable_constants={
            "alpha101.93.const.001": 17.4193,
            "alpha101.93.const.002": 19.848,
            "alpha101.93.const.003": 7.54455,
            "alpha101.93.const.004": 0.524434,
            "alpha101.93.const.005": 0.524434,
            "alpha101.93.const.006": 2.77377,
            "alpha101.93.const.007": 16.2664,
        },
    ),
    94: AlphaProgramDefinition(
        alpha_id=94,
        program=alpha_094_program,
        tunable_constants={
            "alpha101.94.const.001": 11.5783,
            "alpha101.94.const.002": 19.6462,
            "alpha101.94.const.003": 4.02992,
            "alpha101.94.const.004": 18.0926,
            "alpha101.94.const.005": 2.70756,
        },
    ),
    95: AlphaProgramDefinition(
        alpha_id=95,
        program=alpha_095_program,
        tunable_constants={
            "alpha101.95.const.001": 12.4105,
            "alpha101.95.const.002": 19.1351,
            "alpha101.95.const.003": 19.1351,
            "alpha101.95.const.004": 12.8742,
            "alpha101.95.const.005": 5.0,
            "alpha101.95.const.006": 11.7584,
        },
    ),
    96: AlphaProgramDefinition(
        alpha_id=96,
        program=alpha_096_program,
        tunable_constants={
            "alpha101.96.const.001": 3.83878,
            "alpha101.96.const.002": 4.16783,
            "alpha101.96.const.003": 8.38151,
            "alpha101.96.const.004": 7.45404,
            "alpha101.96.const.005": 4.13242,
            "alpha101.96.const.006": 3.65459,
            "alpha101.96.const.007": 12.6556,
            "alpha101.96.const.008": 14.0365,
            "alpha101.96.const.009": 13.4143,
        },
    ),
    97: AlphaProgramDefinition(
        alpha_id=97,
        program=alpha_097_program,
        tunable_constants={
            "alpha101.97.const.001": 0.721001,
            "alpha101.97.const.002": 0.721001,
            "alpha101.97.const.003": 3.3705,
            "alpha101.97.const.004": 20.4523,
            "alpha101.97.const.005": 7.87871,
            "alpha101.97.const.006": 17.255,
            "alpha101.97.const.007": 4.97547,
            "alpha101.97.const.008": 18.5925,
            "alpha101.97.const.009": 15.7152,
            "alpha101.97.const.010": 6.71659,
        },
    ),
    98: AlphaProgramDefinition(
        alpha_id=98,
        program=alpha_098_program,
        tunable_constants={
            "alpha101.98.const.001": 26.4719,
            "alpha101.98.const.002": 4.58418,
            "alpha101.98.const.003": 7.18088,
            "alpha101.98.const.004": 20.8187,
            "alpha101.98.const.005": 8.62571,
            "alpha101.98.const.006": 6.95668,
            "alpha101.98.const.007": 8.07206,
        },
    ),
    99: AlphaProgramDefinition(
        alpha_id=99,
        program=alpha_099_program,
        tunable_constants={
            "alpha101.99.const.001": 19.8975,
            "alpha101.99.const.002": 19.8975,
            "alpha101.99.const.003": 8.8136,
            "alpha101.99.const.004": 6.28259,
        },
    ),
    100: AlphaProgramDefinition(
        alpha_id=100,
        program=alpha_100_program,
        tunable_constants={
            "alpha101.100.const.001": 1.5,
            "alpha101.100.const.002": 5.0,
            "alpha101.100.const.003": 30.0,
        },
    ),
    101: AlphaProgramDefinition(
        alpha_id=101,
        program=alpha_101_program,
        tunable_constants={
            "alpha101.101.const.001": 0.001,
        },
    ),
}

__all__ = [
    "AlphaProgramDefinition",
    "ALPHA_PROGRAM_DEFINITIONS",
    "alpha_001_program",
    "alpha_002_program",
    "alpha_003_program",
    "alpha_004_program",
    "alpha_005_program",
    "alpha_006_program",
    "alpha_007_program",
    "alpha_008_program",
    "alpha_009_program",
    "alpha_010_program",
    "alpha_011_program",
    "alpha_012_program",
    "alpha_013_program",
    "alpha_014_program",
    "alpha_015_program",
    "alpha_016_program",
    "alpha_017_program",
    "alpha_018_program",
    "alpha_019_program",
    "alpha_020_program",
    "alpha_021_program",
    "alpha_022_program",
    "alpha_023_program",
    "alpha_024_program",
    "alpha_025_program",
    "alpha_026_program",
    "alpha_027_program",
    "alpha_028_program",
    "alpha_029_program",
    "alpha_030_program",
    "alpha_031_program",
    "alpha_032_program",
    "alpha_033_program",
    "alpha_034_program",
    "alpha_035_program",
    "alpha_036_program",
    "alpha_037_program",
    "alpha_038_program",
    "alpha_039_program",
    "alpha_040_program",
    "alpha_041_program",
    "alpha_042_program",
    "alpha_043_program",
    "alpha_044_program",
    "alpha_045_program",
    "alpha_046_program",
    "alpha_047_program",
    "alpha_048_program",
    "alpha_049_program",
    "alpha_050_program",
    "alpha_051_program",
    "alpha_052_program",
    "alpha_053_program",
    "alpha_054_program",
    "alpha_055_program",
    "alpha_056_program",
    "alpha_057_program",
    "alpha_058_program",
    "alpha_059_program",
    "alpha_060_program",
    "alpha_061_program",
    "alpha_062_program",
    "alpha_063_program",
    "alpha_064_program",
    "alpha_065_program",
    "alpha_066_program",
    "alpha_067_program",
    "alpha_068_program",
    "alpha_069_program",
    "alpha_070_program",
    "alpha_071_program",
    "alpha_072_program",
    "alpha_073_program",
    "alpha_074_program",
    "alpha_075_program",
    "alpha_076_program",
    "alpha_077_program",
    "alpha_078_program",
    "alpha_079_program",
    "alpha_080_program",
    "alpha_081_program",
    "alpha_082_program",
    "alpha_083_program",
    "alpha_084_program",
    "alpha_085_program",
    "alpha_086_program",
    "alpha_087_program",
    "alpha_088_program",
    "alpha_089_program",
    "alpha_090_program",
    "alpha_091_program",
    "alpha_092_program",
    "alpha_093_program",
    "alpha_094_program",
    "alpha_095_program",
    "alpha_096_program",
    "alpha_097_program",
    "alpha_098_program",
    "alpha_099_program",
    "alpha_100_program",
    "alpha_101_program",
]
