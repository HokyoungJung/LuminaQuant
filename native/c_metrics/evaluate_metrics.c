#include <math.h>
#include <stddef.h>

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

static double max_drawdown(const double *series, int n) {
    if (series == NULL || n <= 0) {
        return 0.0;
    }

    double peak = series[0];
    double max_dd = 0.0;
    for (int i = 0; i < n; ++i) {
        const double value = series[i];
        if (value > peak) {
            peak = value;
        }
        if (peak > 0.0) {
            const double dd = (peak - value) / peak;
            if (dd > max_dd) {
                max_dd = dd;
            }
        }
    }
    return max_dd;
}

EXPORT int evaluate_metrics(
    const double *total_series,
    int n,
    int annual_periods,
    double *out_sharpe,
    double *out_cagr,
    double *out_max_dd
) {
    if (total_series == NULL || out_sharpe == NULL || out_cagr == NULL || out_max_dd == NULL) {
        return 2;
    }

    if (n < 2) {
        *out_sharpe = -999.0;
        *out_cagr = 0.0;
        *out_max_dd = 0.0;
        return 0;
    }

    if (annual_periods <= 0) {
        annual_periods = 252;
    }

    double mean_r = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        const double prev = total_series[i];
        const double next = total_series[i + 1];
        const double den = (prev == 0.0) ? 1.0 : prev;
        mean_r += (next - prev) / den;
    }
    mean_r /= (double)(n - 1);

    double var_r = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        const double prev = total_series[i];
        const double next = total_series[i + 1];
        const double den = (prev == 0.0) ? 1.0 : prev;
        const double ret = (next - prev) / den;
        const double diff = ret - mean_r;
        var_r += diff * diff;
    }
    if (n > 2) {
        var_r /= (double)(n - 2);
    }

    const double std_r = (var_r > 0.0) ? sqrt(var_r) : 0.0;
    *out_sharpe = (std_r > 0.0) ? (mean_r / std_r) * sqrt((double)annual_periods) : -999.0;

    const double initial = total_series[0];
    const double final = total_series[n - 1];
    if (initial <= 0.0) {
        *out_cagr = 0.0;
    } else {
        const double years = (double)n / (double)annual_periods;
        if (years <= 0.0) {
            *out_cagr = 0.0;
        } else {
            *out_cagr = pow(final / initial, 1.0 / years) - 1.0;
        }
    }

    *out_max_dd = max_drawdown(total_series, n);
    return 0;
}
