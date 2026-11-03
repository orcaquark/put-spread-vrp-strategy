# VRP-Gated SPX Put Spread: Results

**Data:** 2008-01-02 to 2026-10-30  
**First tradable day:** 2012-01-03  
**In-sample:** 2012-01-03 to 2022-05-16 (2,610 days)  
**Holdout:** 2022-05-17 to 2026-10-30 (1,119 days)

## Test 1: Correctness and Premium

- **Accounting identity:** Passed (234 trades)
- **Settlement vs. SPX close:** Maximum absolute difference 0.0; AM-settled expiries in data: 0
- **PUT index replica:** Skipped (`put_index.parquet` not provided); 0 rolls
- **Direct premium:** Mean gap 0.02448, HAC t-statistic 7.76, share of years positive 1.00
- **Premium gap by gate status:**
  - Gate off: 0.026242139714325628
  - Gate on: 0.021404868874497874
- **Premium with overnight gap added to RV:** None

### Quote-Drop Fraction by Year

| Year | n_dropped | n_rows | fraction_dropped | flag |
|---|---:|---:|---:|---|
| 2012 | 0 | 3,323 | 0.00000 | False |
| 2013 | 0 | 5,330 | 0.00000 | False |
| 2014 | 1 | 5,804 | 0.00017 | False |
| 2015 | 11 | 5,599 | 0.00196 | False |
| 2016 | 45 | 5,749 | 0.00783 | False |
| 2017 | 75 | 5,585 | 0.01343 | False |
| 2018 | 5 | 5,710 | 0.00088 | False |
| 2019 | 45 | 5,513 | 0.00816 | False |
| 2020 | 111 | 5,454 | 0.02035 | True |
| 2021 | 157 | 5,372 | 0.02923 | False |
| 2022 | 142 | 5,495 | 0.02584 | False |
| 2023 | 8 | 5,643 | 0.00142 | False |
| 2024 | 2 | 5,678 | 0.00035 | False |
| 2025 | 15 | 5,707 | 0.00263 | False |
| 2026 | 4 | 4,798 | 0.00083 | False |

## Test 2: Lookahead

- **Extra one-day lag:**

  | Metric | Base | Lagged |
  |---|---:|---:|
  | n_trades | 231.0 | 237.0 |
  | avg_return_on_risk | 0.022804136874717603 | 0.025658422668249727 |
  | loss_frequency | 0.09523809523809523 | 0.08860759493670886 |
  | CAGR | 0.00490165128730613 | 0.005715679828420583 |
  | sortino | 0.44005770718858256 | 0.5303691717038846 |
  | max_drawdown | -0.035720893896631356 | -0.0369499150834407 |

  Sign flip in average return on risk: False

- **Truncation test:** All match = True (5 dates)

## Test 3: Variants at f = 1.0 (In-Sample)

### Portfolio-Level Performance

| variant | CAGR | ann_vol | sortino | max_drawdown | CVaR_5pct_daily | worst_month | skew | excess_kurtosis | n_trades | avg_pnl_pts | avg_return_on_risk | loss_frequency | worst_trade_pts | worst_trade_ror | alpha_annual | p_alpha | beta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gated spread | 0.0049 | 0.0146 | 0.4401 | -0.0357 | -0.0025 | -0.0149 | -2.1164 | 33.2054 | 231 | 4.0771 | 0.0228 | 0.0952 | -282.9000 | -1.0000 | -0.0017 | 0.5843 | 0.0541 |
| Ungated spread | 0.0138 | 0.0201 | 0.9199 | -0.0336 | -0.0035 | -0.0171 | -1.1623 | 22.8100 | 425 | 4.9038 | 0.0340 | 0.0776 | -311.3000 | -1.0000 | 0.0027 | 0.3949 | 0.0901 |
| VRP-only gate | 0.0073 | 0.0163 | 0.6004 | -0.0345 | -0.0027 | -0.0158 | -1.0862 | 33.8635 | 261 | 5.4682 | 0.0294 | 0.0843 | -311.3000 | -1.0000 | -0.0009 | 0.7686 | 0.0668 |
| Term-structure-only gate | 0.0126 | 0.0192 | 0.8681 | -0.0310 | -0.0034 | -0.0143 | -1.6732 | 23.6046 | 401 | 4.4160 | 0.0331 | 0.0798 | -282.9000 | -1.0000 | 0.0029 | 0.3603 | 0.0795 |

### Per-Trade Comparison

| variant | n_trades | avg_pnl_pts | avg_return_on_risk | loss_frequency | worst_trade_pts | worst_trade_ror | placebo_p_ror |
|---|---|---|---|---|---|---|---|
| gated | 231.0000 | 4.0771 | 0.0228 | 0.0952 | -282.9000 | -1.0000 | 0.8520 |
| ungated | 425.0000 | 4.9038 | 0.0340 | 0.0776 | -311.3000 | -1.0000 | n/a |
| vrp_only | 261.0000 | 5.4682 | 0.0294 | 0.0843 | -311.3000 | -1.0000 | 0.6320 |
| ts_only | 401.0000 | 4.4160 | 0.0331 | 0.0798 | -282.9000 | -1.0000 | 0.4040 |

## Test 4: Cost Sensitivity (Gated, In-Sample)

| f | fee_x | net_pnl_usd | CAGR | sortino | n_trades | avg_return_on_risk |
|---|---|---|---|---|---|---|
| 0.2500 | 0 | 5825.6819 | 0.0055 | 0.4901 | 231 | 0.0252 |
| 0.2500 | 1 | 5797.1186 | 0.0055 | 0.4878 | 231 | 0.0252 |
| 0.2500 | 2 | 5768.5630 | 0.0054 | 0.4855 | 231 | 0.0252 |
| 0.5000 | 0 | 5625.2676 | 0.0053 | 0.4742 | 231 | 0.0244 |
| 0.5000 | 1 | 5596.7850 | 0.0053 | 0.4719 | 231 | 0.0244 |
| 0.5000 | 2 | 5568.3099 | 0.0052 | 0.4696 | 231 | 0.0244 |
| 1.0000 | 0 | 5227.3602 | 0.0049 | 0.4424 | 231 | 0.0228 |
| 1.0000 | 1 | 5199.0375 | 0.0049 | 0.4401 | 231 | 0.0228 |
| 1.0000 | 2 | 5170.7224 | 0.0049 | 0.4378 | 231 | 0.0228 |

**Break-even fill fraction f\*:** In-sample inf

## Test 5: Parameter Grid (f = 1.0)

| dte | short_delta | long_delta | sortino | CVaR_5pct_daily | avg_return_on_risk | n_trades |
|---|---|---|---|---|---|---|
| 30 | 0.1500 | 0.0500 | 0.4165 | -0.0022 | 0.0192 | 231 |
| 30 | 0.1500 | 0.1000 | 0.2229 | -0.0030 | 0.0150 | 229 |
| 30 | 0.2000 | 0.0500 | 0.4401 | -0.0025 | 0.0228 | 231 |
| 30 | 0.2000 | 0.1000 | 0.3351 | -0.0034 | 0.0241 | 231 |
| 30 | 0.2500 | 0.0500 | 0.4872 | -0.0028 | 0.0275 | 231 |
| 30 | 0.2500 | 0.1000 | 0.4166 | -0.0038 | 0.0320 | 231 |

**Neighbor retention:**

```json
{
  "base_avg_ror": 0.022804136874717603,
  "neighbor_mean_avg_ror": 0.023595827223914786,
  "n_neighbors": 3,
  "retention": 1.0347169618191037
}
```

## Test 7: Stress Windows (Conservative Marks, In-Sample)

| window | worst_drawdown | spreads_open_in_window | spreads_at_max_loss | gate_on_going_in |
|---|---|---|---|---|
| aug_2015 | 0.0000 | 0 | 0 | False |
| feb_2018 | 0.0000 | 0 | 0 | True |
| q4_2018 | -0.0307 | 8 | 1 | True |
| covid_2020 | -0.0311 | 4 | 3 | True |
| bear_2022 | -0.0268 | 18 | 0 | True |

**Sizing check:**

```json
{
  "window_drawdowns": {
    "aug_2015": 0.0,
    "feb_2018": 0.0,
    "q4_2018": -0.03065391191611888,
    "covid_2020": -0.031082630789347365,
    "bear_2022": -0.026750256386798288
  },
  "worst": -0.031082630789347365,
  "L": 0.7770657697336841,
  "max_risk_per_trade": 0.048258463389594414,
  "passes": true
}
```

## Test 8: Inference (Gated, In-Sample, f = 1.0)

```json
{
  "alpha_annual": -0.0017356182742932268,
  "alpha_t": -0.5471450984917939,
  "p_alpha": 0.5842790509094942,
  "beta": 0.05414359235410841,
  "n": 2609,
  "boot_mean_ci_annual": [
    -0.0021750954863224373,
    0.013494141418296194
  ],
  "boot_alpha_ci_annual": [
    -0.007400877206614736,
    0.00452690910151825
  ],
  "sharpe_daily": 0.021648922617244538,
  "sharpe_annualized": 0.34366599238626366,
  "n_trials": 9,
  "sr_var_across_trials": 0.00011920490863461323,
  "deflated_sharpe": 0.5992608404197596
}
```

## Appendix A.1: VIX Regimes (In-Sample)

### Terciles

**Overall gate-on rate:** 0.368  
**Gate-on rate by regime:** `{'high': 0.5234375, 'low': 0.23692636072572038, 'mid': 0.6490610328638498}`

| regime | days | mean_gap | std_gap | gap_per_risk | mean_rel_gap |
|---|---|---|---|---|---|
| low | 1,874 | 0.0120 | 0.0162 | 0.7434 | 0.4957 |
| mid | 851 | 0.0260 | 0.0304 | 0.8565 | 0.5191 |
| high | 619 | 0.0608 | 0.0622 | 0.9774 | 0.5872 |

| regime | n | avg_ror | loss_freq | worst_ror |
|---|---|---|---|---|
| high | 79 | 0.0452 | 0.0633 | -1.0000 |
| low | 232 | 0.0373 | 0.0690 | -1.0000 |
| mid | 114 | 0.0197 | 0.1053 | -1.0000 |

### Top 20

**Overall gate-on rate:** 0.368  
**Gate-on rate by regime:** `{'rest': 0.38159187684245005, 'top20': 0.5335463258785943}`

| regime | days | mean_gap | std_gap | gap_per_risk | mean_rel_gap |
|---|---|---|---|---|---|
| rest | 3,047 | 0.0195 | 0.0255 | 0.7650 | 0.5092 |
| top20 | 297 | 0.0770 | 0.0801 | 0.9608 | 0.6150 |

| regime | n | avg_ror | loss_freq | worst_ror |
|---|---|---|---|---|
| rest | 382 | 0.0311 | 0.0838 | -1.0000 |
| top20 | 43 | 0.0596 | 0.0233 | -1.0000 |

## Benchmarks (In-Sample)

**Buy and hold SPX (price only):**

| Metric | Result |
|---|---:|
| CAGR | 0.11665817454670968 |
| ann_vol | 0.16655561165235874 |
| sortino | 1.0374994439325191 |
| max_drawdown | -0.3392495902426059 |
| CVaR_5pct_daily | -0.025622166541043923 |
| worst_month | -0.12511932083595656 |
| skew | -0.6286225199379215 |
| excess_kurtosis | 18.243563064877666 |
| beta | 0.9999999999999994 |
| alpha_annual | -8.255435426558492e-17 |
| p_alpha | 0.024472657539610823 |

---

_Holdout not run yet; kill criterion 8 and the final decision are pending._
