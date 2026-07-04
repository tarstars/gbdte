# OWID country-panel family (positive-result family)

Bootstrap (10 resamples of the training set, fixed forward test window), test RMSE
mean±std. All four datasets are country-year panels from Our World in Data with country
identity WITHHELD (features = continent + initial level), standard 0.75 temporal split,
log target where the process is multiplicative (mortality, GDP). All four were flagged
GREEN by the pre-training `extrapolation_gain` screen. Generated 2026-07-04.

Headline: GBDTE beats the standard constant-leaf boosters (XGBoost/LightGBM/CatBoost) on
3 of 4 and ties on the 4th (GDP), with 12-24% RMSE reductions; on maternal mortality it
is the single best model. Only peer is the manual global-detrend baseline.

## owid_childmort (extrap_gain 0.61) — WIN vs boosters
| model | RMSE |
|---|---|
| detrend_lgbm | 0.320 ± 0.011 |
| **gbdte_auto** | **0.323 ± 0.004** |
| xgb | 0.422 ± 0.012 |
| lgbm | 0.430 ± 0.013 |
| catboost | 0.504 ± 0.014 |
| gbdte_const | 1.151 ± 0.006 |

## owid_maternalmort (extrap_gain 0.24) — WIN (best overall)
| model | RMSE |
|---|---|
| **gbdte_auto** | **0.431 ± 0.010** |
| detrend_lgbm | 0.470 ± 0.013 |
| xgb | 0.491 ± 0.011 |
| lgbm | 0.516 ± 0.013 |
| catboost | 0.638 ± 0.009 |
| gbdte_const | 1.009 ± 0.010 |

## owid_lifeexp (extrap_gain 0.69) — WIN vs boosters
| model | RMSE |
|---|---|
| detrend_lgbm | 4.399 ± 0.081 |
| **gbdte_auto** | **4.691 ± 0.063** |
| xgb | 5.523 ± 0.140 |
| lgbm | 5.566 ± 0.116 |
| catboost | 6.530 ± 0.185 |
| gbdte_const | 14.52 ± 0.103 |

## owid_gdppcap (extrap_gain 0.34) — TIE vs boosters
| model | RMSE |
|---|---|
| xgb | 0.250 ± 0.005 |
| **gbdte_auto** | **0.252 ± 0.009** |
| detrend_lgbm | 0.261 ± 0.005 |
| lgbm | 0.267 ± 0.004 |
| catboost | 0.339 ± 0.006 |
| gbdte_const | 0.498 ± 0.005 |

## Summary (vs best standard booster)
| dataset | extrap_gain | gbdte | best booster | verdict |
|---|---|---|---|---|
| child mortality | 0.61 | 0.323 | 0.422 | WIN (−24%) |
| maternal mortality | 0.24 | 0.431 | 0.491 | WIN (−12%) |
| life expectancy | 0.69 | 4.691 | 5.523 | WIN (−15%) |
| GDP per capita | 0.34 | 0.252 | 0.250 | tie |
