# STEP2_TRAIN_INTERNAL_HOLDOUT_V50

Train-only bounded V5.0 artifact. SWMM, Validation and Final were not accessed.

```json
{
  "contract": "STEP2_TRAIN_INTERNAL_HOLDOUT_V50_TRAIN_ONLY",
  "split": "deterministic SHA256 complete-group split on group identity; no row split",
  "fit_group_count": 240,
  "holdout_group_count": 48,
  "fit_metrics": {
    "D2": {
      "groups": 120,
      "rank": 0.2133778571732396,
      "pairwise": 0.3752599482610845,
      "sign": 0.4240228370663154,
      "top1": 11,
      "mean_regret_m3": 11756.799479166666,
      "max_regret_m3": 47191.65625,
      "spread_ratio": 1.3644460619694935e-08,
      "tfv_mae_m3": 6236.361423577202
    },
    "D3": {
      "groups": 120,
      "rank": 0.10040157919675659,
      "pairwise": 0.5211309523809524,
      "sign": 0.8541666666666666,
      "top1": 17,
      "mean_regret_m3": 81639.49505208334,
      "max_regret_m3": 306374.0,
      "spread_ratio": 4.146092391089107e-07,
      "tfv_mae_m3": 93786.34294331868
    }
  },
  "holdout_metrics": {
    "D2": {
      "groups": 24,
      "rank": 0.2653490669840107,
      "pairwise": 0.39597250568320125,
      "sign": 0.38309727711901626,
      "top1": 3,
      "mean_regret_m3": 11368.463541666666,
      "max_regret_m3": 37200.25,
      "spread_ratio": 1.2981534289278519e-08,
      "tfv_mae_m3": 7692.213639153375
    },
    "D3": {
      "groups": 24,
      "rank": 0.2531382836808311,
      "pairwise": 0.5863095238095238,
      "sign": 0.8177083333333334,
      "top1": 5,
      "mean_regret_m3": 85884.84375,
      "max_regret_m3": 262837.0,
      "spread_ratio": 3.3558017415511e-07,
      "tfv_mae_m3": 96493.73678588867
    }
  },
  "holdout_d3_magnitude": {
    "small": {
      "count": 64,
      "q33": 50246.5,
      "q67": 112447.5,
      "response_ratio": 1.2881885810264539e-06,
      "mae_m3": 26805.561630249023,
      "bias_m3": 18213.124221801758,
      "rank": 0.30638548567386237,
      "pairwise": 0.6351851851851852,
      "sign": 0.71875
    },
    "medium": {
      "count": 64,
      "q33": 50246.5,
      "q67": 112447.5,
      "response_ratio": 4.893932164802211e-07,
      "mae_m3": 84214.42967224121,
      "bias_m3": 77165.56407165527,
      "rank": 0.13363384751883034,
      "pairwise": 0.5574074074074074,
      "sign": 0.953125
    },
    "large": {
      "count": 64,
      "q33": 50246.5,
      "q67": 112447.5,
      "response_ratio": 2.385505072800649e-07,
      "mae_m3": 178461.21905517578,
      "bias_m3": 65685.55920410156,
      "rank": -0.16984126984126982,
      "pairwise": 0.4148148148148148,
      "sign": 0.78125
    }
  },
  "hydraulic_metrics": {
    "groups": 6,
    "depth_rmse": 0.2699100077152252,
    "flooding_rate_rmse": 0.5803422331809998,
    "storage_rmse": 365.68402099609375,
    "managed_flow_rmse": 4.663137435913086
  },
  "validation_accessed": false,
  "final_accessed": false
}
```
