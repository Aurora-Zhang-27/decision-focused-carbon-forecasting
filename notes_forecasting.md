# Forecasting setup (TFT)

- Data: ERCOT / NYISO / PJM, 2020-01-01 ~ 2021-12-31, 1h resolution
- Target: system-level carbon intensity (kgCO2/MWh)
- Forecast window: encoder_len=96, decoder_len=12
- Model: Temporal Fusion Transformer
  - hidden_size=96
  - attention_head_size=4
  - hidden_continuous_size=32
  - dropout=0.2
  - quantiles = [0.1, 0.5, 0.9]
- Features:
  - Time features: hour_sin/hour_cos, dow_sin/dow_cos, is_weekend, is_holiday
  - Rolling: y_roll_mean_24, y_roll_std_24, y_diff_1
  - Valley label flag: is_valley_y
- Training:
  - optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
  - scheduler: ReduceLROnPlateau
  - max_epochs=10 (fast_mode)
  - batch_size=256, CPU training
- Metrics: MAE, MSE, WAPE, sMAPE, MAPE(|y|>=50), MdAPE(|y|>=50)
- Baselines:
  - Naive-last: repeat the last encoder point
  - Hour-mean: global mean by hour of day

# Decision simulation (stage 1)

- Downstream task: given the 12h carbon-intensity path ahead, choose a run window for a fixed load to minimize weighted carbon emissions.
- Oracle: uses the true future carbon intensity.
- TFT: same optimization using the TFT-predicted 12h path.
- Uniform: pick a window uniformly at random (or evenly) within the feasible window.
- Metrics:
  - regret_pred / regret_uniform
  - ratio_pred / ratio_uniform (cost_pred / cost_oracle)
- Summary of findings:
  - ERCOT: ratio_pred ~1.0005 vs 1.163
  - NYISO: ratio_pred ~1.006 vs 1.088
  - PJM:   ratio_pred ~1.03  vs 1.07
