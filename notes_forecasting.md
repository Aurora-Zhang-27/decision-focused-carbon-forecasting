# Forecasting 设置（TFT）

- 数据：ERCOT / NYISO / PJM, 2020-01-01 ~ 2021-12-31，1h 分辨率
- 目标：system-level carbon intensity (kgCO2/MWh)
- 预测窗口：encoder_len=96, decoder_len=12
- 模型：Temporal Fusion Transformer
  - hidden_size=96
  - attention_head_size=4
  - hidden_continuous_size=32
  - dropout=0.2
  - quantiles = [0.1, 0.5, 0.9]
- 特征：
  - 时间特征：hour_sin/hour_cos, dow_sin/dow_cos, is_weekend, is_holiday
  - rolling：y_roll_mean_24, y_roll_std_24, y_diff_1
  - 标签 valley 标记：is_valley_y
- 训练：
  - optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
  - scheduler: ReduceLROnPlateau
  - max_epochs=10（fast_mode）
  - batch_size=256, CPU 训练
- 评估指标：MAE, MSE, WAPE, sMAPE, MAPE(|y|>=50), MdAPE(|y|>=50)
- 基线：
  - Naive-last: 复制 encoder 最后一个点
  - Hour-mean: 按小时的全局平均

# Decision simulation (stage 1)

- 下游任务：给定未来 12h 碳强度路径，选择一段固定负荷的执行时段，使加权碳排放最小。
- Oracle：使用真实未来碳强度。
- TFT：用 tft 预测的 12h 路径做相同优化。
- Uniform：在可行窗口内均匀随机（或简单均匀）选时段。
- 评价指标：
  - regret_pred / regret_uniform
  - ratio_pred / ratio_uniform (cost_pred / cost_oracle)
- 结论摘要：
  - ERCOT: ratio_pred ~1.0005 vs 1.163
  - NYISO: ratio_pred ~1.006 vs 1.088
  - PJM:   ratio_pred ~1.03  vs 1.07