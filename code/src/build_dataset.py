# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet


def build_tsd(df: pd.DataFrame, dataset_cfg: dict):
    """
    Build train/validation TimeSeriesDataSet objects from a DataFrame.

    All column names are taken from `dataset_cfg`.
    Returns:
        (train_tsd, val_tsd)
    """
    # ====== Read config with defaults ======
    time_col   = dataset_cfg.get("datetime_col", "UTC time")
    target_col = dataset_cfg.get("target_col", "carbon_intensity")
    group_col  = dataset_cfg.get("group_col", "region")
    region_def = dataset_cfg.get("region_default", None)

    max_enc = int(dataset_cfg.get("max_encoder_length", 168))
    max_pred = int(dataset_cfg.get("max_prediction_length", 24))

    known_reals   = list(dataset_cfg.get("known_reals", []))
    unknown_reals = list(dataset_cfg.get("unknown_reals", []))

    # ====== Optional region filtering ======
    _df = df.copy()
    if region_def is not None and group_col in _df.columns:
        _df = _df[_df[group_col] == region_def].copy()

    # ====== Basic checks ======
    if time_col not in _df.columns:
        raise KeyError(f"Time column {time_col!r} not found in DataFrame.")
    if target_col not in _df.columns:
        raise KeyError(f"Target column {target_col!r} not found in DataFrame.")
    if group_col not in _df.columns:
        # If no group column is provided, create a constant group
        _df[group_col] = "ALL"

    # ====== Handle timestamps and sorting ======
    _df[time_col] = pd.to_datetime(_df[time_col], utc=True, errors="coerce")
    _df = _df.dropna(subset=[time_col, target_col])
    _df = _df.sort_values([group_col, time_col])

    # Create consecutive time_idx within each group
    _df["time_idx"] = _df.groupby(group_col).cumcount().astype(int)

    # ====== Split train / validation by time ======
    unique_times = np.array(sorted(_df[time_col].unique()))
    if len(unique_times) < max_enc + max_pred + 10:
        # Not enough data: fall back to a simple 90% / 10% split
        val_ratio = 0.1
    else:
        val_ratio = 0.15

    cut_idx = max(int(len(unique_times) * (1 - val_ratio)), 1)
    split_time = unique_times[cut_idx]

    train_df = _df[_df[time_col] <= split_time].copy()
    val_df   = _df[_df[time_col] >  split_time].copy()

    # If validation set is still too small, enlarge it
    if len(val_df) < max_pred * 10:
        cut_idx = max(int(len(unique_times) * 0.85), 1)
        split_time = unique_times[cut_idx]
        train_df = _df[_df[time_col] <= split_time].copy()
        val_df   = _df[_df[time_col] >  split_time].copy()

    print(f"[build_tsd] Split data: train={len(train_df)}, val={len(val_df)}")

    # ====== Columns for TimeSeriesDataSet ======
    # time_idx is usually treated as a known real
    tv_known_reals   = list(dict.fromkeys(known_reals + ["time_idx"]))  # remove duplicates
    tv_unknown_reals = list(dict.fromkeys([target_col] + unknown_reals))

    # ====== Build training dataset ======
    # Use min_encoder_length=1 to avoid "filters remove all entries" assertion
    train_tsd = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=target_col,
        group_ids=[group_col],
        max_encoder_length=max_enc,
        max_prediction_length=max_pred,
        time_varying_known_reals=tv_known_reals,
        time_varying_unknown_reals=tv_unknown_reals,
        static_categoricals=[group_col],
        allow_missing_timesteps=True,
        min_encoder_length=1,
        min_prediction_length=1,
    )

    # ====== Build validation dataset (from training spec) ======
    try:
        val_tsd = TimeSeriesDataSet.from_dataset(
            train_tsd,
            val_df,
            predict=False,
            stop_randomization=True,
        )
    except AssertionError:
        # If validation filtering is too strict and becomes empty, fall back
        # to using the tail of the training set as validation.
        print(
            "[build_tsd] Validation set became empty after filtering; "
            "falling back to using the tail of the training set as validation."
        )
        tail_cut = int(len(train_df) * 0.1)
        fallback_val = train_df.sort_values([group_col, time_col]).tail(
            max(tail_cut, max_pred * 10)
        )
        val_tsd = TimeSeriesDataSet.from_dataset(
            train_tsd,
            fallback_val,
            predict=False,
            stop_randomization=True,
        )

    return train_tsd, val_tsd