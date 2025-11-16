# src/export_forecasts.py
"""
Export multi-step forecasts from a trained TFT model on the validation set.

This script loads:
  - a config file (JSON),
  - the best checkpoint from a given run directory,

and then:
  - rebuilds the dataset and validation dataloader (consistent with train_tft),
  - runs the model on the validation set,
  - saves per-sample, per-horizon predictions (and true values) to
    `forecasts_val.parquet`.

Example:
    python -m src.export_forecasts \
        --config configs/ercot.json \
        --run_dir results_ercot/default/20251108-024136
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from src.data_loader import load_and_merge
from src.build_dataset import build_tsd


# ----------------------------------------------------------------------
# Utility functions (kept consistent with train_tft)
# ----------------------------------------------------------------------


def _safe_inverse(y_tensor: torch.Tensor, target_scale):
    """
    Invert normalization / scaling applied by the TimeSeriesDataSet.

    This is compatible with GroupNormalizer / EncoderNormalizer, where
    target_scale is typically stored in batch["target_scale"].

    Args:
        y_tensor: Tensor of shape [B, dec_len] or [B, dec_len, n_q].
        target_scale: batch["target_scale"], shape [B, scale_dim] (1 or 2).

    Returns:
        Tensor of the same shape as y_tensor, mapped back to original scale.
    """
    if target_scale is None:
        return y_tensor

    if isinstance(target_scale, (list, tuple)):
        target_scale = target_scale[0]

    # Mean + std scaling
    if torch.is_tensor(target_scale) and target_scale.size(-1) >= 2:
        loc, std = target_scale[..., 0], target_scale[..., 1]
        while loc.dim() < y_tensor.dim():
            loc, std = loc.unsqueeze(-1), std.unsqueeze(-1)
        return y_tensor * std + loc

    # Std-only scaling
    if torch.is_tensor(target_scale) and target_scale.size(-1) == 1:
        std = target_scale[..., 0]
        while std.dim() < y_tensor.dim():
            std = std.unsqueeze(-1)
        return y_tensor * std

    return y_tensor


def _median_quantile(yhat: torch.Tensor) -> torch.Tensor:
    """
    Extract a point forecast from quantile outputs by taking the median quantile.

    Args:
        yhat: Tensor of shape [B, dec_len, n_q].

    Returns:
        Tensor of shape [B, dec_len] representing the median-quantile forecast.
    """
    if yhat.dim() == 3 and yhat.size(-1) > 1:
        mid = yhat.size(-1) // 2
        return yhat[..., mid]
    return yhat.squeeze(-1)


# ----------------------------------------------------------------------
# Lightweight Lightning wrapper to reuse the checkpoint
# ----------------------------------------------------------------------


class LitTFT(pl.LightningModule):
    """
    Minimal LightningModule wrapper around a TemporalFusionTransformer.

    This matches the structure used in training so that we can reload
    checkpoints easily and then access the underlying TFT model.
    """

    def __init__(self, tft_model, learning_rate=1e-3, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.tft = tft_model
        self.loss_fn = QuantileLoss(quantiles=list(quantiles))
        self.save_hyperparameters(ignore=["tft_model", "loss_fn"])

    def forward(self, batch):
        x, _ = batch
        out = self.tft(x)
        if isinstance(out, dict):
            out = out.get("prediction", out.get("output", None))
        if isinstance(out, (list, tuple)):
            out = out[0]
        if not torch.is_tensor(out):
            raise TypeError(f"TFT forward must return Tensor, got {type(out)}")
        return out


# ----------------------------------------------------------------------
# Dataset preparation (mirrors train_tft preprocessing)
# ----------------------------------------------------------------------


def prepare_datasets(cfg):
    """
    Build TimeSeriesDataSet objects for training and validation
    using the same preprocessing logic as in train_tft.

    Steps:
      1) Load and merge raw data (forecast, weather, emissions).
      2) Add time-based features (hour-of-day, day-of-week, weekend, holiday).
      3) Clip extreme target values.
      4) Add simple rolling statistics and target-difference features.
      5) Mark low-carbon "valley" periods.
      6) Extend known_reals with newly constructed features.
      7) Call build_tsd to obtain (train_tsd, val_tsd).
    """
    dataset_cfg = cfg["dataset"]
    tgt = dataset_cfg["target_col"]

    # 1) Load raw data
    df, _meta = load_and_merge(cfg)
    print(f"[Data] Loaded merged DataFrame with shape: {df.shape}")

    # 2) Time features
    utc = pd.to_datetime(df[dataset_cfg["datetime_col"]], utc=True)

    df["hour_sin"] = np.sin(2 * np.pi * utc.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * utc.dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * utc.dt.weekday / 7)
    df["dow_cos"] = np.cos(2 * np.pi * utc.dt.weekday / 7)
    df["is_weekend"] = (utc.dt.weekday >= 5).astype(np.int8)

    # US holiday calendar (best-effort; fall back to 0 if unavailable)
    try:
        from pandas.tseries.holiday import USFederalHolidayCalendar

        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=utc.min().date(), end=utc.max().date())
        holiday_dates = set(holidays.date)
        df["is_holiday"] = pd.Series(utc.dt.date).isin(holiday_dates).astype(np.int8)
    except Exception:
        df["is_holiday"] = 0

    # 3) Light clipping of extreme target values
    df[tgt] = df[tgt].clip(lower=0, upper=df[tgt].quantile(0.999))

    # 4) Rolling statistics
    group_col = dataset_cfg.get("group_col", None)
    sort_cols = [dataset_cfg["datetime_col"]]
    if group_col is not None:
        sort_cols = [group_col] + sort_cols
    df = df.sort_values(sort_cols)

    if group_col is not None:
        g = df.groupby(group_col)[tgt]
        df["y_roll_mean_24"] = g.transform(
            lambda s: s.shift(1).rolling(24, min_periods=1).mean()
        )
        df["y_roll_std_24"] = g.transform(
            lambda s: s.shift(1).rolling(24, min_periods=1).std()
        )
        df["y_diff_1"] = g.transform(lambda s: s - s.shift(1))
    else:
        s = df[tgt]
        df["y_roll_mean_24"] = s.shift(1).rolling(24, min_periods=1).mean()
        df["y_roll_std_24"] = s.shift(1).rolling(24, min_periods=1).std()
        df["y_diff_1"] = s - s.shift(1)

    for col in ["y_roll_mean_24", "y_roll_std_24", "y_diff_1"]:
        df[col] = df[col].fillna(df[col].median())

    # 5) Valley indicator (low-CI periods)
    q20 = df[tgt].quantile(0.2)
    df["is_valley_y"] = (df[tgt] <= q20).astype("int8")

    # 6) Add new features to known_reals
    dataset_cfg.setdefault("known_reals", [])
    if not isinstance(dataset_cfg["known_reals"], list):
        dataset_cfg["known_reals"] = list(dataset_cfg["known_reals"])

    for k in [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "is_holiday",
        "is_valley_y",
        "y_roll_mean_24",
        "y_roll_std_24",
        "y_diff_1",
    ]:
        if k not in dataset_cfg["known_reals"]:
            dataset_cfg["known_reals"].append(k)

    # 7) Build TimeSeriesDataSet objects
    train_tsd, val_tsd = build_tsd(df, dataset_cfg)
    print(f"[TSD] Train samples: {len(train_tsd)}, Val samples: {len(val_tsd)}")

    return df, train_tsd, val_tsd


# ----------------------------------------------------------------------
# Main export logic
# ----------------------------------------------------------------------


def export_forecasts(cfg_path: str, run_dir: str):
    run_dir = Path(run_dir).expanduser().resolve()
    print(f"[Run] Using run_dir: {run_dir}")

    # 1) Load config
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    train_cfg = cfg.get("training", {})
    qtls = train_cfg.get("quantiles", [0.1, 0.5, 0.9])

    # 2) Load metadata about the best checkpoint
    best_meta_path = run_dir / "best_model.json"
    if not best_meta_path.exists():
        raise FileNotFoundError(f"best_model.json not found: {best_meta_path}")

    with open(best_meta_path, "r") as f:
        best_meta = json.load(f)

    ckpt_path = best_meta.get("best_ckpt") or best_meta.get("best_model_path")
    if ckpt_path is None:
        raise ValueError(
            f"best_model.json does not contain 'best_ckpt' or 'best_model_path': {best_meta}"
        )

    ckpt_path = Path(ckpt_path).expanduser().resolve()
    print(f"[Checkpoint] Using ckpt: {ckpt_path}")

    # 3) Build datasets and validation dataloader
    df, train_tsd, val_tsd = prepare_datasets(cfg)

    batch_size = cfg["trainer"].get("batch_size", 256)
    num_workers = max(2, os.cpu_count() // 2)
    val_dl = val_tsd.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=max(1, num_workers // 2),
        persistent_workers=False,
        pin_memory=False,
    )

    # 4) Rebuild TFT core with same configuration and load checkpoint
    tft_core = TemporalFusionTransformer.from_dataset(
        train_tsd,
        hidden_size=model_cfg["hidden_size"],
        attention_head_size=model_cfg["attention_head_size"],
        dropout=model_cfg["dropout"],
        hidden_continuous_size=model_cfg["hidden_continuous_size"],
        loss=QuantileLoss(quantiles=qtls),
        learning_rate=train_cfg.get("learning_rate", 1e-3),
        log_interval=10,
    )

    lit_model = LitTFT.load_from_checkpoint(
        ckpt_path,
        tft_model=tft_core,
        learning_rate=train_cfg.get("learning_rate", 1e-3),
        map_location="cpu",
    )
    tft_model = lit_model.tft
    tft_model.eval()
    tft_model.to("cpu")

    # 5) Loop over validation set and collect per-sample forecasts
    rows = []
    sample_id = 0

    with torch.no_grad():
        for batch in val_dl:
            x, y = batch
            y_std = y[0] if isinstance(y, (list, tuple)) else y

            out = tft_model(x)
            if isinstance(out, dict):
                out = out.get("prediction", out.get("output", None))
            if isinstance(out, (list, tuple)):
                out = out[0]
            if not torch.is_tensor(out):
                raise TypeError(
                    f"TFT forward must return Tensor-like output, got {type(out)}"
                )

            # [B, dec_len] median-quantile prediction
            y_hat = _median_quantile(out)
            scale = x.get("target_scale", None)

            # Map back to original scale
            y_pred = _safe_inverse(y_hat, scale).detach().cpu().numpy()  # [B, H]
            y_true = _safe_inverse(y_std, scale).detach().cpu().numpy()  # [B, H]

            B, H = y_pred.shape
            for b in range(B):
                for h in range(H):
                    rows.append(
                        {
                            "sample_id": int(sample_id + b),
                            "horizon": int(h + 1),
                            "y_true": float(y_true[b, h]),
                            "y_pred": float(y_pred[b, h]),
                        }
                    )

            sample_id += B

    out_path = run_dir / "forecasts_val.parquet"
    df_out = pd.DataFrame(rows)
    df_out.to_parquet(out_path, index=False)
    print(f"[Save] Per-sample forecasts written to: {out_path}")
    print(df_out.head())


def main():
    parser = argparse.ArgumentParser(
        description="Export TFT forecasts on the validation set for decision-focused evaluation."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file, e.g. configs/ercot.json.",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the run directory, e.g. results_ercot/default/20251108-024136.",
    )
    args = parser.parse_args()

    pl.seed_everything(2025, workers=True)
    export_forecasts(args.config, args.run_dir)


if __name__ == "__main__":
    main()