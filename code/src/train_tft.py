# ============================
# train_tft.py — CPU-stable TFT trainer
# (AdamW regularization, quiet checkpoint loading,
#  evaluation on original scale, baselines, metrics export)
# ============================
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import re
import argparse
import json
from pathlib import Path
from glob import glob
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
plt.switch_backend("Agg")  # ensure non-interactive plotting works

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from importlib.util import find_spec
from lightning.pytorch import Trainer as _PLTrainer

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

# Project-local utilities
from src.data_loader import load_and_merge          # -> df, meta
from src.build_dataset import build_tsd             # -> (train_tsd, val_tsd)
from src.decision_loss import decision_weighted_mae

# ------------ Global CPU / thread settings ------------
torch.set_num_threads(min(8, os.cpu_count()))
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

# Silence some unimportant warnings
warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", message=r".*Attribute 'loss' is an instance of `nn.Module`.*")
warnings.filterwarnings("ignore", message=r".*Attribute 'logging_metrics' is an instance of `nn.Module`.*")
warnings.filterwarnings("ignore", message=r".*MPS.*fall back.*")


# ===== Safe feature-importance plotting (no weird kwargs into forward) =====
def safe_plot_tft_importance(tft_model, val_loader, save_dir: Path, tag="tft"):
    """
    Robust wrapper around `plot_interpretation` to save feature-importance figures.

    It tries to handle multiple return types across pytorch-forecasting versions:
      - a single Figure,
      - a list/tuple of Figure/Axes,
      - a dict of name -> Figure/Axes/list.

    All figures are saved under `save_dir` on CPU, ignoring MPS fallback warnings.
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from lightning.pytorch import Trainer as _PLTrainer
    import warnings

    warnings.filterwarnings("ignore", message=r".*will fall back to run on the CPU.*MPS.*")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Move model to CPU if possible (some PF versions require this)
    try:
        tft_model.to("cpu")
    except Exception:
        pass

    # Use a simple CPU trainer for prediction / interpretation
    tmp_trainer = _PLTrainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    # Get raw outputs
    try:
        raw = tft_model.predict(val_loader, mode="raw", trainer=tmp_trainer)
    except TypeError:
        # older versions may not accept `trainer` kwarg
        raw = tft_model.predict(val_loader, mode="raw")

    # Convert raw outputs into interpretation dict
    try:
        interpretation = tft_model.interpret_output(raw, reduction="mean")
    except Exception:
        interpretation = None

    # Generic saver: recursively walk through any object and save all Figures/Axes
    def _save_any(obj, stem: str):
        if obj is None:
            return

        # Single Figure
        if isinstance(obj, Figure):
            obj.savefig(save_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
            plt.close(obj)
            return

        # Single Axes
        if isinstance(obj, Axes):
            fig = obj.figure
            if isinstance(fig, Figure):
                fig.savefig(save_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
                plt.close(fig)
            return

        # List / tuple
        if isinstance(obj, (list, tuple)):
            for i, o in enumerate(obj):
                _save_any(o, f"{stem}_{i}")
            return

        # Dict: keys may be strings describing components
        if isinstance(obj, dict):
            i = 0
            for k, v in obj.items():
                # sanitize key for filename
                key = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(k))[:40] or f"item{i}"
                _save_any(v, f"{stem}_{key}_{i}")
                i += 1
            return

        # Other types: ignore
        return

    # Try to plot interpretation
    try:
        ret = tft_model.plot_interpretation(interpretation)
        _save_any(ret, f"{tag}_interpretation")
    except Exception as e:
        print(f"⚠️ plot_interpretation not available: {e}")

    return interpretation


# --------- Minimal Lightning wrapper around TFT ---------
class LitTFT(pl.LightningModule):
    """
    Thin LightningModule wrapper to train a TemporalFusionTransformer
    with optional decision-focused loss.
    """

    def __init__(self, tft_model, cfg, learning_rate=None, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.tft = tft_model
        self.loss_fn = QuantileLoss(quantiles=list(quantiles))

        # Do not store large objects inside the checkpoint
        self.save_hyperparameters(ignore=["tft_model", "loss_fn", "cfg"])

        self.cfg = cfg

        # 1) Learning rate: prefer explicit value, otherwise read from config
        trainer_cfg = cfg.get("trainer", {})
        cfg_lr = trainer_cfg.get("learning_rate", 1e-3)
        self.lr = learning_rate if learning_rate is not None else cfg_lr

        # 2) Decision-focused loss switch
        decision_cfg = cfg.get("decision", {})
        self.use_decision_loss = decision_cfg.get("use_decision_loss", False)
        self.lambda_decision = decision_cfg.get("lambda_decision", 0.5)

    def _split_target(self, y):
        """
        Split target from (target, weight) tuple if present.
        """
        if isinstance(y, (list, tuple)):
            target = y[0]
            weight = y[1] if len(y) > 1 else None
        else:
            target = y
            weight = None
        return target, weight

    def _as_prediction_tensor(self, out):
        """
        Normalize TFT outputs into a single Tensor:
          - if dict, prefer 'prediction' or 'output',
          - if tuple/list, take index 0,
          - final output must be a Tensor.
        """
        if isinstance(out, dict):
            if "prediction" in out:
                out = out["prediction"]
            elif "output" in out:
                out = out["output"]
        if isinstance(out, (list, tuple)):
            out = out[0]
        if not torch.is_tensor(out):
            raise TypeError(f"TFT forward must return Tensor, got {type(out)}")
        return out

    def forward(self, batch):
        x, _ = batch
        out = self.tft(x)
        y_hat = self._as_prediction_tensor(out)
        return y_hat  # [B, dec_len, n_quantiles] or [B, dec_len, 1]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_std, _ = self._split_target(y)  # [B, dec_len]

        out = self.tft(x)
        y_pred_full = self._as_prediction_tensor(out)  # [B, dec_len, n_q]
        y_pred = _median_quantile(y_pred_full)         # [B, dec_len]

        if self.use_decision_loss:
            loss, mae, weighted_mae = decision_weighted_mae(
                y_pred, y_std, lambda_decision=self.lambda_decision
            )
            self.log("train_loss", loss, prog_bar=True)
            self.log("train_mae", mae, prog_bar=False)
            self.log("train_weighted_mae", weighted_mae, prog_bar=False)
        else:
            mae = torch.mean(torch.abs(y_pred - y_std))
            loss = mae
            self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.tft(x)
        y_hat = self._as_prediction_tensor(out)

        y, w = self._split_target(y)
        loss = self.loss_fn(y_hat, y) if w is None else self.loss_fn(y_hat, y, weights=w)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=y.size(0),
        )
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=False,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def _get_logger(save_dir: Path):
    """Prefer TensorBoardLogger, fall back to CSVLogger."""
    if find_spec("tensorboard") is not None:
        return TensorBoardLogger(save_dir=str(save_dir), name="logs")
    return CSVLogger(save_dir=str(save_dir), name="logs_csv")


# --------- Evaluation helpers ---------
def _safe_inverse(y_tensor: torch.Tensor, target_scale):
    """
    Invert the scaling applied by GroupNormalizer / EncoderNormalizer.

    Args:
        y_tensor: [B, dec_len] or [B, dec_len, n_q]
        target_scale: x["target_scale"], shape [B, scale_dim] (1 or 2)
    """
    if target_scale is None:
        return y_tensor

    if isinstance(target_scale, (list, tuple)):
        target_scale = target_scale[0]

    # mean + std scaling
    if torch.is_tensor(target_scale) and target_scale.size(-1) >= 2:
        loc, std = target_scale[..., 0], target_scale[..., 1]
        while loc.dim() < y_tensor.dim():
            loc, std = loc.unsqueeze(-1), std.unsqueeze(-1)
        return y_tensor * std + loc

    # std-only scaling
    if torch.is_tensor(target_scale) and target_scale.size(-1) == 1:
        std = target_scale[..., 0]
        while std.dim() < y_tensor.dim():
            std = std.unsqueeze(-1)
        return y_tensor * std

    return y_tensor


def _feature_idx(tsd, name: str) -> int:
    """
    Find feature index in decoder_cont given its name (assumes it is in tsd.reals).
    """
    if name not in tsd.reals:
        raise KeyError(f"{name} not in dataset.reals: {tsd.reals}")
    return tsd.reals.index(name)


def _hour_from_sin_cos(sin_v, cos_v):
    sin_v = np.asarray(sin_v, dtype=np.float64)
    cos_v = np.asarray(cos_v, dtype=np.float64)
    ang = np.arctan2(sin_v, cos_v)          # [-pi, pi]
    ang = np.where(ang < 0, ang + 2*np.pi, ang)
    return 24.0 * ang / (2 * np.pi)         # [0, 24)


def _smape(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(2.0 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + eps)))


def _wape(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.sum(np.abs(true)) + eps
    return float(np.sum(np.abs(pred - true)) / denom)


def _mdape(pred: np.ndarray, true: np.ndarray, tau: float = 0.0, eps: float = 1e-8) -> float:
    msk = np.abs(true) >= float(tau)
    if not np.any(msk):
        return float("nan")
    ape = np.abs((pred[msk] - true[msk]) / (true[msk] + eps))
    return float(np.median(ape))


def _agg_metrics(pred: np.ndarray, true: np.ndarray, tau: float = 50.0):
    mae = float(np.mean(np.abs(pred - true)))
    mse = float(np.mean((pred - true) ** 2))
    wape = _wape(pred, true)
    smap = _smape(pred, true)
    msk = np.abs(true) >= float(tau)
    mape = float(np.mean(np.abs((pred[msk] - true[msk]) / (true[msk])))) if np.any(msk) else float("nan")
    mdap = _mdape(pred, true, tau=tau)
    return {
        "MAE": mae,
        "MSE": mse,
        "WAPE": wape,
        "sMAPE": smap,
        f"MAPE(|y|>={tau:g})": mape,
        f"MdAPE(|y|>={tau:g})": mdap,
    }


def _median_quantile(yhat: torch.Tensor) -> torch.Tensor:
    """
    Extract median quantile from [B, dec_len, n_q] tensor.
    """
    if yhat.dim() == 3 and yhat.size(-1) > 1:
        mid = yhat.size(-1) // 2
        return yhat[..., mid]
    return yhat.squeeze(-1)


def evaluate_on_loader(tft_model, val_loader, tau=50.0):
    """
    Evaluate a TFT model on a dataloader in original (unscaled) units.
    """
    try:
        tft_model.to("cpu")
    except Exception:
        pass
    tft_model.eval()

    preds_list, trues_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            y_std = y[0] if isinstance(y, (list, tuple)) else y

            out = tft_model(x)
            if isinstance(out, dict):
                out = out.get("prediction", out.get("output", None))
            if isinstance(out, (list, tuple)):
                out = out[0]
            if not torch.is_tensor(out):
                raise TypeError(f"forward must return Tensor-like, got {type(out)}")

            y_hat = _median_quantile(out)
            scale = x.get("target_scale", None)

            y_pred_orig = _safe_inverse(y_hat, scale)
            y_true_orig = _safe_inverse(y_std, scale)

            y_pred_np = y_pred_orig.detach().cpu().numpy().reshape(-1).astype(np.float64)
            y_true_np = y_true_orig.detach().cpu().numpy().reshape(-1).astype(np.float64)

            msk = np.isfinite(y_pred_np) & np.isfinite(y_true_np)
            if not np.any(msk):
                continue
            preds_list.append(y_pred_np[msk])
            trues_list.append(y_true_np[msk])

    if not preds_list:
        return {"MAE": float("nan"), "MSE": float("nan"), f"MAPE(|y|>={tau:g})": float("nan")}

    pred = np.concatenate(preds_list, axis=0)
    true = np.concatenate(trues_list, axis=0)

    mae = float(np.mean(np.abs(pred - true)))
    mse = float(np.mean((pred - true) ** 2))
    msk_mape = np.abs(true) >= float(tau)
    mape = float(np.mean(np.abs((pred[msk_mape] - true[msk_mape]) / true[msk_mape]))) if np.any(msk_mape) else float("nan")

    smape = _smape(pred, true)
    wape = _wape(pred, true)
    mdape = _mdape(pred, true, tau=tau)

    return {
        "MAE": mae,
        "MSE": mse,
        f"MAPE(|y|>={tau:g})": mape,
        "sMAPE": smape,
        "WAPE": wape,
        f"MdAPE(|y|>={tau:g})": mdape,
    }


def eval_naive_last(val_loader, tau=50.0):
    """
    Naive-last baseline: predict each horizon using the last encoder target.
    """
    preds_list, trues_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            y_std = y[0] if isinstance(y, (list, tuple)) else y
            enc = x.get("encoder_target", None)
            if enc is None:
                print(
                    "ℹ️ eval_naive_last: x['encoder_target'] is missing. "
                    "To enable this baseline, construct TimeSeriesDataSet with "
                    "include_target_in_encoder=True. Skipping naive-last baseline."
                )
                return None

            last = enc[..., -1]  # [B]
            naive_std = last.unsqueeze(-1).repeat(1, y_std.size(1))
            scale = x.get("target_scale", None)

            y_pred = _safe_inverse(naive_std, scale).detach().cpu().numpy().reshape(-1).astype(np.float64)
            y_true = _safe_inverse(y_std,    scale).detach().cpu().numpy().reshape(-1).astype(np.float64)
            msk = np.isfinite(y_pred) & np.isfinite(y_true)
            if not np.any(msk):
                continue
            preds_list.append(y_pred[msk])
            trues_list.append(y_true[msk])

    if not preds_list:
        return None

    pred = np.concatenate(preds_list, axis=0)
    true = np.concatenate(trues_list, axis=0)

    mae = float(np.mean(np.abs(pred - true)))
    mse = float(np.mean((pred - true) ** 2))
    msk_mape = np.abs(true) >= float(tau)
    mape = float(np.mean(np.abs((pred[msk_mape] - true[msk_mape]) / true[msk_mape]))) if np.any(msk_mape) else float("nan")
    smape = _smape(pred, true)
    wape = _wape(pred, true)
    mdape = _mdape(pred, true, tau=tau)

    return {
        "MAE": mae,
        "MSE": mse,
        f"MAPE(|y|>={tau:g})": mape,
        "sMAPE": smape,
        "WAPE": wape,
        f"MdAPE(|y|>={tau:g})": mdape,
    }


# ========= Rich evaluation: horizon-wise, slices, baselines, plots =========
def evaluate_full(tft_model, val_loader, val_tsd, save_dir: Path, tag="tft", tau=50.0):
    """
    Compact evaluation routine that computes:

      - overall metrics (MAE, MSE, WAPE, sMAPE, MAPE/MdAPE on high-CI points),
      - per-horizon metrics (per forecast step),
      - a few slices (peak / valley / weekday / weekend),
      - baselines (naive-last, hour-of-day mean),
      - WAPE vs horizon plot,
      - bar chart of overall WAPE (TFT vs baselines),
      - metrics JSON and CSV outputs.

    The results JSON is saved as `{tag}_metrics.json` under `save_dir`.
    """
    tft_model.eval()
    preds, trues = [], []
    hrs, wkends = [], []

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            y_std = y[0] if isinstance(y, (list, tuple)) else y

            out = tft_model(x)
            if isinstance(out, dict):
                out = out.get("prediction", out.get("output", None))
            if isinstance(out, (list, tuple)):
                out = out[0]
            y_hat = _median_quantile(out)

            scale = x.get("target_scale", None)
            y_pred = _safe_inverse(y_hat, scale).detach().cpu().numpy()
            y_true = _safe_inverse(y_std, scale).detach().cpu().numpy()

            # decoder continuous features are required here
            if "decoder_cont" not in x:
                raise KeyError(
                    "evaluate_full expects x['decoder_cont']. "
                    "Ensure TimeSeriesDataSet includes hour_sin/hour_cos/is_weekend "
                    "in time_varying_known_reals."
                )
            dec_cont = x["decoder_cont"].detach().cpu().numpy()
            try:
                hs = dec_cont[..., _feature_idx(val_tsd, "hour_sin")]
                hc = dec_cont[..., _feature_idx(val_tsd, "hour_cos")]
                iw = dec_cont[..., _feature_idx(val_tsd, "is_weekend")]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read hour_sin/hour_cos/is_weekend from decoder_cont: {e}; "
                    f"dataset.reals={getattr(val_tsd, 'reals', None)}"
                )

            preds.append(y_pred)
            trues.append(y_true)
            hrs.append(_hour_from_sin_cos(hs, hc))
            wkends.append(iw)

    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    hour = np.concatenate(hrs, axis=0)
    wknd = np.concatenate(wkends, axis=0)

    # overall metrics
    overall = _agg_metrics(pred.reshape(-1), true.reshape(-1), tau=tau)

    # horizon-wise
    dec_len = pred.shape[1]
    per_h = []
    for h in range(dec_len):
        mh = _agg_metrics(pred[:, h], true[:, h], tau=tau)
        mh["h"] = h + 1
        per_h.append(mh)

    # slices: peak / valley / weekday / weekend
    q20, q80 = np.nanquantile(true, 0.20), np.nanquantile(true, 0.80)
    m_peak = _agg_metrics(pred[true >= q80], true[true >= q80], tau=tau)
    m_valley = _agg_metrics(pred[true <= q20], true[true <= q20], tau=tau)
    m_wkend = _agg_metrics(pred[wknd >= 0.5], true[wknd >= 0.5], tau=tau)
    m_wkday = _agg_metrics(pred[wknd < 0.5],  true[wknd < 0.5],  tau=tau)

    # baselines: naive-last + hour-of-day mean
    base_last = eval_naive_last(val_loader, tau=tau) or {}
    hr_bins = np.floor(hour).astype(int) % 24
    hr_mean = np.zeros(24, dtype=np.float64)
    for h in range(24):
        msk = (hr_bins == h)
        hr_mean[h] = np.mean(true[msk]) if np.any(msk) else np.nan
    base_hour_pred = hr_mean[hr_bins]
    base_hour = _agg_metrics(base_hour_pred.reshape(-1), true.reshape(-1), tau=tau)

    results = {
        "overall": overall,
        "per_horizon": per_h,
        "slices": {
            "peak": m_peak,
            "valley": m_valley,
            "weekday": m_wkday,
            "weekend": m_wkend,
        },
        "baselines": {
            "naive_last": base_last,
            "hour_mean": base_hour,
        },
    }

    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) metrics.json
    with open(save_dir / f"{tag}_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # 2) per_horizon.csv
    pd.DataFrame(per_h).to_csv(save_dir / f"{tag}_per_horizon.csv", index=False)

    # 3) WAPE vs horizon
    plt.figure()
    plt.plot([m["h"] for m in per_h], [m["WAPE"] for m in per_h])
    plt.xlabel("Horizon (h)")
    plt.ylabel("WAPE")
    plt.title(f"{tag} WAPE per horizon")
    plt.tight_layout()
    plt.savefig(save_dir / f"{tag}_wape_per_h.png")
    plt.close()

    # 4) TFT vs baselines (WAPE)
    labels = ["TFT", "Naive-last", "Hour-mean"]
    vals = [overall["WAPE"], base_last.get("WAPE", np.nan), base_hour["WAPE"]]
    plt.figure()
    plt.bar(labels, vals)
    plt.ylabel("WAPE")
    plt.title("Overall WAPE: TFT vs Baselines")
    plt.tight_layout()
    plt.savefig(save_dir / f"{tag}_baselines_wape.png")
    plt.close()

    return results


# --------- Quiet checkpoint loader that prints only useful info ---------
_MISMATCH_RE = re.compile(
    r"size mismatch for ([^:]+): copying a param with shape .* from checkpoint, "
    r"the shape in current model is .*"
)


def _try_load_ckpt(path, tft_core, lr=1e-3, map_location="cpu"):
    """
    Try loading a checkpoint into the LitTFT wrapper.

    If we hit a size mismatch or any other error, return (None, reason)
    and print a short one-line message.
    """
    try:
        m = LitTFT.load_from_checkpoint(
            path,
            tft_model=tft_core,
            learning_rate=lr,
            map_location=map_location,
        )
        return m, None
    except RuntimeError as e:
        msg = str(e)
        mobj = _MISMATCH_RE.search(msg)
        layer = mobj.group(1) if mobj else "unknown layer(s)"
        print(f"⏭️ Skipping {Path(path).name}: size mismatch at {layer}")
        return None, layer
    except Exception as e:
        print(f"⏭️ Skipping {Path(path).name}: {e.__class__.__name__}")
        return None, "unknown"


def pick_best_by_metric(ckpt_dir_in, tft_core_in, val_dl_in, metric_name="MAE", tau=50.0, lr=1e-3):
    """
    Iterate over all checkpoints under `ckpt_dir_in`, evaluate them on `val_dl_in`,
    and select the checkpoint with the smallest value for `metric_name`
    (default: MAE) on the original scale.
    """
    best_path, best_val = None, float("inf")
    for p in sorted(glob(str(Path(ckpt_dir_in) / "*.ckpt"))):
        mdl, _ = _try_load_ckpt(p, tft_core_in, lr=lr, map_location="cpu")
        if mdl is None:
            continue
        mdl.eval()
        try:
            metrics = evaluate_on_loader(mdl.tft, val_dl_in, tau=tau)
            val = metrics.get(metric_name, float("inf"))
            if np.isfinite(val) and val < best_val:
                best_val, best_path = val, p
        except Exception:
            print(f"⏭️ Skipping {Path(p).name}: EvalError")
    return best_path, best_val


# --------- Main training pipeline ---------
def main(config_path: str):
    print(f"🧠 Loading config: {config_path}")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # ========= Read config sections =========
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    train_cfg = cfg.get("training", {})
    out_cfg = cfg["output"]

    # ========= Fast mode: limit max epochs for quick experiments =========
    fast_mode = cfg.get("trainer", {}).get("fast_mode", False)
    max_epochs = cfg["trainer"]["max_epochs"]
    if fast_mode:
        max_epochs = min(max_epochs, 10)

    # ========= Result directory structure =========
    region = str(dataset_cfg.get("region") or dataset_cfg.get("area") or "default")
    run_name = out_cfg.get("run_name") or pd.Timestamp.now(tz="UTC").strftime("%Y%m%d-%H%M%S")
    root_dir = Path(out_cfg["save_dir"]).expanduser().resolve()
    run_dir = root_dir / region / run_name
    ckpt_dir = run_dir / "checkpoints"
    figs_dir = run_dir / "figs"
    logs_dir = run_dir / "logs"
    (logs_dir / "logs").mkdir(parents=True, exist_ok=True)
    for d in (ckpt_dir, figs_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {run_dir}")

    # Save a copy of the config used for this run
    with open(run_dir / "config_used.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ========= Load and inspect data =========
    df, _meta = load_and_merge(cfg)
    print(f"✅ Data loaded: shape={df.shape}")
    print("✅ Columns:", df.columns.tolist())
    print("🔍 Head:")
    print(df.head(3))

    # --------- Time features & light outlier clipping ----------
    utc = pd.to_datetime(df[dataset_cfg["datetime_col"]], utc=True)

    # Daily / weekly cycles
    df["hour_sin"] = np.sin(2 * np.pi * utc.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * utc.dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * utc.dt.weekday / 7)
    df["dow_cos"] = np.cos(2 * np.pi * utc.dt.weekday / 7)

    # Weekend / holiday flags (US calendar; fallback to 0)
    df["is_weekend"] = (utc.dt.weekday >= 5).astype(np.int8)
    try:
        from pandas.tseries.holiday import USFederalHolidayCalendar

        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=utc.min().date(), end=utc.max().date())
        holiday_dates = set(holidays.date)
        df["is_holiday"] = pd.Series(utc.dt.date).isin(holiday_dates).astype(np.int8)
    except Exception:
        df["is_holiday"] = 0

    # Light clipping of extreme target values
    tgt = dataset_cfg["target_col"]
    df[tgt] = df[tgt].clip(lower=0, upper=df[tgt].quantile(0.999))

    # Rolling statistics (by group if present, using past window to avoid leakage)
    group_col = dataset_cfg.get("group_col", None)
    sort_cols = [dataset_cfg["datetime_col"]]
    if group_col is not None:
        sort_cols = [group_col] + sort_cols
    df = df.sort_values(sort_cols)

    if group_col is not None:
        g = df.groupby(group_col)[tgt]
        df["y_roll_mean_24"] = g.transform(lambda s: s.shift(1).rolling(24, min_periods=1).mean())
        df["y_roll_std_24"] = g.transform(lambda s: s.shift(1).rolling(24, min_periods=1).std())
        df["y_diff_1"] = g.transform(lambda s: s - s.shift(1))
    else:
        s = df[tgt]
        df["y_roll_mean_24"] = s.shift(1).rolling(24, min_periods=1).mean()
        df["y_roll_std_24"] = s.shift(1).rolling(24, min_periods=1).std()
        df["y_diff_1"] = s - s.shift(1)

    # Fill rolling NaNs with feature-wise median
    for col in ["y_roll_mean_24", "y_roll_std_24", "y_diff_1"]:
        df[col] = df[col].fillna(df[col].median())

    # Valley indicator based on target quantile
    q20 = df[tgt].quantile(0.2)
    df["is_valley_y"] = (df[tgt] <= q20).astype("int8")

    # Ensure these features are included in known_reals
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

    # ========= Build TimeSeriesDataSet =========
    print("📊 Building TimeSeriesDataSet and splitting train/val ...")
    train_tsd, val_tsd = build_tsd(df, dataset_cfg)
    print(f"🧩 Train samples: {len(train_tsd)}, Val samples: {len(val_tsd)}")

    # Save column names and basic data summary
    pd.Series(df.columns).to_frame("column").to_csv(run_dir / "columns.csv", index=False)
    meta_summary = {
        "train_samples": len(train_tsd),
        "val_samples": len(val_tsd),
        "target": dataset_cfg["target_col"],
    }
    with open(run_dir / "data_summary.json", "w") as f:
        json.dump(meta_summary, f, indent=2)

    # ========= DataLoaders =========
    batch_size = cfg["trainer"].get("batch_size", 256)
    num_workers = max(2, os.cpu_count() // 2)

    train_dl = train_tsd.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=False,
    )

    try:
        val_dl = val_tsd.to_dataloader(
            train=False,
            batch_size=batch_size,
            num_workers=max(1, num_workers // 2),
            persistent_workers=(max(1, num_workers // 2) > 0),
            pin_memory=False,
        )
        _ = next(iter(val_dl))
    except Exception as e:
        print(f"⚠️ Validation dataloader failed to initialize ({e}). Validation will be skipped.")
        val_dl = None

    # Quick check on target values in original scale
    if val_dl is not None:
        try:
            batch = next(iter(val_dl))
            x, y = batch
            y_std = y[0] if isinstance(y, (list, tuple)) else y
            scale = x.get("target_scale", None)
            y_true_orig = _safe_inverse(y_std, scale).detach().cpu().numpy().ravel()
            print(
                "🔎 Quick check on y_true (original scale): "
                f"min={float(np.nanmin(y_true_orig)):.3f}, "
                f"median={float(np.nanmedian(y_true_orig)):.3f}, "
                f"max={float(np.nanmax(y_true_orig)):.3f}"
            )
        except Exception as e:
            print(f"⚠️ Quick check on y_true failed: {e}")

    # ========= Build TFT core (nn.Module) =========
    qtls = train_cfg.get("quantiles", [0.1, 0.5, 0.9])
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

    # ========= Naive-last baseline (before training) =========
    if val_dl is not None:
        base = eval_naive_last(val_dl, tau=50.0)
        if base is not None:
            print(f"📉 Naive-last baseline (original scale): {base}")

    # ========= Wrap TFT in Lightning module =========
    model_cfg = cfg.get("model", {})
    quantiles = model_cfg.get("quantiles", [0.1, 0.5, 0.9])

    lit_model = LitTFT(
        tft_model=tft_core,
        cfg=cfg,
        learning_rate=train_cfg.get("learning_rate", 1e-3),
        quantiles=quantiles,
    )

    # ========= Trainer / callbacks =========
    logger = False  # disable TB/CSV logging for now

    monitor_metric = "val_loss" if val_dl is not None else None
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="tft-{epoch:02d}-{val_loss:.2f}" if monitor_metric else "tft-{epoch:02d}",
        monitor=monitor_metric,
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    es_cb = (
        EarlyStopping(monitor=monitor_metric, min_delta=0.01, mode="min", patience=3)
        if monitor_metric
        else None
    )
    callbacks = [ckpt_cb] + ([es_cb] if es_cb is not None else [])

    # We still detect accelerator but force CPU in Trainer for stability
    if torch.cuda.is_available():
        accelerator_detected = "gpu"
    elif torch.backends.mps.is_available():
        accelerator_detected = "mps"
    else:
        accelerator_detected = "cpu"
    print(f"⚙️ Detected accelerator: {accelerator_detected} (Trainer will use CPU for stability)")

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        deterministic=True,
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        enable_checkpointing=True,
        enable_model_summary=False,
        check_val_every_n_epoch=1 if val_dl is not None else 0,
        enable_progress_bar=False,
        gradient_clip_val=1.0,
        default_root_dir=str(logs_dir),
    )

    acc_name = type(trainer.accelerator).__name__
    device = getattr(trainer.strategy, "root_device", "unknown")
    prec_str = getattr(trainer, "precision", "unknown")
    print(f"⚙️ Trainer accelerator={acc_name}, device={device}, precision={prec_str}")

    print("🚀 Start training Temporal Fusion Transformer ...")
    trainer.fit(lit_model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    print("✅ Training finished.")

    # Show last-epoch metrics if available
    final_metrics = {}
    for k in ("train_loss_epoch", "val_loss"):
        v = trainer.callback_metrics.get(k, None)
        if v is not None:
            try:
                final_metrics[k] = float(v.cpu().item())
            except Exception:
                pass
    if final_metrics:
        print("📉 Final (last epoch) metrics:", final_metrics)

    # ========= After training: select best ckpt by original-scale MAE =========
    if val_dl is not None and any(ckpt_dir.glob("*.ckpt")):
        sel_path, sel_val = pick_best_by_metric(
            ckpt_dir_in=ckpt_dir,
            tft_core_in=tft_core,
            val_dl_in=val_dl,
            metric_name="MAE",
            tau=50.0,
            lr=train_cfg.get("learning_rate", 1e-3),
        )
        if sel_path is not None:
            print(f"🏅 Best checkpoint by original-scale MAE: {sel_path} (MAE={sel_val:.2f})")
            ckpt_cb.best_model_path = str(sel_path)

    # ====== Unified evaluation + save metrics/charts ======
    if val_dl is not None:
        best_model, _ = _try_load_ckpt(
            ckpt_cb.best_model_path,
            tft_core,
            lr=train_cfg.get("learning_rate", 1e-3),
            map_location="cpu",
        )
        if best_model is None:
            best_model = lit_model  # fallback

        res = evaluate_full(best_model.tft, val_dl, val_tsd, figs_dir, tag="tft", tau=50.0)
        print("📦 Saved metrics and charts to:", figs_dir)

        best_meta = {"best_ckpt": ckpt_cb.best_model_path}
        try:
            best_meta["overall"] = res.get("overall", {})
        except Exception:
            pass
        with open(run_dir / "best_model.json", "w") as f:
            json.dump(best_meta, f, indent=2)
        print(f"📝 best_model.json saved to: {run_dir}")

    # ========= Fallback: evaluate current weights if no ckpt saved =========
    if not any(ckpt_dir.glob("*.ckpt")):
        print(f"⚠️ No checkpoints found under {ckpt_dir}. Evaluating current weights once (if val_dl exists).")
        if val_dl is not None:
            lit_model.eval()
            metrics = evaluate_on_loader(lit_model.tft, val_dl, tau=50.0)
            print(f"📈 Eval (no ckpt, current weights): {metrics}")
    else:
        print("✅ Checkpoints and full evaluation saved under figs/ and best_model.json.")


if __name__ == "__main__":
    pl.seed_everything(2025, workers=True)
    parser = argparse.ArgumentParser(
        description="Train a Temporal Fusion Transformer model for carbon-intensity forecasting."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config, e.g. configs/ercot.json.",
    )
    args = parser.parse_args()
    main(args.config)