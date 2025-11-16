# src/decision_loss.py
"""
Phase-2 placeholder for decision-focused training.

Current NeurIPS submission **does not** use this module in the training loop.
We keep it here as a scaffold for future work:
- `decision_focused_loss` will eventually wrap a differentiable optimization
  layer (e.g., cvxpylayers) to define a decision-aware loss.
- `decision_weighted_mae` is a simple prototype that up-weights errors on
  high-carbon-intensity hours.

For now, configs should set:  "use_decision_loss": false
so that training relies on the standard quantile loss only.
"""

import torch
import torch.nn.functional as F


def decision_focused_loss(y_pred, y_true, context=None):
    """
    Placeholder for a fully decision-focused loss.

    Args:
        y_pred: predicted carbon intensity (tensor)
        y_true: ground-truth carbon intensity (tensor)
        context: optional extra information (e.g., capacity E, horizon H, etc.)

    Returns:
        A scalar loss tensor.

    NOTE:
    - In the current version, we simply return an MAE loss as a safe fallback.
    - A future implementation can replace this with a differentiable
      optimization layer that directly measures decision regret.
    """
    if isinstance(y_true, (list, tuple)):
        y_true = y_true[0]

    y_pred = y_pred.float()
    y_true = y_true.float()

    # Safe fallback: plain MAE
    loss = F.l1_loss(y_pred, y_true)
    return loss


def decision_weighted_mae(
    y_pred,
    y_true,
    lambda_decision: float = 0.5,
):
    """
    Decision-weighted MAE prototype.

    - Base term: standard MAE.
    - Decision term: errors in high-carbon-intensity hours get larger weights.
    - Final loss:
        loss = (1 - lambda_decision) * MAE + lambda_decision * weighted_MAE

    Args:
        y_pred: predicted carbon intensity (tensor)
        y_true: ground-truth carbon intensity (tensor)
        lambda_decision: weight in [0, 1] for the decision-aware component

    Returns:
        loss: combined scalar loss tensor
        mae: plain MAE (for logging)
        weighted_mae: decision-weighted MAE (for logging)
    """
    if isinstance(y_true, (list, tuple)):
        y_true = y_true[0]

    # Ensure float tensors and matching shapes
    y_pred = y_pred.float()
    y_true = y_true.float()

    # ----- base MAE -----
    mae = torch.mean(torch.abs(y_pred - y_true))

    # ----- decision-aware weights -----
    # Use the magnitude of carbon intensity as a proxy for decision importance:
    # higher CI -> higher weight.
    abs_y = torch.abs(y_true)

    # Normalize by batch mean to keep numbers in a reasonable range
    mean_abs_y = torch.mean(abs_y) + 1e-6
    weights = abs_y / mean_abs_y

    # Clip very large weights to avoid exploding gradients
    weights = torch.clamp(weights, max=5.0)

    # Weighted MAE
    weighted_mae = torch.mean(torch.abs(y_pred - y_true) * weights)

    # Combine
    loss = (1.0 - lambda_decision) * mae + lambda_decision * weighted_mae
    return loss, mae, weighted_mae