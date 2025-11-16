import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def greedy_schedule(costs, total_energy: float = 12.0, cap_per_step: float = 2.0):
    """
    Simple greedy carbon-aware scheduling.

    Given a vector of per-hour "costs" (here: carbon intensity),
    we allocate a fixed total amount of energy over the horizon
    under a per-step capacity constraint.

    Strategy:
        - Sort hours by cost in ascending order.
        - Fill the cheapest hours first, up to cap_per_step,
          until total_energy is exhausted.

    This serves as a simple and interpretable prototype of a
    "carbon-minimizing" scheduling policy.
    """
    costs = np.asarray(costs, dtype=np.float64)
    H = len(costs)
    order = np.argsort(costs)  # from lowest to highest cost
    q = np.zeros(H, dtype=np.float64)
    remain = float(total_energy)

    for idx in order:
        if remain <= 0:
            break
        alloc = min(cap_per_step, remain)
        q[idx] = alloc
        remain -= alloc

    # If remain > 0, it means cap_per_step is too small to accommodate
    # the entire total_energy within the horizon. We simply return q.
    return q


def simulate_for_region(
    run_dir,
    region_name: str | None = None,
    total_energy: float = 12.0,
    cap_per_step: float = 2.0,
):
    """
    Run decision-focused simulation for a single region.

    For a given training run directory (run_dir), this function:
      - Loads `forecasts_val.parquet` (validation forecasts).
      - For each sample_id (one H-step forecast window), constructs
        three allocation policies:
          1) Oracle (uses true carbon intensity).
          2) Forecast-based (uses predicted carbon intensity).
          3) Uniform (distributes energy evenly across the horizon).
      - Evaluates all policies using *true* carbon intensity.
      - Computes decision-focused metrics: cost, regret, and ratio-to-oracle.
      - Saves results to `decision_sim.parquet`.

    Args:
        run_dir: Path to a single model run directory
                 (e.g., results/ercot/default/20251108-024136).
        region_name: Optional region label to store in the output.
        total_energy: Total flexible energy to allocate across the horizon.
        cap_per_step: Maximum energy that can be scheduled per time step.

    Returns:
        A pandas DataFrame containing per-sample decision metrics.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    fpath = run_dir / "forecasts_val.parquet"
    if not fpath.exists():
        raise FileNotFoundError(f"Forecast file not found: {fpath}")

    print(f"[Load] Reading validation forecasts from: {fpath}")
    df = pd.read_parquet(fpath)

    required_cols = {"sample_id", "horizon", "y_true", "y_pred"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"`forecasts_val.parquet` is missing required columns: {required_cols}. "
            f"Actual columns: {df.columns.tolist()}"
        )

    # Sort to ensure horizons appear in order 1,2,3,...
    df = df.sort_values(["sample_id", "horizon"])

    records = []

    for sid, g in df.groupby("sample_id"):
        true = g["y_true"].to_numpy(dtype=np.float64)  # true carbon intensity
        pred = g["y_pred"].to_numpy(dtype=np.float64)  # predicted carbon intensity
        H = len(true)

        E = float(total_energy)
        cap = float(cap_per_step)

        # 1) Oracle policy: scheduling based on true CI
        q_oracle = greedy_schedule(true, total_energy=E, cap_per_step=cap)
        # 2) Forecast-based: scheduling based on predicted CI, evaluated on true CI
        q_pred = greedy_schedule(pred, total_energy=E, cap_per_step=cap)
        # 3) Uniform baseline: spread energy evenly across the horizon
        q_uniform = np.full(H, E / H, dtype=np.float64)
        if cap > 0:
            q_uniform = np.minimum(q_uniform, cap)

        # Evaluate total emissions (cost) under true CI
        J_oracle = float(np.sum(q_oracle * true))
        J_pred = float(np.sum(q_pred * true))
        J_uniform = float(np.sum(q_uniform * true))

        # Decision-focused metrics
        regret_pred = J_pred - J_oracle
        regret_uniform = J_uniform - J_oracle

        ratio_pred = J_pred / J_oracle if J_oracle > 0 else np.nan
        ratio_uniform = J_uniform / J_oracle if J_oracle > 0 else np.nan

        records.append(
            {
                "region": region_name,
                "sample_id": int(sid),
                "H": H,
                "total_energy": E,
                "cap_per_step": cap,
                "J_oracle": J_oracle,
                "J_pred": J_pred,
                "J_uniform": J_uniform,
                "regret_pred": regret_pred,
                "regret_uniform": regret_uniform,
                "ratio_pred": ratio_pred,
                "ratio_uniform": ratio_uniform,
            }
        )

    res_df = pd.DataFrame(records)

    out_path = run_dir / "decision_sim.parquet"
    res_df.to_parquet(out_path, index=False)
    print(f"[Save] Decision simulation results written to: {out_path}")

    # Aggregate metrics for a quick summary
    mean_regret_pred = res_df["regret_pred"].mean()
    mean_regret_uniform = res_df["regret_uniform"].mean()
    mean_ratio_pred = res_df["ratio_pred"].mean()
    mean_ratio_uniform = res_df["ratio_uniform"].mean()

    print("\n[Decision Metrics] (averaged over all samples)")
    print(f"  mean regret_pred    = {mean_regret_pred:.4f}")
    print(f"  mean regret_uniform = {mean_regret_uniform:.4f}")
    print(f"  mean ratio_pred     = {mean_ratio_pred:.44f}")
    print(f"  mean ratio_uniform  = {mean_ratio_uniform:.4f}")

    return res_df


def main():
    parser = argparse.ArgumentParser(
        description="Decision-focused simulation for a single region."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to a single run directory, e.g. "
             "`results/ercot/default/20251108-024136`.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Optional region name (stored in the output file).",
    )
    parser.add_argument(
        "--total_energy",
        type=float,
        default=12.0,
        help="Total flexible energy to allocate within each H-step window.",
    )
    parser.add_argument(
        "--cap_per_step",
        type=float,
        default=2.0,
        help="Maximum flexible energy per time step.",
    )
    args = parser.parse_args()

    simulate_for_region(
        run_dir=args.run_dir,
        region_name=args.region,
        total_energy=args.total_energy,
        cap_per_step=args.cap_per_step,
    )


if __name__ == "__main__":
    main()