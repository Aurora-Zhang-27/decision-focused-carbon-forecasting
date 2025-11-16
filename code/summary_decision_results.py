import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def find_latest_decision_file(region: str) -> Path:
    """
    Find the most recent decision_sim.parquet for a given region.

    Expected layout (example):
        results_ercot/default/20251108-024136/decision_sim.parquet
    """
    root = Path(__file__).resolve().parent
    region_dir = root / f"results_{region.lower()}" / "default"

    if not region_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {region_dir}")

    run_dirs = sorted([p for p in region_dir.iterdir() if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run subdirectories found under: {region_dir}")

    latest_run = run_dirs[-1]
    dec_path = latest_run / "decision_sim.parquet"

    if not dec_path.exists():
        raise FileNotFoundError(
            f"Missing decision_sim.parquet at: {dec_path}. "
            f"Please run the decision simulation first."
        )

    return dec_path


def summarize_region(region: str) -> Dict[str, float]:
    """
    Load decision_sim.parquet for one region and compute
    average regret and ratio metrics.
    """
    path = find_latest_decision_file(region)
    df = pd.read_parquet(path)

    required_cols: List[str] = [
        "regret_pred",
        "regret_uniform",
        "ratio_pred",
        "ratio_uniform",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(
                f"{region}: decision_sim.parquet is missing column {col!r}. "
                f"Actual columns: {df.columns.tolist()}"
            )

    return {
        "region": region.upper(),
        "mean_regret_pred": df["regret_pred"].mean(),
        "mean_regret_uniform": df["regret_uniform"].mean(),
        "mean_ratio_pred": df["ratio_pred"].mean(),
        "mean_ratio_uniform": df["ratio_uniform"].mean(),
    }


def main() -> None:
    regions = ["ercot", "nyiso", "pjm"]

    rows = [summarize_region(r) for r in regions]
    summary_df = pd.DataFrame(rows)
    print(summary_df)

    # Output directory for tables and figures
    out_dir = Path(__file__).resolve().parent / "decision_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save summary as CSV and JSON
    summary_df.to_csv(out_dir / "decision_summary.csv", index=False)
    with open(out_dir / "decision_summary.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Saved decision summary to: {out_dir}/decision_summary.csv and .json")

    # ===== Figure 1: decision regret (lower is better) =====
    x = range(len(summary_df))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(
        [i - width / 2 for i in x],
        summary_df["mean_regret_pred"],
        width=width,
        label="TFT (regret_pred)",
    )
    plt.bar(
        [i + width / 2 for i in x],
        summary_df["mean_regret_uniform"],
        width=width,
        label="Uniform (regret_uniform)",
    )
    plt.xticks(list(x), summary_df["region"])
    plt.ylabel("Average Regret")
    plt.title("Decision Regret Comparison (lower is better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "regret_comparison.png", dpi=200)
    plt.close()

    # ===== Figure 2: decision ratio (closer to 1 is better) =====
    plt.figure(figsize=(6, 4))
    plt.bar(
        [i - width / 2 for i in x],
        summary_df["mean_ratio_pred"],
        width=width,
        label="TFT (ratio_pred)",
    )
    plt.bar(
        [i + width / 2 for i in x],
        summary_df["mean_ratio_uniform"],
        width=width,
        label="Uniform (ratio_uniform)",
    )
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xticks(list(x), summary_df["region"])
    plt.ylabel("Average Ratio to Oracle")
    plt.title("Decision Quality Ratio (closer to 1 is better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "ratio_comparison.png", dpi=200)
    plt.close()

    print(f"Saved figures to: {out_dir}/regret_comparison.png and ratio_comparison.png")


if __name__ == "__main__":
    main()