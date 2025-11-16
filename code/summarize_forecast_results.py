import json
from pathlib import Path

import pandas as pd

# Root directory of the project (the folder containing this script)
ROOT = Path(__file__).resolve().parent

REGIONS = ["ercot", "nyiso", "pjm"]


def find_latest_metrics(region: str) -> Path:
    """
    Find the most recent tft_metrics.json file for a given region.
    Looks into: results_{region}/default/*/figs/tft_metrics.json
    """
    pattern = ROOT / f"results_{region}" / "default" / "*" / "figs" / "tft_metrics.json"
    candidates = list(pattern.glob("**/*")) if pattern.is_dir() else list(ROOT.glob(str(pattern)))

    # Simpler & safer: just glob once
    candidates = list((ROOT / f"results_{region}").glob("default/*/figs/tft_metrics.json"))

    if not candidates:
        raise FileNotFoundError(f"No tft_metrics.json found for region {region!r}.")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def collect_region_row(region: str) -> dict:
    """
    Load metrics for one region and return a single summary row.
    """
    metrics_path = find_latest_metrics(region)
    with metrics_path.open("r") as f:
        metrics = json.load(f)

    overall = metrics["overall"]
    baselines = metrics["baselines"]

    return {
        "region": region.upper(),
        "MAE": overall["MAE"],
        "WAPE_TFT": overall["WAPE"],
        "sMAPE_TFT": overall["sMAPE"],
        "WAPE_naive": baselines["naive_last"]["WAPE"],
        "WAPE_hour_mean": baselines["hour_mean"]["WAPE"],
    }


def main():
    rows = [collect_region_row(r) for r in REGIONS]
    df = pd.DataFrame(rows)
    print(df)

    out_path = ROOT / "forecast_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved forecast summary to: {out_path}")


if __name__ == "__main__":
    main()