import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Base directory for all result folders (adjust if needed)
base = Path(".")

paths = {
    "ERCOT": base / "results_ercot" / "default" / "20251108-024136" / "decision_sim.parquet",
    "NYISO": base / "results_nyiso" / "default" / "20251108-025342" / "decision_sim.parquet",
    "PJM":   base / "results_pjm"   / "default" / "20251108-030528" / "decision_sim.parquet",
}

rows = []
for region, p in paths.items():
    df = pd.read_parquet(p)
    rows.append(
        {
            "region": region,
            "regret_pred": df["regret_pred"].mean(),
            "regret_uniform": df["regret_uniform"].mean(),
            "ratio_pred": df["ratio_pred"].mean(),
            "ratio_uniform": df["ratio_uniform"].mean(),
        }
    )

summary = pd.DataFrame(rows)
print(summary)

# 1) Decision regret bar plot
plt.figure()
summary.set_index("region")[["regret_pred", "regret_uniform"]].plot(kind="bar")
plt.ylabel("Average regret")
plt.title("Decision regret: prediction-based vs. uniform")
plt.tight_layout()
plt.savefig("fig_decision_regret.png")

# 2) Emission ratio bar plot
plt.figure()
summary.set_index("region")[["ratio_pred", "ratio_uniform"]].plot(kind="bar")
plt.ylabel("Average emission ratio to oracle")
plt.title("Decision emission ratio: prediction-based vs. uniform")
plt.tight_layout()
plt.savefig("fig_decision_ratio.png")