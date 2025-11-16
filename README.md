# Decision-Focused Carbon Forecasting

This repository contains a full pipeline for **carbon intensity forecasting** and  
**decision-focused evaluation** across three U.S. grid regions: **ERCOT, NYISO, PJM**.

The project trains a Temporal Fusion Transformer (TFT) for 96-hour carbon intensity forecasting,  
exports prediction windows, and evaluates how forecast accuracy affects a downstream  
**carbon-aware scheduling** task. The goal is to align forecasting performance with  
operational decision quality.

---

## 📁 Repository Structure
code/
configs/                     # JSON configs for each region (ERCOT, NYISO, PJM)
src/
data_loader.py
build_dataset.py
train_tft.py
export_forecasts.py
decision_simulate.py
decision_loss.py
plot_decision_results.py
run_all.sh                   # Train all regions
summarize_forecast_results.py
summary_decision_results.py

data/
ERCO/
NYISO/
PJM/                         # Raw data files: forecasts, emissions, weather, fuel mix

results/
forecasting/
decision/
figures/                     # Final plots and summary tables
---

## 🚀 Getting Started

### **1. Install dependencies**

```bash
pip install -r requirements.txt
```
### **2. Prepare the dataset**

Place your CSV files under:`data/ERCOT/
data/NYISO/
data/PJM/`

Every config file in code/configs/*.json contains a "data_root" field.
Modify it to point to your local data directory if needed:  
`"data_root": "/absolute/path/to/decision_focused_carbon/data"`

### **3. Train the TFT forecasting models**

Train all regions automatically:
```
cd code
bash run_all.sh
```

Model checkpoints & metrics will be saved to:`results_<region>/default/<timestamp>/`

### **4. Export 96-hour rolling forecasts**

For a specific region:
```
python -m src.export_forecasts \
  --config configs/ercot.json \
  --run_dir results_ercot/default/<timestamp>
```
This generates:`forecasts_val.parquet`

### **5. Decision-focused evaluation**

Run the carbon-aware scheduling evaluation:
```
python -m src.decision_simulate \
  --run_dir results_ercot/default/<timestamp> \
  --region ERCOT
```
Generates:`decision_sim.parquet`

### **6. Summaries and plots**

Summaries for forecasting:`python summarize_forecast_results.py`  
Summaries for decision performance:`python summary_decision_results.py`  
Outputs go to:`results/figures/
results/decision/decision_summary/`

## 📌 Notes
	•	Large files can be excluded using .gitignore (recommended).
	•	The codebase is modular and can be extended to additional regions.
	•	Results are suitable for academic documentation and reproducible ML experiments.
