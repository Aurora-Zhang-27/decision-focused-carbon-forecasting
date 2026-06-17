# Decision-Focused Carbon Forecasting

This repository contains a research prototype for carbon-intensity forecasting and decision-focused evaluation across three U.S. grid regions: **ERCOT**, **NYISO**, and **PJM**.

The project trains Temporal Fusion Transformer models for multi-hour carbon-intensity forecasting, exports rolling forecast windows, and evaluates how forecast quality affects a downstream carbon-aware scheduling task. The goal is to connect forecasting performance with operational decision quality, rather than evaluating forecasts only through scalar error metrics.

## Motivation

Carbon-aware systems can reduce emissions by shifting flexible electricity demand toward lower-carbon hours. Examples include batch computing, electric-vehicle charging, storage operation, and other flexible workloads.

For these applications, a forecast is useful only if it supports a better decision. A model with lower RMSE does not always produce the best scheduling outcome. This project therefore evaluates forecasts through both prediction metrics and a stylized downstream scheduling objective.

## Main Contributions

- Regional carbon-intensity forecasting for ERCOT, NYISO, and PJM
- Temporal Fusion Transformer forecasting prototype
- 96-hour rolling forecast export workflow
- Decision-focused evaluation with a carbon-aware scheduling simulation
- Regret-style comparison against simple scheduling baselines
- Reproducible scripts for training, forecast export, decision evaluation, and plotting

## Repository Structure

```text
code/
  configs/
    ercot.json
    nyiso.json
    pjm.json

  src/
    data_loader.py             Data loading utilities
    build_dataset.py           Dataset construction
    train_tft.py               TFT training entrypoint
    export_forecasts.py        Rolling forecast export
    decision_simulate.py       Carbon-aware scheduling simulation
    decision_loss.py           Decision metrics
    plot_decision_results.py   Plotting utilities

  run_all.sh                   Train all configured regions
  summarize_forecast_results.py
  summary_decision_results.py

data/
  ERCOT/
  NYISO/
  PJM/                         Regional input data

results/
  forecasting/                 Forecast outputs and metrics
  decision/                    Decision-simulation outputs
  figures/                     Final plots and summary tables
```

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Setup

Place regional CSV files under:

```text
data/ERCOT/
data/NYISO/
data/PJM/
```

Each config file under `code/configs/` contains a `data_root` field. Update it if your local data directory differs:

```json
"data_root": "/absolute/path/to/decision_focused_carbon/data"
```

## Train Forecasting Models

Train all configured regions:

```bash
cd code
bash run_all.sh
```

Model checkpoints and forecasting metrics are saved under region-specific result folders, for example:

```text
results_ercot/default/<timestamp>/
```

## Export Rolling Forecasts

Export 96-hour rolling forecasts for a region:

```bash
python -m src.export_forecasts \
  --config configs/ercot.json \
  --run_dir results_ercot/default/<timestamp>
```

This generates:

```text
forecasts_val.parquet
```

## Decision-Focused Evaluation

Run the carbon-aware scheduling evaluation:

```bash
python -m src.decision_simulate \
  --run_dir results_ercot/default/<timestamp> \
  --region ERCOT
```

This generates decision-simulation outputs such as:

```text
decision_sim.parquet
```

## Summaries and Plots

Forecasting summaries:

```bash
python summarize_forecast_results.py
```

Decision-performance summaries:

```bash
python summary_decision_results.py
```

Outputs are written to:

```text
results/figures/
results/decision/decision_summary/
```

## Notes

- Large data and model files should be excluded with `.gitignore`.
- The codebase is modular and can be extended to additional regions.
- The decision simulation is stylized and intended for research evaluation, not direct operational deployment.
- This prototype informed my later work on structured fuel-mix surrogate models for grid carbon-intensity forecasting.

## Relationship to Later Work

This project treated carbon intensity as a direct time-series forecasting target and evaluated forecasts through a downstream scheduling task. My later research extends this direction by forecasting carbon intensity through an interpretable fuel-mix layer, connecting carbon-intensity movement to generation composition and power-system data limitations.

## Tools

`Python` `PyTorch` `pandas` `NumPy` `scikit-learn` `Temporal Fusion Transformer` `time-series forecasting` `decision-focused evaluation`

