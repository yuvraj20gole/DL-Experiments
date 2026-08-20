# Time Series Forecasting — LSTM on Airline Passengers

College deep learning assignment: forecast monthly airline passengers with a stacked LSTM, lag features (sliding window), and RMSE evaluation.

## Objective

- Load the classic AirPassengers time series (Jan 1949 – Dec 1960)
- Scale the series and build lag features with a 12-month lookback
- Train a two-layer LSTM and evaluate train/test RMSE
- Visualize forecast vs actual and the loss curve

## Dataset

**Airline Passengers** (`data/airline_passengers.csv`): 144 monthly totals, columns `Month` / `Passengers`.

On first run, if the CSV is missing locally, the notebook auto-fetches it from:

https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv

and saves it to `data/airline_passengers.csv`.

## Project structure

```
time-series-forecasting/
├── data/
│   └── airline_passengers.csv
├── notebooks/
│   └── 01_time_series_lstm.ipynb
├── models/
├── results/
├── requirements.txt
├── README.md
└── .gitignore
```

## How to run

Use the existing native **arm64** Python venv (avoids Rosetta/TensorFlow crashes on Apple Silicon):

```bash
cd "DL Experiment 6/time-series-forecasting"
# activate your arm64 venv, then:
pip install -r requirements.txt
jupyter notebook notebooks/01_time_series_lstm.ipynb
```

Or run all cells from the terminal:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_time_series_lstm.ipynb \
  --ExecutePreprocessor.kernel_name=arm64-venv \
  --inplace
```

After a successful run you should see:

- `models/lstm_forecast.keras` (or `.h5`)
- `results/forecast_metrics.csv`
- `results/forecast_vs_actual.png`
- `results/loss_curve.png`
