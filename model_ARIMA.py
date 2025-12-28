# -*- coding: utf-8 -*-

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ignore convergence and other warnings
warnings.filterwarnings("ignore")

# =========================
# 1. Configuration
# =========================
TRAIN_PATH = "delhi_climate_train.csv"
TEST_PATH = "delhi_climate_test.csv"
DATE_COL = "date"
TARGET_COL = "meantemp"
OUTPUT_DIR = Path("Outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. Data Transformation Functions
# =========================
def load_and_transform(path: str):
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).drop_duplicates(subset=[DATE_COL])
    df.set_index(DATE_COL, inplace=True)
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").dropna()

    # Log Transformation + First Differencing
    y_log = np.log(y + 1e-8)
    y_log_diff = y_log.diff().dropna()

    return y_log_diff


# =========================
# 3. Main Execution Flow
# =========================
def main():
    # A. Data Preprocessing
    processed_data = load_and_transform(TRAIN_PATH)
    print(f"Data Preprocessing Complete (Log + Diff). Valid Observations: {len(processed_data)}")

    # B. Grid Search (p: 0-5, q: 0-5)
    p_values = range(0, 10)
    q_values = range(0, 10)
    results_list = []

    print(f"\nStarting Grid Search for ARIMA(p, 1, q)...")

    for p in p_values:
        for q in q_values:
            try:
                # Fitting on differenced data, so d=0
                model = ARIMA(processed_data, order=(p, 0, q))
                res = model.fit()
                results_list.append({
                    'p': p, 'q': q,
                    'AIC': res.aic, 'BIC': res.bic,
                    'order': (p, 0, q)
                })
            except:
                continue

    results_df = pd.DataFrame(results_list)
    if results_df.empty:
        print("Grid Search failed to find a valid model.")
        return

    # ==========================================
    # C. Print Top 4 Model Candidates
    # ==========================================
    top_models = results_df.sort_values('BIC').head(4)
    print("\n" + "=" * 50)
    print("--- Top 4 Model Candidates (Ranked by BIC) ---")
    print(top_models[['order', 'AIC', 'BIC']])
    print("=" * 50)

    # D. Select the Best Model (Rank #1)
    best_row = top_models.iloc[0]
    best_order = best_row['order']

    print(f"\n>>> Best Model: ARIMA{best_order}")

    # E. Diagnostics & Parameter Saving (Best Model Only)
    best_res = ARIMA(processed_data, order=best_order).fit()

    summary_path = OUTPUT_DIR / f"ARIMA_{best_order}_Summary.txt"
    with open(summary_path, "w") as f:
        f.write(best_res.summary().as_text())

    print(f"Summary saved to: {summary_path}")

    # --- 2. Statistical Testing (Ljung-Box) ---
    lb_test = acorr_ljungbox(best_res.resid, lags=[10], return_df=True)
    p_lb = lb_test['lb_pvalue'].values[0]
    print(f"Ljung - Box: p={p_lb: .4f}")

    # --- 3. Diagnostic Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residual ACF
    plot_acf(best_res.resid, lags=40, ax=axes[0])
    axes[0].set_title(f"Residual ACF")

    # Q-Q Plot
    stats.probplot(best_res.resid, dist="norm", plot=axes[1])
    axes[1].set_title(f"Q-Q Plot")

    plt.tight_layout()
    diag_plot_path = OUTPUT_DIR / f"Diagnostic_{best_order}.png"
    plt.savefig(diag_plot_path)
    plt.show()

    # ==========================================
    # F. 5-Day Forecast Module (Based on Best Model)
    # ==========================================
    print("\n" + "=" * 50)
    print("--- 5-Day Forecast vs Actual (Best Model) ---")
    print("=" * 50)

    # Predict next 5 steps in log-diff space
    forecast_log_diff = best_res.forecast(steps=5)

    # Inverse Transformation Logic
    df_train_raw = pd.read_csv(TRAIN_PATH, parse_dates=[DATE_COL])
    df_train_raw = df_train_raw.sort_values(DATE_COL).drop_duplicates(subset=[DATE_COL])
    last_val_actual = df_train_raw[TARGET_COL].iloc[-1]
    last_log_val = np.log(last_val_actual + 1e-8)

    # Reconstruct log values (Cumulative Sum) and Apply Exponential
    forecast_log = last_log_val + np.cumsum(forecast_log_diff)
    forecast_celsius = np.exp(forecast_log)

    # Load Test Data for Validation
    if os.path.exists(TEST_PATH):
        df_test = pd.read_csv(TEST_PATH, parse_dates=[DATE_COL])
        df_test = df_test.sort_values(DATE_COL).drop_duplicates(subset=[DATE_COL])
        y_test_actual = df_test[TARGET_COL].head(5).values
        test_dates = df_test[DATE_COL].head(5).values

        # Calculate Error Metrics
        mae = mean_absolute_error(y_test_actual, forecast_celsius)
        rmse = np.sqrt(mean_squared_error(y_test_actual, forecast_celsius))

        # Prepare and Print Comparison Table
        comp_df = pd.DataFrame({
            'Date': test_dates,
            'Actual (°C)': y_test_actual,
            'Predicted (°C)': forecast_celsius.values,
            'Difference': forecast_celsius.values - y_test_actual
        })
        print(comp_df.to_string(index=False))
        print(f"\nEvaluation Metrics: MAE = {mae:.2f}°C, RMSE = {rmse:.2f}°C")
    else:
        print(f"Test file not found: {TEST_PATH}")


if __name__ == "__main__":
    main()