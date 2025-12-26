import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# ==========================================
# 1. Data Loading and Preprocessing
# ==========================================
# Define input paths
train_path = r'Original data/DailyDelhiClimateTrain.csv'
test_path = r'Original data/DailyDelhiClimateTest.csv'

# Load datasets
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Combine datasets and handle date conversion
df_combined = pd.concat([df_train, df_test], ignore_index=True)
df_combined['date'] = pd.to_datetime(df_combined['date'])
df_combined = df_combined.sort_values(by='date').reset_index(drop=True)

# Split the data: Use the last 5 days for validation, the rest for modeling
validation_set = df_combined.tail(5)
model_data = df_combined.iloc[:-5].copy()

# Save processed datasets
df_combined.to_csv('delhi_climate_full.csv', index=False)
model_data.to_csv('delhi_climate_train.csv', index=False)
validation_set.to_csv('delhi_climate_test.csv', index=False)
print("--- Data Splitting Complete: Validation set extracted ---")

# ==========================================
# 2. Data Quality and Descriptive Statistics
# ==========================================
# Assign model_data to df_analysis for further processing
df_analysis = model_data.copy()
df_analysis.set_index('date', inplace=True)

# Extract target series (Mean Temperature)
temp_series = df_analysis['meantemp'].dropna()

# Statistical summary
stats_summary = {
    "Record Count": temp_series.size,
    "Mean": temp_series.mean(),
    "Variance": temp_series.var(),
    "Minimum": temp_series.min(),
    "Maximum": temp_series.max()
}

print("\n--- Descriptive Statistics Results ---")
for key, value in stats_summary.items():
    print(f"{key}: {value:.2f}")

# Check for date continuity
expected_range = pd.date_range(start=temp_series.index.min(), end=temp_series.index.max(), freq='D')
missing_days = expected_range.difference(temp_series.index)
print(f"\n--- Data Quality Check ---")
print(f"Number of missing dates: {len(missing_days)}")

# ==========================================
# 3. Visual Analysis
# ==========================================
# Outlier detection using Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=temp_series, color='lightblue')
plt.title('Temperature Box Plot (Outlier Detection)')
plt.xlabel('Temperature')
plt.show()

# Volatility check (30-day rolling standard deviation)
plt.figure(figsize=(12, 4))
temp_series.rolling(window=30).std().plot(color='purple', label='30-Day Rolling Std')
plt.title('30-Day Rolling Standard Deviation')
plt.ylabel('Standard Deviation')
plt.legend()
plt.show()

# ==========================================
# 4. Stationarity Testing (ADF Test)
# ==========================================
print('\n--- Augmented Dickey-Fuller (ADF) Test (Original Series) ---')
#
adf_result = adfuller(temp_series)
print(f'ADF Statistic: {adf_result[0]:.4f}')
print(f'p-value: {adf_result[1]:.4f}')

if adf_result[1] > 0.05:
    print("Conclusion: The series is non-stationary. Differencing is recommended.")
else:
    print("Conclusion: The series is stationary.")

# Plot Autocorrelation (ACF) and Partial Autocorrelation (PACF)
#
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
plot_acf(temp_series, lags=40, ax=axes[0], title='Autocorrelation (ACF)')
plot_pacf(temp_series, lags=40, ax=axes[1], title='Partial Autocorrelation (PACF)')
plt.tight_layout()
plt.show()