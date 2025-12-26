import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Configuration & Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOOKBACK = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 80
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
FORECAST_STEPS = 5

# --- 1. Data Preparation (Train & Test Files) ---

# Load training/model data
df_train = pd.read_csv('delhi_climate_train.csv')
df_train['date'] = pd.to_datetime(df_train['date'])
df_train = df_train.sort_values('date')
train_series = df_train['meantemp'].dropna()

# Load the separate 5-day test file for final comparison
df_test = pd.read_csv('delhi_climate_test.csv')
df_test['date'] = pd.to_datetime(df_test['date'])
df_test = df_test.sort_values('date')
true_5_values = df_test['meantemp'].values[:FORECAST_STEPS]  # Take first 5 days

# Scale data based on training set ONLY (to avoid data leakage)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_series.values.reshape(-1, 1))


# Function to create sequences
def create_sequences(data, seq_size):
    x, y = [], []
    for i in range(len(data) - seq_size):
        x.append(data[i: i + seq_size])
        y.append(data[i + seq_size])
    return torch.tensor(np.array(x), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)


X, y = create_sequences(scaled_train_data, LOOKBACK)

# Split model_data for internal training and validation (9:1 ratio)
n_obs = len(X)
split_idx = int(0.9 * n_obs)
X_train, y_train = X[:split_idx], y[:split_idx]
X_val, y_val = X[split_idx:], y[split_idx:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)


# --- 2. Model Architecture ---
class OptimizedLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, dropout=0.3):
        super(OptimizedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


model = OptimizedLSTM(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)

# --- 3. Optimizer, Loss, and Scheduler ---
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# --- 4. Training Loop ---
train_losses, val_losses = [], []
print(f"Training on {DEVICE}...")

for epoch in range(EPOCHS):
    model.train()
    batch_train_losses = []
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        batch_train_losses.append(loss.item())

    model.eval()
    batch_val_losses = []
    with torch.no_grad():
        for x_v, y_v in val_loader:
            x_v, y_v = x_v.to(DEVICE), y_v.to(DEVICE)
            v_loss = criterion(model(x_v), y_v)
            batch_val_losses.append(v_loss.item())

    avg_train_loss = np.mean(batch_train_losses)
    avg_val_loss = np.mean(batch_val_losses)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    scheduler.step(avg_val_loss)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

# --- 5. Final Recursive Forecast (Using the end of model_data to predict test_data) ---
model.eval()
# The starting point for future prediction is the last 30 days of our training file
last_window = scaled_train_data[-LOOKBACK:]
current_batch = torch.tensor(last_window.reshape(1, LOOKBACK, 1), dtype=torch.float32).to(DEVICE)

future_forecasts = []
for _ in range(FORECAST_STEPS):
    with torch.no_grad():
        next_pred = model(current_batch)
        future_forecasts.append(next_pred.item())
        new_val = next_pred.view(1, 1, 1)
        current_batch = torch.cat((current_batch[:, 1:, :], new_val), dim=1)

# Inverse transform to actual temperature
final_5_forecast = scaler.inverse_transform(np.array(future_forecasts).reshape(-1, 1)).flatten()

# --- 6. Results Comparison with delhi_climate_test.csv ---
comparison_df = pd.DataFrame({
    'Date': df_test['date'].values[:FORECAST_STEPS],
    'True Value (Test File)': true_5_values,
    'LSTM Prediction': final_5_forecast,
    'Absolute Error': np.abs(true_5_values - final_5_forecast)
})

print("\n--- Final Evaluation against delhi_climate_test.csv ---")
print(comparison_df)
print(f"\nMean Absolute Error over 5 days: {comparison_df['Absolute Error'].mean():.4f}")

# --- 7. Visualization ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Learning Curves')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(true_5_values, 'ko-', label='True Values (Test File)')
plt.plot(final_5_forecast, 'bs--', label='LSTM Forecast')
plt.title('Final 5-Day Forecast Comparison')
plt.xlabel('Days')
plt.ylabel('Temperature')
plt.legend()

plt.tight_layout()
plt.show()