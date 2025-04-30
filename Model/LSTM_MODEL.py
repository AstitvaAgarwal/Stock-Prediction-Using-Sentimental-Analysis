import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# 🗂 Define the path containing stock CSV files
data_folder = "multi_stock_dashboard/Data/Feature_Engineered/"

csv_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith(".csv")]

# 📊 Load and merge all stock datasets
df_list = []
for file in csv_files:
    temp_df = pd.read_csv(file)
    temp_df["Date"] = pd.to_datetime(temp_df["Date"])  # Ensure Date is in datetime format
    temp_df["Stock"] = os.path.basename(file).split("_")[0]  # Identify stock symbol
    df_list.append(temp_df)

# 📌 Combine all stock data
df = pd.concat(df_list, ignore_index=True)

# 🔄 Sort by Date
df = df.sort_values(["Date", "Stock"])

# 🎯 Select relevant features
features = ["Close", "SMA_14", "EMA_14", "RSI_14", "MACD", "MACD_Signal", "VADER_Score", "TextBlob_Score"]
df = df[features].dropna()  # Drop NaN values

# 🔬 Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# 📏 Create sequences for LSTM
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length, 0])  # Predicting 'Close' price
    return np.array(X), np.array(y)

SEQ_LENGTH = 50
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# 🏋️ Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 🏆 Train-test split (80-20)
train_size = int(0.8 * len(X))
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# 🔄 Create PyTorch DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 🎯 Define LSTM Model
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Take the last time step's output
        return out

# 🚀 Initialize Model
input_size = X.shape[2]  # Number of features
model = StockLSTM(input_size)

# 🛠 Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 🎯 Train Model
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        y_pred = model(batch_X)
        loss = criterion(y_pred.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.6f}")

# 🔍 Make Predictions
model.eval()
y_pred_list = []
with torch.no_grad():
    for batch_X, _ in test_loader:
        batch_pred = model(batch_X)
        y_pred_list.append(batch_pred.numpy())

y_pred = np.concatenate(y_pred_list, axis=0)

# 📉 Reverse Scaling
y_test_inv = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((len(y_test), len(features) - 1))]))[:, 0]
y_pred_inv = scaler.inverse_transform(np.hstack([y_pred, np.zeros((len(y_pred), len(features) - 1))]))[:, 0]

# 📊 Evaluate Model
mse = mean_squared_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_inv, y_pred_inv)

print(f"\n📊 Multi-Stock PyTorch LSTM Model Performance:")
print(f"✅ MSE: {mse:.4f}")
print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ R² Score: {r2:.4f}")

