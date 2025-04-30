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

# 🔍 Get all stock CSV files
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

# 🔄 Sort by Date and Stock
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

#  Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

#  Train-test split (80-20)
train_size = int(0.8 * len(X))
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

#  Create PyTorch DataLoader
batch_size = 64  # Increased batch size for stability
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#  Define Optimized Bi-LSTM Model
class BiLSTMStockModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.4):
        super(BiLSTMStockModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.batch_norm(lstm_out[:, -1, :])  # Batch Normalization
        out = self.fc(out)  # Fully Connected Layer
        return out

# 🔄 Initialize Bi-LSTM model
input_size = X.shape[2]  # Number of features
model = BiLSTMStockModel(input_size, hidden_size=256, num_layers=3, dropout=0.4)

# 🏋️ Use AdamW optimizer with weight decay (L2 Regularization)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

# 📁 Create directory for saving models
model_dir = "BiLSTM_MODELS"
os.makedirs(model_dir, exist_ok=True)

# 🔍 Train Model with More Epochs
epochs = 30
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
    
    avg_loss = total_loss / len(train_loader)
    lr_scheduler.step()  # Adjust learning rate if needed

    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

    # Save model at every 10th epoch
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"{model_dir}/bi_lstm_epoch_{epoch+1}.pth")

# 🔬 Evaluate Model
model.eval()
y_pred_list = []
with torch.no_grad():
    for batch_X, _ in test_loader:
        batch_pred = model(batch_X)
        y_pred_list.append(batch_pred.numpy())

y_pred = np.concatenate(y_pred_list, axis=0)

# 📊 Reverse Scaling & Evaluation
y_pred_inv = scaler.inverse_transform(np.hstack([y_pred, np.zeros((len(y_pred), len(features) - 1))]))[:, 0]
y_test_inv = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((len(y_test), len(features) - 1))]))[:, 0]

# 📈 Calculate Metrics
mse = mean_squared_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_inv, y_pred_inv)

# 📊 Display Results
print(f"\n📊 Optimized Bi-LSTM Performance:")
print(f"✅ MSE: {mse:.4f}")
print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ R² Score: {r2:.4f}")

# 📝 Save final model
torch.save(model.state_dict(), f"{model_dir}/final_bi_lstm.pth")
print(f"✅ Model saved at {model_dir}/final_bi_lstm.pth")