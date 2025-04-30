import os
import torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn

# 📂 **Define Paths**
BASE_DIR = r"C:\Users\Admin01\Downloads\Sriagnth_dashboard\multi_stock_dashboard"
base_data_path = os.path.join(BASE_DIR, "Data")
stock_data_path = os.path.join(base_data_path, "Feature_Engineered")
news_data_path = os.path.join(base_data_path, "News")

# ✅ **Check Paths**
if not os.path.exists(stock_data_path):
    st.error(f"❌ Stock data folder not found at: {stock_data_path}")
    st.stop()

if not os.path.exists(news_data_path):
    st.warning(f"⚠️ News data folder missing! Sentiment analysis will be skipped.")

# ✅ **Load Stock Files**
stock_files = [f for f in os.listdir(stock_data_path) if f.endswith("_stock_features.csv")]
if not stock_files:
    st.warning("⚠️ No stock data files found in Feature_Engineered!")
    st.stop()

# 🌟 **Define LSTM Model**
class LSTMStockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, output_size=1, bidirectional=True):
        super(LSTMStockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.batch_norm = nn.BatchNorm1d(hidden_size * (2 if bidirectional else 1))
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers * (2 if self.lstm.bidirectional else 1), x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers * (2 if self.lstm.bidirectional else 1), x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.batch_norm(out[:, -1, :])
        return self.fc(out)

# 🎯 **Load LSTM Model**
st.write("📦 Loading LSTM model...")
model_path = os.path.join(BASE_DIR, "final_stock_lstm.pth")

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at {model_path}")
    st.stop()

model = LSTMStockPredictor(input_size=8)
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    st.success("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Model loading failed: {str(e)}")
    st.stop()

# 📊 **Streamlit UI**
st.title("📈 Multi-Stock Price Prediction & Sentiment Analysis Dashboard")

# ✅ **Required Features**
required_columns = ["Close", "SMA_14", "EMA_14", "RSI_14", "MACD", "MACD_Signal", "VADER_Score", "TextBlob_Score"]

# 🎯 **Function to Compute Sentiment Score**
def compute_sentiment(text):
    if pd.isna(text) or text.strip() == "":
        return 0  # **Neutral sentiment for missing headlines**
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)["compound"]

# 🔄 **Process Each Stock**
st.write("🔄 Processing stock data...")

fig = go.Figure()
predictions_list = []

for stock_file in stock_files:
    stock_name = stock_file.replace("_stock_features.csv", "")

    # 📂 **Load Stock Data**
    stock_df = pd.read_csv(os.path.join(stock_data_path, stock_file))
    
    # 🔍 **Check for 'Date' Column and Rename if Needed**
    stock_df.columns = stock_df.columns.str.strip()
    if "Date" not in stock_df.columns:
        st.warning(f"⚠️ 'Date' column missing in {stock_name}. Skipping...")
        continue

    stock_df["Date"] = pd.to_datetime(stock_df["Date"], errors="coerce")
    stock_df.dropna(subset=required_columns, inplace=True)

    # ✅ **Scale Features**
    features_to_scale = stock_df[required_columns]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features_to_scale)

    # ✅ **Prepare Data for LSTM**
    def prepare_input_data(new_data, seq_length=50):
        if new_data.shape[0] < seq_length:
            return None
        X_new = [new_data[i : i + seq_length] for i in range(len(new_data) - seq_length)]
        return torch.tensor(np.array(X_new), dtype=torch.float32) if X_new else None

    # 🔍 **Function to Predict Stock Price**
    def predict_stock_price(new_stock_data):
        new_X = prepare_input_data(new_stock_data)
        if new_X is None:
            return None  
        with torch.no_grad():
            y_pred = model(new_X).numpy()
            print(f"Raw Predictions: {y_pred}")  # 🛠 Debugging
        y_pred_inv = scaler.inverse_transform(
            np.hstack([y_pred, np.zeros((len(y_pred), len(required_columns) - 1))])
        )[:, 0]
        return y_pred_inv[-1]

    # ✅ **Predict Next Closing Price**
    predicted_price = predict_stock_price(scaled_features)
    if predicted_price is None:
        st.warning(f"⚠️ Skipping {stock_name} - Not enough data for prediction")
        continue

    last_actual_price = stock_df["Close"].iloc[-1]
    price_change = predicted_price - last_actual_price
    price_change_pct = (price_change / last_actual_price) * 100
    price_direction = "🔼 UP" if price_change > 0 else "🔻 DOWN"

    # 📂 **Load News Sentiment Data**
    news_file = os.path.join(news_data_path, f"{stock_name}_news.csv")
    if os.path.exists(news_file):
        news_df = pd.read_csv(news_file)
        
        # 🔍 **Ensure 'Date' Exists in News Data**
        news_df.columns = news_df.columns.str.strip()
        if "Date" not in news_df.columns:
            st.warning(f"⚠️ 'Date' column missing in {stock_name} news. Assigning default dates...")
            news_df["Date"] = pd.to_datetime("today")

        if "Sentiment" not in news_df.columns and "Headline" in news_df.columns:
            news_df["Sentiment"] = news_df["Headline"].apply(compute_sentiment)

        latest_news = news_df[["Date", "Headline", "Sentiment"]].tail(1).to_dict(orient="records")
        news_summary = latest_news[0]["Headline"] if latest_news else "No news available"
    else:
        news_summary = "No news data"

    # 📉 **Store Predictions**
    predictions_list.append([stock_name, last_actual_price, predicted_price, price_change_pct, price_direction, news_summary])

    # 📊 **Add to Plot**
    fig.add_trace(go.Scatter(x=stock_df["Date"][-100:], y=stock_df["Close"][-100:], mode="lines", name=f"{stock_name} (Actual)"))
    fig.add_trace(go.Scatter(x=[stock_df["Date"].iloc[-1]], y=[predicted_price], mode="markers", name=f"{stock_name} (Predicted)", marker=dict(size=8, color="red")))

# 📊 **Display Stock Trends**
st.subheader("📊 Stock Price Trends (Last 100 Days)")
fig.update_layout(title="Stock Price Trends", xaxis_title="Date", yaxis_title="Stock Price")
st.plotly_chart(fig)

# 📌 **Display Predictions Table**
st.subheader("📉 Predicted Stock Prices")
predictions_df = pd.DataFrame(predictions_list, columns=["Stock", "Actual Price", "Predicted Price", "% Change", "Direction", "News Summary"])
st.dataframe(predictions_df)
