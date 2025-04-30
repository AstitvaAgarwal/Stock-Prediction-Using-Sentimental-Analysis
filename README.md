# 📈 Stock Market Prediction and Analysis System

## 🚀 Overview
This project is a **hybrid stock market forecasting system** that integrates **deep learning**, **technical indicators**, and **financial news sentiment analysis** to predict stock prices of **major automotive companies**. By combining historical stock patterns with real-time sentiment data, it offers an intelligent, adaptive, and data-driven decision-making tool for investors and analysts.

## 🔧 Key Components

### 📉 Data Sources
- **Yahoo Finance API** (`yfinance`) for historical stock data (e.g., Tata Motors, Maruti Suzuki)
- **Google News RSS** feeds for recent financial headlines

### 🔄 Data Pipeline
- **Stock Data Preprocessing**:
  - Date parsing, missing value handling
  - Normalization with `MinMaxScaler`
  - Technical indicators: SMA, EMA, RSI, MACD, MACD Signal
- **News Preprocessing**:
  - Sentiment scores using **VADER** and **TextBlob**
  - Merged with stock data by date

### 🧠 Model Architecture
- **Bi-Directional LSTM (Bi-LSTM)** for capturing sequential dependencies
- Trained on 50-day sequences of technical + sentiment features
- Evaluated using:
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - R² Score

### 📊 Dashboard
Built with **Streamlit**, the dashboard allows users to:
- Visualize actual vs. predicted prices
- Track sentiment trends
- Simulate key financial ratios: **P/E**, **ROE**
- (Optional) View AI-generated news summaries

---

## 🎯 Objectives
- 🧩 Develop a hybrid model combining quantitative indicators and qualitative sentiment
- 🎯 Enhance prediction accuracy over traditional models
- 📡 Provide real-time insights responsive to market sentiment
- ⚙️ Create a modular, scalable architecture for multi-stock support

---

## 🧪 Experimental Results
- The Bi-LSTM model showed **high accuracy** during stable market periods
- Sentiment integration improved responsiveness to news-driven price shifts
- Outperformed traditional methods (ARIMA, simple LSTM) in adaptability and precision

---

## ⚠️ Limitations
- Dependent on timely and quality news data
- Sentiment scores from short headlines may lack full context
- No intraday or social media sentiment currently incorporated

---

## 🔮 Future Enhancements
- ✅ Add support for multiple industries and additional stocks
- 🔄 Integrate real-time data feeds and social sentiment (e.g., Twitter, Reddit)
- 🧠 Experiment with transformer models (e.g., BERT, GPT, FinBERT)
- 🔔 Create real-time alert and trading integration
- 🧩 Incorporate AI-generated summaries using LLaMA or similar models

---

## 📦 Requirements
- Python 3.8+
- `pandas`, `numpy`, `yfinance`, `nltk`, `textblob`, `vaderSentiment`
- `tensorflow`, `keras`, `scikit-learn`
- `streamlit`, `matplotlib`, `seaborn`

---

## 🛠️ Setup Instructions
```bash
git clone https://github.com/yourusername/automotive-stock-predictor.git
cd automotive-stock-predictor
pip install -r requirements.txt
streamlit run dashboard.py
