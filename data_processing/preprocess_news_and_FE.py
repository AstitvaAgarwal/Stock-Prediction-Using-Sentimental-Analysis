import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Define input & output folders
preprocessed_stocks_folder = r"multi_stock_dashboard\Data\Preprocessed_Stocks"
news_folder = r"multi_stock_dashboard\Data\News"
feature_engineered_folder = r"multi_stock_dashboard\Data\Feature_Engineered"
os.makedirs(feature_engineered_folder, exist_ok=True)

# List stock and news files
stock_files = {f.split(".")[0]: f for f in os.listdir(preprocessed_stocks_folder) if f.endswith(".csv")}
news_files = {f.split(".")[0]: f for f in os.listdir(news_folder) if f.endswith(".csv")}

# Initialize Sentiment Analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Function to clean numeric columns
def convert_numeric(df, columns):
    """Convert specified columns to numeric, coercing errors."""
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Function to calculate technical indicators
def calculate_technical_indicators(df):
    """Calculate SMA, EMA, RSI, and MACD indicators on numeric columns."""
    if "Close" not in df.columns:
        return df  # Skip if "Close" price is missing

    # Convert Close to numeric
    df = convert_numeric(df, ["Close"])

    # SMA & EMA
    df["SMA_14"] = df["Close"].rolling(window=14).mean()
    df["EMA_14"] = df["Close"].ewm(span=14, adjust=False).mean()

    # RSI Calculation
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD Calculation
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df

# Function to compute sentiment scores
def analyze_sentiment(text):
    """Compute sentiment scores using VADER and TextBlob."""
    if not isinstance(text, str) or text.strip() == "":
        return 0.0, 0.0  # Return neutral scores if text is empty

    vader_score = sentiment_analyzer.polarity_scores(text)["compound"]
    textblob_score = TextBlob(text).sentiment.polarity
    return vader_score, textblob_score

# Processing loop
for stock_name, stock_file in stock_files.items():
    news_key = stock_name.replace("_stock", "_news")  # Ensure correct match

    if news_key not in news_files:
        print(f"⚠ Skipping {stock_name}: No matching news data found! (Expected: {news_key})")
        continue  # Skip if no corresponding news file

    stock_path = os.path.join(preprocessed_stocks_folder, stock_file)
    news_path = os.path.join(news_folder, news_files[news_key])  # Use corrected key

    print(f"\n📌 Processing: {stock_name}")

    # Load stock and news data
    stock_df = pd.read_csv(stock_path)
    news_df = pd.read_csv(news_path)

    # Convert 'Date' to datetime
    if "Date" in stock_df.columns:
        stock_df["Date"] = pd.to_datetime(stock_df["Date"])

    # Fix column names for news_df
    if "Published Date" in news_df.columns:
        news_df.rename(columns={"Published Date": "Date"}, inplace=True)

    if "Date" in news_df.columns:
        news_df["Date"] = pd.to_datetime(news_df["Date"])

    # Select only numeric columns
    numeric_cols = stock_df.select_dtypes(include=[np.number]).columns.tolist()
    stock_df = convert_numeric(stock_df, numeric_cols)

    # Calculate technical indicators
    stock_df = calculate_technical_indicators(stock_df)

    # Extract sentiment scores
    if "Headline" in news_df.columns:
        news_df["VADER_Score"], news_df["TextBlob_Score"] = zip(*news_df["Headline"].apply(analyze_sentiment))
    else:
        print(f"⚠ Skipping sentiment analysis for {stock_name}: 'Headline' column missing.")
        news_df["VADER_Score"], news_df["TextBlob_Score"] = 0.0, 0.0  # Assign neutral values

    # Ensure required columns exist before merging
    if not all(col in news_df.columns for col in ["Date", "VADER_Score", "TextBlob_Score"]):
        print(f"⚠ Skipping merging for {stock_name}: Missing required sentiment columns.")
        continue

    # Merge stock data with sentiment data on 'Date'
    merged_df = pd.merge(stock_df, news_df[["Date", "VADER_Score", "TextBlob_Score"]], on="Date", how="left")

    # Fill missing sentiment values with neutral scores
    merged_df[["VADER_Score", "TextBlob_Score"]] = merged_df[["VADER_Score", "TextBlob_Score"]].fillna(0)

    # Save processed dataset
    output_path = os.path.join(feature_engineered_folder, f"{stock_name}_features.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path}")

print("\n🚀 Feature Engineering Complete! All datasets are enriched with indicators & sentiment scores.")
