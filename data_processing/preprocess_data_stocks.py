import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# Define input & output folders
stock_data_folder = r"multi_stock_dashboard\Data\Stocks"
preprocessed_folder = r"multi_stock_dashboard\Data\Preprocessed_Stocks"
os.makedirs(preprocessed_folder, exist_ok=True)

# List all stock data files
stock_files = [f for f in os.listdir(stock_data_folder) if f.endswith(".csv")]

# Processing Pipeline
def preprocess_stock_data(file_path):
    """Load, clean, and normalize stock data."""
    df = pd.read_csv(file_path)

    # Show raw data preview
    print("\n📜 Raw Data Sample:")
    print(df.head())

    # Convert 'Date' column to datetime and sort
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(by="Date")

    # Ensure numeric columns
    df = df.infer_objects(copy=False)

    # Display column data types
    print("\n📝 Column Data Types:")
    print(df.dtypes)

    # Select only numeric columns
    numeric_columns = df.select_dtypes(include=["number"]).columns

    # Debugging: Show detected numeric columns
    print("\n🔢 Numeric Columns Detected:", list(numeric_columns))

    if numeric_columns.empty:
        print("\n⚠ Warning: No numeric columns found! Skipping scaling.")
    else:
        # Handle missing values (Fill Forward & Interpolation)
        df[numeric_columns] = df[numeric_columns].ffill().bfill()
        df[numeric_columns] = df[numeric_columns].interpolate(method="linear")

        # Normalize only numeric columns
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Show processed data preview
    print("\n✅ Processed Data Sample:")
    print(df.head())

    return df

# Process all stock files in a loop
for file in stock_files:
    input_path = os.path.join(stock_data_folder, file)
    output_path = os.path.join(preprocessed_folder, file)

    print(f"\n📌 Processing: {file}")

    processed_df = preprocess_stock_data(input_path)

    # Save preprocessed dataset
    processed_df.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path}")

print("\n🚀 Preprocessing complete! All stock data is cleaned and normalized.")
