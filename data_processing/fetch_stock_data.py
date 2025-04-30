# Install required libraries
# !pip install yfinance pandas beautifulsoup4 requests lxml

import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

# Define stock tickers for 5 companies
stocks = {
    "Tata Motors": "TATAMOTORS.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Hyundai": "005380.KQ",
    "Maruti Suzuki": "MARUTI.NS",
    "Ashok Leyland": "ASHOKLEY.NS"
}

# # Define time period (last 1 year)
# start_date = "2023-03-01"
# end_date = "2024-03-01"

from datetime import datetime, timedelta

# Calculate past two years from today (March 30, 2025)
today = datetime(2025, 3, 30)
start_date = today - timedelta(days=2*365)

# Format dates as YYYY-MM-DD
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = today.strftime("%Y-%m-%d")

print(f"Start Date: {start_date_str}")
print(f"End Date: {end_date_str}")

# Create a folder for stock data
os.makedirs("Stock_Data", exist_ok=True)
os.makedirs("News_Data", exist_ok=True)

# Fetch stock data for each company
for company, ticker in stocks.items():
    print(f"\n📌 Fetching stock data for: {company} ({ticker})")
    
    df = yf.download(ticker, start=start_date_str, end=end_date_str)
    
    if df.empty:
        print(f"❌ No data found for {company}! Skipping CSV save.\n")
    else:
        print(f"✅ {company} Data Fetched! Showing first 5 rows:")
        print(df.head(), "\n")
        
        df["Company"] = company  
        df["Date"] = df.index  

        # Save as a separate CSV file
        file_path = f"Stock_Data/{company.replace(' ', '_')}_stock.csv"
        df.to_csv(file_path, index=False)
        print(f"📂 Saved: {file_path}\n")

# Function to get news using Google News RSS
def get_news(company):
    """Scrape news headlines from Google News RSS."""
    base_url = f"https://news.google.com/rss/search?q={company.replace(' ', '+')}+stock&hl=en-IN&gl=IN&ceid=IN:en"
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, "xml")

    news_list = []
    for item in soup.findAll("item")[:10]:  # Get latest 10 articles
        title = item.title.text
        link = item.link.text
        pub_date = item.pubDate.text
        news_list.append([company, title, link, pub_date])

    return news_list

# Fetch news for each company
for company in stocks.keys():
    print(f"\n📰 Fetching news for: {company}")
    news = get_news(company)
    
    if not news:
        print(f"❌ No news found for {company}! Skipping CSV save.\n")
    else:
        print(f"✅ Found {len(news)} news articles for {company}")

        # Convert to DataFrame
        news_df = pd.DataFrame(news, columns=["Company", "Headline", "URL", "Published Date"])
        
        # Save as a separate CSV file
        file_path = f"News_Data/{company.replace(' ', '_')}_news.csv"
        news_df.to_csv(file_path, index=False)
        print(f"📂 Saved: {file_path}\n")

print("\n✅ All stock & news data collected and saved separately! 🎉")
import shutil

# Define the output zip file name
zip_file_name = "/content/Stock_News_Data.zip"

# Create a zip file containing both folders
shutil.make_archive(zip_file_name.replace(".zip", ""), 'zip', "/content")

print(f"✅ ZIP file created: {zip_file_name}")