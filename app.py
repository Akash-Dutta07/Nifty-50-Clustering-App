import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import time
import random
import os # For OMP_NUM_THREADS

# --- Configuration ---
st.set_page_config(page_title="Nifty 50 Clustering App", layout="wide")
st.title("📊 Nifty 50 Clustering App")
st.markdown("Analyze Nifty 50 stock clusters based on historical return patterns.")

# Suppress the MKL warning if it's still appearing (optional, as it's just a warning)
# This should be at the very top before any numpy/scikit-learn imports if you want to be pedantic,
# but usually works fine here as well.
os.environ["OMP_NUM_THREADS"] = "1"

# --- Sidebar Inputs ---
n_clusters = st.sidebar.slider("Select Number of Clusters", 2, 10, 4)
n_years = st.sidebar.slider("Years of Historical Data", 1, 10, 2) # Reduced default n_years for faster initial load

# Define tickers
# It's good practice to ensure this list has exactly 50 tickers if possible,
# or adjust the description to reflect the actual number you have.
nifty50_tickers = [
    'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS',
    'KOTAKBANK.NS', 'LT.NS', 'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS',
    'AXISBANK.NS', 'BAJFINANCE.NS', 'HCLTECH.NS', 'ASIANPAINT.NS',
    'MARUTI.NS', 'SUNPHARMA.NS', 'NESTLEIND.NS', 'ULTRACEMCO.NS',
    'TECHM.NS', 'TITAN.NS', 'WIPRO.NS', 'POWERGRID.NS', 'BAJAJFINSV.NS',
    'NTPC.NS', 'ONGC.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'ADANIPORTS.NS',
    'HDFCLIFE.NS', 'GRASIM.NS', 'CIPLA.NS', 'DRREDDY.NS', 'DIVISLAB.NS',
    'BHARTIARTL.NS', 'EICHERMOT.NS', 'BRITANNIA.NS', 'SHREECEM.NS',
    'SBILIFE.NS', 'BPCL.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS',
    'INDUSINDBK.NS', 'TATAMOTORS.NS', 'HINDALCO.NS', 'UPL.NS',
    'TATASTEEL.NS', 'ADANIENT.NS', 'APOLLOHOSP.NS', 'M&M.NS',
    'ICICIPRULI.NS'
]

# --- Data Fetching Function with Caching and Robustness ---
@st.cache_data(ttl=3600, show_spinner="📥 Fetching and processing stock data...") # Cache data for 1 hour (3600 seconds)
def fetch_and_process_data(tickers, n_years_hist):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * n_years_hist)

    all_adj_close_data = pd.DataFrame()
    
    # Attempt bulk download
    bulk_download_success = False
    with st.spinner(f"Attempting bulk download for {len(tickers)} tickers ({n_years_hist} years)..."):
        for attempt in range(3): # Try bulk download 3 times
            try:
                # Removed 'group_by' and 'interval' which are not always reliable or necessary
                # Use progress=False to avoid cluttering Streamlit logs
                df_bulk = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
                
                # Check if 'Adj Close' is directly accessible or if it's a MultiIndex
                if isinstance(df_bulk.columns, pd.MultiIndex):
                    if 'Adj Close' in df_bulk.columns.get_level_values(0):
                        all_adj_close_data = df_bulk['Adj Close'].copy()
                    else: # If 'Adj Close' is not the top level, try to infer or use 'Close' if auto_adjust was true
                        all_adj_close_data = df_bulk.loc[:, (slice(None), 'Close')].droplevel(1, axis=1).copy()
                elif 'Close' in df_bulk.columns: # Single ticker download or already flattened
                    all_adj_close_data = df_bulk['Close'].copy()
                else:
                    st.warning(f"Bulk download attempt {attempt+1}: No 'Adj Close' or 'Close' data found in bulk result. Retrying...")
                    raise ValueError("No relevant columns found in bulk download.")

                if not all_adj_close_data.empty:
                    bulk_download_success = True
                    break
                else:
                    st.warning(f"Bulk download attempt {attempt+1}: Dataframe is empty. Retrying...")

            except Exception as e:
                st.warning(f"Bulk download error (attempt {attempt+1}): {e}. Retrying after delay...")
            time.sleep(2 ** attempt + random.uniform(0, 2)) # Exponential backoff with jitter

    # --- Fallback to Individual Downloads if Bulk Fails or is Incomplete ---
    tickers_to_fetch_individually = []
    if not bulk_download_success:
        st.info("Bulk download failed or was incomplete. Falling back to individual ticker downloads (slower, but more robust).")
        tickers_to_fetch_individually = tickers
    else:
        # Identify missing tickers from the successful bulk download (if any)
        downloaded_tickers = all_adj_close_data.columns.tolist()
        tickers_to_fetch_individually = [t for t in tickers if t not in downloaded_tickers]
        if tickers_to_fetch_individually:
            st.warning(f"Some tickers missing from bulk download. Fetching individually: {len(tickers_to_fetch_individually)} tickers.")

    if tickers_to_fetch_individually:
        individual_progress_bar = st.progress(0)
        successful_individual_fetches = 0
        total_individual_fetches = len(tickers_to_fetch_individually)

        for i, ticker in enumerate(tickers_to_fetch_individually):
            try:
                ticker_df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if not ticker_df.empty and 'Close' in ticker_df.columns:
                    all_adj_close_data[ticker] = ticker_df['Close']
                    successful_individual_fetches += 1
                else:
                    st.warning(f"⚠️ Could not fetch 'Close' data for {ticker} individually.")
            except Exception as e:
                st.error(f"❌ Error fetching {ticker} individually: {e}")
            
            individual_progress_bar.progress((i + 1) / total_individual_fetches)
            time.sleep(random.uniform(5, 10)) # Increased delay for individual fetches

        if successful_individual_fetches > 0:
            st.success(f"Successfully fetched data for {successful_individual_fetches} additional tickers.")
        else:
            st.error("No additional tickers were successfully fetched individually.")

    if all_adj_close_data.empty:
        st.error("❌ No valid stock data could be fetched after all attempts. Please check ticker list or try again later.")
        return None

    # --- Data Cleaning and Transformation ---
    # Drop columns that are entirely NaN (e.g., if a ticker completely failed to download)
    all_adj_close_data = all_adj_close_data.dropna(axis=1, how='all') 
    
    if all_adj_close_data.empty:
        st.error("❌ All stock data columns are empty after initial processing. Cannot proceed.")
        return None

    returns = all_adj_close_data.pct_change().dropna()
    
    if returns.empty:
        st.error("❌ No valid return data after calculating percentage change and dropping NaNs. Check data range or ticker validity.")
        return None

    returns_T = returns.T # Transpose for clustering stocks

    return returns_T

# --- Main App Logic ---
returns_T_data = fetch_and_process_data(nifty50_tickers, n_years)

if returns_T_data is None or returns_T_data.empty:
    st.stop() # Stop the app if no data is available

# Ensure at least 2 samples for clustering/PCA and more features than samples
if returns_T_data.shape[0] < 2 or returns_T_data.shape[1] < 2:
    st.error("❌ Not enough valid data points (stocks or time periods) to perform clustering. Try adjusting years or checking data integrity.")
    st.stop()

# Scaling & Clustering
try:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(returns_T_data)

    # Ensure enough samples for PCA
    n_components_pca = min(2, scaled_data.shape[1], scaled_data.shape[0] -1) # PCA components cannot exceed features or samples-1
    if n_components_pca < 2:
        st.warning(f"Not enough features or samples ({scaled_data.shape[0]} stocks, {scaled_data.shape[1]} features) for 2 PCA components. Using {n_components_pca}.")
        # If n_components_pca is 0 or 1, we can't plot 2D, so we might want to skip PCA or show an error
        st.error("Cannot perform 2D PCA for visualization with current data. Adjust historical years.")
        st.stop()
        
    pca = PCA(n_components=n_components_pca)
    pca_data = pca.fit_transform(scaled_data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto') # n_init='auto' is recommended
    labels = kmeans.fit_predict(scaled_data)

    score = silhouette_score(scaled_data, labels)

    # --- Output ---
    st.write(f"✅ **Silhouette Score**: {round(score, 4)}")

    clustered_df = pd.DataFrame({
        "Ticker": returns_T_data.index,
        "Cluster": labels,
        "PCA1": pca_data[:, 0],
        "PCA2": pca_data[:, 1]
    }).sort_values(by="Cluster")

    st.dataframe(clustered_df)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=clustered_df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=100, ax=ax)
    
    # Add text labels for tickers, handle potential font issues by not using emojis directly
    for i in range(clustered_df.shape[0]):
        ax.text(clustered_df.PCA1.iloc[i] + 0.05, clustered_df.PCA2.iloc[i] + 0.05, clustered_df.Ticker.iloc[i], 
                fontsize=8, alpha=0.7) # Adjusted offset and transparency slightly

    ax.set_title("Nifty 50 Stock Clusters (PCA-Reduced)") # Changed title slightly
    ax.grid(True)
    st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred during clustering or plotting: {e}")
    st.info("This might be due to insufficient data or other processing issues. Try adjusting the 'Years of Historical Data'.")
