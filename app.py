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

# --- Environment Variable Setup (to suppress MKL warning on Windows/Anaconda) ---
# This should ideally be set before any numpy/scikit-learn imports if possible,
# but works fine here for Streamlit context.
os.environ["OMP_NUM_THREADS"] = "1"

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Multi-Asset Clustering App", layout="wide")
st.title("📊 Multi-Asset Clustering App")
st.markdown("Cluster diverse financial assets based on their historical daily return patterns.")

# --- Sidebar Inputs ---
# Reduced default n_years for faster initial load on Streamlit Cloud
n_clusters = st.sidebar.slider("Select Number of Clusters", 2, 10, 4)
# Explanation for Number of Clusters slider
st.sidebar.markdown(
    """
    <small>
    This controls how many distinct groups (clusters) the algorithm will try to find
    among the assets. A higher number means more granular groups.
    </small>
    """, unsafe_allow_html=True
)

n_years = st.sidebar.slider("Years of Historical Data", 1, 10, 2)
# Explanation for Years of Historical Data slider
st.sidebar.markdown(
    """
    <small>
    Determines the length of past data used for analysis. More years provide
    a longer historical context, but may increase loading time.
    </small>
    """, unsafe_allow_html=True
)

# --- Define Diverse Asset Tickers ---
# This list is now the 10 diverse assets we discussed, not just Nifty 50 stocks.
# This helps with faster loading and broader financial insights.
diverse_asset_tickers = [
    'SPY',          # SPDR S&P 500 ETF (US Broad Market)
    'RELIANCE.NS',  # Reliance Industries (Indian Equity)
    'QQQ',          # Invesco QQQ Trust (US Tech/Growth)
    'GLD',          # SPDR Gold Shares ETF (Gold Commodity)
    'USO',          # United States Oil Fund LP (Crude Oil Commodity)
    'BTC-USD',      # Bitcoin (Cryptocurrency)
    'AGG',          # iShares Core US Aggregate Bond ETF (Bonds)
    'VNQ',          # Vanguard Real Estate Index Fund ETF (REITs)
    'HDFCBANK.NS',  # HDFC Bank (Indian Banking)
    'USDINR=X'      # USD/INR Exchange Rate (Currency) - Note: yfinance currency data can sometimes be less robust than others.
]

# --- Data Fetching and Processing Function ---
# Uses st.cache_data to cache results for 1 hour, significantly speeding up reruns.
@st.cache_data(ttl=3600, show_spinner="📥 Fetching and processing asset data...")
def fetch_and_process_data(tickers, n_years_hist):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * n_years_hist)

    all_adj_close_data = pd.DataFrame()
    
    st.info(f"Attempting bulk download for {len(tickers)} assets over {n_years_hist} years...")
    bulk_download_success = False
    for attempt in range(1, 4): # Try bulk download 3 times
        try:
            # Removed 'group_by' which is deprecated and 'interval' (default is '1d')
            df_bulk = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
            
            if not df_bulk.empty:
                # Handle MultiIndex for multiple tickers or single-level for single ticker/flattened data
                if isinstance(df_bulk.columns, pd.MultiIndex):
                    # Attempt to get 'Adj Close' first, then 'Close' if 'Adj Close' is not primary
                    if 'Adj Close' in df_bulk.columns.get_level_values(0):
                        temp_data = df_bulk['Adj Close'].copy()
                    elif 'Close' in df_bulk.columns.get_level_values(0): # Fallback to 'Close' if Adj Close not top level
                        temp_data = df_bulk['Close'].copy()
                    else:
                        raise ValueError("Neither 'Adj Close' nor 'Close' found at top level in bulk download.")
                    
                    # Ensure column names are just tickers
                    temp_data.columns = temp_data.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)
                    all_adj_close_data = temp_data
                elif 'Close' in df_bulk.columns: # Single ticker or already flattened
                    all_adj_close_data = df_bulk['Close'].copy().to_frame() # Ensure it's a DataFrame
                    all_adj_close_data.columns = [tickers[0]] # Rename column if it was a single ticker
                else:
                    raise ValueError("No 'Close' data found in bulk download for single-level DataFrame.")

                # Drop any columns that are entirely NaN after initial fetch (e.g., failed tickers)
                all_adj_close_data = all_adj_close_data.dropna(axis=1, how='all')
                
                if not all_adj_close_data.empty:
                    bulk_download_success = True
                    st.success(f"Bulk download successful on attempt {attempt} for {len(all_adj_close_data.columns)} assets!")
                    break
                else:
                    st.warning(f"Bulk download attempt {attempt}: Dataframe is empty after initial processing. Retrying...")

            else:
                st.warning(f"Bulk download attempt {attempt}: Returned empty DataFrame. Retrying...")

        except Exception as e:
            st.warning(f"Bulk download error (attempt {attempt}): {e}. Retrying after delay...")
        time.sleep(2 ** attempt + random.uniform(0, 3)) # Exponential backoff with jitter

    # --- Fallback to Individual Downloads if Bulk Fails or is Incomplete ---
    tickers_to_fetch_individually = []
    if not bulk_download_success:
        st.info("Bulk download failed or was incomplete. Falling back to individual asset downloads (slower, but more robust).")
        tickers_to_fetch_individually = tickers
    else:
        downloaded_tickers = all_adj_close_data.columns.tolist()
        tickers_to_fetch_individually = [t for t in tickers if t not in downloaded_tickers]
        if tickers_to_fetch_individually:
            st.warning(f"Some assets missing from bulk download. Fetching individually: {len(tickers_to_fetch_individually)} assets.")

    if tickers_to_fetch_individually:
        individual_progress_bar = st.progress(0)
        successful_individual_fetches = 0
        total_individual_fetches = len(tickers_to_fetch_individually)

        for i, ticker in enumerate(tickers_to_fetch_individually):
            # Skip if data for this ticker was already obtained from a partial bulk success
            if ticker in all_adj_close_data.columns and not all_adj_close_data[ticker].dropna().empty:
                successful_individual_fetches += 1
                individual_progress_bar.progress((i + 1) / total_individual_fetches)
                continue

            st.info(f"Fetching data for {ticker} ({i+1}/{total_individual_fetches})...")
            for attempt in range(1, 6): # More attempts for individual
                try:
                    ticker_df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                    if not ticker_df.empty and 'Close' in ticker_df.columns:
                        all_adj_close_data[ticker] = ticker_df['Close']
                        successful_individual_fetches += 1
                        st.success(f"Successfully fetched {ticker}.")
                        break
                    else:
                        st.warning(f"No 'Close' data for {ticker} (Attempt {attempt}/5). Retrying...")
                except Exception as e:
                    st.error(f"❌ Error fetching {ticker} (Attempt {attempt}/5): {e}")
                time.sleep(random.uniform(7, 12)) # Increased delay for individual fetches (crucial for cloud)
            
            individual_progress_bar.progress((i + 1) / total_individual_fetches)
            time.sleep(random.uniform(2, 4)) # Small delay between individual tickers even if successful

        if successful_individual_fetches > 0:
            st.success(f"Successfully fetched data for {successful_individual_fetches} additional assets.")
        else:
            st.error("No additional assets were successfully fetched individually.")

    if all_adj_close_data.empty:
        st.error("❌ No valid asset data could be fetched after all attempts. Please check ticker list or try again later.")
        return None

    # --- Final Data Cleaning and Transformation ---
    # Drop any columns that are entirely NaN after all attempts
    all_adj_close_data = all_adj_close_data.dropna(axis=1, how='all') 
    
    if all_adj_close_data.empty:
        st.error("❌ All asset data columns are empty after final processing. Cannot proceed.")
        return None

    returns = all_adj_close_data.pct_change().dropna()
    
    if returns.empty:
        st.error("❌ No valid return data after calculating percentage change and dropping NaNs. Check data range or asset validity.")
        return None

    returns_T = returns.T # Transpose for clustering assets

    return returns_T

# --- Main App Logic ---
# Call the robust data fetching function
returns_T_data = fetch_and_process_data(diverse_asset_tickers, n_years)

# Stop the app gracefully if data fetching failed
if returns_T_data is None or returns_T_data.empty:
    st.info("Please adjust parameters or try reloading the app.")
    st.stop() 

# Ensure enough samples for clustering/PCA
if returns_T_data.shape[0] < n_clusters or returns_T_data.shape[1] < 2:
    st.error(f"❌ Not enough valid data points ({returns_T_data.shape[0]} assets, {returns_T_data.shape[1]} features) to perform clustering with {n_clusters} clusters. Try reducing clusters or adjusting historical years.")
    st.stop()

# --- Scaling & Clustering ---
try:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(returns_T_data)

    # Determine optimal n_components for PCA, ensuring it's at least 2 for 2D plot
    # and not more than min(n_samples, n_features) - 1
    n_components_pca = min(2, scaled_data.shape[1], scaled_data.shape[0] - 1)
    
    if n_components_pca < 2:
        st.warning(f"Not enough features or samples ({scaled_data.shape[0]} assets, {scaled_data.shape[1]} features) for 2 PCA components. Using {n_components_pca}.")
        st.error("Cannot perform 2D PCA for visualization with current data. Try adjusting historical years or reducing the number of assets if too many failed.")
        st.stop()
        
    pca = PCA(n_components=n_components_pca)
    pca_data = pca.fit_transform(scaled_data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto') # n_init='auto' is recommended
    labels = kmeans.fit_predict(scaled_data)

    # Calculate Silhouette Score (only if there's more than 1 cluster and enough samples)
    score = -1 # Default to -1 if score cannot be calculated
    if len(np.unique(labels)) > 1 and scaled_data.shape[0] > 1:
        try:
            score = silhouette_score(scaled_data, labels)
        except Exception as e:
            st.warning(f"Could not calculate Silhouette Score: {e}. This can happen with very few data points or specific clustering outcomes.")
            score = np.nan # Set to NaN if calculation fails

    # --- Output Results ---
    if not np.isnan(score):
        st.write(f"✅ **Silhouette Score**: {round(score, 4)}")
    else:
        st.write("✅ **Silhouette Score**: N/A (Could not be calculated)")

    # Create a display-friendly Ticker column for the DataFrame
    clustered_df = pd.DataFrame({
        "Ticker": returns_T_data.index,
        "Cluster": labels,
        "PCA1": pca_data[:, 0],
        "PCA2": pca_data[:, 1]
    }).sort_values(by="Cluster")

    # Create a display-friendly Ticker column for the DataFrame and plot
    # Remove '=X' suffix for currency tickers for cleaner display
    clustered_df['Display_Ticker'] = clustered_df['Ticker'].apply(lambda x: x.replace('=X', '') if '=X' in x else x)

    st.subheader("Clustered Assets")
    st.dataframe(clustered_df[['Display_Ticker', 'Cluster', 'PCA1', 'PCA2']]) # Display the new column

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 8))
    # Use 'Display_Ticker' for the text labels on the plot
    sns.scatterplot(data=clustered_df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=100, ax=ax)
    
    # Add text labels for tickers, using Display_Ticker for cleaner labels
    for i in range(clustered_df.shape[0]):
        # Adjust text offset slightly for better readability
        ax.text(clustered_df.PCA1.iloc[i] + 0.05, clustered_df.PCA2.iloc[i] + 0.05, 
                clustered_df.Display_Ticker.iloc[i], fontsize=8, alpha=0.8, 
                ha='left', va='bottom') # Horizontal/Vertical alignment for better positioning

    ax.set_title("Clustering of Diverse Financial Assets (PCA-Reduced)")
    ax.grid(True)
    st.pyplot(fig)

    # --- Explanation for PCA Axes ---
    st.markdown("---") # Separator for clarity
    st.subheader("Understanding the PCA Plot Axes")
    st.markdown(
        """
        This scatter plot visualizes the assets in a reduced 2-dimensional space using **Principal Component Analysis (PCA)**.
        
        * **PCA1 (Principal Component 1):** This axis captures the **largest amount of variance** (information) in the original high-dimensional daily return data. Assets that are far apart along this axis show the biggest differences in their overall return patterns.
        * **PCA2 (Principal Component 2):** This axis captures the **second largest amount of variance** in the data, independent of PCA1. It helps to differentiate assets further in a direction not explained by PCA1.
        
        **Interpretation:** Assets that are plotted closer together in this 2D space have more similar historical daily return behaviors. The clusters (indicated by different colors) group these similar assets together. The exact numerical values on the axes (e.g., -10, 0, 10) are abstract and represent positions in this transformed space, not direct financial metrics.
        """
    )

except Exception as e:
    st.error(f"An error occurred during clustering or plotting: {e}")
    st.info("This might be due to insufficient valid data points for the selected parameters. Try adjusting the 'Years of Historical Data' or 'Number of Clusters'.")

