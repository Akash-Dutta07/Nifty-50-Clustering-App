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

st.set_page_config(page_title="Nifty 50 Clustering App", layout="wide")

st.title("📊 Nifty 50 Clustering App")
st.markdown("Upload or fetch data dynamically to view stock clusters.")

# Sidebar
n_clusters = st.sidebar.slider("Select Number of Clusters", 2, 10, 4)
n_years = st.sidebar.slider("Years of Historical Data", 1, 10, 5)

# Define tickers
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

start_date = datetime.now() - timedelta(days=365 * n_years)
end_date = datetime.now()

# Download data
@st.cache_data
def fetch_data():
    try:
        st.info("📥 Fetching data from Yahoo Finance...")
        df = yf.download(nifty50_tickers, start=start_date, end=end_date, interval='1d', group_by='ticker', auto_adjust=True)
        adj_close = pd.DataFrame()
        for ticker in nifty50_tickers:
            try:
                adj_close[ticker] = df[ticker]['Close']
            except Exception:
                st.warning(f"⚠️ Skipping {ticker} due to missing data")
        return adj_close.dropna(axis=1)
    except Exception as e:
        st.error(f"❌ Failed to fetch data: {e}")
        return None

# Try live data first
adj_close = fetch_data()

# Fallback to local CSV
if adj_close is None or adj_close.empty:
    st.error("❌ No valid stock data fetched. Using fallback file instead.")
    try:
        adj_close = pd.read_csv("returns.csv", index_col=0, parse_dates=True)
        st.success("✅ Loaded from fallback returns.csv")
    except:
        st.stop()

# Proceed with clustering
returns = adj_close.pct_change().dropna()
returns_T = returns.T

# Scaling & Clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(returns_T)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(scaled_data)

score = silhouette_score(scaled_data, labels)

# Output
st.write(f"✅ **Silhouette Score**: {round(score, 4)}")

clustered_df = pd.DataFrame({
    "Ticker": returns_T.index,
    "Cluster": labels,
    "PCA1": pca_data[:, 0],
    "PCA2": pca_data[:, 1]
}).sort_values(by="Cluster")

st.dataframe(clustered_df)

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(data=clustered_df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=100)
for i in range(clustered_df.shape[0]):
    ax.text(clustered_df.PCA1.iloc[i]+0.2, clustered_df.PCA2.iloc[i]+0.2, clustered_df.Ticker.iloc[i], fontsize=8)
ax.set_title("📍 Nifty 50 Clusters")
ax.grid(True)
st.pyplot(fig)
