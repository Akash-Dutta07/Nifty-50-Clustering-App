# 📊 Nifty 50 Stock Clustering & Analysis App

## 🧠 Project Overview

This project features an interactive **Streamlit** web application designed to **cluster Nifty 50 stocks** based on the similarities in their historical daily return patterns.

The application provides a dynamic way to visualize and understand how different stocks within the Indian market move together — offering insights valuable for:

- Portfolio analysis  
- Diversification strategies  
- Identifying thematic groupings  

---

## ✨ Features

- **📈 Dynamic Data Fetching:** Pulls historical **Adjusted Close prices** for 48 Nifty 50 constituents directly from Yahoo Finance.

- **📅 Configurable History:** Users can select the number of years of historical data (1–10 years) via a **sidebar slider**.

- **🔁 Robust API Handling:** Implements retry mechanisms with **exponential backoff** and **individual ticker fallbacks** to handle yfinance API rate limits gracefully.

- **🧪 Automated Feature Engineering:** Calculates **daily percentage returns** and preprocesses the dataset for clustering.

- **🧠 Unsupervised Clustering:** Uses **KMeans** to group stocks based on return similarities.

- **📉 Dimensionality Reduction:** Applies **PCA (Principal Component Analysis)** to reduce high-dimensional data to 2D for visualization.

- **📊 Interactive Visualization:** Clustered stocks are shown on a **Plotly Express** scatter plot with **hover tooltips** for ticker insight.

- **🌐 User-Friendly UI:** Entire app is built using **Streamlit** for smooth interaction.

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- yFinance  
- Pandas  
- NumPy  
- Scikit-learn (`StandardScaler`, `PCA`, `KMeans`)  
- Plotly Express  

---

## 🚀 Installation & Setup

1. **Clone the Repository**

```bash
git clone <your-repo-url>
cd <your-repo-name>
