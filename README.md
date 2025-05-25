# 📊 Multi-Asset Clustering & Analysis App

## 🚀 Project Overview

This project features an interactive **Streamlit** web application designed to cluster diverse financial assets (including **stocks, ETFs, commodities, cryptocurrencies, bonds, and real estate**) based on the similarities in their historical **daily return patterns**. It helps users visualize and understand how different asset classes move together—insightful for **portfolio diversification**, **risk management**, and **investing**.

---

## ✨ Features

- **Dynamic Data Fetching**: Pulls historical Adjusted Close prices directly from Yahoo Finance.
- **Configurable History**: Users can choose 1–10 years of historical data via sidebar slider.
- **Automated Feature Engineering**: Computes daily returns and prepares data for clustering.
- **Unsupervised Clustering**: Groups assets using the KMeans algorithm.
- **Dimensionality Reduction**: Applies PCA to reduce features to 2D for clear visualizations.
- **Interactive Visualization**: Hoverable scatter plot with ticker labels using Matplotlib/Seaborn.
- **User-Friendly UI**: Built entirely with Streamlit for smooth interaction.

---

## 🛠️ Technical Stack

- Python
- Streamlit
- yfinance
- Pandas
- NumPy
- scikit-learn (StandardScaler, PCA, KMeans)
- Matplotlib
- Seaborn
- Plotly Express *(optional dependency for future enhancements)*

---

## 📦 Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
