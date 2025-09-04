# app.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests
import os
import io
from pathlib import Path
from dotenv import load_dotenv

# --------------------
# Config
# --------------------
st.set_page_config(
    page_title="Crypto Macro Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Crypto Macro Risk Dashboard")

# Load environment variables (for API key)
load_dotenv()
API_KEY = os.getenv("GECKO_KEY")

# --------------------
# Data Loader
# --------------------
data_dir = Path(r"D:\My folder\Crypto-Macro-Economic-Dashboard\dashboard")

files = {
    "CPI": "cpi_data.csv",
    "Interest Rates": "federal_funds_rate_data.csv",
    "S&P 500": "SP500_data.csv",
    "BTC Market Cap": "bitcoin_mcap.csv",
    "Total Market Cap": "global_mcap.csv",
    "BTC Dominance": "btc_dominance_monthly.csv",
    "Hashrate": "Hashrate.csv"
}

datasets = {}
for name, file in files.items():
    try:
        df = pd.read_csv(data_dir / file)

        # Drop index col if exists
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        # Find date column
        date_col = [col for col in df.columns if "date" in col.lower()][0]

        # Set index as date
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)

        # Keep numeric
        df = df.apply(pd.to_numeric, errors="coerce")

        # Monthly resample
        df = df.resample("ME").mean()

        datasets[name] = df
    except Exception as e:
        st.warning(f"⚠️ Could not load {file}: {e}")


# --------------------
# Normalization + Scores
# --------------------
def normalize_series(series, invert=False):
    series = series.fillna(method="ffill")
    norm = (series - series.min()) / (series.max() - series.min()) * 100
    if invert:
        norm = 100 - norm
    return norm

def compute_macro_score(cpi, rates, sp500):
    cpi_score = normalize_series(cpi["value"], invert=True)   # lower CPI = better
    rate_score = normalize_series(rates["value"], invert=True)  # lower rates = better
    sp_score = normalize_series(sp500["value"], invert=False)   # higher S&P = better
    return (cpi_score + rate_score + sp_score) / 3

def compute_crypto_flow_score(btc_dom, total_mcap):
    dom_score = normalize_series(btc_dom["btc_dominance"], invert=True)  # falling dom = risk-on
    mcap_score = normalize_series(total_mcap["total_mcap"], invert=False)
    return (dom_score + mcap_score) / 2

def compute_miner_health(hashrate):
    return normalize_series(hashrate["hashrate"], invert=False)  # rising/stable = better


def compute_thermometer(datasets):
    macro = compute_macro_score(datasets["CPI"], datasets["Interest Rates"], datasets["S&P 500"])
    flow = compute_crypto_flow_score(datasets["BTC Dominance"], datasets["Total Market Cap"])
    miner = compute_miner_health(datasets["Hashrate"])

    thermometer = 0.4 * macro + 0.4 * flow + 0.2 * miner
    return thermometer.to_frame("thermometer")


# --------------------
# Coin Price Fetcher
# --------------------
def fetch_coin_price(coin_id="bitcoin"):
    url = f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": "max"}
    headers = {"accept": "application/json", "x-cg-pro-api-key": API_KEY}

    try:
        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status()
        data = res.json()["prices"]

        df = pd.DataFrame(data, columns=["timestamp", "price"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("date", inplace=True)
        df = df.resample("ME").mean()
        return df
    except Exception as e:
        st.error(f"⚠️ Error fetching {coin_id} price: {e}")
        return pd.DataFrame()


# --------------------
# Helper: download buttons
# --------------------
def download_csv_button(df, filename):
    csv = df.to_csv().encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

def download_plot_button(fig, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    st.download_button(
        label="🖼 Download Plot (PNG)",
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png"
    )



# --------------------
# Streamlit Tabs
# --------------------
tab1, tab2 = st.tabs(["📂 Metrics Explorer", "🌡 Risk Thermometer"])

# --------------------
# Tab 1: Metrics Explorer
# --------------------
with tab1:
    st.sidebar.title("ℹ️ About (Metrics Explorer)")
    st.sidebar.info(
        """
        Explore macro & crypto metrics:
        - CPI  
        - Interest Rates  
        - S&P 500  
        - BTC Dominance  
        - Market Caps  
        - Hashrate  

        Data is loaded from saved CSVs.
        """
    )

    selected_metric = st.selectbox("Select a Metric", list(datasets.keys()))
    df = datasets[selected_metric]

    view_option = st.radio("View Mode:", ["Table", "Plot"], horizontal=True)

    if view_option == "Table":
        st.write(f"📄 Latest {selected_metric} data")
        st.dataframe(df.tail(10), use_container_width=True)

    elif view_option == "Plot":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        col_to_plot = numeric_cols[0] if len(numeric_cols) == 1 else st.selectbox("Select column", numeric_cols)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df[col_to_plot], label=selected_metric, linewidth=2)
        ax.set_title(f"{selected_metric} Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel(col_to_plot)
        ax.legend()
        st.pyplot(fig)


# --------------------
# Tab 2: Risk Thermometer
# --------------------
with tab2:
    st.sidebar.title("ℹ️ About (Risk Thermometer)")
    st.sidebar.info(
        """
        The **Crypto Risk Thermometer** (0–100) combines:
        - Macro context (CPI, Rates, S&P) → 40%  
        - Crypto flows (BTC Dominance, Market Cap) → 40%  
        - Miner health (Hashrate) → 20%  

        Zones:  
        - 0–30 (Red): Defensive  
        - 30–60 (Amber): Mixed  
        - 60–100 (Green): Risk-On  
        """
    )

    thermometer = compute_thermometer(datasets)

    coin_choice = st.selectbox("Overlay with Coin Price", ["bitcoin", "ethereum", "solana"])
    price_df = fetch_coin_price(coin_choice)

    # Merge thermometer and coin prices
    merged_df = thermometer.copy()
    if not price_df.empty:
        merged_df = merged_df.join(price_df[["price"]], how="inner")

    # ---- Plot ----
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(thermometer.index, thermometer["thermometer"], color="black", label="Risk Thermometer")
    ax1.set_ylabel("Risk Thermometer (0–100)")
    ax1.set_ylim(0, 100)

    # Zones
    ax1.axhspan(0, 30, color="red", alpha=0.2, label="Defensive")
    ax1.axhspan(30, 60, color="yellow", alpha=0.2, label="Mixed")
    ax1.axhspan(60, 100, color="green", alpha=0.2, label="Risk-On")

    if not price_df.empty:
        ax2 = ax1.twinx()
        ax2.plot(price_df.index, price_df["price"], color="blue", label=f"{coin_choice.title()} Price (log)")
        ax2.set_yscale("log")
        ax2.set_ylabel(f"{coin_choice.title()} Price (log scale)")

    ax1.set_title("Crypto Risk Thermometer with Overlay")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels() if not price_df.empty else ([], [])
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    st.pyplot(fig)

    # ---- Downloads ----
    st.markdown("### 📥 Download Data & Plot")
    download_csv_button(merged_df, f"thermometer_{coin_choice}.csv")
    download_plot_button(fig, f"thermometer_{coin_choice}.png")
