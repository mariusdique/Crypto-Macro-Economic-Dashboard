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
from streamlit_js_eval import streamlit_js_eval

# --------------------
# Config
# --------------------
st.set_page_config(
    page_title="Crypto Macro Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


#st.title("📊 Crypto Macro Risk Dashboard")
# --------------------
# Title & About Section
# --------------------

# --------------------
# Title with Background
# --------------------
st.markdown(
    """
    <div style="background-color:#1E90FF;padding:15px;border-radius:10px">
        <h1 style="color:white;text-align:center;">
            📊 Crypto Macro Risk Dashboard
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)


# About Section (Expandable)
with st.expander("ℹ️ About this Dashboard"):
    st.markdown(
        """
        Welcome to the **Crypto Macro Risk Dashboard** 🎯  

        This dashboard integrates **macro-economic indicators**  
        (CPI, Interest Rates, S&P 500) with **crypto-native metrics**  
        (BTC Market Cap, Total Market Cap, BTC Dominance, Hashrate)  
        to build a **Crypto Risk Thermometer**.  

        🔹 **Features:**  
        - Explore macro & crypto datasets  
        - Track market risk sentiment with the Thermometer  
        - Compare multiple coins (BTC, ETH, SOL, …)  
        - Correlation explorer between variables  
        
        ✅ Built with Streamlit, Python, and CoinGecko API  
        """
    )


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
tab1, tab2, tab3, tab4 = st.tabs(["📂 Metrics Explorer", "🌡 Risk Thermometer","🔗Correlation Explorer", "💹 Multi-Coin Overlay"])

# --------------------
# Tab 1: Metrics Explorer
# --------------------
with tab1:
    st.markdown("<div id='metrics_about'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background-color:#f0f8ff;padding:10px;border-radius:8px">
            <h2 style="color:#1E90FF;">📂 Metrics Explorer</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

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

    # Auto-scroll here
    streamlit_js_eval(
        js_expressions=[
            """
            (() => {
                setTimeout(() => {
                    const el = document.getElementById("metrics_about");
                    if (el) {
                        el.scrollIntoView({behavior: "smooth", block: "center"});
                    }
                }, 300);
                return true;
            })()
            """
        ],
        key="metrics_scroll"
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
    st.markdown("<div id='thermo_about'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background-color:#FFF5F5;padding:10px;border-radius:8px">
            <h2 style="color:#E53E3E;">🌡 Risk Thermometer</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

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

    # Auto-scroll here
    streamlit_js_eval(
        js_expressions=[
            """
            (() => {
                setTimeout(() => {
                    const el = document.getElementById("thermo_about");
                    if (el) {
                        el.scrollIntoView({behavior: "smooth", block: "center"});
                    }
                }, 300);
                return true;
            })()
            """
        ],
        key="thermo_scroll"
    )

    # Compute thermometer
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

# --------------------
# Tab 3: Correlation Explorer
# --------------------
with tab3:
    st.markdown("<div id='corr_about'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background-color:#FAF5FF;padding:10px;border-radius:8px">
            <h2 style="color:#6B46C1;">🔗 Correlation Explorer</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.title("ℹ️ About (Correlation Explorer)")
    st.sidebar.info(
        """
        Explore correlations between macro & crypto variables.

        Features:
        - Pick **any two series** from datasets or live prices  
        - Compute **Pearson correlation**  
        - Rolling correlation (default: 12 months)  
        """
    )

    # Auto-scroll here
    streamlit_js_eval(
        js_expressions=[
            """
            (() => {
                setTimeout(() => {
                    const el = document.getElementById("corr_about");
                    if (el) {
                        el.scrollIntoView({behavior: "smooth", block: "center"});
                    }
                }, 300);
                return true;
            })()
            """
        ],
        key="corr_scroll"
    )


    # User selects metrics
    metric_options = list(datasets.keys()) + ["Bitcoin Price", "Ethereum Price", "Solana Price"]
    col1, col2 = st.columns(2)
    with col1:
        metric_x = st.selectbox("Select First Metric", metric_options, index=0)
    with col2:
        metric_y = st.selectbox("Select Second Metric", metric_options, index=1)

    # Load chosen series
    def load_series(metric):
        if metric in datasets:
            df = datasets[metric]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            col = numeric_cols[0]
            return df[[col]].rename(columns={col: metric})
        else:
            coin_id = metric.split()[0].lower()  # e.g. "Bitcoin Price" -> "bitcoin"
            df = fetch_coin_price(coin_id)
            return df[["price"]].rename(columns={"price": metric})

    series_x = load_series(metric_x)
    series_y = load_series(metric_y)

    # Merge
    merged = series_x.join(series_y, how="inner")

    if merged.empty:
        st.warning("⚠️ Not enough data to compute correlation.")
    else:
        # Correlation coefficient
        corr_value = merged[metric_x].corr(merged[metric_y])
        st.metric(label="📈 Pearson Correlation", value=f"{corr_value:.2f}")

        # Rolling correlation
        window = st.slider("Rolling Window (months)", 3, 24, 12)
        rolling_corr = merged[metric_x].rolling(window).corr(merged[metric_y])

        # ---- Plot ----
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(merged.index, merged[metric_x], label=metric_x, color="blue")
        ax.plot(merged.index, merged[metric_y], label=metric_y, color="orange")

        ax.set_ylabel("Metric Values")
        ax.set_title(f"{metric_x} vs {metric_y}")
        ax.legend(loc="upper left")
        st.pyplot(fig)

        # Rolling correlation plot
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(rolling_corr.index, rolling_corr, color="green", label="Rolling Correlation")
        ax2.axhline(0, color="black", linestyle="--", linewidth=1)
        ax2.set_title(f"{window}-Month Rolling Correlation")
        ax2.set_ylabel("Correlation")
        ax2.legend()
        st.pyplot(fig2)

        # ---- Downloads ----
        st.markdown("### 📥 Download Data & Plots")
        download_csv_button(merged, f"correlation_{metric_x}_{metric_y}.csv")
        download_plot_button(fig, f"series_{metric_x}_{metric_y}.png")
        download_plot_button(fig2, f"rolling_corr_{metric_x}_{metric_y}.png")

# --------------------
# Tab 4: Multi-Coin Overlay
# --------------------
#tab4 = st.tabs(["📊 Multi-Coin Overlay"])[0]

with tab4:
    st.markdown("<div id='multi_about'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background-color:#E6FFFA;padding:10px;border-radius:8px">
            <h2 style="color:#00665C;">📊 Multi-Coin Overlay</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.title("ℹ️ About (Multi-Coin Overlay)")
    st.sidebar.info(
        """
        Compare the relative performance of multiple coins over time.  
        - Select coins (BTC, ETH, SOL, etc.)  
        - Prices are normalized (start = 100) for fair comparison  
        - Useful to see relative outperformance  
        """
    )

    # Auto-scroll here
    streamlit_js_eval(
        js_expressions=[
            """
            (() => {
                setTimeout(() => {
                    const el = document.getElementById("multi_about");
                    if (el) {
                        el.scrollIntoView({behavior: "smooth", block: "center"});
                    }
                }, 300);
                return true;
            })()
            """
        ],
        key="multi_scroll"
    )
    
    # Coin options
    coin_choices = ["bitcoin", "ethereum", "solana", "ripple", "litecoin", "cardano", "polkadot", "dogecoin", "avalanche", "chainlink"]
    selected_coins = st.multiselect("Select Coins to Compare", coin_choices, default=["bitcoin", "ethereum"])

    price_data = {}

    for coin in selected_coins:
        df = fetch_coin_price(coin)
        if not df.empty:
            # Normalize to start = 100
            df["normalized"] = df["price"] / df["price"].iloc[0] * 100
            price_data[coin.title()] = df["normalized"]

    if price_data:
        combined_df = pd.DataFrame(price_data)

        # ---- Plot ----
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in combined_df.columns:
            ax.plot(combined_df.index, combined_df[col], label=col, linewidth=2)

        ax.set_title("Multi-Coin Performance (Normalized to 100)")
        ax.set_ylabel("Normalized Performance (Start=100)")
        ax.legend()
        st.pyplot(fig)

        # ---- Downloads ----
        st.markdown("### 📥 Download Data & Plot")
        download_csv_button(combined_df, "multi_coin_overlay.csv")
        download_plot_button(fig, "multi_coin_overlay.png")
    else:
        st.warning("⚠️ No coin data available. Try selecting different coins.")
