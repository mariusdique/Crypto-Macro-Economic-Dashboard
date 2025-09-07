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
from streamlit.components.v1 import html as components_html

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
# Title with Background
# --------------------
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    ">
        <h1 style="
            color: white;
            text-align: center;
            font-size: 2.5rem;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            margin: 0;
        ">
            📊 Crypto Macro Risk Dashboard
        </h1>
        <p style="
            color: rgba(255,255,255,0.9);
            text-align: center;
            font-size: 1.1rem;
            margin: 10px 0 0 0;
        ">
            Advanced Analytics for Crypto Market Intelligence
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Enhanced CSS styling
st.markdown("""
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app background with animated gradient */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Inter', sans-serif;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Main content area with glassmorphism */
    .main .block-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
    }
    
    /* Headings with modern styling */
    h1, h2, h3, h4, h5, h6 {
        color: #2d3748;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.5);
    }
    
    /* Sidebar with dark theme */
    .css-1d391kg, .css-1lcbmhc {
        background: rgba(30, 30, 30, 0.9) !important;
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .css-1d391kg .sidebar-content {
        color: white;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: #2d3748;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric cards with modern design */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 1rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Buttons with gradient styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* Multiselect styling */
    .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        color: #2d3748;
        font-weight: 500;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 0 0 12px 12px;
        backdrop-filter: blur(5px);
    }
    
    /* Warning and info boxes */
    .stAlert > div {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: rgba(30, 30, 30, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Success messages */
    .stSuccess > div {
        background: rgba(72, 187, 120, 0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(72, 187, 120, 0.3);
        border-radius: 12px;
    }
    
    /* Error messages */
    .stError > div {
        background: rgba(245, 101, 101, 0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(245, 101, 101, 0.3);
        border-radius: 12px;
    }
    
    /* Warning messages */
    .stWarning > div {
        background: rgba(237, 137, 54, 0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(237, 137, 54, 0.3);
        border-radius: 12px;
    }
    
    /* Info messages */
    .stInfo > div {
        background: rgba(66, 153, 225, 0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(66, 153, 225, 0.3);
        border-radius: 12px;
    }
    
    /* Dataframe styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #667eea;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    /* Responsive design for mobile */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        
        h1 {
            font-size: 2rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)


# About Section (Expandable)
with st.expander("ℹ️ About this Dashboard"):
    st.markdown(
        """
        Welcome to the **Crypto Macro Risk Dashboard** 🎯  

        This dashboard integrates **macro-economic indicators**  (CPI, Interest Rates, S&P 500) with **crypto-native metrics**  
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
data_dir = Path(__file__).parent / "dashboard"

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
# --------------------
# Tab 1: Metrics Explorer (replace your current About + scroll code in tab1)
# --------------------
    st.markdown("<div id='metrics_about'></div>", unsafe_allow_html=True)

    st.markdown(
    """
    <div style="background-color:#f0f8ff;padding:10px;border-radius:8px">
        <h2 style="color:#1E90FF;">📂 Metrics Explorer</h2>
    </div>
    """,
    unsafe_allow_html=True
    )
    

    st.sidebar.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px;
            border-radius: 12px;
            margin: 15px 0;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
        ">
            <h3 style="color: white; margin: 0 0 10px 0; text-align: center;">ℹ️ About (Metrics Explorer)</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # KEEP THIS EXISTING CODE
    
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


    # Auto-scroll via small iframe (retries until anchor found)
    scroll_js = """
    <script>
    (function(){
    function tryScroll(){
    try {
      var el = parent.document.getElementById("metrics_about");
      if(el){ el.scrollIntoView({behavior:"smooth", block:"center"}); return true; }
    } catch(e){ /* ignore */ }
    return false;
    }
    var tries = 0;
    var interval = setInterval(function(){
    tries++;
    if(tryScroll() || tries>25){ clearInterval(interval); }
    }, 200);
    })();
    </script>
    """
    components_html(scroll_js, height=10)


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
        # --------------------
    # Tab 2: Risk Thermometer (replace your current About + scroll code in tab2)
    # --------------------
    st.markdown("<div id='thermo_about'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="background-color:#FFF3E6;padding:10px;border-radius:8px">
            <h2 style="color:#994C00;">🌡 Risk Thermometer</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # SIDEBAR DESIGN AND COLOURS
    st.sidebar.markdown("""
        <div style="
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            padding: 15px;
            border-radius: 12px;
            margin: 15px 0;
            box-shadow: 0 4px 15px rgba(255, 154, 158, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
        ">
            <h3 style="color: #8b1538; margin: 0 0 10px 0; text-align: center;">ℹ️ About (Risk Thermometer)</h3>
        </div>
    """, unsafe_allow_html=True)

    # SIDEBAR CONTENT
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

    scroll_js = """
    <script>
    (function(){
    function tryScroll(){
        try {
        var el = parent.document.getElementById("thermo_about");
        if(el){ el.scrollIntoView({behavior:"smooth", block:"center"}); return true; }
        } catch(e){}
        return false;
    }
    var tries = 0;
    var interval = setInterval(function(){
        tries++;
        if(tryScroll() || tries>25){ clearInterval(interval); }
    }, 200);
    })();
    </script>
    """
    components_html(scroll_js, height=10)



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
        # --------------------
    # Tab 3: Correlation Explorer (replace your current About + scroll code in tab3)
    # --------------------
    st.markdown("<div id='corr_about'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="background-color:#F9F0FF;padding:10px;border-radius:8px">
            <h2 style="color:#4B0082;">🔗 Correlation Explorer</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    #SIDEBAR DESIGN AND COLOURS
    st.sidebar.markdown("""
        <div style="
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 15px;
            border-radius: 12px;
            margin: 15px 0;
            box-shadow: 0 4px 15px rgba(168, 237, 234, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
        ">
            <h3 style="color: #2d3748; margin: 0 0 10px 0; text-align: center;">ℹ️ About (Correlation Explorer)</h3>
        </div>
    """, unsafe_allow_html=True)

    # SIDEBAR CONTENT
    st.sidebar.info(
        """
        Explore correlations between macro & crypto variables.

        Features:
        - Pick any two series from datasets or live prices  
        - Compute Pearson correlation  
        - Rolling correlation (default: 12 months)  
        """
    )

    scroll_js = """
    <script>
    (function(){
    function tryScroll(){
        try {
        var el = parent.document.getElementById("corr_about");
        if(el){ el.scrollIntoView({behavior:"smooth", block:"center"}); return true; }
        } catch(e){}
        return false;
    }
    var tries = 0;
    var interval = setInterval(function(){
        tries++;
        if(tryScroll() || tries>25){ clearInterval(interval); }
    }, 200);
    })();
    </script>
    """
    components_html(scroll_js, height=10)




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
#tab4 = st.tabs(["📊 Multi-Coin Overlay"])

with tab4:
        # --------------------
    # Tab 4: Multi-Coin Overlay (replace your current About + scroll code in tab4)
    # --------------------
    st.markdown("<div id='multi_about'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='background-color:#E6FFFA; padding:10px; border-radius:8px;'>
            <h2 style='color:#00665C;'>📊 Multi-Coin Overlay</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    #SIDEBAR DESIGN AND COLOURS
    st.sidebar.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px;
            border-radius: 12px;
            margin: 15px 0;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
        ">
            <h3 style="color: white; margin: 0 0 10px 0; text-align: center;">ℹ️ About (Multi-Coin Overlay)</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # SIDEBAR CONTENT
    st.sidebar.info(
        """
        Compare the relative performance of multiple coins over time.  
        - Select coins (BTC, ETH, SOL, etc.)  
        - Prices are normalized (start = 100) for fair comparison  
        - Useful to see relative outperformance  
        """
    )


    scroll_js = """
    <script>
    (function(){
    function tryScroll(){
        try {
        var el = parent.document.getElementById("multi_about");
        if(el){ el.scrollIntoView({behavior:"smooth", block:"center"}); return true; }
        } catch(e){}
        return false;
    }
    var tries = 0;
    var interval = setInterval(function(){
        tries++;
        if(tryScroll() || tries>25){ clearInterval(interval); }
    }, 200);
    })();
    </script>
    """
    components_html(scroll_js, height=10)


    
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

