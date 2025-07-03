# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import streamlit as st
import base64
from io import BytesIO
from PIL import Image
import math
import logging
import numpy as np
from pathlib import Path
import time

# --- SET PAGE CONFIG (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Phronesis Pulse 2.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 1. Phronesis Apex Theme Configuration Constants ---
PRIMARY_ACCENT_COLOR = "#cd669b"
PRIMARY_ACCENT_COLOR_RGB = "205, 102, 155"
CARD_TEXT_COLOR = "#a9b4d2"
CARD_TITLE_TEXT_COLOR = PRIMARY_ACCENT_COLOR
MAIN_TITLE_COLOR = "#f0f8ff"
BODY_TEXT_COLOR = "#ffff" # White for general text and labels
SUBTITLE_COLOR = "#8b98b8" # Grey for less important text like help/captions
MAIN_BACKGROUND_COLOR = "#0b132b"
CARD_BACKGROUND_COLOR = "#1c2541"
SIDEBAR_BACKGROUND_COLOR = "#121a35"
HOVER_GLOW_COLOR = f"rgba({PRIMARY_ACCENT_COLOR_RGB}, 0.4)"
CONTAINER_BG_COLOR = "rgba(11, 19, 43, 0.0)"
CONTAINER_BORDER_RADIUS = "15px"
INPUT_BG_COLOR = "#1c2541"
INPUT_BORDER_COLOR = "#3a506b"
INPUT_TEXT_COLOR = BODY_TEXT_COLOR
BUTTON_PRIMARY_BG = PRIMARY_ACCENT_COLOR # Pink background
BUTTON_PRIMARY_TEXT = "#FFFFFF" # White text for primary buttons
BUTTON_PRIMARY_BORDER = PRIMARY_ACCENT_COLOR # Pink border
DATAFRAME_HEADER_BG = "#1c2541"
DATAFRAME_HEADER_TEXT = MAIN_TITLE_COLOR
DATAFRAME_CELL_BG = MAIN_BACKGROUND_COLOR
DATAFRAME_CELL_TEXT = BODY_TEXT_COLOR
CHART_SUCCESS_COLOR = "#2ecc71"
CHART_WARNING_COLOR = "#f39c12"
CHART_ERROR_COLOR = "#e74c3c"
TITLE_FONT = "'Montserrat', sans-serif"
BODY_FONT = "'Roboto', sans-serif"
CARD_TITLE_FONT = "'Montserrat', sans-serif"

# --- Logo Configuration ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
LOGO_PATH = current_dir / "ppl_logo.png"

# --- App Specific Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEFAULT_TICKERS = "GOOGL,AAPL,MSFT,AMZN"

# *** MODIFIED: Added 'Address' to the selection list ***
FINANCIAL_COLUMNS_TO_SELECT = [
    'Ticker', 'LongName', 'Long_Business_Summary', 'Country', 'Address', 'Sector', 'Industry',
    'Full_Time_Employees', 'Website', 'Phone', 'Full_Date', 'Year_Index',
    'Currency', 'Financial_Currency', 'Total Revenue', 'Operating Revenue',
    'Cost Of Revenue', 'Gross Profit', 'Operating Expense',
    'Selling General And Administrative', 'Selling General And Administration',
    'Operating Income', 'EBIT', 'Normalized EBITDA', 'Net Income'
]
NUMERIC_COLUMNS_TO_CLEAN = [
    'Total Revenue', 'Operating Revenue', 'Cost Of Revenue', 'Gross Profit',
    'Operating Expense', 'Selling General And Administrative',
    'Selling General And Administration', 'Operating Income', 'EBIT',
    'Normalized EBITDA', 'Net Income',
]

# --- Function to load and encode image to base64 ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f: data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.warning(f"Warning: Logo file not found at {bin_file}.")
        return None
    except Exception as e:
        st.error(f"Error loading logo: {e}")
        return None

logo_base64 = get_base64_of_bin_file(LOGO_PATH)
logo_html = f'<img src="data:image/png;base64,{logo_base64}" alt="Phronesis Partners Logo" class="logo">' if logo_base64 else '<div class="logo-placeholder">Logo</div>'

# --- 2. Apex Theme CSS Styling (Forcing Button and Metric Label Colors) ---
# ... (CSS is unchanged, so it is omitted here for brevity)
APP_STYLE = f"""
<style>
    /* --- Import Fonts --- */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400&display=swap');

    /* --- Global Body & Streamlit App Styling --- */
    body {{ background-color: {MAIN_BACKGROUND_COLOR}; color: {BODY_TEXT_COLOR}; font-family: {BODY_FONT}; }}
    .stApp {{ background-color: {MAIN_BACKGROUND_COLOR}; color: {BODY_TEXT_COLOR}; }}
    .stApp > header {{ background-color: transparent; border-bottom: none; }}

    /* --- Main Content Area Container --- */
    .main .block-container {{ max-width: 1100px; padding: 2rem 1rem 4rem 1rem; background-color: {CONTAINER_BG_COLOR}; border-radius: {CONTAINER_BORDER_RADIUS}; color: {BODY_TEXT_COLOR}; margin: auto; font-family: {BODY_FONT}; }}

    /* --- Header Section CSS --- */
    .header-container {{ display: flex; flex-direction: row; align-items: center; width: fit-content; margin-left: auto; margin-right: auto; margin-bottom: 3rem; text-align: left; }}
    .logo {{ height: 80px; width: auto; margin-right: 1.5rem; margin-bottom: 0; flex-shrink: 0; vertical-align: middle; }}
    .logo-placeholder {{ height: 80px; width: 80px; margin-right: 1.5rem; background-color: #333; border: 1px dashed #555; display: flex; align-items: center; justify-content: center; color: #888; font-size: 0.9em; text-align: center; border-radius: 5px; flex-shrink: 0; }}
    .title {{ font-family: {TITLE_FONT}; font-size: 2.8rem; font-weight: 700; color: {MAIN_TITLE_COLOR}; letter-spacing: 1px; margin: 0; padding: 0; line-height: 1.2; text-shadow: 0 0 8px rgba({PRIMARY_ACCENT_COLOR_RGB}, 0.2); }}

    /* --- General Headings (st.subheader) --- */
    h2, h3 {{ font-family: {TITLE_FONT}; color: {PRIMARY_ACCENT_COLOR}; margin-top: 2.5rem; margin-bottom: 1.5rem; border-bottom: 1px solid rgba({PRIMARY_ACCENT_COLOR_RGB}, 0.3); padding-bottom: 0.6rem; font-weight: 600; font-size: 1.7rem; }}

    /* --- Button Styling (Applying Primary Style by Default with !important) --- */
    div[data-testid="stButton"] > button, div[data-testid="stDownloadButton"] > button {{
        border-radius: 20px; padding: 0.6rem 1.6rem; font-weight: 600; font-family: {BODY_FONT};
        transition: all 0.3s ease;
        border: 1px solid {BUTTON_PRIMARY_BORDER} !important; /* Use primary border */
        background-color: {BUTTON_PRIMARY_BG} !important; /* Use primary background */
        color: {BUTTON_PRIMARY_TEXT} !important; /* Use primary text color */
        cursor: pointer;
    }}
    /* Hover for ALL buttons */
    div[data-testid="stButton"] > button:hover, div[data-testid="stDownloadButton"] > button:hover {{
         background-color: {PRIMARY_ACCENT_COLOR} !important; /* Keep accent color */
         border-color: {PRIMARY_ACCENT_COLOR} !important; /* Keep accent border */
         color: {BUTTON_PRIMARY_TEXT} !important; /* Keep white text */
         box-shadow: 0 6px 15px rgba({PRIMARY_ACCENT_COLOR_RGB}, 0.3);
         opacity: 0.9;
         transform: translateY(-2px);
    }}
    /* Active state */
    div[data-testid="stButton"] > button:active, div[data-testid="stDownloadButton"] > button:active {{
        transform: translateY(0px); box-shadow: none; opacity: 1;
        background-color: {PRIMARY_ACCENT_COLOR} !important; /* Ensure active maintains color */
    }}
    /* Disabled state */
    div[data-testid="stButton"] > button:disabled, div[data-testid="stDownloadButton"] > button:disabled {{
        background-color: rgba({PRIMARY_ACCENT_COLOR_RGB}, 0.3) !important; /* Use dim accent background */
        color: rgba(255, 255, 255, 0.5) !important; /* Dim white text */
        border-color: rgba({PRIMARY_ACCENT_COLOR_RGB}, 0.2) !important; /* Dim accent border */
        cursor: not-allowed;
        opacity: 0.7; /* Slightly less opacity */
    }}
    div[data-testid="stButton"] > button:disabled:hover, div[data-testid="stDownloadButton"] > button:disabled:hover {{
         box-shadow: none; transform: none; opacity: 0.7;
         background-color: rgba({PRIMARY_ACCENT_COLOR_RGB}, 0.3) !important;
    }}

    /* --- Input Element Styling --- */
    div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {{ background-color: {INPUT_BG_COLOR} !important; color: {INPUT_TEXT_COLOR} !important; border: 1px solid {INPUT_BORDER_COLOR} !important; border-radius: 8px !important; box-shadow: none !important; }}
    div[data-testid="stTextInput"] label, div[data-testid="stTextArea"] label {{ color: {BODY_TEXT_COLOR} !important; font-weight: 600; font-family: {BODY_FONT}; margin-bottom: 0.5rem; }}
    .stTextArea small, .stTextInput small {{ color: {SUBTITLE_COLOR} !important; opacity: 0.8; }}

    /* --- Data Editor / DataFrame Styling --- */
    div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {{ border: 1px solid {INPUT_BORDER_COLOR}; border-radius: 8px; background-color: {DATAFRAME_CELL_BG}; margin-top: 1rem; }}
    .stDataFrame th, .stDataEditor th {{ background-color: {DATAFRAME_HEADER_BG} !important; color: {DATAFRAME_HEADER_TEXT} !important; font-weight: 600; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.5px; border-radius: 0 !important; border-bottom: 2px solid {PRIMARY_ACCENT_COLOR} !important; padding: 0.7rem 0.7rem; }}
    .stDataFrame td, .stDataEditor td {{ font-size: 0.9rem; vertical-align: middle; padding: 0.6rem 0.7rem; color: {DATAFRAME_CELL_TEXT}; border-bottom: 1px solid {INPUT_BORDER_COLOR}; border-right: 1px solid {INPUT_BORDER_COLOR}; }}
    div[data-testid="stDataFrame"] > div > div > div > div {{ width: 100% !important; }}

    /* --- Markdown & Misc Elements --- */
    .stMarkdown p, .stMarkdown li {{ color: {BODY_TEXT_COLOR}; line-height: 1.6; }}
    .stMarkdown a {{ color: {PRIMARY_ACCENT_COLOR}; text-decoration: none; }} .stMarkdown a:hover {{ text-decoration: underline; }}
    .stCaption {{ color: {SUBTITLE_COLOR}; font-size: 0.85rem; }}
    div[data-testid="stText"] {{ margin-bottom: 0.8rem; font-family: {BODY_FONT}; color: {BODY_TEXT_COLOR}; line-height: 1.6; }}
    hr {{ border-top: 1px solid {INPUT_BORDER_COLOR}; margin-top: 2rem; margin-bottom: 2rem; }}

    /* --- Alert Styling --- */
    div[data-testid="stAlert"] {{ border-radius: 8px !important; border: 1px solid {INPUT_BORDER_COLOR} !important; border-left-width: 5px !important; padding: 1rem 1.2rem !important; margin-top: 1rem; margin-bottom: 1rem; }}
    div[data-testid="stAlert"] div[role="alert"] {{ font-family: {BODY_FONT}; font-size: 0.95rem; font-weight: 500; }}
    div[data-testid="stAlert"][data-baseweb="notification-info"] {{ border-left-color: {PRIMARY_ACCENT_COLOR} !important; background-color: rgba({PRIMARY_ACCENT_COLOR_RGB}, 0.1) !important; }} div[data-testid="stAlert"][data-baseweb="notification-info"] div[role="alert"] {{ color: {PRIMARY_ACCENT_COLOR} !important; }} div[data-testid="stAlert"][data-baseweb="notification-info"] svg {{ fill: {PRIMARY_ACCENT_COLOR} !important; }}
    div[data-testid="stAlert"][data-baseweb="notification-warning"] {{ border-left-color: {CHART_WARNING_COLOR} !important; background-color: rgba(243, 156, 18, 0.1) !important; }} div[data-testid="stAlert"][data-baseweb="notification-warning"] div[role="alert"] {{ color: {CHART_WARNING_COLOR} !important; }} div[data-testid="stAlert"][data-baseweb="notification-warning"] svg {{ fill: {CHART_WARNING_COLOR} !important; }}
    div[data-testid="stAlert"][data-baseweb="notification-error"] {{ border-left-color: {CHART_ERROR_COLOR} !important; background-color: rgba(231, 76, 60, 0.1) !important; }} div[data-testid="stAlert"][data-baseweb="notification-error"] div[role="alert"] {{ color: {CHART_ERROR_COLOR} !important; }} div[data-testid="stAlert"][data-baseweb="notification-error"] svg {{ fill: {CHART_ERROR_COLOR} !important; }}
    div[data-testid="stAlert"][data-baseweb="notification-success"] {{ border-left-color: {CHART_SUCCESS_COLOR} !important; background-color: rgba(46, 204, 113, 0.1) !important; }} div[data-testid="stAlert"][data-baseweb="notification-success"] div[role="alert"] {{ color: {CHART_SUCCESS_COLOR} !important; }} div[data-testid="stAlert"][data-baseweb="notification-success"] svg {{ fill: {CHART_SUCCESS_COLOR} !important; }}

    /* --- Metrics Styling (Forcing White Label) --- */
    .stMetric {{ background-color: {CARD_BACKGROUND_COLOR}; border-radius: 10px; padding: 1.2rem 1.5rem; border: 1px solid {INPUT_BORDER_COLOR}; border-left: 4px solid {PRIMARY_ACCENT_COLOR}; margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
    .stMetric > label {{ /* Target the label specifically */
        color: {BODY_TEXT_COLOR} !important; /* FORCE WHITE COLOR */
        font-size: 0.9em !important; /* Ensure size is applied */
        font-weight: 600 !important; /* Ensure weight is applied */
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 0.3rem !important;
    }}
    .stMetric > div:nth-of-type(1) {{ color: {MAIN_TITLE_COLOR} !important; font-size: 1.8em; font-weight: 700; font-family: {TITLE_FONT}; line-height: 1.2; }}
    .stMetric .stMetricDelta {{ font-size: 0.9em; font-weight: 600; margin-top: 0.2rem; }}
    .stMetric .stMetricDelta > div[data-delta-direction="increase"] {{ color: {CHART_SUCCESS_COLOR} !important; }}
    .stMetric .stMetricDelta > div[data-delta-direction="decrease"] {{ color: {CHART_ERROR_COLOR} !important; }}

    /* --- Footer Styling --- */
    .footer {{ text-align: center; color: {SUBTITLE_COLOR}; opacity: 0.7; margin: 4rem auto 1rem auto; font-size: 0.9rem; max-width: 1100px; padding-bottom: 1rem; }}
    .footer p {{ font-size: 0.9rem !important; color: {SUBTITLE_COLOR} !important; margin: 0; }}
    .footer a {{ color: {PRIMARY_ACCENT_COLOR}; text-decoration: none; }} .footer a:hover {{ text-decoration: underline; }}

    /* --- Streamlit Cleanup --- */
    header[data-testid="stHeader"], footer {{ display: none !important; }}
    div[data-testid="stDecoration"] {{ display: none !important; }}

    /* --- Responsive Adjustments --- */
    @media (max-width: 768px) {{
        .main .block-container {{ padding: 2rem 1rem 3rem 1rem; }} .header-container {{ margin-bottom: 2.5rem; }} .logo {{ height: 60px; margin-right: 1rem;}} .logo-placeholder {{ height: 60px; width: 60px; }} .title {{ font-size: 2.2rem; }} h2, h3 {{ font-size: 1.5rem; }}
        div[data-testid="stButton"] > button, div[data-testid="stDownloadButton"] > button {{ padding: 0.5rem 1.2rem; }} .footer {{ margin-top: 2rem; font-size: 0.8rem; }} .stMetric {{ padding: 1rem 1.2rem; }} .stMetric > div:nth-of-type(1) {{ font-size: 1.6em; }}
    }}
     @media (max-width: 480px) {{
         .header-container {{ flex-direction: column; text-align: center; gap: 0.8rem; margin-bottom: 2rem; }} .logo {{ margin-right: 0; height: 50px; }} .logo-placeholder {{ margin-right: 0; height: 50px; width: 50px; }} .title {{ font-size: 2rem; }}
         div[data-testid="stDataFrame"] td {{ font-size: 0.8rem; padding: 0.5rem 0.5rem; }} div[data-testid="stDataFrame"] th {{ font-size: 0.75rem; padding: 0.6rem 0.5rem; }} .stMetric {{ padding: 0.8rem 1rem; }} .stMetric > div:nth-of-type(1) {{ font-size: 1.4em; }}
     }}
</style>
"""

# --- 3. Inject the custom CSS ---
st.markdown(APP_STYLE, unsafe_allow_html=True)

# --- Header ---
st.markdown(
    f"""
    <div class="header-container">
        {logo_html}
        <h1 class="title">Phronesis Pulse 2.0</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Helper Functions ---

# *** NEW: Helper function for address formatting ***
def get_formatted_address(info: dict) -> str:
    """Formats the address from the yfinance info dictionary."""
    address_parts = [
        info.get('address1'),
        f"{info.get('city', '')}, {info.get('state', '')} {info.get('zip', '')}".strip(', '),
        info.get('country')
    ]
    # Filter out None or empty parts and join them with newlines
    full_address = "\n".join(part for part in address_parts if part)
    return full_address if full_address else "N/A"

# *** MODIFIED: TTM values are now handled correctly to avoid 0 for NaN sums ***
def get_financial_data(ticker: str) -> pd.DataFrame:
    """ Fetches annual and TTM financial data, using np.nan for missing TTM numerics. """
    try:
        logging.info(f"Fetching financial data for {ticker}...")
        stock = yf.Ticker(ticker)
        info = stock.info
        financials = stock.financials

        if financials.empty:
            logging.warning(f"No annual financial data found for {ticker}.")
            return pd.DataFrame({'Ticker': [ticker]})

        df = financials.T.copy()
        df['Ticker'] = ticker
        df['Full_Date'] = pd.to_datetime(df.index).strftime('%Y-%m-%d')
        df = df.reset_index(drop=True)
        df['Year_Index'] = df.index + 1
        df['Currency'] = info.get('currency', 'N/A')
        df['Financial_Currency'] = info.get('financialCurrency', 'N/A')

        q_financials = stock.quarterly_financials
        ttm_data = {'Ticker': ticker, 'Full_Date': "TTM", 'Year_Index': 0}
        ttm_data['Currency'] = info.get('currency', 'N/A')
        ttm_data['Financial_Currency'] = info.get('financialCurrency', 'N/A')

        potential_numeric_cols = list(set(df.select_dtypes(include=np.number).columns.tolist() + NUMERIC_COLUMNS_TO_CLEAN))
        
        if not q_financials.empty and q_financials.shape[1] >= 4:
            last_4_quarters = q_financials.iloc[:, :4]
            ttm_series = last_4_quarters.sum(axis=1, numeric_only=True)
            
            # --- FIX FOR TTM ZEROS ---
            # If all 4 quarters for a metric are NaN, its sum is 0. Replace this with NaN.
            for metric in ttm_series.index:
                if last_4_quarters.loc[metric].isnull().all():
                    ttm_series[metric] = np.nan # Set to NaN instead of 0
            # --- END FIX ---
            
            common_metrics = df.columns.intersection(ttm_series.index)
            for metric in common_metrics:
                 if metric not in ttm_data:
                     ttm_data[metric] = ttm_series.get(metric)
        else:
            logging.info(f"Insufficient quarterly data for TTM for {ticker}. TTM values set to NaN.")
            for metric in df.columns:
                if metric in potential_numeric_cols:
                    ttm_data[metric] = np.nan

        ttm_df = pd.DataFrame([ttm_data])
        final_df = pd.concat([ttm_df, df], ignore_index=True, sort=False)
        
        logging.info(f"Successfully fetched financial data for {ticker}.")
        return final_df

    except Exception as e:
        logging.warning(f"Error getting financial data for {ticker}: {e}")
        return pd.DataFrame({'Ticker': [ticker]})

# *** MODIFIED: Integrated call to get_formatted_address ***
def get_profile_data(ticker: str) -> pd.DataFrame:
    """ Fetches company profile data, including the formatted address. """
    try:
        logging.info(f"Fetching profile data for {ticker}...")
        stock = yf.Ticker(ticker)
        info = stock.info
        
        address = get_formatted_address(info) # Call the new helper function

        company_info = {
            'Ticker': ticker,
            'LongName': info.get('longName', 'N/A'),
            'Long_Business_Summary': info.get('longBusinessSummary', 'N/A'),
            'Country': info.get('country', 'N/A'),
            'Address': address, # Add the formatted address
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Full_Time_Employees': info.get('fullTimeEmployees'),
            'Website': info.get('website', 'N/A'),
            'Phone': info.get('phone', 'N/A')
        }
        
        fte = company_info['Full_Time_Employees']
        if fte is None:
            company_info['Full_Time_Employees'] = 'N/A'
        else:
            try:
                company_info['Full_Time_Employees'] = f"{pd.to_numeric(fte):,.0f}"
            except (ValueError, TypeError):
                company_info['Full_Time_Employees'] = str(fte)
                
        logging.info(f"Successfully fetched profile data for {ticker}.")
        return pd.DataFrame([company_info])
        
    except Exception as e:
        logging.warning(f"Error getting profile data for {ticker}: {e}")
        return pd.DataFrame({'Ticker': [ticker]})

def create_excel_download(df: pd.DataFrame, filename: str) -> bytes:
    """Creates an Excel file in memory for downloading."""
    output = BytesIO()
    df_display = df.copy()
    for col in df_display.select_dtypes(include=['object']).columns:
        df_display[col] = df_display[col].astype(str)
        
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_display.to_excel(writer, index=False, sheet_name='Data')
        worksheet = writer.sheets['Data']
        for i, col in enumerate(df_display.columns):
            try:
                max_len_data = df_display[col].map(len).max()
                if pd.isna(max_len_data): max_len_data = 0
                max_len_col = len(str(col))
                column_width = max(int(max_len_data), max_len_col) + 2
                worksheet.set_column(i, i, min(column_width, 50))
            except Exception as width_e:
                logging.warning(f"Could not calculate width for column {col}: {width_e}")
                worksheet.set_column(i, i, 20)
    return output.getvalue()

# --- Initialize Session State ---
if 'tickers_to_process' not in st.session_state: st.session_state.tickers_to_process = []
if 'processed_data' not in st.session_state: st.session_state.processed_data = pd.DataFrame()
if 'all_extracted_data' not in st.session_state: st.session_state.all_extracted_data = pd.DataFrame()
if 'last_run_summary' not in st.session_state: st.session_state.last_run_summary = {}
if 'last_ticker_input' not in st.session_state: st.session_state.last_ticker_input = DEFAULT_TICKERS

# --- Streamlit App Layout ---

# --- Input Section ---
st.subheader("Enter Stock Tickers")
ticker_input_area = st.text_area(
    "Enter ticker symbols separated by commas (e.g., GOOGL,AAPL,MSFT). Avoid spaces.",
    value=st.session_state.last_ticker_input,
    height=80, key="ticker_input_area",
    help="Provide comma-separated stock ticker symbols like 'MSFT,GOOGL'. Data from Yahoo Finance."
)

# --- Action Button (Centered) ---
col_spacer1, col_button, col_spacer2 = st.columns([1, 2, 1])
with col_button:
    get_data_pressed = st.button("📊 Get Company Data", key="get_data_button", use_container_width=True)

if get_data_pressed:
    # 1. Validate Tickers
    tickers_raw = [ticker.strip().upper() for ticker in ticker_input_area.split(',') if ticker.strip()]
    st.session_state.last_ticker_input = ticker_input_area

    if not tickers_raw:
        st.warning("Please enter valid ticker symbols.")
        st.session_state.tickers_to_process = []
        st.session_state.processed_data = pd.DataFrame()
        st.session_state.all_extracted_data = pd.DataFrame()
        st.session_state.last_run_summary = {}
        st.rerun()
    else:
        st.session_state.tickers_to_process = tickers_raw
        st.session_state.processed_data = pd.DataFrame()
        st.session_state.all_extracted_data = pd.DataFrame()
        st.session_state.last_run_summary = {}

        # 2. Fetch Data (with Spinner)
        all_financial_dfs = []
        all_profile_dfs = []
        failed_tickers_profile = []
        failed_tickers_financial = []
        total_tickers_to_process = len(st.session_state.tickers_to_process)

        with st.spinner(f"Fetching data for {total_tickers_to_process} ticker(s)... Please wait."):
            for ticker in st.session_state.tickers_to_process:
                profile_df = get_profile_data(ticker)
                financial_df = get_financial_data(ticker)
                
                if len(profile_df.columns) > 1: all_profile_dfs.append(profile_df)
                else: failed_tickers_profile.append(ticker)
                
                if len(financial_df.columns) > 1: all_financial_dfs.append(financial_df)
                else: failed_tickers_financial.append(ticker)

        # 3. Process & Consolidate Data (Post-Spinner)
        processing_status = st.empty()
        # This consolidation logic handles cases where one of the dataframes might be empty
        # and merges them correctly. No major changes were needed here, as the fixes were in the data-gathering functions.
        try:
            if not all_profile_dfs and not all_financial_dfs:
                processing_status.error("No data could be extracted for any ticker.")
                # Clear state if total failure
                st.session_state.processed_data = pd.DataFrame()
                st.session_state.all_extracted_data = pd.DataFrame()
            else:
                processing_status.info("Consolidating and cleaning data...")
                
                combined_profile_df = pd.concat(all_profile_dfs, ignore_index=True) if all_profile_dfs else pd.DataFrame()
                combined_financial_df = pd.concat(all_financial_dfs, ignore_index=True) if all_financial_dfs else pd.DataFrame()

                if not combined_profile_df.empty and not combined_financial_df.empty:
                    final_df = pd.merge(combined_profile_df, combined_financial_df, on='Ticker', how='outer')
                elif not combined_profile_df.empty:
                    final_df = combined_profile_df
                elif not combined_financial_df.empty:
                    final_df = combined_financial_df
                else:
                    final_df = pd.DataFrame() # Should not happen if first check passes

                if final_df.empty:
                    processing_status.error("Data consolidation resulted in an empty DataFrame.")
                else:
                    st.session_state.all_extracted_data = final_df.copy()
                    
                    # Ensure all desired columns exist, adding missing ones with NaN
                    for col in FINANCIAL_COLUMNS_TO_SELECT:
                        if col not in final_df.columns:
                            final_df[col] = np.nan
                    
                    final_display_dt = final_df[FINANCIAL_COLUMNS_TO_SELECT].copy()

                    for col in NUMERIC_COLUMNS_TO_CLEAN:
                        if col in final_display_dt.columns:
                            final_display_dt[col] = pd.to_numeric(final_display_dt[col], errors='coerce')
                    
                    # Clean up string columns for display
                    str_cols_to_clean = ['Full_Time_Employees', 'Address']
                    for col in str_cols_to_clean:
                        if col in final_display_dt.columns:
                            final_display_dt[col] = final_display_dt[col].astype(str).replace(['nan', 'None', '<NA>'], 'N/A', regex=False)

                    final_display_dt = final_display_dt.sort_values(by=['Ticker', 'Year_Index'], ascending=[True, False], na_position='last')
                    st.session_state.processed_data = final_display_dt.reset_index(drop=True)
                    
                    processing_status.success("Data processing complete.")
                    time.sleep(1.5)
                    processing_status.empty()

        except Exception as merge_error:
             processing_status.error(f"An error occurred during data consolidation/cleaning: {merge_error}")
             logging.exception("Data merging error:")
             st.session_state.processed_data = pd.DataFrame()
             st.session_state.all_extracted_data = pd.DataFrame()

        # 4. Store Summary Info
        total_submitted = len(st.session_state.get('tickers_to_process', []))
        successful_tickers_count = 0
        if 'processed_data' in st.session_state and not st.session_state.processed_data.empty:
            successful_tickers_count = st.session_state.processed_data['Ticker'].nunique()
            
        failed_tickers_all = list(set(failed_tickers_profile + failed_tickers_financial))
        total_failed_count = len(failed_tickers_all)
        
        st.session_state.last_run_summary = {
            "submitted": total_submitted,
            "successful": successful_tickers_count,
            "failed_total": total_failed_count,
            "failed_list": failed_tickers_all
        }

        st.rerun()

# --- Display Results Section ---
if 'processed_data' in st.session_state and not st.session_state.processed_data.empty:
    st.subheader("Processed Data")
    st.dataframe(st.session_state.processed_data)

    # --- Download Buttons ---
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        try:
            excel_display_data = create_excel_download(st.session_state.processed_data, "Pulse_yf_FormattedData.xlsx")
            st.download_button(
                label="📥 Download Formatted Data",
                data=excel_display_data,
                file_name='Pulse_yf_FormattedData.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key="download_display_button",
                help="Downloads the cleaned and formatted table shown above."
            )
        except Exception as e:
            st.error(f"Error creating formatted download: {e}")
            logging.error(f"Error creating display data Excel: {e}")
            
    with col_dl2:
       if 'all_extracted_data' in st.session_state and not st.session_state.all_extracted_data.empty:
            try:
                # Re-order the raw data for a more logical export
                all_cols = st.session_state.all_extracted_data.columns.tolist()
                ordered_cols = [col for col in FINANCIAL_COLUMNS_TO_SELECT if col in all_cols] + \
                               [col for col in all_cols if col not in FINANCIAL_COLUMNS_TO_SELECT]
                sort_cols = ['Ticker']
                if 'Year_Index' in ordered_cols:
                    sort_cols.append('Year_Index')
                    
                all_data_ordered = st.session_state.all_extracted_data[ordered_cols].copy()
                
                # Convert numeric-like columns to strings for raw export, replacing NaNs
                for col in all_data_ordered.columns:
                    if pd.api.types.is_numeric_dtype(all_data_ordered[col]):
                        all_data_ordered[col] = all_data_ordered[col].astype(str).replace(['nan', 'None', '<NA>'], '', regex=False)

                all_data_ordered = all_data_ordered.sort_values(by=sort_cols, ascending=[True, False], na_position='last').reset_index(drop=True)
                excel_all_data = create_excel_download(all_data_ordered, "Pulse_yf_AllExtractedData.xlsx")
                st.download_button(
                    label="📦 Download All Raw Data",
                    data=excel_all_data,
                    file_name='Pulse_yf_AllExtractedData.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    key="download_all_button",
                    help="Downloads all columns retrieved before extensive cleaning/formatting."
                )
            except Exception as e:
                st.error(f"Error creating raw download: {e}")
                logging.error(f"Error creating all data Excel: {e}")

# --- Display Summary Section ---
if 'last_run_summary' in st.session_state and st.session_state.last_run_summary:
    st.markdown("---")
    st.subheader("Last Run Summary")
    summary = st.session_state.last_run_summary
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1: st.metric("Tickers Submitted", summary.get("submitted", 0))
    with metric_col2: st.metric("Tickers Successful", summary.get("successful", 0))
    with metric_col3: st.metric("Tickers with Issues", summary.get("failed_total", 0), delta=f"{summary.get('failed_total', 0)} issues", delta_color="inverse" if summary.get("failed_total", 0) > 0 else "off")
    if summary.get("failed_list"):
        with st.expander(f"View Tickers with Issues ({len(summary['failed_list'])})"):
            st.warning("Could not retrieve complete data for the following tickers:")
            for ticker in sorted(summary['failed_list']): st.markdown(f"- {ticker}")

# --- Footer ---
st.markdown("---")
st.markdown(
    f"""
    <div class="footer">
        <p>© {pd.Timestamp.now().year} Phronesis Partners. Phronesis Pulse v2.0 - Data sourced via yfinance.</p>
    </div>
    """,
    unsafe_allow_html=True
)
