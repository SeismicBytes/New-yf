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

# --- ENHANCEMENT: You can now add Balance Sheet/Cash Flow items here ---
# Example: Add 'Total Assets', 'Free Cash Flow' to see them in the main table
FINANCIAL_COLUMNS_TO_SELECT = [
    'Ticker', 'LongName', 'Long_Business_Summary', 'Country', 'Sector', 'Industry',
    'Full_Time_Employees', 'Website', 'Phone', 'Full_Date', 'Year_Index',
    'Currency', 'Financial_Currency', 'Total Revenue', 'Operating Revenue',
    'Cost Of Revenue', 'Gross Profit', 'Operating Expense',
    'Selling General And Administrative', 'Selling General And Administration',
    'Operating Income', 'EBIT', 'Normalized EBITDA', 'Net Income',
    # --- Example of new columns you can add ---
    'Total Assets', 'Total Liabilities Net Minority Interest', 'Free Cash Flow'
]
NUMERIC_COLUMNS_TO_CLEAN = [
    'Total Revenue', 'Operating Revenue', 'Cost Of Revenue', 'Gross Profit',
    'Operating Expense', 'Selling General And Administrative',
    'Selling General And Administration', 'Operating Income', 'EBIT',
    'Normalized EBITDA', 'Net Income',
    # --- Add corresponding new numeric columns here as well ---
    'Total Assets', 'Total Liabilities Net Minority Interest', 'Free Cash Flow'
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

# --- 2. Apex Theme CSS Styling (Unchanged) ---
APP_STYLE = f"""
<style>
    /* --- Import Fonts --- */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400&display=swap');
    /* All other CSS is unchanged... */
    body {{ background-color: {MAIN_BACKGROUND_COLOR}; color: {BODY_TEXT_COLOR}; font-family: {BODY_FONT}; }}
    .stApp {{ background-color: {MAIN_BACKGROUND_COLOR}; color: {BODY_TEXT_COLOR}; }}
    .main .block-container {{ max-width: 1100px; padding: 2rem 1rem 4rem 1rem; }}
    .header-container {{ display: flex; align-items: center; justify-content: center; margin-bottom: 3rem; }}
    .logo {{ height: 80px; margin-right: 1.5rem; }}
    .title {{ font-family: {TITLE_FONT}; font-size: 2.8rem; color: {MAIN_TITLE_COLOR}; }}
    h2, h3 {{ font-family: {TITLE_FONT}; color: {PRIMARY_ACCENT_COLOR}; border-bottom: 1px solid rgba({PRIMARY_ACCENT_COLOR_RGB}, 0.3); padding-bottom: 0.6rem; }}
    div[data-testid="stButton"] > button, div[data-testid="stDownloadButton"] > button {{ border: 1px solid {BUTTON_PRIMARY_BORDER} !important; background-color: {BUTTON_PRIMARY_BG} !important; color: {BUTTON_PRIMARY_TEXT} !important; }}
    .stMetric > label {{ color: {BODY_TEXT_COLOR} !important; }}
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


# --- Helper Functions (REVISED AND CORRECTED) ---

def get_financial_data(ticker: str) -> pd.DataFrame:
    """
    Fetches comprehensive financial data, ensuring consistent chronological ordering
    and calculating TTM only when a full four quarters of data are available.
    """
    try:
        logging.info(f"Fetching financial data for {ticker}...")
        stock = yf.Ticker(ticker)
        info = stock.info

        # Fetch from all three financial statements
        annual_fin = stock.financials
        annual_bs = stock.balance_sheet
        annual_cf = stock.cashflow
        annual_data = pd.concat([annual_fin, annual_bs, annual_cf])
        annual_data = annual_data.loc[~annual_data.index.duplicated(keep='first')]

        if annual_data.empty:
            logging.warning(f"No annual financial data found for {ticker}.")
            return pd.DataFrame({'Ticker': [ticker]})

        # --- FIX #1: ENFORCE CONSISTENT DATE SORTING FOR YEAR_INDEX ---
        df = annual_data.T.copy()
        # Convert index to datetime objects to ensure correct sorting
        df.index = pd.to_datetime(df.index)
        # Explicitly sort by date descending. This guarantees the most recent year is always first.
        df = df.sort_index(ascending=False)
        
        # Now proceed with creating the index, which will now be consistent
        df['Ticker'] = ticker
        df['Full_Date'] = df.index.strftime('%Y-%m-%d')
        df = df.reset_index(drop=True)
        # Year_Index=1 will now reliably be the most recent annual data
        df['Year_Index'] = df.index + 1
        df['Currency'] = info.get('currency', 'N/A')
        df['Financial_Currency'] = info.get('financialCurrency', 'N/A')

        # --- FIX #2: STRICT TTM CALCULATION (Integrated from test code) ---
        q_fin = stock.quarterly_financials
        q_bs = stock.quarterly_balance_sheet
        q_cf = stock.quarterly_cashflow
        q_data = pd.concat([q_fin, q_bs, q_cf])
        q_data = q_data.loc[~q_data.index.duplicated(keep='first')]

        if q_data.empty:
            logging.warning(f"No quarterly data found for {ticker}.")
            ttm_data = {'Ticker': ticker, 'Full_Date': "TTM", 'Year_Index': 0}
            ttm_data['Currency'] = info.get('currency', 'N/A')
            ttm_data['Financial_Currency'] = info.get('financialCurrency', 'N/A')
            for metric in annual_data.index:
                ttm_data[metric] = np.nan
            ttm_df = pd.DataFrame([ttm_data])
            final_df = pd.concat([ttm_df, df], ignore_index=True, sort=False)
            return final_df

        # Enforce sorting: Convert columns to datetime and sort descending (most recent first)
        q_data.columns = pd.to_datetime(q_data.columns)
        q_data = q_data.sort_index(axis=1, ascending=False)  # Columns now sorted: newest to oldest
        logging.info(f"Quarterly data columns (dates) for {ticker}: {q_data.columns.tolist()}")

        ttm_data = {'Ticker': ticker, 'Full_Date': "TTM", 'Year_Index': 0}
        ttm_data['Currency'] = info.get('currency', 'N/A')
        ttm_data['Financial_Currency'] = info.get('financialCurrency', 'N/A')

        # Check for sufficient data
        if q_data.shape[1] >= 4:
            logging.info(f"Calculating TTM for {ticker} from quarterly data.")
            for metric in annual_data.index:
                if metric in q_data.index:
                    last_four_quarters = q_data.loc[metric].iloc[:4]  # iloc[:4] since sorted descending
                    
                    # *** CORE TTM FIX: Only calculate if we have exactly 4 non-NaN quarters ***
                    if len(last_four_quarters) == 4 and last_four_quarters.count() == 4 and not last_four_quarters.isnull().any():
                        # For Balance Sheet items, take the most recent quarter's value.
                        if any(keyword in metric for keyword in ['Asset', 'Liabilities', 'Equity', 'Capital', 'Stock']):
                            ttm_data[metric] = last_four_quarters.iloc[0]
                        # For Income/Cash Flow items, sum the four quarters.
                        else:
                            ttm_data[metric] = last_four_quarters.sum()
                    else:
                        logging.warning(f"Skipping TTM for '{metric}' in {ticker}: Incomplete data (len={len(last_four_quarters)}, count={last_four_quarters.count()}, has NaN={last_four_quarters.isnull().any()})")
                        ttm_data[metric] = np.nan
                else:
                    ttm_data[metric] = np.nan
        else:
            logging.warning(f"Insufficient quarterly columns ({q_data.shape[1]} < 4) for TTM in {ticker}. Setting all TTM values to NaN.")
            for metric in annual_data.index:
                ttm_data[metric] = np.nan

        ttm_df = pd.DataFrame([ttm_data])
        final_df = pd.concat([ttm_df, df], ignore_index=True, sort=False)
        
        logging.info(f"Successfully fetched and processed financial data for {ticker}.")
        return final_df

    except Exception as e:
        logging.error(f"CRITICAL error in get_financial_data for {ticker}: {e}")
        return pd.DataFrame({'Ticker': [ticker]})

def get_profile_data(ticker: str) -> pd.DataFrame:
    """ Fetches company profile data. (Unchanged) """
    try:
        logging.info(f"Fetching profile data for {ticker}..."); stock = yf.Ticker(ticker); info = stock.info
        company_info = {'Ticker': ticker, 'LongName': info.get('longName', 'N/A'), 'Long_Business_Summary': info.get('longBusinessSummary', 'N/A'), 'Country': info.get('country', 'N/A'), 'Sector': info.get('sector', 'N/A'), 'Industry': info.get('industry', 'N/A'), 'Full_Time_Employees': info.get('fullTimeEmployees'), 'Website': info.get('website', 'N/A'), 'Phone': info.get('phone', 'N/A')}
        fte = company_info['Full_Time_Employees']
        if fte is None: company_info['Full_Time_Employees'] = 'N/A'
        else:
            try: company_info['Full_Time_Employees'] = f"{pd.to_numeric(fte):,.0f}"
            except (ValueError, TypeError): company_info['Full_Time_Employees'] = str(fte)
        logging.info(f"Successfully fetched profile data for {ticker}.")
        return pd.DataFrame([company_info])
    except Exception as e: logging.warning(f"Error getting profile data for {ticker}: {e}"); return pd.DataFrame({'Ticker': [ticker]})

def create_excel_download(df: pd.DataFrame, filename: str) -> bytes:
    """Creates an Excel file in memory for downloading. (Unchanged)"""
    output = BytesIO(); df_display = df.copy()
    for col in df_display.select_dtypes(include=['object']).columns: df_display[col] = df_display[col].astype(str)
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_display.to_excel(writer, index=False, sheet_name='Data'); worksheet = writer.sheets['Data']
        for i, col in enumerate(df_display.columns):
            try:
                max_len_data = df_display[col].astype(str).map(len).max();
                if pd.isna(max_len_data): max_len_data = 0
                max_len_col = len(str(col)); column_width = max(int(max_len_data), max_len_col) + 2
                worksheet.set_column(i, i, min(column_width, 50))
            except Exception as width_e: logging.warning(f"Could not calculate width for column {col}: {width_e}"); worksheet.set_column(i, i, 20)
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

# --- Main Application Logic ---
if get_data_pressed:
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

        all_financial_dfs = []; all_profile_dfs = []
        failed_tickers_profile = []; failed_tickers_financial = []
        total_tickers_to_process = len(st.session_state.tickers_to_process)

        with st.spinner(f"Fetching data for {total_tickers_to_process} ticker(s)... Please wait."):
            for ticker in st.session_state.tickers_to_process:
                profile_df = get_profile_data(ticker)
                financial_df = get_financial_data(ticker)
                
                if profile_df is not None and len(profile_df.columns) > 1: all_profile_dfs.append(profile_df)
                else: failed_tickers_profile.append(ticker)

                if financial_df is not None and len(financial_df.columns) > 1: all_financial_dfs.append(financial_df)
                else: failed_tickers_financial.append(ticker)

        processing_status = st.empty()
        if not all_financial_dfs:
            processing_status.error("No financial data could be extracted for any ticker. Cannot proceed.")
        else:
            processing_status.info("Consolidating and cleaning data...")
            try:
                combined_financial_df = pd.concat(all_financial_dfs, ignore_index=True)
                if all_profile_dfs:
                    combined_profile_df = pd.concat(all_profile_dfs, ignore_index=True)
                    final_df = pd.merge(combined_profile_df, combined_financial_df, on='Ticker', how='outer')
                else:
                    final_df = combined_financial_df

                if final_df.empty:
                    processing_status.error("Data merging resulted in an empty DataFrame.")
                else:
                    st.session_state.all_extracted_data = final_df.copy()
                    existing_display_columns = [col for col in FINANCIAL_COLUMNS_TO_SELECT if col in final_df.columns]
                    final_display_dt = final_df[existing_display_columns].copy()
                    
                    all_numeric_cols = list(final_df.select_dtypes(include=np.number).columns)
                    for col in all_numeric_cols:
                        if col in final_display_dt.columns:
                            final_display_dt[col] = pd.to_numeric(final_display_dt[col], errors='coerce')
                            
                    if 'Full_Time_Employees' in final_display_dt.columns:
                         final_display_dt['Full_Time_Employees'] = final_display_dt['Full_Time_Employees'].astype(str).replace(['nan', 'None', '<NA>'], 'N/A', regex=False)

                    # THE FINAL SORT - This will now work correctly because Year_Index is consistent
                    # Sorts by Ticker, then puts TTM (0) first, then Year 1 (most recent), Year 2, etc.
                    final_display_dt = final_display_dt.sort_values(by=['Ticker', 'Year_Index'], ascending=[True, True])
                    st.session_state.processed_data = final_display_dt.reset_index(drop=True)
                    
                    processing_status.success("Data processing complete.")
                    time.sleep(1.5)
                    processing_status.empty()
            except Exception as merge_error:
                 processing_status.error(f"An error occurred during data consolidation/cleaning: {merge_error}")
                 logging.exception("Data merging error:")
                 st.session_state.processed_data = pd.DataFrame(); st.session_state.all_extracted_data = pd.DataFrame()

        total_submitted = len(st.session_state.get('tickers_to_process', [])); successful_tickers_count = 0
        if 'processed_data' in st.session_state and not st.session_state.processed_data.empty: successful_tickers_count = st.session_state.processed_data['Ticker'].nunique()
        failed_tickers_all = list(set(failed_tickers_profile + failed_tickers_financial)); total_failed_count = len(failed_tickers_all)
        st.session_state.last_run_summary = {"submitted": total_submitted, "successful": successful_tickers_count, "failed_total": total_failed_count, "failed_list": failed_tickers_all}

        st.markdown("---")
        summary = st.session_state.last_run_summary
        if summary.get("failed_total", 0) > 0: st.warning(f"Extraction attempted for {total_submitted} tickers. Issues encountered for {summary['failed_total']}.")
        elif total_submitted > 0 : st.success(f"Successfully processed data for {summary['successful']} of {total_submitted} submitted tickers.")

        st.rerun()

# --- Display Results Section ---
if 'processed_data' in st.session_state and not st.session_state.processed_data.empty:
    st.subheader("Processed Data")
    df_display = st.session_state.processed_data.copy()
    numeric_cols_in_df = [col for col in df_display.columns if pd.api.types.is_numeric_dtype(df_display[col]) and col not in ['Year_Index']]
    format_dict = {col: '{:,.0f}' for col in numeric_cols_in_df}
    st.dataframe(df_display.style.format(format_dict, na_rep="N/A"))

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        try:
            excel_display_data = create_excel_download(st.session_state.processed_data, "Pulse_yf_FormattedData.xlsx")
            st.download_button( label="📥 Download Formatted Data", data=excel_display_data, file_name='Pulse_yf_FormattedData.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key="download_display_button", help="Downloads the cleaned and formatted table shown above." )
        except Exception as e: st.error(f"Error creating formatted download: {e}"); logging.error(f"Error creating display data Excel: {e}")
    with col_dl2:
       if 'all_extracted_data' in st.session_state and not st.session_state.all_extracted_data.empty:
            try:
                all_cols = st.session_state.all_extracted_data.columns.tolist()
                ordered_cols = [col for col in FINANCIAL_COLUMNS_TO_SELECT if col in all_cols] + [col for col in all_cols if col not in FINANCIAL_COLUMNS_TO_SELECT]
                sort_cols = ['Ticker']
                if 'Year_Index' in ordered_cols: sort_cols.append('Year_Index')
                
                all_data_ordered = st.session_state.all_extracted_data[ordered_cols].copy()
                for col in all_data_ordered.select_dtypes(include=['object']).columns:
                    all_data_ordered[col] = all_data_ordered[col].astype(str).replace(['nan', 'None', '<NA>'], '', regex=False)
                
                all_data_ordered = all_data_ordered.sort_values(by=sort_cols, ascending=[True, True], na_position='last').reset_index(drop=True)
                excel_all_data = create_excel_download(all_data_ordered, "Pulse_yf_AllExtractedData.xlsx")
                st.download_button( label="📦 Download All Raw Data", data=excel_all_data, file_name='Pulse_yf_AllExtractedData.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key="download_all_button", help="Downloads ALL columns retrieved from Income, Balance, and Cash Flow statements." )
            except Exception as e: st.error(f"Error creating raw download: {e}"); logging.error(f"Error creating all data Excel: {e}")

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
