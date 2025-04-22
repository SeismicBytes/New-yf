import yfinance as yf
import pandas as pd
import streamlit as st
import base64
from io import BytesIO
from PIL import Image
import math
import logging
import numpy as np # Import NumPy

# --- Configuration ---
# (Keep your existing configuration)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGO_PATH = "ppl_logo.jpg"
BACKGROUND_PATH = "wp.jpg"
DEFAULT_TICKERS = "GOOGL,AAPL,MSFT,AMZN"
FINANCIAL_COLUMNS_TO_SELECT = [
    'Ticker', 'LongName', 'Long_Business_Summary', 'Country', 'Sector', 'Industry',
    'Full_Time_Employees', 'Website', 'Phone', 'Full_Date', 'Year_Index',
    'Currency', 'Financial_Currency', 'Total Revenue', 'Operating Revenue',
    'Cost Of Revenue', 'Gross Profit', 'Operating Expense',
    'Selling General And Administrative', 'Selling General And Administration',
    'Operating Income', 'EBIT', 'Normalized EBITDA', 'Net Income'
]
# Identify columns that SHOULD be numeric after fetching/merging
NUMERIC_COLUMNS_TO_CLEAN = [
    'Total Revenue', 'Operating Revenue', 'Cost Of Revenue', 'Gross Profit',
    'Operating Expense', 'Selling General And Administrative',
    'Selling General And Administration', 'Operating Income', 'EBIT',
    'Normalized EBITDA', 'Net Income',
    # Add 'Full_Time_Employees' here if you want it treated strictly as a number
    # 'Full_Time_Employees'
]


# --- Helper Functions ---

# set_background function remains the same as the previous good version
def set_background(image_file: str):
    """Sets the background image and applies improved CSS styling for readability."""
    try:
        with open(image_file, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode()
        st.markdown(
            f"""
            <style>
            /* --- Base Background --- */
            .stApp {{
                background-image: url(data:image/png;base64,{encoded_string});
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed; /* Keeps background fixed */
            }}

            /* --- General Text Readability --- */
            body, .stMarkdown, p, label, .stException {{
                color: white !important;
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7);
            }}
            .stSuccess > div > div > div {{ color: #90EE90 !important; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7); }}
            .stWarning > div > div > div {{ color: #FFD700 !important; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7); }}
            .stError   > div > div > div {{ color: #F08080 !important; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7); }}
            .stInfo    > div > div > div {{ color: #ADD8E6 !important; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7); }}

            /* --- Headers --- */
            h1, h2, h3, h4, h5, h6 {{
                 color: white !important;
                 background-color: rgba(0, 0, 0, 0.65);
                 padding: 10px 15px;
                 border-radius: 8px;
                 margin-top: 15px;
                 margin-bottom: 10px;
                 border-left: 5px solid #4F8BF9;
                 text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
            }}
            .main-title {{
                 background-color: rgba(0, 0, 0, 0.75) !important;
                 border-left: 8px solid #4F8BF9 !important;
                 text-align: center !important;
                 margin-top: 20px !important;
            }}
            .subtitle {{
                text-align: center !important;
                color: lightgrey !important;
                text-shadow: 1px 1px 2px black !important;
                background-color: transparent !important;
                padding: 0 !important;
                border: none !important;
                margin-top: -10px !important;
                margin-bottom: 15px !important;
            }}

            /* --- Buttons --- */
            .stButton>button {{
                color: #FFFFFF;
                background-color: #4F8BF9;
                border: 1px solid #FFFFFF;
                border-radius: 5px;
                padding: 0.6em 1.2em;
                font-weight: bold;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
                transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
            }}
            .stButton>button:hover {{
                background-color: #3F6EBD;
                color: #FFFFFF;
                border-color: #E0E0E0;
                box-shadow: 3px 3px 7px rgba(0, 0, 0, 0.4);
            }}
            .stButton>button:active {{
                 background-color: #2E4C8A;
                 box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
            }}
            .stButton>button:focus {{
                outline: none !important;
                box-shadow: 0 0 0 2px rgba(79, 139, 249, 0.5);
            }}

            /* --- Text Input / Text Area --- */
            .stTextInput>div>div>input, .stTextArea>div>div>textarea {{
                background-color: rgba(255, 255, 255, 0.9);
                color: #212529;
                border: 1px solid #ced4da;
                border-radius: 5px;
                padding: 10px;
                text-shadow: none;
            }}
            .stTextInput label, .stTextArea label {{
                 color: white !important;
                 text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7);
                 margin-bottom: 5px;
             }}
            .stTextArea small, .stTextInput small {{
                 color: #E0E0E0 !important;
                 text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.6);
            }}

            /* --- Dataframes --- */
            .stDataFrame {{
                 background-color: rgba(255, 255, 255, 0.92);
                 border-radius: 8px;
                 padding: 5px;
                 border: 1px solid rgba(0,0,0,0.1);
            }}
             .stDataFrame table {{ color: #212529; }}
             .stDataFrame th {{
                 background-color: #e9ecef;
                 color: #212529;
                 text-shadow: none;
                 font-weight: bold;
                 border-bottom: 2px solid #dee2e6;
             }}
             .stDataFrame td {{
                 text-shadow: none;
                 border-top: 1px solid #dee2e6;
             }}
             .stDataFrame tr:nth-child(even) td {{
                background-color: rgba(0, 0, 0, 0.03);
             }}

            /* --- Logo --- */
            .stImage img {{ border-radius: 5px; }}

            /* --- Footer --- */
            footer {{ color: lightgrey !important; text-shadow: 1px 1px 2px black; }}
            footer p {{
                color: lightgrey !important;
                text-shadow: 1px 1px 2px black !important;
                font-size: small !important;
                text-align: center !important;
            }}
            footer a {{ color: #ADD8E6 !important; }}

            /* --- Progress Bar --- */
            .stProgress > div > div > div > div {{ background-color: #4F8BF9; }}
            .stProgress > div > div > div {{ background-color: rgba(255, 255, 255, 0.5); }}

            /* --- Separator --- */
            hr {{ border-top: 1px solid rgba(255, 255, 255, 0.3); margin-top: 1.5rem; margin-bottom: 1.5rem; }}

            /* --- Metrics --- */
            .stMetric {{
                 background-color: rgba(0, 0, 0, 0.5);
                 border-radius: 8px;
                 padding: 15px;
                 border-left: 5px solid #4F8BF9;
                 margin-bottom: 10px;
            }}
            .stMetric > label {{
                color: #E0E0E0 !important;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.7);
                font-size: 0.9em;
            }}
            .stMetric > div {{
                 color: white !important;
                 text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
                 font-size: 1.5em;
                 font-weight: bold;
            }}
            .stMetric .stMetricDelta {{ color: #90EE90 !important; }}
            </style>
            """,
            unsafe_allow_html=True
        )
        logging.info(f"Background image '{image_file}' set and custom CSS applied.")
    except FileNotFoundError:
        st.error(f"Background image file not found: {image_file}")
        logging.error(f"Background image file not found: {image_file}")
    except Exception as e:
        st.error(f"Error setting background or applying CSS: {e}")
        logging.error(f"Error setting background or applying CSS: {e}")


# *** MODIFIED get_financial_data ***
def get_financial_data(ticker: str) -> pd.DataFrame:
    """ Fetches annual and TTM financial data, using np.nan for missing TTM numerics. """
    try:
        logging.info(f"Fetching financial data for {ticker}...")
        stock = yf.Ticker(ticker)
        info = stock.info # Fetch info once

        # --- Annual Data ---
        financials = stock.financials
        if financials.empty:
            st.warning(f"No annual financial data found for {ticker}.")
            return pd.DataFrame({'Ticker': [ticker]}) # Return minimal DF

        df = financials.T.copy()
        df['Ticker'] = ticker
        df['Full_Date'] = pd.to_datetime(df.index).strftime('%Y-%m-%d')
        df = df.reset_index(drop=True)
        df['Year_Index'] = df.index + 1

        df['Currency'] = info.get('currency', 'N/A')
        df['Financial_Currency'] = info.get('financialCurrency', 'N/A')

        # --- TTM Data ---
        q_financials = stock.quarterly_financials
        ttm_data = {'Ticker': ticker, 'Full_Date': "TTM", 'Year_Index': 0}
        ttm_data['Currency'] = info.get('currency', 'N/A')
        ttm_data['Financial_Currency'] = info.get('financialCurrency', 'N/A')

        # Identify potential numeric columns expected from financials
        # (Be conservative, include likely candidates)
        potential_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        # Also add columns commonly returned as numbers even if sometimes object
        potential_numeric_cols.extend([
            'Total Revenue', 'Operating Revenue', 'Cost Of Revenue', 'Gross Profit',
            'Operating Expense', 'Selling General And Administrative', 'Selling General And Administration',
            'Operating Income', 'EBIT', 'Normalized EBITDA', 'Net Income'
        ])
        potential_numeric_cols = list(set(potential_numeric_cols)) # Unique list

        if not q_financials.empty and q_financials.shape[1] >= 4:
            ttm_series = q_financials.iloc[:, :4].sum(axis=1, numeric_only=True)
            common_metrics = df.columns.intersection(ttm_series.index)
            for metric in common_metrics:
                 if metric not in ttm_data:
                    ttm_data[metric] = ttm_series.get(metric) # Use .get for safety
        else:
            st.info(f"Insufficient quarterly data to calculate TTM for {ticker}. TTM financial values set to NaN.")
            # *** CHANGE: Use np.nan instead of None for missing numeric TTM data ***
            financial_metrics = [col for col in df.columns if col not in ['Ticker', 'Full_Date', 'Year_Index', 'Currency', 'Financial_Currency']]
            for metric in financial_metrics:
                # Only set to np.nan if it's likely a numeric column, otherwise keep as None/missing
                if metric in potential_numeric_cols:
                    ttm_data[metric] = np.nan # Use NaN for missing numerics
                else:
                    pass # Let it be missing, pandas will handle object columns

        ttm_df = pd.DataFrame([ttm_data])

        # Combine TTM and annual data
        final_df = pd.concat([ttm_df, df], ignore_index=True, sort=False)

        logging.info(f"Successfully fetched financial data for {ticker}.")
        return final_df

    except Exception as e:
        st.warning(f"Error getting financial data for {ticker}: {e}")
        logging.warning(f"Error getting financial data for {ticker}: {e}")
        return pd.DataFrame({'Ticker': [ticker]}) # Return minimal DF

# get_profile_data remains the same as previous good version
def get_profile_data(ticker: str) -> pd.DataFrame:
    """ Fetches company profile data for a given stock ticker using yfinance. """
    try:
        logging.info(f"Fetching profile data for {ticker}...")
        stock = yf.Ticker(ticker)
        info = stock.info

        company_info = {
            'Ticker': ticker,
            'LongName': info.get('longName', 'N/A'),
            'Long_Business_Summary': info.get('longBusinessSummary', 'N/A'),
            'Country': info.get('country', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Full_Time_Employees': info.get('fullTimeEmployees'), # Get raw value
            'Website': info.get('website', 'N/A'),
            'Phone': info.get('phone', 'N/A')
        }

        fte = company_info['Full_Time_Employees']
        if fte is None:
            company_info['Full_Time_Employees'] = 'N/A' # Keep as N/A string if None initially
        else:
            # Attempt to convert to numeric first for cleaning/formatting
            try:
                fte_numeric = pd.to_numeric(fte)
                company_info['Full_Time_Employees'] = f"{fte_numeric:,.0f}" # Format number with commas
            except (ValueError, TypeError):
                 # If conversion fails, keep original string representation
                 company_info['Full_Time_Employees'] = str(fte)

        logging.info(f"Successfully fetched profile data for {ticker}.")
        return pd.DataFrame([company_info])

    except Exception as e:
        st.warning(f"Error getting profile data for {ticker}: {e}")
        logging.warning(f"Error getting profile data for {ticker}: {e}")
        return pd.DataFrame({'Ticker': [ticker]}) # Return minimal DF

# create_excel_download remains the same as previous good version
def create_excel_download(df: pd.DataFrame, filename: str) -> bytes:
    """Creates an Excel file in memory for downloading."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
        worksheet = writer.sheets['Data']
        for i, col in enumerate(df.columns):
            try: # Add try-except for robust column width calculation
                max_len_data = df[col].astype(str).map(len).max()
                if pd.isna(max_len_data): max_len_data = 0 # Handle all-NA columns
                max_len_col = len(col)
                column_width = max(int(max_len_data), max_len_col) + 2
                worksheet.set_column(i, i, min(column_width, 50))
            except Exception as width_e:
                 logging.warning(f"Could not calculate width for column {col}: {width_e}")
                 worksheet.set_column(i, i, 20) # Default width on error
    return output.getvalue()


# --- Streamlit App ---
st.set_page_config(page_title="Phronesis Pulse 2.0", page_icon="📊", layout="wide")
set_background(BACKGROUND_PATH)

# --- Header ---
col_logo, col_title_spacer = st.columns([1, 5])
with col_logo:
    try:
        logo = Image.open(LOGO_PATH)
        st.image(logo, width=130)
    except FileNotFoundError:
        st.error(f"Logo image not found: {LOGO_PATH}")
    except Exception as e:
        st.error(f"Error loading logo: {e}")
st.markdown("<h1 class='main-title'>Phronesis Pulse 2.0</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Financial Data Extractor</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Initialize Session State ---
if 'tickers' not in st.session_state: st.session_state.tickers = []
if 'processed_data' not in st.session_state: st.session_state.processed_data = pd.DataFrame()
if 'all_extracted_data' not in st.session_state: st.session_state.all_extracted_data = pd.DataFrame()

# --- Ticker Input Area ---
st.subheader("1. Enter Stock Tickers")
ticker_input = st.text_area(
    "Enter ticker symbols separated by commas (e.g., GOOGL,AAPL,MSFT). Avoid spaces.",
    value=DEFAULT_TICKERS, height=80, key="ticker_input_area",
    help="Provide comma-separated stock ticker symbols like 'MSFT,GOOGL'. Data from Yahoo Finance."
)
if st.button("Load Tickers", key="load_tickers_button"):
    tickers_raw = [ticker.strip().upper() for ticker in ticker_input.split(',') if ticker.strip()]
    if tickers_raw:
        st.session_state.tickers = tickers_raw
        st.success(f"{len(st.session_state.tickers)} tickers loaded: {', '.join(st.session_state.tickers)}")
        st.session_state.processed_data = pd.DataFrame()
        st.session_state.all_extracted_data = pd.DataFrame()
    else:
        st.warning("Please enter valid ticker symbols.")
        st.session_state.tickers = []

# --- Data Extraction Section ---
if st.session_state.tickers:
    st.markdown("---")
    st.subheader("2. Configure and Extract Data")
    if st.button("Extract Financial Data", key="extract_data_button"):
        # (Data fetching logic remains largely the same)
        all_financial_dfs = []
        all_profile_dfs = []
        failed_tickers_profile = []
        failed_tickers_financial = []
        total_tickers_to_process = len(st.session_state.tickers)

        st.info(f"Starting data extraction for {total_tickers_to_process} tickers...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, ticker in enumerate(st.session_state.tickers):
            current_progress = (i) / total_tickers_to_process
            progress_bar.progress(current_progress)
            status_text.text(f"Processing {ticker} ({i+1}/{total_tickers_to_process})...")

            profile_df = get_profile_data(ticker)
            financial_df = get_financial_data(ticker)

            if len(profile_df.columns) > 1: all_profile_dfs.append(profile_df)
            else: failed_tickers_profile.append(ticker)

            if len(financial_df.columns) > 1: all_financial_dfs.append(financial_df)
            else: failed_tickers_financial.append(ticker)

            progress_bar.progress((i + 1) / total_tickers_to_process)

        status_text.success(f"Data extraction complete for {total_tickers_to_process} tickers.")
        progress_bar.empty()

        if failed_tickers_profile: st.warning(f"Could not retrieve profile data for: {', '.join(failed_tickers_profile)}")
        if failed_tickers_financial: st.warning(f"Could not retrieve financial data for: {', '.join(failed_tickers_financial)}")

        if not all_profile_dfs or not all_financial_dfs:
            st.error("No data could be extracted successfully. Please check tickers and network connection.")
            st.session_state.processed_data = pd.DataFrame()
            st.session_state.all_extracted_data = pd.DataFrame()
        else:
            st.markdown("---")
            st.subheader("3. Processed Results")
            try:
                combined_profile_df = pd.concat(all_profile_dfs, ignore_index=True)
                combined_financial_df = pd.concat(all_financial_dfs, ignore_index=True)
                final_df = pd.merge(combined_profile_df, combined_financial_df, on='Ticker', how='inner')

                if final_df.empty:
                    st.error("Data merging resulted in an empty DataFrame. No tickers had both profile and financial data.")
                    st.session_state.processed_data = pd.DataFrame()
                    st.session_state.all_extracted_data = pd.DataFrame()
                else:
                    st.session_state.all_extracted_data = final_df.copy() # Store before selecting/cleaning

                    existing_display_columns = [col for col in FINANCIAL_COLUMNS_TO_SELECT if col in final_df.columns]
                    missing_display_columns = [col for col in FINANCIAL_COLUMNS_TO_SELECT if col not in final_df.columns]
                    if missing_display_columns:
                        st.info(f"Note: Columns not found and omitted from display: {', '.join(missing_display_columns)}")

                    # Create the display dataframe
                    final_display_dt = final_df[existing_display_columns].copy() # Use .copy() to avoid SettingWithCopyWarning

                    # *** ADDED DATA CLEANING STEP ***
                    st.write("Cleaning data types...") # Temporary message
                    for col in NUMERIC_COLUMNS_TO_CLEAN:
                        if col in final_display_dt.columns:
                            # Convert to numeric, forcing errors to NaN
                            final_display_dt[col] = pd.to_numeric(final_display_dt[col], errors='coerce')
                            # Optional: Convert Int columns containing NaN to nullable Int type
                            # if final_display_dt[col].isnull().any() and pd.api.types.is_integer_dtype(final_display_dt[col].dropna()):
                            #      try:
                            #          final_display_dt[col] = final_display_dt[col].astype('Int64') # Pandas nullable integer
                            #      except Exception: # Fallback if conversion fails
                            #          pass
                    # Handle 'Full_Time_Employees' separately if kept as object/string but needs cleaning
                    if 'Full_Time_Employees' in final_display_dt.columns and 'Full_Time_Employees' not in NUMERIC_COLUMNS_TO_CLEAN:
                         # Example: Ensure it's string, replace non-standard missing values if necessary
                         final_display_dt['Full_Time_Employees'] = final_display_dt['Full_Time_Employees'].astype(str).replace('nan', 'N/A')


                    final_display_dt = final_display_dt.sort_values(by=['Ticker', 'Year_Index'], ascending=[True, False])
                    st.session_state.processed_data = final_display_dt.reset_index(drop=True)
                    st.write("Data cleaning complete.") # Temporary message

            except Exception as merge_error:
                 st.error(f"An error occurred during data consolidation/cleaning: {merge_error}")
                 logging.exception("Data merging/consolidation/cleaning error:") # Log full traceback
                 st.session_state.processed_data = pd.DataFrame()
                 st.session_state.all_extracted_data = pd.DataFrame()


# --- Display Results and Download ---
if not st.session_state.processed_data.empty:
    st.subheader("4. View and Download Data")

    # Debug: Print dtypes just before display to confirm cleaning worked
    # st.write("Debug: Final dtypes before st.dataframe:")
    # st.write(st.session_state.processed_data.dtypes)

    # Display the cleaned DataFrame
    st.dataframe(st.session_state.processed_data) # This should now work

    # --- Download Buttons ---
    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        try:
            # Use the cleaned data for the "Displayed Data" download
            excel_display_data = create_excel_download(st.session_state.processed_data, "Pulse_yf_FormattedData.xlsx")
            st.download_button(label="📥 Download Displayed Data (Excel)", data=excel_display_data,
                               file_name='Pulse_yf_FormattedData.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                               key="download_display_button", help="Downloads the cleaned table shown above.")
        except Exception as e:
            st.error(f"Error creating displayed data download file: {e}")
    with col_dl2:
       if not st.session_state.all_extracted_data.empty:
            try:
                # Download raw(ish) data (before numeric coercion)
                # Reorder and sort the 'all_extracted_data' before download
                all_cols = st.session_state.all_extracted_data.columns.tolist()
                ordered_cols = [col for col in FINANCIAL_COLUMNS_TO_SELECT if col in all_cols] + \
                               [col for col in all_cols if col not in FINANCIAL_COLUMNS_TO_SELECT]
                all_data_ordered = st.session_state.all_extracted_data[ordered_cols].sort_values(
                    by=['Ticker', 'Year_Index'], ascending=[True, False]
                ).reset_index(drop=True)

                excel_all_data = create_excel_download(all_data_ordered, "Pulse_yf_AllExtractedData.xlsx")
                st.download_button(label="📦 Download All Extracted Data (Excel)", data=excel_all_data,
                                   file_name='Pulse_yf_AllExtractedData.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                   key="download_all_button", help="Downloads all columns retrieved before cleaning/formatting.")
            except Exception as e:
                 st.error(f"Error creating all data download file: {e}")

    # --- Summary Statistics ---
    st.markdown("---")
    st.subheader("Extraction Summary")
    total_submitted = len(st.session_state.tickers)
    successful_tickers_count = st.session_state.processed_data['Ticker'].nunique()
    failed_count = total_submitted - successful_tickers_count
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1: st.metric("Tickers Submitted", total_submitted)
    with metric_col2: st.metric("Tickers Successfully Processed", successful_tickers_count)
    with metric_col3: st.metric("Tickers with Issues", failed_count)

elif 'tickers' in st.session_state and st.session_state.tickers and st.session_state.processed_data.empty:
    st.info("Click 'Extract Financial Data' above to begin processing the loaded tickers.")

# --- Footer ---
st.markdown("---")
st.markdown("<p>Phronesis Pulse v2.0 - Powered by yfinance and Streamlit</p>", unsafe_allow_html=True)
