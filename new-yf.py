import yfinance as yf
import pandas as pd
import streamlit as st
import base64
from io import BytesIO
from PIL import Image
import math
import logging # Use logging for cleaner error/info messages

# --- Configuration ---
# Use logging instead of just print/st.error for more structured output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for file paths and default values
LOGO_PATH = "ppl_logo.jpg"  # Make sure this file exists in the same directory
BACKGROUND_PATH = "wp.jpg" # Make sure this file exists in the same directory
DEFAULT_TICKERS = "GOOGL,AAPL,MSFT,AMZN" # Example default tickers
FINANCIAL_COLUMNS_TO_SELECT = [
    # Profile Info (Merged)
    'Ticker', 'LongName', 'Long_Business_Summary', 'Country', 'Sector', 'Industry',
    'Full_Time_Employees', 'Website', 'Phone',
    # Financial Info
    'Full_Date', 'Year_Index', 'Currency', 'Financial_Currency',
    # Key Financial Metrics (Ensure these match yfinance output)
    'Total Revenue', 'Operating Revenue', 'Cost Of Revenue', 'Gross Profit',
    'Operating Expense', 'Selling General And Administration', # Often listed under Operating Expense
    'Selling General And Administration', # Check if yfinance uses this exact name
    'Operating Income', 'EBIT', 'Normalized EBITDA', # EBIT/EBITDA might not always be directly available or need calculation
    'Net Income'
]

# --- Helper Functions ---

def set_background(image_file: str):
    """Sets the background image and applies improved CSS styling for readability."""
    try:
        with open(image_file, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode()

        # --- Improved CSS ---
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
            /* Make default text white with a shadow for contrast */
            body, .stMarkdown, p, label, .stException {{
                color: white !important; /* Force white text */
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7); /* Dark shadow for pop */
            }}
             /* Specific styling for message types */
            .stSuccess > div > div > div {{ color: #90EE90 !important; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7); }} /* Light Green for success */
            .stWarning > div > div > div {{ color: #FFD700 !important; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7); }} /* Gold for warning */
            .stError   > div > div > div {{ color: #F08080 !important; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7); }} /* Light Coral for error */
            .stInfo    > div > div > div {{ color: #ADD8E6 !important; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7); }} /* Light Blue for info */


            /* --- Headers --- */
            /* Darker background, more padding, white text */
            h1, h2, h3, h4, h5, h6 {{
                 color: white !important;
                 background-color: rgba(0, 0, 0, 0.65); /* Dark semi-transparent */
                 padding: 10px 15px;
                 border-radius: 8px;
                 margin-top: 15px; /* Add some space above headers */
                 margin-bottom: 10px; /* Add some space below headers */
                 border-left: 5px solid #4F8BF9; /* Accent border */
                 text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
            }}
            /* Custom style for main title if needed, e.g., via markdown */
            .main-title {{
                 background-color: rgba(0, 0, 0, 0.75) !important;
                 border-left: 8px solid #4F8BF9 !important;
                 text-align: center !important; /* Center align */
                 margin-top: 20px !important; /* Adjust top margin specifically */
            }}
             /* Custom style for subtitle if needed */
            .subtitle {{
                text-align: center !important;
                color: lightgrey !important;
                text-shadow: 1px 1px 2px black !important;
                background-color: transparent !important; /* No background for subtitle */
                padding: 0 !important;
                border: none !important;
                margin-top: -10px !important; /* Pull subtitle closer to title */
                margin-bottom: 15px !important;
            }}

             /* --- Buttons --- */
             /* Solid background, clear text color */
            .stButton>button {{
                color: #FFFFFF; /* White text */
                background-color: #4F8BF9; /* Primary button color */
                border: 1px solid #FFFFFF; /* White border */
                border-radius: 5px;
                padding: 0.6em 1.2em;
                font-weight: bold;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
                transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease; /* Smooth transition */
            }}
            .stButton>button:hover {{
                background-color: #3F6EBD; /* Darker blue on hover */
                color: #FFFFFF;
                border-color: #E0E0E0;
                box-shadow: 3px 3px 7px rgba(0, 0, 0, 0.4);
            }}
            .stButton>button:active {{ /* Style for when button is clicked */
                 background-color: #2E4C8A;
                 box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
            }}
            .stButton>button:focus {{ /* Remove default browser focus outline if desired */
                outline: none !important;
                box-shadow: 0 0 0 2px rgba(79, 139, 249, 0.5); /* Custom focus indicator */
            }}


            /* --- Text Input / Text Area --- */
            /* Input element styling */
            .stTextInput>div>div>input, .stTextArea>div>div>textarea {{
                background-color: rgba(255, 255, 255, 0.9); /* Almost opaque white */
                color: #212529; /* Dark text color for input */
                border: 1px solid #ced4da;
                border-radius: 5px;
                padding: 10px;
                text-shadow: none; /* Remove inherited text shadow */
            }}
             /* Style the label associated with text area/input (it's usually part of the subheader/markdown) */
             /* label for text input/area (if Streamlit renders one separately) */
             .stTextInput label, .stTextArea label {{
                 color: white !important;
                 text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7);
                 margin-bottom: 5px; /* Add space below label */
             }}

             /* Helper text below input/text area */
            .stTextArea small, .stTextInput small {{
                 color: #E0E0E0 !important; /* Lighter text for helper */
                 text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.6);
            }}


             /* --- Dataframes --- */
            .stDataFrame {{
                 background-color: rgba(255, 255, 255, 0.92); /* Slightly opaque white */
                 border-radius: 8px;
                 padding: 5px; /* Add slight padding around table */
                 border: 1px solid rgba(0,0,0,0.1); /* Subtle border */
            }}
             /* Ensure dataframe text is readable (usually black by default) */
             .stDataFrame table {{
                 color: #212529; /* Dark text for table content */
             }}
             .stDataFrame th {{ /* Table headers */
                 background-color: #e9ecef; /* Light grey header */
                 color: #212529; /* Dark header text */
                 text-shadow: none;
                 font-weight: bold;
                 border-bottom: 2px solid #dee2e6;
             }}
             .stDataFrame td {{ /* Table cells */
                 text-shadow: none;
                 border-top: 1px solid #dee2e6; /* Add lines between rows */
             }}
             .stDataFrame tr:nth-child(even) td {{ /* Zebra striping */
                background-color: rgba(0, 0, 0, 0.03);
             }}


            /* --- Logo --- */
            .stImage img {{
                 border-radius: 5px;
                 /* Optional: add a background if logo has transparency issues */
                 /* background-color: rgba(255, 255, 255, 0.8); */
                 /* padding: 5px; */
            }}

            /* --- Footer --- */
            footer {{
                color: lightgrey !important; /* Ensure footer is visible */
                text-shadow: 1px 1px 2px black;
            }}
            footer p {{ /* Target the paragraph inside the footer specifically */
                color: lightgrey !important;
                text-shadow: 1px 1px 2px black !important;
                font-size: small !important;
                text-align: center !important;
            }}
            footer a {{ /* Style links in footer */
                color: #ADD8E6 !important; /* Light blue links */
            }}

            /* --- Progress Bar --- */
            .stProgress > div > div > div > div {{
                background-color: #4F8BF9; /* Solid color progress bar */
                /* OR use a gradient: */
                /* background-image: linear-gradient(to right, #4F8BF9 , #3F6EBD); */
            }}
            .stProgress > div > div > div {{ /* Background of the progress bar track */
                background-color: rgba(255, 255, 255, 0.5);
            }}

            /* --- Separator --- */
            hr {{
                border-top: 1px solid rgba(255, 255, 255, 0.3); /* Lighter separator */
                margin-top: 1.5rem; /* Add space around separator */
                margin-bottom: 1.5rem;
            }}

            /* --- Metrics --- */
            .stMetric {{
                 background-color: rgba(0, 0, 0, 0.5); /* Semi-transparent dark background */
                 border-radius: 8px;
                 padding: 15px;
                 border-left: 5px solid #4F8BF9; /* Accent border */
                 margin-bottom: 10px; /* Spacing between metrics */
            }}
            .stMetric > label {{ /* Metric label */
                color: #E0E0E0 !important; /* Lighter grey label */
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.7);
                font-size: 0.9em;
            }}
            .stMetric > div {{ /* Metric value */
                 color: white !important;
                 text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
                 font-size: 1.5em; /* Make value larger */
                 font-weight: bold;
            }}
            .stMetric .stMetricDelta {{ /* Optional: style delta indicator */
                color: #90EE90 !important; /* Default to positive color, adjust as needed */
            }}


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


def get_financial_data(ticker: str) -> pd.DataFrame:
    """
    Fetches annual and TTM financial data for a given stock ticker using yfinance.

    Args:
        ticker: The stock ticker symbol (e.g., "AAPL").

    Returns:
        A pandas DataFrame containing financial data, including a TTM row.
        Returns a DataFrame with only the 'Ticker' column if an error occurs.
    """
    try:
        logging.info(f"Fetching financial data for {ticker}...")
        stock = yf.Ticker(ticker)
        info = stock.info # Fetch info once

        # --- Annual Data ---
        financials = stock.financials
        if financials.empty:
            st.warning(f"No annual financial data found for {ticker}.")
            return pd.DataFrame({'Ticker': [ticker]})

        df = financials.T.copy() # Transpose for years as rows
        df['Ticker'] = ticker
        df['Full_Date'] = pd.to_datetime(df.index).strftime('%Y-%m-%d') # Format date
        df = df.reset_index(drop=True)
        df['Year_Index'] = df.index + 1 # Annual rows start from 1

        # Add currency info from general info
        df['Currency'] = info.get('currency', 'N/A')
        df['Financial_Currency'] = info.get('financialCurrency', 'N/A')

        # --- TTM Data ---
        q_financials = stock.quarterly_financials
        ttm_data = {'Ticker': ticker, 'Full_Date': "TTM", 'Year_Index': 0}
        ttm_data['Currency'] = info.get('currency', 'N/A')
        ttm_data['Financial_Currency'] = info.get('financialCurrency', 'N/A')

        if not q_financials.empty and q_financials.shape[1] >= 4:
            # Sum the latest four quarters for TTM
            ttm_series = q_financials.iloc[:, :4].sum(axis=1, numeric_only=True)
            # Include only metrics present in the annual data columns
            common_metrics = df.columns.intersection(ttm_series.index)
            for metric in common_metrics:
                 if metric not in ttm_data: # Avoid overwriting Ticker, Date etc.
                    ttm_data[metric] = ttm_series.get(metric) # Use .get for safety
        else:
            st.info(f"Insufficient quarterly data to calculate TTM for {ticker}. TTM financial values set to None.")
            # Set financial metrics to None if TTM cannot be calculated
            financial_metrics = [col for col in df.columns if col not in ['Ticker', 'Full_Date', 'Year_Index', 'Currency', 'Financial_Currency']]
            for metric in financial_metrics:
                ttm_data[metric] = None

        ttm_df = pd.DataFrame([ttm_data])

        # Combine TTM and annual data
        final_df = pd.concat([ttm_df, df], ignore_index=True, sort=False) # Ensure TTM is first

        logging.info(f"Successfully fetched financial data for {ticker}.")
        return final_df

    except Exception as e:
        st.warning(f"Error getting financial data for {ticker}: {e}")
        logging.warning(f"Error getting financial data for {ticker}: {e}")
        # Return a minimal DataFrame to allow merging later
        return pd.DataFrame({'Ticker': [ticker]})

def get_profile_data(ticker: str) -> pd.DataFrame:
    """
    Fetches company profile data for a given stock ticker using yfinance.

    Args:
        ticker: The stock ticker symbol (e.g., "AAPL").

    Returns:
        A pandas DataFrame containing profile data.
        Returns a DataFrame with only the 'Ticker' column if an error occurs.
    """
    try:
        logging.info(f"Fetching profile data for {ticker}...")
        stock = yf.Ticker(ticker)
        info = stock.info

        # Use .get() with default values for robustness
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
        # Format employees: handle None, format numbers nicely
        fte = company_info['Full_Time_Employees']
        if fte is None:
            company_info['Full_Time_Employees'] = 'N/A'
        elif isinstance(fte, (int, float)):
             try:
                company_info['Full_Time_Employees'] = f"{fte:,.0f}" # Format with commas
             except (ValueError, TypeError):
                 company_info['Full_Time_Employees'] = str(fte) # Fallback to string if formatting fails
        else:
            company_info['Full_Time_Employees'] = str(fte) # Convert other types to string


        logging.info(f"Successfully fetched profile data for {ticker}.")
        return pd.DataFrame([company_info])

    except Exception as e:
        st.warning(f"Error getting profile data for {ticker}: {e}")
        logging.warning(f"Error getting profile data for {ticker}: {e}")
        # Return a minimal DataFrame to allow merging later
        return pd.DataFrame({'Ticker': [ticker]})

def create_excel_download(df: pd.DataFrame, filename: str) -> bytes:
    """Creates an Excel file in memory for downloading."""
    output = BytesIO()
    # Use ExcelWriter to potentially add more sheets or formatting later
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
        # Optional: Auto-adjust column widths (can be slow for large data)
        worksheet = writer.sheets['Data']
        for i, col in enumerate(df.columns):
            # Find max length of data in the column
            max_len_data = df[col].astype(str).map(len).max()
            # Find length of column header
            max_len_col = len(col)
            # Set column width to max of data or header length, plus some buffer
            column_width = max(max_len_data, max_len_col) + 2
            # Limit max width to avoid excessively wide columns
            worksheet.set_column(i, i, min(column_width, 50))
    return output.getvalue()

# --- Streamlit App ---

# Page Configuration (MUST be the first Streamlit command)
st.set_page_config(
    page_title="Phronesis Pulse 2.0",
    page_icon="📊", # Add a relevant emoji icon
    layout="wide"
)

# Apply background image and custom CSS (Place this right after set_page_config)
set_background(BACKGROUND_PATH)

# --- Header ---
# Use columns for layout, adjust ratios as needed
col_logo, col_title_spacer = st.columns([1, 5]) # Spacer column to push title

with col_logo:
    try:
        logo = Image.open(LOGO_PATH)
        st.image(logo, width=130) # Slightly larger logo
    except FileNotFoundError:
        st.error(f"Logo image not found: {LOGO_PATH}")
        logging.error(f"Logo image not found: {LOGO_PATH}")
    except Exception as e:
        st.error(f"Error loading logo: {e}")
        logging.error(f"Error loading logo: {e}")

# Add title and subtitle using markdown with custom classes for styling
st.markdown("<h1 class='main-title'>Phronesis Pulse 2.0</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Financial Data Extractor</p>", unsafe_allow_html=True)


st.markdown("---") # Visual separator

# --- Initialize Session State ---
if 'tickers' not in st.session_state:
    st.session_state.tickers = []
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = pd.DataFrame()
if 'all_extracted_data' not in st.session_state:
    st.session_state.all_extracted_data = pd.DataFrame()


# --- Ticker Input Area ---
# Optional: Wrap input section in a container for visual grouping (CSS can target containers)
# with st.container():
st.subheader("1. Enter Stock Tickers")
ticker_input = st.text_area(
    "Enter ticker symbols separated by commas (e.g., GOOGL,AAPL,MSFT). Avoid spaces.",
    value=DEFAULT_TICKERS,
    height=80, # Slightly smaller height
    key="ticker_input_area", # Unique key for the widget
    help="Provide comma-separated stock ticker symbols like 'MSFT,GOOGL'. Data from Yahoo Finance."
)

if st.button("Load Tickers", key="load_tickers_button"):
    # Basic validation: split, strip whitespace, remove empty strings, convert to uppercase
    tickers_raw = [ticker.strip().upper() for ticker in ticker_input.split(',') if ticker.strip()]
    if tickers_raw:
        st.session_state.tickers = tickers_raw
        st.success(f"{len(st.session_state.tickers)} tickers loaded: {', '.join(st.session_state.tickers)}")
        # Clear previous results when new tickers are loaded
        st.session_state.processed_data = pd.DataFrame()
        st.session_state.all_extracted_data = pd.DataFrame()
    else:
        st.warning("Please enter valid ticker symbols.")
        st.session_state.tickers = [] # Clear if input is invalid


# --- Data Extraction Section ---
if st.session_state.tickers:
    st.markdown("---")
    st.subheader("2. Configure and Extract Data")

    if st.button("Extract Financial Data", key="extract_data_button"):
        all_financial_dfs = []
        all_profile_dfs = []
        failed_tickers_profile = []
        failed_tickers_financial = []
        total_tickers_to_process = len(st.session_state.tickers)

        st.info(f"Starting data extraction for {total_tickers_to_process} tickers...")
        progress_bar = st.progress(0)
        status_text = st.empty() # Placeholder for status updates

        # Process all tickers in one go
        for i, ticker in enumerate(st.session_state.tickers):
            current_progress = (i) / total_tickers_to_process # Progress before processing current ticker
            progress_bar.progress(current_progress)
            status_text.text(f"Processing {ticker} ({i+1}/{total_tickers_to_process})...")

            profile_df = get_profile_data(ticker)
            financial_df = get_financial_data(ticker)

            # Check if data fetching was successful (minimal df indicates failure)
            if len(profile_df.columns) > 1: # More than just 'Ticker'
                 all_profile_dfs.append(profile_df)
            else:
                 failed_tickers_profile.append(ticker)

            if len(financial_df.columns) > 1: # More than just 'Ticker'
                all_financial_dfs.append(financial_df)
            else:
                 failed_tickers_financial.append(ticker)

            # Update progress after processing
            progress_bar.progress((i + 1) / total_tickers_to_process)

        status_text.success(f"Data extraction complete for {total_tickers_to_process} tickers.")
        progress_bar.empty() # Remove progress bar after completion

        # Report failures
        if failed_tickers_profile:
             st.warning(f"Could not retrieve profile data for: {', '.join(failed_tickers_profile)}")
        if failed_tickers_financial:
             st.warning(f"Could not retrieve financial data for: {', '.join(failed_tickers_financial)}")

        # Check if *any* data was successfully fetched before proceeding
        if not all_profile_dfs or not all_financial_dfs:
            st.error("No data could be extracted successfully. Please check tickers and network connection.")
            st.session_state.processed_data = pd.DataFrame()
            st.session_state.all_extracted_data = pd.DataFrame()
        else:
            # --- Consolidate and Process Data ---
            st.markdown("---") # Separator moved here
            st.subheader("3. Processed Results")

            # Combine all successfully fetched dataframes
            # Use inner join to only include tickers where both profile and financials were found
            try:
                combined_profile_df = pd.concat(all_profile_dfs, ignore_index=True)
                combined_financial_df = pd.concat(all_financial_dfs, ignore_index=True)

                # Merge profile and financial data using an inner join
                # This ensures we only keep rows where we have both profile and financial data.
                final_df = pd.merge(combined_profile_df, combined_financial_df, on='Ticker', how='inner')

                if final_df.empty:
                    st.error("Data merging resulted in an empty DataFrame. This might happen if no ticker had both profile and financial data successfully retrieved.")
                    st.session_state.processed_data = pd.DataFrame()
                    st.session_state.all_extracted_data = pd.DataFrame()
                else:
                    # Store the full merged data before selecting columns
                    st.session_state.all_extracted_data = final_df.copy()

                    # --- Select and Order Display Columns ---
                    # Get columns that actually exist in the merged dataframe
                    existing_display_columns = [col for col in FINANCIAL_COLUMNS_TO_SELECT if col in final_df.columns]
                    missing_display_columns = [col for col in FINANCIAL_COLUMNS_TO_SELECT if col not in final_df.columns]

                    if missing_display_columns:
                         st.info(f"Note: The following requested columns were not found in the data and will be omitted from the default view: {', '.join(missing_display_columns)}")

                    # Create the display dataframe
                    final_display_dt = final_df[existing_display_columns]

                    # Format specific columns for better display (optional)
                    # Example: Format numerical columns (handle potential errors if data isn't numeric)
                    # numerical_cols = ['Full_Time_Employees', 'Total Revenue', 'Gross Profit', 'Net Income'] # Add more as needed
                    # for col in numerical_cols:
                    #     if col in final_display_dt.columns:
                    #         # Use errors='coerce' to turn non-numeric into NaN, then fillna
                    #         final_display_dt[col] = pd.to_numeric(final_display_dt[col], errors='coerce')
                    #         # You might want different formatting based on the column
                    #         if 'Employees' in col:
                    #              final_display_dt[col] = final_display_dt[col].map('{:,.0f}'.format, na_action='ignore')
                    #         else: # Assume currency/large numbers
                    #              final_display_dt[col] = final_display_dt[col].map('{:,.0f}'.format, na_action='ignore')


                    # Sort data: Ticker ascending, then Year_Index descending (TTM first)
                    final_display_dt = final_display_dt.sort_values(by=['Ticker', 'Year_Index'], ascending=[True, False])

                    # Store the processed data for display
                    st.session_state.processed_data = final_display_dt.reset_index(drop=True) # Reset index after sorting

            except Exception as merge_error:
                 st.error(f"An error occurred during data consolidation: {merge_error}")
                 logging.error(f"Data merging/consolidation error: {merge_error}")
                 st.session_state.processed_data = pd.DataFrame()
                 st.session_state.all_extracted_data = pd.DataFrame()


# --- Display Results and Download ---
if not st.session_state.processed_data.empty:
    # Separator and header moved to section 3, display starts here
    st.subheader("4. View and Download Data")

    st.dataframe(st.session_state.processed_data)

    # --- Download Buttons ---
    st.markdown("---") # Separator before downloads
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        # Download Button for Displayed/Formatted Data
        try:
            excel_display_data = create_excel_download(
                st.session_state.processed_data,
                "Pulse_yf_FormattedData.xlsx"
            )
            st.download_button(
                label="📥 Download Displayed Data (Excel)",
                data=excel_display_data,
                file_name='Pulse_yf_FormattedData.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key="download_display_button",
                help="Downloads the table shown above with selected columns."
            )
        except Exception as e:
            st.error(f"Error creating displayed data download file: {e}")
            logging.error(f"Error creating displayed data download file: {e}")


    with col_dl2:
       # Download Button for All Extracted Data
       if not st.session_state.all_extracted_data.empty:
            try:
                # Optional: Reorder columns for the "All Data" file (display columns first)
                all_cols = st.session_state.all_extracted_data.columns.tolist()
                ordered_cols = [col for col in FINANCIAL_COLUMNS_TO_SELECT if col in all_cols] + \
                               [col for col in all_cols if col not in FINANCIAL_COLUMNS_TO_SELECT]
                # Sort the full data similarly to the displayed data
                all_data_ordered = st.session_state.all_extracted_data[ordered_cols].sort_values(
                    by=['Ticker', 'Year_Index'], ascending=[True, False]
                ).reset_index(drop=True)


                excel_all_data = create_excel_download(
                    all_data_ordered, # Use the reordered and sorted dataframe
                    "Pulse_yf_AllExtractedData.xlsx"
                )
                st.download_button(
                    label="📦 Download All Extracted Data (Excel)",
                    data=excel_all_data,
                    file_name='Pulse_yf_AllExtractedData.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    key="download_all_button",
                    help="Downloads all columns retrieved from yfinance before selection/formatting."
                )
            except Exception as e:
                 st.error(f"Error creating all data download file: {e}")
                 logging.error(f"Error creating all data download file: {e}")


    # --- Summary Statistics ---
    st.markdown("---")
    st.subheader("Extraction Summary")
    total_submitted = len(st.session_state.tickers)
    # Count unique tickers in the *final processed* data (accounts for merge failures)
    successful_tickers_count = st.session_state.processed_data['Ticker'].nunique()
    # Calculate failed based on the difference between submitted and successfully processed *and merged*
    failed_count = total_submitted - successful_tickers_count

    # Use columns for better layout of metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Tickers Submitted", total_submitted)
    with metric_col2:
        st.metric("Tickers Successfully Processed", successful_tickers_count)
    with metric_col3:
        st.metric("Tickers with Issues", failed_count)


elif 'tickers' in st.session_state and st.session_state.tickers and st.session_state.processed_data.empty:
    # Show this specifically if tickers are loaded but processing hasn't happened or failed entirely
    st.info("Click 'Extract Financial Data' above to begin processing the loaded tickers.")

# --- Footer ---
st.markdown("---")
# Use markdown paragraph for footer styling consistency
st.markdown("<p>Phronesis Pulse v2.0 - Powered by yfinance and Streamlit</p>", unsafe_allow_html=True)
