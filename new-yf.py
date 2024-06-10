import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.options import Options
import streamlit as st
import warnings
from io import BytesIO
import os

pd.options.display.float_format = '{:.0f}'.format
warnings.filterwarnings("ignore")

# Function to switch to classic Yahoo Finance
def switch_to_classic(driver):
    driver.get("https://finance.yahoo.com/")
    driver.implicitly_wait(4)
    try:
        switch_button = driver.find_element(by=By.XPATH, value="//a[contains(@href, '/go-back') and contains(@class, 'opt-in-link')]")
        switch_button.click()
        time.sleep(2)  # Give some time for the switch to take effect
    except Exception as e:
        st.warning(f"Error switching to classic: {e}")

def scrape_financial_data(driver, ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/financials?p={ticker}"
    driver.get(url)
    driver.implicitly_wait(4)  # Wait for the page to load

    # Clicking the 'Expand All' button if present
    try:
        expand_button = driver.find_element(by=By.XPATH, value="//*[@id='Col1-1-Financials-Proxy']/section/div[2]/button/div/span")
        expand_button.click()
    except Exception:
        pass  # If the button is not found, proceed

    html = driver.execute_script('return document.body.innerHTML;')
    soup = BeautifulSoup(html, 'lxml')

    features = soup.find_all('div', class_='D(tbr)')
    if not features:
        return pd.DataFrame()  # Return empty DataFrame if no data found

    # Extracting header dates
    header_dates = []
    # Correctly escape parentheses in class names for CSS selectors
    date_headers = soup.select('.D\\(tbhg\\) .Ta\\(c\\)')
    for header in date_headers:
        date_span = header.find('span')
        if date_span:
            header_dates.append(date_span.text)
    
    headers = [item.text for item in features[0].find_all('div', class_='D(ib)')]
    final_data = []
    for feature in features[1:]:
        temp_data = [item.text for item in feature.find_all('div', class_='D(tbc)')]
        final_data.append(temp_data)

    df = pd.DataFrame(final_data)
    if len(df.columns) == len(headers):
        df.columns = headers
        df = df.set_index(headers[0]).T
        df['Ticker'] = ticker

        # Extracting currency information
        features_Curr = soup.find_all('div', class_='C($tertiaryColor) Fz(12px)')
        if features_Curr:
            df['Currency'] = features_Curr[0].text[-4:]

        features_Curr_2 = soup.find_all('span', class_='Fz(xs) C($tertiaryColor) Mstart(25px) smartphone_Mstart(0px) smartphone_D(b) smartphone_Mt(5px)')
        if features_Curr_2:
            df['Financial_Currency'] = features_Curr_2[0].text.split(".")[0][-3:]

        # Assign a year index, corresponding year, and full date to each row
        df['Year_Index'] = range(len(header_dates))
        df['Full_Date'] = header_dates

        return df

    return pd.DataFrame()  # Return empty DataFrame if headers and data columns don't match


known_countries = ['Afghanistan', 'Aland Islands', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Anguilla', 'Antarctica', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Bolivia, Plurinational State of', 'Bonaire, Sint Eustatius and Saba', 'Bosnia and Herzegovina', 'Botswana', 'Bouvet Island', 'Brazil', 'British Indian Ocean Territory', 'British Virgin Islands', 'Brunei Darussalam', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Cayman Islands', 'Central African Republic', 'Chad', 'Chile', 'China', 'Christmas Island', 'Cocos (Keeling) Islands', 'Colombia', 'Comoros', 'Congo', 'Congo, The Democratic Republic of the', 'Cook Islands', 'Costa Rica', "Côte d'Ivoire", 'Croatia', 'Cuba', 'Curaçao', 'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Falkland Islands (Malvinas)', 'Faroe Islands', 'Fiji', 'Finland', 'France', 'French Guiana', 'French Polynesia', 'French Southern Territories', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Gibraltar', 'Greece', 'Greenland', 'Grenada', 'Guadeloupe', 'Guam', 'Guatemala', 'Guernsey', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Heard Island and McDonald Islands', 'Holy See (Vatican City State)', 'Honduras', 'Hong Kong', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran, Islamic Republic of', 'Iraq', 'Ireland', 'Isle of Man', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jersey', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', "Korea, Democratic People's Republic of", 'Korea, Republic of', 'Kuwait', 'Kyrgyzstan', "Lao People's Democratic Republic", 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macao', 'Macau','Macedonia', 'Republic of', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Martinique', 'Mauritania', 'Mauritius', 'Mayotte', 'Mexico', 'Micronesia, Federated States of', 'Moldova, Republic of', 'Monaco', 'Mongolia', 'Montenegro', 'Montserrat', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Niue', 'Norfolk Island', 'Northern Mariana Islands', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestinian Territory, Occupied', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Pitcairn', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Reunion', 'Romania', 'Russia', 'Russian Federation', 'Rwanda', 'Saint Barthélemy', 'Saint Helena, Ascension and Tristan da Cunha', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Martin (French part)', 'Saint Pierre and Miquelon', 'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Sint Maarten (Dutch part)', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa','South Korea', 'South Georgia and the South Sandwich Islands', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'South Sudan', 'Svalbard and Jan Mayen', 'Swaziland', 'Sweden', 'Switzerland', 'Syrian Arab Republic', 'Taiwan, Province of China','Taiwan', 'Tajikistan', 'Tanzania, United Republic of', 'Thailand', 'Timor-Leste', 'Togo', 'Tokelau', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Turks and Caicos Islands', 'Tuvalu', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States', 'United States Minor Outlying Islands', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela, Bolivarian Republic of', 'Vietnam', 'Virgin Islands, British', 'Virgin Islands, U.S.', 'Wallis and Futuna', 'Yemen', 'Zambia', 'Zimbabwe']

def scrape_profile_data(driver, ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/profile?p={ticker}"
    driver.get(url)
    driver.implicitly_wait(4)  # Wait for page to load

    html = driver.execute_script('return document.body.innerHTML;')
    soup = BeautifulSoup(html, 'lxml')

    company_info = {}
    company_info['Ticker'] = ticker

    try:
        company_info['LongName'] = soup.find('h3', class_='Fz(m) Mb(10px)').text
        company_info['Long_Business_Summary'] = soup.find('p', class_='Mt(15px) Lh(1.6)').text
        
        address_block = driver.find_element(by=By.XPATH, value="//*[@id='Col1-0-Profile-Proxy']/section/div[1]/div/div/p[1]")
        address_lines = address_block.text.split("\n")

        # Extract the country from the address lines
        country = None
        for line in address_lines:
            # Check if this line is a known country
            if line in known_countries:
                country = line
                break

        # If country is found, assign it, otherwise assign a default value or keep it None
        company_info['Country'] = country if country else 'Not Found'

        labels = soup.find_all('span', text=['Sector(s)', 'Industry', 'Full Time Employees'])
        for label in labels:
            value = label.find_next('span', class_='Fw(600)')
            if value and value.text.isdigit():
                # Special handling for Full Time Employees since it's a number
                company_info['Full_Time_Employees'] = value.text
            elif value:
                # For Sector and Industry
                key = label.text[:-1] if label.text.endswith('(s)') else label.text  # Remove the trailing (s) from "Sector(s)"
                company_info[key] = value.text

        # Extracting additional link texts
        features_5 = soup.find_all('a', class_='C($linkColor)')
        if len(features_5) >= 2:
            company_info['Website'] = features_5[-1].text
            company_info['Phone'] = features_5[-2].text

    except Exception as e:
        st.warning(f"Error scraping {ticker}: {e}")

    return pd.DataFrame([company_info])

# Streamlit app
st.title("Yahoo Finance Scraper")

tickers = st.text_area("Enter tickers (comma-separated)", "JEL.L,ARTNA").split(',')

if st.button("Scrape Data"):
    # Define the paths to Chromium and ChromeDriver
    chrome_path = "/usr/bin/chromium-browser"
    chrome_driver_path = "/usr/bin/chromedriver"

    options = Options()
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    options.add_argument('--headless')  # Ensure headless mode is enabled
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.binary_location = chrome_path

    # Initialize the Chrome driver
    driver = webdriver.Chrome(options=options)

    # Switch to classic Yahoo Finance
    switch_to_classic(driver)

    financial_dfs = []
    profile_dfs = []
    for ticker in tickers:
        ticker = ticker.strip()
        financial_df = scrape_financial_data(driver, ticker)
        profile_df = scrape_profile_data(driver, ticker)
        financial_dfs.append(financial_df)
        profile_dfs.append(profile_df)

    combined_financial_df = pd.concat(financial_dfs, ignore_index=True)
    combined_profile_df = pd.concat(profile_dfs, ignore_index=True)

    # Merge the two DataFrames on the 'Ticker' column
    final_df = pd.merge(combined_profile_df, combined_financial_df, on='Ticker')

    driver.quit()

    # List of columns you want to select
    columns_to_select = [
        'Ticker', 'Full_Date', 'Year_Index', 'LongName', 'Long_Business_Summary',
        'Currency', 'Financial_Currency', 'Sector(s', 'Industry',
        'Full Time Employees', 'Full_Time_Employees', 'Website', 'Phone', 'Country',
        'Total Revenue', 'Operating Revenue', 'Cost of Revenue', 'Gross Profit',
        'Operating Expense', 'Selling General and Administrative',
        'Selling & Marketing Expense', 'EBIT', 'Normalized EBITDA',
        'Net Income from Continuing & Discontinued Operation', 'Operating Income'
    ]

    # Filter out only the columns that exist in your DataFrame
    existing_columns = [col for col in columns_to_select if col in final_df.columns]

    # Select only the existing columns from your DataFrame
    Final_DT = final_df[existing_columns]

    # Display the first few rows of the DataFrame
    st.dataframe(Final_DT)

    # Convert specified columns to numeric and multiply by 1000
    columns_to_convert = [
        'Total Revenue', 'Operating Revenue', 'Cost of Revenue', 'Gross Profit',
        'Operating Expense', 'Selling General and Administrative', 'EBIT',
        'Normalized EBITDA',
        'Net Income from Continuing & Discontinued Operation',
        'Operating Income'
    ]

    for column in columns_to_convert:
        if column in Final_DT.columns:
            Final_DT[column] = Final_DT[column].str.replace(',', '').str.replace('$', '')
            Final_DT[column] = pd.to_numeric(Final_DT[column], errors='coerce')
            Final_DT[column] = Final_DT[column].apply(lambda x: x * 1000)

    # Provide download link
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        Final_DT.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()

    st.download_button(
        label="Download data as Excel",
        data=processed_data,
        file_name='financial_data.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
