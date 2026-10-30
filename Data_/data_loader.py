import cot_reports as cot
import pandas as pd
import numpy as np
import os
import yfinance as yf

#Download the COT data
COT_data = cot.cot_all(cot_report_type= 'disaggregated_fut')
data_path = "Data_"
COT_data = COT_data.to_csv(os.path.join(data_path, "cot_disaggregated.csv"), index=False)

#Extract COMMO Data & CLEAN

df = pd.read_csv(
    os.path.join(data_path, 'cot_disaggregated.csv'),
    dtype={'CFTC_Contract_Market_Code': str},
    parse_dates=['Report_Date_as_YYYY-MM-DD'],
    low_memory=False
    )


name_list = df['Market_and_Exchange_Names'].drop_duplicates().tolist()

filtered = []

from PARAM_Config import tickers_label
for ticker, (commodity, exchange) in tickers_label.items():
    step1 = [name for name in name_list if commodity.upper() in name.upper()]
    step2 = [name for name in step1 if exchange.upper() in name.upper()]
    filtered.extend(step2)

name_filtered = filtered

exclude = ['MINI', 'MICRO', 'SWAP', 'FINANCIAL', 'CALENDAR', 'LAST DAY',
           'BLACK SEA', 'HRW', 'SOYBEAN OIL', 'SOYBEAN MEAL',
           'WHEAT - CHICAGO', 'COPPER- #1']


name_clean = [name for name in name_filtered
              if not any(word in name.upper() for word in exclude)]

name_list = name_clean

market_code_list = [
    df.loc[df["Market_and_Exchange_Names"] == name, "CFTC_Contract_Market_Code"].iloc[0]
    for name in name_list
]

columns = [
    "Report_Date_as_YYYY-MM-DD",
    "Market_and_Exchange_Names",
    "CFTC_Contract_Market_Code",
    "M_Money_Positions_Long_All", "M_Money_Positions_Short_All",
    "Prod_Merc_Positions_Long_All", "Prod_Merc_Positions_Short_All",
    "Swap_Positions_Long_All", "Swap__Positions_Short_All",
    "Other_Rept_Positions_Long_All", "Other_Rept_Positions_Short_All",
    ]
from PARAM_Config import start_date
from PARAM_Config import end_date

COT_data_cat = df[df["CFTC_Contract_Market_Code"].isin(market_code_list)]
COT_data_cat = COT_data_cat[columns].copy()
COT_data_cat["Report_Date_as_YYYY-MM-DD"] = pd.to_datetime(COT_data_cat["Report_Date_as_YYYY-MM-DD"]).dt.strftime('%Y-%m-%d')
COT_data_cat = COT_data_cat.set_index("Report_Date_as_YYYY-MM-DD").sort_index()
COT_data_cat = COT_data_cat.loc[start_date:end_date]


# COMMO Return data import & cleaning
from PARAM_Config import tickers
all_tickers = [ticker for group in tickers.values() for ticker in group]

raw = yf.download(
    all_tickers,
    start=start_date,
    end="2025-01-01",
    interval="1d",
    auto_adjust=False,
    progress=False,
)
prices_daily = raw["Close"]

COT_data_p = prices_daily.resample("W-TUE").last()
COT_data_p = COT_data_p.loc[start_date:end_date]





