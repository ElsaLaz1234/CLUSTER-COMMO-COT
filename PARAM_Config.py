# Data - Tickers

tickers = {
    'Energy' : ['CL=F', 'NG=F', 'RB=F'],
    'Metals' : ['GC=F','SI=F','HG=F'],
    'Grains' : ['ZC=F','ZW=F','ZS=F'],
    'Softs'  : ['KC=F','SB=F', 'CC=F', 'CT=F'],
    'Livestocks' : ['LE=F', 'HE=F', 'GF=F']
    }

tickers_label = {
    'CL=F': ['Crude Oil', 'New York Mercantile Exchange'],
    'NG=F': ['Natural Gas', 'New York Mercantile Exchange'],
    'RB=F': ['Gasoline RBOB', 'New York Mercantile Exchange'],
    'GC=F': ['Gold', 'Commodity Exchange Inc.'],
    'SI=F': ['Silver', 'Commodity Exchange Inc.'],
    'HG=F': ['Copper', 'Commodity Exchange Inc.'],
    'ZC=F': ['Corn', 'Chicago Board of Trade'],
    'ZW=F': ['Wheat', 'Chicago Board of Trade'],
    'ZS=F': ['Soybean', 'Chicago Board of Trade'],
    'KC=F': ['Coffee', 'ICE Futures U.S.'],
    'SB=F': ['Sugar', 'ICE Futures U.S.'],
    'CC=F': ['Cocoa', 'ICE Futures U.S.'],
    'CT=F': ['Cotton', 'ICE Futures U.S.'],
    'LE=F': ['Live Cattle', 'Chicago Mercantile Exchange'],
    'HE=F': ['Lean Hogs', 'Chicago Mercantile Exchange'],
    'GF=F': ['Feeder Cattle', 'Chicago Mercantile Exchange'],
}
sector_c = {
    'Energy': 'red',
    'Metals': 'grey',
    'Grains': 'green',
    'Softs': 'yellow',
    'Livestocks': 'blue'
    }

# Data - COT

COT_category = {
    'MM': 'Managed Money',
    'PM': 'Producer/Merchant',
    'SD': 'Swap Dealers',
    'OR': 'Other Reportables'
    }

# PARAM TIME & CLUESTER
start_date = '2006-01-31'
end_date = '2024-12-31'

k_c = 5

lags = [0,4,8,12]
ref_lag = 8
lag_labels = {
    0 : 'h=0',
    4 : 'h=1',
    8 : 'h=2',
    12: 'h=3',
}

rolling_window_COT = 52 #COT feature
rolling_window_r = 16 #Return co-movement feature

# Subperiods

subperiods ={
    "GFC_2008_2009":       ("2008-01-01", "2009-12-31"),
    "Oil_crash_2014_2015": ("2014-07-01", "2016-01-01"),
    "COVID_2020":          ("2020-01-01", "2020-12-31"),
    "Gas_shock_2021_2022": ("2021-01-01", "2022-12-31"),
    "Baseline_2016_2017":  ("2016-02-01", "2017-12-31"),
    "Baseline_2018_2019":  ("2018-01-01", "2019-12-31"),
    "Baseline_2023":       ("2023-01-01", "2023-12-31"),
    }
calm_periods = ["Baseline_2016_2017", "Baseline_2018_2019", "Baseline_2023"]
crisis_periods =["GFC_2008_2009", "Oil_crash_2014_2015",  "COVID_2020", "Gas_shock_2021_2022"]

# Output repertory
data_rep = 'data'
results_rep = 'results'
