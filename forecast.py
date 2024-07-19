import pandas as pd
import numpy as np

forecast_df = pd.read_csv("C:/Users/Bhoomika/Documents/Financial Analysis Tool/data/Forecast_8191_Final.csv", index_col = [0]).sort_index()
forecast_df['DATES'] = pd.to_datetime(forecast_df['DATES'])
forecast_df = forecast_df.loc[(forecast_df['DATES']>="2023-08-01") & (forecast_df['DATES']<="2025-04-01")]


grouped = forecast_df.groupby(['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID'])['MPG_ID'].count()
forecast_combinations = []

for group, mpg_count in grouped.items():                     # Loop through each group and its corresponding MPG_ID count in the grouped dictionary
    market, account_id, channel_id, mpg_id = group            # From the group tuple assin values into individual variables for easier access
    forecast_combinations.append([market, account_id, channel_id, mpg_id])

error_metrics = pd.read_csv("C:/Users/Bhoomika/Documents/Financial Analysis Tool/data/ErrorMetrics (1).csv")

best_model = pd.read_csv("C:/Users/Bhoomika/Documents/Financial Analysis Tool/data/best_model_8191 (1).csv", index_col=[0])
