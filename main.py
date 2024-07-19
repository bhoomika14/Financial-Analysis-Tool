import pandas as pd
import numpy as np
from data_loading import df, data, months_seq, combinations_greater_24, combinations_lesser_24
# from data_transform import get_timeseries
# from models import prediction_models, forecast_models

df_24 = pd.DataFrame()
df, data, months_seq, combinations_greater_24, combinations_lesser_24 = df, data, months_seq, combinations_greater_24, combinations_lesser_24

###### Get the data that has greater than 24 months of data and lesser than 24 months of data ################################


for ids in combinations_greater_24:
    dataframe = data[(data['MARKET'] == ids[0]) &         # Filter the data DataFrame where the 'MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID' column matches the element of ids
                        (data['ACCOUNT_ID'] == ids[1]) &
                        (data['CHANNEL_ID'] == ids[2]) &
                        (data['MPG_ID'] == ids[3])]
    
    df_24 = pd.concat([df_24, dataframe], ignore_index=True)

# test_pred_df = pd.DataFrame(columns = ['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID', 'TYPE', 'MODEL', 'ACTUAL_AMOUNT', 'AMOUNT'])     # Initialize an empty DataFrame to store test predictions
# pred_df = pd.DataFrame()          # Initialize an empty DataFrame to combine test_predictions of every combinations
# k = 0  

# # Code block to create predictions dataframe with predicted AMOUNT and ACTUAL_AMOUNT
# for i in combinations:          # Loop through each combination in the combinations list
#   series = get_timeseries(i)      # Get the data based on ids using combinations list

#   for key, value in prediction_models.items():
#     #test_pred_df['AMOUNT'] = value[k].pd_series()
#     test_pred_df = pd.DataFrame({'MARKET': i[0], 'ACCOUNT_ID': i[1], 'CHANNEL_ID': i[2], 'MPG_ID': i[3], 'TYPE': 'past', 'MODEL': key, 'ACTUAL_AMOUNT': series.pd_series()[36:], 'AMOUNT': value[k].pd_series()})

#     test_pred_df.to_csv("Forecast_8034.csv", mode = 'a', header = False)  # Concatenate the updated test_pred_df to the pred_df DataFrame
#   k += 1      


# forecast_df = pd.DataFrame() # Dataframe to save forecasted data
# k = 0
# for i in combinations:          # Loop through each combination in the combinations list
#   series = get_timeseries(i)      # Get the data based on ids using combinations list

#   for key, value in forecast_models.items():
#     forecast_df = pd.DataFrame({'MARKET': i[0], 'ACCOUNT_ID': i[1], 'CHANNEL_ID': i[2], 'MPG_ID': i[3], 'TYPE': 'future', 'MODEL': key, 'ACTUAL_AMOUNT': np.nan, 'AMOUNT': value[k].pd_series()})

#     forecast_df.to_csv("Forecast_8034.csv", mode = 'a', header = False)         # Append new rows to csv file
#   k += 1                                            
