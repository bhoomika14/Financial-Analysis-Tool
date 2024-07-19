import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_data(file):
    # loading data as pandas dataframe
    df = pd.read_csv(file, index_col=[0])

    # assigning the data type of DATES_UPD column as datetime
    df['DATES_UPD'] = pd.to_datetime(df['DATES_UPD'])

    df.drop(['CURRENCY_ID', 'PERIOD_DATE', 'PROD_ID' , 'month_index'], axis=1, inplace=True)

    # filtering the original dataframe to one specific market
    data = df[df['MARKET']==8191]

    # setting the DATES_UPD column as index
    #data.set_index("DATES_UPD", inplace=True)

    #data[['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID']] = data[['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID']].astype(str)

    # getting the total date range from the DATES_UPD column
    months_seq = pd.date_range(start = min(data['DATES_UPD']), end=max(data['DATES_UPD']), freq='MS')

    # grouping the filtered data to get the combinations and getting the count of MPG_ID for each combinations
    grouped = data.groupby(['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID'])['MPG_ID'].count()


    combinations_greater_24 = []                                           # Initialize an empty list to store all unique combinations
    combinations_lesser_24 = []

    for group, mpg_count in grouped.items():                     # Loop through each group and its corresponding MPG_ID count in the grouped dictionary
        market, account_id, channel_id, mpg_id = group            # From the group tuple assin values into individual variables for easier access
        if mpg_count >= 24:                                      # Check if the count of MPG_IDs in the group is greater than or equal to 36                         # Iterate mpg_count times
            combinations_greater_24.append([market, account_id, channel_id, mpg_id])  # Append the current group values to the combinations list
        else:
            combinations_lesser_24.append([market, account_id, channel_id, mpg_id])
    return df, data, months_seq, combinations_greater_24, combinations_lesser_24


file = "C:/Users/Bhoomika/Documents/Financial Analysis Tool/data/gm_input_NA_updated (1).csv"
df, data, months_seq, combinations_greater_24, combinations_lesser_24 = load_data(file)