import pandas as pd
import numpy as np
import streamlit as st
from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values
from darts.dataprocessing.transformers import Scaler
from data_loading import data, months_seq, combinations_greater_24

@st.cache_data
#function that returns all the rows of each combination in the form of Darts Time Series
def get_timeseries(ids):  
            #list of unique combinations/sequence - ex: [8034, -2, 9054, 376995]
    dataset = data[(data['MARKET'] == ids[0]) &         # Filter the data DataFrame where the 'MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID' column matches the element of ids
                        (data['ACCOUNT_ID'] == ids[1]) &
                        (data['CHANNEL_ID'] == ids[2]) &
                        (data['MPG_ID'] == ids[3])] 
        
    dataset.set_index("DATES_UPD", inplace = True)                                        #past_dataset is a function that takes grouped dataset. ids paramter consist of ids of one combination

    if len(dataset) < 45:                                                   # Check if the length of the dataset of one combination is less than 45
        missing_dates = months_seq[~months_seq.isin(dataset.index)]         # Identify the dates that are missing in the dataset

        # Concatenate the 'AMOUNT' column of the dataset with reindexed 'AMOUNT' column
        dataset = pd.concat([dataset['AMOUNT'], dataset['AMOUNT'].reindex(missing_dates, fill_value = np.nan)])

        # Convert the dataset into a Darts TimeSeries object
        series = TimeSeries.from_series(dataset)

        # Fill the missing values in the TimeSeries using linear interpolation
        timeseries = fill_missing_values(series, fill = 'auto', method = 'linear')

    # If the dataset has 45 or more entries
    else:
        # Convert the 'AMOUNT' column directly into a TimeSeries object
        timeseries = TimeSeries.from_series(dataset['AMOUNT'])

    # Return the TimeSeries object
    return timeseries

@st.cache_data
def transform_data(train_data):
    transform = Scaler()
    train_data = transform.fit_transform(train_data)
    return train_data

@st.cache_data
def data_inverse_transform(train_data, preds):
  transform = Scaler()
  transform.fit(train_data)
  predictions = transform.inverse_transform(preds)
  return predictions

k = 1                       # Initializing a variable k with the value 1; Used as index for splitting data
train_dict = {}             # Initialize an empty dictionary to store train data of all combinations in the form of TimeSeries object
test_dict = {}              # Initialize an empty dictionary to store test data of all combinations in the form of TimeSeries object
data_dict = {}              # Initialize an empty dictionary to store 45 data points of all combinations in the form of TimeSeries object

for i in combinations_greater_24:                                              # Loop through each combination in the combinations list
    # Get the data based on ids using combinations list
    series = get_timeseries(i)                         # Call the get_timeseries function with the current combination to get the time series data
    train_data, test_data = series[:36], series[36:]                 # Generate the training data from the series using the train_series function

    train_dict[f'train_{k}'] = train_data           # Store the training data in the train_dict with a key of the format 'train_k'
    test_dict[f'test_{k}'] = test_data              # Store the test data in the test_dict with a key of the format 'test_k'
    data_dict[f'data_{k}'] = series
    k += 1                                          # Increment k by 1 to ensure unique keys for the next iteration

training_data = list(train_dict.values())          #Store all the training data values that is in the train_dict as a list
data_series = list(data_dict.values())             #Store all the data values that is in the data_dict as a list
