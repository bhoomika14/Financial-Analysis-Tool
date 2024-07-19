from darts.models import NBEATSModel, BlockRNNModel
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data_transform import training_data, data_series, transform_data, data_inverse_transform
import streamlit as st

test_size = 9
# stopper = EarlyStopping(
#       monitor="val_loss",
#       patience=5,
#       min_delta=0.05,
#       mode='min',
#   )
# pl_trainer_kwargs = {"callbacks": [stopper]}

@st.cache_resource
def nbeats(train_data):
  nbeats_model = NBEATSModel(         # Initialize an NBEATSModel with specified parameters
      input_chunk_length=24,          # Set the input sequence length to 24 time steps
      output_chunk_length=9,  # Set the output sequence length to the value of test_size
      random_state=0                 # Set the random seed for reproducibility
      )  # Pass the trainer keyword arguments -> early stopping callback
  nbeats_model.fit(             # Train the NBEATS model
      train_data,               # Provide the training data series
      epochs=100               # Train the model for up to 100 epochs
)                               # Use the training data series also as validation data
  return nbeats_model

@st.cache_resource
def gru(train_data):
  gru_model = BlockRNNModel(
      model = 'GRU',
      input_chunk_length=24,
      output_chunk_length=9,
      n_rnn_layers=50,
      hidden_dim=50,
      dropout=0.2,
      random_state = 0,
      
  )
  gru_model.fit(train_data, epochs = 100)
  return gru_model

@st.cache_resource
def lstm(train_data):
    lstm_model = BlockRNNModel(
        model = 'LSTM',
        input_chunk_length=24,
        output_chunk_length=9,
        n_rnn_layers=50,
        hidden_dim=50,
        dropout=0.2,
        random_state = 0,
    )
    lstm_model.fit(train_data, epochs = 100)
    return lstm_model

@st.cache_resource
def model_training(training_data):
  nbeats_model = nbeats(training_data)
  gru_model = gru(transform_data(training_data))
  lstm_model = lstm(transform_data(training_data))
  return nbeats_model, gru_model, lstm_model

@st.cache_resource
def prediction(model, training, time_step):
   pred = model.predict(time_step, series = training)
   return pred

#Models training
nbeats_for_prediction, gru_for_prediction, lstm_for_prediction = model_training(training_data)
nbeats_for_forecast, gru_for_forecast, lstm_for_forecast = model_training(data_series)

#Models prediction
nbeats_prediction = prediction(nbeats_for_prediction, training_data, test_size)
gru_prediction = data_inverse_transform(training_data, prediction(gru_for_prediction, transform_data(training_data), test_size))
lstm_prediction = data_inverse_transform(training_data, prediction(lstm_for_prediction, transform_data(training_data), test_size))

#Models forecast
nbeats_forecast = prediction(nbeats_for_forecast, data_series, 24)
gru_forecast = data_inverse_transform(data_series, prediction(gru_for_forecast, transform_data(data_series), 24))
lstm_forecast = data_inverse_transform(data_series, prediction(lstm_for_forecast, transform_data(data_series), 24))

prediction_models = {'NBEATS': nbeats_prediction, 'GRU': gru_prediction, 'LSTM': lstm_prediction}
forecast_models = {'NBEATS': nbeats_forecast, 'GRU': gru_forecast, 'LSTM': lstm_forecast}

