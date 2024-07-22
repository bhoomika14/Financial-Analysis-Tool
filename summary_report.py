from scipy import stats
from tsmoothie.smoother import LowessSmoother
import pandas as pd
from main import df, data, months_seq, combinations_greater_24, combinations_lesser_24
from scipy import stats
from darts import TimeSeries
from darts.utils.utils import ModelMode
from darts.utils.statistics import extract_trend_and_seasonality, check_seasonality

def get_outlier_count(series):

  # Reshape the 'AMOUNT' column values from the data DataFrame to a 2D array with 1 row
  d = series.values.reshape(1, -1)

  # Create a LowessSmoother object with specified parameters for smoothing
  smoother = LowessSmoother(smooth_fraction = 0.4, iterations = 4)

  # Apply the smoother to the data to obtain smoothed values
  smooth_data = smoother.smooth(d)

  # Initialize an empty list to store identified outlier values
  outlier_val = []
  outlier_index = []

  # Generate prediction intervals from the smoother
  low, up = smoother.get_intervals('prediction_interval')

  # Extract the smoothed data points and the upper and lower bounds of the intervals
  points = smoother.data[0]  # Smoothed data points
  up_points = up[0]          # Upper bounds of prediction interval
  low_points = low[0]        # Lower bounds of prediction interval

  # Loop through the smoothed data points in reverse order
  for i in range(len(points)-1, 0, -1):
    current_point = points[i]
    current_up = up_points[i]
    current_low = low_points[i]

    # Check if the current point is outside the prediction interval bounds
    if current_point > current_up or current_point < current_low:
        outlier_val.append(current_point)                   # Add the current point to the outlier list
  return len(outlier_val), outlier_val

def summary_stats(dataframe):
    df = dataframe
    summ_stats = (df.groupby(['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID']).agg(
    Frequency = ('MPG_ID', 'count'),
    Zero_Count = ('AMOUNT', lambda x: (x == 0).sum()),
    Negative_Count = ('AMOUNT', lambda x: (x < 0).sum()),
    Outlier_Count = ('AMOUNT', get_outlier_count),
    Missing_Count = ('DATES_UPD', lambda x: len(months_seq) - x.count()),
    Missing_Values = ('DATES_UPD', lambda x: ', '.join([date.strftime("%m/%d/%Y") for date in months_seq[~months_seq.isin(x)]])))
    ).reset_index()
    summ_stats['Freq_Range'] = summ_stats['Frequency'].apply(lambda x: 1 if x <= 12 else 2 if x <= 24 else 3 if x <= 36 else 4)
    print(summ_stats)
    return summ_stats


def trend_seasonality(dt):
    try:
      ts = pd.DataFrame()
      series = TimeSeries.from_dataframe(dt, time_col = "DATES", value_cols="AMOUNT", fill_missing_dates=True, freq="MS")
      trend_season = extract_trend_and_seasonality(series, method='naive', model = ModelMode.ADDITIVE)
      ts['trend'] = trend_season[0].pd_series()
      ts['seasonality'] = trend_season[1].pd_series()
      return ts
    except Exception as e:
       print(e)


#trend, seasonality = trend_seasonality(data)