from hampel import hampel
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

def normalize_minmax(data):
    """
    Min-Max normalization scales the data to the [0, 1] range.
    Advantages:
    - Preserves the relative ordering of data.
    - Easy to understand and implement.
    Disadvantages:
    - Sensitive to outliers, which can affect scaling.
    """

    minmax_scaler = MinMaxScaler()
    normalized_data = minmax_scaler.fit_transform(data.reshape(-1, 1))
    return normalized_data

def compute_ema(data, alpha):
    ema = [data[0]]  # Initial EMA is the same as the first data point

    for value in data[1:]:
        ema_value = alpha * value + (1 - alpha) * ema[-1]
        ema.append(ema_value)

    return np.array(ema)

def normalize_column(df, subset_metric_name = None, column = "metric_value", remove_outliers = None, inverse_data = None, smooth = None, normalize = None):
    
     # Filter rows where metric_name column is "synergy"
    if subset_metric_name is not None:
        df.reset_index(drop=True, inplace=True)
        subset_data = df[df['metric_name'] == subset_metric_name]
        data_to_normalize = subset_data[column].to_numpy()
        
        
    else:
        data_to_normalize = df[column].to_numpy()

    if remove_outliers:
        data_to_normalize = hampel(data_to_normalize, window_size=3, n_sigma=3.0).filtered_data

    if inverse_data:
        data_to_normalize = -data_to_normalize
        
    if normalize:
        data_to_normalize = normalize_minmax(data_to_normalize)
        
    if smooth:
        data_to_normalize = compute_ema(data_to_normalize, 0.1)

    if subset_metric_name is not None:
        df.loc[subset_data.index, column] = data_to_normalize
    else:
        df[column] = data_to_normalize
  
    return df