import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import timedelta

# Helper functions
def setup_dataframe(df):
    # basic checks and setup
    df = df[df['y'].notnull()].copy()
    df['y'] = pd.to_numeric(df['y'])
    if np.isinf(df['y'].values).any():
        raise ValueError("Found infinity in column y")
    df['ds'] = pd.to_datetime(df['ds'])
    if df['ds'].isnull().any():
        raise ValueError("Found NaN in column ds")
    df = df.sort_values('ds')
    df.reset_index(inplace=True, drop=True)
    return df

def fourier_series(dates, period, order):
    # to days since epoch
    t = np.array((dates - pd.datetime(1970, 1, 1))
                 .dt.total_seconds()
                 .astype(np.float)) / (3600 * 24.0)
    return np.column_stack([
        fun((2.0 * (i + 1) * np.pi * t / period))
        for i in range(order)
        for fun in (np.sin, np.cos)
    ])
    
def seasonal_feature(dates, period, fourier_order, name):
    features = fourier_series(dates, period, fourier_order)
    columns = ['{}_delim_{}'.format(name, i + 1) for i in range(features.shape[1])]
    return pd.DataFrame(features, columns=columns)
    
def make_seasonality_features(history, yearly=True, weekly=True, holidays=None, prior_scale=5.0):
    start = history['ds'].min()
    end = history['ds'].max()
    dt = history['ds'].diff()
    min_dt = dt.iloc[dt.nonzero()[0]].min() # spacing

    seasonal_features = []
    prior_scales = []
    
    # Year seasonality
    # yearly_disable = end - start < pd.Timedelta(days=730)
    if yearly:
        features = seasonal_feature(history['ds'],
                                    period=365.25,
                                    fourier_order=10,
                                    name='yearly')
        seasonal_features.append(features)
        prior_scales.extend([prior_scale] * features.shape[1])
        
    
    # Weekly seasonality
    # weekly_disable = ((end - start < pd.Timedelta(weeks=2)) or
    #                  (min_dt >= pd.Timedelta(weeks=1)))
    if weekly:
        features = seasonal_feature(history['ds'],
                                    period=7,
                                    fourier_order=3,
                                    name='weekly')
        seasonal_features.append(features)
        prior_scales.extend([prior_scale] * features.shape[1])
        
    if holidays is not None:
        features, holiday_priors = make_holiday_features(history['ds'], holidays)
        seasonal_features.append(features)
        prior_scales.extend(holiday_priors)
                                
    if len(seasonal_features) == 0:
        seasonal_features.append(
            pd.DataFrame({'zeros': np.zeros(history.shape[0])})
        )
        prior_scales.append(1.0)
    return pd.concat(seasonal_features, axis=1), prior_scales

def make_holiday_features(dates, holidays, holidays_prior_scale=10.0):
    """Construct a dataframe of holiday features. Returns:
    holiday_features: pd.DataFrame with a column for each holiday.
    prior_scale_list: List of prior scales for each holiday column.
    """
    
    expanded_holidays = defaultdict(lambda: np.zeros(dates.shape[0]))
    prior_scales = {}
    # Makes an index so we can perform `get_loc` below.
    # Strip to just dates.
    row_index = pd.DatetimeIndex(dates.apply(lambda x: x.date()))

    for _ix, row in holidays.iterrows():
        dt = row.ds.date()
        # Gets holidy range
        try:
            lw = int(row.get('lower_window', 0))
        except ValueError:
            lw = 0
        try:
            uw = int(row.get('upper_window', 0))
        except ValueError:
            uw = 0
                
        ps = float(row.get('prior_scale', holidays_prior_scale))
        if np.isnan(ps):
            ps = float(holidays_prior_scale)
        prior_scales[row.holiday] = ps

        # Find dates in history range
        for offset in range(lw, uw + 1):
            occurrence = dt + timedelta(days=offset)
            try:
                loc = row_index.get_loc(occurrence)
            except KeyError:
                loc = None
            key = '{}_delim_{}{}'.format(row.holiday,
                                         '+' if offset >= 0 else '-',
                                         abs(offset))
            if loc is not None:
                expanded_holidays[key][loc] = 1.0 # binary features
            else:
                # Access key to generate value
                expanded_holidays[key]
        
    holiday_features = pd.DataFrame(expanded_holidays)
    prior_scale_list = [
        prior_scales[h.split('_delim_')[0]] for h in holiday_features.columns
    ]
    return holiday_features, prior_scale_list
    
def get_changepoints(history, n_changepoints=25):
    # Place potential changepoints evenly through first 80% of history
    # Return changepoints_t in t index
    
    hist_size = np.floor(history.shape[0] * 0.8)
    if n_changepoints == -1 or n_changepoints + 1 > hist_size:
        n_changepoints = hist_size - 1
            
    # set changepoints in df['ds'] timestamps
    if n_changepoints == 0:
        changepoints = [] # no changepoints
    else:
        cp_indexes = (
            np.linspace(0, hist_size, n_changepoints + 1)
            .round()
            .astype(np.int)
        )
        changepoints = history.iloc[cp_indexes]['ds'].tail(-1)
    
    # get changepoints_t in scaled t index
    if len(changepoints) > 0:
        start = history['ds'].min()
        t_scale = history['ds'].max() - start
        changepoints_t = np.sort(np.array((changepoints - start) / t_scale))
    else:
        changepoints_t = np.array([0])  # dummy changepoint
    return changepoints_t 

def get_changepoint_matrix(df, changepoints_t): 
    A = np.zeros((df.shape[0], len(changepoints_t)))
    for i, t_i in enumerate(changepoints_t):
        A[df['t'].values >= t_i, i] = 1
    return A