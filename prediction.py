import pandas as pd
import numpy as np

def piecewise_linear(t, deltas, k, m, changepoint_ts):
    # Intercept changes
    gammas = -changepoint_ts * deltas
    # Get cumulative slope and intercept at each t
    k_t = k * np.ones_like(t)
    m_t = m * np.ones_like(t)
    for s, t_s in enumerate(changepoint_ts):
        indx = t >= t_s
        k_t[indx] += deltas[s]
        m_t[indx] += gammas[s]
    return k_t * t + m_t

def add_group_component(components, name, group):
    new_comp = components[components['component'].isin(set(group))].copy()
    new_comp['component'] = name
    components = components.append(new_comp)
    return components
    
def predict_seasonal_components(df, params, data, interval_width=0.8):
    seasonal_features = data["X"]
    lower_p = 100 * (1.0 - interval_width) / 2 
    upper_p = 100 * (1.0 + interval_width) / 2
    
    components = pd.DataFrame({
        'col': np.arange(seasonal_features.shape[1]),
        'component': [x.split('_delim_')[0] for x in seasonal_features.columns],
    })
    
    # Add a total for seasonal 
    components = components.append(pd.DataFrame({
        'col': np.arange(seasonal_features.shape[1]),
        'component': 'seasonal',
    }))
    
    X = seasonal_features.as_matrix()
    data = {}
    for component, features in components.groupby('component'):
        cols = features.col.tolist()
        comp_beta = params['beta'][:, cols]
        comp_features = X[:, cols]
        comp = (np.matmul(comp_features, comp_beta.transpose()))
        data[component] = np.nanmean(comp, axis=1)
        data[component + '_lower'] = np.nanpercentile(comp, lower_p, axis=1)
        data[component + '_upper'] = np.nanpercentile(comp, upper_p, axis=1)
    return pd.DataFrame(data)
    

def predict_fixed(df, params, data, nsample=500):
    # get posterior predictive mean
    k = np.nanmean(params['k'])
    m = np.nanmean(params['m'])
    deltas = np.nanmean(params['delta'], axis=0)
    
    # predict trend
    df['trend'] = piecewise_linear(np.array(df['t']), deltas, k, m, data["t_change"])
    # predict seasonal components 
    seasonal_components = predict_seasonal_components(df, params, data)
    #TODO: intervals = predict_uncertainty(df)

    df = pd.concat([df, seasonal_components], axis=1)
    df['y'] = df['trend'] + df['seasonal']
    return df


def make_future_dataframe(history, periods, freq='D'):
    # create future time series for forecasting
    t_start = history['ds'].min()
    last_d = history['ds'].max()
    t_scale = last_d - t_start
    dates = pd.date_range(start=last_d, periods=periods + 1, freq=freq)
    dates = dates[dates > last_d]
    dates = dates[:periods]
    
    future = pd.DataFrame({"ds": dates})
    future['ds'] = pd.to_datetime(future['ds'])
    future.reset_index(inplace=True, drop=True)
    future['t'] = (future['ds'] - t_start) / t_scale
    return future