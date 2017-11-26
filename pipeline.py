import os
import edward as ed
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import timedelta
from edward.models import Normal, Laplace, Empirical

from model import *


def get_timeseries(path):
    df = pd.read_csv(path)
    timeseries = {}
    print("Loading timeseries:")
    for _, row in df.iterrows():
        ts = pd.DataFrame({"ds": row.index[1:], "views": row.values[1:]})
        ts["y"] = ts["views"].astype(float)
        timeseries[row.Page] = ts
        print(row.Page)
    return timeseries

def evaluate(y_true, y_pred, prefix=""):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = np.mean(np.abs((y_true - y_pred)) / (np.abs((y_true + y_pred)))) * 100
    mse = ((y_true - y_pred) ** 2).mean()
    return {prefix+"_MAPE": mape,
            prefix+"_SMAPE": smape,
            prefix+"_MSE": mse}


def iteration(df):
    SDATE = pd.datetime(2017, 7, 10)
    df["y"] = np.log(df["y"])
    df  = setup_dataframe(df)

    # Split data into train and test
    history = df[df['ds'] <= SDATE].copy()
    future = df[df['ds'] > SDATE].copy()

    # Add a scaled t (time index) and y (#views)
    t_start = history['ds'].min()
    t_scale = history['ds'].max() - t_start
    if t_scale == 0:
        raise ValueError("Timeseries start == end")
    y_scale = history['y'].max()
    if y_scale == 0:
        y_scale = 1
    history['t'] = (history['ds'] - t_start) / t_scale
    history['y_scaled'] = history['y'] / y_scale


    holiday_en_us = ['2015-01-01', '2015-01-19', '2015-05-25', '2015-07-03',
                     '2015-09-07', '2015-11-26', '2015-11-27', '2015-12-25',
                     '2016-01-01', '2016-01-18', '2016-05-30', '2016-07-04',
                     '2016-09-05', '2016-11-11', '2016-11-24', '2016-12-26',
                     '2017-01-01', '2017-01-02', '2017-01-16', '2017-05-29',
                     '2017-07-04', '2017-09-04', '2017-11-10', '2017-11-23',
                     '2017-12-25', '2015-02-14', '2016-02-14', '2017-02-14']
    holidays = pd.DataFrame({
        'holiday': 'US public holiday',
        'ds': pd.to_datetime(holiday_en_us),
        'lower_window': 0,
        'upper_window': 0,
        'prior_scale': 10.0
    })
    holidays = None

    seasonal_features, prior_scales = make_seasonality_features(history, yearly=True, weekly=True,
                                                                holidays=holidays)

    K = seasonal_features.shape[1] # number of seasonal factors
    changepoints_t = get_changepoints(history, n_changepoints=25)
    S = len(changepoints_t) # number of change points
    changepoint_prior_scale = 0.05

    X_train = {
        't': history['t'].as_matrix(), # day
        'A': get_changepoint_matrix(history, changepoints_t), # split indicator
        'X': seasonal_features, # seasonal vectors
        'sigmas': prior_scales, # scale on seasonality prior
    }

    Y_train = history['y_scaled'].as_matrix()


    # Model
    t = tf.placeholder(tf.float32, shape=None, name="t")        # time index
    A = tf.placeholder(tf.float32, shape=(None, S), name="A")      # changepoint indicators
    t_change = tf.placeholder(tf.float32, shape=(S), name="t_change") # changepoints_t
    X = tf.placeholder(tf.float32, shape=(None, K), name="X")      # season vectors
    sigmas = tf.placeholder(tf.float32, shape=(K,), name="sigmas")  # scale on seasonality prior
    tau = tf.placeholder(tf.float32, shape=(), name="tau")      # scale on changepoints prior

    k = Normal(loc=tf.zeros(1), scale=5.0*tf.ones(1))           # initial slope
    m = Normal(loc=tf.zeros(1), scale=5.0*tf.ones(1))           # initial intercept
    sigma_obs = Normal(loc=tf.zeros(1), scale=0.5*tf.ones(1))   # noise

    delta = Laplace(loc=tf.zeros(S), scale=tau*tf.ones(S))      # changepoint rate adjustment
    gamma = tf.multiply(-t_change, delta, name="gamma")

    beta = Normal(loc=tf.zeros(K), scale=sigmas*tf.ones(K))     # seasonal

    trend_loc = (k + ed.dot(A, delta)) * t + (m + ed.dot(A, gamma))
    seas_loc = ed.dot(X, beta)
    y = Normal(loc = trend_loc + seas_loc, scale = sigma_obs)


    ## Inference

    ITR = 5000                       # Number of samples.

    # Init k, m
    def init_km(df):
        i0, i1 = df['ds'].idxmin(), df['ds'].idxmax()
        T = df['t'].iloc[i1] - df['t'].iloc[i0]
        k = (df['y_scaled'].iloc[i1] - df['y_scaled'].iloc[i0]) / T
        m = df['y_scaled'].iloc[i0] -  k * df['t'].iloc[i0]
        return (k, m)

    kinit, minit = init_km(history)
    qk = Empirical(params=tf.Variable(kinit * tf.ones([ITR, 1])))
    qm = Empirical(params=tf.Variable(minit * tf.ones([ITR, 1])))
    qsigma_obs = Empirical(params=tf.Variable(tf.ones([ITR, 1])))
    qbeta = Empirical(params=tf.Variable(tf.zeros([ITR, K])))
    qdelta = Empirical(params=tf.Variable(tf.zeros([ITR, S])))

    inference = ed.HMC({k: qk, m: qm, sigma_obs: qsigma_obs, beta: qbeta, delta:qdelta},
                       data={y: Y_train,
                             t: X_train['t'],
                             A: X_train['A'],
                             X: X_train['X'].as_matrix(),
                             sigmas: X_train['sigmas'],
                             t_change: changepoints_t,
                            tau: changepoint_prior_scale})
    inference.run(step_size=5e-4)


    ## Criticism

    sess = ed.get_session()
    kmean, kstddev = sess.run([qk.mean(), qk.stddev()])
    mmean, mstddev = sess.run([qm.mean(), qm.stddev()])
    sigma_obs_mean, sigma_obs_stddev = sess.run([qsigma_obs.mean(), qsigma_obs.stddev()])
    post = {
        "k_mean": kmean, "k_stddev": kstddev,
        "m_mean": kmean, "m_stddev": mstddev,
        "sigma_obs_mean": sigma_obs_mean, "sigma_obs_stddev": sigma_obs_stddev,
    }


    ## Evaluation / Prediction

    # Add scaled t and y
    future['t'] = (future['ds'] - t_start) / t_scale
    future['y_scaled'] = future['y'] / y_scale

    # Extract seasonality features
    future_seasonal, future_prior_scales = make_seasonality_features(future,
                                                                     yearly=True, weekly=True,
                                                                     holidays=holidays)
    assert(future_seasonal.shape[1] == K)
    assert(all(future_seasonal.columns == seasonal_features.columns))

    X_test = {
        't': future['t'].as_matrix(), # day
        'A': get_changepoint_matrix(future, changepoints_t), # split indicator
        'X': future_seasonal, # seasonal vectors
        'sigmas': future_prior_scales, # scale on seasonality prior
    }

    Y_test = future['y_scaled'].as_matrix()

    sess = ed.get_session()

    y_post = ed.copy(y, {k: qk, m: qm, sigma_obs: qsigma_obs, beta: qbeta, delta:qdelta})
    y_pred = np.array([sess.run([y_post],
                       feed_dict={t: X_test['t'],
                                  A: X_test['A'],
                                  X: X_test['X'].as_matrix(),
                                  sigmas: X_test['sigmas'],
                                  t_change: changepoints_t,
                                  tau: changepoint_prior_scale}) for _ in range(500)]).mean(axis=0)[0]

    metrics = {}
    metrics.update(evaluate(future['y_scaled'], y_pred, prefix="test_log"))
    metrics.update(evaluate(future['views'], np.exp(y_pred * y_scale), prefix="test"))

    # Training error
    y_train_pred = np.array([sess.run([y_post],
                      feed_dict={t: X_train['t'],
                                 A: X_train['A'],
                                 X: X_train['X'].as_matrix(),
                                 sigmas: X_train['sigmas'],
                                 t_change: changepoints_t,
                                 tau: changepoint_prior_scale}) for _ in range(500)]).mean(axis=0)[0]

    metrics.update(evaluate(history['y_scaled'], y_train_pred, prefix="train_log"))
    metrics.update(evaluate(history['views'], np.exp(y_train_pred * y_scale), prefix="train"))


    # Benchmark: median model
    def get_median(history_y, size, p=50):
        last_p = np.array(history_y.values[-p:], dtype=float)
        return np.ones(size) * np.nan_to_num(np.nanmedian(last_p))

    y_pred_median = get_median(history["views"], size=future.shape[0])
    metrics.update(evaluate(future['views'], y_pred_median, prefix="test_median"))
    return post, metrics



if __name__ == "__main__":
    FPATH = "./data/nfl_teams.csv"
#    OPATH = os.path.join("./results", os.path.basename(FPATH))
    OPATH = "./results/nfl_teams.csv"

    timeseries = get_timeseries(FPATH)
    timeseries_dfs = []
    for k, ts in timeseries.items():
        print(k)
        post, metrics = iteration(ts)

        data = {"Page": k}
        data.update(post)
        data.update(metrics)

        timeseries_dfs.append(pd.DataFrame.from_dict(data))

    timeseries_df = pd.concat(timeseries_dfs)
    timeseries_df.to_csv(OPATH, header=True, index=False)

    print("Average test SMAPE: %f" % (np.mean(timeseries_df["test_SMAPE"])))
    print("Average train SMAPE: %f" % (np.mean(timeseries_df["train_SMAPE"])))

    print("Average test SMAPE (Median): %f" % (np.mean(timeseries_df["test_median_SMAPE"])))







