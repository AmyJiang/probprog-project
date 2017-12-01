import edward as ed
import numpy as np
import tensorflow as tf
import pandas as pd
# import matplotlib.pyplot as plt

from utils import get_changepoint_matrix, get_changepoints, \
    make_seasonality_features
from prediction import predict_fixed, make_future_dataframe


def get_timeseries(path):
    df = pd.read_csv(path)
    timeseries = {}
    print("Loading timeseries:")
    for i, row in df.iterrows():
        ts = pd.DataFrame({"ds": row.index[1:], "views": row.values[1:]})
        timeseries[row.Page] = ts
    return timeseries


def split_train_test(df, sdate=pd.datetime(2017, 7, 10), edate=None):
    ''' Split timeseries dataframe into train (history) and test (future) '''

    if edate is None:
        edate = df['ds'].max()
    history = df[df['ds'] <= sdate].copy()
    future = df[df['ds'] <= edate]
    future = future[future['ds'] > sdate].copy()
    print("[+] History: %d, Future: %d" % (history.shape[0], future.shape[0]))

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
    future['t'] = (future['ds'] - t_start) / t_scale
    future['y_scaled'] = future['y'] / y_scale

#    plt.plot(history['ds'],history['y'])
#    plt.plot(future['ds'],future['y'])
#    plt.xticks(rotation=90)
#    plt.show()

    return (history, future, y_scale)


def extract_features(df, n_changepoints=25, changepoints_t=None,
                     holidays=None):
    seasonal_features, prior_scales = \
        make_seasonality_features(df, yearly=True, weekly=True,
                                  holidays=None)
    K = seasonal_features.shape[1]  # number of seasonal factors
    print("[+] %d Seasonal features" % K)
    print("\t[+]", list(seasonal_features.columns))
    if holidays is not None:
        holiday_ds = {}
        for feature in seasonal_features:
            if feature.split("_delim_")[0] in set(holidays['holiday']):
                holiday_ds[feature] = seasonal_features[
                    seasonal_features[feature] == 1.0].shape[0]
        print("\t[+] %d Holidays" % len(holiday_ds), holiday_ds)

    if changepoints_t is None:
        changepoints_t = get_changepoints(df, n_changepoints=n_changepoints)
    # number of change points
    print("[+] %d changepoints" % len(changepoints_t))
    print("\t", changepoints_t)
    return {
        't': df['t'].as_matrix(),  # time index
        'A': get_changepoint_matrix(df, changepoints_t),  # split indicator
        'X': seasonal_features,  # seasonal vectors
        'sigmas': prior_scales,  # scale on seasonality prior
        't_change': changepoints_t
    }


def evaluate(y_true, y_pred, prefix=""):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = np.mean(np.abs((y_true - y_pred)) /
                    (np.abs((y_true + y_pred)))) * 200
    mse = ((y_true - y_pred) ** 2).mean()
    return {prefix + "-MAPE": mape,
            prefix + "-SMAPE": smape,
            prefix + "-MSE": mse}


def get_posts(params, posts, i=None):
    if i is None:
        return {
            params[i][k]: v for i, ps in posts.items() for k, v in ps.items()
        }
    else:
        p = {params[i][k]: v for k, v in posts[i].items()}
        if -1 in posts:  # common parameters
            p.update({params[-1][k]: v for k, v in posts[-1].items()})
        return p


def pipeline(ts_data, model, train_data, test_data,
             ITR=5000, STEP_SIZE=5e-4, N_STEPS=2,
             CI=[20, 80]):
    print("[+] Building model")
    model.set_values(len(ts_data),  # number of timeseries
                     len(train_data["t_change"]),  # number of change points
                     train_data["X"].shape[1])  # number of seasonal factors
    model.build_model()
    model.build_posts(ITR, ts_data)

    print("[+] Running inference")
    with tf.name_scope(model.name):
        data_dict = {model.data[k]: v for k, v in train_data.items()}
        # add y true
        for i, ts in enumerate(ts_data):
            y_true = ts["history"]["y_scaled"].as_matrix()
            data_dict.update({
                model.params[i]["y"]: y_true
            })

        posts_dict = get_posts(model.params, model.posts)
        inference = ed.HMC(posts_dict, data=data_dict)
        inference.run(step_size=STEP_SIZE, n_steps=N_STEPS)

        print("[+] Making prediction")
        predictions = []
        metrics = []
        nburn = int(ITR / 2)
        stride = 5
        model.post_params = {}
        for i, ts in enumerate(ts_data):
            sess = ed.get_session()
            post_params = {
                "k": model.posts[i]["k"].params.eval()[nburn:ITR:stride],
                "m": model.posts[i]["m"].params.eval()[nburn:ITR:stride],
                "beta": model.posts[i]["beta"].params.eval()[nburn:ITR:stride],
                "delta": model.posts[i]["delta"].params.eval()[nburn:ITR:stride]
            }
            model.post_params[i] = post_params
            df = make_future_dataframe(ts["history"], ts["future"].shape[0])
            df = predict_fixed(df, post_params, test_data)

            predictions.append(df)
            y_true = ts["future"]["y_scaled"].as_matrix()
            metrics.append(evaluate(y_true, df["y"], prefix=model.name))

            # plt.plot(ts["future"]["ds"], ts["future"]["y_scaled"],
            #          lw=4, color='b')
            # plt.plot(ts["future"]["ds"], df["y"], color='r')
            # plt.xticks(rotation=90)
            # plt.show()
            # plt.plot(ts["future"]["ds"], y_true)
            # plt.plot(ts["future"]["ds"], y_pred_mean)
            # plt.plot(ts["future"]["ds"], y_pred_lower,lw=4)
            # plt.plot(ts["future"]["ds"], y_pred_upper,lw=4)
            # plt.fill_between(ts["future"].index, y_pred_lower,
            # y_pred_upper,color='k',alpha=.5)
            # plt.show()

        return predictions, metrics
