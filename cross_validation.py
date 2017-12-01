from datetime import timedelta
from pipeline import extract_features, split_train_test, pipeline
from pandas import pd


def cross_validation(ts_dfs, model, ITR=5000,
                     start=pd.datetime(2016, 7, 1),
                     end=None,
                     period=timedelta(days=90),
                     length=timedelta(days=60)):
    #  Cross-validate from #start to #end, cut off every #period of time,
    #  and predict #next into future.

    if end is None:
        end = ts_dfs[0]["ds"].max()

    predictions = []
    metrics = []
    while start < end:
        print("[+] Validation: train [--%s], test [%s-%s]" % (
            start, start, start + length))

        #  split history (train) / future (test)
        ts_data = []
        for df in ts_dfs:
            history, future, y_scale \
                = split_train_test(df, start, start + length)

            ts_data.append({
                "history": history, "future": future, "y_scale": y_scale
            })

        #  same feature matrix for all test series
        ts = ts_data[0]
        train_data = extract_features(ts["history"])
        test_data = extract_features(ts["future"],
                                     changepoints_t=train_data["t_change"])
        assert(all(train_data["X"].columns == test_data["X"].columns))
        assert(all(train_data["t_change"] == test_data["t_change"]))

        ps, ms = pipeline(ts_data, model, train_data, test_data, ITR=5000)
        for i, p in enumerate(ps):
            p["y_pred"] = p["y_scaled_pred"] * ts_data[i]["y_scale"]
        predictions.append(ps)
        metrics.append(ms)

        start = start + period

    return predictions, metrics
