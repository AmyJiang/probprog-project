import matplotlib.pyplot as plt


def plot_component(results, results_train, component, ts_data, ts_idx, cs):
    for j, r in enumerate(results):
        df = r["predictions"][ts_idx]
        plt.plot(ts_data[ts_idx]["future"]["t"], df[component], color=cs[j])
        if results_train is not None:
            df_train = results_train[j]["predictions"][ts_idx]
            plt.plot(ts_data[ts_idx]["history"]["t"],
                     df_train[component].as_matrix(), color=cs[j])
    plt.legend(loc=2, prop={'size': 24})
    plt.show()


def visualize_results(results, results_train, ts_data, ts_idx=0):
    ts = ts_data[ts_idx]
    cs = ['C1', 'C3', 'C5']

    # train and test split
    plt.axvline(x=1., color='k', linestyle='--')

    # True data
    plt.plot(ts["history"]["t"], ts["history"]["y_scaled"].as_matrix(),
             lw=1, color='C2', label='y_true')
    plt.plot(ts["future"]["t"], ts["future"]["y_scaled"].as_matrix(),
             lw=1, color='C2', label='_nolegend_')

    for j, r in enumerate(results):
        df = r["predictions"][ts_idx]
        plt.plot(ts["future"]["t"], df["y"].as_matrix(),
                 label="Model %d" % j, color=cs[j])
        if results_train is not None:
            df_train = results_train[j]["predictions"][ts_idx]
            plt.plot(ts["history"]["t"], df_train["y"],
                     color=cs[j], label='_nolegend_')
    plt.legend(loc=2, prop={'size': 24})
    # plt.legend(handles=[y_true_train,y_true_test])
    plt.show()

    plt.subplot(411)
    plot_component(results, results_train, "trend", ts_data, ts_idx, cs)
    plt.subplot(412)
    plot_component(results, results_train, "yearly", ts_data, ts_idx, cs)
    plt.subplot(413)
    plot_component(results, results_train, "weekly", ts_data, ts_idx, cs)
    plt.subplot(414)

    # Residual
    for j, r in enumerate(results):
        df = r["predictions"][ts_idx]
        plt.plot(ts["future"]["t"],
                 ts["future"]["y_scaled"].as_matrix() - df["y"].as_matrix(),
                 color=cs[j], label='residual')
        if results_train is not None:
            df_train = results_train[j]["predictions"][ts_idx]
            plt.plot(ts["history"]["t"],
                     ts["history"]["y_scaled"].as_matrix() -
                     df_train["y"].as_matrix(), color=cs[j])
    plt.legend(loc=2, prop={'size': 24})
    plt.show()
