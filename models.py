import tensorflow as tf
import pandas as pd
import edward as ed
from edward.models import Normal, Laplace, Empirical
import numpy as np

def build_data(name, N_TS, S, K):
    with tf.name_scope(name):
        t = tf.placeholder(tf.float32, shape=None, name="t")              # time index
        A = tf.placeholder(tf.float32, shape=(None, S), name="A")         # changepoint indicators
        t_change = tf.placeholder(tf.float32, shape=(S), name="t_change") # changepoints_t
        X = tf.placeholder(tf.float32, shape=(None, K), name="X")         # season vectors
        sigmas = tf.placeholder(tf.float32, shape=(K,), name="sigmas")    # scale on seasonality prior
        return {
            "t": t, "A": A, "t_change": t_change, "X": X, "sigmas": sigmas
        }


def init_km(df):
    i0, i1 = df['ds'].idxmin(), df['ds'].idxmax()
    T = df['t'].iloc[i1] - df['t'].iloc[i0]
    k = (df['y_scaled'].iloc[i1] - df['y_scaled'].iloc[i0]) / T
    m = df['y_scaled'].iloc[i0] -  k * df['t'].iloc[i0]
    return (k, m)

class Model1(object):
    def set_values(self, N_TS, S, K):
        self.N_TS = N_TS
        self.S = S
        self.K = K
        self.name = "model1"

    def build_model(self):
        with tf.name_scope(self.name):
            self.data = build_data(self.name, self.N_TS, self.S, self.K)
            self.params = {}
            for i in range(self.N_TS):
                k = Normal(loc=tf.zeros(1), scale=1.0*tf.ones(1))     # initial slope
                m = Normal(loc=tf.zeros(1), scale=1.0*tf.ones(1))     # initial intercept
                sigma_obs = Normal(loc=tf.zeros(1), scale=0.5*tf.ones(1))   # noise
                tau = Normal(loc=tf.ones(1) * 0.05, scale=1.*tf.ones(1))    # changepoint prior scale
                delta = Laplace(loc=tf.zeros(self.S), scale=tau*tf.ones(self.S))    # changepoint rate adjustment
                gamma = tf.multiply(-self.data["t_change"], delta)
                beta = Normal(loc=tf.zeros(self.K),
                              scale=self.data["sigmas"]*tf.ones(self.K))      # seasonal
                trend_loc = (k + ed.dot(self.data["A"], delta)) * self.data["t"] + \
                            (m + ed.dot(self.data["A"], gamma))
                seas_loc = ed.dot(self.data["X"], beta)
                y = Normal(loc = trend_loc + seas_loc, scale = sigma_obs)
                self.params[i] = {
                    "k": k, "m": m,
                    "sigma_obs": sigma_obs,
                    "tau": tau, "delta": delta,
                    "beta": beta,
                    "trend_loc": trend_loc, "seas_loc": seas_loc,
                    "y": y
                }

    def build_posts(self, ITR, ts_data):
        assert(self.N_TS == len(ts_data))
        self.ITR = ITR
        self.posts = {}
        with tf.name_scope(self.name):
            for i in range(self.N_TS):
                kinit, minit = init_km(ts_data[i]["history"])
                print("[+] Initial slope / intercept: %f, %f" % (kinit, minit))
                qbeta = Empirical(params=tf.Variable(tf.zeros([ITR, self.K])))
                qk = Empirical(params=tf.Variable(kinit * tf.ones([ITR, 1])))
                qm = Empirical(params=tf.Variable(minit * tf.ones([ITR, 1])))
                qsigma_obs = Empirical(params=tf.Variable(tf.ones([ITR, 1])))
                qdelta = Empirical(params=tf.Variable(tf.zeros([ITR, self.S])))
                qtau = Empirical(params=tf.Variable(0.05 * tf.ones([ITR, 1])))
                self.posts[i] = {
                    "k": qk, "m": qm,
                    "sigma_obs": qsigma_obs,
                    "delta": qdelta,
                    "tau": qtau,
                    "beta": qbeta
                }


class Model3(object):
    def set_values(self, N_TS, S, K):
        self.N_TS = N_TS
        self.S = S
        self.K = K
        self.name = "model3"

    def build_model(self):
        with tf.name_scope(self.name):
            self.data = build_data(self.name, self.N_TS, self.S, self.K)
            self.params = {}

            # Common prior
            gbeta = Normal(loc=tf.zeros(self.K), scale=self.data["sigmas"]*tf.ones(self.K))
            self.params[-1] = {"gbeta": gbeta}

            for i in range(self.N_TS):
                k = Normal(loc=tf.zeros(1), scale=1.0*tf.ones(1))     # initial slope
                m = Normal(loc=tf.zeros(1), scale=1.0*tf.ones(1))     # initial intercept
                sigma_obs = Normal(loc=tf.zeros(1), scale=0.5*tf.ones(1))   # noise
                tau = Normal(loc=tf.ones(1) * 0.05, scale=1.*tf.ones(1))    # changepoint prior scale
                delta = Laplace(loc=tf.zeros(self.S), scale=tau*tf.ones(self.S))    # changepoint rate adjustment
                gamma = tf.multiply(-self.data["t_change"], delta)
                trend_loc = (k + ed.dot(self.data["A"], delta)) * self.data["t"] + \
                            (m + ed.dot(self.data["A"], gamma))
                beta = Normal(loc=tf.zeros(self.K), scale=self.data["sigmas"]*tf.ones(self.K))
                seas_loc = ed.dot(self.data["X"], beta)
                y = Normal(loc = trend_loc + seas_loc, scale = sigma_obs)
                self.params[i] = {
                    "k": k, "m": m,  "sigma_obs": sigma_obs,
                    "tau": tau, "delta": delta, "beta": beta,
                    "trend_loc": trend_loc, "seas_loc": seas_loc, "y": y
                }

    def build_posts(self, ITR, ts_data):
        assert(self.N_TS == len(ts_data))
        self.ITR = ITR
        self.posts = {}
        with tf.name_scope(self.name):
            qgbeta = Empirical(params=tf.Variable(tf.zeros([ITR, self.K])))
            self.posts[-1] = {"gbeta": qgbeta}

            for i in range(self.N_TS):
                kinit, minit = init_km(ts_data[i]["history"])
                print("[+] Initial slope / intercept: %f, %f" % (kinit, minit))
                qk = Empirical(params=tf.Variable(kinit * tf.ones([ITR, 1])))
                qm = Empirical(params=tf.Variable(minit * tf.ones([ITR, 1])))
                qsigma_obs = Empirical(params=tf.Variable(tf.ones([ITR, 1])))
                qdelta = Empirical(params=tf.Variable(tf.zeros([ITR, self.S])))
                qtau = Empirical(params=tf.Variable(0.05 * tf.ones([ITR, 1])))
                qbeta = Empirical(params=tf.Variable(tf.zeros([ITR, self.K])))
                self.posts[i] = {
                    "k": qk, "m": qm,
                    "sigma_obs": qsigma_obs,
                    "delta": qdelta, "beta": qbeta,
                    "tau": qtau,
                }

class Model4(object):
    def set_values(self, N_TS, S, K):
        self.N_TS = N_TS
        self.S = S
        self.K = K
        self.name = "model4"

    def build_model(self):
        with tf.name_scope(self.name):
            self.data = build_data(self.name, self.N_TS, self.S, self.K)
            self.params = {}

            # Common prior
            gk = Normal(loc=tf.zeros(1), scale=5.0*tf.ones(1))     # initial slope
            self.params[-1] = {"gk": gk}


            for i in range(self.N_TS):
                k = Normal(loc=gk, scale=1.0*tf.ones(1))     # initial slope
                m = Normal(loc=tf.zeros(1), scale=1.0*tf.ones(1))     # initial intercept
                sigma_obs = Normal(loc=tf.zeros(1), scale=0.5*tf.ones(1))   # noise
                tau = Normal(loc=tf.ones(1) * 0.05, scale=1.*tf.ones(1))    # changepoint prior scale
                delta = Laplace(loc=tf.zeros(self.S), scale=tau*tf.ones(self.S))    # changepoint rate adjustment
                gamma = tf.multiply(-self.data["t_change"], delta)
                beta = Normal(loc=tf.zeros(self.K),
                              scale=self.data["sigmas"]*tf.ones(self.K))      # seasonal
                trend_loc = (k + ed.dot(self.data["A"], delta)) * self.data["t"] + \
                            (m + ed.dot(self.data["A"], gamma))
                seas_loc = ed.dot(self.data["X"], beta)
                y = Normal(loc = trend_loc + seas_loc, scale = sigma_obs)
                self.params[i] = {
                    "k": k, "m": m,
                    "sigma_obs": sigma_obs,
                    "tau": tau, "delta": delta,
                    "beta": beta,
                    "trend_loc": trend_loc, "seas_loc": seas_loc,
                    "y": y
                }

    def build_posts(self, ITR, ts_data):
        assert(self.N_TS == len(ts_data))
        self.ITR = ITR
        self.posts = {}
        with tf.name_scope(self.name):
            qgk = Empirical(params=tf.Variable(tf.zeros([ITR, 1])))
            self.posts[-1] = {"gk": qgk}
            for i in range(self.N_TS):
                kinit, minit = init_km(ts_data[i]["history"])
                print("[+] Initial slope / intercept: %f, %f" % (kinit, minit))
                qbeta = Empirical(params=tf.Variable(tf.zeros([ITR, self.K])))
                qk = Empirical(params=tf.Variable(kinit * tf.ones([ITR, 1])))
                qm = Empirical(params=tf.Variable(minit * tf.ones([ITR, 1])))
                qsigma_obs = Empirical(params=tf.Variable(tf.ones([ITR, 1])))
                qdelta = Empirical(params=tf.Variable(tf.zeros([ITR, self.S])))
                qtau = Empirical(params=tf.Variable(0.05 * tf.ones([ITR, 1])))
                self.posts[i] = {
                    "k": qk, "m": qm,
                    "sigma_obs": qsigma_obs,
                    "delta": qdelta,
                    "tau": qtau,
                    "beta": qbeta
                }


