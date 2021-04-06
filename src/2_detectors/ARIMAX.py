"""
@Author: Simona Bernardi, Raúl Javierre
@Date: updated 13/11/2020

@Review: Simona Bernardi - 06/04/2021

This module provides the functionality of an ARIMAX detector with exogenous variables (Fourier)
"""


import numpy as np
import pmdarima as pm
from pmdarima.arima.utils import ndiffs  # , nsdiffs
from pmdarima.preprocessing import FourierFeaturizer
from src.detectors.Detector import Detector
from time import time

freq = 48  # number of readings in a seasonal cycle: every half-an-hour in a day


class ARIMAX(Detector):

    def build_model(self, training_dataset):
        t0 = time()
        return buildARIMAXModel(training_dataset.Usage), time() - t0

    def predict(self, testing_dataset, model):
        t0 = time()
        pred, conf = predict_consumption_arimax(testing_dataset.Usage, model)
        n_false, obs = compute_outliers_arimax(pred, conf, testing_dataset.Usage)
        return n_false, obs, time() - t0


def buildARIMAXModel(train):
    # Precomputing d
    # Estimate the number of differences using an ADF tests
    # Other possibilities: kpss, pp
    n_adf = ndiffs(train, test='adf')

    # Use of exogenous variables to capture seasonality:
    # the use of seasonality feature is too much time-expensive (not scalable)
    # The parameter values corresponds to the seasonality of the time series: in
    # our case every day (half-an-hour observations: frequency of observations = 2x24)

    trans = FourierFeaturizer(freq)
    train, exog = trans.fit_transform(train)

    # Looking for the best params with auto_arima -- time expensive!!
    # model(p,d,q)
    model = pm.auto_arima(train,  # the time series to be fitted
                          exogenous=exog,  # exogenous variables (default None)
                          # start_p=5, #initial p value (default 2)
                          d=n_adf,  # estimated a priori (default None)
                          # start_q=0, #initial q value (default 2)
                          # max_p=5, #max p value (default 5)
                          # max_d=2, #max d value (default 2)
                          # max_q=5, #max q value (default 5)
                          # start_P=1, #initial P value (default 1)
                          # D=n_ch, #(default None)
                          # start_Q=1, #initial Q value (default 1)
                          # max_P = 2, #max P value (default 2)
                          # max_D = 1, #max D value (default 1)
                          # max_Q = 2, #max Q value (default 2)
                          # max_order=5, #p+q+P+Q if stepwise=False (default 5)
                          # m=48, #period for seasonal differencing (default 1)
                          seasonal=False,  # whether to fit a seasonal ARIMA (default True)
                          # stationary=False, #the time-series is stationary (default False)
                          information_criterion='bic',  # default 'aic'
                          # alpha=0.05, #level of the tests for testing significance (default 0.05)
                          # tests='adf', #type of tests to determine d (d=None); default 'kpss'
                          # seasonal_test ='ch', #type of tests to determine D (D=None); default 'ocsb'
                          stepwise=True,  # follows the Hyndman and Khandakar approach (default True)
                          # n_jobs=1, #number of models to fit in parallel in case stepwise=False (default 1)
                          # start_params=None, #starting params for ARMA(p,q); default None
                          # method='nm', #optimization method (default 'lbfgs')
                          # trend=None, #trend parameter (default None)
                          maxiter=20,  # max no. of function evaluations (default 50)
                          suppress_warnings=True,
                          error_action='warn',  # error-handling behavior in case of unable to fit (default 'warn')
                          trace=False  # print the status of the fit (default False)
                          # random=False, #random search over a hyper-parameter space (default False)
                          # random_state=None, #the PRNG when random=True
                          # n_fits=10, #number of ARIMA models to be fit, when random=True
                          # return_valid_fits=False, #return all valid ARIMA fits if True (default False)
                          # out_of_sample_size=0, #portion of the data to be hold out and use as validation (default 0)
                          # scoring='mse', #if out_of_sample_size>0, metric to use for validation purposes (default 'mse')
                          # scoring_args=None, #dictionary to be passed to the scoring metric (default None)
                          # with_intercept='auto', #whether to include an intercept term (default 'auto')
                          # sarimax_kwargs=None #arguments to pase to the ARIMA constructor (default None)
                          )
    return model


def predict_consumption_arimax(test, model):
    # The exogenous variables need to be used also in the forecast
    trans = FourierFeaturizer(freq)
    test, exog = trans.fit_transform(test)

    pred, conf = model.predict(n_periods=exog.shape[0], exogenous=exog, return_conf_int=True, alpha=0.05)

    return pred, conf


def compute_outliers_arimax(pred, conf, actual):
    # Pre: pred.size = actual.size = conf.shape[0]
    # Setting a counter vector to zero
    n_out = np.zeros(pred.size)
    # Count if actual is outside of the confidence interval
    n_out = np.where(((actual < conf[:, 0]) | (actual > conf[:, 1]))
                     , n_out + 1, n_out)

    return np.sum(n_out), pred.size