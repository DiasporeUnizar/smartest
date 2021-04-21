"""
@Author: Raúl Javierre
@Date: updated 13/11/2020

@Review: Simona Bernardi - 19/03/2021

This module provides the functionality of a model that detects anomalous weeks.
It marks a week as anomalous if the mean of the usages of that week is less than the minimum
average of the weeks of the training set.
"""

from .Detector import Detector
from time import time

#SB The following commented constants seems not to be used in this module: remove?
#nObs = 336                  # number of readings in a week
#freq = 48                   # number of readings in a seasonal cycle: every half-an-hour in a day
MAX_USAGE = 9999999         # used to get the min average (comparing first with a big number)
FIRST_WEEK_TESTING = 61


class MinAverage(Detector):

    def build_model(self, training_dataset):
        """
        Returns the minimum of the averages of the consumption readings in each of the 60 weeks of training (dframe)
        and time to compute the method
        """
        t0 = time()
        min_avg = MAX_USAGE
        min_day = int(training_dataset.DT.min() / 100) * 100  # Getting the min_day DT (19500)
        max_day = int(training_dataset.DT.max() / 100) * 100  # Getting the max_day DT (62100)
        day = min_day
        week = 0
        while day < max_day:
            avg = training_dataset.query('DT >= @day & DT < (@day + 7*100)')['Usage'].mean()  # Getting the avg of the usages of the week
            if avg < min_avg:
                min_avg = avg

            day += 7 * 100
            week += 1

        return min_avg, time() - t0

    def predict(self, testing_dataset, model):
        """
        Returns a dictionary of 15 elements. Example: {61: True, 62: False, 63: False, ..., 75: True}
        If a key (week) has a value True means that the detector has marked that week as an anomalous week.
        Otherwise, means that the detector has marked that week as a normal week. Also returns the time
        to compute the method
        """
        t0 = time()
        min_day = int(testing_dataset.DT.min() / 100) * 100  # Getting the min_day DT (62200)
        max_day = int(testing_dataset.DT.max() / 100) * 100  # Getting the max_day DT (72600)
        week = FIRST_WEEK_TESTING
        weeks = {}

        day = min_day
        while day <= max_day:
            avg = testing_dataset.query('DT >= @day & DT < (@day + 7*100)')['Usage'].mean()  # Getting the avg of the usages of the week

            if avg < model:
                weeks[week] = True   # Anomalous week
            elif avg >= model:
                weeks[week] = False  # Normal week

            day += 7 * 100
            week += 1

        return weeks, len(weeks), time() - t0

    def compute_outliers(self, testing_len, predictions, is_attack_behavior):
        n_tp, n_tn, n_fp, n_fn = 0, 0, 0, 0
        if is_attack_behavior:  # if attacks were detected, they were true positives
            n_tp = sum(value for value in predictions.values())
            n_fn = sum(not value for value in predictions.values())
        else:  # if attacks were detected, they were false positives
            n_fp = sum(value for value in predictions.values())
            n_tn = sum(not value for value in predictions.values())

        return n_tp, n_tn, n_fp, n_fn
