"""
@Author: Raúl Javierre
@Date: updated 13/11/2020

This module provides a Detector abstract class
"""

from abc import ABC, abstractmethod
import pandas as pd
import os

FIRST_WEEK_TRAINING_ELECTRICITY = 0
LAST_WEEK_TRAINING_ELECTRICITY = 60
FIRST_WEEK_TESTING_ELECTRICITY = 61
LAST_WEEK_TESTING_ELECTRICITY = 75

FIRST_WEEK_TRAINING_GAS = 0
LAST_WEEK_TRAINING_GAS = 60
FIRST_WEEK_TESTING_GAS = 61
LAST_WEEK_TESTING_GAS = 77


class Detector(ABC):

    @abstractmethod
    def build_model(self, training_dataset):
        pass

    @abstractmethod
    def predict(self, testing_dataset, model):
        pass

    def compute_outliers(self, testing_len, predictions, is_attack_behavior):
        n_tp, n_tn, n_fp, n_fn = 0, 0, 0, 0
        if is_attack_behavior:  # if attacks were detected, they were true positives
            n_tp = predictions
            n_fn = testing_len - predictions
        else:  # if attacks were detected, they were false positives
            n_fp = predictions
            n_tn = testing_len - predictions

        return n_tp, n_tn, n_fp, n_fn

    def print_metrics(self, meterID, detector, attack, time_model_creation, time_model_prediction, n_tp, n_tn, n_fp, n_fn):
        print("\n\nMeterID:\t\t\t", meterID)
        print("Detector:\t\t\t", detector)
        print("Attack:\t\t\t\t", attack)
        print("Exec. time of model creation:\t", time_model_creation, "seconds")
        print("Exec. time of model prediction:\t", time_model_prediction, "seconds")
        print("Accuracy:\t\t\t", (n_tp + n_tn) / (n_tp + n_tn + n_fp + n_fn))
        print("Number of true positives:\t", n_tp)
        print("Number of false negatives:\t", n_fn)
        print("Number of true negatives:\t", n_tn)
        print("Number of false positives:\t", n_fp)
        print("[", n_tp, n_fp, "]")
        print("[", n_fn, n_tn, "]\n\n")

    def metrics_to_csv(self, meterID, detector, attack, time_model_creation, time_model_prediction, n_tp, n_tn, n_fp, n_fn, type_of_dataset):
        resulting_csv_path = "./script_results/" + type_of_dataset + "_detector_comparer_results.csv"

        df = pd.DataFrame({'meterID': meterID,
                           'detector': detector,
                           'attack': attack,
                           'time_model_creation': time_model_creation,
                           'time_model_prediction': time_model_prediction,
                           'n_tp': n_tp,
                           'n_tn': n_tn,
                           'n_fp': n_fp,
                           'n_fn': n_fn,
                           'accuracy': (n_tp + n_tn) / (n_tp + n_tn + n_fp + n_fn)},
                          index=[0])

        df.to_csv(resulting_csv_path, mode='a', header=not os.path.exists(resulting_csv_path), index=False)

    def get_training_dataset(self, meterID, type_of_dataset):
        """
        Returns the training dataset for the meterID passed
        """
        if type_of_dataset == "electricity":
            FIRST_WEEK_TRAINING = FIRST_WEEK_TRAINING_ELECTRICITY
            LAST_WEEK_TRAINING = LAST_WEEK_TRAINING_ELECTRICITY
        else:
            FIRST_WEEK_TRAINING = FIRST_WEEK_TRAINING_GAS
            LAST_WEEK_TRAINING = LAST_WEEK_TRAINING_GAS

        return pd.read_csv("./script_results/" + type_of_dataset + "_training_data/" + str(meterID) + "_" + str(FIRST_WEEK_TRAINING) + "_" + str(LAST_WEEK_TRAINING) + ".csv")

    def get_testing_dataset(self, attack, meterID, type_of_dataset):
        """
        Returns the testing dataset for the meterID passed
        """
        if type_of_dataset == "electricity":
            FIRST_WEEK_TESTING = FIRST_WEEK_TESTING_ELECTRICITY
            LAST_WEEK_TESTING = LAST_WEEK_TESTING_ELECTRICITY
        else:
            FIRST_WEEK_TESTING = FIRST_WEEK_TESTING_GAS
            LAST_WEEK_TESTING = LAST_WEEK_TESTING_GAS

        if attack:
            testing_dataset = pd.read_csv("./script_results/" + type_of_dataset + "_testing_data/" + str(meterID) + "_" + attack + "_" + str(FIRST_WEEK_TESTING) + "_" + str(LAST_WEEK_TESTING) + ".csv")
        else:
            testing_dataset = pd.read_csv("./script_results/" + type_of_dataset + "_testing_data/" + str(meterID) + "_" + str(FIRST_WEEK_TESTING) + "_" + str(LAST_WEEK_TESTING) + ".csv")

        return testing_dataset
