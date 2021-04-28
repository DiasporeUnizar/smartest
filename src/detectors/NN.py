"""
@Author: Raúl Javierre
@Date: updated 05/04/2021

@Review: Simona Bernardi, 28/03/2021

This module provides the functionality of a deep learning detector.
"""

from tensorflow import keras
from time import time
from src.detectors.Detector import Detector
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os

FIRST_WEEK_TRAINING_ELECTRICITY = 0
LAST_WEEK_TRAINING_ELECTRICITY = 60
FIRST_WEEK_TESTING_ELECTRICITY = 61
LAST_WEEK_TESTING_ELECTRICITY = 75

FIRST_WEEK_TRAINING_GAS = 0
LAST_WEEK_TRAINING_GAS = 60
FIRST_WEEK_TESTING_GAS = 61
LAST_WEEK_TESTING_GAS = 77


class NN(Detector):

    def get_training_dataset(self, meterID, type_of_dataset):
        list_of_dataframes = list()
        d_path = "./script_results/" + type_of_dataset + "_training_data/"
        dir_files = os.listdir(d_path)
        dir_files.sort()
        for file in dir_files:
            if file.startswith(str(meterID)):
                training_scenario = pd.read_csv(d_path + file)
                list_of_dataframes.append(training_scenario)

        # list_of_dataframes[0] -> normal
        return list_of_dataframes

    def build_model(self, training_dataset):
        t0 = time()

        window_len = 96     # two days window

        model = keras.Sequential()
        model.add(keras.layers.Dense(units=10000, activation='relu'))
        model.add(keras.layers.Dense(units=len(training_dataset), activation='softmax'))  # one neuron for each class
        model.compile(loss='binary_crossentropy', optimizer='adam')

        x = []
        y = []

        # For each kind of behaviour
        for i in range(0, len(training_dataset)):   # i:0 -> normal

            # split into windows of two days
            list_df = [training_dataset[i][w:w + window_len] for w in range(0, training_dataset[i].shape[0], window_len)]

            # Generate training set for each window of two days
            for j in range(0, len(list_df)):
                x = x + [generate_input(list_df[j])]
                y = y + [generate_label(i, number_of_classes=len(training_dataset))]

        x = np.array(x)
        y = np.array(y)
        model.fit(x=x, y=y, epochs=10, verbose=0)

        return model, time() - t0

    def predict(self, testing_dataset, model):
        t0 = time()

        window_len = 96
        list_df = [testing_dataset[w:w + window_len] for w in range(0, testing_dataset.shape[0], window_len)]
        n_attacks = 0

        for j in range(0, len(list_df)):
            predicted = model.predict([generate_input(list_df[j])])[0]
            #print(predicted)    # If we uncomment it we can see that it can distinguish the different kind of attacks

            if np.argmax(predicted) >= 1:
                n_attacks += 1

        return n_attacks, len(list_df), time() - t0


def generate_input(df):
    return preprocessing.scale(pd.DataFrame(
        {
            'mean': df['Usage'].mean(),
            'std': df['Usage'].std(),
            'q1': df['Usage'].quantile(0.25),
            'q2': df['Usage'].quantile(0.5),
            'q3': df['Usage'].quantile(0.75),
            'iqr': df['Usage'].quantile(0.75) - df['Usage'].quantile(0.25),
            'last_minus_first': df.tail(1)['Usage'].values[0] - df.head(1)['Usage'].values[0]
        },
        index=[0]).values.flatten().tolist()).tolist()


def generate_label(kind, number_of_classes):
    labels = np.zeros(number_of_classes, dtype=int)
    labels[kind] = 1
    return labels
