"""
@Author: Raúl Javierre, Simona Bernardi
@Date: updated 13/11/2020

@Review: Simona Bernardi - 20/03/2021

Builds the KLD model based on the training set and the number of bins and
makes prediction based on the model, the significance level and the testing set.
"""

import numpy as np
from scipy.stats import entropy     # it is used to compute the KLD measure
from src.detectors.Detector import Detector
from time import time

nObs = 336  # number of readings in a week


class KLD(Detector):

    def build_model(self, training_dataset):
        t0 = time()
        kld = KLDdetector()
        kld.buildModel(training_dataset['Usage'].to_numpy())
        return kld, time() - t0

    def predict(self, testing_dataset, model):
        t0 = time()
        Ka = model.predictConsumption(testing_dataset['Usage'].to_numpy())
        n_false, obs = model.computeOutliers(Ka)
        return n_false, obs, time() - t0


class KLDmodel:

    def __init__(self, min_v, max_v):
        self.m = min_v
        self.M = max_v

    def setXdist(self, P_Xi):
        nWeeks = P_Xi.shape[0]
        nBins = P_Xi.shape[1]

        PT = np.transpose(P_Xi)
        # PX[j] = number of values of X that belong to bin_j
        self.PX = np.zeros(nBins)
        for j in range(nBins):
            self.PX[j] = np.sum(PT[j])

        #X distribution of relative frequencies of the overall training period
        self.PX = self.PX / nWeeks

    def setKdist(self, P_Xi):
        nWeeks = P_Xi.shape[0]
        self.K = np.zeros(nWeeks)

        for i in range(nWeeks):
            #KLD measure with log_2
            self.K[i] = entropy(P_Xi[i], self.PX, base=2)


class KLDdetector:

    def __init__(self, bins=30, signLevel=5):
        self.nbins = bins          #param: number of bins of the histograms
        self.signLevel = signLevel  #param: (100-alpha)-percentile

    def getXiDist(self, ds):

        # Matrix X:  dimension [nWeeks, nObs]
        nWeeks = int(ds.shape[0] / nObs)

        # Week 0 for the first week
        X = ds[0:nObs]
        # Rest of the weeks: one per row
        for i in range(nWeeks - 1):
            X = np.block([[X], [ds[nObs * (i + 1):nObs * (i + 2)]]])

        # P[i,j]= number of values of X[i] that belong to each bin j
        P = np.zeros([nWeeks, self.nbins])
        for i in range(nWeeks):
            P[i], b_edges = np.histogram(X[i], bins=self.nbins, range=(self.model.m, self.model.M))

        # Normalization: relative frequencies
        P = P / nObs

        return P

    def buildModel(self, train):

        # Min, max values of the training set
        m = np.min(train)
        M = np.max(train)

        # Create model and set m,M
        self.model = KLDmodel(m, M)

        # Compute Xi distributions (nTrainWeeks,nbins)
        P_Xi = self.getXiDist(train)

        # Set X distribution (nTrainWeeks, nbins)
        self.model.setXdist(P_Xi)

        # Set KLD distribution (nTrainWeeks)
        self.model.setKdist(P_Xi)

    def predictConsumption(self, test):

        # Compute Xi distributions (nTestWeeks,nbins) and bin edges
        P_Xa = self.getXiDist(test)

        nTestWeeks = P_Xa.shape[0]
        Ka = np.zeros(nTestWeeks)

        for i in range(nTestWeeks):
            #KLD measure with log_2
            Ka[i] = entropy(P_Xa[i], self.model.PX, base=2)

        return Ka

    def computeOutliers(self, Ka):

        # Compute the (100-alpha) percentile of the K distribution
        perc = np.percentile(self.model.K, 100 - self.signLevel)

        # Setting a counter vector to zero
        n_out = np.zeros(Ka.size)
        # If Ka > percentile of K distribution => Week_a is anomalous
        n_out = np.where(Ka > perc, n_out + 1, n_out)

        return np.sum(n_out), Ka.size

