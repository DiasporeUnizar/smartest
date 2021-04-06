"""
@Author: Simona Bernardi, Raúl Javierre
@Date: updated 05/04/2021

@Review: Simona Bernardi - 19/03/2021

¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡This detector is designed ONLY for Energy dataset (ISSDA-CER)!!!!!!!!!!!!!!!!!!!!!!!

This module encapsulates:
- PCA functionality for a given dataset
- DBSCAN clustering algorithm
"""

import os
import pandas as pd
import numpy as np
from src import meterIDsEnergy
from robustbase import Sn
from sklearn.cluster import DBSCAN
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.decomposition import PCA
from src.detectors.Detector import Detector
from time import time

nObs = 336  # number of readings

FIRST_WEEK_TRAINING_ENERGY = 0
LAST_WEEK_TRAINING_ENERGY = 60
FIRST_WEEK_TESTING_ENERGY = 61
LAST_WEEK_TESTING_ENERGY = 75

FIRST_WEEK_TRAINING_GAS = 0
LAST_WEEK_TRAINING_GAS = 60
FIRST_WEEK_TESTING_GAS = 61
LAST_WEEK_TESTING_GAS = 77


class PCADBSCAN(Detector):

    def build_model(self, training_dataset):
        t0 = time()
        mg = dataAnalyzerPCA()
        mg.set_dataframe(training_dataset)
        mg.updateMeterID("./ISSDA-CER/Energy/documentation/customerClassification.csv")

        A = mg.df.pivot(index='DT', columns='ID', values='Usage').to_numpy()
        B = getMatrixB(A)

        # PCA for matrix B
        Y_B, P_B = mg.PCA(np.transpose(B), 2)

        # Transform the B_pca[nCust X nweek, 2] matrix into a dataframe with
        # the week and meterIdx columns
        Y_B = mg.transformToDataFrame(Y_B)

        P_B = pd.DataFrame(P_B)

        return dataAnalyzerDBSCAN(Y_B, P_B), time() - t0

    def predict(self, testing_dataset, model):
        t0 = time()

        model.set_testing_dataset(testing_dataset)
        model.set_Y_BmeterID()

        # Convert the selected columns to numpy array
        model.Y_BmeterID = model.Y_BmeterID[['0', '1']].to_numpy()

        # Compute the pairwise Euclidean distance
        paird = pairwise_distances(model.Y_BmeterID, metric='euclidean')

        # The radius is set calculating the Sn measure of Rousseeuw and Croux flatten the distance matrix paird (ravel)
        radiusSn = Sn(paird.ravel())

        # minPts: The (min) number of points in a neighborhood for a point to be considered as a core point.
        # The number of points is the majority of the entire set of points
        minPts = int(model.Y_BmeterID.shape[0] / 2) + 1

        # ===========================================================
        # Build DBSCAN model
        # ===========================================================
        db = DBSCAN(eps=radiusSn, min_samples=minPts).fit(model.Y_BmeterID)

        # Compute outliers using the tests dataset
        try:
            n_false, nObservations = model.computeOutliers(db, radiusSn)
        except ValueError:  # Error: 60 noisy points fitting -> 0 clusters
            return -1, -1, -1,

        return n_false, nObservations, time() - t0

    def compute_outliers(self, testing_len, predictions, is_attack_behavior):
        if predictions == -1:
            return -1, -1, -1, -1

        n_tp, n_tn, n_fp, n_fn = 0, 0, 0, 0
        if is_attack_behavior:  # if attacks were detected, they were true positives
            n_tp = predictions
            n_fn = testing_len - predictions
        else:  # if attacks were detected, they were false positives
            n_fp = predictions
            n_tn = testing_len - predictions

        return n_tp, n_tn, n_fp, n_fn

    def get_training_dataset(self, meterID, type_of_dataset):
        if type_of_dataset == "energy":
            FIRST_WEEK_TRAINING = FIRST_WEEK_TRAINING_ENERGY
            LAST_WEEK_TRAINING = LAST_WEEK_TRAINING_ENERGY
        else:   # gas -> MIGRATED BUT NOT APPLICABLE
            FIRST_WEEK_TRAINING = FIRST_WEEK_TESTING_GAS
            LAST_WEEK_TRAINING = LAST_WEEK_TRAINING_GAS

        if os.path.isfile("./script_results/pca_dbscan_training.csv"):
            training_dataset = pd.read_csv("./script_results/pca_dbscan_training.csv")
        else:  # Heavy! Generating file once and next invocations going to "if branch"
            mg = dataAnalyzerPCA()
            files = mg.getDataSetFiles("./ISSDA-CER/Energy/data/data_all_filtered")
            training_dataset = mg.loadFilteredData(files, FIRST_WEEK_TRAINING, LAST_WEEK_TRAINING)
            training_dataset.to_csv("./script_results/pca_dbscan_training.csv", index=False)

        return training_dataset

    def get_testing_dataset(self, attack, meterID, type_of_dataset):
        testing_dataset = super(PCADBSCAN, self).get_testing_dataset(attack, meterID, type_of_dataset)
        return insertWeekColumnToTestingDataframe(testing_dataset)


class dataAnalyzerPCA:

    def __init__(self):
        # dataframe with all the readings: columns ['ID','DT','Usage']
        self.df = pd.DataFrame()
        # list of selected (complete) meterIDs together with their classification
        self.meterID = pd.DataFrame()

    def set_dataframe(self, df):
        self.df = df

    def getDataSetFiles(self, dpath):
        dirFiles = os.listdir(dpath)
        myfiles = []
        for files in dirFiles:
                pathfile = (dpath + '/') + files
                myfiles.append(pathfile)
                myfiles.sort(key=lambda x: int(x.split()[-1]))

        return myfiles

    def loadFilteredData(self, files, firstWeek, lastWeek):
        df = pd.DataFrame()
        i = 1
        for file in files[firstWeek:lastWeek]:  # not +1 because there's not week 36
            #print("Reading File: ", file)
            dset = pd.read_csv(file)  # load as a pd.dataFrame
            # insert a new column for weeks i-1
            dset.insert(1, "Week", (i - 1) * np.ones(dset.shape[0], dtype='int'), True)
            df = pd.concat([df, dset])
            # Get only selected meterIDs (500 from energy_customer_analysis.py)
            df = df[df['ID'].isin(meterIDsEnergy)]
            i += 1

        return df

    def updateMeterID(self, clfile):
        ID = self.df.groupby('ID').count().index.to_numpy()
        ID = pd.DataFrame(ID, columns=['ID'])
        clset = pd.read_csv(clfile, sep=';', index_col='ID')
        self.meterID = pd.merge(ID, clset, on='ID', how='inner')

        return 0

    ####################################################################
    # Methods for the PCA analysis
    ####################################################################
    def PCA(self, mat, comp):

        # keep the first "comp" principal components of the data
        pca = PCA(n_components=comp, svd_solver='full')
        # fit PCA model to the dataset

        pca.fit(mat)

        # transform data onto the first "comp" principal components
        mat_pca = pca.transform(mat)

        return mat_pca, pca.components_

    def transformToDataFrame(self, mat):
        ncust = self.meterID.shape[0]
        rowid = np.array([])  # an array with the meterID indexes 0-335
        ID = self.meterID['ID'].to_numpy()
        nWeeks = int(mat.shape[0] / ncust)
        week = np.array([])
        for i in range(nWeeks):
            week = np.concatenate([week, i * np.ones(ncust)])
            rowid = np.concatenate([rowid, ID])

        week = np.transpose(np.array([week]))
        rowid = np.transpose(np.array([rowid]))

        mat = np.block([mat, week, rowid])

        # B_pca dataframe
        mat_pca = pd.DataFrame(mat, columns=['0', '1', 'week', 'meterID'])

        return mat_pca


class dataAnalyzerDBSCAN:

    def __init__(self, Y_B, P_B):

        # Columns of Y_B: [0,1,week,meterID]
        self.Y_B = Y_B

        # Columns of P_B: nObs=336  (each half-an-hour)
        self.P_B = P_B.to_numpy()

    def set_testing_dataset(self, TS):
        self.TS = TS

    def set_Y_BmeterID(self):
        meterID = self.TS['ID'].values[0]
        self.Y_BmeterID = self.Y_B[self.Y_B.meterID == meterID]

    '''
    SB: The following method is not used in the experiments.
    def plotPoints(self, db, radiusSn):
        # Pre: there is only one cluster (0)
        # Core point mask
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        # Number of clusters in labels, ignoring noise if present.
        n_noise_ = list(db.labels_).count(-1)

        # Points belonging to the cluster 0
        class_member_mask = (db.labels_ == 0)

        # Core points of the cluster 0
        coreP = self.Y_BmeterID[core_samples_mask]

        # Plot eps neighborhoods of core points
        npoints = 2  # dimension of a point in the plot
        plt.plot(coreP[:, 0], coreP[:, 1], 'o', color='yellow', alpha=0.4,
                 markersize=5.0 * npoints * radiusSn)

        # Plot core points of the cluster 0
        plt.plot(coreP[:, 0], coreP[:, 1], 'o', color='green', markersize=npoints)

        # Fringe points of the cluster 0
        fringeP = self.Y_BmeterID[class_member_mask & ~core_samples_mask]
        # Plot fringe points of the cluster 0
        plt.plot(fringeP[:, 0], fringeP[:, 1], 'v', color='blue', markersize=npoints)

        # Noise points (outside the cluster 0)
        noise_member_mask = (db.labels_ == -1)
        noiseP = self.Y_BmeterID[noise_member_mask]
        # Plot noise points = anomalous weeks
        plt.plot(noiseP[:, 0], noiseP[:, 1], 'x', color='red', markersize=npoints)

        plt.title("Core, fringe and noise points of tested meterID")
        plt.xlabel("Principal component 1")
        plt.ylabel("Principal component 2")
        plt.show()
    '''

    def computeOutliers(self, db, radius):

        # Calculate matrices A and B for the testSet
        A = self.TS.pivot(index='DT', columns='ID', values='Usage').to_numpy()
        B = getMatrixB(A)

        # Get the reduced matrix for the testSet
        Y_Btest = np.matmul(self.P_B, B)

        # Getting the transpose
        Y_Btest = np.transpose(Y_Btest)

        # Get the core points of the cluster 0
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        coreP = self.Y_BmeterID[core_samples_mask]

        # Detect the number of points outside the cluster
        count = 0
        for p in Y_Btest:
            min_dist = np.min(euclidean_distances(coreP, [p]))
            if min_dist <= radius:
                count += 1

        return count, len(Y_Btest)


def insertWeekColumnToTestingDataframe(dframe):
    min_day = int(dframe.DT.min() / 100) * 100  # 622|00
    max_day = int(dframe.DT.max() / 100) * 100  # 726|00
    week = 60

    df = pd.DataFrame()

    day = min_day
    while day <= max_day:
        aux = dframe.query('DT >= @day & DT < (@day + 7*100)')
        aux.insert(1, "Week", week * np.ones(aux.shape[0], dtype='int'), True)
        df = pd.concat([df, aux])

        day += 7 * 100
        week += 1

    return df


def getMatrixB(mat):
    # Matrix B: rearranged from df dimension B[nObs=336,nMeterID X nweeks]
    nWeeks = int(mat.shape[0] / nObs)

    # Week 0 for the first week
    B = mat[0:nObs, :]

    for i in range(nWeeks - 1):
        B = np.block([B, mat[nObs * (i + 1):nObs * (i + 2), :]])

    return B