"""
@Author: Raúl Javierre
@Date: updated 04/03/2021

This module provides a DetectorFactory class
"""

from .ARIMA import ARIMA
from .ARIMAX import ARIMAX
from .FisherJenks import FisherJenks
from .IsolationForest import Isolation
from .JSD import JSD
from .KLD import KLD
from .KMeans import K_Means
from .MinAverage import MinAverage
from .MiniBatchKMeans import MiniBatchK_Means
from .NN import NN
from .PCADBSCAN import PCADBSCAN
from .TEG import TEG


class DetectorFactory:
    """
    It provides a static method to create concrete detectors
    """

    @staticmethod
    def create_detector(detector):
        if detector == "Min-Avg":
            return MinAverage()
        elif detector == "JSD":
            return JSD()
        elif detector == "ARIMA":
            return ARIMA()
        elif detector == "ARIMAX":
            return ARIMAX()
        elif detector == "FisherJenks":
            return FisherJenks()
        elif detector == "KLD":
            return KLD()
        elif detector == "K-Means":
            return K_Means()
        elif detector == "MiniBatchK-Means":
            return MiniBatchK_Means()
        elif detector == "PCA-DBSCAN":
            return PCADBSCAN()
        elif detector == "IsolationForest":
            return Isolation()
        elif detector == "NN":
            return NN()
        elif detector == "TEG_Cosine":
            return TEG("Cosine")
        elif detector == "TEG_Hamming":
            return TEG("Hamming")
        elif detector == "TEG_KLD":
            return TEG("KLD")
        else:
            raise KeyError("Detector " + detector + " not found. You must add a conditional branch in /src/detectors/DetectorFactory.py")
