"""
@Author: Raúl Javierre
@Date: updated 04/03/2021

This module provides a DetectorFactory class
"""

from .ARIMA import ARIMA
from .ARIMAX import ARIMAX
from .JSD import JSD
from .KLD import KLD
from .NN import NN
from .PCADBSCAN import PCADBSCAN
from .MinAverage import MinAverage

class DetectorFactory:
    """
    It provides a static method to create concrete detectors
    """

    @staticmethod
    def create_detector(detector):
        if detector == "JSD":
            return JSD()
        elif detector == "ARIMA":
            return ARIMA()
        elif detector == "ARIMAX":
            return ARIMAX()
        elif detector == "KLD":
            return KLD()
        elif detector == "PCA-DBSCAN":
            return PCADBSCAN()
        elif detector == "NN":
            return NN()
        elif detector == "Min-Avg":
            return MinAverage()
        else:
            raise KeyError("Detector " + detector + " not found.")
