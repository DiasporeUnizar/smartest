"""
@Author: Raúl Javierre
@Date: updated 18/03/2021
@Review: Simona Bernardi, 30/03/2021

The main program generates some *basic metrics* to facilitate the comparison of the *detectors* for each *scenario*.
The results with all the metrics are generated in /script_results/<dataset>_detector_comparer_results.csv

**Metrics:**
- Execution time of model creation
- Execution time of model prediction
- Accuracy = (TP + TN) /(TP + TN + FP + FN) 
- Number of TP (True Positives)
- Number of FP (False Positives)
- Number of TN (True Negatives)
- Number of FN (False Negatives)

**Detectors:**
- Min-Avg
- ARIMAX
- ARIMA
- PCA-DBSCAN
- NN
- KLD
- JSD

**Scenarios:**
- False : Normal
- RSA_0.25_1.1
- RSA_0.5_3
- Avg
- Swap
- FDI10
- FDI30
"""

import sys
from src.detectors.DetectorFactory import DetectorFactory
from src import meterIDsGas, meterIDsElectricity
from time import time

# The dataset is set in the first parameter
# The meterIDs used are specific for each dataset
# You can customize the attacks and the detectors you want to use here
tuple_of_attacks = (False, "RSA_0.25_1.1", "RSA_0.5_3", "Avg", "Swap", "FDI10", "FDI30")
list_of_detectors = ["ARIMAX", "ARIMA", "Min-Avg", "PCA-DBSCAN", "KLD", "JSD", "NN"]

if __name__ == '__main__':
    """
    args:
    sys.argv[1]:dataset ("electricity" or "gas")
    """
    if sys.argv[1] != "electricity" and sys.argv[1] != "gas":
        print("Usage: python3 detector_comparer.py <electricity/gas> <test>")
        exit(85)

    if sys.argv[1] == "electricity":
        list_of_meterIDs = meterIDsElectricity
    else:
        list_of_meterIDs = meterIDsGas

    processed_meterIDs = 0
    t0 = time()

    for meterID in list_of_meterIDs:

        for name_of_detector in list_of_detectors:
            detector = DetectorFactory.create_detector(name_of_detector)
            training_dataset = detector.get_training_dataset(meterID, sys.argv[1])
            model, time_model_creation = detector.build_model(training_dataset)

            for attack in tuple_of_attacks:
                testing_dataset = detector.get_testing_dataset(attack, meterID, sys.argv[1])
                predictions, obs, time_model_prediction = detector.predict(testing_dataset, model)
                n_tp, n_tn, n_fp, n_fn = detector.compute_outliers(obs, predictions, attack)

                detector.print_metrics(meterID, name_of_detector, attack, time_model_creation, time_model_prediction, n_tp, n_tn, n_fp, n_fn)
                detector.metrics_to_csv(meterID, name_of_detector, attack, time_model_creation, time_model_prediction, n_tp, n_tn, n_fp, n_fn, sys.argv[1])

        processed_meterIDs += 1

        remaining_meterIDs = len(list_of_meterIDs) - processed_meterIDs
        avg_time = (time() - t0) / processed_meterIDs

        print(str(remaining_meterIDs) + " meterIDs remaining. It will be completed in " + str(remaining_meterIDs * avg_time) + " seconds (aprox.)")
