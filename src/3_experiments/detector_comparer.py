"""
@Author: Raúl Javierre
@Date: updated 18/03/2021
@Review: Simona Bernardi, 30/03/2021

The main program generates some *basic metrics* to facilitate the comparison of the *detectors* for each *attack*.
The results with all the metrics are generated in /script_results/dataset_detector_comparer_results.csv

**Metrics:**
- Execution time of model creation
- Execution time of model prediction
- Accuracy = (TP + TN) /(TP + TN + FP + FN) 
/// SB: the accuracy is actually a "derived metric" that can be posponed (computed with the dashboard)
- Number of TP (True Positives)
- Number of FP (False Positives)
- Number of TN (True Negatives)
- Number of FN (False Negatives)

**Detectors:**
- Min-Avg
- ARIMAX
- ARIMA
- PCA-DBSCAN
- K-Means
- MiniBatchK-Means
- FisherJenks
- NN
- KLD
- JSD
- IsolationForest
- TEG_Hamming
- TEG_Cosine
- TEG_KLD

**Attacks:**
- False
- RSA_0.5_1.5
- RSA_0.25_1.1
- RSA_0.5_3
- Avg
- Min-Avg
- Swap
- FDI0
- FDI5
- FDI10
- FDI20
- FDI30
"""

import sys
from src.detectors.DetectorFactory import DetectorFactory
from src.experiments import meterIDsGas, meterIDsEnergy, test_tuple_of_attacks, test_list_of_detectors
from time import time

# The dataset is set in the first parameter
# The meterIDs used are specific for each dataset
# You can customize the attacks and the detectors you want to use here
tuple_of_attacks = (False, "RSA_0.5_1.5", "RSA_0.25_1.1", "RSA_0.5_3", "Avg", "Min-Avg", "Swap", "FDI0", "FDI5", "FDI10", "FDI20", "FDI30")
list_of_detectors = ["PCA-DBSCAN"]

if __name__ == '__main__':
    """
    args:
    sys.argv[1]:dataset ("energy" or "gas")
    sys.argv[2]:test ("on" or None, if it is set, it uses only 1 meterID, all the attacks and only the lightest detectors)
    """
    if sys.argv[1] != "energy" and sys.argv[1] != "gas":
        print("Usage: python3 training_and_testing_generator.py <energy/gas> <test>")
        exit(85)

    if sys.argv[1] == "energy":
        list_of_meterIDs = meterIDsEnergy
    else:
        list_of_meterIDs = meterIDsGas

    test_mode = len(sys.argv) == 3 and sys.argv[2] == "on"
    if test_mode:
        list_of_meterIDs = [list_of_meterIDs[0]]
        tuple_of_attacks = test_tuple_of_attacks
        list_of_detectors = test_list_of_detectors

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
                detector.metrics_to_csv(meterID, name_of_detector, attack, time_model_creation, time_model_prediction, n_tp, n_tn, n_fp, n_fn, sys.argv[1] if not test_mode else "test_" + sys.argv[1])

        processed_meterIDs += 1

        remaining_meterIDs = len(list_of_meterIDs) - processed_meterIDs
        avg_time = (time() - t0) / processed_meterIDs

        print(str(remaining_meterIDs) + " meterIDs remaining. It will be completed in " + str(remaining_meterIDs * avg_time) + " seconds (aprox.)")
