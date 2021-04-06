"""
@Author: Simona Bernardi, Raúl Javierre
@Date: updated 06/04/2021
The main program generates the results used for the EDCC21 paper:
    - The percentage of PCA-DBSCAN experiments that could not be completed
    ---> Refined results
    - Boxplots of metrics (tpr,tnr--> equal to accuracy due to partial confusion matrices) 
      for each detector, attack, dataset
    ---> Summary results
    - Generates a report to the global metrics (stdout, csv, latex and html formats)
    - Plots comparing the usages between different behaviours. It is hardcoded in the main function. Moreover, it prints
    quantitative information for each behaviour (mean, standard deviation, q1, q2, q3, iqr, last minus first).
    - The utility of each attack (how much money does an attacker save?) comparing the different bills.
from:
    ./script_results/<dataset>_detector_comparer_results.csv
    ./script_results/.../<meterID>_<behaviour>_<first_week>_<last_week>.csv    | if behaviour is omitted -> Normal/False
"""

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import sqrt
from statsmodels.stats.weightstats import zconfint
from src import meterIDsEnergy, meterIDsGas


def add_recall_precision_and_fmeasure_to_dataframe(df):
    df['recall'] = df['n_tp'] / (df['n_tp'] + df['n_fn'])
    df['precision'] = df['n_tp'] / (df['n_tp'] + df['n_fp'])
    df['f_measure'] = 2 * df['recall'] * df['precision'] / (df['recall'] + df['precision'])


def add_classification_column_to_dataframe(df, type_of_dataset):
    classifications = pd.read_csv("./script_results/meterID_" + type_of_dataset + ".csv")
    for index, row in classifications.iterrows():
        meterID = row['ID']
        classification = row['Code']
        df.loc[df['meterID'] == meterID, 'classification'] = str(classification)


def generate_metric_plot_for_each_detector(behaviour, detectors, metric, classification, df_behaviour):
    data = []
    # Selecting detector
    for detector in detectors:
        df_behaviour_detector = df_behaviour[df_behaviour['detector'] == detector][metric]
        data.append(df_behaviour_detector)

    fig, ax = plt.subplots(figsize=(8,4))
    #ax.set_title("Detectors " + metric + " | behaviour = " + behaviour + " | Classification = " + classification)
    ax.tick_params(axis='x', labelsize=12)
    ax.boxplot(data, labels=detectors)
    plt.show()


def set_and_get_variables(type_of_dataset):
    if type_of_dataset == "energy":
        classifications = ['1'] #, '2', '3']
        meterIDs = meterIDsEnergy
    else:                   #gas
        classifications = ['1'] #, '2', '3', '4', '5']
        meterIDs = meterIDsGas

    detector_comparer = pd.read_csv("./script_results/" + type_of_dataset + "_detector_comparer_results.csv")
    add_recall_precision_and_fmeasure_to_dataframe(detector_comparer)
    add_classification_column_to_dataframe(detector_comparer, type_of_dataset)
    detector_comparer = detector_comparer.fillna(0)

    return detector_comparer, classifications, meterIDs


def compare_the_detectors_for_each_behaviour_for_each_metric_for_each_classification(df, detectors, behaviours, metrics, classifications):
    # Selecting behaviour
    for behaviour in behaviours:
        df_behaviour = df[df['attack'] == behaviour]

        # Selecting metric
        for metric in metrics:

            # Selecting classification
            for classification in classifications:
                df_behaviour_classification = df_behaviour[df_behaviour['classification'] == classification]

                generate_metric_plot_for_each_detector(behaviour, detectors, metric, classification, df_behaviour_classification)


def compare_the_detectors_for_each_behaviour_for_each_metric(df, detectors, behaviours, metrics):
    # Selecting behaviour
    for behaviour in behaviours:
        df_behaviour = df[df['attack'] == behaviour]
        # Selecting metric
        for metric in metrics:
            generate_metric_plot_for_each_detector(behaviour, detectors, metric, 'ALL', df_behaviour)





def print_quantitative_stats_of_a_behaviour(name_of_behaviour, usages_of_behaviour):
    print("\n\nName of behaviour:\t" + name_of_behaviour)
    print("Mean:\t\t\t\t" + str(usages_of_behaviour.mean()))
    print("Standard deviation:\t" + str(usages_of_behaviour.std()))
    print("Q1:\t\t\t\t\t" + str(usages_of_behaviour.quantile(0.25)))
    print("Q2:\t\t\t\t\t" + str(usages_of_behaviour.quantile(0.5)))
    print("Q3:\t\t\t\t\t" + str(usages_of_behaviour.quantile(0.75)))
    print("IQR:\t\t\t\t" + str(usages_of_behaviour.quantile(0.75) - usages_of_behaviour.quantile(0.25)))
    print("Last minus first:\t" + str(usages_of_behaviour.tail(1).values[0] - usages_of_behaviour.head(1).values[0]))


def compare_behaviours(label_behaviour1, label_behaviour2, file_behaviour1, file_behaviour2, title):
    number_of_meditions = range(96) #two days

    behaviour1_usages = pd.read_csv(file_behaviour1)['Usage'].head(len(number_of_meditions))
    behaviour2_usages = pd.read_csv(file_behaviour2)['Usage'].head(len(number_of_meditions))

    plt.title(title)
    plt.xlabel('Time (every half an hour during two days)')
    plt.ylabel('Usage (kWh)')
    plt.plot(number_of_meditions, behaviour1_usages, label=label_behaviour1)
    plt.plot(number_of_meditions, behaviour2_usages, label=label_behaviour2)
    plt.legend()

    plt.show()

    print_quantitative_stats_of_a_behaviour(label_behaviour1, behaviour1_usages)
    print_quantitative_stats_of_a_behaviour(label_behaviour2, behaviour2_usages)

def compare_normal_with_attack_classes(label_behaviour1, label_behaviour2, file_behaviour1, file_behaviour2):
    number_of_meditions = range(96) #two days

    behaviour1_usages = pd.read_csv(file_behaviour1)['Usage'].head(len(number_of_meditions))
    #plt.title(title)
    plt.plot(number_of_meditions, behaviour1_usages, label=label_behaviour1)

    for i in range(len(file_behaviour2)):
    	behaviour2_usages = pd.read_csv(file_behaviour2[i])['Usage'].head(len(number_of_meditions))
    	plt.plot(number_of_meditions, behaviour2_usages, label=label_behaviour2[i], linestyle=':') #dotted lines

    plt.legend()
    plt.xlabel('Time (every half an hour)')
    plt.ylabel('Usage (kWh)')

    plt.show()



def compare_normal_behaviour_with(another_behaviour, dataset):
    try:
        if dataset == 'energy':
            """It uses the meterID 1014 and the Energy ISSDA CER dataset"""
            compare_behaviours(label_behaviour1="Normal", label_behaviour2=another_behaviour,
                               file_behaviour1="./script_results/energy_training_data/1014_0_60.csv",
                               file_behaviour2="./script_results/energy_training_data/1014_" + another_behaviour + "_0_60.csv",
                               title="Normal behaviour VS " + another_behaviour + " behaviour (Energy ISSDA CER | ID=1014)")

        elif dataset == 'gas':
            """It uses the meterID 2513 and the Gas ISSDA CER dataset"""
            compare_behaviours(label_behaviour1="Normal", label_behaviour2=another_behaviour,
                               file_behaviour1="./script_results/gas_training_data/2513_0_60.csv",
                               file_behaviour2="./script_results/gas_training_data/2513_" + another_behaviour + "_0_60.csv",
                               title="Normal behaviour VS " + another_behaviour + " behaviour (Gas ISSDA CER | ID=2513)")
    except FileNotFoundError:
        pass  # when another_behaviour=False, compare_behaviours will throw an exception


def generate_avg_bill_for_each_behaviour_for_60_weeks(behaviours, type_of_dataset, meterIDs):
    AVG_PRICE_KWH_OFF_PEAK_PERIOD = 0.04  # €
    AVG_PRICE_KWH_PEAK_PERIOD = 0.11  # €

    # Off peak period is from 00:00 to 09:00 and peak period is from 09:00 to 24:00
    NINE_AM = 19

    for behaviour in behaviours:
        behaviour_total_price = 0.0

        for meterID in meterIDs:
            if behaviour == 'False':
                df_meterID = pd.read_csv('./script_results/' + type_of_dataset + '_training_data/' + str(meterID) + '_0_60.csv')
            else:
                df_meterID = pd.read_csv('./script_results/' + type_of_dataset + '_training_data/' + str(meterID) + '_' + behaviour + '_0_60.csv')

            # takes the records between 09:00 and 24:00
            df_prices_peak = df_meterID[df_meterID.DT % 100 > NINE_AM]['Usage']

            # takes the records between 00:00 and 09:00
            df_prices_off_peak = df_meterID[df_meterID.DT % 100 <= NINE_AM]['Usage']

            behaviour_total_price += df_prices_peak.sum() * AVG_PRICE_KWH_PEAK_PERIOD +\
                                     df_prices_off_peak.sum() * AVG_PRICE_KWH_OFF_PEAK_PERIOD

        print("\nBehaviour:\t" + behaviour + "\nAvg Bill:\t" + str(behaviour_total_price / len(meterIDs)) + " €")


def treat_PCA_DBSCAN_anomalies(detector_results):
    # Print percentage of PCA-DBSCAN experiments that could be completed
    pca_dbscan_experiments = detector_results[detector_results.detector == 'PCA-DBSCAN']
    uncompleted_pca_dbscan_experiments = pca_dbscan_experiments[pca_dbscan_experiments.n_tp == -1]

    n_pca_dbscan_experiments = len(pca_dbscan_experiments.index)
    n_uncompleted_pca_dbscan_experiments = len(uncompleted_pca_dbscan_experiments.index)
    print("PCA-DBSCAN experiments:\t\t\t\t" + str(n_pca_dbscan_experiments))
    print("PCA-DBSCAN uncompleted experiments:\t" + str(n_uncompleted_pca_dbscan_experiments))
    print(
        "% PCA-DBSCAN uncompleted:\t\t\t" + str(100 * n_uncompleted_pca_dbscan_experiments / n_pca_dbscan_experiments))

    # Remove uncompleted PCA-DBSCAN experiments from the detector_results variable
    return detector_results[detector_results.n_tp >= 0]


def compute_experiment_exec_time_in_days(df,behaviours):
	SECONDS_IN__ONE_DAY = 86400
	totExecTime = 0.0
	for behaviour in behaviours:
		df_behaviour = df[df['attack'] == behaviour]
		totExecTime += (df_behaviour['time_model_creation'].sum() + df_behaviour['time_model_prediction'].sum())
	return totExecTime / SECONDS_IN__ONE_DAY



def generate_scenario_metric_for_each_detector(df,behaviours,detectors,type):
    formatter = "{:.3f}"
    acc_m = []
    for behaviour in behaviours:
        df_behaviour = df[df['attack'] == behaviour]
        acc_v =[]
        for detector in detectors:
            df_detector = df_behaviour[df_behaviour['detector'] == detector]
            if behaviour == 'False':
                col = [df_detector['n_tn'].sum(), df_detector['n_fp'].sum()]
            else:
                col = [df_detector['n_tp'].sum(), df_detector['n_fn'].sum()]
                
            #Accuracy formula: sensitivity for atttack scenarios, specificity for normal
            acc_v.append(col[0] / (col[0]+col[1]))

            #print(detector,  formatter.format(acc))
        acc_m.append(acc_v)

    scenario_acc = pd.DataFrame(np.array(acc_m), columns=detectors, index=behaviours)
    outputFile= './script_results/' + type + 'scenariosAccuracy'
    scenario_acc.to_csv(outputFile, index=True, float_format="{:.3f}".format)
    scenario_acc.to_latex(outputFile +'.tex', index=True, float_format="{:.3f}".format)
    scenario_acc.to_html(outputFile + '.html', index=True, float_format="{:.3f}".format)


#Matthews correlation coefficient: useful when negative/positive classes are not balanced 
def compute_MCC(mat):
    
    D = (mat[0,0] + mat[0,1]) * (mat[0,0] + mat[1,0]) * (mat[1,1] + mat[0,1]) * (mat[1,1] + mat[1,0])
    return ((mat[0,0] * mat[1,1]) - (mat[0,1] * mat[1,0])) / sqrt(D)

def save_global_metrics_for_each_detector(df,type): 
	outputFile= './script_results/' + type + 'globalMetrics'
	df.to_csv(outputFile, index=False, float_format="{:.3f}".format)
	df.to_latex(outputFile +'.tex', index=False, float_format="{:.3f}".format)
	df.to_html(outputFile + '.html', index=False, float_format="{:.3f}".format)

def get_performance_metrics(df): 
	formatter = "{:.3f}"    
	t = dict()
	t["min"] = formatter.format(df.min())
	t["mean"] = formatter.format(df.mean())
	t["std"] = formatter.format(df.std())
	t["Q1"] = formatter.format(df.quantile(0.25))
	t["Q2"] = formatter.format(df.quantile(0.5))
	t["Q3"] = formatter.format(df.quantile(0.75))
	t["max"] = formatter.format(df.max())
	lower, upper = zconfint(df.to_numpy(), alpha=0.05, alternative='two-sided',  ddof=1.0)
	t["lower-95%"] = formatter.format(lower)
	t["upper-95%"] = formatter.format(upper)
	t["CI-half-length"] = formatter.format((upper-lower)/2)
	
	return t



def generate_global_metrics_for_each_detector(df, detectors):
	acc = []
	misclas = []
	rec = []
	fnr = []
	tnr = []
	fpn = []
	prec = []
	npv = []
	bacc = []
	f1_score = []
	mcc = []
	prev = []
	time_build = []
	time_pred = []
	for detector in detectors:
		df_detector = df[df['detector'] == detector]
		positives = [df_detector['n_tp'].sum(), df_detector['n_fp'].sum()]
		negatives = [df_detector['n_fn'].sum(), df_detector['n_tn'].sum()]
		matrix = np.array([positives, negatives])
		#Accuracy: How often it is correct 
		a = (matrix[0,0]+ matrix[1,1]) / matrix.sum()
		acc.append(a)
		misclas.append(1 - a)
        	#Recall (=Sensitivity=True Positive Rate): when it is actually "Yes", how often it predicts "Yes" 
		r = matrix[0,0] / (matrix[0,0]+matrix[1,0])
		rec.append(r)
        	#False Negative Rate: when it is actually "Yes", how often it predicts "No"
		fnr.append(matrix[1,0] / (matrix[0,0]+matrix[1,0]))
        	#True Negative Rate (=Specificity): when it is actually "No", how often it predicts "No"
		s = matrix[1,1] / (matrix[0,1]+matrix[1,1]) 
		tnr.append(s)
        	#False Positive Rate: when it is actually "No", how often it predicts "Yes"
		fpn.append(matrix[0,1] / (matrix[0,1]+matrix[1,1]))
        	#Precision: when it predicts "Yes", how often is correct
		p = matrix[0,0] / (matrix[0,0]+matrix[0,1])
		prec.append(p)
        	#Negative predictive value: when it predict "no" how often it is correct
		npv.append(matrix[1,1] / (matrix[1,0]+matrix[1,1]))
        	#Balanced accuracy (useful for imbalanced data)
		bacc.append((r + s) / 2)
        	#F1-score:  to be used instead of accuracy when the sample has a large number of negative values 
		f1_score.append(2 * p * r / (p+r))
        	#Matthews correlation coefficient (useful for imbalanced data)
		mcc.append(compute_MCC(matrix))
        	#Prevalence: how often the "yes" condition actually occur in the sample
		prev.append((matrix[0,0]+matrix[1,0]) / matrix.sum())
		
		df_detector_TMC  = df_detector['time_model_creation']
		df_detector_TMP  = df_detector['time_model_prediction'] 
		
		tb = get_performance_metrics(df_detector_TMC)
		time_build.append(tb)

		tp = get_performance_metrics(df_detector_TMP)
		time_pred.append(tp)
	
	# Add a comma and keep to two d.p.
	pd.options.display.float_format = '{:,.3f}'.format	
	df = pd.DataFrame({'detector': detectors,
			'accuracy': acc,
			'misclass': misclas,
			'recall': rec,
			'FNR': fnr,
			'TNR': tnr,
			'FPN': fpn,
			'precision': prec,
			'NPV': npv,
			'balancedAccuracy': bacc,
			'f1-score': f1_score,
			'mcc': mcc,
			'prevalence': prev,
			'tb' : time_build,
			'tp' : time_pred})
	return df



if __name__ == '__main__':
    """
    args: 
    sys.argv[1]:type (energy, gas)
    """
    type_of_dataset = sys.argv[1]

    # Creating the dataframe and getting the classification array
    detector_results, classifications, meterIDs = set_and_get_variables(type_of_dataset)

    # Listing behaviours, detectors, metrics and classifications
    #behaviours = detector_results['attack'].unique()

    #In the order they appear in the paper
    behaviours = ['False', 'RSA_0.25_1.1', 'RSA_0.5_3', 'Avg', 'Swap', 'FDI10', 'FDI30']
    detectors = ['Min-Avg', 'ARIMA', 'ARIMAX', 'PCA-DBSCAN',  'KLD', 'JSD' , 'NN' ]
    metrics = ['accuracy']

    # Gas dataset wasn't tested with PCA-DBSCAN
    if type_of_dataset != "gas":
        detector_results = treat_PCA_DBSCAN_anomalies(detector_results)

    #Fine-grained results: boxplots
    compare_the_detectors_for_each_behaviour_for_each_metric(detector_results, detectors, behaviours, metrics)
    #compare_the_detectors_for_each_behaviour_for_each_metric_for_each_classification(detector_results, detectors, behaviours, metrics, classifications)

    #Scenario level metrics: sensitivity/specificity (=accuracy, special case)
    generate_scenario_metric_for_each_detector(detector_results,behaviours,detectors,type_of_dataset)

    # Summary report: global quality and performance metrics
    gmetrics = generate_global_metrics_for_each_detector(detector_results, detectors)
    save_global_metrics_for_each_detector(gmetrics,type_of_dataset)


    # Compare normal behaviours and the attack classes, meterID 1014
    file_RSAbehaviours = ["./script_results/energy_training_data/1014_RSA_0.5_3_0_60.csv",
    					  "./script_results/energy_training_data/1014_RSA_0.25_1.1_0_60.csv"]
    RSA_behaviours = [ "RSA[0.5,3]", "RSA[0.25,1.1]"]

    compare_normal_with_attack_classes(label_behaviour1="Normal", label_behaviour2=RSA_behaviours,
                               file_behaviour1="./script_results/energy_training_data/1014_0_60.csv",
                               file_behaviour2=file_RSAbehaviours)

    file_Avgbehaviours = ["./script_results/energy_training_data/1014_Avg_0_60.csv" ] 
                          
    Avg_behaviours = ["Avg[0.5,1.5]" ] 
                     

    compare_normal_with_attack_classes(label_behaviour1="Normal", label_behaviour2=Avg_behaviours,
                               file_behaviour1="./script_results/energy_training_data/1014_0_60.csv",
                               file_behaviour2=file_Avgbehaviours)

    file_Swapbehaviours = ["./script_results/energy_training_data/1014_Swap_0_60.csv"]
    Swap_behaviours = ["Swap"]

    compare_normal_with_attack_classes(label_behaviour1="Normal", label_behaviour2=Swap_behaviours,
                               file_behaviour1="./script_results/energy_training_data/1014_0_60.csv",
                               file_behaviour2=file_Swapbehaviours)

    file_FDIbehaviours = [ "./script_results/energy_training_data/1014_FDI10_0_60.csv",
    						"./script_results/energy_training_data/1014_FDI30_0_60.csv"]
    FDI_behaviours = [ "FDI_10", "FDI_30"]

    compare_normal_with_attack_classes(label_behaviour1="Normal", label_behaviour2=FDI_behaviours,
                               file_behaviour1="./script_results/energy_training_data/1014_0_60.csv",
                               file_behaviour2=file_FDIbehaviours)

    #Execution time of the experiment
    t = compute_experiment_exec_time_in_days(detector_results,behaviours)
    print("Total execution time of the experiment (in days): ", t)

    # Generate avg bill for the meterID 1014 (energy)
    meterIDs = [1014]
    generate_avg_bill_for_each_behaviour_for_60_weeks(behaviours, type_of_dataset, meterIDs)


