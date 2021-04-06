"""
@Author: Raul Javierre
@Date: updated 18/03/2021

The main program generates:
    1. Training csv files on /script_results/energy_training_data (GDrive)
        with syntax: meterID_0_60.csv | example: 2458_0_60.csv  (no attack)
        or syntax: meterID_kind_0_60.csv | example: 2458_Avg_0_60.csv   (attack = kind)
    2. Testing csv files on /script_results/energy_testing_data (GDrive)
        with syntax: meterID_kind_x_y.csv | example: 2458_Avg_x_y.csv   (attack = kind)
        or syntax: meterID_x_y.csv | example: 2458_x_y.csv  (no attack)

        (x and y are the first and last testing weeks, respectively)

for a list of 500 meterIDs previously selected (see <dataset>_customer_analysis.py)
"""

import pandas as pd
import numpy as np
import random
import sys
from src import meterIDsGas, meterIDsEnergy

NINE_AM = 19

FIRST_WEEK_TRAINING = 0
LAST_WEEK_TRAINING = 60
FIRST_WEEK_TESTING = LAST_WEEK_TRAINING + 1
LAST_WEEK_TESTING_ENERGY = 75
LAST_WEEK_TESTING_GAS = 77

SEVEN_DAYS_PER_WEEK = 7
NOBS_PER_DAY = 48

THIRTY_PER_CENT = 0.3
TWENTY_PER_CENT = 0.2
TEN_PER_CENT = 0.1
FIVE_PER_CENT = 0.05

MAX_MIN_AVG = 9999999
SEED = 19990722
dataset = None


def get_the_eighteen_highest_usage_rows_of_peak_period(df):
    df_09_24 = df[df.DT % 100 > NINE_AM]   # takes the records between 09:00 and 24:00
    return df_09_24.nlargest(18, 'Usage')   # returns the 18 maximum values of Usage between 09:00 and 24:00


def get_the_off_peak_period(df):
    return df[df.DT % 100 < NINE_AM]   # returns the records between 00:00 and 09:00


def swap_usages(df):
    list_df = [df[i:i + NOBS_PER_DAY] for i in range(0, df.shape[0], NOBS_PER_DAY)]     # get a dataframe for each day

    for i in range(0, len(list_df)):
        eighteen_highest_peak_period = get_the_eighteen_highest_usage_rows_of_peak_period(list_df[i])
        off_peak_period = get_the_off_peak_period(list_df[i])

        dt_eighteen_highest_peak_period = eighteen_highest_peak_period['DT'].to_list()
        eighteen_highest_peak_period['DT'] = off_peak_period['DT'].to_list()
        off_peak_period['DT'] = dt_eighteen_highest_peak_period

        list_dts_modified = eighteen_highest_peak_period['DT'].to_list() + off_peak_period['DT'].to_list()
        list_df[i] = list_df[i].query('DT not in @list_dts_modified')
        list_df[i] = list_df[i].append(eighteen_highest_peak_period)
        list_df[i] = list_df[i].append(off_peak_period)
        list_df[i] = list_df[i].sort_values(by=['DT'])

    return pd.concat(list_df)['Usage'].to_list()


def get_min_avg_of_training_weeks(caseID):   # Heavy!!! and a little bit legacy...
    min_avg = MAX_MIN_AVG

    for i in range(0, LAST_WEEK_TRAINING + 1):  # training weeks [week0, week60]
        filename = "./ISSDA-CER/" + dataset + "/data/data_all_filtered/" + dataset + "DataWeek " + str(i)
        try:
            dset = pd.read_csv(filename)
        except FileNotFoundError:  # Range is not complete or is out of range
            continue
        dset = dset[dset.ID == int(caseID)]
        avg = dset['Usage'].mean()

        if avg < min_avg:
            min_avg = avg

    return min_avg


def load_week_files(firstWeek, lastWeek, caseID=None):
    df = pd.DataFrame()
    for i in range(firstWeek, lastWeek + 1):
        filename = "./ISSDA-CER/" + dataset + "/data/data_all_filtered/" + dataset + "DataWeek " + str(i)
        try:
            dset = pd.read_csv(filename)
            print(filename, "successfully obtained")
        except FileNotFoundError:  # Range is not complete or is out of range
            continue

        if caseID is not None:
            dset = dset[dset.ID == int(caseID)]

        df = df.append(dset)

    return df


class AttackInjector:
    """General attack injector"""

    def __init__(self, caseID=None):
        super().__init__()
        self.caseID = caseID

    def inject_attack(self, original_consume, a, b):
        np.random.seed(SEED)
        noise = float(a) + np.random.rand(original_consume.size) * (float(b) - float(a))
        return original_consume * noise

    def attack_dataset(self, data, kind, a=None, b=None):
        consumed_faked = None

        if kind.startswith('RSA'):
            consumed_faked = self.inject_attack(data.Usage.to_numpy(), a, b)
        elif kind == 'Avg':
            mean_Kw = []
            # Split the dataframe into nWeeks dataframes
            data_weeks = np.array_split(data, len(data.index) / (NOBS_PER_DAY * SEVEN_DAYS_PER_WEEK))
            for data_week in data_weeks:
                # Create an array with len = NOBS_PER_DAY * 7 times
                mean_data_week = np.empty(len(data_week.index))
                # Fill the array with the mean of the week
                mean_data_week.fill(data_week['Usage'].mean())
                mean_Kw.append(mean_data_week)
            consumed_faked = self.inject_attack(np.array(mean_Kw).flatten(), a, b)
        elif kind == 'Min-Avg':
            min_avg = get_min_avg_of_training_weeks(self.caseID)
            consumed_faked = []
            random.seed(SEED)
            for i in range(0, len(data)):
                consumed_faked.append(random.uniform(min_avg, min_avg + 1))
        elif kind == 'Swap':
            consumed_faked = swap_usages(data)
        elif kind == 'FDI0':
            consumed_faked = np.zeros(len(data))
        elif kind == 'FDI5':
            data['5%'] = data['Usage'] * FIVE_PER_CENT
            consumed_faked = data['5%'].tolist()
        elif kind == 'FDI10':
            data['10%'] = data['Usage'] * TEN_PER_CENT
            consumed_faked = data['10%'].tolist()
        elif kind == 'FDI20':
            data['20%'] = data['Usage'] * TWENTY_PER_CENT
            consumed_faked = data['20%'].tolist()
        elif kind == 'FDI30':
            data['30%'] = data['Usage'] * THIRTY_PER_CENT
            consumed_faked = data['30%'].tolist()
        else:
            print("Error: kind", kind, "not found")
            exit(1)

        return pd.DataFrame({'ID': self.caseID, 'DT': data.DT, 'Usage': consumed_faked})


def generate_attacked_file(attack_injector, meterID, testing_set, d_type, first_week, last_week, dir, kind, a=None, b=None):
    print("Generating file:", "./script_results/" + d_type.lower() + "_" + dir + "/" + str(meterID) + "_" + kind + "_" + str(first_week) + "_" + str(last_week) + ".csv")
    test_set = attack_injector.attack_dataset(data=testing_set, kind=kind, a=a, b=b)
    test_set.to_csv("./script_results/" + d_type.lower() + "_" + dir + "/" + str(meterID) + "_" + kind + "_" + str(first_week) + "_" + str(last_week) + ".csv", index=False)  # GDrive


if __name__ == '__main__':
    '''
    args: 
    sys.argv[1]:dataset ("Energy" or "Gas")
    '''

    if len(sys.argv) != 2 or sys.argv[1] != "Energy" and sys.argv[1] != "Gas":
        print("Usage: python3 training_and_testing_generator.py <Energy/Gas>")
        exit(85)

    # Global variable
    dataset = sys.argv[1]

    if dataset == "Energy":
        first_week_train = FIRST_WEEK_TRAINING
        last_week_train = LAST_WEEK_TRAINING
        first_week_test = FIRST_WEEK_TESTING
        last_week_test = LAST_WEEK_TESTING_ENERGY
        meterIDs = meterIDsEnergy
        dataset_training_all_meter_ids = load_week_files(firstWeek=first_week_train, lastWeek=last_week_train)
        dataset_testing_all_meter_ids = load_week_files(firstWeek=first_week_test, lastWeek=last_week_test)
    elif dataset == "Gas":
        first_week_train = FIRST_WEEK_TRAINING
        last_week_train = LAST_WEEK_TRAINING
        first_week_test = FIRST_WEEK_TESTING
        last_week_test = LAST_WEEK_TESTING_GAS
        meterIDs = meterIDsGas
        dataset_training_all_meter_ids = load_week_files(firstWeek=first_week_train, lastWeek=last_week_train)
        dataset_testing_all_meter_ids = load_week_files(firstWeek=first_week_test, lastWeek=last_week_test)
    else:
        print("Usage: python3 training_and_testing_generator.py <Energy/Gas>")
        exit(85)

    for meterID in meterIDs:
        attack_injector = AttackInjector(caseID=meterID)

        # Training weeks
        training_set = dataset_training_all_meter_ids.query('ID == @meterID')
        print("\nGenerating file:", "./script_results/" + dataset.lower() + "_training_data/" + str(meterID) + "_" + str(first_week_train) + "_" + str(last_week_train) + ".csv")
        training_set.to_csv("./script_results/" + dataset.lower() + "_training_data/" + str(meterID) + "_" + str(first_week_train) + "_" + str(last_week_train) + ".csv", index=False)   # GDrive
        generate_attacked_file(attack_injector, meterID, training_set, dataset, first_week_train, last_week_train, "training_data", "Swap")
        generate_attacked_file(attack_injector, meterID, training_set, dataset, first_week_train, last_week_train, "training_data", "RSA_0.5_1.5", 0.5, 1.5)
        generate_attacked_file(attack_injector, meterID, training_set, dataset, first_week_train, last_week_train, "training_data", "Avg", 0.5, 1.5)
        generate_attacked_file(attack_injector, meterID, training_set, dataset, first_week_train, last_week_train, "training_data", "Min-Avg")
        generate_attacked_file(attack_injector, meterID, training_set, dataset, first_week_train, last_week_train, "training_data", "FDI0")
        generate_attacked_file(attack_injector, meterID, training_set, dataset, first_week_train, last_week_train, "training_data", "FDI5")
        generate_attacked_file(attack_injector, meterID, training_set, dataset, first_week_train, last_week_train, "training_data", "FDI10")
        generate_attacked_file(attack_injector, meterID, training_set, dataset, first_week_train, last_week_train, "training_data", "FDI20")
        generate_attacked_file(attack_injector, meterID, training_set, dataset, first_week_train, last_week_train, "training_data", "FDI30")
        generate_attacked_file(attack_injector, meterID, training_set, dataset, first_week_train, last_week_train, "training_data", "RSA_0.25_1.1", 0.25, 1.1)
        generate_attacked_file(attack_injector, meterID, training_set, dataset, first_week_train, last_week_train, "training_data", "RSA_0.5_3", 0.5, 3)

        # Testing weeks
        testing_set = dataset_testing_all_meter_ids.query('ID == @meterID')
        print("Generating file:", "./script_results/" + dataset.lower() + "_testing_data/" + str(meterID) + "_" + str(first_week_test) + "_" + str(last_week_test) + ".csv")
        testing_set.to_csv("./script_results/" + dataset.lower() + "_testing_data/" + str(meterID) + "_" + str(first_week_test) + "_" + str(last_week_test) + ".csv", index=False)   # GDrive
        generate_attacked_file(attack_injector, meterID, testing_set, dataset, first_week_test, last_week_test, "testing_data", "Swap")
        generate_attacked_file(attack_injector, meterID, training_set, dataset, first_week_train, last_week_train,"testing_data", "RSA_0.5_1.5", 0.5, 1.5)
        generate_attacked_file(attack_injector, meterID, testing_set, dataset, first_week_test, last_week_test, "testing_data", "Avg", 0.5, 1.5)
        generate_attacked_file(attack_injector, meterID, testing_set, dataset, first_week_test, last_week_test, "testing_data", "Min-Avg")
        generate_attacked_file(attack_injector, meterID, testing_set, dataset, first_week_test, last_week_test, "testing_data", "FDI0")
        generate_attacked_file(attack_injector, meterID, testing_set, dataset, first_week_test, last_week_test, "testing_data", "FDI5")
        generate_attacked_file(attack_injector, meterID, testing_set, dataset, first_week_test, last_week_test, "testing_data", "FDI10")
        generate_attacked_file(attack_injector, meterID, testing_set, dataset, first_week_test, last_week_test, "testing_data", "FDI20")
        generate_attacked_file(attack_injector, meterID, testing_set, dataset, first_week_test, last_week_test, "testing_data", "FDI30")
        generate_attacked_file(attack_injector, meterID, testing_set, dataset, first_week_test, last_week_test, "testing_data", "RSA_0.25_1.1", 0.25, 1.1)
        generate_attacked_file(attack_injector, meterID, testing_set, dataset, first_week_test, last_week_test, "testing_data", "RSA_0.5_3", 0.5, 3)
