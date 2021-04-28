"""
Updated on April 21, 2021

@author: Simona Bernardi, Raúl Javierre
Preprocessing of the ISSDA dataset in order to generate a set of files (each one referring to a week of readings)
and /script_results/meterID_<electricity/gas>.csv

Take care with the restrictions indicated on "preconditions()" function.

└── smartest/
    ├── Electricity/
    │   ├── data/
    │   │   ├── data_original (6 ISSDA-CER's txt files)
    │   │   ├── data_all
    │   │   └── data_all_filtered
    │   └── doc/
    │       └── customerClassification.csv (*) (from ISSDA-CER: SME and Residential allocations.xlsx)
    └── Gas/
        ├── data/
        │   ├── data_original (78 ISSDA-CER's files)
        │   ├── data_all
        │   └── data_all_filtered
        └── doc/
            └── customerClassification.csv (**) (from ISSDA-CER: Residential allocations.xlsx)

(*) -> First and second column:
            ID;Code
            1000;...
            1001;...

(**) -> First and third column:
            ID;Code
            1000;...
            1001;...
"""

import sys
import os
import csv
import pandas as pd
import numpy as np

first = 19501  # first reading: 14/7/2009 tuesday -- first half an hour
last = 73048  # last reading
nObs = 336  # number of readings


class dataAnalyzer:

    def __init__(self):
        # dataframe with all the readings: columns ['ID','DT','Usage']
        self.df = pd.DataFrame()
        # list of selected (complete) meterIDs together with their classification
        self.meterID = pd.DataFrame()

    def getDataSetFiles(self, dpath, prefixFilename):
        # This part is very dependent on how the dataset is structured
        # Loads the list of filenames in my_files array and then sort the files
        # according to the number of week
        dirFiles = os.listdir(dpath)
        myfiles = []
        for files in dirFiles:
            if files.startswith(prefixFilename):
                pathfile = dpath + files
                myfiles.append(pathfile)
                # wont work if len(myfiles) >= 10
                myfiles = sorted(myfiles)

        return myfiles

    def writeFilteredData(self, files, firstWeek, lastWeek, type):
        if type == "Electricity":
            data_all_filtered = "./ISSDA-CER/Electricity/data/data_all_filtered/"
        else:
            data_all_filtered = "./ISSDA-CER/Gas/data/data_all_filtered/"

        # Consider all the weeks between the first and the last one (included)
        ID = self.meterID.index.to_numpy()
        groupID = pd.DataFrame(ID, columns=['ID'])
        for file in files[firstWeek:lastWeek + 1]:
            print(file)
            dset = pd.read_csv(file)  # load as a pd.dataFrame
            # Select those observations where the timestamp timecode is between 0-48 hours
            dset = dset[(dset.DT % 100 >= 0) & (dset.DT % 100 <= 48)]
            # Inner join to select just the complete meterIDs
            dset = pd.merge(dset, groupID, on='ID', how='inner')
            dset.round(3)
            # Same file name but in different directory
            filef = data_all_filtered + file.split("/")[-1]
            dset.to_csv(filef, index=False)

    def filterDataSetFirstRound(self, files, firstWeek, lastWeek, clfile, type):
        def getCustomerClassification(file, missingID):
            # Classification according to the file "SME and Residential allocations"
            # (CER Electricity Revised March 2012)
            d_set = pd.read_csv(file, sep=';', index_col='ID')
            customers = d_set.drop(missingID, axis=0)

            # customers is a dataFrame with index='ID' and column='Code'
            return customers

        ID = np.int_([])  # np array initialization
        # Consider all the weeks between the first and the last one (included)
        for file in files[firstWeek:lastWeek + 1]:
            print(file)
            dset = pd.read_csv(file)  # load as a pd.dataFrame

            # Group by meterID and count the number of observations
            groupID = dset.groupby('ID').count()

            # Select those that have the number of observation less than nObs
            groupID = groupID[groupID.Usage < nObs]

            # Concatenate the missing IDs in the ID array
            ID = np.concatenate((ID, groupID.index.to_numpy()))

            # Remove the ID repetitions
            ID = np.unique(ID)

        # load the classification of the customers with complete observations
        self.meterID = getCustomerClassification(clfile, ID)

        # Generate filtered dataset in data_all_filtered: very heavy to be executed once
        self.writeFilteredData(files, firstWeek, lastWeek, type)

    def filterDataSetSecondRound(self, data_all_filtered, type, firstWeek, lastWeek):
        # Last step: from 4733 to 4626 meterIDs. Some meterIDs are not complete for all the weeks.
        # Update and store the list of complete meterIDs together with their classification
        files = self.getDataSetFiles(data_all_filtered, type + "DataWeek")
        self.update_df(files, firstWeek, lastWeek)
        uncompleted_meter_ids = self.get_uncompleted_meter_ids(files)
        # result (for Electricity):
        # uncompleted_meter_ids = [5640, 7176, 2573, 1038, 6675, 1048, 1572, 4647, 7223, 5690, 3646, 3651, 5699, 5705, 2122,
        #                           4687, 5202, 1621, 2150, 6248, 3179, 3691, 7275, 6255, 5234, 3190, 4729, 6777, 5759,
        #                           3202, 3719, 5255, 5769, 4238, 4240, 6290, 3735, 6296, 2204, 5792, 7331, 3237, 2726,
        #                           6823, 3250, 2227, 6842, 1211, 1738, 5327, 6361, 5341, 2274, 5858, 2788, 1771, 3313,
        #                           1785, 7419, 6912, 3329, 3331, 4872, 6929, 6932, 2325, 3864, 1308, 5916, 1826, 2339,
        #                           2866, 3893, 6455, 2362, 4418, 6979, 4943, 4432, 4944, 5478, 5479, 2419, 6515, 2934,
        #                           6015, 4996, 2949, 2961, 2451, 3475, 7096, 6075, 6587, 5063, 5066, 4044, 6615, 3549,
        #                           3552, 3045, 2535, 2538, 6125, 3568, 1010, 3575]
        # result (for Gas):
        # uncompleted_meter_ids = []
        for meter_id in uncompleted_meter_ids:
            for file in files:
                print("Removing meterID", meter_id, "from file", file)
                df = pd.read_csv(file)
                df = df[df.ID != meter_id]
                df.to_csv(file, index=False)

        customerClassification = "./ISSDA-CER/" + type + "/doc/customerClassification.csv"
        self.updateMeterID(customerClassification)

    def generateSortedFiles(self, files):
        for file in files:
            print("Sorting file:", file)
            sortedfile = "./ISSDA-CER/Electricity/data/data_original/sorted" + file[-9:]

            # Pandas instead of csv library: efficiency
            dset = pd.read_csv(file, sep=' ', names=['ID', 'DT', 'Usage'])

            # Sort by meterID and by timestamp
            dset = dset.sort_values(by=["ID", "DT"], ascending=True)

            # Store... Generate sorted file
            dset.to_csv(sortedfile, index=False, header=False, float_format='%.3f')  # 3 decimal points (even ".000")

    def generateWeekDataFiles(self, files):
        # Setting the name of the week files
        weekfiles = []
        iweek = []
        path = "./ISSDA-CER/Electricity/data/data_all/ElectricityDataWeek "
        firstday = int(first / 100)
        lastday = int(last / 100)
        nWeeks = int((lastday - firstday) / 7) + 1  # number of weeks
        for idx in range(nWeeks):
            weekfiles.append(path + str(idx))
            iweek.append(firstday + idx * 7)

        for file in files:
            print("Reading file: ", file)
            inp = open(file, "r")
            dset = csv.reader(inp, delimiter=',')
            out = []
            oset = []
            header = ["ID", "DT", "Usage"]
            for i in range(nWeeks):
                out.append(open(weekfiles[i], "a"))
                oset.append(csv.writer(out[i], delimiter=','))
                if file == files[0]:
                    oset[i].writerow(header)

            for row in dset:
                idweek = int(row[1][0:3])  # day identifier
                for i in range(nWeeks - 1):
                    if iweek[i] <= idweek < iweek[i + 1]:
                        oset[i].writerow(row)

            out[i].close()
        inp.close()

        # Removing this weeks
        os.remove(path + '76')
        os.remove(path + '36')

    def get_uncompleted_meter_ids(self, files):
        prev_meter_ids = set(pd.read_csv(files[0])['ID'])  # Initial file 0 has the 4733 meterIDs
        uncompleted_meter_ids = prev_meter_ids - set(self.df['ID'].unique())
        print("Uncompleted meter ids:", uncompleted_meter_ids)  # with len(uncompleted_meter_ids) = 4733 - 4626

        return uncompleted_meter_ids

    def update_df(self, files, firstWeek, lastWeek):
        df = pd.DataFrame()
        i = 1
        for file in files[firstWeek:lastWeek + 1]:
            print("Reading File: ", file)
            dset = pd.read_csv(file)  # load as a pd.dataFrame
            df = pd.concat([df, dset])
            if dset.shape != df.shape:
                groupID = df.groupby('ID').count()
                groupID = groupID[groupID.Usage == i * nObs]
                ID = groupID.index.to_numpy()
                groupID = pd.DataFrame(ID, columns=['ID'])
                # Filtering: Inner join to select just the complete meterIDs
                df = pd.merge(df, groupID, on='ID', how='inner')
            i += 1
        print("Dimension: ", df.groupby('ID').count().describe())
        self.df = df

    def updateMeterID(self, clfile):
        ID = self.df.groupby('ID').count().index.to_numpy()
        ID = pd.DataFrame(ID, columns=['ID'])
        clset = pd.read_csv(clfile, sep=';', index_col='ID')
        self.meterID = pd.merge(ID, clset, on='ID', how='inner')


def electricity_preprocessing():
    # data_original -> data_all
    data_original_dir = "./ISSDA-CER/Electricity/data/data_original/"
    data_all = "./ISSDA-CER/Electricity/data/data_all/"
    data_all_filtered = "./ISSDA-CER/Electricity/data/data_all_filtered/"

    # new dataAnalyzer object
    mg = dataAnalyzer()

    # Sort the files (can be removed)
    files = mg.getDataSetFiles(data_original_dir, "File")
    mg.generateSortedFiles(files)

    # Generate data week sorted files. They could be in a different directory. Just looking for similarity with Gas
    files = mg.getDataSetFiles(data_original_dir, "sortedFile")
    mg.generateWeekDataFiles(files)

    # data_all -> data_all_filtered
    files = mg.getDataSetFiles(data_all, "ElectricityDataWeek")
    customerClassification = "./ISSDA-CER/Electricity/doc/customerClassification.csv"
    mg.filterDataSetFirstRound(files, 0, len(files), customerClassification, "Electricity")

    mg.filterDataSetSecondRound(data_all_filtered, "Electricity", 0, 74)

    mg.meterID.to_csv("./script_results/meterID_electricity.csv", index=False)


def gas_preprocessing():
    # Gas hasn't got to sort the original data and generate the WeekDataFiles

    # data_original -> data_all
    data_original_dir = "./ISSDA-CER/Gas/data/data_original/"
    data_all = "./ISSDA-CER/Gas/data/data_all/"
    data_all_filtered = "./ISSDA-CER/Gas/data/data_all_filtered/"

    files = os.listdir(data_original_dir)
    for file in files:
        count = 0
        with open(data_original_dir + file, 'r') as f:
            for _ in f:
                count += 1
        # Copy this file to data all dir if it has enough meditions (501649)
        if count == 501649:
            # Delete rare characters (" ", " "" ") of the file and copy
            with open(data_original_dir + file, 'r') as infile, \
                    open(data_all + file, 'w') as outfile:
                data = infile.read()
                data = data.replace("\"", "")
                data = data.replace(" ", "")
                outfile.write(data)

    # new dataAnalyzer object
    mg = dataAnalyzer()

    # data_all -> data_all_filtered
    files = mg.getDataSetFiles(data_all, "GasDataWeek")
    customerClassification = "./ISSDA-CER/Gas/doc/customerClassification.csv"
    mg.filterDataSetFirstRound(files, 0, len(files), customerClassification, "Gas")

    # Gas doesn't remove any meterID in this step
    mg.filterDataSetSecondRound(data_all_filtered, "Gas", 0, 73)

    mg.meterID.to_csv("./script_results/meterID_gas.csv", index=False)


def check_preconditions():
    # Checking invocation parameters
    if len(sys.argv) != 2 or (sys.argv[1] != "Electricity" and sys.argv[1] != "Gas"):
        print("Usage: python3 issdacer.py <Electricity/Gas>")
        exit(85)

    # Preventing execution error (Electricity)
    electricity_data_directories_not_exists = not os.path.isdir('./ISSDA-CER/Electricity/data/data_original') or \
                                         not os.path.isdir('./ISSDA-CER/Electricity/data/data_all') or \
                                         not os.path.isdir('./ISSDA-CER/Electricity/data/data_all_filtered') or \
                                         not os.path.exists('./ISSDA-CER/Electricity/doc/customerClassification.csv')

    if electricity_data_directories_not_exists:
        print("The following directories are needed:")
        print("./ISSDA-CER/Electricity/data/data_original/")
        print("./ISSDA-CER/Electricity/data/data_all/")
        print("./ISSDA-CER/Electricity/data/data_all_filtered/")
        print("./ISSDA-CER/Electricity/doc/customerClassification.csv")
        exit(1)

    # Preventing execution error (Gas)
    gas_data_directories_not_exists = not os.path.isdir('./ISSDA-CER/Gas/data/data_original') or \
                                      not os.path.isdir('./ISSDA-CER/Gas/data/data_all') or \
                                      not os.path.isdir('./ISSDA-CER/Gas/data/data_all_filtered') or \
                                      not os.path.exists('./ISSDA-CER/Gas/doc/customerClassification.csv')

    if gas_data_directories_not_exists:
        print("The following directories are needed:")
        print("./ISSDA-CER/Gas/data/data_original/")
        print("./ISSDA-CER/Gas/data/data_all/")
        print("./ISSDA-CER/Gas/data/data_all_filtered/")
        print("./ISSDA-CER/Gas/doc/customerClassification.csv")
        exit(1)

    # Preventing bad behaviour with appending mode
    electricity_data_all_is_not_empty = len(os.listdir("./ISSDA-CER/Electricity/data/data_all/")) > 0
    if electricity_data_all_is_not_empty and sys.argv[1] == "Electricity":
        print("Remove all files of ./ISSDA-CER/Electricity/data/data_all/ before executing this script")
        exit(1)


# data_original -> data_all_filtered

# ELECTRICITY: data_original (from 6 to one per week) -> data_all (remove meterIDs without enough nObs)
#               -> data_all_filtered (remove some meterIDs that are not complete for ALL weeks) -> data_all_filtered

# GAS: data_original (remove files with less than 501649 obs) -> data_all (remove some meterIDs) -> data_all_filtered
if __name__ == '__main__':
    """
    args: 
    sys.argv[1]:type (Electricity, Gas)
    """

    check_preconditions()
    print("Preconditions OK, starting with the process\n")

    if sys.argv[1] == "Electricity":
        electricity_preprocessing()
    else:
        gas_preprocessing()
