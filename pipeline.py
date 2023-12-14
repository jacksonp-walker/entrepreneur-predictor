import os
import time
from multiprocessing import Process
from multiprocessing import Queue
import concurrent.futures as cf

import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport

import data_groups




# ---------------------- FILE MANAGEMENT -------------------------- #

def open_csv_file(filename):
    try:
        return pd.read_csv('{}/data/{}.csv'.format(os.getcwd(), filename))

    except Exception as e:
        print(".{} file not found".format(filename))
        print(e)


def open_stata_file(filename):
    try:
        return pd.read_stata('{}/data/{}.dta'.format(os.getcwd(), filename), convert_categoricals=False)

    except Exception as e:
        print(".{} file not found".format(filename))
        print(e)


def to_csv(df, filename):
    df.to_csv('{}/data/{}.csv'.format(os.getcwd(), filename))
    print(filename + '.csv successfully saved')

# --------- PREP --------- #
def prep(dfs):
    # Drop columns
    # keep minsamp 4 from org1 and minsamp 8 from org2
    # create total_id var
    # sort by total_id
    # return df

    print('Merging Years...\n')
    # Drop the drop_columns from the dataframes
    for i in range(0, len(dfs)):
        dfs[i] = dfs[i].drop(data_groups.drop_columns, axis=1)
    # narrow down to month 4 in first file and month 8 in second file
    dfs[0] = dfs[0].loc[dfs[0]['minsamp'] == 4]
    dfs[1] = dfs[1].loc[dfs[1]['minsamp'] == 8]
    # combine the data from the two files
    df = pd.concat(dfs)
    # add total id
    print('Adding total_id...\n')
    df['total_id'] = df['hhid'] + df['hhid2']
    # sort by total id
    df = df.sort_values(by='total_id')
    return df

def check_48(x):
    return 4 in x and 8 in x

def pm_age(x):
    age1 = x[0]
    age2 = x[1]
    if (18 <= age1 <= 65):
        return abs(age1 - age2) <= 2
    return False

# ----- Feature Engineering ----- #
def se_change(cow):
    self_employed_codes = [6, 7]
    return 1 if (cow[0] not in self_employed_codes and cow[1] in self_employed_codes) else 0

def new_se(job1, job2):
    # checks if Self-Employment status (in either first or second job) changes from NO in minsamp 4 to YES minsamp 8
    # If change occurs, then 1, else 0
    return 1 if (job1 == 1 or job2 == 1) else 0

def se(m, cow1, cow2):
    # Checks if Self-Employed (in either first or second job) for a specified month in sample m
    # If self-employed then 1, else 0
    self_employed_codes = [6, 7]
    return 1 if (cow1[m] in self_employed_codes or cow2[m] in self_employed_codes) else 0

def new_se_iu(code, cow1, cow2):
    # Note if they become self-employed not incorporated or incorporated based on code given
    # 6: incorporated
    # 7: not incorporated
    self_employed_codes = [code]
    self_employed_job1 = cow1[0] not in self_employed_codes and cow1[1] in self_employed_codes
    self_employed_job2 = cow2[0] not in self_employed_codes and cow2[1] in self_employed_codes
    return 1 if (self_employed_job1 or self_employed_job2) else 0

def first_gen(m, f):
    # if either parent was not born in US, considered first generation
    return 1 if (m[0] != 57 or f[0] != 57) else 0

def full_time(uhourse):
    # Full-time (1) if usual hours main job is >= 40 hours per week, else part-time (0)
    if np.isnan(uhourse[0]):
        return uhourse[0]
    return 1.0 if uhourse[0] >= 40.0 else 0.0

def college(col):
    # 1 if above "Associate degree-occupational/vocational", 0 if below
    return 1 if col[0] >= 13 else 0

def children(num):
    if np.isnan(num[0]):
        return num[0]
    return 1.0 if num[0] >= 1.0 else 0.0


def get_minsamp_x(x, col):
    return col[x]

class TestData:
    def __init__(self, df):
        self.df = df

    def get_df(self):
        return self.df
    def group(self):
        # Group individuals based on 'total_id' 'hhnum' 'wbhaom' 'female' and 'lineno, then confirm 4 and 8 in minsamp
        # Confirm matches by checking that age changed by no more than 2 years from minsamp 4 to minsamp 8
        t1 = time.perf_counter()

        # Group by ['total_id', 'hhnum', 'wbhaom', 'female', 'lineno']
        print('Grouping inidividuals by total_id, hhnum, wbhaom, female, and lineno...')
        self.df = self.df.groupby(['total_id', 'hhnum', 'wbhaom', 'female', 'lineno']).agg(list)
        print('Grouping complete!\n')

        # Check for minsamp 4 and 8 record

        self.df = self.df.loc[np.vectorize(check_48)(self.df['minsamp'])]
        print('Minsamp 4 and 8 check complete!\n')

        # Check if age within 2 years and that 18 <= age <= 65
        self.df = self.df.loc[np.vectorize(pm_age)(self.df['age'])]
        print('Age check complete!\n')

        t2 = time.perf_counter()
        print(f'Finished in {t2 - t1} seconds\n')

        print('GROUPED DATAFRAME')
        print(self.df)

    def new_features(self):
        t1 = time.perf_counter()

        self.df['first_gen'] = np.vectorize(first_gen)(self.df['pemntvty'], self.df['pefntvty'])
        print('first_gen added')

        # Full time (1) or part time (0)
        self.df['full_time'] = np.vectorize(full_time)(self.df['uhourse'])
        print('full_time added')

        # College (1) or less than college (0)?
        self.df['college'] = np.vectorize(college)(self.df['educ92'])
        print('college added')

        # Children present (1) or children not present (0)
        self.df['children'] = np.vectorize(children)(self.df['ownchild'])
        print('children added')

        # if SE changed in Job 1 or Job 2
        self.df['new_se_job1'] = np.vectorize(se_change)(self.df['cow1'])
        self.df['new_se_job2'] = np.vectorize(se_change)(self.df['cow2'])
        print('new_se_job1 and new_se_job2 added')

        # Create 'new_se' var (if change to SE from minsamp 4 to 8)
        self.df['new_se'] = np.vectorize(new_se)(self.df['new_se_job1'], self.df['new_se_job2'])
        print('new_se added')

        # Create 'se4' var (if SE in minsamp 4)
        self.df['se4'] = np.vectorize(se)(0, self.df['cow1'], self.df['cow2'])
        print('se4 added')

        # Create 'se8' var (if SE in minsamp 8)
        self.df['se8'] = np.vectorize(se)(1, self.df['cow1'], self.df['cow2'])
        print('se8 added')

        # Whether they are new SE inc. or new SE uninc.
        self.df['new_se_i'] = np.vectorize(new_se_iu)(6, self.df['cow1'], self.df['cow2'])
        self.df['new_se_u'] = np.vectorize(new_se_iu)(7, self.df['cow1'], self.df['cow2'])
        print('new_se_i and new_se_u added\n')

        t2 = time.perf_counter()
        print(f'Finished in {t2 - t1} seconds\n')

        print('NEW FEATURES ADDED')
        print(self.df)

    def keep_4(self, column):
        columns8 = ['pdemp1', 'pdemp2', 'nmemp1', 'nmemp2', 'ind_2d',
                    'ind14', 'ind_m03', 'docc03', 'occ12', 'occ_m03']
        if column in columns8:
            # Save minsamp 8 values for ^^ columns
            self.df[column + '_8'] = np.vectorize(get_minsamp_x)(1, self.df[column])
        # Save only the month 4 value
        self.df[column] = np.vectorize(get_minsamp_x)(0, self.df[column])

    def make_categorical(self):
        for col in data_groups.categorical:
            self.df[col] = np.vectorize(str)(self.df[col])
        self.df = pd.get_dummies(self.df)
        print(self.df)

    def make_pairs(self):
        print(self.df)
        self.group()
        self.new_features()
        columns = self.df.columns.values[:-11]
        for column in columns:
            self.keep_4(column)
            print(column + ' complete.')
        to_csv(self.df, 'test_pipeline_1819')

        
def pipline(data_years):
    combined_datasets = []
    for year in data_years:		 # for each pair of files
        y1 = [year[0]]
        y2 = [year[1]]
        
		# Open raw data files
        df1 = open_stata_file(f'cepr_org_20{y1}')
        df2 = open_stata_file(f'cepr_org_20{y2}')
        
        df1 = df1.loc[df1['minsamp'] == 4]		# get month 4 responses
        df2 = df2.loc[df2['minsamp'] == 8]		# get month 8 responses
        
        df = pd.concat([df1, df2])				# combine 2 years of data
        
		# MATCH RESPONDENTS ACROSS YEARS
        
        df = df.groupby(['total_id', 'hhnum', 'wbhaom', 'female', 'lineno']).agg(list)	# Group by ['total_id', 'hhnum', 'wbhaom', 'female', 'lineno']
        print('Grouping complete!\n')

        # Check for minsamp 4 and 8 record
        df = df.loc[np.vectorize(check_48)(df['minsamp'])]
        print('Minsamp 4 and 8 check complete!\n')

        # Check if age within 2 years and that 18 <= age <= 65
        self.df = self.df.loc[np.vectorize(pm_age)(self.df['age'])]
        print('Age check complete!\n')

		# combine files
			# take month 4 samples from first file and month 8 samples from second file
			# merge these dfs
			# groupby...
			# check for minsamp 4 and 8 records
			# add to list of combined dfs
		# match individuals in file
		# Drop columns
		# keep minsamp 4 from org1 and minsamp 8 from org2
		# create total_id var
		# sort by total_id
		# return df

files = [[14, 15], [15, 16], [16, 17], [17, 18], [18, 19]]
processed_datasets = []
for file in files:
	df1 = open_stata_file(f'cepr_org_20{file[0]}')
	df2 = open_stata_file(f'cepr_org_20{file[1]}')
	processed_datasets.append(prep([df1, df2]))

# def sort_minsamp(id):
#     return (df.loc[df['total_id'] == id]).sort_values(by='minsamp')

# if __name__ == '__main__':
#     print('Sorting by total_id and minsamp...\n')
#     ids = df['total_id'].unique()
#     with cf.ProcessPoolExecutor() as executor:
#         results = executor.map(sort_minsamp, ids)
#         df = pd.concat(results)
#         print('Sorting complete!\n')
#         print(df)

#     test_data = TestData(df)
#     test_data.make_pairs()

df = open_csv_file('test_pipeline_1718')
df = df.loc[df['new_se'] == 0]
df = df
# test_data = TestData(df)
# test_data.make_categorical()
# print(test_data.get_df())
# to_csv(test_data.get_df(), 'org1718_training_dummies')

# df = open_csv_file('test_pipeline_1617')
# print(df)
# print(df['new_se'].value_counts())
#
# df1 = open_csv_file('test_pipeline_1516')
# print(df1)
# print(df1['new_se'].value_counts())
#
# df2 = open_csv_file('org1819_training_dummies')
# print(df2)
# print(df2['new_se'].value_counts())

# df = open_csv_file('test_pipeline_1516')
# test_data = TestData(df)
# test_data.make_categorical()
# df = test_data.get_df()
# to_csv(df, 'org1516_training_dummies')



# orgs = prep(orgs)
# to_csv(prep(orgs), 'org1819')