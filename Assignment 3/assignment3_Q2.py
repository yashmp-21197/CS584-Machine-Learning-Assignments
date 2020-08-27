# load necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from itertools import combinations
import math
import warnings
warnings.filterwarnings("ignore")


# function that calculate entropy for ordinal or interval variables
def EntropyOrdinalSplit(in_data, split):
    data_table = in_data
    data_table['LE_Split'] = (data_table.iloc[:, 0] <= split)

    cross_table = pd.crosstab(index=data_table['LE_Split'], columns=data_table.iloc[:, 1], margins=True, dropna=True)
    # print(cross_table)

    n_rows = cross_table.shape[0]
    n_columns = cross_table.shape[1]

    table_entropy = 0
    for i_row in range(n_rows - 1):
        row_entropy = 0
        for i_column in range(n_columns):
            proportion = cross_table.iloc[i_row, i_column] / cross_table.iloc[i_row, (n_columns - 1)]
            if proportion > 0:
                row_entropy -= proportion * np.log2(proportion)
        # print('Row = ', i_row, 'Entropy =', row_entropy)
        # print(' ')
        table_entropy += row_entropy * cross_table.iloc[i_row, (n_columns - 1)]
    table_entropy = table_entropy / cross_table.iloc[(n_rows - 1), (n_columns - 1)]

    return cross_table, table_entropy


# function that check all the posibilities in ordinal or interval variables and return possibility that has smallest
# entropy
def FindMinOrdinalEntropy(in_data, set_intervals):
    min_entropy = sys.float_info.max
    min_interval = None
    min_table = None

    for i in range(set_intervals[0], set_intervals[len(set_intervals) - 1]):
        ret_table, ret_entropy = EntropyOrdinalSplit(in_data=in_data, split=i + 0.5)
        if ret_entropy < min_entropy:
            min_entropy = ret_entropy
            min_interval = i + 0.5
            min_table = ret_table

    return min_table, min_entropy, min_interval


# function that calculate entropy for nominal variables
def EntropyNominalSplit(in_data, subset):
    data_table = in_data
    data_table['LE_Split'] = data_table.iloc[:, 0].apply(lambda x: True if x in subset else False)

    cross_table = pd.crosstab(index=data_table['LE_Split'], columns=data_table.iloc[:, 1], margins=True, dropna=True)
    # print(cross_table)

    n_rows = cross_table.shape[0]
    n_columns = cross_table.shape[1]

    table_entropy = 0
    for i_row in range(n_rows - 1):
        row_entropy = 0
        for i_column in range(n_columns):
            proportion = cross_table.iloc[i_row, i_column] / cross_table.iloc[i_row, (n_columns - 1)]
            if proportion > 0:
                row_entropy -= proportion * np.log2(proportion)
        # print('Row = ', i_row, 'Entropy =', row_entropy)
        # print(' ')
        table_entropy += row_entropy * cross_table.iloc[i_row, (n_columns - 1)]
    table_entropy = table_entropy / cross_table.iloc[(n_rows - 1), (n_columns - 1)]

    return cross_table, table_entropy


# function that create all the possible combinations for nominal variables and return possibility that has smallest
# entropy
def FindMinNominalEntropy(in_data, set):
    subset_map = {}
    for i in range(1, (int(len(set) / 2)) + 1):
        subsets = combinations(set, i)
        for ss in subsets:
            remaining = tuple()
            for ele in set:
                if ele not in ss:
                    remaining += (ele,)
            if subset_map.get(remaining) == None:
                subset_map[ss] = remaining

    min_entropy = sys.float_info.max
    min_subset1 = None
    min_subset2 = None
    min_table = None

    for subset in subset_map:
        ret_table, ret_entropy = EntropyNominalSplit(in_data=in_data, subset=subset)
        if ret_entropy < min_entropy:
            min_entropy = ret_entropy
            min_subset1 = subset
            min_subset2 = subset_map.get(subset)
            min_table = ret_table

    return min_table, min_entropy, min_subset1, min_subset2


print("=" * 50)
print("=" * 50)
print("ML-Assignment 3-Question 2")
print("=" * 50)
print("=" * 50)

# loading data from data file
claim_history_data = pd.read_csv('claim_history.csv', delimiter=',')
claim_history_data = claim_history_data[["CAR_TYPE", "OCCUPATION", "EDUCATION", "CAR_USE"]].dropna()
data_shape = claim_history_data.shape

# map ordinal variable with numeric data
claim_history_data['EDUCATION'] = claim_history_data['EDUCATION'].map(
    {'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})
# print(claim_history_data)

# splitting data into training and testing part
p_training, p_testing = 0.7, 0.3
claim_history_data_train, claim_history_data_test = train_test_split(claim_history_data, train_size=p_training,
                                                                     test_size=p_testing, random_state=27513)
train_data = claim_history_data_train
test_data = claim_history_data_test

print()
print("=" * 50)
print("ML-Assignment 3-Question 2-Section a)")
print("=" * 50)
# a)	(5 points). What is the entropy value of the root node?

p_com_train = train_data.groupby('CAR_USE').size()['Commercial'] / train_data.shape[0]
p_pri_train = train_data.groupby('CAR_USE').size()['Private'] / train_data.shape[0]

root_entropy = -((p_com_train * math.log2(p_com_train)) + (p_pri_train * math.log2(p_pri_train)))
print(f'root node entropy : {root_entropy}')

print()
print("=" * 50)
print("ML-Assignment 3-Question 2-Section b)")
print("=" * 50)
# b)	(5 points). What is the split criterion (i.e., predictor name and values in the two branches) of the first layer?

# for layer 0 split
cross_table_edu, table_entropy_edu, interval_edu = FindMinOrdinalEntropy(in_data=train_data[['EDUCATION', 'CAR_USE']],
                                                                         set_intervals=[0, 1, 2, 3, 4])
print(
    f'layer0-education:\n '
    f'cross table: \n{cross_table_edu}\n '
    f'entropy: {table_entropy_edu}\n '
    f'split interval: {interval_edu}\n')

cross_table_car_type, table_entropy_car_type, subset1_car_type, subset2_car_type = FindMinNominalEntropy(
    in_data=train_data[['CAR_TYPE', 'CAR_USE']], set=['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van'])
print(
    f'layer0-car-type:\n '
    f'cross table: \n{cross_table_car_type}\n '
    f'entropy: {table_entropy_car_type}\n '
    f'left subset: {subset1_car_type}\n '
    f'right subset: {subset2_car_type}\n')

cross_table_occu, table_entropy_occu, subset1_occu, subset2_occu = FindMinNominalEntropy(
    in_data=train_data[['OCCUPATION', 'CAR_USE']],
    set=['Blue Collar', 'Clerical', 'Doctor', 'Home Maker', 'Lawyer', 'Manager', 'Professional', 'Student', 'Unknown'])
print(
    f'layer0-occupation:\n '
    f'cross table: \n{cross_table_occu}\n '
    f'entropy: {table_entropy_occu}\n '
    f'left subset: {subset1_occu}\n '
    f'right subset: {subset2_occu}\n')

print(f'split criterion for first layer')
print(f'predictor name: OCCUPATION')
print(f'predictor value:\n left subset: {subset1_occu}\n right subset: {subset2_occu}')

print()
print("=" * 50)
print("ML-Assignment 3-Question 2-Section c)")
print("=" * 50)
# c)	(10 points). What is the entropy of the split of the first layer?

# for layer 1 left node split
train_data_left_branch = train_data[train_data['OCCUPATION'].isin(subset1_occu)]

layer1_cross_table_edu, layer1_table_entropy_edu, layer1_interval_edu = FindMinOrdinalEntropy(
    in_data=train_data_left_branch[['EDUCATION', 'CAR_USE']], set_intervals=[0, 1, 2, 3, 4])
print(
    f'layer1-left-node-education:\n cross table: \n{layer1_cross_table_edu}\n entropy: {layer1_table_entropy_edu}\n '
    f'split interval: {layer1_interval_edu}\n')

layer1_cross_table_car_type, layer1_table_entropy_car_type, layer1_subset1_car_type, layer1_subset2_car_type = FindMinNominalEntropy(
    in_data=train_data_left_branch[['CAR_TYPE', 'CAR_USE']],
    set=['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van'])
print(
    f'layer1-left-node-car-type:\n '
    f'cross table: \n{layer1_cross_table_car_type}\n '
    f'entropy: {layer1_table_entropy_car_type}\n '
    f'left subset: {layer1_subset1_car_type}\n '
    f'right subset: {layer1_subset2_car_type}\n')

layer1_cross_table_occu, layer1_table_entropy_occu, layer1_subset1_occu, layer1_subset2_occu = FindMinNominalEntropy(
    in_data=train_data_left_branch[['OCCUPATION', 'CAR_USE']], set=subset1_occu)
print(
    f'layer1-left-node-occupation:\n '
    f'cross table: \n{layer1_cross_table_occu}\n '
    f'entropy: {layer1_table_entropy_occu}\n '
    f'left subset: {layer1_subset1_occu}\n '
    f'right subset: {layer1_subset2_occu}\n')

# for layer 1 right node split
train_data_right_branch = train_data[train_data['OCCUPATION'].isin(subset2_occu)]

layer1_cross_table_edu, layer1_table_entropy_edu, layer1_interval_edu = FindMinOrdinalEntropy(
    in_data=train_data_right_branch[['EDUCATION', 'CAR_USE']], set_intervals=[0, 1, 2, 3, 4])
print(
    f'layer1-right-node-education:\n '
    f'cross table: \n{layer1_cross_table_edu}\n '
    f'entropy: {layer1_table_entropy_edu}\n '
    f'split interval: {layer1_interval_edu}\n')

layer1_cross_table_car_type, layer1_table_entropy_car_type, layer1_subset1_car_type, layer1_subset2_car_type = FindMinNominalEntropy(
    in_data=train_data_right_branch[['CAR_TYPE', 'CAR_USE']],
    set=['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van'])
print(
    f'layer1-right-node-car-type:\n '
    f'cross table: \n{layer1_cross_table_car_type}\n '
    f'entropy: {layer1_table_entropy_car_type}\n '
    f'left subset: {layer1_subset1_car_type}\n '
    f'right subset: {layer1_subset2_car_type}\n')

layer1_cross_table_occu, layer1_table_entropy_occu, layer1_subset1_occu, layer1_subset2_occu = FindMinNominalEntropy(
    in_data=train_data_right_branch[['OCCUPATION', 'CAR_USE']], set=subset2_occu)
print(
    f'layer1-left-node-occupation:\n '
    f'cross table: \n{layer1_cross_table_occu}\n '
    f'entropy: {layer1_table_entropy_occu}\n '
    f'left subset: {layer1_subset1_occu}\n '
    f'right subset: {layer1_subset2_occu}\n')

print(
    f'entropy of the split of the first layer:\n '
    f'for left node: {layer1_table_entropy_edu}\n '
    f'for right node: {layer1_table_entropy_car_type}\n')

print()
print("=" * 50)
print("ML-Assignment 3-Question 2-Section d,e)")
print("=" * 50)
# d)	(5 points). How many leaves?
print(f'there are four leaves')

# e)	(15 points). Describe all your leaves.  Please include the decision rules and the counts of the target values.

# data of leave 1
train_data_left_left_branch = train_data_left_branch[train_data_left_branch['EDUCATION'] <= layer1_interval_edu]
total_count_l1 = train_data_left_left_branch.shape[0]
com_count_l1 = train_data_left_left_branch[train_data_left_left_branch['CAR_USE'] == 'Commercial'].shape[0]
pri_count_l1 = train_data_left_left_branch[train_data_left_left_branch['CAR_USE'] == 'Private'].shape[0]
p_com_l1 = com_count_l1 / total_count_l1
p_pri_l1 = pri_count_l1 / total_count_l1
entropy_l1 = -((p_com_l1 * math.log2(p_com_l1)) + (p_pri_l1 * math.log2(p_pri_l1)))
class_l1 = 'Commercial' if com_count_l1 > pri_count_l1 else 'Private'
print(
    f'leave 1:\n entropy: {entropy_l1}\n total count: {total_count_l1}\n commercial count: {com_count_l1}\n '
    f'private count: {pri_count_l1}\n commercial probability: {p_com_l1}\n private probability: {p_pri_l1}\n '
    f'class: {class_l1}\n')

# data of leave 2
train_data_right_left_branch = train_data_left_branch[train_data_left_branch['EDUCATION'] > layer1_interval_edu]
total_count_l2 = train_data_right_left_branch.shape[0]
com_count_l2 = train_data_right_left_branch[train_data_right_left_branch['CAR_USE'] == 'Commercial'].shape[0]
pri_count_l2 = train_data_right_left_branch[train_data_right_left_branch['CAR_USE'] == 'Private'].shape[0]
p_com_l2 = com_count_l2 / total_count_l2
p_pri_l2 = pri_count_l2 / total_count_l2
entropy_l2 = -((p_com_l2 * math.log2(p_com_l2)) + (p_pri_l2 * math.log2(p_pri_l2)))
class_l2 = 'Commercial' if com_count_l2 > pri_count_l2 else 'Private'
print(
    f'leave 2:\n entropy: {entropy_l2}\n total count: {total_count_l2}\n commercial count: {com_count_l2}\n '
    f'private count: {pri_count_l2}\n commercial probability: {p_com_l2}\n private probability: {p_pri_l2}\n '
    f'class: {class_l2}\n')

# data of leave 3
train_data_left_right_branch = train_data_right_branch[
    train_data_right_branch['CAR_TYPE'].isin(layer1_subset1_car_type)]
total_count_l3 = train_data_left_right_branch.shape[0]
com_count_l3 = train_data_left_right_branch[train_data_left_right_branch['CAR_USE'] == 'Commercial'].shape[0]
pri_count_l3 = train_data_left_right_branch[train_data_left_right_branch['CAR_USE'] == 'Private'].shape[0]
p_com_l3 = com_count_l3 / total_count_l3
p_pri_l3 = pri_count_l3 / total_count_l3
entropy_l3 = -((p_com_l3 * math.log2(p_com_l3)) + (p_pri_l3 * math.log2(p_pri_l3)))
class_l3 = 'Commercial' if com_count_l3 > pri_count_l3 else 'Private'
print(
    f'leave 3:\n entropy: {entropy_l3}\n total count: {total_count_l3}\n commercial count: {com_count_l3}\n '
    f'private count: {pri_count_l3}\n commercial probability: {p_com_l3}\n private probability: {p_pri_l3}\n '
    f'class: {class_l3}\n')

# data of leave 4
train_data_right_right_branch = train_data_right_branch[
    train_data_right_branch['CAR_TYPE'].isin(layer1_subset2_car_type)]
total_count_l4 = train_data_right_right_branch.shape[0]
com_count_l4 = train_data_right_right_branch[train_data_right_right_branch['CAR_USE'] == 'Commercial'].shape[0]
pri_count_l4 = train_data_right_right_branch[train_data_right_right_branch['CAR_USE'] == 'Private'].shape[0]
p_com_l4 = com_count_l4 / total_count_l4
p_pri_l4 = pri_count_l4 / total_count_l4
entropy_l4 = -((p_com_l4 * math.log2(p_com_l4)) + (p_pri_l4 * math.log2(p_pri_l4)))
class_l4 = 'Commercial' if com_count_l4 > pri_count_l4 else 'Private'
print(
    f'leave 4:\n entropy: {entropy_l4}\n total count: {total_count_l4}\n commercial count: {com_count_l4}\n '
    f'private count: {pri_count_l4}\n commercial probability: {p_com_l4}\n private probability: {p_pri_l4}\n '
    f'class: {class_l4}\n')
