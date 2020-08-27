# load necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

print("=" * 50)
print("=" * 50)
print("ML-Assignment 3-Question 1")
print("=" * 50)
print("=" * 50)
# Please provide information about your Data Partition step.

# load data of claim history
claim_history_data = pd.read_csv('claim_history.csv', delimiter=',')
claim_history_data = claim_history_data[["CAR_TYPE", "OCCUPATION", "EDUCATION", "CAR_USE"]].dropna()
data_shape = claim_history_data.shape
print(f'\nclaim history data :\n{claim_history_data}')
print(f'claim history data shape : {data_shape}')

# take partition of data into training and testing data
p_training = 0.7
p_testing = 0.3
claim_history_data_train, claim_history_data_test = train_test_split(claim_history_data, train_size=p_training,
                                                                     test_size=p_testing,
                                                                     random_state=27513, stratify=claim_history_data["CAR_USE"])

train_data = claim_history_data_train
test_data = claim_history_data_test
print('number of Observations in training data = ', train_data.shape[0])
print('number of Observations in testing data = ', test_data.shape[0])

print()
print("=" * 50)
print("ML-Assignment 3-Question 1-Section a)")
print("=" * 50)
# a)	(5 points). Please provide the frequency table (i.e., counts and proportions) of the target variable in the
# Training partition?

print(f'count of target variable in train data :\n{train_data.groupby("CAR_USE").size()}')
print(f'proportion of target variable in train data :\n {train_data.groupby("CAR_USE").size() / train_data.shape[0]}')

print()
print("=" * 50)
print("ML-Assignment 3-Question 1-Section b)")
print("=" * 50)
# b)	(5 points). Please provide the frequency table (i.e., counts and proportions) of the target variable in the
# Test partition?

print(f'count of target variable in test data :\n{test_data.groupby("CAR_USE").size()}')
print(f'proportion of target variable in test data :\n {test_data.groupby("CAR_USE").size() / test_data.shape[0]}')

print()
print("=" * 50)
print("ML-Assignment 3-Question 1-Section c)")
print("=" * 50)
# c)	(5 points). What is the probability that an observation is in the Training partition given that CAR_USE =
# Commercial?

p_com_given_training = train_data.groupby("CAR_USE").size()["Commercial"] / train_data.shape[0]
p_com_given_testing = test_data.groupby("CAR_USE").size()["Commercial"] / test_data.shape[0]
p_com = (p_com_given_training * p_training) + (p_com_given_testing * p_testing)
p_training_given_com = (p_com_given_training * p_training) / p_com
print(
    f'probability that an observation is in the Training partition given that CAR_USE = Commercial : {p_training_given_com}')

print()
print("=" * 50)
print("ML-Assignment 3-Question 1-Section d)")
print("=" * 50)
# d)	(5 points). What is the probability that an observation is in the Test partition given that CAR_USE = Private?

p_pri_given_testing = test_data.groupby("CAR_USE").size()["Private"] / test_data.shape[0]
p_pri_given_training = train_data.groupby("CAR_USE").size()["Private"] / train_data.shape[0]
p_pri = (p_pri_given_testing * p_testing) + (p_pri_given_training * p_training)
p_testing_given_pri = (p_pri_given_testing * p_testing) / p_pri
print(f'probability that an observation is in the Test partition given that CAR_USE = Private : {p_testing_given_pri}')
