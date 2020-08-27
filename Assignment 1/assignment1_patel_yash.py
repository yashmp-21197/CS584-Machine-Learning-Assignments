import math as m
import numpy as np
from numpy import linalg as la
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Question 1
print('<===============Question 1===============>')
normalDataSample = pd.read_csv('NormalSample.csv', delimiter=',', usecols=['group', 'x'], )
normalDataSample = normalDataSample.dropna()
print('NormalSample.csv :\n', normalDataSample)

normalDataSampleDescription = normalDataSample.describe()
print('Description of input data :\n', normalDataSampleDescription)

min_x = normalDataSampleDescription['x']['min']
max_x = normalDataSampleDescription['x']['max']
int_min_x = m.floor(min_x)
int_max_x = m.ceil(max_x)

binWidth = [0.1, 0.5, 1, 2]

for h in binWidth:
    print(f'bin-width={h}')
    Nh = h * normalDataSampleDescription['x']['count']
    currentMidPoint = int_min_x + (h / 2.0)
    for currentMidPoint in np.arange(int_min_x + (h / 2.0), int_max_x, h):
        p_midPoint = 0
        for x in normalDataSample['x']:
            w_x = (x - currentMidPoint) / h
            if -0.5 < w_x <= 0.5:
                p_midPoint += 1
        p_midPoint /= Nh
        print(f'({currentMidPoint} , {p_midPoint})')

    binCount = m.ceil((int_max_x - int_min_x) / h)
    print(f'Bin Count={binCount}')

    plt.hist(normalDataSample['x'], color='blue', bins=binCount)
    plt.title(f'Histogram of x in NormalSample for bin width h={h}')
    plt.xlabel('x in NormalSample')
    plt.ylabel('frequency of x')
    plt.show()

# Question  2
print('<===============Question 2===============>')
normalDataSample = pd.read_csv('NormalSample.csv', delimiter=',', usecols=['group', 'x'])
normalDataSample = normalDataSample.dropna()
normalDataSampleGroup0 = normalDataSample[normalDataSample['group'] == 0]
normalDataSampleGroup1 = normalDataSample[normalDataSample['group'] == 1]
print('NormalSample.csv :\n', normalDataSample)
print('NormalSample.csv where group=0 :\n', normalDataSampleGroup0)
print('NormalSample.csv where group=1 :\n', normalDataSampleGroup1)

normalDataSampleDescription = normalDataSample.describe()  # For question 2 section a and c
normalDataSampleDescriptionGroupWise = normalDataSample.groupby(by='group').describe()  # For question 2 section b and d
print('Description of input data:\n', normalDataSampleDescription)
print('Description of input data group wise:\n', normalDataSampleDescriptionGroupWise)

# For question 2 section a and c
plt.boxplot(normalDataSample['x'], vert=False, labels=['x'])
plt.show()

# For question 2 section b and d
combinedData = [normalDataSample['x'], normalDataSampleGroup0['x'], normalDataSampleGroup1['x']]
boxPlot = plt.boxplot(combinedData, vert=True, labels=['x', 'x:group=0', 'x:group=1'])
plt.show()

# Question 3
print('<===============Question 3===============>')
fraudData = pd.read_csv('Fraud.csv', delimiter=',')
fraudData = fraudData.dropna()
print(f'Fraud Data : \n {fraudData}')

fraudDataDescription = fraudData.describe()
print(f'Fraud Data Description: \n {fraudDataDescription}')
fraudDataDescriptionGroupbyFRAUD = fraudData.groupby('FRAUD').describe()
print(f'Fraud Data Description group by FRAUD: \n {fraudDataDescriptionGroupbyFRAUD}')

totalSpendData = [fraudData[fraudData['FRAUD'] == 0]['TOTAL_SPEND'], fraudData[fraudData['FRAUD'] == 1]['TOTAL_SPEND']]
plt.boxplot(totalSpendData, vert=False, labels=['fraud=0', 'fraud=1'])
plt.title('boxplot for total_spend')
plt.show()

doctorVisitsData = [fraudData[fraudData['FRAUD'] == 0]['DOCTOR_VISITS'],
                    fraudData[fraudData['FRAUD'] == 1]['DOCTOR_VISITS']]
plt.boxplot(doctorVisitsData, vert=False, labels=['fraud=0', 'fraud=1'])
plt.title('boxplot for doctor_Visits')
plt.show()

numClaimsData = [fraudData[fraudData['FRAUD'] == 0]['NUM_CLAIMS'], fraudData[fraudData['FRAUD'] == 1]['NUM_CLAIMS']]
plt.boxplot(numClaimsData, vert=False, labels=['fraud=0', 'fraud=1'])
plt.title('boxplot for num_claims')
plt.show()

memberDurationData = [fraudData[fraudData['FRAUD'] == 0]['MEMBER_DURATION'],
                      fraudData[fraudData['FRAUD'] == 1]['MEMBER_DURATION']]
plt.boxplot(memberDurationData, vert=False, labels=['fraud=0', 'fraud=1'])
plt.title('boxplot for member_duration')
plt.show()

optomPrescData = [fraudData[fraudData['FRAUD'] == 0]['OPTOM_PRESC'], fraudData[fraudData['FRAUD'] == 1]['OPTOM_PRESC']]
plt.boxplot(optomPrescData, vert=False, labels=['fraud=0', 'fraud=1'])
plt.title('boxplot for optom_presc')
plt.show()

numMembersData = [fraudData[fraudData['FRAUD'] == 0]['NUM_MEMBERS'], fraudData[fraudData['FRAUD'] == 1]['NUM_MEMBERS']]
plt.boxplot(numMembersData, vert=False, labels=['fraud=0', 'fraud=1'])
plt.title('boxplot for num_members')
plt.show()

print("Number of Dimensions = ", fraudData.ndim)
print("Number of Rows = ", np.size(fraudData, 0))
print("Number of Columns = ", np.size(fraudData, 1))

x = np.matrix(
    fraudData[['TOTAL_SPEND', 'DOCTOR_VISITS', 'NUM_CLAIMS', 'MEMBER_DURATION', 'OPTOM_PRESC', 'NUM_MEMBERS']].values)
xtx = x.transpose() * x
print("t(x) * x = \n", xtx)

# Eigenvalue decomposition
evals, evecs = la.eigh(xtx)
print("Eigenvalues of x = \n", evals)
print("Eigenvectors of x = \n", evecs)

# transformation matrix
transf = evecs * la.inv(np.sqrt(np.diagflat(evals)))
print("Transformation Matrix = \n", transf)

# Here is the transformed X
transf_x = x * transf
print("The Transformed x = \n", transf_x)

# Check columns of transformed X
xtx = transf_x.transpose() * transf_x
print("Expect an Identity Matrix = \n", xtx)

trainData = transf_x
targetData = fraudData['FRAUD']

knn = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='euclidean', )
nbrs = knn.fit(trainData, targetData)
accuracy = nbrs.score(trainData, targetData)
print('Accuracy of the model is ', accuracy)

# See the classification probabilities
testData = [7500, 15, 3, 127, 2, 2]
testData = testData * transf
print('transformed test data =\n', testData)
knbrs = nbrs.kneighbors(testData)
print('5 nearest neighbors = \n', knbrs)
predict_fraud = nbrs.predict(testData)
print('predicted fraud = ', predict_fraud)
# See the classification probabilities
fraud_prob = nbrs.predict_proba(testData)
print('fraud probability = ', fraud_prob)

# Calculate the Misclassification Rate
class_prob = nbrs.predict_proba(trainData)
targetClass = [0, 1]

nMissClass = 0
for i in range(trainData.shape[0]):
    j = np.argmax(class_prob[i][:])
    predictClass = targetClass[j]
    if predictClass != targetData.iloc[i]:
        nMissClass += 1

print('number of missclassified data = ', nMissClass)
rateMissClass = nMissClass / trainData.shape[0]
print('Misclassification Rate = ', rateMissClass)
