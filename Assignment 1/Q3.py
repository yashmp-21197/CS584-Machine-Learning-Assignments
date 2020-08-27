import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from sklearn.neighbors import KNeighborsClassifier

fraudData = pd.read_csv('Fraud.csv', delimiter=',')
fraudData = fraudData.dropna()
print(f'Fraud Data : \n {fraudData}')

fraudDataDescription = fraudData.describe()
print(f'Fraud Data Description: \n {fraudDataDescription}')
fraudDataDescriptionGroupbyFRAUD = fraudData.groupby('FRAUD').describe()
print(f'Fraud Data Description group by FRAUD: \n {fraudDataDescriptionGroupbyFRAUD}')

# totalSpendData = [fraudData[fraudData['FRAUD'] == 0]['TOTAL_SPEND'], fraudData[fraudData['FRAUD'] == 1]['TOTAL_SPEND']]
# plt.boxplot(totalSpendData, vert=False, labels=['fraud=0', 'fraud=1'])
# plt.title('boxplot for total_spend')
# plt.show()
#
# doctorVisitsData = [fraudData[fraudData['FRAUD'] == 0]['DOCTOR_VISITS'], fraudData[fraudData['FRAUD'] == 1]['DOCTOR_VISITS']]
# plt.boxplot(doctorVisitsData, vert=False, labels=['fraud=0', 'fraud=1'])
# plt.title('boxplot for doctor_Visits')
# plt.show()
#
# numClaimsData = [fraudData[fraudData['FRAUD'] == 0]['NUM_CLAIMS'], fraudData[fraudData['FRAUD'] == 1]['NUM_CLAIMS']]
# plt.boxplot(numClaimsData, vert=False, labels=['fraud=0', 'fraud=1'])
# plt.title('boxplot for num_claims')
# plt.show()
#
# memberDurationData = [fraudData[fraudData['FRAUD'] == 0]['MEMBER_DURATION'], fraudData[fraudData['FRAUD'] == 1]['MEMBER_DURATION']]
# plt.boxplot(memberDurationData, vert=False, labels=['fraud=0', 'fraud=1'])
# plt.title('boxplot for member_duration')
# plt.show()
#
# optomPrescData = [fraudData[fraudData['FRAUD'] == 0]['OPTOM_PRESC'], fraudData[fraudData['FRAUD'] == 1]['OPTOM_PRESC']]
# plt.boxplot(optomPrescData, vert=False, labels=['fraud=0', 'fraud=1'])
# plt.title('boxplot for optom_presc')
# plt.show()
#
# numMembersData = [fraudData[fraudData['FRAUD'] == 0]['NUM_MEMBERS'], fraudData[fraudData['FRAUD'] == 1]['NUM_MEMBERS']]
# plt.boxplot(numMembersData, vert=False, labels=['fraud=0', 'fraud=1'])
# plt.title('boxplot for num_members')
# plt.show()

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
# trainData = fraudData[['TOTAL_SPEND', 'DOCTOR_VISITS', 'NUM_CLAIMS', 'MEMBER_DURATION', 'OPTOM_PRESC', 'NUM_MEMBERS']]
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
    if (predictClass != targetData.iloc[i]):
        nMissClass += 1

print('number of missclassified data = ', nMissClass)
rateMissClass = nMissClass / trainData.shape[0]
print('Misclassification Rate = ', rateMissClass)
