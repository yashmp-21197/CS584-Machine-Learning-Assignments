import pandas as pd
import matplotlib.pyplot as plt

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
# plt.boxplot(normalDataSample['x'], vert=False, labels=['x'])
# plt.show()


# For question 2 section b and d
combinedData = [normalDataSample['x'], normalDataSampleGroup0['x'], normalDataSampleGroup1['x']]

boxPlot = plt.boxplot(combinedData, vert=True, labels=['x', 'x:group=0', 'x:group=1'])
plt.show()
