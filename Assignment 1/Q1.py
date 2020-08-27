import math as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
