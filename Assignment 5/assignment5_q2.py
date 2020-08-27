# load necessary libraries
import pandas
import numpy
import sklearn.metrics as metrics
import sklearn.svm as svm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

print()
print("=" * 50)
print("=" * 50)
print("ML-Assignment 5-Question 2")
print("=" * 50)
print("=" * 50)
# Please answer the following questions based on your model.
# load data from csv file
spiral_with_cluster = pandas.read_csv('SpiralWithCluster.csv', delimiter=',', usecols=['x', 'y', 'SpectralCluster'])

# build svm classifier
x_train = spiral_with_cluster[['x', 'y']]
y_train = spiral_with_cluster['SpectralCluster']

svm_model = svm.SVC(kernel='linear', decision_function_shape='ovr', random_state=20191108, max_iter=-1)
this_fit = svm_model.fit(x_train, y_train)

print()
print("=" * 50)
print("ML-Assignment 5-Question 2-Section a)")
print("=" * 50)
# a)	(5 points) What is the equation of the separating hyperplane?  Please state the coefficients up to seven
# decimal places.
# equation is 𝑤_0+𝐰_1*𝐱_1+w_2*x_2=𝟎
# 𝑤_0 = intercept
# w = coefficients
print(f'Intercept = {this_fit.intercept_}')
print(f'Coefficients = {numpy.round(this_fit.coef_, 7)}')
print(
    f'Equation of the separating hyperplane is "𝑤_0+𝐰_1*𝐱_1+w_2*x_2=𝟎"  ==> '
    f'({numpy.round(this_fit.intercept_[0], 7)}) '
    f'+ ({numpy.round(this_fit.coef_[0][0], 7)}*x_1) '
    f'+ ({numpy.round(this_fit.coef_[0][1], 7)}*x_2) = 𝟎')

print()
print("=" * 50)
print("ML-Assignment 5-Question 2-Section b)")
print("=" * 50)
# b)	(5 points) What is the misclassification rate?
y_predict_class = this_fit.predict(x_train)
# calculating accuracy and misclassification rate
accuracy = metrics.accuracy_score(y_train, y_predict_class)
misclassification_rate = 1 - accuracy
print(f'Accuracy of the SVM is {accuracy}')
print(f'Misclassification rate of the SVM is {misclassification_rate}')

print()
print("=" * 50)
print("ML-Assignment 5-Question 2-Section c)")
print("=" * 50)
# c)	(5 points) Please plot the y-coordinate against the x-coordinate in a scatterplot.  Please color-code the
# points using the predicted SpectralCluster (0 = Red and 1 = Blue).  Besides, plot the hyperplane as a dotted line
# to the graph.  To obtain the full credits, you should properly label the axes, the legend, and the chart title.
# Also, grid lines should be added to the axes.
spiral_with_cluster['pred_class'] = y_predict_class

# get the separating hyperplane
xx = numpy.linspace(-5, 5)
yy = numpy.zeros((len(xx), 1))
for j in range(1):
    w = this_fit.coef_[j, :]
    a = -w[0] / w[1]
    yy[:, j] = a * xx - (this_fit.intercept_[j]) / w[1]

# plot the line, the coordinates, and the nearest vectors to the plane
color_array = ['red', 'blue']
for i in range(2):
    x_y = spiral_with_cluster[spiral_with_cluster['pred_class'] == i]
    plt.scatter(x_y['x'], x_y['y'], c=color_array[i], label=i)
plt.plot(xx, yy[:, 0], color='black', linestyle='-')
plt.xlabel('x axis of coordinates')
plt.ylabel('y axis of coordinates')
plt.title('Support Vector Machines on Two Segments')
plt.legend(title='Predicted_Cluster', loc='best')
plt.grid(True)
plt.show()

print()
print("=" * 50)
print("ML-Assignment 5-Question 2-Section d)")
print("=" * 50)
# d)	(10 points) Please express the data as polar coordinates.  Please plot the theta-coordinate against the
# radius-coordinate in a scatterplot.  Please color-code the points using the SpectralCluster variable (0 = Red and 1
# = Blue).  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also,
# grid lines should be added to the axes.
# Convert to the polar coordinates


def customArcTan(z):
    theta = numpy.where(z < 0.0, 2.0 * numpy.pi + z, z)
    return (theta)


# get radius and theta coordinates
spiral_with_cluster['radius'] = numpy.sqrt(spiral_with_cluster['x'] ** 2 + spiral_with_cluster['y'] ** 2)
spiral_with_cluster['theta'] = numpy.arctan2(spiral_with_cluster['y'], spiral_with_cluster['x']).apply(customArcTan)

# plot the polar coordinates
color_array = ['red', 'blue']
for i in range(2):
    x_y = spiral_with_cluster[spiral_with_cluster['SpectralCluster'] == i]
    plt.scatter(x_y['radius'], x_y['theta'], c=color_array[i], label=i)
plt.xlabel('Radius-Coordinates')
plt.ylabel('Theta-Coordinates')
plt.ylim(-1, 7)
plt.title('Support Vector Machines on Two Segments')
plt.legend(title='Spectral_Cluster', loc='best')
plt.grid(True)
plt.show()

print()
print("=" * 50)
print("ML-Assignment 5-Question 2-Section e)")
print("=" * 50)
# e)	(10 points) You should expect to see three distinct strips of points and a lone point.  Since the
# SpectralCluster variable has two values, you will create another variable, named Group, and use it as the new
# target variable. The Group variable will have four values. Value 0 for the lone point on the upper left corner of
# the chart in (d), values 1, 2,and 3 for the next three strips of points. Please plot the theta-coordinate against
# the radius-coordinate in a scatterplot.  Please color-code the points using the new Group target variable (0 = Red,
# 1 = Blue, 2 = Green, 3 = Black).  To obtain the full credits, you should properly label the axes, the legend,
# and the chart title.  Also, grid lines should be added to the axes.
group = numpy.zeros(spiral_with_cluster.shape[0])

# create four group by using the location of the coordinates
for index, row in spiral_with_cluster.iterrows():
    if row['radius'] < 1.5 and row['theta'] > 6:
        group[index] = 0
    elif row['radius'] < 2.5 and row['theta'] > 3:
        group[index] = 1
    elif 2.5 < row['radius'] < 3 and row['theta'] > 5.5:
        group[index] = 1
    elif row['radius'] < 2.5 and row['theta'] < 3:
        group[index] = 2
    elif 3 < row['radius'] < 4 and 3.5 < row['theta'] < 6.5:
        group[index] = 2
    elif 2.5 < row['radius'] < 3 and 2 < row['theta'] < 4:
        group[index] = 2
    elif 2.5 < row['radius'] < 3.5 and row['theta'] < 2.25:
        group[index] = 3
    elif 3.55 < row['radius'] and row['theta'] < 3.25:
        group[index] = 3

spiral_with_cluster['group'] = group
# plot coordinates divided into four group
color_array = ['red', 'blue', 'green', 'black']
for i in range(4):
    x_y = spiral_with_cluster[spiral_with_cluster['group'] == i]
    plt.scatter(x=x_y['radius'], y=x_y['theta'], c=color_array[i], label=i)
plt.xlabel('Radius-Coordinates')
plt.ylabel('Theta-Coordinates')
plt.title('Support Vector Machines on Four Segments')
plt.legend(title='Group', loc='best', )
plt.grid(True)
plt.show()

print()
print("=" * 50)
print("ML-Assignment 5-Question 2-Section f)")
print("=" * 50)
# f)	(10 points) Since the graph in (e) has four clearly separable and neighboring segments, we will apply the
# Support Vector Machine algorithm in a different way.  Instead of applying SVM once on a multi-class target
# variable, you will SVM three times, each on a binary target variable. SVM 0: Group 0 versus Group 1 SVM 1: Group 1
# versus Group 2 SVM 2: Group 2 versus Group 3 Please give the equations of the three hyperplanes.

# build SVM 0: Group 0 versus Group 1
svm_1 = svm.SVC(kernel="linear", random_state=20191108, decision_function_shape='ovr', max_iter=-1)
subset1 = spiral_with_cluster[spiral_with_cluster['group'] == 0]
subset1 = subset1.append(spiral_with_cluster[spiral_with_cluster['group'] == 1])
train_subset1 = subset1[['radius', 'theta']]
svm_1.fit(train_subset1, subset1['SpectralCluster'])

# build SVM 1: Group 1 versus Group 2
svm_2 = svm.SVC(kernel="linear", random_state=20191108, decision_function_shape='ovr', max_iter=-1)
subset2 = spiral_with_cluster[spiral_with_cluster['group'] == 1]
subset2 = subset2.append(spiral_with_cluster[spiral_with_cluster['group'] == 2])
train_subset2 = subset2[['radius', 'theta']]
svm_2.fit(train_subset2, subset2['SpectralCluster'])

# build SVM 2: Group 2 versus Group 3
svm_3 = svm.SVC(kernel="linear", random_state=20191108, decision_function_shape='ovr', max_iter=-1)
subset3 = spiral_with_cluster[spiral_with_cluster['group'] == 2]
subset3 = subset3.append(spiral_with_cluster[spiral_with_cluster['group'] == 3])
train_subset3 = subset3[['radius', 'theta']]
svm_3.fit(train_subset3, subset3['SpectralCluster'])

print(
    f'Equation of the separating hyperplane for SVM 0 is "𝑤_0+𝐰_1*𝐱_1+w_2*x_2=𝟎"  ==> '
    f'({numpy.round(svm_1.intercept_[0] ,7)})'
    f' + ({numpy.round(svm_1.coef_[0][0], 7)}*x_1)'
    f' + ({numpy.round(svm_1.coef_[0][1], 7)}*x_2) = 𝟎')
print(
    f'Equation of the separating hyperplane for SVM 1 is "𝑤_0+𝐰_1*𝐱_1+w_2*x_2=𝟎"  ==> '
    f'({numpy.round(svm_2.intercept_[0] ,7)})'
    f' + ({numpy.round(svm_2.coef_[0][0], 7)}*x_1)'
    f' + ({numpy.round(svm_2.coef_[0][1], 7)}*x_2) = 𝟎')
print(
    f'Equation of the separating hyperplane for SVM 2 is "𝑤_0+𝐰_1*𝐱_1+w_2*x_2=𝟎"  ==> '
    f'({numpy.round(svm_3.intercept_[0] ,7)})'
    f' + ({numpy.round(svm_3.coef_[0][0], 7)}*x_1)'
    f' + ({numpy.round(svm_3.coef_[0][1], 7)}*x_2) = 𝟎')

print()
print("=" * 50)
print("ML-Assignment 5-Question 2-Section g)")
print("=" * 50)
# g)	(5 points) Please plot the theta-coordinate against the radius-coordinate in a scatterplot.  Please color-code
# the points using the new Group target variable (0 = Red, 1 = Blue, 2 = Green, 3 = Black). Please add the
# hyperplanes to the graph. To obtain the full credits, you should properly label the axes, the legend, and the chart
# title.  Also, grid lines should be added to the axes.
# getting hyperplanes for all the SVM
w = svm_1.coef_[0]
a = -w[0] / w[1]
xx1 = numpy.linspace(1, 4)
yy1 = a * xx1 - (svm_1.intercept_[0]) / w[1]
w = svm_2.coef_[0]
a = -w[0] / w[1]
xx2 = numpy.linspace(1, 4)
yy2 = a * xx2 - (svm_2.intercept_[0]) / w[1]
w = svm_3.coef_[0]
a = -w[0] / w[1]
xx3 = numpy.linspace(1, 4)
yy3 = a * xx3 - (svm_3.intercept_[0]) / w[1]
# plot polar coordinates and hyperplanes
for i in range(4):
    x_y = spiral_with_cluster[spiral_with_cluster['group'] == i]
    plt.scatter(x_y['radius'], x_y['theta'], c=color_array[i], label=i)
plt.plot(xx1, yy1, color='black', linestyle='-')
plt.plot(xx2, yy2, color='black', linestyle='-')
plt.plot(xx3, yy3, color='black', linestyle='-')
plt.xlabel('Radius-Coordinates')
plt.ylabel('Theta-Coordinates')
plt.title('Support Vector Machines on Four Segments')
plt.legend(title='Group', loc='best', )
plt.grid(True)
plt.show()

print()
print("=" * 50)
print("ML-Assignment 5-Question 2-Section h)")
print("=" * 50)
# h)	(10 points) Convert the observations along with the hyperplanes from the polar coordinates back to the
# Cartesian coordinates. Please plot the y-coordinate against the x-coordinate in a scatterplot.  Please color-code
# the points using the SpectralCluster (0 = Red and 1 = Blue). Besides, plot the hyper-curves as dotted lines to the
# graph.  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also,
# grid lines should be added to the axes. Based on your graph, which hypercurve do you think is not needed?
# Back-transform the hyperplane from the Polar coordinates to the Cartesian coordinates
h1_xx1 = xx1 * numpy.cos(yy1)
h1_yy1 = xx1 * numpy.sin(yy1)
h2_xx2 = xx2 * numpy.cos(yy2)
h2_yy2 = xx2 * numpy.sin(yy2)
h3_xx3 = xx3 * numpy.cos(yy3)
h3_yy3 = xx3 * numpy.sin(yy3)
# plot the line, the coordinates, and the nearest vectors to the plane
color_array = ['red', 'blue']
for i in range(2):
    x_y = spiral_with_cluster[spiral_with_cluster['SpectralCluster'] == i]
    plt.scatter(x_y['x'], x_y['y'], c=color_array[i], label=i)
plt.plot(h1_xx1, h1_yy1, color='green', linestyle='-')
plt.plot(h2_xx2, h2_yy2, color='black', linestyle='-')
plt.plot(h3_xx3, h3_yy3, color='black', linestyle='-')
plt.xlabel('x axis of coordinates')
plt.ylabel('y axis of coordinates')
plt.title('Support Vector Machines on Two Segments')
plt.legend(title='Spectral_Cluster', loc='best')
plt.grid(True)
plt.show()
print(f'Based on my graph, I think green colored hypercurve is not needed.')
