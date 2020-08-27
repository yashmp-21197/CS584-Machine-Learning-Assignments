# load necessary libraries
import pandas
import numpy
import sklearn.neural_network as nn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

print()
print("=" * 50)
print("=" * 50)
print("ML-Assignment 5-Question 1")
print("=" * 50)
print("=" * 50)
# Please answer the following questions based on your model.
# read data from csv file
spiral_with_cluster = pandas.read_csv('SpiralWithCluster.csv', delimiter=',',
                                      usecols=['x', 'y', 'SpectralCluster'])

print()
print("=" * 50)
print("ML-Assignment 5-Question 1-Section a)")
print("=" * 50)
# a)	(5 points) What percent of the observations have SpectralCluster equals to 1?
no_obs = spiral_with_cluster.shape[0]
no_obs_spe_clu_1 = spiral_with_cluster[spiral_with_cluster['SpectralCluster'] == 1].shape[0]
per_obs_spe_clu_1 = (no_obs_spe_clu_1) / no_obs
print(f'Percent of the observations have SpectralCluster equals to 1 is {100 * per_obs_spe_clu_1}')

print()
print("=" * 50)
print("ML-Assignment 5-Question 1-Section b)")
print("=" * 50)
# b)	(15 points) You will search for the neural network that yields the lowest loss value and the lowest
# misclassification rate.  You will use your answer in (a) as the threshold for classifying an observation into
# SpectralCluster = 1. Your search will be done over a grid that is formed by cross-combining the following
# attributes: (1) activation function: identity, logistic, relu, and tanh; (2) number of hidden layers: 1, 2, 3, 4,
# and 5; and (3) number of neurons: 1 to 10 by 1.  List your optimal neural network for each activation function in a
# table.  Your table will have four rows, one for each activation function.  Your table will have five columns: (1)
# activation function, (2) number of layers, (3) number of neurons per layer, (4) number of iterations performed,
# (5) the loss value, and (6) the misclassification rate.
activation_functions = ['identity', 'logistic', 'relu', 'tanh']
no_layers = range(1, 6, 1)
no_neurons_per_layer = range(1, 11, 1)
result = pandas.DataFrame(
    columns=['Index', 'ActivationFunction', 'nLayers', 'nNeuronsPerLayer', 'nIterations', 'Loss',
             'MisclassificationRate'])
index = 0
# run loop for all possible networks
for act_fun in activation_functions:
    min_loss = float("inf")
    min_misclassification_rate = float("inf")
    nlyrs = -1
    nnpl = -1
    niter = -1
    for no_lyrs in no_layers:
        for no_npl in no_neurons_per_layer:
            # build neural network
            nn_obj = nn.MLPClassifier(hidden_layer_sizes=(no_npl,) * no_lyrs, activation=act_fun, verbose=False,
                                     solver='lbfgs', learning_rate_init=0.1, max_iter=5000, random_state=20191108)
            this_fit = nn_obj.fit(spiral_with_cluster[['x', 'y']], spiral_with_cluster[['SpectralCluster']])
            y_pred = nn_obj.predict_proba(spiral_with_cluster[['x', 'y']])
            n_iter = nn_obj.n_iter_
            loss = nn_obj.loss_

            target_y = spiral_with_cluster[['SpectralCluster']]
            no_targets = target_y.shape[0]

            # determining the predicted class
            pred_y = numpy.empty_like(target_y)
            threshold = per_obs_spe_clu_1
            for i in range(no_targets):
                if y_pred[i][0] >= threshold:
                    pred_y[i] = 0
                else:
                    pred_y[i] = 1

            # calculating accuracy and misclassification rate
            accuracy = metrics.accuracy_score(target_y, pred_y)
            misclassification_rate = 1 - accuracy

            # find neural network with minimum loss and misclassification rate
            if loss <= min_loss and misclassification_rate <= min_misclassification_rate:
                min_loss = loss
                min_misclassification_rate = misclassification_rate
                nlyrs = no_lyrs
                nnpl = no_npl
                niter = n_iter

    result = result.append(
        pandas.DataFrame([[index, act_fun, nlyrs, nnpl, niter, min_loss, min_misclassification_rate]],
                         columns=['Index', 'ActivationFunction', 'nLayers', 'nNeuronsPerLayer',
                                  'nIterations', 'Loss', 'MisclassificationRate']))
    index += 1
result = result.set_index('Index')
pandas.set_option('display.max_columns', 10)
print(result)

print()
print("=" * 50)
print("ML-Assignment 5-Question 1-Section d)")
print("=" * 50)
# d)	(5 points) Which activation function, number of layers, and number of neurons per layer give the lowest loss
# and the lowest misclassification rate?  What are the loss and the misclassification rate?  How many iterations are
# performed?
min_loss = float("inf")
min_misclassification_rate = float("inf")
index = None
# find activation function, number of layers, and number of neurons per layer with minimum loss and misclassification rate
for ind, row in result.iterrows():
    if row['Loss'] <= min_loss and row['MisclassificationRate'] <= min_misclassification_rate:
        index = ind
        min_loss = row['Loss']
        min_misclassification_rate = row['MisclassificationRate']

print(result.loc[index])


# build neural network with minimum loss and misclassification rate
nn_obj = nn.MLPClassifier(hidden_layer_sizes=(result.loc[index]['nNeuronsPerLayer'],) * result.loc[index]['nLayers'],
                         activation=result.loc[index]['ActivationFunction'], verbose=False,
                         solver='lbfgs', learning_rate_init=0.1, max_iter=5000, random_state=20191108)
this_fit = nn_obj.fit(spiral_with_cluster[['x', 'y']], spiral_with_cluster[['SpectralCluster']])
y_pred = nn_obj.predict_proba(spiral_with_cluster[['x', 'y']])

target_y = spiral_with_cluster[['SpectralCluster']]
no_targets = target_y.shape[0]

# determining the predicted class
pred_y = numpy.empty_like(target_y)
threshold = per_obs_spe_clu_1
for i in range(no_targets):
    if y_pred[i][0] >= threshold:
        pred_y[i] = 0
    else:
        pred_y[i] = 1

spiral_with_cluster['y_pred_0'] = y_pred[:, 0]
spiral_with_cluster['y_pred_1'] = y_pred[:, 1]
spiral_with_cluster['class'] = pred_y

target_y = spiral_with_cluster[['SpectralCluster']]
accuracy = metrics.accuracy_score(target_y, pred_y)
misclassification_rate = 1 - accuracy
print()
print(f'Accuracy is {accuracy * 100.0}%')
print(f'Misclassification rate is {misclassification_rate}')


print()
print("=" * 50)
print("ML-Assignment 5-Question 1-Section c)")
print("=" * 50)
# c)	(5 points) What is the activation function for the output layer?
print(f'The activation function for the output layer is "{nn_obj.out_activation_}"')

print()
print("=" * 50)
print("ML-Assignment 5-Question 1-Section e)")
print("=" * 50)
# e)	(5 points) Please plot the y-coordinate against the x-coordinate in a scatterplot.  Please color-code the
# points using the predicted SpectralCluster (0 = Red and 1 = Blue) from the optimal MLP in (d).  To obtain the full
# credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to
# the axes.
color_array = ['red', 'blue']
for i in range(2):
    x_y = spiral_with_cluster[spiral_with_cluster['class'] == i]
    plt.scatter(x_y['x'], x_y['y'], c=color_array[i], label=i)
plt.xlabel('x axis of coordinates')
plt.ylabel('y axis of coordinates')
plt.title('Scatter Plot of Spiral Cluster Coordinates')
plt.legend(title='Predicted_Cluster', loc='best')
plt.grid(True)
plt.show()

print()
print("=" * 50)
print("ML-Assignment 5-Question 1-Section f)")
print("=" * 50)
# f)	(5 points) What is the count, the mean and the standard deviation of the predicted probability Prob(
# SpectralCluster = 1) from the optimal MLP in (d) by value of the SpectralCluster?  Please give your answers up to
# the 10 decimal places.
pandas.set_option('float_format', '{:.10f}'.format)
print(spiral_with_cluster[spiral_with_cluster['class'] == 1]['y_pred_1'].describe())
