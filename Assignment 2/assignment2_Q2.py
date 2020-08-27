# Load the libraries
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.neighbors as nbrs
import numpy as np
import math
from numpy import linalg as LA

# load data file
spiral_data = pd.read_csv('Spiral.csv', delimiter=',')
spiral_data = spiral_data.dropna()
no_objs = spiral_data.shape[0]

# Q1-a) Generate a scatterplot of y (vertical axis) versus x (horizontal axis).  How many clusters will you say by
# visual inspection?
print('\n<========(Q2-a)========>')
plt.scatter(spiral_data['x'], spiral_data['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# Q2-b) Apply the K-mean algorithm directly using your number of clusters that you think in (a). Regenerate the
# scatterplot using the K-mean cluster identifier to control the color scheme?
print('\n<========(Q2-b)========>')
# finding clusters in plot
no_clusters = 2
train_data = spiral_data[['x', 'y']]
k_means = cluster.KMeans(n_clusters=no_clusters, random_state=60616).fit(train_data)

print("cluster centroids = \n", k_means.cluster_centers_)

spiral_data['k_mean_cluster'] = k_means.labels_

# printing cluster data
for i in range(no_clusters):
    print("\ncluster label = ", i)
    print(f"{spiral_data.loc[spiral_data['k_mean_cluster'] == i]}")

plt.scatter(spiral_data['x'], spiral_data['y'], c=spiral_data['k_mean_cluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# Q2-c) Apply the nearest neighbor algorithm using the Euclidean distance.  How many nearest neighbors will you use?
# Remember that you may need to try a couple of values first and use the eigenvalue plot to validate your choice.
print('\n<========(Q2-c)========>')
# k nearest neighbors (k=3)
# finding number of nearest neighbors
k_nrst_nbrs = 3
knn_spec = nbrs.NearestNeighbors(n_neighbors=k_nrst_nbrs, algorithm='brute', metric='euclidean')
k_nbrs = knn_spec.fit(train_data)
dis, ind = k_nbrs.kneighbors(train_data)

# Retrieve the distances among the observations
dist_object = nbrs.DistanceMetric.get_metric('euclidean')
distances = dist_object.pairwise(train_data)

# Create the Adjacency and the Degree matrices
adjacency = np.zeros((no_objs, no_objs))
degree = np.zeros((no_objs, no_objs))

for i in range(no_objs):
    for j in ind[i]:
        if (i <= j):
            adjacency[i, j] = math.exp(- distances[i][j])
            adjacency[j, i] = adjacency[i, j]

for i in range(no_objs):
    sum = 0
    for j in range(no_objs):
        sum += adjacency[i, j]
    degree[i, i] = sum

l_matrix = degree - adjacency

# Q2-c,d) Retrieve the first two eigenvectors that correspond to the first two smallest eigenvalues.  Display up to
# ten decimal places the means and the standard deviation of these two eigenvectors.  Also, plot the first
# eigenvector on the horizontal axis and the second eigenvector on the vertical axis.
print('\n<========(Q2-d)========>')
e_vals, e_vecs = LA.eigh(l_matrix)

# Series plot of the smallest ten eigenvalues to determine the number of clusters
plt.scatter(np.arange(0, 9, 1), e_vals[0:9, ])
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.grid(True)
plt.show()

# Inspect the values of the selected eigenvectors
z = e_vecs[:, [0, 1]]

plt.scatter(z[[0]], z[[1]])
plt.xlabel('z[0] (first eigen vector)')
plt.ylabel('z[1] (second eigen vector)')
plt.grid(True)
plt.show()

# mean and standard deviation of first two eigen vectors
print(f'mean of first evec : {z[[0]].mean()}')
print(f'mean of second evec : {z[[1]].mean()}')
print(f'std of first evec : {np.std(z[[0]], dtype=np.float64)}')
print(f'std of second evec : {np.std(z[[1]], dtype=np.float64)}')

# Q2-c,e) Apply the K-mean algorithm on your first two eigenvectors that correspond to the first two smallest
# eigenvalues. Regenerate the scatterplot using the K-mean cluster identifier to control the color scheme?
print('\n<========(Q2-e)========>')
k_means_spectral = cluster.KMeans(n_clusters=no_clusters, random_state=60616).fit(z)

spiral_data['spectral_cluster'] = k_means_spectral.labels_

plt.scatter(spiral_data['x'], spiral_data['y'], c=spiral_data['spectral_cluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
