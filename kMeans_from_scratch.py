from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics.pairwise import euclidean_distances
import sys 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import random as rd
import math

if len(sys.argv) != 3:
    print("Usage: kMeans.py  <path_to_dataset>  <K>")
    sys.exit(1)

filename = sys.argv[1]
K = int(sys.argv[2])

# Read dataset file
if 'unbalance' in filename:
    dataset = pd.read_csv(f'{filename}', sep=" ", header=None)
else:
    dataset = pd.read_csv(f'{filename}', sep="\t", header=None)
X = dataset.iloc[:, [0, 1]].values

def find_clusters(X, n_clusters, rseed=0):
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    
    while True:
        # Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        
        # Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        
        # Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels

centers, labels = find_clusters(X, K)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap = 'viridis')
plt.scatter(centers[:,0],centers[:,1],marker='^', c = 'black')
plt.show()