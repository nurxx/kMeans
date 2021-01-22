import sys 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

# K-Means using random initialization
# kmeans = KMeans(n_clusters = K, init = "random", random_state = 0)

# K-Means++ initialization
kmeans = KMeans(n_clusters = K, init = "k-means++")

# Same as fit(X) and then predict(X)
y_kmeans = kmeans.fit_predict(X)
for i in range(0, K):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], label = 'Cluster ' + str(i+1))

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'black', marker='^', label = 'Centroids')
plt.legend()
plt.show()