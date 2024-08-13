from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, AffinityPropagation, SpectralClustering, Birch, MeanShift
from collections import defaultdict
from copy import deepcopy
import numpy as np

def hierarchical_clustering(X, n_clusters, metric = 'euclidean', linkage_method = 'ward', verbose = None, normalize = False):
    """
    if normalize:
        X = deepcopy(X)
        X = (X - X.mean(axis=1)[:, np.newaxis]) / np.std(X, axis=1)[:, np.newaxis].astype(np.float32)
    """
    
    # Using Ward linkage for hierarchical clustering
    #linkage_matrix = linkage(X.T, method=linkage_method)
   
    # Performing clustering based on the linkage matrix
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method, metric = metric)
    
    
    labels = clustering.fit_predict(X.T)
   
   

    # Creating a list to store clusters
    #clusters_features = [[] for _ in range(n_clusters)]
    clustered_labels_to_neurons = defaultdict(list)

    # Filling the list with the subset feature vectors
    for i, label in enumerate(labels):
        #clusters_features[label].append(X[:, i][:, np.newaxis])  # Adding a new axis to maintain 2D shape
        clustered_labels_to_neurons[label].append(i)

    if verbose:
        # Printing the shape of each cluster
        for label, indices in clustered_labels_to_neurons.items():
            print(f"Cluster {label}: {len(indices)}")
    
    return clustered_labels_to_neurons