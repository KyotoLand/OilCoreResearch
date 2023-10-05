import numpy as np
from pyclustering.cluster.center_initializer import random_center_initializer, kmeans_plusplus_initializer
from pyclustering.cluster.encoder import cluster_encoder
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils import distance_metric
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tools import calculate_wavelet_coeffs


def get_kmeans_with_specified_metric(X, dist_measure, num_of_clusters):
    initial_centers = kmeans_plusplus_initializer(X, num_of_clusters).initialize()
    # instance created for respective distance metric
    instance_km = kmeans(X, initial_centers=initial_centers, metric=distance_metric(dist_measure))
    # perform cluster analysis
    instance_km.process()
    # cluster analysis results - clusters and centers
    py_clusters = instance_km.get_clusters()
    py_centers = instance_km.get_centers()
    # enumerate encoding type to index labeling to get labels
    py_encoding = instance_km.get_cluster_encoding()
    py_encoder = cluster_encoder(py_encoding, py_clusters, X)
    py_labels = py_encoder.set_encoding(0).get_clusters()
    # function purity score is defined in previous section
    return X, py_labels


def get_dbscan_with_wavelet_transform(data, wavelet='mexh'):
    # Calculate wavelet coefficients
    if wavelet is not None:
        X = calculate_wavelet_coeffs(data, wavelet)
    else:
        # Use raw input data if wavelet is None
        X = np.array(list(data.values()))
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=1.3, min_samples=100)
    labels = dbscan.fit_predict(X)

    return X, labels