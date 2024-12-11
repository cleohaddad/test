import logging
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Configuration de base des logs
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("project_logs.log"),  # Logs dans un fichier
        logging.StreamHandler()  # Affichage dans la console
    ]
)

# Fonction de détection d'anomalies par kNN
def knn_outlier_detection(data, k=5, threshold=1.5):
    """
    Detect anomalies in a dataset using the k-Nearest Neighbors algorithm.

    Args:
        data (numpy.ndarray): The input dataset of shape (n_samples, n_features).
        k (int): The number of neighbors to consider.
        threshold (float): The mean distance threshold for detecting anomalies.

    Returns:
        tuple:
            - numpy.ndarray: A boolean array indicating which points are outliers.
            - numpy.ndarray: The mean distances for all points in the dataset.
    """
    logging.info(f"Starting kNN outlier detection with k={k} and threshold={threshold}")
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(data)
    logging.debug("kNN model trained")

    distances, _ = neigh.kneighbors(data)
    logging.debug(f"Distances calculated: {distances}")

    mean_distances = distances.mean(axis=1)
    logging.debug(f"Mean distances: {mean_distances}")

    outliers = mean_distances > threshold
    logging.info(f"Number of outliers detected: {np.sum(outliers)}")

    return outliers, mean_distances

# Fonction de détection d'anomalies par Isolation Forest
def isolation_forest_outlier_detection(data, contamination=0.015):
    """
    Detect anomalies in a dataset using the Isolation Forest algorithm.

    Args:
        data (numpy.ndarray): The input dataset of shape (n_samples, n_features).
        contamination (float): The proportion of anomalies in the dataset.

    Returns:
        numpy.ndarray: A boolean array indicating which points are outliers.
    """
    logging.info(f"Starting Isolation Forest with contamination={contamination}")
    isolation_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = isolation_forest.fit_predict(data)
    logging.debug("Isolation Forest model trained")

    outliers = predictions == -1
    logging.info(f"Number of outliers detected: {np.sum(outliers)}")

    return outliers

# Fonction de détection d'anomalies par One-Class SVM
def one_class_svm_outlier_detection(data, nu=0.015, gamma="scale"):
    """
    Detect anomalies in a dataset using the One-Class SVM algorithm.

    Args:
        data (numpy.ndarray): The input dataset of shape (n_samples, n_features).
        nu (float): An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
        gamma (str or float): Kernel coefficient for the RBF kernel.

    Returns:
        numpy.ndarray: A boolean array indicating which points are outliers.
    """
    logging.info(f"Starting One-Class SVM with nu={nu}, gamma={gamma}")
    oc_svm = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    predictions = oc_svm.fit_predict(data)
    logging.debug("One-Class SVM model trained")

    outliers = predictions == -1
    logging.info(f"Number of outliers detected: {np.sum(outliers)}")

    return outliers

# Fonction de détection d'anomalies par PCA
def pca_outlier_detection(data, percentile=95):
    """
    Detect anomalies in a dataset using Principal Component Analysis (PCA).

    Args:
        data (numpy.ndarray): The input dataset of shape (n_samples, n_features).
        percentile (float): The percentile for determining the anomaly threshold based on reconstruction error.

    Returns:
        numpy.ndarray: A boolean array indicating which points are outliers.
    """
    logging.info(f"Starting PCA-based outlier detection with percentile={percentile}")
    pca = PCA(n_components=2)
    data_transformed = pca.fit_transform(data)
    logging.debug("PCA model trained")

    data_reconstructed = pca.inverse_transform(data_transformed)
    reconstruction_error = np.mean((data - data_reconstructed) ** 2, axis=1)
    logging.debug(f"Reconstruction errors: {reconstruction_error}")

    threshold = np.percentile(reconstruction_error, percentile)
    logging.info(f"Threshold for anomaly detection: {threshold}")

    outliers = reconstruction_error > threshold
    logging.info(f"Number of outliers detected: {np.sum(outliers)}")

    return outliers

# Exemple d'exécution et visualisation pour chaque méthode
def visualize_results(data, outliers, method_name):
    """
    Visualize the results of anomaly detection.

    Args:
        data (numpy.ndarray): The input dataset of shape (n_samples, n_features).
        outliers (numpy.ndarray): A boolean array indicating which points are outliers.
        method_name (str): The name of the anomaly detection method.
    """
    plt.scatter(data[:, 0], data[:, 1], c='blue', label="Normal data")
    plt.scatter(data[outliers][:, 0], data[outliers][:, 1], c='red', label="Anomalies", edgecolor='k')
    plt.legend()
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"Anomaly Detection using {method_name}")
    plt.show()

# Génération de données synthétiques
data, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
outliers = np.array([[8, 8], [9, 9], [-6, 6], [6, -10]])
data_with_outliers = np.vstack([data, outliers])

# Exemple d'utilisation : kNN
knn_outliers, _ = knn_outlier_detection(data_with_outliers, k=6, threshold=2.0)
visualize_results(data_with_outliers, knn_outliers, "kNN")

# Exemple d'utilisation : Isolation Forest
iso_outliers = isolation_forest_outlier_detection(data_with_outliers, contamination=0.015)
visualize_results(data_with_outliers, iso_outliers, "Isolation Forest")

# Exemple d'utilisation : One-Class SVM
svm_outliers = one_class_svm_outlier_detection(data_with_outliers, nu=0.015)
visualize_results(data_with_outliers, svm_outliers, "One-Class SVM")

# Exemple d'utilisation : PCA
pca_outliers = pca_outlier_detection(data_with_outliers, percentile=95)
visualize_results(data_with_outliers, pca_outliers, "PCA")
