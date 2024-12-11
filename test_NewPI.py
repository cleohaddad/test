import pytest
import numpy as np
from NewPI import knn_outlier_detection, isolation_forest_outlier_detection, one_class_svm_outlier_detection, pca_outlier_detection

# Générer des données pour les tests
data = np.array([[1, 1], [2, 2], [3, 3], [100, 100]])

def test_knn_outlier_detection():
    outliers, _ = knn_outlier_detection(data, k=2, threshold=10)
    assert np.sum(outliers) == 1  # Un seul point est un outlier
    assert outliers[-1] == True  # Le dernier point est l'outlier

def test_isolation_forest_outlier_detection():
    outliers = isolation_forest_outlier_detection(data, contamination=0.25)
    assert np.sum(outliers) == 1  # Un seul point est un outlier
    assert outliers[-1] == True  # Le dernier point est l'outlier

def test_one_class_svm_outlier_detection():
    outliers = one_class_svm_outlier_detection(data, nu=0.1)
    assert np.sum(outliers) == 1  # Un seul point est un outlier
    assert outliers[-1] == True  # Le dernier point est l'outlier

def test_pca_outlier_detection():
    outliers = pca_outlier_detection(data, percentile=95)
    assert np.sum(outliers) == 1  # Un seul point est un outlier
    assert outliers[-1] == True  # Le dernier point est l'outlier
