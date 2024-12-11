#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 08:20:58 2024

@author: cleohaddad
"""

# Load data 

'''
import pandas as pd

# Exemple pour charger un jeu de données CSV
data = pd.read_csv('chemin/vers/ton/fichier.csv')
print(data.head())

'''

from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# Création d'un jeu de données avec des clusters et quelques outliers
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Ajout de quelques points aléatoires comme outliers
outliers = np.random.uniform(low=-10, high=10, size=(20, 2))
X = np.vstack([X, outliers])

# Visualisation des données
plt.scatter(X[:, 0], X[:, 1], c='b', marker='o', label="Données")
plt.scatter(outliers[:, 0], outliers[:, 1], c='r', marker='x', label="Outliers")
plt.legend()
plt.show()



#%%

# KNN

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Fonction de détection d'anomalies par kNN
def knn_outlier_detection(data, k=5, threshold=1.5):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(data)
    distances, _ = neigh.kneighbors(data)
    mean_distances = distances.mean(axis=1)
    outliers = mean_distances > threshold
    return outliers, mean_distances

# Génération d'un jeu de données synthétique avec make_blobs
data, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Ajout de quelques points comme outliers
outliers = np.array([[8, 8], [9, 9], [-6, 6], [6, -10]])
data_with_outliers = np.vstack([data, outliers])

# Détection d'anomalies avec kNN
outliers_detected, scores = knn_outlier_detection(data_with_outliers, k=6, threshold=2.0)

# Affichage des résultats
plt.scatter(data_with_outliers[:, 0], data_with_outliers[:, 1], c='blue', label="Données normales")
plt.scatter(data_with_outliers[outliers_detected][:, 0], data_with_outliers[outliers_detected][:, 1], 
            c='red', label="Anomalies détectées", edgecolor='k')
plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Détection d'anomalies avec kNN")
plt.show()


#%%


# Isolation forest 

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Génération d'un jeu de données synthétique avec make_blobs
data, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Ajout de quelques points comme outliers
outliers = np.array([[8, 8], [9, 9], [-6, 6], [6, -10]])
data_with_outliers = np.vstack([data, outliers])

# Application de l'Isolation Forest
isolation_forest = IsolationForest(contamination=0.015, random_state=42)
predictions = isolation_forest.fit_predict(data_with_outliers)

# Les prédictions de l'Isolation Forest : -1 pour les anomalies, 1 pour les points normaux
outliers_detected = predictions == -1

# Affichage des résultats
plt.scatter(data_with_outliers[:, 0], data_with_outliers[:, 1], c='blue', label="Données normales")
plt.scatter(data_with_outliers[outliers_detected][:, 0], data_with_outliers[outliers_detected][:, 1], 
            c='red', label="Anomalies détectées", edgecolor='k')
plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Détection d'anomalies avec Isolation Forest")
plt.show()

#%%

#SVM

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Génération de données normales avec make_blobs
data, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Ajout de quelques outliers
outliers = np.array([[8, 8], [9, 9], [-6, 6], [6, -10]])
data_with_outliers = np.vstack([data, outliers])

# Application de One-Class SVM
oc_svm = OneClassSVM(kernel="rbf", nu=0.015, gamma="scale")
predictions = oc_svm.fit_predict(data_with_outliers)

# Les prédictions de One-Class SVM : -1 pour les anomalies, 1 pour les points normaux
outliers_detected = predictions == -1

# Affichage des résultats
plt.scatter(data_with_outliers[:, 0], data_with_outliers[:, 1], c='blue', label="Données normales")
plt.scatter(data_with_outliers[outliers_detected][:, 0], data_with_outliers[outliers_detected][:, 1], 
            c='red', label="Anomalies détectées", edgecolor='k')
plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Détection d'anomalies avec One-Class SVM")
plt.show()

#%%

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Génération d'un jeu de données synthétique avec make_blobs
data, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Ajout de quelques points comme outliers
outliers = np.array([[8, 8], [9, 9], [-6, 6], [6, -10]])
data_with_outliers = np.vstack([data, outliers])

# Initialisation de l'ACP pour réduire les données à 2 composantes principales
pca = PCA(n_components=2)
data_transformed = pca.fit_transform(data_with_outliers)

# Reconstruction des données
data_reconstructed = pca.inverse_transform(data_transformed)

# Calcul de l'erreur de reconstruction
reconstruction_error = np.mean((data_with_outliers - data_reconstructed) ** 2, axis=1)

# Définition d'un seuil pour détecter les anomalies
# Ici, on choisit un seuil à 95e percentile
threshold = np.percentile(reconstruction_error, 95)
outliers_detected = reconstruction_error > threshold

# Affichage des résultats
plt.scatter(data_with_outliers[:, 0], data_with_outliers[:, 1], c='blue', label="Données normales")
plt.scatter(data_with_outliers[outliers_detected][:, 0], data_with_outliers[outliers_detected][:, 1], 
            c='red', label="Anomalies détectées", edgecolor='k')
plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Détection d'anomalies avec PCA")
plt.show()



