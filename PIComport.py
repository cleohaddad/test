#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:31:49 2024

@author: cleohaddad
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist

# 1. Générer des données synthétiques
# On crée deux clusters bien distincts pour simuler des comportements normaux
X_normal, _ = make_blobs(n_samples=500, centers=[[2, 2], [7, 7]], cluster_std=0.8, random_state=42)

# Ajouter des anomalies de type changement de comportement (nouveau cluster)
X_anomalies, _ = make_blobs(n_samples=10, centers=[[12, 12]], cluster_std=0.5, random_state=42)
X = np.vstack([X_normal, X_anomalies])

# 2. Appliquer K-Means avec 2 clusters (en supposant qu'on ne connaît pas les anomalies)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# 3. Calculer la distance de chaque point à son centroïde
# Pour chaque point, trouver la distance au centroïde du cluster auquel il appartient
distances = cdist(X, kmeans.cluster_centers_).min(axis=1)

# 4. Définir un seuil basé sur la distribution des distances pour détecter les anomalies
# Ici, on utilise le 95e percentile comme seuil pour définir les points anormaux
threshold = np.percentile(distances, 98)
anomalies = distances > threshold

# Affichage des résultats
plt.scatter(X[:, 0], X[:, 1], c='blue', label="Points normaux")
plt.scatter(X[anomalies][:, 0], X[anomalies][:, 1], c='red', label="Anomalies", edgecolor='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='yellow', s=100, label="Centroïdes")
plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Détection d'anomalies par K-Means (changement de comportement)")
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

# 1. Générer des données synthétiques
# Créer des clusters normaux pour simuler des comportements réguliers
X_normal, _ = make_blobs(n_samples=500, centers=[[2, 2], [7, 7]], cluster_std=0.8, random_state=42)

# Ajouter des anomalies de type changement de comportement (nouveau cluster éloigné)
X_anomalies, _ = make_blobs(n_samples=10, centers=[[12, 12]], cluster_std=0.5, random_state=42)
X = np.vstack([X_normal, X_anomalies])

# 2. Appliquer Isolation Forest pour la détection d'anomalies
# On spécifie un pourcentage des points à considérer comme anomalies
isolation_forest = IsolationForest(contamination=0.035, random_state=42)
y_pred = isolation_forest.fit_predict(X)

# 3. Les points avec une prédiction de -1 sont des anomalies
anomalies = y_pred == -1

# Affichage des résultats
plt.scatter(X[:, 0], X[:, 1], c='blue', label="Points normaux")
plt.scatter(X[anomalies][:, 0], X[anomalies][:, 1], c='red', label="Anomalies", edgecolor='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Détection d'anomalies par Isolation Forest (changement de comportement)")
plt.legend()
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs

# 1. Générer des données synthétiques
# Création de deux clusters représentant des comportements normaux
X_normal, _ = make_blobs(n_samples=500, centers=[[2, 2], [7, 7]], cluster_std=0.8, random_state=42)

# Ajouter des anomalies représentant un changement de comportement
X_anomalies, _ = make_blobs(n_samples=10, centers=[[12, 12]], cluster_std=0.5, random_state=42)
X = np.vstack([X_normal, X_anomalies])

# 2. Appliquer One-Class SVM pour la détection d'anomalies
# Utilisation d'un noyau gaussien 'rbf' et réglage de nu (nu est un paramètre de régularisation)
one_class_svm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
y_pred = one_class_svm.fit_predict(X)

# 3. Identifier les anomalies (les points avec une prédiction de -1)
anomalies = y_pred == -1

# Affichage des résultats
plt.scatter(X[:, 0], X[:, 1], c='blue', label="Points normaux")
plt.scatter(X[anomalies][:, 0], X[anomalies][:, 1], c='red', label="Anomalies", edgecolor='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Détection d'anomalies par One-Class SVM (changement de comportement)")
plt.legend()
plt.show()

