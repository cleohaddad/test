# Anomaly Detection Project

## Description
Ce projet implémente plusieurs techniques de détection d'anomalies en utilisant des algorithmes classiques comme :
- **k-Nearest Neighbors (kNN)**
- **Isolation Forest**
- **One-Class SVM**
- **Principal Component Analysis (PCA)**

Chaque méthode est appliquée sur des données synthétiques générées avec des anomalies simulées. Le projet inclut également des logs détaillés et des tests unitaires pour garantir le bon fonctionnement des algorithmes.

---

## Features
- Détection d'anomalies avec des algorithmes basés sur :
  - Distance (`kNN`)
  - Isolation aléatoire (`Isolation Forest`)
  - Support vector machine (`One-Class SVM`)
  - Reconstruction d'erreurs (`PCA`)
- Visualisation des résultats avec Matplotlib.
- Intégration de logs à plusieurs niveaux (`DEBUG`, `INFO`).
- Tests unitaires pour valider les fonctions principales.

---

## Prérequis
1. Python 3.7 ou supérieur.
2. Biblioteques python numpy, sklearn et matplotlib

---

## Installation
1. Clone le dépôt :
   ```bash
   git clone https://github.com/cleohaddad/anomaly-detection.git
   cd anomaly-detection


## Utilisation
1. Exécute le fichier principal pour lancer les algorithmes :
python NewPI.py

Les résultats seront affichés sous forme de graphiques, indiquant les anomalies détectées en rouge.

## Tests Unitaires
Des tests unitaires sont disponibles pour valider les implémentations :

pytest test_NewPI.py

## Structure du projet

anomaly-detection/
├── NewPI.py                 # Fichier principal avec les algorithmes
├── test_NewPI.py            # Tests unitaires
├── README.md                # Documentation du projet

## Auteurs
Cleo Haddad (@cleohaddad)
