# Tutorial for Anomaly Detection Project

## Overview

This tutorial will guide you through the steps to use the Anomaly Detection Project. You will learn how to:
1. Set up the project.
2. Run anomaly detection algorithms.
3. Visualize results.

The project includes implementations of algorithms like **k-Nearest Neighbors (kNN)**, **Isolation Forest**, **One-Class SVM**, and **PCA** for detecting anomalies in datasets.

---

## Prerequisites

- Python 3.7 or above installed on your system.
- Basic understanding of Python and data analysis.
- Required libraries:
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

Make sure to install the dependencies by running:
```bash
pip install -r requirements.txt
```

---

## Step 1: Clone the Repository

Start by cloning the GitHub repository:
```bash
git clone https://github.com/cleohaddad/anomaly-detection.git
cd anomaly-detection
```

---

## Step 2: Run the Main File

Execute the main file `NewPI.py` to see the implemented anomaly detection algorithms in action:
```bash
python NewPI.py
```

This will:
- Generate a synthetic dataset with anomalies.
- Run each detection algorithm.
- Display results as scatter plots where anomalies are highlighted.

---

## Step 3: Customize Parameters

Each algorithm allows you to customize parameters. For example, in the `knn_outlier_detection` function, you can change:
- `k` (number of neighbors).
- `threshold` (distance threshold for anomaly detection).

Example usage in code:
```python
knn_outliers, _ = knn_outlier_detection(data_with_outliers, k=6, threshold=2.0)
```

Similarly, you can modify parameters for other algorithms like contamination rate for **Isolation Forest**:
```python
iso_outliers = isolation_forest_outlier_detection(data_with_outliers, contamination=0.015)
```

---

## Step 4: Visualize Results

The project includes a generic visualization function that displays normal data points and anomalies. For instance:
```python
visualize_results(data_with_outliers, knn_outliers, "kNN")
```

This produces a scatter plot with anomalies marked in red and normal points in blue.

---

## Step 5: Extend the Project

Feel free to extend the project by:
- Adding new algorithms for anomaly detection.
- Integrating real-world datasets.
- Enhancing visualization options.

---

## Conclusion

You now have a complete walkthrough to set up, run, and customize the Anomaly Detection Project. Explore the codebase, modify parameters, and experiment with your datasets to learn more about anomaly detection techniques!

