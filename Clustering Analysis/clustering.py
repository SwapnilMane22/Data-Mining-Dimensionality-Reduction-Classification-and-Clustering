# -*- coding: utf-8 -*-
"""A3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cjfmUclHCHfa4l2mDERaTJt2VZJnLSla
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import calinski_harabasz_score

import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)

data = pd.read_csv("/content/drive/MyDrive/Data/Spring 2024/DM/Content 2024/assignment3/synthetic_control.data", header=None, sep='\s+')

scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

"""## Question 1"""

def modified_kmeans(data, k, max_iterations=100):
    centroids = data[np.random.choice(len(data), k, replace=False)]

    for _ in range(max_iterations):
        labels = [np.argmin([euclidean(point, centroid) for centroid in centroids]) for point in data]

        new_centroids = [np.mean(data[np.array(labels) == i], axis=0) for i in range(k)]

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids

inertia = []
k_values = range(1, 11)
for k in k_values:
    labels, centroids = modified_kmeans(data_normalized, k)
    inertia.append(np.sum([euclidean(data_normalized[i], centroids[labels[i]])**2 for i in range(len(data_normalized))]))

def modified_kmeans(data, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels

"""## Question 2"""

plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

def modified_kmeans_elbow(data):
    distortions = []
    max_k = 11
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)

    plt.plot(range(1, max_k + 1), distortions, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.show()

def evaluate_clustering(labels, true_labels):
    ari = adjusted_rand_score(true_labels, labels)
    nmi = normalized_mutual_info_score(true_labels, labels)
    return ari, nmi

ground_truth_labels = np.repeat(range(6), 100)

modified_kmeans_elbow(data_normalized)

k = 4
cluster_labels = modified_kmeans(data_normalized, k)

ari, nmi = evaluate_clustering(cluster_labels, ground_truth_labels)
print("\nAdjusted Rand Index (ARI):", ari)
print("\nNormalized Mutual Information (NMI):", nmi)

# Iterate over different values of k and evaluate clustering performance
for k in range(2, 11):
    # Perform clustering
    cluster_labels = modified_kmeans(data_normalized, k)

    # Evaluate clustering performance
    ari, nmi = evaluate_clustering(cluster_labels, ground_truth_labels)

    # Print results
    print(f"\nNumber of Clusters (k): {k}")
    print("Adjusted Rand Index (ARI):", ari)
    print("Normalized Mutual Information (NMI):", nmi)

"""## Question 3"""

# The dataset for clustering using the classic modified K-means algorithm was sourced from the "relation.csv" file,
# as specified in the assignment3.doc. Initially, the dataset was copied and pasted into a notepad. Subsequently,
# the content from the notepad was saved as a .txt file. Finally, the .txt file was converted into a .csv file format
# for further processing and analysis. The relation.txt and relation.csv files are attached in the submitted documnent.
relation = pd.read_csv("/content/drive/MyDrive/Data/Spring 2024/DM/Content 2024/assignment3/relation.csv")

relation.rename(columns={relation.columns[0]: 'Index'}, inplace=True)

relation = relation[~relation['Id'].isin(['Id']) & ~relation['Id'].isna()]

relation.reset_index(drop=True, inplace=True)

print(relation)

relation['Amount'] = pd.to_numeric(relation['Amount'], errors='coerce')

collectedSamples = relation.groupby(['From', 'To']).agg({'Amount': ['sum', 'mean', 'count']}).reset_index()

collectedSamples.columns = ['From', 'To', 'Total_Amount', 'Average_Amount', 'Transaction_Count']

print(collectedSamples)

"""## Question 4"""

k = 2

collectedSamples[['Total_Amount', 'Average_Amount', 'Transaction_Count']] = collectedSamples[['Total_Amount', 'Average_Amount', 'Transaction_Count']].apply(pd.to_numeric)

unique_entities = pd.unique(relation[['From', 'To']].values.ravel('K'))

entity_labels = np.arange(len(unique_entities))

entity_label_map = dict(zip(unique_entities, entity_labels))

relation['From_Label'] = relation['From'].map(entity_label_map)
relation['To_Label'] = relation['To'].map(entity_label_map)

ground_truth_labels = relation.groupby(['From_Label', 'To_Label']).ngroup()

ground_truth_labels_aggregated = relation.groupby(['From_Label', 'To_Label']).size().reset_index(name='Transaction_Count')

cluster_labels = modified_kmeans(collectedSamples[['Total_Amount', 'Average_Amount', 'Transaction_Count']], k)

ari, nmi = evaluate_clustering(cluster_labels, ground_truth_labels_aggregated['To_Label']) #

modified_kmeans_elbow(collectedSamples[['Total_Amount', 'Average_Amount', 'Transaction_Count']])

print("\nAdjusted Rand Index (ARI):", ari)
print("\nNormalized Mutual Information (NMI):", nmi)

calinski_score = calinski_harabasz_score(collectedSamples[['Total_Amount', 'Average_Amount', 'Transaction_Count']], cluster_labels)

print("\nCalinski-Harabasz Score:", calinski_score)