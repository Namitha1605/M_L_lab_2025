from ISLP import load_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


def load_dataset():
    data = load_data('NCI60')
    x = data['data']
    y = data['labels']
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print("Unique labels:", y['label'].unique())

    # Encode the labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y['label'])

    return x, y_encoded


def split_data(x, y):
    return train_test_split(x, y, test_size=0.3, random_state=30)


def H_clusterning(x_train, n_clusters=200):
    # Cluster genes (columns of x_train)
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    gene_clusters = agg_clustering.fit_predict(x_train.T)  # Transpose so that genes are clustered
    return gene_clusters


def reduce_by_gene_clusters(x_data, gene_clusters, n_clusters):
    # Create reduced features by averaging expression within each gene cluster
    x_reduced = np.zeros((x_data.shape[0], n_clusters))
    for i in range(n_clusters):
        cluster_indices = np.where(gene_clusters == i)[0]
        x_reduced[:, i] = x_data[:, cluster_indices].mean(axis=1)
    return x_reduced


def pca_reduction(x_train, n_components=30):
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    x_reduced = pca.fit_transform(x_train)
    return x_reduced, pca


def train_classification_model(x_train, y_train, x_test, y_test):
    # Train a Logistic Regression classifier
    clf = LogisticRegression(max_iter=10000,random_state=30,)
    clf.fit(x_train, y_train)

    # Predict and compute accuracy
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification accuracy: {accuracy:.4f}")
    return accuracy


def plot_dendrogram(x_train):
    # Visualize hierarchical clustering of genes
    Z = linkage(x_train.T, 'ward')
    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title("Gene Clustering Dendrogram")
    plt.xlabel("Gene Index")
    plt.ylabel("Distance")
    plt.show()


def main():
    x, y = load_dataset()

    # Split into train and test sets
    x_train, x_test, y_train, y_test = split_data(x, y)

    # Standardize
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Feature reduction using Hierarchical Clustering
    print("\nFeature reduction using Hierarchical Clustering:")
    n_clusters = 14
    gene_clusters = H_clusterning(x_scaled, n_clusters=n_clusters)
    x_clustered = reduce_by_gene_clusters(x_scaled, gene_clusters, n_clusters)
    x_test_clustered = reduce_by_gene_clusters(x_test_scaled, gene_clusters, n_clusters)
    accuracy_hc = train_classification_model(x_clustered, y_train, x_test_clustered, y_test)

    # Feature reduction using PCA
    print("\nFeature reduction using PCA:")
    x_pca, pca = pca_reduction(x_scaled, n_components=n_clusters)
    x_test_pca = pca.transform(x_test_scaled)
    accuracy_pca = train_classification_model(x_pca, y_train, x_test_pca, y_test)

    # Comparison
    print("\nComparison of the two approaches:")
    print(f"Accuracy with Hierarchical Clustering: {accuracy_hc:.4f}")
    print(f"Accuracy with PCA: {accuracy_pca:.4f}")

    # Optional dendrogram plot
    plot_dendrogram(x_scaled)


if __name__ == '__main__':
    main()
