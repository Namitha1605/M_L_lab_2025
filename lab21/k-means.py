import numpy as np
import matplotlib.pyplot as plt

# Step (a): Define the observations
X = np.array([
    [1, 4],
    [1, 3],
    [0, 4],
    [5, 1],
    [4, 2],
    [5, 0]
])

# Step (b): Randomly assign cluster labels
np.random.seed(0)
labels = np.random.choice([0, 1], size=len(X))

def compute_centroids(X, labels, K):
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        centroids[k] = X[labels == k].mean(axis=0)
    return centroids

def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

# Step (c-f): Iterate until convergence
K = 2
iteration = 0
while True:
    iteration += 1
    centroids = compute_centroids(X, labels, K)
    new_labels = assign_clusters(X, centroids)
    if np.all(labels == new_labels):
        break
    labels = new_labels

# Final centroids
centroids = compute_centroids(X, labels, K)

# Step (f): Plot with cluster labels
colors = ['red' if label == 0 else 'blue' for label in labels]
plt.scatter(X[:, 0], X[:, 1], c=colors, s=100)
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroids')
plt.title('Final Clustering Result')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
plt.legend()
plt.show()
