# %%
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate synthetic data for KMeans clustering

X, _ = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=1.20,
    random_state=13
)
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], s=30, cmap='viridis')
plt.title('Generated Data for KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Create a KMeans model
kmeans = KMeans(n_clusters=4, random_state=13)
# Fit the model to the data
kmeans.fit(X) 

labels = kmeans.labels_

plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')
plt.title('KMeans Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show() 
# %
# %%
