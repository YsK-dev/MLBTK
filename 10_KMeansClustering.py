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

## hierarchical clustering
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage 


# Generate synthetic data for hierarchical clustering
X_blobs, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.110,
                        random_state=42)
# Create a hierarchical clustering model

plt.scatter(X_blobs[:, 0], X_blobs[:, 1], s=30, cmap='viridis')
plt.title('Generated Data for Hierarchical Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

linkage_methods = ['ward', 'complete', 'average', 'single']
for method in linkage_methods:
    clustering = AgglomerativeClustering(n_clusters=4, linkage=linkage_methods.index(method))
    # 'ward' minimizes the variance of clusters being merged
    # 'complete' minimizes the maximum distance between points in clusters
    # 'average' minimizes the average distance between points in clusters
    # 'single' minimizes the distance between the closest points in clusters
    # Note: The linkage parameter in AgglomerativeClustering accepts 'ward', 'complete', 'average', or 'single'
    if method == 'ward':
        clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
    elif method == 'complete':
        clustering = AgglomerativeClustering(n_clusters=4, linkage='complete')
    elif method == 'average':
        clustering = AgglomerativeClustering(n_clusters=4, linkage='average')
    elif method == 'single':
        clustering = AgglomerativeClustering(n_clusters=4, linkage='single')    
    # Fit the model to the data and predict cluster labels
    labels = clustering.fit_predict(X_blobs)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels, s=30, cmap='viridis')
    plt.title(f'Hierarchical Clustering with {method} Linkage')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()




# Create a dendrogram to visualize the hierarchical clustering
for method in linkage_methods:
    plt.figure(figsize=(10, 7))
    plt.title(f'Dendrogram for {method} Linkage')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    
    # Generate the linkage matrix
    Z = linkage(X_blobs, method=method)  # Using the current method for dendrogram
    
    dendrogram(Z, leaf_rotation=90, leaf_font_size=12)
    plt.show()

# denogram and clustering results show together
for method in linkage_methods:  
    plt.figure(figsize=(12, 6))
    
    # Dendrogram
    plt.subplot(1, 2, 1)
    plt.title(f'Dendrogram for {method} Linkage')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    
    Z = linkage(X_blobs, method=method)
    dendrogram(Z, leaf_rotation=90, leaf_font_size=12)
    
    # Clustering results
    plt.subplot(1, 2, 2)
    clustering = AgglomerativeClustering(n_clusters=4, linkage=method)
    labels = clustering.fit_predict(X_blobs)
    
    plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels, s=30, cmap='viridis')
    plt.title(f'Hierarchical Clustering with {method} Linkage')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
#
# %%


import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN
# Generate synthetic data for DBSCAN clustering
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
plt.scatter(X_moons[:, 0], X_moons[:, 1], s=30, cmap='viridis')
plt.title('Generated Data for DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()      

X_circles, _ = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
plt.scatter(X_circles[:, 0], X_circles[:, 1], s=30, cmap='viridis')
plt.title('Generated Data for DBSCAN Clustering (Circles)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()  


# %%
