import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r'C:\Users\KIIT\Documents\Assignment06\kmeans - kmeans_blobs.csv')
X = df.values

# Standardize the dataset
def normalize_features(X):
    mean_vals = np.mean(X, axis=0)
    std_vals = np.std(X, axis=0)
    return (X - mean_vals) / std_vals


X = normalize_features(X)

# Randomly initialize cluster centers
def initialize_random_centers(X, num_clusters):
    np.random.seed(42)# it sets a fixed seed for NumpY's RANDOM No generator and ensures the same random numbers are generated each time the function runs
    indices = np.random.choice(X.shape[0], num_clusters, replace=False)#selects random indices from the dataset (num_cluster)
    
    return X[indices]#extracts and lauta deta hai datapoints selected indices pe 



# data points ko allot krte jao nearest cluster ke hisaab se
def get_cluster_assignments(X, centers):
    distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)#euclidean distance calculate ho rha hai yaha par
    return np.argmin(distances, axis=1)#new axis add ho rhi hai (broadcasting) each data point ko subtract kr rhe hai har cluster ke centre ke saath 
# reshaping is tareke se ho rhi hai (n_samples,1,n_features)
                                   #(n_samples,num_clusters,n_features) 
                                   
    #result is a 1D array of shape (n_samples,), where each entry represents the assigned cluster.                               
                                   
                                   
                                   
# Compute new cluster centers by taking mean of them
def compute_new_centers(X, cluster_labels, num_clusters):
    # X yaha pe humne Numpy Array hai jo data points contain kr rha hai (n_samples,n_features)
    #cluster_labels ek NumPy array hai jiske andar n_samples.,n_features hao 
    #num_clusters hai total clusters K Means ke andar use jo kre hai 
    #for wala iterate krega 0 to num_clusters-1 tak jaha pe i current cluster index represent kr rha hai 
    #cluster_label==i ek Boolean mask create kr rha hai jo ki true ho rha hai for rows in X assigned to cluster i nahi toh false hoga
    #np.array(...) convert krdeta into a NumPy array of shape (num_clusters, n_features).  
    
    return np.array([X[cluster_labels == i].mean(axis=0) for i in range(num_clusters)])#generates a list of new cluster centers

# K-Means clustering algorithm
def kmeans_clustering(X, num_clusters, max_iterations=100, tolerance=1e-4):
    centers = initialize_random_centers(X, num_clusters)
    for _ in range(max_iterations):
        cluster_labels = get_cluster_assignments(X, centers)
        new_centers = compute_new_centers(X, cluster_labels, num_clusters)
        if np.linalg.norm(new_centers - centers) < tolerance:
            break#If the change is small (< tolerance), the algorithm stops early.
        centers = new_centers
    return cluster_labels, centers

# Run K-Means for k=2 and k=3
for num_clusters in [2, 3]:
    cluster_labels, centers = kmeans_clustering(X, num_clusters)
    
    # Plot the results
    plt.figure(figsize=(8, 6))
    for i in range(num_clusters):
        plt.scatter(X[cluster_labels == i, 0], X[cluster_labels == i, 1], label=f'Cluster {i}')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200, label='Centroids')#Uses an "X" marker for centroids.
    plt.title(f'K-Means Clustering (k={num_clusters})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
