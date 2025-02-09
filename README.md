# K-Means Clustering Project

## Overview
This project implements the **K-Means clustering algorithm** from scratch using **NumPy** and **Pandas**. The algorithm groups similar data points into clusters based on their features. The dataset is standardized before clustering, and results are visualized using **Matplotlib**.

## Features Implemented
- **Data Preprocessing:** Standardizes the dataset.
- **Random Initialization:** Selects initial cluster centers randomly.
- **Cluster Assignment:** Assigns data points to the nearest cluster center.
- **Centroid Update:** Computes new cluster centers based on the mean of assigned points.
- **Convergence Check:** Stops iterations when cluster centers do not change significantly.
- **Visualization:** Plots the clustered data along with centroids.

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib

## How It Works
1. **Load Dataset:** Reads the CSV file (`kmeans_blobs.csv`) using Pandas.
2. **Normalize Features:** Standardizes the dataset to improve clustering.
3. **Initialize Centers:** Selects `k` random data points as initial cluster centers.
4. **Assign Clusters:** Computes the **Euclidean distance** and assigns each point to the nearest cluster.
5. **Update Centers:** Recalculates cluster centers as the mean of assigned points.
6. **Check Convergence:** Stops when the difference between old and new centers is minimal.
7. **Visualization:** Displays the clusters along with centroids for `k=2` and `k=3`.

## Running the Project
1. Ensure you have the required libraries installed:
   ```bash
   pip install numpy pandas matplotlib
   ```
2. Place the dataset (`kmeans_blobs.csv`) in the specified directory.
3. Run the Python script.
4. View the plotted clusters and centroids.

## Expected Output
- **Scatter plots** showing clusters with different colors.
- **Black 'X' markers** representing centroids.

## Applications
- Customer segmentation
- Image compression
- Anomaly detection
- Document clustering

 
This implementation demonstrates a **custom-built K-Means algorithm** and serves as a foundation for further exploration into clustering techniques.

