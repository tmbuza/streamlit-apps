import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Title and Description
st.title("K-Means Clustering Demo")
st.write(
    """
    This application demonstrates K-Means clustering, an unsupervised learning technique for grouping data points into clusters.
    """
)

# Generate Synthetic Dataset
@st.cache_data
def load_data():
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
    df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
    return df

df = load_data()
st.write("### Sample Data", df.head())

# User Input for Number of Clusters
st.sidebar.write("### K-Means Settings")
n_clusters = st.sidebar.slider("Select Number of Clusters:", 1, 10, 4)

# Perform K-Means Clustering
@st.cache_resource
def perform_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans.cluster_centers_

clusters, centers = perform_kmeans(df, n_clusters)

# Display Clustered Data
df["Cluster"] = clusters
st.write("### Clustered Data Sample", df.head())

# Plotting Clusters
st.write("### K-Means Clustering Plot")
fig, ax = plt.subplots()
for cluster in range(n_clusters):
    subset = df[df["Cluster"] == cluster]
    ax.scatter(subset["Feature 1"], subset["Feature 2"], label=f"Cluster {cluster}", alpha=0.6)
ax.scatter(centers[:, 0], centers[:, 1], c="black", marker="X", s=100, label="Centroids")  # Changed to black
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()
st.pyplot(fig)