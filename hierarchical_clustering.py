import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Title and Description
st.title("Hierarchical Clustering Demo")
st.write(
    """
    This application demonstrates Hierarchical Clustering, an unsupervised learning technique that organizes data into a tree-like structure.
    """
)

# Generate Synthetic Dataset
@st.cache_data
def load_data():
    X, _ = make_blobs(n_samples=150, centers=3, cluster_std=1.0, random_state=42)
    df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
    return df

df = load_data()
st.write("### Sample Data", df.head())

# User Input for Number of Clusters
st.sidebar.write("### Hierarchical Clustering Settings")
n_clusters = st.sidebar.slider("Select Number of Clusters:", 2, 6, 3)
linkage_method = st.sidebar.selectbox("Select Linkage Method:", ["single", "complete", "average", "ward"])

# Perform Hierarchical Clustering
@st.cache_resource
def perform_hierarchical_clustering(data, n_clusters, linkage_method):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    clusters = model.fit_predict(data)
    Z = linkage(data, method=linkage_method)
    return clusters, Z

clusters, Z = perform_hierarchical_clustering(df, n_clusters, linkage_method)

# Display Clustered Data
df["Cluster"] = clusters
st.write("### Clustered Data Sample", df.head())

# Plotting Dendrogram
st.write("### Dendrogram Plot")
fig, ax = plt.subplots(figsize=(10, 6))
dendrogram(Z, truncate_mode="level", p=5, ax=ax)
ax.set_xlabel("Data Points")
ax.set_ylabel("Distance")
st.pyplot(fig)

# Plotting Hierarchical Clusters
st.write("### Hierarchical Clustering Plot")
fig, ax = plt.subplots()
for cluster in range(n_clusters):
    subset = df[df["Cluster"] == cluster]
    ax.scatter(subset["Feature 1"], subset["Feature 2"], label=f"Cluster {cluster}", alpha=0.6)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()
st.pyplot(fig)