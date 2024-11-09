import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Title and Description
st.title("Principal Component Analysis (PCA) Demo")
st.write(
    """
    This application demonstrates the concept of Principal Component Analysis (PCA),
    a dimensionality reduction technique. PCA projects data onto fewer dimensions while
    retaining as much variance as possible.
    """
)

# Load Sample Dataset
@st.cache_data
def load_data():
    data = load_iris(as_frame=True)
    df = data.frame
    return df

df = load_data()
st.write("### Sample Data", df.head())

# User Input for Number of Components
st.sidebar.write("### PCA Settings")
n_components = st.sidebar.slider("Select Number of Components:", 1, min(df.shape[1], 3), 2)

# Perform PCA
@st.cache_resource
def perform_pca(data, n_components):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(data)
    explained_variance = pca.explained_variance_ratio_
    return components, explained_variance

X = df.drop("target", axis=1)
components, explained_variance = perform_pca(X, n_components)

# Display PCA Results
st.write("### PCA Components")
pca_df = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)])
pca_df["target"] = df["target"]
st.write(pca_df.head())

# Plotting
st.write("### Variance Explained by Each Component")
st.bar_chart(explained_variance)

# Scatter Plot of First Two Principal Components
if n_components >= 2:
    st.write("### PCA Scatter Plot")
    fig, ax = plt.subplots()
    for target in pca_df["target"].unique():
        subset = pca_df[pca_df["target"] == target]
        ax.scatter(subset["PC1"], subset["PC2"], label=f"Class {target}", alpha=0.7)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend()
    st.pyplot(fig)