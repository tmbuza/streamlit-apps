import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Title and description
st.title("K-Nearest Neighbors (KNN) Classifier Model")
st.write("""
This app demonstrates a basic K-Nearest Neighbors (KNN) classifier using synthetic data.
Adjust the sample size below to generate a dataset and view model performance.
""")

# Sidebar with a heading
st.sidebar.header("Control Panel")

# Sample size control in the sidebar
sample_size = st.sidebar.slider("Sample Size", min_value=50, max_value=500, value=100, step=10)
n_neighbors = st.sidebar.slider("Number of Neighbors (k)", min_value=1, max_value=15, value=3, step=1)

# Generate synthetic data
@st.cache_data
def generate_data(n):
    np.random.seed(0)
    X = np.random.rand(n, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Simple binary classification problem
    return pd.DataFrame({"Feature 1": X[:, 0], "Feature 2": X[:, 1], "Target": y})

data = generate_data(sample_size)

# Display synthetic data sample
st.write("### Synthetic Data Sample")
st.write(data.head())

# Split data
X = data[["Feature 1", "Feature 2"]]
y = data["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the KNN model
model = KNeighborsClassifier(n_neighbors=n_neighbors)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display metrics
st.write("### Model Metrics")
st.write(f"Accuracy: {accuracy:.2f}")

# Confusion matrix plot
st.write("### Confusion Matrix")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="coolwarm", xticklabels=["0", "1"], yticklabels=["0", "1"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
st.pyplot(fig)

# User information section
st.write("""
### About K-Nearest Neighbors (KNN)

KNN is a simple, instance-based learning algorithm that classifies data points based on their proximity to other points. It is ideal for classification tasks where similarity is a good predictor of class membership.
""")

# Learn more section
st.write("""
---

### Learn More

For a deeper understanding of the K-Nearest Neighbors (KNN) technique and other machine learning models, check out the full **Practical Guide to Mastering Machine Learning Techniques**.

[Read the full practical guide here](https://complexdatainsights.com/machine-learning/chapter7)

This guide offers a hands-on exploration of key concepts, including model implementation, training, evaluation, and deployment. Perfect for beginners and those looking to deepen their knowledge of machine learning.
""")