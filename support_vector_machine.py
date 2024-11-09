import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Title and Introduction
st.title("Support Vector Machine (SVM) Classifier")
st.write("""
This app demonstrates a basic Support Vector Machine (SVM) classifier using synthetic data.
Adjust the sample size below to generate a dataset, train the SVM model, and evaluate its performance.
SVMs are powerful for binary classification, aiming to find the optimal boundary (hyperplane) that best separates classes.
""")

# Sidebar with a heading
st.sidebar.header("Control Panel")

# Sample size control in the sidebar
sample_size = st.sidebar.slider("Sample Size", min_value=50, max_value=500, value=100, step=10)

# Generate synthetic data
@st.cache_data
def generate_data(n):
    np.random.seed(0)
    X = np.random.rand(n, 2)  # Generate two random features
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary target based on a threshold
    return pd.DataFrame({"Feature 1": X[:, 0], "Feature 2": X[:, 1], "Target": y})

data = generate_data(sample_size)

# Display synthetic data sample
st.write("### Synthetic Data Sample")
st.write("Below is a preview of the generated data based on the selected sample size.")
st.write(data.head())

# Split data into training and test sets
X = data[["Feature 1", "Feature 2"]]
y = data["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
model = SVC(kernel="linear", random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display model accuracy
st.write("### Model Performance Metrics")
st.write(f"**Accuracy**: {accuracy:.2f}")
st.write("""
Accuracy reflects the model's ability to correctly classify samples. 
A high accuracy score indicates that the SVM model effectively separates the classes.
""")

# Display the confusion matrix
st.write("### Confusion Matrix")
st.write("""
The confusion matrix below shows actual vs. predicted values for each class, helping to identify correct and incorrect predictions.
""")

# Plotting the confusion matrix
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Purples", 
            xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
st.pyplot(fig)

# Visualizing the decision boundary (optional for 2D data)
st.write("### Decision Boundary Visualization")
st.write("""
The plot below illustrates the decision boundary determined by the SVM, along with the actual data points.
Points near the boundary are the 'support vectors' that the SVM uses to maximize class separation.
""")

# Prepare a mesh grid for plotting the decision boundary
h = .02  # step size in the mesh
x_min, x_max = X["Feature 1"].min() - 0.1, X["Feature 1"].max() + 0.1
y_min, y_max = X["Feature 2"].min() - 0.1, X["Feature 2"].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
fig, ax = plt.subplots(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.3)
plt.scatter(X["Feature 1"], X["Feature 2"], c=y, edgecolors="k", cmap="coolwarm")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM Decision Boundary")
st.pyplot(fig)

# Learn more section
st.write("""
---

### Learn More

For an in-depth understanding of Support Vector Machines (SVMs) and other machine learning models, explore the full **Practical Guide to Mastering Machine Learning Techniques**.

[Read the full practical guide here](https://complexdatainsights.com/machine-learning/chapter6)

This guide covers a wide range of machine learning models, including SVMs, and provides hands-on examples for model training, evaluation, and deployment.
""")