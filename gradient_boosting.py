import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Title and description
st.title("Gradient Boosting Model")
st.write("""
This app demonstrates a Gradient Boosting model using synthetic data. 
Adjust the sample size below to generate a dataset and view model performance.
""")

# Sidebar with a heading
st.sidebar.header("Control Panel")

# Sample size control in the sidebar
sample_size = st.sidebar.slider("Sample Size", min_value=50, max_value=500, value=100, step=10)

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

# Train the Gradient Boosting model
model = GradientBoostingClassifier(random_state=42)
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
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
st.pyplot(fig)

# Plot decision boundary
st.write("""
### Decision Boundary Visualization
In this section, we visualize the decision boundary of the Gradient Boosting classifier in 2D space.
The decision boundary is the line where the classifier changes its prediction between the two classes.
""")

# Create meshgrid for decision boundary plot
h = .02  # Step size in mesh
x_min, x_max = X["Feature 1"].min() - 1, X["Feature 1"].max() + 1
y_min, y_max = X["Feature 2"].min() - 1, X["Feature 2"].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict the class for each point in the meshgrid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(xx, yy, Z, alpha=0.75, cmap="coolwarm")

# Plot the training points
ax.scatter(X["Feature 1"], X["Feature 2"], c=y, s=30, edgecolors="k", cmap="coolwarm", marker="o")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title("Decision Boundary and Data Points")
st.pyplot(fig)

# Transition to full practical guide link
st.write("""
---

### Learn More

For a deeper understanding of the Gradient Boosting technique and other machine learning models, check out the full **Practical Guide to Mastering Machine Learning Techniques**.

[Read the full practical guide here](https://complexdatainsights.com//machine-learning/chapter9)

This guide offers a hands-on exploration of key concepts, including model implementation, training, evaluation, and deployment. Perfect for beginners and those looking to deepen their knowledge of machine learning.
""")