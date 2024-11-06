import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

# Title and description
st.title("Decision Tree Classifier Model")
st.write("""
This app demonstrates a basic Decision Tree classifier using synthetic data. 
Adjust the sample size below to generate a dataset and view model performance.
""")

# Sidebar for input options
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

# Train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
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

# Decision Tree visualization
st.write("### Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(8, 6))
plot_tree(model, filled=True, feature_names=["Feature 1", "Feature 2"], class_names=["0", "1"], ax=ax)
st.pyplot(fig)