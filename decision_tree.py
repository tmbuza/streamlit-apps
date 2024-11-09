import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Title and Introduction
st.title("Decision Tree Classifier Model")
st.write("""
This app demonstrates a basic Decision Tree classifier using synthetic data for binary classification. 
Adjust the sample size below to generate a dataset, train the model, and assess its performance.
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
st.write("Here is a preview of the generated data based on your selected sample size.")
st.write(data.head())

# Split data into training and test sets
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

# Display model accuracy
st.write("### Model Performance Metrics")
st.write(f"**Accuracy**: {accuracy:.2f}")
st.write("""
The accuracy metric indicates the proportion of correct predictions made by the model.
An accuracy closer to 1.0 signifies a high rate of accurate predictions.
""")

# Display the confusion matrix
st.write("### Confusion Matrix")
st.write("""
The confusion matrix provides an overview of actual vs. predicted values for each class. 
It helps identify where the model is making correct or incorrect predictions.
""")

# Plotting the confusion matrix
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
st.pyplot(fig)

# Visualize the Decision Tree
st.write("### Decision Tree Visualization")
st.write("""
The tree structure below shows how the model splits data based on feature values. 
This helps to understand the model's decision-making process at each node, making it interpretable.
""")
fig, ax = plt.subplots(figsize=(8, 6))
plot_tree(model, filled=True, feature_names=["Feature 1", "Feature 2"], class_names=["Class 0", "Class 1"], ax=ax)
st.pyplot(fig)

# Learn more section
st.write("""
---

### Learn More

For a deeper understanding of Decision Trees and their application in classification tasks, check out the full **Practical Guide to Mastering Machine Learning Techniques**.

[Read the full practical guide here](https://complexdatainsights.com/machine-learning/chapter4)

This guide covers various machine learning models, including Decision Trees, and provides practical examples of model implementation, evaluation, and deployment.
""")