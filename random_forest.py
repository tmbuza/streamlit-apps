import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Title and Introduction
st.title("Random Forest Classifier Model")
st.write("""
This app demonstrates a Random Forest Classifier using synthetic data. 
Adjust the sample size below to generate a dataset, train the model, and evaluate its performance.
Random forests are an ensemble learning method that uses multiple decision trees to improve prediction accuracy and robustness.
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

# Train the Random Forest model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display model accuracy
st.write("### Model Performance Metrics")
st.write(f"**Accuracy**: {accuracy:.2f}")
st.write("""
The accuracy metric shows the proportion of correct predictions. 
With Random Forests, high accuracy indicates strong performance from the ensemble of decision trees.
""")

# Display the confusion matrix
st.write("### Confusion Matrix")
st.write("""
The confusion matrix below shows actual vs. predicted values for each class. 
It helps identify where the model made correct or incorrect predictions.
""")

# Plotting the confusion matrix
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", 
            xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
st.pyplot(fig)

# Feature Importance Visualization
st.write("### Feature Importance")
st.write("""
Random Forests allow us to examine the importance of each feature in the prediction process.
The plot below shows how much each feature contributed to the model's decision-making.
""")
feature_importances = model.feature_importances_
features = ["Feature 1", "Feature 2"]

# Plotting feature importances
fig, ax = plt.subplots()
sns.barplot(x=feature_importances, y=features, palette="viridis")
plt.xlabel("Importance")
plt.title("Feature Importance in Random Forest Classifier")
st.pyplot(fig)

# Learn more section
st.write("""
---

### Learn More

For a deeper understanding of Random Forests and their application in classification tasks, check out the full **Practical Guide to Mastering Machine Learning Techniques**.

[Read the full practical guide here](https://complexdatainsights.com/machine-learning/chapter5)

This guide covers various machine learning models, including Random Forests, and provides practical examples of model implementation, evaluation, and deployment.
""")