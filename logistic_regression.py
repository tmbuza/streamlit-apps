import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Title and Introduction
st.title("Logistic Regression Model")
st.write("""
This app demonstrates a basic Logistic Regression model using synthetic data for binary classification. 
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
    X = np.random.rand(n, 2)  # Two random features
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary target based on feature sum
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

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display model accuracy
st.write("### Model Performance Metrics")
st.write(f"**Accuracy**: {accuracy:.2f}")
st.write("""
The accuracy metric indicates the proportion of correct predictions. An accuracy closer to 1.0 signifies a higher 
rate of accurate predictions by the model.
""")

# Display the confusion matrix
st.write("### Confusion Matrix")
st.write("""
The confusion matrix provides an overview of actual vs. predicted values for each class. It helps identify 
where the model is making correct or incorrect predictions.
""")

# Plotting the confusion matrix
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="viridis", 
            xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
st.pyplot(fig)

# Learn more section
st.write("""
---

### Learn More

For a deeper understanding of Logistic Regression and its application in classification tasks, check out the full **Practical Guide to Mastering Machine Learning Techniques**.

[Read the full practical guide here](https://complexdatainsights.com/machine-learning/chapter3)

This guide covers various machine learning models, including Logistic Regression, and provides practical examples of model implementation, evaluation, and deployment.
""")