import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Title and description
st.title("Linear Regression Model")
st.write("""
This app demonstrates a basic Linear Regression model using synthetic data. 
Adjust the sample size below to generate a dataset and view model performance.
""")

# Sidebar for input options
sample_size = st.sidebar.slider("Sample Size", min_value=50, max_value=500, value=100, step=10)

# Generate synthetic data function (with caching for efficiency)
@st.cache_data
def generate_data(n):
    # Creates a synthetic dataset with a linear relationship and some noise
    np.random.seed(0)
    X = 2 * np.random.rand(n, 1)
    y = 4 + 3 * X + np.random.randn(n, 1) * 0.5
    return pd.DataFrame({"Feature": X.flatten(), "Target": y.flatten()})

# Generate data based on sample size input
data = generate_data(sample_size)

# Display sample of synthetic data
st.write("### Synthetic Data Sample")
st.write(data.head())

# Prepare data for training
X = data[["Feature"]]
y = data["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display model performance metrics
st.write("### Model Metrics")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R-squared (R²): {r2:.2f}")

# Plotting actual vs. predicted values
st.write("### Scatter Plot of Predictions vs. Actual Values")
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.scatter(X_test, y_pred, color="red", label="Predicted")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression Predictions")
plt.legend()
st.pyplot(plt)