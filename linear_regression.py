import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px

# Title and introduction
st.title("Linear Regression Model")
st.write("""
This app demonstrates a simple Linear Regression model using synthetic data with two features.
Linear regression is a model that fits a plane to data points, helping to predict the relationship 
between multiple input features and a target variable.
""")

# Sample size control in the main panel
st.write("### Step 1: Adjust Sample Size")
st.write("Use the slider below to change the number of data points generated for the model.")
sample_size = st.slider("Sample Size", min_value=50, max_value=500, value=100, step=10)

# Generate synthetic data with two features
@st.cache_data
def generate_data(n):
    np.random.seed(0)
    X1 = 2.5 * np.random.rand(n, 1)  # First feature
    X2 = 1.5 * np.random.rand(n, 1)  # Second feature
    y = 5 + 2 * X1 + 3 * X2 + np.random.randn(n, 1) * 0.5  # Target with both features influencing it
    return pd.DataFrame({"Feature1": X1.flatten(), "Feature2": X2.flatten(), "Target": y.flatten()})

data = generate_data(sample_size)

# Display synthetic data sample
st.write("### Step 2: Synthetic Data Sample")
st.write("Here is a small sample of the synthetic data generated based on the sample size you chose.")
st.write(data.head())

# Split data into features (X) and target (y)
X = data[["Feature1", "Feature2"]]
y = data["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
st.write("### Step 3: Training the Model")
st.write("""
The app uses a Linear Regression model to fit a plane in 3D space based on two input features.
This is a step up from single-feature linear regression and better represents real-world scenarios, where models 
often include multiple features to make predictions. Once trained, the model can make predictions on unseen data.
""")
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display metrics in main panel
st.write("### Step 4: Model Evaluation Metrics")
st.write("""
The following metrics help evaluate the model's accuracy:
- **Mean Absolute Error (MAE)**: The average absolute difference between actual and predicted values.
- **Mean Squared Error (MSE)**: The average squared difference, which penalizes larger errors.
- **R² Score**: The proportion of the variance in the target variable that is predictable from the input features.
""")
st.write(f"- Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"- Mean Squared Error (MSE): {mse:.2f}")
st.write(f"- R² Score: {r2:.2f}")

import plotly.graph_objects as go

# Interactive plot using Plotly
st.write("""
### Step 5: Visualizing Predictions with Plotly
The plot below compares actual target values (blue) with predicted values (red).
Plotly is used here for interactive visualization, allowing you to hover over data points and explore the relationship 
between actual and predicted values. This makes it easier to assess the model’s performance in an interactive way.
""")

# Create the Plotly figure
fig = go.Figure()

# Add scatter plot for actual target values
fig.add_trace(go.Scatter(
    x=y_test, y=y_test, mode='markers', name='Actual Target', 
    marker=dict(color='blue', size=8),
    hovertemplate="<b>Actual Target:</b> %{x}<br><b>Predicted Target:</b> %{y}<br><extra></extra>"  # Custom hover text
))

# Add scatter plot for predicted target values
fig.add_trace(go.Scatter(
    x=y_test, y=y_pred, mode='markers', name='Predicted Target', 
    marker=dict(color='red', size=8),
    hovertemplate="<b>Actual Target:</b> %{x}<br><b>Predicted Target:</b> %{y}<br><extra></extra>"  # Custom hover text
))

# Update layout for better readability
fig.update_layout(
    title="Actual vs Predicted Target Values",
    xaxis_title="Actual Values",
    yaxis_title="Predicted Values",
    showlegend=True,
    hoverlabel=dict(
        font=dict(size=10)  # Adjust hover text font size
    )
)

# Show the plot in the Streamlit app
st.plotly_chart(fig)


st.write("""

### Learn More

For a deeper understanding of the Linear Regression technique and other machine learning models, check out the full **Practical Guide to Mastering Machine Learning Techniques**.

[Read the full practical guide here](https://complexdatainsights.com//machine-learning/chapter2)

This guide offers a hands-on exploration of key concepts, including model implementation, training, evaluation, and deployment. Perfect for beginners and those looking to deepen their knowledge of machine learning.
""")