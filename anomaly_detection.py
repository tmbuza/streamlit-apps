import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt

# Title and description
st.title("Anomaly Detection with Isolation Forest")
st.write("""
    This application demonstrates Anomaly Detection using the Isolation Forest algorithm.
    We use a small dataset where we introduce some anomalies, and the algorithm detects them.
""")

# Generate sample data with anomalies
# Creating a simple 2D dataset with some anomalies
np.random.seed(42)
normal_data = np.random.randn(100, 2)  # Normal points (100)
outliers = np.random.uniform(low=-6, high=6, size=(10, 2))  # Anomalies (10)

# Combine the data
data = np.vstack([normal_data, outliers])

df = pd.DataFrame(data, columns=['Feature 1', 'Feature 2'])

st.write("### Sample Data", df.head())

# Applying Isolation Forest
st.write("### Running Isolation Forest for Anomaly Detection")
clf = IsolationForest(contamination=0.1)  # Assume 10% of the points are anomalies
pred = clf.fit_predict(df)

# Add anomaly predictions to the DataFrame (1 = normal, -1 = anomaly)
df['Anomaly'] = pred

# Visualizing the data and anomalies
st.write("### Anomaly Detection Results")

# Scatter plot of the data points
plt.figure(figsize=(8, 6))
# Fix the color mapping: red for anomalies (-1), blue for normal (1)
plt.scatter(df['Feature 1'], df['Feature 2'], c=df['Anomaly'], cmap='coolwarm_r', edgecolors='k', s=100)
plt.title("Anomaly Detection with Isolation Forest")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Anomaly (-1) / Normal (1)")

# Show the plot in Streamlit
st.pyplot(plt)

# Application and Interpretation Section
st.write("### Application and Interpretation of Anomaly Detection")

st.write("""
    **Anomaly Detection** is useful for identifying unusual or rare data points that could indicate problems, fraud, or new, interesting phenomena.
    
    ### Key Concepts:
    - **Isolation Forest** works by isolating observations that are few and different. The algorithm identifies anomalies by constructing trees where the anomalies are more likely to be isolated.
    - **Contamination**: In this example, we set the contamination parameter to 0.1, which means we expect 10% of the data to be anomalies.
    - **Anomalies** are marked with `-1` and normal points with `1`.
    
    ### Practical Use:
    Anomaly detection can be applied to:
    - **Fraud Detection**: Identifying unusual credit card transactions that may indicate fraud.
    - **Network Security**: Detecting unusual behavior in network traffic that might indicate a security breach.
    - **Manufacturing**: Identifying defective products in a production line.
    - **Healthcare**: Detecting unusual medical readings that could indicate health issues.
    
    ### Example Interpretation:
    In the scatter plot above:
    - Points marked in **red** represent anomalies.
    - The model's task is to find out which of the data points deviate significantly from the normal pattern.
""")