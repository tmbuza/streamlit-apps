import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt


# Title and description
st.title("Anomaly Detection")
st.write("""
    This application demonstrates anomaly detection using an Isolation Forest model.
    Anomalies, or outliers, are data points that deviate significantly from other observations.
    Detecting them is important in various fields, such as fraud detection, network security, and quality control.
""")

# Generate sample data with anomalies
st.write("### Generating Sample Data with Anomalies")
st.write("""
    In this section, we create a synthetic dataset that includes both normal data points 
    and a few anomalies. These anomalies are intentionally crafted as points that lie far 
    from the normal distribution of data, allowing the model to detect them as outliers.
""")

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
st.write("""
    The Isolation Forest algorithm is a tree-based model that identifies anomalies by isolating 
    observations that differ significantly from others in the dataset. Unlike traditional density- or 
    distance-based methods, Isolation Forest isolates anomalies by partitioning the data. Anomalies 
    are isolated faster as they are fewer and farther from the dense regions of normal data. 

    Here, we set the contamination level to 5%, meaning the model will consider the top 5% of points 
    most different from others as anomalies.
""")

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