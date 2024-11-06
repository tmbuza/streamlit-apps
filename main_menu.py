import streamlit as st

# Main title
st.title("Model Deployment Hub")

# Description
st.write("""
Welcome to the Model Deployment Hub! Select a model from the options below to see an example deployment with synthetic data. 
Each model app demonstrates basic functionality and visualization for educational purposes.
""")

# Model selection
option = st.selectbox(
    "Choose a model to explore:",
    ("Select a model", "Linear Regression", "Logistic Regression", "Decision Tree")
)

# Navigation based on selected model
if option == "Linear Regression":
    st.write("### Linear Regression Model")
    st.write("To use the Linear Regression model, open `linear_regression.py`.")
    st.write("You can view synthetic data, basic metrics, and visualizations related to linear regression.")
elif option == "Logistic Regression":
    st.write("### Logistic Regression Model")
    st.write("To use the Logistic Regression model, open `logistic_regression.py`.")
    st.write("This app will show synthetic data, model metrics, and basic classification visualizations.")
elif option == "Decision Tree":
    st.write("### Decision Tree Model")
    st.write("Explore the Decision Tree model by opening `decision_tree.py`.")
    st.write("View synthetic data and model results with decision tree-based insights.")
else:
    st.write("Please select a model from the dropdown to begin.")