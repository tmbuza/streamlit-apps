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
    st.write("To explore the Linear Regression model, visit the app using the link below:")
    st.markdown("[Open Linear Regression App](https://tmbuza-linear-regression-demo.streamlit.app/)")
elif option == "Logistic Regression":
    st.write("### Logistic Regression Model")
    st.write("The Logistic Regression model will be available soon.")
elif option == "Decision Tree":
    st.write("### Decision Tree Model")
    st.write("The Decision Tree model will be available soon.")
else:
    st.write("Please select a model from the dropdown to begin.")