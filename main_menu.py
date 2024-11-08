import streamlit as st

# Main title
st.title("Model Deployment Hub")

# Description
st.write("""
Welcome to the Model Deployment Hub! Select a model from the options below to see an example deployment with synthetic data. 
Each model app demonstrates basic functionality and visualization for educational purposes.
""")

# Model selection
st.write("### Choose a Model to Explore:")
option = st.radio(
    "",  # Empty label to keep the radio button section clean
    ["Select a model", "Linear Regression", "Logistic Regression", "Decision Tree"]
)

# Navigation based on selected model
if option == "Linear Regression":
    st.write("### Linear Regression Model")
    st.write("To explore the Linear Regression model, visit the app using the link below:")
    st.markdown("[Open Linear Regression App](https://tmbuza-linear-regression-demo.streamlit.app/)")
elif option == "Logistic Regression":
    st.write("### Logistic Regression Model")
    st.write("To explore the Logistic Regression model, visit the app using the link below:")
    st.markdown("[Open Logistic Regression App](https://tmbuza-logistic-regression-demo.streamlit.app/)")
elif option == "Decision Tree":
    st.write("### Decision Tree Model")
    st.write("To explore the Decision Tree model, visit the app using the link below:")
    st.markdown("[Open Decision Tree App](https://tmbuza-decision-tree-demo.streamlit.app/)")
elif option == "Select a model":
    st.write("Please select a model from the options above to begin.")