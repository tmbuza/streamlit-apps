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
    "Select a model",  # Non-empty label for accessibility
    ["Select a model", "Linear Regression", "Logistic Regression", "Decision Tree", 
     "Random Forest", "Support Vector Machine", "K-Nearest Neighbors", "Naive Bayes", "Gradient Boosting"],
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
elif option == "Random Forest":
    st.write("### Random Forest Model")
    st.write("To explore the Random Forest model, visit the app using the link below:")
    st.markdown("[Open Random Forest App](https://tmbuza-random-forest-demo.streamlit.app/)")
elif option == "Support Vector Machine":
    st.write("### Support Vector Machine Model")
    st.write("To explore the Support Vector Machine model, visit the app using the link below:")
    st.markdown("[Open Support Vector Machine App](https://tmbuza-support-vector-machine-demo.streamlit.app/)")
elif option == "K-Nearest Neighbors":
    st.write("### K-Nearest Neighbors Model")
    st.write("To explore the K-Nearest Neighbors model, visit the app using the link below:")
    st.markdown("[Open K-Nearest Neighbors App](https://tmbuza-k-nearest-neighbor-demo.streamlit.app/)")
elif option == "Naive Bayes":
    st.write("### Naive Bayes Model")
    st.write("To explore the Naive Bayes model, visit the app using the link below:")
    st.markdown("[Open Naive Bayes App](https://tmbuza-naive-bayes-demo.streamlit.app/)")
elif option == "Gradient Boosting":
    st.write("### Gradient Boosting Model")
    st.write("To explore the Gradient Boosting model, visit the app using the link below:")
    st.markdown("[Open Gradient Boosting App](https://tmbuza-gradient-boosting-demo.streamlit.app/)")
elif option == "Select a model":
    st.write("Please select a model from the options above to begin.")
    