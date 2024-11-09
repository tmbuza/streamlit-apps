import streamlit as st

# Function to load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load custom CSS (for styling)
local_css("style.css")

# Work in Progress Note
st.write("## Work in Progress")
st.write("""
    This portfolio is a work in progress as I continue to update and refine my projects and resources. 
    Please explore the current content, and stay tuned for new additions in the future!
""")

# Header with profile picture and introduction
st.header("TMB Professional Portfolio (Just Testing)")
st.image("images/portfolio0.jpg", width=150)  # Profile picture

# Introduction with a smoother transition
st.subheader("Hi, I'm Teresia Mrema Buza, PhD")
st.write("""
    I am a passionate Data Scientist and Bioinformatician with expertise in Microbiome research, 
    Data Science, Bioinformatics, and Machine Learning. My goal is to leverage data-driven insights 
    to drive breakthroughs in health and environmental sciences, particularly through machine learning 
    applications in bioinformatics.
""")

# Transition to Model Demos section
st.write("""
    Below, you'll find interactive demos showcasing various machine learning models that I've built 
    and deployed. These models represent my approach to solving real-world problems using data science.
    Feel free to explore them to get hands-on experience with the techniques and methodologies I work with.
""")

# Model Demos Section
st.write("## Model Demo Apps")
st.write("Explore live, interactive apps demonstrating machine learning models:")

# Navigation to model demos
model_options = [
    "Select a model", "Linear Regression", "Logistic Regression", "Decision Tree", 
    "Random Forest", "Support Vector Machine", "K-Nearest Neighbors", 
    "Naive Bayes", "Gradient Boosting"
]
option = st.selectbox("Choose a model to explore:", model_options)

# Display model-specific links based on selection
if option == "Linear Regression":
    st.write("### Linear Regression Model")
    st.markdown("[Open Linear Regression App](https://tmbuza-linear-regression-demo.streamlit.app/)")
elif option == "Logistic Regression":
    st.write("### Logistic Regression Model")
    st.markdown("[Open Logistic Regression App](https://tmbuza-logistic-regression-demo.streamlit.app/)")
elif option == "Decision Tree":
    st.write("### Decision Tree Model")
    st.markdown("[Open Decision Tree App](https://tmbuza-decision-tree-demo.streamlit.app/)")
elif option == "Random Forest":
    st.write("### Random Forest Model")
    st.markdown("[Open Random Forest App](https://tmbuza-random-forest-demo.streamlit.app/)")
elif option == "Support Vector Machine":
    st.write("### Support Vector Machine Model")
    st.markdown("[Open Support Vector Machine App](https://tmbuza-svm-demo.streamlit.app/)")
elif option == "K-Nearest Neighbors":
    st.write("### K-Nearest Neighbors Model")
    st.markdown("[Open K-Nearest Neighbors App](https://tmbuza-knn-demo.streamlit.app/)")
elif option == "Naive Bayes":
    st.write("### Naive Bayes Model")
    st.markdown("[Open Naive Bayes App](https://tmbuza-naive-bayes-demo.streamlit.app/)")
elif option == "Gradient Boosting":
    st.write("### Gradient Boosting Model")
    st.markdown("[Open Gradient Boosting App](https://tmbuza-gradient-boosting-demo.streamlit.app/)")

# Transition from Model Demos to Practical Guides
st.write("""
    After exploring the interactive model demos, you can deepen your understanding of the underlying 
    machine learning techniques by visiting the **Practical Guides** section. Each chapter in this section 
    offers a comprehensive, hands-on walkthrough of the models and their real-world applications.
""")

# Practical Guides Section
st.write("## Practical Guides")
st.write("""
    For an in-depth understanding and hands-on guidance on each model, visit the Practical Guides section.
    Each chapter provides a detailed walkthrough of a specific ML technique, offering both theory and 
    practical examples to enhance your skills.
""")
st.markdown("[Explore Practical Guides on Machine Learning](https://complexdatainsights.com/machine-learning/)")

# Transition from Practical Guides to Projects
st.write("""
    After gaining a deeper understanding of machine learning techniques through the Practical Guides, 
    you can explore some of my **projects** that apply these concepts to real-world challenges. 
    Each project showcases how these models are used to solve problems in areas such as health, bioinformatics, and more.
""")

# Projects Section
st.write("## Projects")
st.write("Explore some of my projects below, where I apply machine learning techniques to address complex challenges.")

# Project 1: Interactive Data Science Demo
st.write("### Project 1: Interactive Data Science Analysis")
st.write("An interactive demo that showcases data cleaning, exploratory analysis, and visualization techniques.")
st.write("[Launch Interactive Demo](https://tmbuza-linear-regression-demo.streamlit.app/)")
st.image("images/project1_image.png", width=300)

# Project 2: Machine Learning Model Demo
st.write("### Project 2: Machine Learning Model")
st.write("A hands-on machine learning model project demonstrating predictive techniques.")
st.write("[Launch Model Demo](https://tmbuza-logistic-regression-demo.streamlit.app/)")
st.image("images/project2_image.png", width=300)

# Case Studies Section
st.write("## Case Studies and Jupyter Notebooks")
st.write("""
    Detailed case studies and notebooks that demonstrate problem-solving and analytical thinking.
    - [Case Study 1: Data Analysis on Genomics Data](https://github.com/username/project1)
    - [Notebook: Predictive Modeling Techniques](https://github.com/username/project2)
    """)

# Testimonials Section
st.write("## Testimonials")
st.write("""
    "Working with Teresia was a fantastic experience. Her ability to interpret complex biological data was invaluable." - Colleague
    "Teresia is a brilliant mentor, always making complex data science concepts accessible and practical." - Student
    """)

# Sample Reports Section
st.write("## Sample Reports & Articles")
st.write("""
    Here are some articles and reports I’ve authored on microbiome bioinformatics and data science:
    - [Report on Bioinformatics Workflow](https://example.com/report)
    - [Article on Machine Learning in Genomics](https://example.com/article)
    """)

# Career Goals / Vision Section
st.write("## Career Vision")
st.write("""
    My goal is to harness the power of machine learning and data science to drive groundbreaking advancements in bioinformatics. 
    I am particularly passionate about exploring health and environmental data to uncover insights that can lead to transformative discoveries.

    I aim to innovate within these fields by applying cutting-edge technologies to analyze complex biological and environmental data, and I am deeply committed to using these innovations to improve human health and the environment.

    In addition to my technical pursuits, I am deeply committed to mentoring the next generation of data scientists and bioinformaticians. Through writing practical guides and educational resources, I aim to empower others by providing clear, hands-on learning experiences. 

    My goal is to help aspiring data scientists develop the necessary skills to thrive in this rapidly evolving field, with a particular focus on machine learning, bioinformatics, and data science techniques. By mentoring through my guides, I hope to inspire others to make meaningful contributions to the world of health and environmental data analysis.
""")

# Resources Section
st.write("## Resources")
st.write("""
    For more, check out my work on [GitHub](https://github.com/username) or explore my detailed guides:
    - [Practical Machine Learning Guide](https://complexdatainsights.com/machine-learning/)
    - [Download My Resume](assets/resume.pdf)
    """)

# Footer Section
st.write("---")
st.write("Built using Streamlit and Python")
