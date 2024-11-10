import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules

# Title and description
st.title("Apriori Rule Learning")
st.write("""
    This application demonstrates Association Rule Learning using the Apriori algorithm.
    It finds interesting relationships between items in a dataset. Here, we use a small 
    transactional dataset as an example and apply the Apriori algorithm to discover association rules.
""")

# Example transactional data (1 means the item is present, 0 means it's absent)
data = {
    'Bread': [1, 1, 0, 1, 1],
    'Milk': [1, 1, 1, 0, 1],
    'Butter': [0, 1, 1, 1, 1],
    'Cheese': [1, 0, 0, 1, 1],
}

# Convert the dataset into a DataFrame and display
df = pd.DataFrame(data)
st.write("### Sample Dataset", df)

# Convert the DataFrame to frequent itemsets using the Apriori algorithm
st.write("### Running Apriori Algorithm")
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# Calculate association rules using MLxtend's `association_rules` function
# Add the required 'num_itemsets' parameter to avoid the TypeError
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5, num_itemsets=len(frequent_itemsets))

# If rules are found, display them; otherwise, show a message
if rules.empty:
    st.write("No association rules found based on the given thresholds.")
else:
    st.write("### Association Rules Found:")
    st.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Application and Interpretation Section
st.write("### Application and Interpretation of Association Rules")
st.write("""
    In association rule learning, we aim to find interesting relationships between items in a dataset. 
    The rules consist of:
    - **Antecedents**: Items that imply the presence of other items.
    - **Consequents**: Items that are likely to be present if the antecedents are present.

    ### Key Metrics:
    - **Support**: This measures how frequently the itemsets appear in the dataset.
    - **Confidence**: This measures the likelihood that the consequent item will be bought when the antecedent item is bought.
    - **Lift**: This measures the strength of a rule by comparing the confidence of the rule with the expected confidence assuming the items are independent.

    ### Example Interpretation:
    - If a rule shows that `Bread -> Milk` with high confidence, this indicates that when customers buy bread, they are likely to buy milk as well.
    - A lift greater than 1 suggests that the two items appear together more often than by random chance, meaning the two items are associated.

    ### Practical Use:
    These rules can be applied in various domains such as:
    - **Market Basket Analysis**: To suggest additional items to customers based on their purchasing history.
    - **Recommendation Systems**: To recommend products or services to users based on their previous behavior.
    - **Inventory Management**: To understand product associations and optimize stock levels accordingly.
""")