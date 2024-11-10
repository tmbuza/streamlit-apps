import pandas as pd
import streamlit as st
from apyori import apriori
import plotly.express as px

# Sidebar for parameter settings
st.sidebar.title("Apriori Parameters")
min_support = st.sidebar.slider("Minimum Support", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
min_confidence = st.sidebar.slider("Minimum Confidence", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
min_lift = st.sidebar.slider("Minimum Lift", min_value=1.0, max_value=5.0, value=1.0, step=0.1)

# Title and description
st.title("Apriori Rule Learning")
st.write("""
    This application demonstrates Apriori Rule Learning, an association rule learning algorithm.
    It finds interesting relationships between items in a dataset. In this example, we use a small transactional dataset.
    We will apply the Apriori algorithm to discover association rules and interpret the results.
""")

# Example transactional data (1 means the item is present, 0 means it's absent)
data = {
    'Bread': [1, 1, 0, 1, 1],
    'Milk': [1, 1, 1, 0, 1],
    'Butter': [0, 1, 1, 1, 1],
    'Cheese': [1, 0, 0, 1, 1],
}

# Convert the dataset into a list of transactions
df = pd.DataFrame(data)

# Display the original dataset for reference
st.write("### Dataset Sample", df)

# Convert the DataFrame to a list of transactions (1s as presence, 0s as absence)
transactions = []
for index, row in df.iterrows():
    transaction = [item for item, value in row.items() if value == 1]
    transactions.append(transaction)

# Display the list of transactions for transparency
st.write("### List of Transactions", transactions)

# Apply Apriori algorithm to find frequent itemsets with dynamic parameters
st.write("### Running Apriori Algorithm")
rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift, min_length=2)

# Convert the results into a list
rules_list = list(rules)

# If rules are found, process and display them
if len(rules_list) == 0:
    st.write("No association rules found based on the given thresholds.")
else:
    st.write("### Association Rules Found:")

    # Initialize lists to store rule details
    rule_antecedents = []
    rule_consequents = []
    rule_support = []
    rule_confidence = []
    rule_lift = []

    # Extract the rule details
    for rule in rules_list:
        for ordered_stat in rule.ordered_statistics:
            antecedents = ', '.join(list(ordered_stat.items_base))
            consequents = ', '.join(list(ordered_stat.items_add))
            confidence = ordered_stat.confidence
            lift = ordered_stat.lift
            support = rule.support

            rule_antecedents.append(antecedents)
            rule_consequents.append(consequents)
            rule_support.append(support)
            rule_confidence.append(confidence)
            rule_lift.append(lift)

    # Create a DataFrame to display the rules
    rules_df = pd.DataFrame({
        'Antecedents': rule_antecedents,
        'Consequents': rule_consequents,
        'Support': rule_support,
        'Confidence': rule_confidence,
        'Lift': rule_lift
    })

    # Display the rules in a table
    st.write(rules_df)

    # Interactive Scatter Plot for Support, Confidence, and Lift
    st.write("### Interactive Visualization of Association Rules")
    fig = px.scatter(
        rules_df,
        x="Support",
        y="Confidence",
        size="Lift",
        color="Lift",
        hover_data=['Antecedents', 'Consequents'],
        title="Association Rules - Support vs Confidence",
        labels={"Support": "Support", "Confidence": "Confidence", "Lift": "Lift"}
    )
    st.plotly_chart(fig)

    # Application and Interpretation Section
    st.write("### Application and Interpretation of Association Rules")
    st.write("""
    In association rule learning, we aim to find interesting relationships between items in a dataset. 
    The rules consist of:
    - **Antecedents**: Items that imply the presence of other items.
    - **Consequents**: Items that are likely to be present if the antecedents are present.

    ### Key Metrics:
    - **Support**: This measures how frequently the itemsets appear in the dataset. A higher support means the itemset is more common in the transactions.
    - **Confidence**: This measures the likelihood that the consequent item will be bought when the antecedent item is bought. Higher confidence means stronger certainty that the rule is valid.
    - **Lift**: This measures the strength of a rule by comparing the confidence of the rule with the expected confidence assuming the items are independent. A lift value greater than 1 indicates a positive association, meaning the items are likely to occur together more often than expected by chance.

    ### Example Interpretation:
    - If a rule shows that `Bread -> Milk` with high confidence, this indicates that when customers buy bread, they are likely to buy milk as well.
    - A lift greater than 1 suggests that the two items appear together more often than by random chance, meaning the two items are associated.
    
    ### Practical Use:
    These rules can be applied in various domains such as:
    - **Market Basket Analysis**: To suggest additional items to customers based on their purchasing history.
    - **Recommendation Systems**: To recommend products or services to users based on their previous behavior.
    - **Inventory Management**: To understand product associations and optimize stock levels accordingly.

    **In conclusion**, association rules help identify relationships between products, which can be used to drive business decisions, such as product placement, recommendations, and cross-selling strategies.
    """)