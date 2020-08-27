# Load the libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# load data file
groceries_data = pd.read_csv('Groceries.csv', delimiter=',')
groceries_data = groceries_data.dropna()

# Q1-a) Create a dataset which contains the number of distinct items in each customer’s market basket. Draw a
# histogram of the number of unique items.  What are the median, the 25th percentile and the 75th percentile in this
# histogram?
print('\n<========(Q1-a)========>')
items_group_count = groceries_data.groupby(['Customer'])['Item'].count()
items_group_count = items_group_count.sort_values()
print(f'groceries data : \n {groceries_data}')
print(f'\ncustomers\' distinct items : \n {items_group_count}')

median = np.percentile(items_group_count, 50)
quartile1 = np.percentile(items_group_count, 25)
quartile3 = np.percentile(items_group_count, 75)
print(f'\nmedian (quartile 2): {median}')
print(f'25% (quartile 1): {quartile1}')
print(f'75% (quartile 3): {quartile3}')

#ploting histogram
plt.hist(items_group_count)
plt.axvline(quartile1, color='orange', linewidth=1, alpha=1)
plt.axvline(median, color='red', linewidth=1, alpha=1)
plt.axvline(quartile3, color='orange', linewidth=1, alpha=1)
plt.title('Histogram of the number of unique items')
plt.xlabel('Item Label')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Q1-b) If you are interested in the k-itemsets which can be found in the market baskets of at least seventy five (
# 75) customers.  How many itemsets can you find?  Also, what is the largest k value among your itemsets?
print('\n<========(Q1-b)========>')
# Convert the Sale Receipt data to the Item List format
items_group_count = groceries_data.groupby(['Customer'])['Item'].count()
item_group_list = groceries_data.groupby(['Customer'])['Item'].apply(list).values.tolist()
# Convert the Item List format to the Item Indicator format
te = TransactionEncoder()
te_ary = te.fit(item_group_list).transform(item_group_list)
item_indicator = pd.DataFrame(te_ary, columns=te.columns_)
min_support = 75 / len(items_group_count)
# Find the frequent itemsets
frequent_itemset = apriori(item_indicator, min_support=min_support, use_colnames=True, max_len=median)
k_itemset_largest = len(frequent_itemset['itemsets'][len(frequent_itemset) - 1])

print(f"frequent item sets : \n{frequent_itemset['itemsets']}")
print(f"\ntotal number of item sets found = {frequent_itemset.shape[0]}")
print(f"\nthe largest value of k = {k_itemset_largest}")

# Q1-c) Find out the association rules whose Confidence metrics are at least 1%.  How many association rules have you
# found?  Please be reminded that a rule must have a non-empty antecedent and a non-empty consequent.  Also,
# you do not need to show those rules.
print('\n<========(Q1-c)========>')
# Discover the association rules
asso_rules = association_rules(frequent_itemset, metric="confidence", min_threshold=0.01)
print(f"Total number of association rules found = {asso_rules.shape[0]}")

# Q1-d) Graph the Support metrics on the vertical axis against the Confidence metrics on the horizontal axis for the
# rules you found in (c).  Please use the Lift metrics to indicate the size of the marker.
print('\n<========(Q1-d)========>')
plt.scatter(asso_rules['confidence'], asso_rules['support'], c=asso_rules['lift'], s=asso_rules['lift'])
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.grid(True)
cbar = plt.colorbar()
cbar.set_label('lift', labelpad=+1)
plt.show()

# Q1-e)
# List the rules whose Confidence metrics are at least 60%.  Please include their Support and Lift metrics.
print('\n<========(Q1-e)========>')
# Discover the association rules
asso_rules = association_rules(frequent_itemset, metric="confidence", min_threshold=0.6)
print(f'association rules : \n{asso_rules.to_string()}')
