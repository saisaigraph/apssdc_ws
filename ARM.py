#!pip install mlxtend

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import association_rules


import os
print(os.getcwd())
os.chdir('F:\\Locker\\Sai\\SaiHCourseNait\\DecBtch\\R_Datasets\\')
print(os.getcwd())

#data = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx')
data = pd.read_excel('Online Retail.xlsx')

data.head()

alst = data.columns
alst

alst = alst.str.strip()
alst

alst = alst.str.lower()
alst

alst = alst.str.replace(' ','-')
alst

alst = alst.str.replace('(','').str.replace(')','')
alst

# standardize the columns to 
data.head()

data.columns = data.columns.str.strip().str.lower().str.replace(' ','-').str.replace('(','').str.replace(')','')

data.head()


data.info()

print('Data dimension (row count, col count): {dim}'.format(dim= data. shape) )

print('Count of unique invoice numvers: {cnt}'. format (cnt=data. invoiceno.nunique()))

print('Count of unique customer ids: {cnt}'.format (cnt=data.customerid.nunique()))

data['invoiceno'].value_counts()

#*** 

# drop the rows that dont have invoice numbers and remove the credit transactions 
# (those with invoice numbers containing C).
len(data)

#data.dropna(axis=0, subset=['invoiceno'], inplace=True)
#len(data)

data.info()

data['invoiceno'] = data['invoiceno' ].astype('str')
data.info()

print(len(data))

data.head()

data = data[~data[ 'invoiceno'].str.contains('C')]
print(len(data))

data.head()

data['country'].unique()


len(data['country'].unique())

basket = (data[data['country']=='Australia']
			.groupby(['invoiceno', 'description' ])['quantity']
			.sum().unstack().reset_index().fillna(0)
			.set_index('invoiceno'))
basket.head()
''' a lot of zeros in the data but we also need to make sure any positive values are 
converted to a 1 and anything less the 0 is set to 0. 
apply one hot encoding of the data and remove the postage column; 
we are not going to explore postage.'''

print(basket.shape)

def encode_units(x):
	if x <= 0:
		return 0
	if x>=1:
		return 1

basket.head(2)

basket_sets = basket.applymap(encode_units)

basket_sets.head(2)

basket_sets[['POSTAGE']].head()

print(len(basket_sets.columns))

basket_sets.drop('POSTAGE', inplace=True, axis=1)

print(len(basket_sets.columns))

print(len(basket_sets))

basket_sets.head(2)

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# generate the rules with their corresponding support, confidence and lift
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
print (frequent_itemsets)
'''frequent_itemsets = apriori(basket_sets, min_support=0.1, use_colnames=True)
print (frequent_itemsets)'''


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()

rules.shape
# 800 rules

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=14)
rules.head()

rules.shape
#346,9

dfTemp = rules[['antecedents','consequents','support','confidence','lift']]
dfTemp

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.hist('confidence',grid=False,bins=30)
plt.title('Confidence')


rules.hist('lift',grid=False,bins=30)
plt.title('Lift')


#support=rules.as_matrix(columns=['support'])
#support

support = rules['support'].values
support

print(len(support))

#confidence=rules.as_matrix(columns=['confidence'])
#print(len(confidence))
confidence = rules['confidence'].values
confidence

#***
# scatter plot for support and confidence showing the association rules 
# (first 10 rules) for the data set.
import numpy as np
for i in range (len(support)) :
	support[i] = support[i] + 0.0025 * (np.random.randint(1,10) - 5)
	confidence[i] = confidence[i] + 0.0025 * (np.random.randint(1,10) - 5)

plt.scatter(support, confidence, alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()



df1 = rules[(rules['lift']>=6) & (rules['confidence']>=0.8)]
df1.head()

print(basket['RED RETROSPOT CAKE STAND'].sum())

print(basket['36 PENCILS TUBE RED RETROSPOT'].sum())
# compared to 385 numbers of 36 PENCILS TUBE RED RETROSPOT 
# but only 
# 73 numbers of RED RETROSPOT CAKE STAND so maybe 
# business has to take some strategy here to bring both at par.


# how the combinations vary by country of purchase. 
# check out what some popular combinations in France.
basket2 = (data[data['country'] =="France"].groupby(['invoiceno', 'description' ])['quantity'].sum().unstack().reset_index().fillna(0).set_index('invoiceno'))
basket_sets2 = basket2.applymap(encode_units)
basket_sets2.drop('POSTAGE', inplace=True, axis=1)
frequent_itemsets2 = apriori(basket_sets2, min_support=0.05, use_colnames=True)
print (frequent_itemsets2)
rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)
rules2[ (rules2['lift'] >= 4) & (rules2['confidence'] >= 0.5)]
 
print('Number of association {}'.format(rules2.shape[0]))
'''compare and prepare an analysis report. Based on the association rule that we 
have defined, we can find significant correlation among some of the products. 
The priori algorithm applied here with the certain threshold. 
experiment with different threshold value. 
Larger the lift means more interesting association. 
Association rules with high support are potentially interesting rules. 
Similarly, rules with high confidence would be interesting rules '''

print(basket2['RED RETROSPOT CAKE STAND'].sum())
print(basket2['36 PENCILS TUBE RED RETROSPOT'].sum())
