# prompt: same output as above set, but without the frozenset line

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
dataset = [['Milk', 'Onion','Nutmeg','Kideny Beans','Eggs','Yogurt'],
           ['Dill','Onion','Nutmeg','Kideny Beans','Eggs','Yogurt'],
           ['Milk','Apple','Kideny Beans','Eggs'],
           ['Milk','Unicorn','Corn','Kideny Beans','Yogurt'],
           ['Corn','Onion','Kideny Beans','Ice Cream','Eggs']
]
!pip install mlxtend
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
# Association Rule Mining
res = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
# remove frozenset
res['antecedents'] = [tuple(x) for x in res['antecedents']]
res['consequents'] = [tuple(x) for x in res['consequents']]
res1 = res[['antecedents','consequents','support','support', 'confidence','lift']]
res2 = res1[res1['confidence']>=1]
for i in res2.index:
  print(res2['antecedents'][i], '=>', res2['consequents'][i])
