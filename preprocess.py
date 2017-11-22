
# coding: utf-8

# ## Preprocess
# 
# ### download raw data from [kaggle: web traffic time series forecasting]
#  (https://www.kaggle.com/c/web-traffic-time-series-forecasting/data)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
from random import sample

def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org',page)
    if res:
        return res[0][0:2]
    return 'na'


#read the csv into a dataframe
train = pd.read_csv('./data/input/train_2.csv',sep=',', error_bad_lines=False, warn_bad_lines=True).fillna(0)
train['lang'] = train.Page.map(get_language)
print(Counter(train.lang))


lang_sets = {}
lang_sets['en'] = train[train.lang=='en'].iloc[:,0:-1]
lang_sets['ja'] = train[train.lang=='ja'].iloc[:,0:-1]
lang_sets['de'] = train[train.lang=='de'].iloc[:,0:-1]
lang_sets['na'] = train[train.lang=='na'].iloc[:,0:-1]
lang_sets['fr'] = train[train.lang=='fr'].iloc[:,0:-1]
lang_sets['zh'] = train[train.lang=='zh'].iloc[:,0:-1]
lang_sets['ru'] = train[train.lang=='ru'].iloc[:,0:-1]
lang_sets['es'] = train[train.lang=='es'].iloc[:,0:-1]

# top10median = {}
# for key in lang_sets:
#     if key == 'en':
#         print(key)
#         lang_sets[key]['sum'] = lang_sets[key].sum(axis=1)
#         lang_sets[key]['mean'] = lang_sets[key].mean(axis=1)
#         lang_sets[key]['median'] = lang_sets[key].median(axis=1)
#         top10median[key] = lang_sets[key].sort_values('median',ascending = False).groupby('Page').head(10)[0:-3]


idx = lang_sets['en'].index.values.tolist()
for i in sample(idx,5):
    ts = lang_sets['en'].loc[i]
    data = pd.DataFrame({'ds':ts.index[1:], 'y':ts.values[1:]})
    data.to_csv('./data/'+'%s.csv' % ts.values[0], sep=',',encoding='utf-8',header=True,index=False)


