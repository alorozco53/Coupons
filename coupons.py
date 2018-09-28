
# coding: utf-8

# # Coupons Challenge for Kueski Data Science Position
# https://www.kaggle.com/c/coupon-purchase-prediction/

# In[176]:



import pandas as pd
import numpy as np
import json
import codecs
import dateparser

from pprint import pprint
from random import randint
from datetime import datetime
from matplotlib import pyplot as plt


# ## Translation Japansese - English

# In[64]:


with codecs.open('data/translations.json', 'r') as f:
    translations = json.load(f)

def translate(jpn_word):
    try:
        return translations[jpn_word]
    except:
        return jpn_word


# In[37]:


print(translations)


# ## Reading and parsing of files using Pandas
#
# Japanese cells are translated, as well

# In[65]:


coupon_list_train = pd.read_csv('data/coupon_list_train.csv', index_col='COUPON_ID_hash')
coupon_list_train = coupon_list_train.apply(lambda row: row.apply(translate), axis=1)
print(coupon_list_train.head())


# In[35]:


user_list = pd.read_csv('data/user_list.csv', index_col='USER_ID_hash')
user_list = user_list.apply(lambda row: row.apply(translate), axis=1)
print(user_list.head())


# In[36]:


coupon_visit_train = pd.read_csv('data/coupon_visit_train.csv')
print(coupon_visit_train.head())


# In[100]:


def get_hour(row):
    datestring = row['I_DATE']
    date = dateparser.parse(datestring)
    row['I_DATE'] = date.timestamp()
    return row
print(coupon_visit_train.head().apply(lambda row: get_hour(row), axis=1))


# In[211]:


avgs = []
for i in range(1000):
    stats = []
    delta = 2000
    lower_b = randint(0, len(coupon_list_train) - delta)
    upper_b = lower_b + delta
    dates = [coupon_visit_train.I_DATE.iloc[i] for i in range(lower_b, upper_b)]
    times = np.array(list(map(lambda d: dateparser.parse(d).timestamp(), dates)))
    std = np.std(times)
    stats = [datetime.fromtimestamp(t) for t in times]
    stats = [conv.hour + (conv.minute / 60) for conv in stats]
    mean = np.mean(stats)
    if i % 10 == 0 and avgs:
        print('current mean:', np.mean(avgs))
    avgs.append(np.mean(mean))


# In[204]:


print(np.mean(avgs))
