#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd
import arrow
from dateutil.parser import parse
from dateutil.tz import gettz
import datetime
from pprint import pprint
import urllib,time,datetime
import sys


# In[2]:


def get_quote_data(symbol='iwm', data_range='1d', data_interval='60m'):
    res = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={data_range}&interval={data_interval}'.format(**locals()))
    data = res.json()
    body = data['chart']['result'][0]
    dt = datetime.datetime
    dt = pd.Series(map(lambda x: arrow.get(x).to('EST').datetime.replace(tzinfo=None), body['timestamp']), name='dt')
    df = pd.DataFrame(body['indicators']['quote'][0], index=dt)
    dg = pd.DataFrame(body['timestamp'])
    return df.loc[:, ('open', 'high', 'low', 'close', 'volume')]


# In[3]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




