#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Functions.Indicators as ind
from inspect import getmembers, isfunction


# In[2]:


df = pd.read_csv()


# In[3]:


index_threshold = 19
price_threshold = 0.25


# In[4]:


df.reset_index(drop=True,inplace=True)


# In[5]:


df.rename(columns={'high':'High','low':'Low','open':'Open','close':'Close','dt':'Date', 'volume':'Volume'},inplace=True)


# In[6]:


#Calls
calls = []
for i in range(len(df['Open'])):
    counter = 1
    try:
        while True:
            if df.iloc[i+counter]['High'] - df.iloc[i]['Open'] > price_threshold and df.iloc[i+counter]['High'] > df.iloc[i]['Open']:
                calls.append(counter)
                break
            elif counter > index_threshold:
                calls.append(counter)
                break
            else:
                counter += 1
        
    except IndexError:
        calls.append(counter)
        pass


# In[7]:


#Puts
puts = []
for i in range(len(df['Open'])):
    counter = 1
    try:
        while True:
            if abs(df.iloc[i]['Open'] - df.iloc[i+counter]['Low'])  > price_threshold and df.iloc[i+counter]['Low'] < df.iloc[i]['Open']:
                puts.append(counter)
                break
            elif counter > index_threshold:
                puts.append(counter)
                break
            else:
                counter += 1
        
    except IndexError:
        puts.append(counter)
        pass


# In[8]:


df['CallTimeReqs'] = np.array(calls)
df['PutTimeReqs'] = np.array(puts)


# In[9]:


def optionaction(calltime,puttime):
    if calltime < puttime:
        return 'Call'
    elif calltime > puttime:
        return 'Put'
    else:
        return 'No Action'


# In[10]:


df['Target'] = df.apply(lambda x: optionaction(x.CallTimeReqs, x.PutTimeReqs), axis=1)


# In[11]:


df.drop(['CallTimeReqs','PutTimeReqs'], inplace=True, axis=1)


# In[12]:


indicators = [o for o in getmembers(ind) if isfunction(o[1])]


# In[13]:


for i in indicators:
    df[i[0]] = i[1](df).iloc[:,-1]


# In[14]:


df['psardiff'] = df['psar'] - df['Open']


# In[15]:


df.dropna(inplace=True, thresh=0.4)


# In[16]:


df.to_csv('C:\\Users\\zackarys\\Functions\\Stock Data\\prepared_data_cron.csv',index=None)


# In[ ]:





# In[ ]:





# In[ ]:




