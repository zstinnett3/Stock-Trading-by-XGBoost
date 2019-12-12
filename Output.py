#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle


# In[ ]:





# In[2]:


filename = 'finalized_model_cron.sav'
loaded_model = pickle.load(open(filename, 'rb'))


# In[3]:


data = pd.read_csv('prepared_data_cron.csv')


# In[4]:


data['Day'] = data['Date'].apply(lambda x:pd.to_datetime(x).day)
data['Hour'] = data['Date'].apply(lambda x:pd.to_datetime(x).hour)


# In[5]:


data.drop(['Date'],inplace=True, axis=1)


# In[6]:


X = data.drop(['Target'],axis=1)


# In[7]:


model_input = X[-1:]


# In[8]:


model_input


# In[9]:


loaded_model.predict(model_input)


# In[10]:


from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)


# In[ ]:




