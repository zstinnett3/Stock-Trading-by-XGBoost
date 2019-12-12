#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


#Loads a custom library to load stck data
import pandas as pd
import Functions.RetrieveIntradayData as ret


# In[2]:


#Read in previously saved data
cron = pd.read_csv()

#Set the datetime column as the index
cron.set_index('dt',inplace=True)


#Uses a custom function to load stock data from Google Finance. Goes back only 60days.
cron2 = ret.get_quote_data('cron', '60d', '15m')

#Concatenates the new data with the old
cron = pd.concat([cron,cron2],sort=True)

cron.drop_duplicates(inplace=True)

cron.reset_index(inplace=True)
cron.to_csv()
