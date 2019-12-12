#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import seaborn as sns
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:





# In[2]:


data = pd.read_csv()


# In[3]:


label_encoder = LabelEncoder()


# In[4]:


data['Target'] = label_encoder.fit_transform(data['Target'])


# In[5]:


data.dropna(inplace=True)
data.reset_index(inplace=True,drop=True)


# In[6]:


data.head()


# In[7]:


data['Day'] = data['Date'].apply(lambda x:pd.to_datetime(x).day)
data['Hour'] = data['Date'].apply(lambda x:pd.to_datetime(x).hour)


# In[8]:


data.drop(['Date'],inplace=True, axis=1)


# In[9]:


X = data.drop(['Target'],axis=1)
y = data['Target']


# In[ ]:





# In[10]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[11]:


from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer

clf_xgb = XGBClassifier(objective = 'binary:logistic')
param_dist = {'n_estimators': stats.randint(150, 500),
              'learning_rate': stats.uniform(0.01, 0.1),
              'subsample': stats.uniform(0.3, 0.7),
              'max_depth': range (2, 15, 1),
              'colsample_bytree': stats.uniform(0.5, 0.45),
              'min_child_weight': [1, 2, 3]
             }


# In[12]:


clf = RandomizedSearchCV(clf_xgb, param_distributions = param_dist, n_iter = 100, scoring ='f1_micro', error_score = 0, verbose = 0, n_jobs = -1)


# In[13]:


numFolds = 10
folds = KFold(n_splits = numFolds, shuffle = True)


# In[14]:


estimators = []
results = np.zeros(len(X))
score = 0.0
for train_index, test_index in folds.split(X):
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index].values.ravel(), y.iloc[test_index].values.ravel()
    clf.fit(X_train, y_train)
    results[test_index] = clf.predict(X_test)
    f1 = f1_score(y_test, results[test_index], average='micro')
    score += f1
    if f1>0.71:
        estimators.append((f1,clf))
score /= numFolds


# In[ ]:





# In[15]:


model=sorted(estimators, key=lambda tup: tup[0], reverse=True)[0][1]


# In[16]:


import pickle


# In[17]:


filename = 'C:\\Users\\zackarys\\Functions\\finalized_model_cron.sav'
pickle.dump(model, open(filename, 'wb'))


# In[18]:


y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
C = confusion_matrix(y_test, predictions)


# In[19]:


C


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




