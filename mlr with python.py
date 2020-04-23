#!/usr/bin/env python
# coding: utf-8

# In[1]:


import statsmodels.formula.api as smf


# In[2]:


import pandas as pd


# In[3]:


from sklearn.model_selection import train_test_split


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl 
import seaborn as sns 
import matplotlib.pyplot as plt 
import statsmodels.formula.api as smf 
import statsmodels.graphics.api as smg 
import pandas as pd 
import numpy as np 
import patsy 
from statsmodels.graphics.correlation import plot_corr 
from sklearn.model_selection import train_test_split 
plt.style.use('seaborn')


# In[5]:


rawBostonData = pd.read_csv('https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter02/Dataset/Boston.csv')


# In[6]:


rawBostonData.head()


# In[8]:


rawBostonData = rawBostonData.dropna()


# In[9]:


rawBostonData = rawBostonData.drop_duplicates()


# In[10]:


list(rawBostonData.columns)


# In[11]:


renamedBostonData = rawBostonData.rename(columns = {'CRIM':'crimeRatePerCapita', 
 ' ZN ':'landOver25K_sqft', 
 'INDUS ':'non-retailLandProptn', 
 'CHAS':'riverDummy', 
 'NOX':'nitrixOxide_pp10m', 
 'RM':'AvgNo.RoomsPerDwelling', 
 'AGE':'ProptnOwnerOccupied', 
 'DIS':'weightedDist', 
 'RAD':'radialHighwaysAccess', 
 'TAX':'propTaxRate_per10K', 
 'PTRATIO':'pupilTeacherRatio', 
 'LSTAT':'pctLowerStatus', 
 'MEDV':'medianValue_Ks'}) 
renamedBostonData.head() 


# In[12]:


renamedBostonData.info()


# In[13]:


renamedBostonData.describe(include=[np.number]).T


# In[14]:


X = renamedBostonData.drop('crimeRatePerCapita', axis = 1)
y = renamedBostonData[['crimeRatePerCapita']]
seed = 10
test_data_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_data_size, random_state = seed)
train_data = pd.concat([X_train, y_train], axis = 1)
test_data = pd.concat([X_test, y_test], axis = 1)


# In[16]:


multiLinearModel = smf.ols(formula='crimeRatePerCapita ~ pctLowerStatus + radialHighwaysAccess +medianValue_Ks + nitrixOxide_pp10m', data=train_data)


# In[17]:


multiLinearModelResult=multiLinearModel.fit()


# In[18]:


print(multiLinearModelResult.summary())


# In[ ]:




