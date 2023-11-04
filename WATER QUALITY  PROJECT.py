#!/usr/bin/env python
# coding: utf-8

# ## Data Gathering

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("water_potability.csv")
data


# In[3]:


# o - quality , 1 - no quality 


# In[4]:


data.isnull().sum()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


# Binary classification -- 0 and 1


# In[8]:


# Multi classification -- 1 , 2 , 3 , 4 , 5 -- gd,bad,excellent,..


# In[9]:


data['ph'].describe()


# In[10]:


data['ph'].isnull().sum()


# In[11]:


#fill the ph column with mean()
data['ph'].fillna(data['ph'].mean(),inplace=True)


# In[12]:


data['ph'].isnull().sum()


# In[13]:


data['Sulfate'].describe()


# In[14]:


data['Sulfate'].isnull().sum()


# In[15]:


#fill the ph column with mean()
data['Sulfate'].fillna(data['Sulfate'].mean(),inplace=True)


# In[16]:


data['Sulfate'].isnull().sum()


# In[17]:


data['Trihalomethanes'].describe()


# In[18]:


data['Trihalomethanes'].isnull().sum()


# In[19]:


data['Trihalomethanes'].fillna(data['Trihalomethanes'].mean(),inplace=True)


# In[20]:


data['Trihalomethanes'].isnull().sum()


# In[21]:


data.info()


# In[61]:


# ALL NULL VALUES FILL AT A TIME
data.fillna(data.mean(),inplace=True)
data.isnull().sum()


# In[23]:


data.Potability.value_counts()


# In[24]:


sns.countplot(data['Potability'])
plt.show()


# In[54]:


#Pie chart -->we can go for above method or plt.pie()

data['Potability'].value_counts()
#we will split index and values
a = data['Potability'].value_counts().index.tolist()
a
b = data['Potability'].value_counts().values.tolist()
b


# In[56]:


1998/(1998+1278)


# In[59]:


plt.pie(b,labels=a,colors=['yellow','blue'],autopct='%.1f%%')
plt.title("Potability Ratio")
plt.legend()
plt.show()


# In[25]:


# seaborn
plt.figure(figsize=(8,8))
sns.heatmap(data.corr(),annot=True,cmap='crest')
plt.show()


# In[26]:


# boxplot

data.boxplot(figsize=(12,6))
plt.show()


# In[60]:


# histogram
data.hist(figsize=(12,12),color='red')
plt.show()


# In[28]:


x = data.iloc[:,:-1].values
 # independent variable


# In[29]:


y = data.iloc[:,-1].values
 #dependent variable


# In[30]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)


# In[31]:


x_train.shape


# In[32]:


x_test.shape


# In[33]:


#feature scaling
from sklearn.preprocessing import StandardScaler # 0 and 1
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[34]:


x_train


# In[35]:


x_test


# In[36]:



#LogisticRegression
from sklearn.linear_model import LogisticRegression


# In[37]:


# object of logisticRegression
model = LogisticRegression()


# In[44]:


# Training
model.fit(x_train , y_train)


# In[47]:


# prediction
y_pred = model.predict(x_test)


# In[66]:


from sklearn.metrics import mean_squared_error,r2_score
print(mean_squared_error(y_test,y_pred))
print(r2_score(y_test,y_pred))


# In[40]:


from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(y_test , y_pred))
print(confusion_matrix(y_test,y_pred))


# In[41]:


#we can reduce false negative - 371 - water quality is not purely but the prediction is water is quality
                                                                                                                                      


# In[ ]:




