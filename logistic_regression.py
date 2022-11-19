#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('titanic_train.csv')


# In[3]:


train.head()


# ### missing Data
# ### بررسی دیتا های از دست رفته

# In[4]:


train.info()


# In[5]:


train.isnull()


# In[6]:


train.isnull().sum()


# In[7]:


687/891


# In[8]:


#sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. We'll probably drop this later, or change it to another feature like "Cabin Known: 1 or 0"
# 
# Let's continue on by visualizing some more of the data!

# In[9]:


#sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')


# In[10]:


#sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[11]:


#sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[12]:


sns.distplot(train['Age'].dropna(),kde=False,bins=30)


# In[13]:


train['Age'].plot.hist(bins=30)


# In[14]:


sns.countplot(x='SibSp',data=train)


# In[15]:


sns.countplot(x='Parch',data=train)


# ## Data Cleaning
# We want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation). However we can be smarter about this and check the average age by passenger class. For example:

# In[16]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[17]:


train[train['Pclass']==1]['Age'].mean()


# In[18]:


train[train['Pclass']==1]


# In[19]:


train[train['Pclass']==1]['Age']


# In[20]:


print(train[train['Pclass']==1]['Age'].mean())
print(train[train['Pclass']==2]['Age'].mean())
print(train[train['Pclass']==3]['Age'].mean())


# We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these average age values to impute based on Pclass for Age.

# In[21]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 38

        elif Pclass == 2:
            return 29

        else:
            return 25

    else:
        return Age


# Now apply that function!

# In[22]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[23]:


#Now let's check that heat map again!
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[24]:


#Great! Let's go ahead and drop the Cabin column
#and the row in Embarked that is NaN.
train.drop('Cabin',axis=1,inplace=True)


# In[25]:


train['Embarked'].value_counts()


# In[26]:


train['Embarked'].replace(np.nan, 'S', inplace=True)


# In[27]:


train.info()


# In[28]:


train.head()


# ## Converting Categorical Features
# We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.

# In[29]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
pclass = pd.get_dummies(train['Pclass'],drop_first=True)


# In[30]:


sex


# In[31]:


train.drop(['PassengerId','Sex','Embarked','Name','Ticket','Pclass'],axis=1,inplace=True)


# In[32]:


train = pd.concat([train,sex,embark,pclass],axis=1)


# In[33]:


train.head()


# #### Great! Our data is ready for our model!

# ### Building a Logistic Regression model
# Let's start by splitting our data into a training set and test set (there is another test.csv file that you can play around with in case you want to use all this data for training).
# 
# Train Test Split

# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),  train['Survived'], test_size=0.30,  random_state=101)


# ## Training and Predicting
# 

# In[36]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[37]:


predictions = logmodel.predict(X_test)


# In[38]:


predictions


# In[39]:


y_test


# ### Let's move on to evaluate our model!
# ## evaluation
# Let's bring Confusion Matrix!
# 
# 
# 
# 

# In[40]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[41]:


from sklearn.metrics import classification_report


# In[42]:


print(classification_report(y_test,predictions))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




