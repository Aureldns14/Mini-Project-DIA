#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()


# In[51]:


train.describe()


# In[52]:


train.info()


# In[99]:


sns.countplot(x='Survived', data=train)

plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Number of Passengers')
plt.title('Survival Count in Titanic Training Data')

plt.show()


# In[101]:


sns.countplot(x='Pclass', data=train)

plt.xlabel('Passenger Class')
plt.ylabel('Number of Passengers')
plt.title('Passenger Class Distribution in Titanic Training Data')


# In[102]:


train_data = pd.read_csv('train.csv')

sns.countplot(x='Sex', data=train)

plt.xlabel('Sex')
plt.ylabel('Number of Passengers')
plt.title('Passenger Sex Distribution in Titanic Training Data')


# In[103]:


sns.countplot(x='SibSp', data=train)

plt.xlabel('Number of Siblings/Spouses Aboard')
plt.ylabel('Number of Passengers')
plt.title('Distribution of Siblings/Spouses Aboard in Titanic Training Data')

plt.show()


# In[104]:


sns.countplot(x='Parch', data=train)

plt.xlabel('Number of Parents/Children Aboard')
plt.ylabel('Number of Passengers')
plt.title('Distribution of Parents/Children Aboard in Titanic Training Data')

plt.show()


# In[105]:


sns.countplot(x='Embarked', data=train)

plt.xlabel('Port of Embarkation')
plt.ylabel('Number of Passengers')
plt.title('Passenger Distribution by Embarked Port in Titanic Training Data')

plt.show()


# In[60]:


sns.distplot(train['Age'])


# In[61]:


sns.distplot(train['Fare'])


# In[62]:


class_fare = train.pivot_table(index='Pclass', values='Fare')
class_fare.plot(kind='bar')
plt.xlabel('Pclass')
plt.ylabel('Avg. Fare')
plt.xticks(rotation=0)
plt.show()


# In[63]:


class_fare = train.pivot_table(index='Pclass', values='Fare', aggfunc=np.sum)
class_fare.plot(kind='bar')
plt.xlabel('Pclass')
plt.ylabel('Total Fare')
plt.xticks(rotation=0)
plt.show()


# In[64]:


sns.barplot(data=train, x='Pclass', y='Fare', hue='Survived')


# In[65]:


sns.barplot(data=train, x='Survived', y='Fare', hue='Pclass')


# In[66]:


train_len = len(train)
# combine two dataframes
df = pd.concat([train, test], axis=0)
df = df.reset_index(drop=True)
df.head()


# In[67]:


df.isnull().sum()


# In[68]:


df = df.drop(columns=['Cabin'], axis=1)


# In[69]:


df['Age'].mean()


# In[70]:


df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())


# In[71]:


df['Embarked'].mode()[0]


# In[72]:


df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


# In[73]:


sns.distplot(df['Fare'])


# In[74]:


df['Fare'] = np.log(df['Fare']+1)


# In[75]:


sns.distplot(df['Fare'])


# In[76]:


corr = df.corr()
plt.figure(figsize=(15, 9))
sns.heatmap(corr, annot=True, cmap='coolwarm')


# In[77]:


df.head()
df = df.drop(columns=['Name', 'Ticket'], axis=1)
df.head()


# In[78]:


from sklearn.preprocessing import LabelEncoder
cols = ['Sex', 'Embarked']
le = LabelEncoder()

for col in cols:
    df[col] = le.fit_transform(df[col])
df.head()


# In[79]:


train = df.iloc[:train_len, :]
test = df.iloc[train_len:, :]


# In[80]:


X = train.drop(columns=['PassengerId', 'Survived'], axis=1)
y = train['Survived']


# In[81]:


from sklearn.model_selection import train_test_split, cross_val_score
# classify column
def classify(model):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(x_train, y_train)
    print('Accuracy:', model.score(x_test, y_test))
    
    score = cross_val_score(model, X, y, cv=5)
    print('CV Score:', np.mean(score))


# In[92]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model)


# In[93]:


model = LGBMClassifier()
model.fit(X, y)


# In[94]:


test.head()


# In[95]:


X_test = test.drop(columns=['PassengerId', 'Survived'], axis=1)


# In[96]:


X_test.head()


# In[97]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model)


# In[ ]:





# In[ ]:





# In[ ]:




