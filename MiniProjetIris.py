#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


df=pd.read_csv('IRIS_ Flower_Dataset.csv')
df.head()


# In[3]:


df.describe()


# In[4]:


df.info()


# In[5]:


df['species'].value_counts()


# In[6]:


df.isnull().sum()


# In[7]:


df['sepal_length'].hist()


# In[8]:


df['sepal_width'].hist()


# In[9]:


df['petal_length'].hist()


# In[10]:


df['petal_width'].hist()


# In[11]:


colors=['red','orange','blue']
species=['Iris-setosa','Iris-versicolor','Iris-virginica']


# In[12]:


for i in range(3):
    x=df[df['species']==species[i]]
    plt.scatter(x['sepal_length'],x['sepal_width'],c=colors[i],label=species[i])
    plt.xlabel("sepal_length")
    plt.ylabel("sepal_width")
    plt.legend()


# In[13]:


for i in range(3):
    x=df[df['species']==species[i]]
    plt.scatter(x['petal_length'],x['petal_width'],c=colors[i],label=species[i])
    plt.xlabel("petal_length")
    plt.ylabel("petal_width")
    plt.legend()


# In[14]:


for i in range(3):
    x=df[df['species']==species[i]]
    plt.scatter(x['sepal_length'],x['petal_length'],c=colors[i],label=species[i])
    plt.xlabel("sepal_length")
    plt.ylabel("petal_length")
    plt.legend()


# In[15]:


for i in range(3):
    x=df[df['species']==species[i]]
    plt.scatter(x['sepal_width'],x['petal_width'],c=colors[i],label=species[i])
    plt.xlabel("sepal_width")
    plt.ylabel("petal_width")
    plt.legend()


# In[16]:


df.corr()


# In[17]:


corr=df.corr()
fig,ax=plt.subplots(figsize=(5,4))
sns.heatmap(corr,annot=True,ax=ax,cmap='coolwarm')


# In[18]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[53]:


df['species']=le.fit_transform(df['species'])
df.head()


# In[54]:


from sklearn.model_selection import train_test_split
X=df.drop(columns=['species'])
Y=df['species']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.30)


# In[62]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=110, random_state=42)


# In[63]:


model.fit(x_train,y_train)


# In[64]:


print("Accuracy: ",model.score(x_test,y_test)*100)


# In[65]:


y_pred = model.predict(x_test)

# Print accuracy
accuracy = model.score(x_test, y_test)
print("Accuracy:", accuracy * 100)

# Plot Training and Validation Curves
# Note: RandomForestClassifier doesn't provide built-in history, so we won't have validation curves.
# We can, however, visualize feature importances.

# Feature Importances
feature_importances = model.feature_importances_
feature_names = X.columns

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances, align='center')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.show()


# In[66]:


from sklearn.svm import SVC
model=SVC(kernel='linear', C=1.1, random_state=42)


# In[67]:


model.fit(x_train,y_train)


# In[68]:


print("Accuracy: ",model.score(x_test,y_test)*100)


# In[76]:


# Tracer la frontière de décision
plt.scatter(x_train['sepal_length'], x_train['sepal_width'], c=y_train, cmap='viridis', label='Train')
plt.scatter(x_test['sepal_length'], x_test['sepal_width'], c=y_test, cmap='viridis', marker='x', label='Test')

# Tracer la frontière de décision du SVM
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Créer une grille pour évaluer le modèle
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))

# Prédire la classe pour chaque point de la grille
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Tracer la frontière de décision et les marges du SVM
ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Marquer les vecteurs de support
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k', marker='o', label='Support Vectors')

plt.title('SVM Decision Boundary')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




