#!/usr/bin/env python
# coding: utf-8

# # Build a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data. Use a dataset such as the Bank Marketing dataset from the UCI Machine Learning Repository.

# In[1]:


from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
bank_marketing = fetch_ucirepo(id=222) 
  
# data (as pandas dataframes) 
X = bank_marketing.data.features 
y = bank_marketing.data.targets 
  
# metadata 
print(bank_marketing.metadata) 
  
# variable information 
print(bank_marketing.variables) 


# In[2]:


X['deposit']=y
X


# In[3]:


X=X.dropna()
X.isna().sum().sum()
X['job'].value_counts()


# In[4]:


X


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# In[6]:


lb=LabelEncoder()


# In[7]:


import pandas as pd
X= pd.DataFrame(X)
X1=X.apply(lb.fit_transform)


# In[8]:


X1


# In[9]:


# Split the dataset into training and testing sets
y=X1['deposit']
x=X1.drop(['deposit'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)


# In[10]:


# Initialize the decision tree classifier
clf = DecisionTreeClassifier(criterion="gini",max_depth=5,min_samples_split=15)


# In[11]:


# Train the classifier on the training data
clf.fit(X_train, y_train)


# In[12]:


# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[13]:


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Extract feature names
cols = list(X_train.columns)

# Plot the decision tree
plt.figure(figsize=(20,20))  # Adjust the figure size as needed
plot_tree(clf, feature_names=cols, filled=True)
plt.show()

