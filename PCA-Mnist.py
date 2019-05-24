
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.sparse


# In[2]:


# Import MNIST Dataset
from sklearn.datasets import fetch_mldata
mnist=fetch_mldata('MNIST original')
mnist


# In[10]:


type(mnist)


# In[3]:


X,y=mnist["data"],mnist["target"]
X.shape


# In[4]:


y=y.astype(int)
y


# In[5]:


# Dataset is splitted to Train and Test Set.
X_train, X_test=X[:60000],X[60000:]
y_train, y_test=y[:60000],y[60000:]


# In[6]:


# Train a Random Forest classifier on the dataset and time how long it takes,
# then evaluate the resulting model on the test set.
 
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import accuracy_score

start = time.time()

rfclf=RandomForestClassifier(n_estimators=500, random_state=42)
clf = rfclf
clf.fit(X_train,y_train)

end = time.time()
print(end - start)

y_pred=clf.predict(X_test)
print(clf.__class__.__name__,accuracy_score(y_test,y_pred))


# In[17]:


# Use PCA (Principal Component Analysis) to reduce 
# the dataset’s dimensionality, with an explained variance ratio of 95%. 
from sklearn.decomposition import PCA

precent_of_variance_explained = .95
pca = PCA(n_components=precent_of_variance_explained)
pca_data = pca.fit_transform(X)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)


# In[20]:


# Train a new Random Forest classifier on the reduced dataset
# and see how long it takes again.

X_train, X_test=X_pca[:60000],X_pca[60000:]
y_train, y_test=y[:60000],y[60000:]

start = time.time()

rfclf=RandomForestClassifier(n_estimators=500, random_state=42)
clf = rfclf
clf.fit(X_train,y_train)

end = time.time()
print(end - start)

y_pred=clf.predict(X_test)
print(clf.__class__.__name__,accuracy_score(y_test,y_pred))

