
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


# In[3]:


X,y=mnist["data"],mnist["target"]


# In[4]:


#K means on the MNIST dataset
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 10, random_state = 111)
kmeans.fit(X)


# In[5]:


correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
print(correct/len(X))


# In[6]:


# Use PCA (Principal Component Analysis) to reduce 
# the dataset’s dimensionality, with an explained variance ratio of 85%. 
from sklearn.decomposition import PCA

precent_of_variance_explained = .85
pca = PCA(n_components=precent_of_variance_explained)
pca_data = pca.fit_transform(X)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)


# In[8]:


#K means on the dimentionality reduced MNIST dataset
kmeans = KMeans(n_clusters = 10, random_state = 111)
kmeans.fit(X_pca)


# In[9]:


correct = 0
for i in range(len(X_pca)):
    predict_me = np.array(X_pca[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
print(correct/len(X_pca))

