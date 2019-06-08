
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.sparse


# In[2]:


from sklearn.datasets import fetch_mldata
mnist=fetch_mldata('MNIST original')
mnist


# In[3]:


X,y=mnist["data"],mnist["target"]
X.shape


# In[4]:


y=y.astype(int)
y


# In[8]:


X_train1 = X[:5000]
y_train1 = y[:5000]
X_validation1 = X[5000:6000]
y_validation1 = y[5000:6000]

X_train2 = X[6000:11000]
y_train2 = y[6000:11000]
X_validation2 = X[11000:12000]
y_validation2 = y[11000:12000]

X_train3 = X[12000:17000]
y_train3 = y[12000:17000]
X_validation3= X[17000:18000]
y_validation3 = y[17000:18000]

X_train4 = X[18000:23000]
y_train4 = y[18000:23000]
X_validation4= X[23000:24000]
y_validation4 = y[23000:24000]

X_train5 = X[24000:29000]
y_train5 = y[24000:29000]
X_validation5= X[29000:30000]
y_validation5 = y[29000:30000]

X_train6 = X[30000:35000]
y_train6 = y[30000:35000]
X_validation6= X[35000:36000]
y_validation6 = y[35000:36000]

X_train7 = X[36000:41000]
y_train7 = y[36000:41000]
X_validation7= X[41000:42000]
y_validation7= y[41000:42000]

X_train8 = X[42000:47000]
y_train8 = y[42000:47000]
X_validation8= X[47000:48000]
y_validation8= y[47000:48000]

X_train9 = X[42000:47000]
y_train9 = y[42000:47000]
X_validation9 = X[47000:48000]
y_validation9 = y[47000:48000]

X_train10 = X[48000:53000]
y_train10 = y[48000:53000]
X_validation10 = X[53000:54000]
y_validation10= y[53000:54000]

X_train11 = X[54000:59000]
y_train11 = y[54000:59000]
X_validation11 = X[59000:60000]
y_validation11 = y[59000:60000]


# In[9]:


X_train = np.concatenate((X_train1, X_train2, X_train3,X_train4,X_train5,X_train6, X_train7, X_train8,X_train9,X_train10,X_train11), axis=0)


# In[10]:


y_train = np.concatenate((y_train1, y_train2, y_train3,y_train4,y_train5,y_train6, y_train7, y_train8,y_train9,y_train10,y_train11), axis=0)


# In[11]:


X_validation = np.concatenate((X_validation1,X_validation2,X_validation3,X_validation4,X_validation5,X_validation6,X_validation7,X_validation8,X_validation9,X_validation10,X_validation11), axis =0)


# In[12]:


y_validation = np.concatenate((y_validation1,y_validation2,y_validation3,y_validation4,y_validation5,y_validation6,y_validation7,y_validation8,y_validation9,y_validation10,y_validation11), axis = 0)


# In[13]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

nbclf=MultinomialNB()
rfclf=RandomForestClassifier(n_estimators=500, random_state=42)
svm_clf = SVC(gamma=0.1, kernel='poly', random_state = 0)


# In[14]:


from sklearn.metrics import accuracy_score

clf = nbclf
clf.fit(X_train,y_train)
y_pred=clf.predict(X_validation)
print(clf.__class__.__name__,accuracy_score(y_validation,y_pred))


# In[15]:


clf = svm_clf
clf.fit(X_train,y_train)
y_pred=clf.predict(X_validation)
print(clf.__class__.__name__,accuracy_score(y_validation,y_pred))


# In[16]:


clf = rfclf
clf.fit(X_train,y_train)
y_pred=clf.predict(X_validation)
print(clf.__class__.__name__,accuracy_score(y_validation,y_pred))


# In[17]:


X_test = X[60000:]
y_test = y[60000:]


# In[20]:


#HARD VOTING CLASSIFIER
hard_voting_clf=VotingClassifier(estimators=[('rf',rfclf),('svc',svm_clf),('nb',nbclf)],
                           voting='hard')
clf = hard_voting_clf
clf.fit(X_train,y_train)
y_pred=clf.predict(X_validation)
print(clf.__class__.__name__,accuracy_score(y_validation,y_pred))


# In[21]:


#SVM ON TEST SET
clf = svm_clf
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(clf.__class__.__name__,accuracy_score(y_test,y_pred))

