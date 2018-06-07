
# coding: utf-8

# In[1]:


from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import numpy as np


# In[2]:


# Data and labels
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']


# In[3]:


clf = tree.DecisionTreeClassifier()
clf1 = SVC()
clf2 = Perceptron()
clf3 = KNeighborsClassifier()
clf4 = GaussianNB()


# In[4]:


clf.fit(X,Y)
clf1.fit(X,Y)
clf2.fit(X,Y)
clf3.fit(X,Y)
clf4.fit(X,Y)


# In[5]:


pred_tree = clf.predict(X)
acc_tree = accuracy_score(Y,pred_tree) * 100
print 'Accuracy for DecisionTree: {}'.format(acc_tree)


# In[12]:


pred_svc = clf1.predict(X)
acc_svc = accuracy_score(Y,pred_svc) * 100
print 'Accuracy for SVC: {}'.format(acc_svc)


# In[13]:


pred_perc = clf2.predict(X)
acc_perc = accuracy_score(Y,pred_perc) * 100
print 'Accuracy for Perceptron: {}'.format(acc_perc)


# In[14]:


pred_neigh = clf3.predict(X)
acc_neigh = accuracy_score(Y,pred_neigh) * 100
print 'Accuracy for KNeighborsClassifier: {}'.format(acc_neigh)


# In[15]:


pred_gaus = clf4.predict(X)
acc_gaus = accuracy_score(Y,pred_gaus) * 100
print 'Accuracy for GaussianNB: {}'.format(acc_gaus)


# In[17]:


maxval = np.argmax([acc_tree,acc_svc,acc_perc,acc_neigh,acc_gaus])
classifiers = {0: 'SVM', 1: 'Perceptron', 2: 'KNN', 3: 'GaussianNB'}
print('Best gender classifier is {}'.format(classifiers[maxval]))

