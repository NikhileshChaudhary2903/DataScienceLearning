
# coding: utf-8

# In[335]:


import numpy as np
import sklearn
# get_ipython().magic('matplotlib inline')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import random_projection


## load training data from files
train_df = np.loadtxt('train.dat')
X_training = pd.DataFrame(train_df)

training_labels=np.loadtxt('train.labels')
Y_training=pd.DataFrame(training_labels)

test_df = np.loadtxt('test.dat')
X_test = pd.DataFrame(test_df)


# In[336]:


from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X_training, Y_training, random_state=0)

# In[337]:

#
# X_train.shape
#
#
# # In[338]:
#
#
# X_test.shape
#
#
# # In[339]:
#
#
# y_train.shape
#
#
# # In[340]:
#
#
# y_test.shape


# In[341]:


# from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_training)
# X_test = scaler.transform(X_test)


# In[333]:


# from sklearn.decomposition import PCA

# # Make an instance of the Model
# #pca = PCA(n_components=40,svd_solver='randomized')
# pca = PCA(n_components=30, svd_solver='randomized')

# pca.fit(X_train)

# X_train = pca.transform(X_train)
# X_test = pca.transform(X_test)


# In[342]:


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=30)
svd.fit(X_training)

X_train = svd.transform(X_training)
X_test = svd.transform(X_test)


# rp = random_projection.SparseRandomProjection(n_components=30)
# rp.fit(X_training)

# X_train = rp.transform(X_training)
# X_test = rp.transform(X_test)


# In[219]:
#
#
# from sklearn.linear_model import LogisticRegression
#
#
# logreg = LogisticRegression(solver = 'newton-cg', multi_class ='multinomial')
# logreg.fit(X_train, y_train)
#
# print('Accuracy of Logistic regression classifier on training set: {:.2f}'
#      .format(logreg.score(X_train, y_train)))
# print('Accuracy of Logistic regression classifier on test set: {:.2f}'
#      .format(logreg.score(X_test, y_test)))


# In[324]:


# from sklearn.tree import DecisionTreeClassifier
#
# clf = DecisionTreeClassifier().fit(X_train, y_train)
#
# print('Accuracy of Decision Tree classifier on training set: {:.5f}'
#      .format(clf.score(X_train, y_train)))
# print('Accuracy of Decision Tree classifier on test set: {:.5f}'
#      .format(clf.score(X_test, y_test)))


# In[343]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5, algorithm='kd_tree')
knn.fit(X_train, np.ravel(Y_training))
# print('Accuracy of K-NN classifier on training set: {:.2f}'
#      .format(knn.score(X_train, y_train)))
# print('Accuracy of K-NN classifier on test set: {:.2f}'
#      .format(knn.score(X_test, y_test)))

pred = knn.predict(X_test)

# print(classification_report(y_test, pred))


# In[90]:


# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#
# lda = LinearDiscriminantAnalysis()
# lda.fit(X_train, y_train)
# print('Accuracy of LDA classifier on training set: {:.2f}'
#      .format(lda.score(X_train, y_train)))
# print('Accuracy of LDA classifier on test set: {:.2f}'
#      .format(lda.score(X_test, y_test)))


# In[220]:

#
# from sklearn.naive_bayes import GaussianNB
#
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# print('Accuracy of GNB classifier on training set: {:.2f}'
#      .format(gnb.score(X_train, y_train)))
# print('Accuracy of GNB classifier on test set: {:.2f}'
#      .format(gnb.score(X_test, y_test)))
#
#
# # In[221]:
#
#
# from sklearn.svm import SVC
#
# svm = SVC()
# svm.fit(X_train, y_train)
# print('Accuracy of SVM classifier on training set: {:.2f}'
#      .format(svm.score(X_train, y_train)))
# print('Accuracy of SVM classifier on test set: {:.2f}'
#      .format(svm.score(X_test, y_test)))
#
#
# # In[95]:
#
#
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# pred = knn.predict(X_test)
# print(confusion_matrix(y_test, pred))
# print(classification_report(y_test, pred))


df_pred = pd.DataFrame(pred)

predicted_cls = df_pred.astype('int', copy=False)

np.savetxt('predictions.txt',
predicted_cls, fmt='%s', delimiter="\n")