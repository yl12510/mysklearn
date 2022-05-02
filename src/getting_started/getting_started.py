#!/usr/bin/env python

# # Getting Started
#
# to practise main features of `scikit-learn`

# ## Fitting and predicting: estimator basics
#
# estimators: built-in machine learning algorithms and models

# In[1]:


# simple example of using RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=0)

X = [[1, 2, 3], [11, 12, 13]]  # sample matrix (n_samples, n_features)
y = [0, 1]  # target values
clf.fit(X, y)


# In[4]:


# using the fitted estimator to predict target values of training data
clf.predict(X)


# In[5]:


# predict target values of new data
clf.predict(
    [
        [
            4,
            5,
            6,
        ],
        [14, 15, 16],
    ]
)


# ## Transformers and pre-processors
#
# A typical machine learning pipeline:
# 1. preprocessing: transforms or imputes the data
# 2. predicting: predicts the targeted value
#
# pre-processors, transformers, estimators all inherit from the `BaseEstimator` class
# - pre-processors & transformers don't have a predict method, but have a transform method
# - `ColumnTransformer` is designed for applying different transformations to different features

# In[7]:


# An example of using StandardScaler
from sklearn.preprocessing import StandardScaler

X = [[0, 15], [1, -10]]
StandardScaler().fit(X).transform(X)


# In[ ]:
