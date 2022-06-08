# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3.9.12 ('base')
#     language: python
#     name: python3
# ---

# + [markdown] pycharm={"name": "#%% md\n"}
# # Getting Started
#
# to practise main features of `scikit-learn`

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Fitting and predicting: estimator basics
#
# estimators: built-in machine learning algorithms and models

# + pycharm={"name": "#%%\n"}
# simple example of using RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=0)

X = [[1, 2, 3], [11, 12, 13]]  # sample matrix (n_samples, n_features)
y = [0, 1]  # target values
clf.fit(X, y)

# + pycharm={"name": "#%%\n"}
# using the fitted estimator to predict target values of training data
clf.predict(X)

# + pycharm={"name": "#%%\n"}
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

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Transformers and pre-processors
#
# A typical machine learning pipeline:
# 1. preprocessing: transforms or imputes the data
# 2. predicting: predicts the targeted value
#
# pre-processors, transformers, estimators all inherit from the `BaseEstimator` class
# - pre-processors & transformers don't have a predict method, but have a transform method
# - for certain use-cases, `ColumnTransformer` is designed for applying different transformations to different features

# + pycharm={"name": "#%%\n"}
# An example of using StandardScaler
from sklearn.preprocessing import StandardScaler

X = [[0, 15], [1, -10]]
StandardScaler().fit(X).transform(X)
# -

# ## Pipelines: chaining pre-processors and estimators
