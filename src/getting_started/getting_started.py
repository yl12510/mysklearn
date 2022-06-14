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

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Pipelines: chaining pre-processors and estimators
# - a pipeline offers the same API functions e.g. `fit` and `predict` as a regular estimator
# - using a pipeline can prevent from disclosing testing data in training data (i.e. data leakage)

# + pycharm={"name": "#%%\n"}
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# + pycharm={"name": "#%%\n"}
from sklearn.preprocessing import StandardScaler

# create a pipeline object
pipe = make_pipeline(StandardScaler(), LogisticRegression())

# load the iris dataset and split into train and test sets
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# fit the whole pipeline
pipe.fit(X_train, y_train)

# use it to predict over test data set and calculate accuracy score
accuracy_score(pipe.predict(X_test), y_test)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Model evaluation
#
# A model needs to be evaluated to see if it can predict well over unseen data.
# - Cross validation is a particular tool for model evaluation
# - sklearn provides a `cross_validate` helper, which by default will perform a 5-fold cross validation
# - it is also possible to do manual iteration over folds, use different data splitting strategies, and use custom scoring functions

# + pycharm={"name": "#%%\n"}
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

X, y = make_regression(n_samples=1000, random_state=0)
lr = LinearRegression()

result = cross_validate(lr, X, y)
result

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Automatic parameter searches
#
# How good the generalisation of an estimator highly depends on a number of parameters (or hyper-parameters).
# - sklearn provides tools to automatically search the parameter space and find the best combination.

# + pycharm={"name": "#%%\n"}
from scipy.stats import randint
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# define the parameter space to search over
param_spaces = {"n_estimators": randint(1, 5), "max_depth": randint(5, 10)}

# create a searchCV object and fit it to the data
search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=0),
    n_iter=5,
    param_distributions=param_spaces,
    random_state=0,
)

search.fit(X_train, y_train)

search.best_params_

# + pycharm={"name": "#%%\n"}
search.score(X_test, y_test)

# + pycharm={"name": "#%%\n"}
