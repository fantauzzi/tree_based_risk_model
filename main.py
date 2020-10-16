import shap
import sklearn
import itertools
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import Image

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer, SimpleImputer
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GridSearchCV
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import roc_auc_score
import catboost

from util import load_data, cindex


class C_index(object):
    def is_max_optimal(self):
        # Returns whether great values of metric are better
        return True  # Higher is better

    def evaluate(self, approxes, target, weight):
        # approxes is a list of indexed containers
        # (containers with only __len__ and __getitem__ defined),
        # one container per approx dimension.
        # Each container contains floats.
        # weight is a one dimensional indexed container.
        # target is a one dimensional indexed container.

        # weight parameter can be None.
        # Returns pair (error, weights sum)
        pass

    def get_final_error(self, error, weight):
        # Returns final value of metric based on error and weight
        pass

seed = 42
iterations = 200

# Load the NHANES I epidemiology dataset
X_dev, X_test, y_dev, y_test = load_data(10)

# Convert categorical features from float to int, as that is what CatBoost expects
X_dev = X_dev.astype({'Sex': int, 'Race': int})
y_dev = y_dev.astype(int)
X_test = X_test.astype({'Sex': int, 'Race': int})
y_test = y_test.astype(int)


# Find out how many samples have missing data in one or more variables (columns)
def count_samples_with_missing_data(df):
    res = sum(df.isnull().any(axis='columns'))
    return res


dev_missing_count = count_samples_with_missing_data(X_dev)
test_missing_count = count_samples_with_missing_data(X_test)

print('Dev. set missing data in', dev_missing_count, 'samples out of', len(X_dev))
print('Test set missing data in', test_missing_count, 'samples out of', len(X_test))

# Split the dev set into training and validation. The latter will be used for hyper-parameters tuning.
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.2, random_state=seed)

# Make a dataset after dropping samples with missing data (note, no samples with missing data in test set)
X_dev_dropped = X_dev.dropna(axis='rows')
y_dev_dropped = y_dev.loc[X_dev_dropped.index]
X_train_dropped = X_train.dropna(axis='rows')
y_train_dropped = y_train.loc[X_train_dropped.index]
X_val_dropped = X_val.dropna(axis='rows')
y_val_dropped = y_val.loc[X_val_dropped.index]

cat_features = [3, 11]  # Categorical features are race and sex
dev_pool = Pool(data=X_dev_dropped, label=y_dev_dropped, cat_features=cat_features)
train_pool = Pool(data=X_train_dropped, label=y_train_dropped, cat_features=cat_features)
val_pool = Pool(data=X_val_dropped, label=y_val_dropped, cat_features=cat_features)
test_pool = Pool(data=X_test, label=y_test, cat_features=cat_features)

# Fit a model on the dataset from where samples with missing data have been dropped
print('Fitting a model on the dataset from where samples with missing data have been dropped')
model = CatBoostClassifier(iterations=iterations, random_state=seed)
model.fit(train_pool, eval_set=val_pool, verbose=iterations // 10)

# Compute the C-index on the train/val/test dataset
y_train_preds = model.predict_proba(X_train_dropped)[:, 1]
print(f"Train C-Index: {cindex(y_train_dropped.values, y_train_preds)}")
print(f'Train ROC AUC: {roc_auc_score(y_train_dropped.values, y_train_preds)}')

y_val_preds = model.predict_proba(X_val_dropped)[:, 1]
print(f"Val C-Index: {cindex(y_val_dropped.values, y_val_preds)}")
print(f'Val ROC AUC: {roc_auc_score(y_val_dropped.values, y_val_preds)}')

y_test_preds = model.predict_proba(X_test)[:, 1]
print(f"Test C-Index: {cindex(y_test.values, y_test_preds)}")
print(f'Train ROC AUC: {roc_auc_score(y_test.values, y_test_preds)}')

# Grid-search with scikit-learn
# grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
# results = grid_search.fit(X_train, y_train)
# print(results.best_estimator_.get_params())

# Perform grid-search to optimize some of the hyper-parameters
print('Performing grid-search for hyper-parameters optimization, without samples with missing data')
param_grid = {
    'learning_rate': [.01, .05, .06, .07, 0.08, .1, .2],
    'depth': [2, 3, 4, 5, 6, 7, 8]
}


# Determine how many combinations of parameter values are in the grid
def compute_n_combinations(grid):
    res = 1
    for _, value in grid.items():
        res *= len(value)
    return res


n_combos = compute_n_combinations(param_grid)

clf = CatBoostClassifier(iterations=iterations, random_state=seed, cat_features=cat_features)

# Argument search_by_train_test_split can be set to False to use x-validation at every evaluation of hyperparameters value
# If search_by_train_test_split is set to True, it is still possible to use x-validation in evaluating the final
# model by setting calc_cv_statistics to True

''' Grid-search is done on the dev. set, as the grid-search takes care of splitting it into training and validation.
Note: if `search_by_train_test_split` is set to True, every combination of values of the hyper-parameters is evaluated
with a basic training/val. split of the dataset; if set to False, thene very combination is evaluated with x-evaluation.
Once method grid_search() has found the best combination of hyper-parameters, fits a model with it. The final model 
can be evaluated with x-evaluation by setting parameter `calc_cv_statistics` to True (default). '''
grid_search_results = clf.grid_search(param_grid=param_grid,
                                      X=dev_pool,
                                      partition_random_seed=seed,
                                      cv=5,
                                      search_by_train_test_split=True,
                                      verbose=True)


def print_grid_search_results(results):
    best_iter = results['cv_results']['iterations'][
        np.argmin(results['cv_results']['test-Logloss-mean'])]
    best_loss = np.min(results['cv_results']['test-Logloss-mean'])
    print('Best params', results['params'], 'obtained at iteration', best_iter, 'with logloss', best_loss)


print_grid_search_results(grid_search_results)


# Compute and print C-Indices
def print_dev_test_c_indices(classifier, X_dev, y_dev, X_test, y_test):
    y_dev_preds = classifier.predict_proba(X_dev)[:, 1]
    print(f"Dev. C-Index on best model after grid-search: {cindex(y_dev.values, y_dev_preds)}")

    y_test_preds = clf.predict_proba(X_test)[:, 1]
    print(f"Test C-Index on best model after grid-search: {cindex(y_test.values, y_test_preds)}")


print_dev_test_c_indices(clf, X_dev, y_dev, X_test, y_test)

# Now impute missing values using the mean, instead of dropping samples containing them
imputer = SimpleImputer(strategy='mean')
imputer.fit(X_dev)
X_dev_mean_imputed = pd.DataFrame(imputer.transform(X_dev), columns=X_dev.columns)
# imputer.transform() above has converted the int columns with categories into float, need to be converted back to int
X_dev_mean_imputed = X_dev_mean_imputed.astype({'Sex': int, 'Race': int})
dev_pool_mean_imputed = Pool(data=X_dev_mean_imputed, label=y_dev, cat_features=cat_features)

''' Note: CatBoost doean't allow to re-tune the hyper-parameters of a model that has been fitted already; need to
instantiate a new model '''
clf2 = CatBoostClassifier(iterations=200, random_state=seed, cat_features=cat_features)
print('Performing grid-search for hyper-parameters optimization, with missing data replaced with a mean imputer')
grid_search_results = clf2.grid_search(param_grid=param_grid,
                                      X=dev_pool_mean_imputed,
                                      search_by_train_test_split=True,
                                      cv=5,
                                      partition_random_seed=seed,
                                      verbose=True)
print_grid_search_results(grid_search_results)

print_dev_test_c_indices(clf2, X_dev, y_dev, X_test, y_test)

''' Now instead of dropping samples with missing data, replace the missing data with an iterative imputer. 
Do it on the dev set as later will use grid-search, which does its own training/val. split.'''
imputer = IterativeImputer(random_state=seed, sample_posterior=False, max_iter=1, min_value=0)
imputer.fit(X_dev)
X_dev_iter_imputed = pd.DataFrame(imputer.transform(X_dev), columns=X_dev.columns)
# imputer.transform() above has converted the int columns with categories into float, need to be converted back to int
X_dev_iter_imputed = X_dev_iter_imputed.astype({'Sex': int, 'Race': int})
dev_pool_iter_imputed = Pool(data=X_dev_iter_imputed, label=y_dev, cat_features=cat_features)

clf3 = CatBoostClassifier(iterations=200, random_state=seed, cat_features=cat_features)
print('Performing grid-search for hyper-parameters optimization, with missing data replaced with an iterative imputer')
grid_search_results = clf3.grid_search(param_grid=param_grid,
                                       X=dev_pool_iter_imputed,
                                       search_by_train_test_split=True,
                                       cv=5,
                                       partition_random_seed=seed,
                                       verbose=True)
print_grid_search_results(grid_search_results)

print_dev_test_c_indices(clf, X_dev, y_dev, X_test, y_test)

''' TODO
Introduce hyperopt
Use c-index (custom metric) to select models instead of loss. Is it true that the c-index is the same as the ROC AUC?
   see https://www.statisticshowto.com/c-statistic/#:~:text=A%20weighted%20c-index%20is,correctly%20predicting%20a%20negative%20outcome
   and also https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4886856/  and https://bit.ly/3dvUh07
Leverage Tensorboard
How to display CatBoost charts outside of notebook? Is it possible?
Explore Seaborne
Use the whole HANES dataset from CDC
Try other strategies for imputation based on mean encoding and similar
Instead of checking if survival after 10 years, estimate the number of years of survival
Unbalanced dataset, try using weights
How does CatBoost deal with missing data (None/NaN)?
'''
