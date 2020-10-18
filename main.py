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
# from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer, SimpleImputer
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GridSearchCV
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, space_eval, hp
import catboost

from util import load_data, cindex


def print_train_val_test_c_indices(classifier,
                                   X_train,
                                   y_train,
                                   X_val,
                                   y_val,
                                   X_test,
                                   y_test):
    y_train_preds = classifier.predict_proba(X_train)[:, 1]
    print(f'Train ROC AUC: {roc_auc_score(y_train, y_train_preds)}')

    y_val_preds = classifier.predict_proba(X_val)[:, 1]
    print(f'Val ROC AUC: {roc_auc_score(y_val, y_val_preds)}')

    y_test_preds = classifier.predict_proba(X_test)[:, 1]
    print(f'Test ROC AUC: {roc_auc_score(y_test, y_test_preds)}')


def make_imputed_pool(X, y, imputer, cat_features):
    X_imputed = X if imputer is None else pd.DataFrame(imputer.transform(X), columns=X.columns)
    # imputer.transform() above has converted the int columns with categories into float, need to be converted back to int
    X_imputed = X_imputed.astype({'Sex': int, 'Race': int})
    pool = Pool(data=X_imputed, label=y, cat_features=cat_features)
    return pool, X_imputed


seed = 42
iterations = 100
hyper_iterations = 100

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

print('Performing grid-search for hyper-parameters optimization, after dropping samples with missing data')

''' Grid-search is done on the dev. set, as the grid-search takes care of splitting it into training and validation.
Note: if `search_by_train_test_split` is set to True, every combination of values of the hyper-parameters is evaluated
with a basic training/val. split of the dataset; if set to False, then every combination is evaluated with x-evaluation.
Once method grid_search() has selected the best combination of hyper-parameters, fits a model with it. The final model 
can be evaluated with x-evaluation by setting parameter `calc_cv_statistics` to True (default). 

Note 2: CatBoost grid search chooses the best values for the hyper-parameters based on the loss, not on the eval metric 
set for the model (AUC). '''


# TODO: Check the above, might be not true!

def run_exp_grid_hyperparams_opt(X, y, cat_features, seed, iterations, param_grid, imputer=None):

    dev_pool, X_inputed = make_imputed_pool(X, y, imputer, cat_features)
    model = CatBoostClassifier(iterations=iterations,
                               eval_metric='AUC:hints=skip_train~false',
                               cat_features=cat_features,
                               random_state=seed)

    grid_search_results = model.grid_search(X=dev_pool,
                                            param_grid=param_grid,
                                            search_by_train_test_split=True,
                                            calc_cv_statistics=True,
                                            cv=5,
                                            partition_random_seed=seed,
                                            verbose=True)

    best_iter = np.argmax(grid_search_results['cv_results']['test-AUC-mean'])
    best_AUC = grid_search_results['cv_results']['test-AUC-mean'][best_iter]
    loss_for_best_AUC = grid_search_results['cv_results']['test-Logloss-mean'][best_iter]
    print('Best params', grid_search_results['params'], 'with AUC', best_AUC, 'obtained at iteration', best_iter, 'with Logloss', loss_for_best_AUC)
    return model
    # y_preds = model.predict_proba(X)[:, 1]
    # print(f'ROC AUC on best model after grid-search: {roc_auc_score(y.values, y_preds)}')


param_grid = {'learning_rate': [.01, .05, .06, .07, 0.08, .1, .2],
              'depth': [2, 3, 4, 5, 6, 7, 8]}

run_exp_grid_hyperparams_opt(X=X_dev_dropped,
                             y=y_dev_dropped,
                             cat_features=cat_features,
                             seed=seed,
                             iterations=iterations,
                             param_grid=param_grid,
                             imputer=None)

print('\nPerforming grid-search for hyper-parameters optimization, with missing data replaced with a mean imputer')

# Now impute missing values using the mean, instead of dropping samples containing them
mean_imputer = SimpleImputer(strategy='mean')
mean_imputer.fit(X_dev)
run_exp_grid_hyperparams_opt(X=X_dev,
                             y=y_dev,
                             seed=seed,
                             iterations=iterations,
                             cat_features=cat_features,
                             param_grid=param_grid,
                             imputer=mean_imputer)

print(
    '\nPerforming grid-search for hyper-parameters optimization, with missing data replaced with an iterative imputer')

# Now try with an iterative imputer instead
iter_imputer = IterativeImputer(random_state=seed, sample_posterior=False, max_iter=1, min_value=0)
iter_imputer.fit(X_dev)
run_exp_grid_hyperparams_opt(X=X_dev,
                             y=y_dev,
                             seed=seed,
                             iterations=iterations,
                             cat_features=cat_features,
                             param_grid=param_grid,
                             imputer=iter_imputer)


def run_exp_bayes_hyperparams_opt(X_train, y_train, X_val, y_val, cat_features, param_space, max_evals, imputer):
    train_pool, X_train_imputed = make_imputed_pool(X_train,
                                                    y=y_train,
                                                    imputer=imputer,
                                                    cat_features=cat_features)

    val_pool, X_val_imputed = make_imputed_pool(X_val,
                                                y=y_val,
                                                imputer=imputer,
                                                cat_features=cat_features)

    # The objective function, that hyperopt will minimize
    def objective(params):
        model = CatBoostClassifier(iterations=params['iterations'],
                                   eval_metric='AUC',
                                   learning_rate=params['learning_rate'],
                                   depth=params['depth'],
                                   random_state=params['seed'])
        training_res = model.fit(train_pool, eval_set=val_pool, verbose=False)
        auc = training_res.best_score_['validation']['AUC']
        return -auc  # The objective function is minimized

    rstate = np.random.RandomState(seed)
    best = fmin(fn=objective, space=param_space, algo=tpe.suggest, max_evals=max_evals, rstate=rstate)
    print('Re-fitting the model with the best hyper-parameter values found:', best)
    refit_model = CatBoostClassifier(iterations=param_space['iterations'],
                                     eval_metric='AUC',
                                     **best,
                                     random_state=param_space['seed'])
    training_res = refit_model.fit(train_pool, eval_set=val_pool, verbose=iterations // 10)

    print_train_val_test_c_indices(refit_model,
                                   X_train_imputed,
                                   y_train.values,
                                   X_val_imputed,
                                   y_val.values,
                                   X_test,
                                   y_test.values)


''' Use the iterative imputer, but use Bayesian optimization for the hyper-parameters, instead of grid search. Here
we use the train/val data sets

Note: passing a CatBoost Pool() instance in the param_space values here below doesn't work, because hyperopt would
throw an exception during optimization.'''

param_space = {'learning_rate': hp.uniform('learning_rate', .01, .1),
               'depth': hp.quniform('depth', 2, 8, 1),
               'seed': seed,  # hyperopt accepts constant value parameters
               'iterations': iterations
               }

print('Performing Bayesian search for hyper-parameters optimization, with missing data replaced with iterative imputer')

run_exp_bayes_hyperparams_opt(X_train,
                              y_train,
                              X_val,
                              y_val,
                              cat_features=cat_features,
                              param_space=param_space,
                              max_evals=hyper_iterations,
                              imputer=iter_imputer)

print('Performing Bayesian search for hyper-parameters optimization, without replacement of missing data')

run_exp_bayes_hyperparams_opt(X_train,
                              y_train,
                              X_val,
                              y_val,
                              cat_features=cat_features,
                              param_space=param_space,
                              max_evals=hyper_iterations,
                              imputer=None)

''' TODO
Check the warning: [IterativeImputer] Early stopping criterion not reached.
Check the loss/ROC issue filed on GitHub
Try Karpathy approach
Unbalanced dataset, try using weights
Leverage Tensorboard
How to display CatBoost charts outside of notebook? Is it possible?
Explore Seaborne
Use the whole HANES dataset from CDC, and also try with GPU
Try other strategies for imputation based on mean encoding and similar
Instead of checking if survival after 10 years, estimate the number of years of survival
C-index is the same as the ROC AUC for logistic regression.
   see https://www.statisticshowto.com/c-statistic/#:~:text=A%20weighted%20c-index%20is,correctly%20predicting%20a%20negative%20outcome
   and also https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4886856/  and https://bit.ly/3dvUh07

'''
