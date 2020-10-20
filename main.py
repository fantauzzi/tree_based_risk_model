import shap
import sklearn
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from catboost import CatBoostClassifier, Pool, cv
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, space_eval, hp
from time import time

from util import load_data


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


def make_imputed_pool(X, y, imputer, cat_features, weight=None):
    X_imputed = X if imputer is None else pd.DataFrame(imputer.transform(X), columns=X.columns)
    # imputer.transform() above has converted the int columns with categories into float, need to be converted back to int
    X_imputed = X_imputed.astype({'Sex': int, 'Race': int})
    pool = Pool(data=X_imputed, label=y, cat_features=cat_features, weight=weight)
    return pool, X_imputed


#############################################################################################################
seed = 42  # For random numbers generation
iterations = 20  # Max number of iterations at every run of gradien boosting (max number of trees built)
hyper_iterations = 20  # Number of iterations required during each Bayesian optimization of hyper-parameters
cv_folds = 4  # Number of folds used for k-folds cross-validation
logs_dir = Path('catboost_logs')  # Relative to the directory where the program is running
task_type = 'GPU'  # Can be 'CPU' or 'GPU'
#############################################################################################################

start_time = time()

# Make the logs directory, if it doesn't exist already, and make sure it is empty
logs_dir.mkdir(exist_ok=True)
for item in logs_dir.iterdir():
    if item.is_dir():
        shutil.rmtree(str(item))

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

# Now impute missing values using the mean, instead of dropping samples containing them
mean_imputer = SimpleImputer(strategy='mean', verbose=0)
mean_imputer.fit(X_train)

iter_imputer = IterativeImputer(random_state=seed, sample_posterior=False, max_iter=10, min_value=0, verbose=0)
iter_imputer.fit(X_train)

cat_features = [3, 11]  # Categorical features are race and sex


def compute_weights(y):
    """
    Computes and returns the weights for every sample, such that if the positive samples are n
    times the negative samples, then their weight is 1/n times the weight of the negative samples, and such that
    the sum of the weights of all the samples is equal to the total number of samples. One weight value is assigned
    to all positive samples, and another to all negative samples.
    :param y: an array-like with the ground truth for the samples, with 1 for positive and 0 for negative.
    :return: a pair, the first element is a numpy array with the requested weights per sample, the second is a
    dictionary providing the count of positive and negative elements, and the assigned respective weights.
    """
    total_pos = sum(y)
    total_neg = len(y) - total_pos
    # pos_weight = total_neg / total_pos
    # neg_weight = total_pos / total_neg
    pos_weight = total_neg * len(y) / (2 * total_neg * total_pos)
    neg_weight = total_pos * len(y) / (2 * total_neg * total_pos)
    assert np.isclose(pos_weight * total_pos + neg_weight * total_neg, len(y))
    w = np.full_like(y, neg_weight)
    w[y == 1] = pos_weight
    return w, {'total_pos': total_pos, 'total_neg': total_neg, 'pos_weight': pos_weight, 'neg_weight': neg_weight}


def run_exp_bayes_hyperparams_opt(X_train,
                                  y_train,
                                  X_val,
                                  y_val,
                                  cat_features,
                                  param_space,
                                  max_evals,
                                  imputer,
                                  train_dir=None,
                                  weights=None):
    train_pool, X_train_imputed = make_imputed_pool(X_train,
                                                    y=y_train,
                                                    imputer=imputer,
                                                    cat_features=cat_features,
                                                    weight=weights)

    val_pool, X_val_imputed = make_imputed_pool(X_val,
                                                y=y_val,
                                                imputer=imputer,
                                                cat_features=cat_features)

    # The objective function, that hyperopt will minimize
    def objective(params):
        model = CatBoostClassifier(iterations=params['iterations'],
                                   eval_metric='AUC:hints=skip_train~false',
                                   learning_rate=params['learning_rate'],
                                   depth=params['depth'],
                                   random_state=params['seed'],
                                   task_type=params['task_type'])
        training_res = model.fit(train_pool, eval_set=val_pool, verbose=False)
        auc = training_res.best_score_['validation']['AUC']
        return -auc  # The objective function is minimized

    rstate = np.random.RandomState(seed)
    best = fmin(fn=objective, space=param_space, algo=tpe.suggest, max_evals=max_evals, rstate=rstate)
    print('Re-fitting the model with the best hyper-parameter values found:', best)
    refit_model = CatBoostClassifier(iterations=param_space['iterations'],
                                     eval_metric='AUC:hints=skip_train~false',
                                     **best,
                                     random_state=param_space['seed'],
                                     train_dir=train_dir,
                                     task_type=param_space['task_type'])
    training_res = refit_model.fit(train_pool,
                                   eval_set=val_pool,
                                   verbose=False)

    y_train_preds = refit_model.predict_proba(X_train)[:, 1]
    training_AUC = roc_auc_score(y_train, y_train_preds)
    print(f"Training: Log loss={training_res.best_score_['learn']['Logloss']}   ROC AUC={training_AUC}")
    print(
        f"Validation: Log loss={training_res.best_score_['validation']['Logloss']}   ROC AUC={training_res.best_score_['validation']['AUC']}")
    print(f'Best iteration: {training_res.best_iteration_}')
    return refit_model


''' Use the iterative imputer, but use Bayesian optimization for the hyper-parameters, instead of grid search. Here
we use the train/val data sets

Note: passing a CatBoost Pool() instance in the param_space values here below doesn't work, because hyperopt would
throw an exception during optimization.'''
param_space = {'learning_rate': hp.uniform('learning_rate', .01, .1),
               'depth': hp.quniform('depth', 4, 12, 1),
               'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 9),
               'bagging_temperature': hp.uniform('bagging_temperature', 0, 2),
               'seed': seed,  # hyperopt accepts constant value parameters
               'iterations': iterations,
               'task_type': task_type
               }

print('\nPerforming Bayesian search for hyper-parameters optimization, after dropping samples with missing data')

run_exp_bayes_hyperparams_opt(X_train_dropped,
                              y_train_dropped,
                              X_val_dropped,
                              y_val_dropped,
                              cat_features=cat_features,
                              param_space=param_space,
                              max_evals=hyper_iterations,
                              imputer=None,
                              train_dir=str(logs_dir / 'catboost_logs_drop'))

print('\nPerforming Bayesian search for hyper-parameters optimization, with missing data replaced with mean imputer')

run_exp_bayes_hyperparams_opt(X_train,
                              y_train,
                              X_val,
                              y_val,
                              cat_features=cat_features,
                              param_space=param_space,
                              max_evals=hyper_iterations,
                              imputer=mean_imputer,
                              train_dir=str(logs_dir / 'catboost_logs_mean_imputer'))

print(
    '\nPerforming Bayesian search for hyper-parameters optimization, with missing data replaced with iterative imputer')

selected_model_imputed = run_exp_bayes_hyperparams_opt(X_train,
                                                       y_train,
                                                       X_val,
                                                       y_val,
                                                       cat_features=cat_features,
                                                       param_space=param_space,
                                                       max_evals=hyper_iterations,
                                                       imputer=iter_imputer,
                                                       train_dir=str(logs_dir / 'catboost_logs_iter_imputer'))

print('\nPerforming Bayesian search for hyper-parameters optimization, without replacement of missing data')

selected_model = run_exp_bayes_hyperparams_opt(X_train,
                                               y_train,
                                               X_val,
                                               y_val,
                                               cat_features=cat_features,
                                               param_space=param_space,
                                               max_evals=hyper_iterations,
                                               imputer=None,
                                               train_dir=str(logs_dir / 'catboost_logs_keep_nan'))

print('\nPerforming Bayesian search for hyper-parameters optimization, without replacement and with weights')

w, stats = compute_weights(y_train)
print('Computed weights')
print('For', stats['total_pos'], 'positive samples:', stats['pos_weight'])
print('For', stats['total_neg'], 'negative samples:', stats['neg_weight'])

run_exp_bayes_hyperparams_opt(X_train,
                              y_train,
                              X_val,
                              y_val,
                              cat_features=cat_features,
                              param_space=param_space,
                              max_evals=hyper_iterations,
                              imputer=None,
                              weights=w,
                              train_dir=str(logs_dir / 'catboost_logs_weights'))

''' Grid-search is done on the dev. set, as the grid-search takes care of splitting it into training and validation.
Note: if `search_by_train_test_split` is set to True, every combination of values of the hyper-parameters is evaluated
with a basic training/val. split of the dataset; if set to False, then every combination is evaluated with x-evaluation.
Once method grid_search() has selected the best combination of hyper-parameters, fits a model with it. The final model 
can be evaluated with x-evaluation by setting parameter `calc_cv_statistics` to True (default). '''


def run_exp_grid_hyperparams_opt(X, y, cat_features, seed, iterations, param_grid, cv_folds, imputer=None,
                                 train_dir=None, task_type='CPU'):
    dev_pool, X_inputed = make_imputed_pool(X, y, imputer, cat_features)
    model = CatBoostClassifier(iterations=iterations,
                               eval_metric='AUC:hints=skip_train~false',
                               cat_features=cat_features,
                               random_state=seed,
                               train_dir=train_dir,
                               task_type=task_type)

    grid_search_results = model.grid_search(X=dev_pool,
                                            param_grid=param_grid,
                                            search_by_train_test_split=True,
                                            calc_cv_statistics=True,
                                            cv=cv_folds,
                                            partition_random_seed=seed,
                                            verbose=True)

    best_iter = np.argmax(grid_search_results['cv_results']['test-AUC-mean'])
    best_AUC = grid_search_results['cv_results']['test-AUC-mean'][best_iter]
    loss_for_best_AUC = grid_search_results['cv_results']['test-Logloss-mean'][best_iter]
    print('Best params', grid_search_results['params'], 'with AUC', best_AUC, 'obtained at iteration', best_iter,
          'with Log loss', loss_for_best_AUC)
    return model


param_grid = {'learning_rate': np.arange(.01, .2, .01),
              'depth': list(range(2, 8))}

# Count the number of hyper-parameter combinations in the grid
n_grid_points = 1
for _, param in param_grid.items():
    n_grid_points *= len(param)

print('\nPerforming grid-search for hyper-parameters optimization while maintaining missing data')
print(f'Searching over {n_grid_points} combinations')

run_exp_grid_hyperparams_opt(X=X_dev,
                             y=y_dev,
                             cat_features=cat_features,
                             seed=seed,
                             iterations=iterations,
                             param_grid=param_grid,
                             cv_folds=cv_folds,
                             imputer=None,
                             train_dir=str(logs_dir / 'catboost_logs_grid_search'),
                             task_type=task_type)

# Cross-validate the two selected models, and test them on the test set

''' Make a new imputer for cross-validation over the dev set. It would be more correct to re-compute the imputer
at every fold of cross-validation, but CatBoost() cv doesn't contemplate the possibility.
'''

dev_iter_imputer = IterativeImputer(random_state=seed, sample_posterior=False, max_iter=10, min_value=0, verbose=0)
dev_iter_imputer.fit(X_dev)

for model, descr, imputer in zip((selected_model, selected_model_imputed), ('without_imputation', 'with_imputation'),
                                 (None, dev_iter_imputer)):
    print(f'\nCross-validating model {descr}.')
    params = model.get_params()
    params['loss_function'] = 'Logloss'
    params['eval_metric'] = 'AUC:hints=skip_train~false'
    params['train_dir'] = str(logs_dir / ('catboost_logs_cv_' + descr))
    params['task_type'] = task_type
    X_pool, _ = make_imputed_pool(X_dev, y_dev, imputer=imputer, cat_features=cat_features, weight=None)
    cv_results = cv(pool=X_pool,
                    params=params,
                    iterations=iterations,
                    fold_count=4,
                    partition_random_seed=seed,
                    stratified=True,
                    verbose=False)
    # Find the iteration with the best test AUC, its AUC and other train and test stats.
    best_cv_iter = np.argmax(cv_results['test-AUC-mean'])  # All the stats retrieved will refer to this same iteration
    best_cv_val_AUC = cv_results['test-AUC-mean'][best_cv_iter]
    best_cv_val_Logloss = cv_results['test-Logloss-mean'][best_cv_iter]
    best_cv_train_AUC = cv_results['train-AUC-mean'][best_cv_iter]
    best_cv_train_Logloss = cv_results['train-Logloss-mean'][best_cv_iter]
    print('Best validation with parameters', params, 'achieved at iteration', best_cv_iter)
    print(f'Training: Logloss {best_cv_train_Logloss}   ROC AUC {best_cv_train_AUC}')
    print(f'Validation: Logloss {best_cv_val_Logloss}   ROC AUC {best_cv_val_AUC}')

    print('Re-fitting the model and testing it')
    test_pool, _ = make_imputed_pool(X_test, y_test, imputer=None, cat_features=cat_features, weight=None)
    params['iterations'] = best_cv_iter + 1  # Shrink the model to the best iteration found during cross-validation
    cv_model = CatBoostClassifier(**params)
    training_res = cv_model.fit(X_pool, eval_set=test_pool, verbose=False)
    print('Iteration:', training_res.best_iteration_)
    y_train_preds = cv_model.predict_proba(X_dev)[:, 1]
    training_AUC = roc_auc_score(y_dev, y_train_preds)
    print(f"Training (on dev. set): Log loss={training_res.best_score_['learn']['Logloss']}   ROC AUC={training_AUC}")
    print(
        f"Test (on test set): Log loss={training_res.best_score_['validation']['Logloss']}   ROC AUC={training_res.best_score_['validation']['AUC']}")

print(f'Overall run time: {round(time() - start_time)}s')

''' TODO: misc
Try Karpathy approach
Leverage Tensorboard
Explore Seaborne
Use the whole HANES dataset from CDC, and also try with GPU
Try other strategies for imputation based on mean encoding and similar
Instead of checking if survival after 10 years, estimate the number of years of survival
C-index is the same as the ROC AUC for logistic regression.
   see https://www.statisticshowto.com/c-statistic/#:~:text=A%20weighted%20c-index%20is,correctly%20predicting%20a%20negative%20outcome
   and also https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4886856/  and https://bit.ly/3dvUh07

'''
