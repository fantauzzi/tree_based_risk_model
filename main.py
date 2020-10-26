# from matplotlib import use
# use('TkAgg')
import matplotlib.pyplot as plt
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
import shutil
from pathlib import Path
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.experimental import enable_iterative_imputer
from catboost import CatBoostClassifier, Pool, cv, EFstrType
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from hyperopt import fmin, tpe, hp
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from time import time

from util import load_data


def make_imputed_pool(X, y, imputer, cat_features, weight=None):
    X_imputed = X if imputer is None else pd.DataFrame(imputer.transform(X), columns=X.columns)
    # imputer.transform() above has converted the int columns with categories into float, need to be converted back to int
    X_imputed = X_imputed.astype({'Sex': int, 'Race': int})
    pool = Pool(data=X_imputed, label=y, cat_features=cat_features, weight=weight)
    return pool, X_imputed


def run_exp_nn(X_train,
               y_train,
               X_val,
               y_val,
               params,
               max_evals,
               imputer,
               train_dir,
               seed):
    X_train_imputed = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
    X_train_imputed = X_train_imputed.astype({'Sex': int, 'Race': int})
    X_val_imputed = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
    X_val_imputed = X_val_imputed.astype({'Sex': int, 'Race': int})

    minmax_scaler = MinMaxScaler()
    columns = [col for col in X_train_imputed.columns if col not in {'Sex', 'Race'}]
    X_train_scaled = minmax_scaler.fit_transform(X_train_imputed[columns])
    X_train_imputed = pd.concat((pd.DataFrame(X_train_scaled, columns=columns), X_train_imputed[['Sex', 'Race']]),
                                axis='columns')
    X_val_scaled = minmax_scaler.transform(X_val_imputed[columns])
    X_val_imputed = pd.concat((pd.DataFrame(X_val_scaled, columns=columns), X_val_imputed[['Sex', 'Race']]),
                              axis='columns')

    X_train_1_hot = pd.get_dummies(X_train_imputed, columns=['Sex', 'Race'])
    X_val_1_hot = pd.get_dummies(X_val_imputed, columns=['Sex', 'Race'])

    n_vars = len(X_train_1_hot.columns)

    # Calculate bias for initialization of the final layer
    p = sum(y_val) / len(y_val)  # Frequency of positive training samples
    bias_init = - np.log(1 / p - 1)

    def make_nn_model(dropout_rate, learning_rate):
        model = tf.keras.Sequential([
            Dense(n_vars,
                  input_shape=(n_vars,),
                  # kernel_regularizer=tf.keras.regularizers.l1_l2(l1=.001, l2=.001),
                  activation='relu'),
            Dropout(rate=dropout_rate),  # rate is fraction of units to drop
            Dense(n_vars,
                  input_shape=(n_vars,),
                  # kernel_regularizer=tf.keras.regularizers.l1_l2(l1=.001, l2=.001),
                  activation='relu'),
            Dropout(rate=dropout_rate),
            Dense(1,
                  bias_initializer=tf.keras.initializers.Constant(bias_init),
                  activation='sigmoid')  # TF unable to compute metric AUC if activation here is linear
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=[tf.keras.metrics.AUC(name='auc')])
        return model


    def objective(params):
        tf.random.set_seed(seed)
        model = make_nn_model(dropout_rate=params['dropout_rate'],
                              learning_rate=params['learning_rate'])
        # print(model.summary())
        # pre = model.predict_proba(X_val_1_hot)
        res = model.fit(X_train_1_hot,
                        y_train,
                        validation_data=(X_val_1_hot, y_val),
                        epochs=params['epochs'],
                        batch_size=int(params['batch_size']),
                        verbose=0)

        best_epoch = np.argmax(res.history['val_auc'])
        val_auc = res.history['val_auc'][best_epoch]
        return -val_auc

    # Make an empty model, and then delete it, just to open CUDA libraries
    dummy = tf.keras.Sequential()
    del dummy
    rstate = np.random.RandomState(seed)
    best = fmin(fn=objective, space=params, algo=tpe.suggest, max_evals=max_evals, rstate=rstate)
    all_best_params = params
    for key, value in best.items():
        all_best_params[key] = value
    print('Best validated model was trained with hyper-parameters', all_best_params)
    print('Retraining and validating it')  # TODO change to x-validation
    model = make_nn_model(dropout_rate=all_best_params['dropout_rate'],
                              learning_rate=all_best_params['learning_rate'])
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=train_dir, histogram_freq=1, profile_batch=0)
    res = model.fit(X_train_1_hot,
                    y_train,
                    validation_data=(X_val_1_hot, y_val),
                    epochs=all_best_params['epochs'],
                    batch_size=int(all_best_params['batch_size']),
                    callbacks=[tensorboard_callback],
                    verbose=1)

    best_epoch = np.argmax(res.history['val_auc'])
    val_auc = res.history['val_auc'][best_epoch]
    val_loss = res.history['val_loss'][best_epoch]
    train_auc = res.history['auc'][best_epoch]
    train_loss = res.history['loss'][best_epoch]
    print('Best model epoch is',best_epoch)
    print(f'Training: loss={train_loss}   auc={train_auc}')
    print(f'Validation: loss={val_loss}   auc={val_auc}')

    # TODO Check https://github.com/tensorflow/tensorflow/issues/36465 and also https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth


def run_exp_log_regr(X_train,
                     y_train,
                     X_val,
                     y_val,
                     param_space,
                     max_evals,
                     imputer,
                     seed):
    # Build dataset with categorical variables one-hot encoded for logistic regression
    X_train_imputed_for_1_hot = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
    X_train_imputed_for_1_hot = X_train_imputed_for_1_hot.astype({'Sex': int, 'Race': int})
    X_val_imputed_for_1_hot = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
    X_val_imputed_for_1_hot = X_val_imputed_for_1_hot.astype({'Sex': int, 'Race': int})

    print('\nFitting logistic regression with iterative imputation')
    X_train_1_hot = pd.get_dummies(X_train_imputed_for_1_hot, columns=['Sex', 'Race'])
    X_val_1_hot = pd.get_dummies(X_val_imputed_for_1_hot, columns=['Sex', 'Race'])

    def objective(params):
        log_regr_model = LogisticRegression(**params)
        log_regr_model.fit(X_train_1_hot, y_train)
        y_log_regr_pred = log_regr_model.predict_proba(X_val_1_hot)[:, 1]
        log_regr_auc = roc_auc_score(y_val, y_log_regr_pred)
        return -log_regr_auc  # Hyperopt optimizes for minimum

    rstate = np.random.RandomState(seed)
    best_log_reg_params = fmin(fn=objective,
                               space=param_space,
                               algo=tpe.suggest,
                               max_evals=max_evals,
                               rstate=rstate)
    all_best_log_reg_params = param_space
    for key, value in best_log_reg_params.items():
        all_best_log_reg_params[key] = value

    log_regr_model = LogisticRegression(**all_best_log_reg_params)
    log_regr_model.fit(X_train_1_hot, y_train)
    y_log_regr_pred = log_regr_model.predict_proba(X_val_1_hot)[:, 1]
    log_regr_auc = roc_auc_score(y_val, y_log_regr_pred)
    print('Best model found has parameters', best_log_reg_params)
    print(f'Validation AUC is {log_regr_auc} achieved in {log_regr_model.n_iter_[0]} iterations')
    return log_regr_model


# Find out how many samples have missing data in one or more variables (columns)
def count_samples_with_missing_data(df):
    res = sum(df.isnull().any(axis='columns'))
    return res


def run_exp_bayes_hyperparams_opt(X_train,
                                  y_train,
                                  X_val,
                                  y_val,
                                  cat_features,
                                  param_space,
                                  max_evals,
                                  imputer,
                                  train_dir=None,
                                  weights=None,
                                  seed=None):
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
                                   task_type=params['task_type'],
                                   # early_stopping_rounds = params['early_stopping_rounds'],
                                   od_type=params['od_type'],
                                   od_wait=params['od_wait'])
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
                                     task_type=param_space['task_type'],
                                     od_type=param_space['od_type'],
                                     od_wait=param_space['od_wait'])
    training_res = refit_model.fit(train_pool,
                                   eval_set=val_pool,
                                   verbose=False)

    best_val_iter = training_res.best_iteration_  # This is the best iter. based on the validation metric, ROC AUC
    assert best_val_iter == np.argmax(training_res.evals_result_['validation']['AUC'])
    # Collect metrics at the iteration with the best validation metric (the iteration where the model is shrunk to)
    train_AUC = training_res.evals_result_['learn']['AUC'][best_val_iter]
    train_Logloss = training_res.evals_result_['learn']['Logloss'][best_val_iter]
    val_AUC = training_res.evals_result_['validation']['AUC'][best_val_iter]
    val_Logloss = training_res.evals_result_['validation']['Logloss'][best_val_iter]
    print(f"Training: Log loss={train_Logloss}   ROC AUC={train_AUC}")
    print(f"Validation: Log loss={val_Logloss}   ROC AUC={val_AUC}")
    print(f'Best iteration: {training_res.best_iteration_}')
    return refit_model


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
    w = np.full_like(y, neg_weight, dtype=float)
    w[y == 1] = pos_weight
    return w, {'total_pos': total_pos, 'total_neg': total_neg, 'pos_weight': pos_weight, 'neg_weight': neg_weight}


def run_exp_grid_hyperparams_opt(X,
                                 y,
                                 cat_features,
                                 iterations,
                                 param_grid,
                                 cv_folds,
                                 early_stopping_iters,
                                 imputer=None,
                                 train_dir=None,
                                 task_type='CPU',
                                 seed=None):
    dev_pool, X_inputed = make_imputed_pool(X, y, imputer, cat_features)
    model = CatBoostClassifier(iterations=iterations,
                               eval_metric='AUC:hints=skip_train~false',
                               cat_features=cat_features,
                               random_state=seed,
                               train_dir=train_dir,
                               task_type=task_type,
                               # early_stopping_rounds=True,
                               od_type='Iter',
                               od_wait=early_stopping_iters)

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
    print('Best params', grid_search_results['params'])
    print('Cross-validation: best AUC', best_AUC, 'obtained at iteration', best_iter,
          'with log-loss', loss_for_best_AUC)
    return model


def run_exp_sanity_check(X_train,
                         y_train,
                         max_evals,
                         train_dir='None',
                         task_type='CPU',
                         seed=None):
    model = CatBoostClassifier(iterations=max_evals,
                               random_state=seed,
                               task_type=task_type,
                               train_dir=train_dir,
                               custom_metric='AUC', )
    model.fit(X_train, y_train, verbose=False)
    scores = model.get_best_score()['learn']
    print(f"Final Logloss {scores['Logloss']}   final AUC {scores['AUC']}")

    return model


def main():
    #############################################################################################################
    seed = 42  # For random numbers generation
    iterations = 300  # Max number of iterations at every run of gradient boosting (max number of trees built)
    epochs = 50  # Max number of epochs for the NN model
    hyper_iterations = 3  # Number of iterations required during each Bayesian optimization of hyper-parameters
    log_regs_hyper_iterations = 10  # Number of iterations for hyper-parameters optimization for logistic regression
    cv_folds = 4  # Number of folds used for k-folds cross-validation
    logs_dir = Path('catboost_logs')  # Relative to the directory where the program is running
    task_type = 'GPU'  # Can be 'CPU' or 'GPU'
    early_stopping_iters = 10000  # Effectively disabled, as there is an issue with displaying the charts see https://github.com/catboost/catboost/issues/1468
    #############################################################################################################

    start_time = time()

    # Make the logs directory, if it doesn't exist already, and ensure it is empty
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

    dev_missing_count = count_samples_with_missing_data(X_dev)
    test_missing_count = count_samples_with_missing_data(X_test)

    print('\nDev. set missing data in', dev_missing_count, 'samples out of', len(X_dev))
    print('Test set missing data in', test_missing_count, 'samples out of', len(X_test))

    # Split the dev set into training and validation. The latter will be used for hyper-parameters tuning.
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.2, random_state=seed)

    # Make a dataset after dropping samples with missing data (note, no samples with missing data in test set)
    # X_dev_dropped = X_dev.dropna(axis='rows')
    # y_dev_dropped = y_dev.loc[X_dev_dropped.index]
    X_train_dropped = X_train.dropna(axis='rows')
    y_train_dropped = y_train.loc[X_train_dropped.index]
    X_val_dropped = X_val.dropna(axis='rows')
    y_val_dropped = y_val.loc[X_val_dropped.index]

    # Now impute missing values using the mean, instead of dropping samples containing them
    mean_imputer = SimpleImputer(strategy='mean', verbose=0)
    mean_imputer.fit(X_train)

    iter_imputer = IterativeImputer(random_state=seed, sample_posterior=False, max_iter=10, min_value=0, verbose=0)
    iter_imputer.fit(X_train)

    X_sanity = X_dev.dropna(axis='rows')
    y_sanity = y_dev.loc[X_sanity.index]
    X_sanity = pd.concat(
        [X_sanity[y_sanity == 1].sample(n=32, random_state=seed),
         X_sanity[y_sanity == 0].sample(n=32, random_state=seed)])
    y_sanity = y_sanity.loc[X_sanity.index]

    params = {'learning_rate': hp.loguniform('learning_rate', np.log(.0001), np.log(.01)),
              # 'learning_rate': 3e-4,
              'epochs': epochs,
              'batch_size': hp.quniform('batch_size', 16, 64, 1),
              'dropout_rate': hp.uniform('dropout_rate', .0, .5)}

    run_exp_nn(X_train,
               y_train,
               X_val,
               y_val,
               params=params,
               max_evals=hyper_iterations,
               imputer=iter_imputer,
               train_dir=str(logs_dir / 'tensorflow_logs_nn'),
               seed=seed)

    print('\nRunning sanity check')
    run_exp_sanity_check(X_sanity,
                         y_sanity,
                         max_evals=1000,
                         task_type=task_type,
                         seed=seed,
                         train_dir=str(logs_dir / 'catboost_logs_sanity'))

    log_regr_params = {'penalty': 'elasticnet',
                       'C': hp.uniform('C', .25, 4),
                       'class_weight': None,
                       'random_state': seed,
                       'solver': 'saga',
                       'max_iter': 10000,
                       'multi_class': 'ovr',
                       'n_jobs': -1,
                       'l1_ratio': hp.uniform('l1_ratio', .0, 1)}

    run_exp_log_regr(X_train,
                     y_train,
                     X_val,
                     y_val,
                     param_space=log_regr_params,
                     max_evals=log_regs_hyper_iterations,
                     imputer=iter_imputer,
                     seed=seed)

    cat_features = [3, 11]  # Categorical features are race and sex

    ''' Use the iterative imputer, but use Bayesian optimization for the hyper-parameters, instead of grid search. Here
    we use the train/val data sets

    Note: passing a CatBoost Pool() instance in the param_space values here below doesn't work, because hyperopt would
    throw an exception during optimization.'''

    ############################################################################################################
    param_space = {'learning_rate': hp.loguniform('learning_rate', np.log(.001), np.log(.2)),
                   # 'learning_rate': hp.uniform('learning_rate', .01, .1),
                   'depth': hp.quniform('depth', 4, 12, 1),
                   'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 9),
                   'bagging_temperature': hp.uniform('bagging_temperature', 0, 2),
                   'seed': seed,  # hyperopt accepts constant value parameters
                   'iterations': iterations,
                   'task_type': task_type,
                   # 'early_stopping_rounds': True,
                   'od_type': 'Iter',
                   'od_wait': early_stopping_iters
                   }
    ############################################################################################################

    print('\nPerforming Bayesian search for hyper-parameters optimization, after dropping samples with missing data')

    run_exp_bayes_hyperparams_opt(X_train_dropped,
                                  y_train_dropped,
                                  X_val_dropped,
                                  y_val_dropped,
                                  cat_features=cat_features,
                                  param_space=param_space,
                                  max_evals=hyper_iterations,
                                  imputer=None,
                                  train_dir=str(logs_dir / 'catboost_logs_drop'),
                                  seed=seed)

    print(
        '\nPerforming Bayesian search for hyper-parameters optimization, with missing data replaced with mean imputer')

    run_exp_bayes_hyperparams_opt(X_train,
                                  y_train,
                                  X_val,
                                  y_val,
                                  cat_features=cat_features,
                                  param_space=param_space,
                                  max_evals=hyper_iterations,
                                  imputer=mean_imputer,
                                  train_dir=str(logs_dir / 'catboost_logs_mean_imputer'),
                                  seed=seed)

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
                                                           train_dir=str(logs_dir / 'catboost_logs_iter_imputer'),
                                                           seed=seed)

    print('\nPerforming Bayesian search for hyper-parameters optimization, without replacement of missing data')

    selected_model = run_exp_bayes_hyperparams_opt(X_train,
                                                   y_train,
                                                   X_val,
                                                   y_val,
                                                   cat_features=cat_features,
                                                   param_space=param_space,
                                                   max_evals=hyper_iterations,
                                                   imputer=None,
                                                   train_dir=str(logs_dir / 'catboost_logs_keep_nan'),
                                                   seed=seed)

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
                                  train_dir=str(logs_dir / 'catboost_logs_weights'),
                                  seed=seed)

    ''' Grid-search is done on the dev. set, as the grid-search takes care of splitting it into training and validation.
    Note: if `search_by_train_test_split` is set to True, every combination of values of the hyper-parameters is evaluated
    with a basic training/val. split of the dataset; if set to False, then every combination is evaluated with x-evaluation.
    Once method grid_search() has selected the best combination of hyper-parameters, fits a model with it. The final model 
    can be evaluated with x-evaluation by setting parameter `calc_cv_statistics` to True (default). '''

    param_grid = {'learning_rate': np.arange(.01, .2, .02),
                  'depth': list(range(6, 10))}

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
                                 early_stopping_iters=early_stopping_iters,
                                 train_dir=str(logs_dir / 'catboost_logs_grid_search'),
                                 task_type=task_type)

    # Cross-validate the two selected models, and test them on the test set

    model = selected_model
    imputer = None

    print(f'\nCross-validating selected model.')
    params = model.get_params()
    params['loss_function'] = 'Logloss'
    params['eval_metric'] = 'AUC:hints=skip_train~false'
    params['train_dir'] = str(logs_dir / ('catboost_logs_cv_selected'))
    params['task_type'] = task_type
    # params['early_stopping_rounds'] = True
    params['od_type'] = 'Iter'
    params['od_wait'] = early_stopping_iters
    ''' Make a new imputer for cross-validation over the dev set. '''
    X_pool, _ = make_imputed_pool(X_dev, y_dev, imputer=imputer, cat_features=cat_features, weight=None)
    cv_results = cv(pool=X_pool,
                    params=params,
                    iterations=iterations,
                    fold_count=cv_folds,
                    partition_random_seed=seed,
                    stratified=True,
                    verbose=False)
    # Find the iteration with the best test AUC, its AUC and other train and test stats.
    best_cv_iter = np.argmax(cv_results['test-AUC-mean'])  # All the stats retrieved will refer to this same iteration
    best_cv_val_AUC = cv_results['test-AUC-mean'][best_cv_iter]
    best_cv_val_Logloss = cv_results['test-Logloss-mean'][best_cv_iter]
    best_cv_train_AUC = cv_results['train-AUC-mean'][best_cv_iter]
    best_cv_train_Logloss = cv_results['train-Logloss-mean'][best_cv_iter]
    print('Parameters:')
    for key, value in sorted(params.items()):
        print(f'   {key}={value}')
    print('Best cross-validation achieved at iteration', best_cv_iter)
    print(f'Training: Logloss {best_cv_train_Logloss}   ROC AUC {best_cv_train_AUC}')
    print(f'Validation: Logloss {best_cv_val_Logloss}   ROC AUC {best_cv_val_AUC}')

    print('Re-fitting the model on the dev. set and testing it')
    params['iterations'] = best_cv_iter + 1
    params['train_dir'] = None
    cv_model = CatBoostClassifier(**params)
    # training_res = cv_model.fit(X_pool, verbose=False)
    training_res = cv_model.fit(X_pool, verbose=False)
    # print('Iteration:', training_res.best_iteration_)
    y_test_preds = cv_model.predict_proba(X_test)[:, 1]
    test_AUC = roc_auc_score(y_test, y_test_preds)
    test_Logloss = log_loss(y_test, y_test_preds)
    print(f"Test on test set: Log loss={test_Logloss}   ROC AUC={test_AUC}")

    print(f'Overall train, validation and test run time: {round(time() - start_time)}s')

    print('\nFetaures importance based on prediction values change (%)')
    feature_importances = cv_model.get_feature_importance(X_pool)
    feature_names = X_dev.columns
    for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
        print('{}: {}'.format(name, score))

    print('\nFetaures importance based on loss (ROC AUC) values change')
    feature_importances_loss = cv_model.get_feature_importance(X_pool, type=EFstrType.LossFunctionChange)
    for score, name in sorted(zip(feature_importances_loss, feature_names), reverse=True):
        print('{}: {}'.format(name, score))

    # Plot a ROC curve for the x-validated model over the test set
    y_test_preds = cv_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_test_preds)
    fig, ax = plt.subplots()
    ax.set_title('ROC Curve')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True Positive rate')
    ax.set_ylim((0, 1))
    ax.set_xlim((0, 1))
    ax.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.plot([0, 1], [0, 1], color='blue', ls='--', lw=.5)
    ax.plot(fpr, tpr, color='blue', label='ROC')
    # ax.legend(loc='lower center')
    plt.show()


if __name__ == '__main__':
    main()

''' TODO: misc
Add early stoppoing to NN
Add SHAP to the jupyter Notebook
Use the whole HANES dataset from CDC or another survivale dataset e.g. https://archive.ics.uci.edu/ml/datasets/HCC+Survival explore Seaborne for preliminary data analysis
Instead of checking if survival after 10 years, estimate the number of years of survival
C-index is the same as the ROC AUC for logistic regression.
   see https://www.statisticshowto.com/c-statistic/#:~:text=A%20weighted%20c-index%20is,correctly%20predicting%20a%20negative%20outcome
   and also https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4886856/  and https://bit.ly/3dvUh07

'''
