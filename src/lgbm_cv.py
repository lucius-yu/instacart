#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 01:18:55 2017

@author: eyulush
"""

import common
import copy
import lightgbm as lgb
import pandas as pd
import numpy as np

import json

from common import TrainingCtrl, none_product
from pNoneEstimator import pNoneEstimator

from data_processing import get_extra_cat_data
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from f1_optimal import get_best_prediction
# control
cross_validation = True

# logging configuration
logger = common.get_logger("lightbm_cv", "../logs/lightgbm_cv.log")

# load data
df_train = common.load_df('../data/','df_imba_train')

# load extra cat data 150, 300
df_train = get_extra_cat_data(df_train)

# choose feature to use
f_to_use = ['aisle_id', 'aisle_orders', 'aisle_reorders',
 'days_since_prior_order',
 'department_id', 'department_orders', 'department_reorders',
 'order_dow', 'order_hour_of_day', 'order_number',
 'pnone',
 'prod_avg_add_to_cart_order',
 'prod_avg_order_days_intervals',
 'prod_avg_order_nums_intervals',
 'prod_orders',
 'prod_reorder_rate',
 'prod_reorders',
  # 'product_id',
  # 'reordered',
  #' user_avg_order_size',
  #' user_id',
 'user_orders',
 'user_prod_avg_add_to_cart_order',
 'user_prod_days_since_last_ordered',
 'user_prod_no_of_orders',
 'user_prod_order_days_expect',
 'user_prod_order_days_prob',
 'user_prod_order_nums_expect',
 'user_prod_order_nums_prob',
 'user_prod_order_rate',
 'user_prod_orders_since_first_ordered',
 'user_prod_orders_since_last_ordered',
 'user_prod_reorder_rate',
 'user_reorders',
 'user_total_items',
 'user_total_orders',
 'user_cat_20', 'user_cat_50', 'user_cat_100', 'user_cat_150', 'user_cat_300',
 'prod_cat_20', 'prod_cat_50', 'prod_cat_100', 'prod_cat_150', 'prod_cat_300',
 'user_prod_match_20', 'user_prod_match_50', 'user_prod_match_100', 'user_prod_match_150', 'user_prod_match_300',
 'aisle_reorder_rate',
 'department_reorder_rate',
 'prod_cat_20_reorder_rate',
 'user_cat_20_prod_reorder_rate',
 'user_cat_20_prod_cat_20_reorder_rate',
 'prod_cat_50_reorder_rate',
 'user_cat_50_prod_reorder_rate',
 'user_cat_50_prod_cat_50_reorder_rate',
 'prod_cat_100_reorder_rate',
 'user_cat_100_prod_reorder_rate',
 'user_cat_100_prod_cat_100_reorder_rate',
 'dep_products',
 #'dep_reordered',
 'aisle_products',
 #'aisle_reordered',
 #'prod_users_unq_reordered',
 # eyulus: remove extra features
 #'up_orders', 
 'up_first_order', 'up_last_order', 'up_mean_cart_position',
 'up_median_cart_position', 'days_since_prior_order_mean',
 #'days_since_prior_order_median',
 'order_dow_mean',
 #'order_dow_median',
 'order_hour_of_day_mean',
 #'order_hour_of_day_median',
 'add_to_cart_order_inverted_mean',
 'add_to_cart_order_inverted_median',
 'add_to_cart_order_relative_mean',
 'add_to_cart_order_relative_median',
 'user_order_starts_at', 'user_mean_days_since_prior',
 'user_median_days_since_prior', 'user_reorder_ratio',
  #'up_order_rate',
  #'dep_reordered_ratio',
 'last', 'prev1', 'prev2', 'median', 'mean',
 '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
 '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
 '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
 '30', '31']

# evaluation on valid data
def cv_evaluate(model, df_data, selected_iteration=None):
    if selected_iteration:
        preds = model.predict(df_data[f_to_use], num_iteration=selected_iteration)
    else:
        if model.best_iteration > 10:
            preds = model.predict(df_data[f_to_use], num_iteration=model.best_iteration)
        else:    
            preds = model.predict(df_data[f_to_use])

    data_eval = df_data[['order_id','product_id', 'reordered', 'pnone_pred']].copy()
    data_eval['preds'] = preds
    
    # F1 opt
    best_result = common.pget_best_prediction(data_eval)
            
    # evaluation
    best_result.rename(inplace=True,columns={"order_id": "group", "reordered": "label", "bst_preds" : "pred"})
    data_eval_score=common.f1_none_evaluate(best_result)
    print(data_eval_score.mean())

    # compare result 0.40511968 vs 0.4044575
    '''
    wn_data_eval = df_data[['order_id','product_id', 'reordered']].copy()
    wn_data_eval['preds'] = preds
    
    # F1 opt
    wn_best_result = common.pget_best_prediction(wn_data_eval)
    
    wn_best_result.rename(inplace=True,columns={"order_id": "group", "reordered": "label", "bst_preds" : "pred"})
    wn_data_eval_score=common.f1_none_evaluate(wn_best_result)
    print(wn_data_eval_score.mean())
    '''
    return data_eval_score.mean()

# run one cross validation for a params
# cv_iter marks only 3 iteration is done even we have 5 fold
# cv_iter <= n_splits
def run_cross_validation(df_data, params, n_splits=5, f1_eval=False, cv_iter=None):
    # parameter processing
    ori_params=copy.deepcopy(params)
    
    if cv_iter==None:
        cv_iter= n_splits
    
    # records
    cv_result=pd.DataFrame(columns=['best_training_iteration',\
                                    'best_training_score',\
                                    'best_valid_iteration',\
                                    'best_valid_score',\
                                    'best_eval_score',\
                                    'params'])

    # eval_threshold=params.pop('threshold',0.21)
    train_init_learning_rate=params.pop('init_learning_rate',0.1)
    train_decay=params.pop('decay',0.99)
    train_min_learning_rate=params.pop('min_learning_rate',0.01)
    
    # handle training parameters
    fitParams = ['early_stopping_rounds','num_boost_round','verbose_eval']
    fitKwargs = dict()
    for key in params.keys():
        if key in fitParams:
            fitKwargs[key] = params.pop(key)

    # setup k fold
    df_pnone_labels=df_data.groupby('order_id').apply(lambda x: 0 if x.reordered.sum()>0 else 1)
    skf = StratifiedKFold(n_splits=n_splits, random_state=1234)
    skf_iterator = skf.split(df_pnone_labels.index.values, df_pnone_labels.values.tolist())

    # train_group_indices, test_group_indices = skf_iterator.next()
    n_iter = 0
    for train_group_indices, test_group_indices in skf_iterator:
        if n_iter >= cv_iter:
            break
        else: 
            n_iter = n_iter + 1
        # fetch data
        df_cv_train=df_data[df_data.order_id.isin(df_pnone_labels.index.values[train_group_indices])]
        df_cv_valid=df_data[df_data.order_id.isin(df_pnone_labels.index.values[test_group_indices])]
        # construct the d_train and d_value
        d_train = lgb.Dataset(df_cv_train[f_to_use],
                              label=df_cv_train['reordered'],
                              free_raw_data=False)
        d_valid = lgb.Dataset(df_cv_valid[f_to_use],
                              label=df_cv_valid['reordered'],
                              free_raw_data=False)
        
        tctrl = TrainingCtrl(init_learning_rate=train_init_learning_rate,\
                             decay=train_decay,\
                             min_learning_rate=train_min_learning_rate)
        
        evals_result = dict()
        
        model = lgb.train(params, d_train, valid_sets=[d_train, d_valid],\
                          learning_rates=tctrl.get_learning_rate,\
                          evals_result=evals_result,**fitKwargs)
        # add pnone_pred
        pne= pNoneEstimator()
        df_cv_valid = df_cv_valid.merge(pne.get_pnone(df_cv_train, df_cv_valid), \
                                        how='left', on='order_id')

        
        best_training_score=min(evals_result['training']['binary_logloss'])
        best_training_iteration=np.argmin(evals_result['training']['binary_logloss'])
        best_valid_score=min(evals_result['valid_1']['binary_logloss'])
        best_valid_iteration=np.argmin(evals_result['valid_1']['binary_logloss'])
        best_eval_score=cv_evaluate(model,df_cv_valid) if f1_eval else 0

        ori_params.pop('metric',None)
        cv_result=cv_result.append(pd.DataFrame({\
                    'best_training_score': [best_training_score],\
                    'best_training_iteration': [best_training_iteration],\
                    'best_valid_score': [best_valid_score],\
                    'best_valid_iteration': [best_valid_iteration],\
                    'best_eval_score': [best_eval_score],
                    'lgb_version': [lgb.__version__],
                    'params':json.dumps(ori_params)}))
        # explore
        print("Features importance...")
        gain = model.feature_importance('gain')
        ft = pd.DataFrame({'feature':model.feature_name(), 'split':model.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
        print(ft)
        
        logger.debug('train and valid loss')
        common.logging_dict(logger, evals_result, 'evals result')
        logger.debug(ft.to_string())
        logger.debug([best_eval_score, best_training_score, best_training_iteration, best_valid_score, best_valid_iteration])
    
        del df_cv_train
        del df_cv_valid
        del d_train
        del d_valid
        del model

    print cv_result.best_eval_score.mean()
    return cv_result

def get_bst_cv(cv_file='../output/lightgbm_cv.csv'):
    import os
    if os.path.isfile('../output/lightgbm_cv.csv'):
        bst_cv = pd.read_csv('../output/lightgbm_cv.csv')
    else:
        bst_cv=pd.DataFrame()
    return bst_cv
        
if cross_validation:
    gridParams = {
        'task': ['train'],
        'min_data_in_leaf' : [100,80], # new version of default parameters as 20
        'min_sum_hessian_in_leaf' : [20], # new version of default parameters as 1e-3
        'boosting_type': ['gbdt'],
        'objective': ['binary'],
        'metric': {'binary_logloss'},
        'num_leaves':  [256],
        'max_depth': [10],
        'scale_pos_weight': [1.0],
        'is_unbalance' : [False],
        'feature_fraction': [0.56, 0.6],
        'early_stopping_rounds' : [60], # early_stopping_rounds is important. only when early_stopping happens, the best_iteration will be returned.
        'num_boost_round': [6000], # num_boost_round, Number of boosted trees to fit 
        'decay': [0.995],
        'min_learning_rate': [0.02],
        'verbose_eval' : [True]
    }
    
    # should we update lr to 0.01 after 1000 rounds?
    
    # start to crossvaliation
    parameters = list(ParameterGrid(gridParams))     
    print('Total number of combinations ' + str(len(parameters)))

    n_fold=5 # 5 fold cv, which will close final training
    for i in range(len(parameters)):
        print('current number %d in total combination %d'%(i, len(parameters)))
        
        logging_params = parameters[i]
        params = copy.deepcopy(logging_params)
        
        common.logging_dict(logger,logging_params,'cv parameters')

        
        df_cv_result = run_cross_validation(df_train, params, n_splits=n_fold, f1_eval=True, cv_iter=3)
        
        # save and logging
        bst_cv=get_bst_cv(cv_file='../output/lightgbm_cv.csv')
        df_cv_result['num_rounds'] = bst_cv.num_rounds.max() + 1 if bst_cv.shape[0] > 0 else 1
        bst_cv = bst_cv.append(df_cv_result)
        bst_cv.to_csv('../output/lightgbm_cv.csv',index=False)
        
        logger.debug('one cv result \n' + df_cv_result.to_string())
        print df_cv_result
        
        
    # common.logging_dict(logger, bst_cv, 'cv results')
