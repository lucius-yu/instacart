#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 12:17:02 2017

@author: eyulush
"""

import common
import copy
import lightgbm as lgb
import pandas as pd
import numpy as np

from pNoneEstimator import pNoneEstimator
from common import TrainingCtrl, none_product
from data_processing import get_extra_cat_data
from f1_optimal import get_best_prediction

# control
training = True
train_validation = False
submission = False
sh1ng_f1=False

# logging configuration
logger = common.get_logger("lightbm_train", "../logs/lightgbm_train.log")

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
 'up_orders', 'up_first_order', 'up_last_order', 'up_mean_cart_position',
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

    # prediction = model.predict(data_val)
    orders = df_data.order_id.values
    products = df_data.product_id.values
    reordered = df_data.reordered.values
    data_eval = pd.DataFrame({'product_id': products, 'order_id': orders, 'prediction': preds, 'label' : reordered})
    
    # F1 opt and evaluation
    # old_best_result = common.pget_best_prediction(data_eval)
    # old_best_result.rename(inplace=True,columns={"order_id": "group", "reordered": "label", "bst_preds" : "pred"})
    # old_data_eval_score=common.f1_evaluate(old_best_result, 0.5)
    # print(old_data_eval_score.mean())
    
    # F1 opt, to be changed
    best_result = get_best_prediction(data_eval)
    best_result_dict=list()
    # order_id, product_id, reordered, predict
    # 1, 5000, 1, 0
    # 1, 36, 1, 1
    for index, row in best_result.iterrows():
        for pid in  row['products'].split():
            if pid == 'None':
                best_result_dict.append({'order_id': row['order_id'], 'product_id':none_product, 'pred': 1})
            else:
                best_result_dict.append({'order_id' : row['order_id'], 'product_id': int(pid), 'pred' : 1 })
    best_result=pd.DataFrame(best_result_dict)
   
    # data_eval processing
    df_none_eval = data_eval.groupby('order_id').apply(lambda x: pd.Series({'product_id': none_product, 'label': 1}) if x.label.sum()==0\
                                     else pd.Series({'product_id': none_product, 'label': 0}))
    df_none_eval.reset_index(inplace=True)
    df_none_eval = pd.concat([data_eval[df_none_eval.keys()], df_none_eval])
    
    # merge with best prediction
    best_result = df_none_eval.merge(best_result, how='left', on=['order_id','product_id']).fillna(0)

    # evaluation
    best_result.rename(inplace=True,columns={"order_id": "group", "reordered": "label"})
    best_result_score=common.f1_none_evaluate(best_result)
    return best_result_score.mean()

### training the model
def run_training(df_train, params, valid_train=True):
    d_train = lgb.Dataset(df_train[f_to_use],
                          label=df_train['reordered'],
                          free_raw_data=False)
    
    # parameter processing
    train_init_learning_rate=params.pop('init_learning_rate',0.1)
    train_decay=params.pop('decay',0.99)
    train_min_learning_rate=params.pop('min_learning_rate',0.01)
    
    # handle training parameters
    fitParams = ['early_stopping_rounds','num_boost_round','verbose_eval']
    fitKwargs = dict()
    for key in params.keys():
        if key in fitParams:
            fitKwargs[key] = params.pop(key)
                
    tctrl = TrainingCtrl(init_learning_rate=train_init_learning_rate,\
                         decay=train_decay,\
                         min_learning_rate=train_min_learning_rate)
    evals_result = dict()
    
    # train the model and use train_set as validation
    if valid_train:
        model = lgb.train(params, d_train, valid_sets=d_train, learning_rates=tctrl.get_learning_rate, evals_result=evals_result,**fitKwargs)
    else:
        model = lgb.train(params, d_train, learning_rates=tctrl.get_learning_rate, evals_result=evals_result,**fitKwargs)
    
    common.logging_dict(logger, evals_result, 'evals result')
    
    return model

if training:
    trainParams = {
        'task': 'train',
        'min_data_in_leaf' : 100, # new version of default parameters as 20
        'min_sum_hessian_in_leaf' : 20, # new version of default parameters as 1e-3
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves':  256,
        'max_depth': 10,
        'scale_pos_weight': 1.0,
        'is_unbalance' : False,
        'feature_fraction': 0.56,
        #'bagging_fraction': 0.95,
        #'bagging_freq': 5,
        'early_stopping_rounds' : 100, # early_stopping_rounds is important. only when early_stopping happens, the best_iteration will be returned.
        'num_boost_round': 3000, # num_boost_round, Number of boosted trees to fit 
        'decay': 0.995,
        'min_learning_rate': 0.02,
        'verbose_eval' : True
    }

    print(trainParams)
    common.logging_dict(logger,trainParams,'test logging')
    logger.debug('lgb_version=%f'%lgb.__version__)
    
    # load the data
    df_train = common.load_df('../data/','df_imba_train')
    df_train['aisle_id']=df_train['aisle_id'].astype('category')
    df_train['department_id']=df_train['department_id'].astype('category')
    
    # load extra cat data 150, 300
    df_train = get_extra_cat_data(df_train)
    print (df_train.dtypes['user_cat_150'])
    print (df_train.dtypes['prod_cat_150'])
    
    # the bst_model_id decided load model or do training
    bst_model_id = -1

    if bst_model_id < 0:
        print("execute the training")
        # run the training
        model=run_training(df_train, copy.deepcopy(trainParams))
    
        if train_validation:
            # training self evaluation
            train_eval_score = cv_evaluate(model, df_train, selected_iteration=None)

        # logging
        trainParams.pop('metric',None)
        common.save_model(model, trainParams, \
                          best_score=train_eval_score.mean() if train_validation else 0.0,\
                          best_iteration=model.current_iteration(),
                          notes='lgb_version='+str(lgb.__version__))
    
        detail_result={'best_iteration': model.best_iteration,
                       'best_score': train_eval_score.mean() if train_validation else 0.0,
                       'current_iteration': model.current_iteration(),
                       'current_score': train_eval_score.mean() if train_validation else 0.0,
                       'lgb_version' : lgb.__version__}
        common.logging_dict(logger, detail_result, 'train result') 
    else: 
        print("load the trained model")
        model, model_params = common.load_lgb_model(model_id=bst_model_id)
    
    # load the test data
    df_test = common.load_df('../data/','df_imba_test')
    df_test['aisle_id']=df_test['aisle_id'].astype('category')
    df_test['department_id']=df_test['department_id'].astype('category')
    
    # load extra cat data 150, 300
    df_test = get_extra_cat_data(df_test)
    
    # predict and submit
    preds = model.predict(df_test[f_to_use])
    
    # get F1 opt result
    if sh1ng_f1:
        orders = df_test.order_id.values
        products = df_test.product_id.values
        test_eval = pd.DataFrame({'product_id': products, 'order_id': orders, 'prediction': preds})

        test_result = get_best_prediction(test_eval)
        test_result[['order_id', 'products']].to_csv('../submit/sub.csv', index=False)
    else: # faron's f1 with pnone estimation +  add pnone_pred
        pne= pNoneEstimator()
        test_eval = df_test[['order_id','product_id']].copy()
        test_eval['preds'] = preds
        test_eval = test_eval.merge(pne.get_pnone(df_train, df_test), how='left', on='order_id')
        test_result =  common.pget_best_prediction(test_eval)
        test_result['preds']=test_result['bst_preds']
        test_result['order_id'] = test_result['order_id'].astype(np.int32)
        test_result['product_id'] = test_result['product_id'].astype(np.int32)
        test_submission = common.mk_none_submission('../submit/sub.csv', test_result)
    # cleanup
    del df_train
    del df_test
