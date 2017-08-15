#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 22:04:33 2017

@author: eyulush
"""

import common
import copy
import lightgbm as lgb
import pandas as pd
import numpy as np
import scipy as sp

from data_processing import get_extra_cat_data
from sklearn.model_selection import ParameterGrid


# logging configuration
logger = common.get_logger("lightbm_f1opt_pnone", "../logs/lightgbm_f1opt_pnone.log")

class pNoneEstimator():
    def __init__(self):
        self.reset()
        
    def reset(self):
        # used feature
        self.f_to_use = ['days_since_prior_order',
                         'order_dow', 'order_hour_of_day', 'order_number',
                         'pnone', 'user_avg_order_size', # 'user_id', why I use user_id??
                         'user_orders', 'user_reorders', 'user_total_items',
                         'user_total_orders', 'user_cat_20', 'user_cat_50', 'user_cat_100',
                         'user_order_starts_at', 'user_mean_days_since_prior', 'user_median_days_since_prior',
                         'user_reorder_ratio', 'user_cat_150', 'user_cat_300']
        self.f_to_data = copy.deepcopy(self.f_to_use)
        self.f_to_data.append('order_id')
        # parameters
        self.params= {'boosting_type': 'gbdt',
                      'is_unbalance': False,
                      'learning_rate': 0.02,
                      'max_depth': 4,
                      'metric': 'binary_logloss',
                      'min_data_in_leaf': 100,
                      'min_sum_hessian_in_leaf': 20,
                      'num_leaves': 16,
                      'objective': 'binary',
                      'scale_pos_weight': 1.0,
                      'task': 'train',
                      'feature_fraction': 0.9,
                      'verbose': 1}
        # training parameters
        self.num_boost_round = 1100
        self.d_train= None
    
    def set_features(self, f_list):
        self.f_to_use = f_list
        self.f_to_data = copy.deepcopy(self.f_to_use)
        self.f_to_data.append('order_id')
    
    def set_params(self, params):
        self.params = params
    
    def init_cv(self,df_train):
        # preprocessing df_train to get label for pnone
        # when label is 1, pnone is true, i.e. 0 reordered product happened
        df_pnone_labels=df_train.groupby('order_id').apply(lambda x: 0 if x.reordered.sum()>0 else 1)

        ### prepare data
        df_pnone_train=df_train[self.f_to_data].drop_duplicates()
        df_pnone_train.set_index('order_id',inplace=True)
        df_pnone_train=pd.concat([df_pnone_train,df_pnone_labels.rename('label')],axis=1)

        ### setup a model to run training
        self.d_train = lgb.Dataset(df_pnone_train[self.f_to_use], 
                          label=df_pnone_train.label.values)

    def cv(self, df_train):
        if self.d_train == None:
            self.init_cv(df_train)
            
        params = copy.deepcopy(self.params)
        cv_result = lgb.cv(params, self.d_train, num_boost_round=3000, nfold=5, stratified=True, \
                           shuffle=True, init_model=None, feature_name='auto', \
                           categorical_feature='auto', early_stopping_rounds=300, \
                           fpreproc=None, verbose_eval=True, show_stdv=True, seed=1234, callbacks=None)
        
        self.print_cv_result(cv_result)
        # best results shown as follow, finally, choose max_depth=4, leavs=16, lr=0.02, ff=0.95

        # best 6, 48, 555, lr=0.02 0.193635
        # best 6, 32, 564, lr=0.02 0.193498
        # best 4, 32 or 16, 1034, lr=0.02 0.193194
        # best 4, 32 or 16, 1016, ff=0.95 lr=0.02 0.193161
        # best 4, 16, 982, ff=0.95, lr=0.02, full candidates, features 0.191541
        # best 4, 16, 1100, ff=0.9, lr=0.02, full candidates, features 0.191541
        
        # best 4, 16, 843, ff=0.6, lr=0.02, full + extra user_cat_20 detail features, 0.19145244312795784
        return cv_result
    
    @staticmethod
    def print_cv_result(cv_result):
        for key in cv_result.keys():
            print("[%s] min=[%f] iter=[%d]"%(key, np.min(cv_result[key]), np.argmin(cv_result[key])))
        
    def get_pnone(self, df_train, df_test):
        ### prepare data
        df_pnone_labels=df_train.groupby('order_id').apply(lambda x: 0 if x.reordered.sum()>0 else 1)
        df_pnone_train=df_train[self.f_to_data].drop_duplicates()
        df_pnone_train.set_index('order_id',inplace=True)
        df_pnone_train=pd.concat([df_pnone_train,df_pnone_labels.rename('label')],axis=1)

        ### setup a model to run training
        d_train = lgb.Dataset(df_pnone_train[self.f_to_use], 
                          label=df_pnone_train.label.values)

        model = lgb.train(self.params, d_train, num_boost_round=self.num_boost_round)
        
        df_pnone_test=df_test[self.f_to_data].drop_duplicates()
        df_pnone_test.set_index('order_id',inplace=True)
        
        df_pnone_test['pnone_pred'] = model.predict(df_pnone_test)
        df_pnone_test['order_id'] = df_pnone_test.index
        return df_pnone_test[['order_id','pnone_pred']]

def get_bst_cv(cv_file='../output/lightgbm_pnone_cv.csv'):
    import os
    if os.path.isfile('../output/lightgbm_pnone_cv.csv'):
        bst_cv = pd.read_csv('../output/lightgbm_pnone_cv.csv')
    else:
        bst_cv=pd.DataFrame()
    return bst_cv

def get_list_stats(df_x):
    return pd.Series([np.mean(df_x[0]), np.std(df_x[0]), sp.stats.skew(df_x[0]), sp.stats.kurtosis(df_x[0])], 
                      index=['user_cat_mean', 'user_cat_std', 'user_cat_skew', 'user_cat_kur'])

if __name__ == '__main__':
    # get data
    # df_train,df_test = get_processed_data()
    
    #  try to use extra data
    df_train = common.load_df('../data/','df_imba_train')
    df_train['aisle_id']=df_train['aisle_id'].astype('category')
    df_train['department_id']=df_train['department_id'].astype('category')
    
    # load extra cat data 150, 300
    df_train = get_extra_cat_data(df_train)
    print (df_train.dtypes['user_cat_150'])
    print (df_train.dtypes['prod_cat_150'])

    # find the features which could be used for prediction pnone
    pne = pNoneEstimator()

    find_features = False
    if find_features:
        df_features = df_train[df_train.user_id<10000]
        order_numbers = df_features.order_id.unique().shape[0]
        feature_candidates = list()
        for key in df_features.keys():
            if (df_features.groupby('order_id').apply(lambda x: x[key].unique().shape[0] == 1)).sum() == order_numbers:
                feature_candidates.append(key)
                print(key)
        feature_candidates.remove('eval_set') if 'eval_set' in feature_candidates else None
        feature_candidates.remove('order_id') if 'order_id' in feature_candidates else None
        pne.set_features(feature_candidates)
    
    gridParams = {
        'task': ['train'],
        'min_data_in_leaf' : [100, 80], # new version of default parameters as 20
        'min_sum_hessian_in_leaf' : [10, 20], # new version of default parameters as 1e-3
        'boosting_type': ['gbdt'],
        'objective': ['binary'],
        'metric': {'binary_logloss'},
        'num_leaves':  [16, 48],
        'max_depth': [4, 6],
        'scale_pos_weight': [1.0],
        'is_unbalance' : [False],
        'feature_fraction': [0.6, 0.5, 0.4],
        'learning_rate' : [0.02, 0.01]
        #'early_stopping_rounds' : [20], # early_stopping_rounds is important. only when early_stopping happens, the best_iteration will be returned.
        #'decay': [0.995],
        #'min_learning_rate': [0.02, 0.01],
        # 'verbose_eval' : [True]
    }

    ## shrink the df_train for saving memory
    drop_cols = [k for k in df_train.keys() if k not in (pne.f_to_data + ['user_id', 'reordered'])]
    df_train.drop(drop_cols, axis=1, inplace=True)
    # df_train.drop_duplicates(inplace=True) # we can not drop duplicates because reordered information is kepted.

    ## load extra detailed category data, extra 
    num_topics = 50
    user_cat_name = "prior_user_cat_"+str(num_topics)
    df_user_cat = common.load_df("../data/", user_cat_name)
    df_user_cat.drop(['user_cat','user_cat_list'], axis=1, inplace=True)
    # rename columns
    ren_dict=dict()
    for k in df_user_cat.keys():
        if k != 'user_id':
            ren_dict[k] = '_'+k
    df_user_cat.rename(columns=ren_dict,inplace=True)    
    # merge to df_train
    df_train_debug=df_train.merge(df_user_cat, how='left', on='user_id').drop('user_id',axis=1)
    pne.set_features(pne.f_to_use + ren_dict.values())
    print(pne.f_to_data)
    
    '''
    ## extra stats information from user_cat, test with 100, not help
    nt_list = [20, 50, 100, 150, 300]
    nt_list = [100]
    for num_topics in nt_list:
        user_cat_name = "prior_user_cat_"+str(num_topics)
        df_user_cat = common.load_df("../data/", user_cat_name)
        df_user_cat_stats = df_user_cat.user_cat_list.apply(get_list_stats)
        df_user_cat_stats['user_id'] = df_user_cat['user_id']
        df_train_debug=df_train.merge(df_user_cat_stats, how='left', on='user_id').drop('user_id',axis=1)
        pne.set_features(pne.f_to_use + df_user_cat_stats.keys().drop('user_id').tolist())
        print(pne.f_to_data)
    '''
    
    # start to crossvaliation
    parameters = list(ParameterGrid(gridParams))     
    print('Total number of combinations ' + str(len(parameters)))

        
    for i in range(len(parameters)):
        print('current number %d in total combination %d'%(i, len(parameters)))
        
        logging_params = parameters[i]
        params = copy.deepcopy(logging_params)
        common.logging_dict(logger,logging_params,'cv parameters')
        
        pne.set_params(params)
        cv_result = pne.cv(df_train_debug)

        bst_cv=get_bst_cv(cv_file='../output/lightgbm_pnone_cv.csv')
        cv_result_dict = {'num_rounds' : [bst_cv.num_rounds.max() + 1 if bst_cv.shape[0] > 0 else 1], 
                          'best_score' : [np.min(cv_result['binary_logloss-mean'])],
                          'best_iteration' : [np.argmin(cv_result['binary_logloss-mean'])]}
        bst_cv = bst_cv.append(pd.DataFrame(cv_result_dict))
        bst_cv.to_csv('../output/lightgbm_pnone_cv.csv',index=False)
        
        common.logging_dict(logger, cv_result_dict, 'one cv result')
        
    # load the test data
    df_test = common.load_df('../data/','df_imba_test')
    df_test['aisle_id']=df_test['aisle_id'].astype('category')
    df_test['department_id']=df_test['department_id'].astype('category')
    
    # load extra cat data 150, 300
    df_test = get_extra_cat_data(df_test)
    print (df_test.dtypes['user_cat_150'])
    print (df_test.dtypes['prod_cat_150'])
    
    
    
    df_pnone_test = pne.get_pnone(df_train, df_test)
    
