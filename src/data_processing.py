#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 10:44:48 2017

@author: eyulush
"""
import common
import os
import pandas as pd
import numpy as np
import scipy as sp

from common import grouper,papply
from timeline_processing import get_timeline_data

# get product statistics from prior
# orders, for a product how many times has been ordered
# reorders, for a product how many times has benn re-ordered
# avg_add_to_cart_order, for a product the average order of 'add-to-cart' when it has been added to cart
def get_prod_stats_from_prior(priors):
    prods = priors.drop_duplicates(subset='product_id')[['product_id','aisle_id','department_id']]
    prods.index = prods.product_id
    prods.sort_index(inplace=True)
    prods['prod_orders'] = priors.groupby(priors.product_id).size().astype(np.int32)
    prods['prod_reorders'] = priors[priors.reordered==1].groupby(priors.product_id).size().astype(np.int32)
    prods['prod_avg_add_to_cart_order'] = priors[priors.reordered==1].groupby(priors.product_id).add_to_cart_order.mean().astype(np.float16)
    
    aisles = pd.DataFrame()
    aisles['aisle_orders'] = priors.groupby('aisle_id').size().astype(np.int32)
    aisles['aisle_reorders'] = priors[priors.reordered==1].groupby(priors.aisle_id).size().astype(np.int32)
    aisles['aisle_id'] = aisles.index
    
    departments = pd.DataFrame()
    departments['department_orders'] = priors.groupby('department_id').size().astype(np.int32)
    departments['department_reorders'] = priors[priors.reordered==1].groupby(priors.department_id).size().astype(np.int32)
    departments['department_id'] = departments.index.astype(np.uint8)
    
    return (prods.merge(aisles, how='left', on='aisle_id')).merge(departments, how='left', on='department_id')

# support functions 
# combine list in rows which grouped by product.
# time consuming operation, use multi-processing
def combine_list_by_prod(df_data):
    return df_data.groupby('product_id').agg({'user_prod_reorder_nums_intervals':lambda x: sum(x.tolist(),[]),
                                              'user_prod_reorder_days_intervals':lambda x: sum(x.tolist(),[]) })

# get product statistics from timeline data
# reorder-rate, sum(the given product has been reordered by given user) divided by sum(the orders which user has made after the product first ordered)
# avg_order_nums_intervals, average of reorder numbers intervals 
# avg_order_days_intervals, average of reorder days intervals
def get_prod_stats_from_td(df_timeline):

    prods = pd.DataFrame()

    # reorder_rate
    prods['prod_reorder_rate']=df_timeline.groupby('product_id').\
        apply(lambda order: (sum(order.user_prod_no_of_orders-1) / float(sum(order.user_prod_orders_since_first_ordered))) if sum(order.user_prod_orders_since_first_ordered) > 0 else 0.0)
    # combine reorder_nums_interval list which grouped by product_id
    group_size = 10000
    product_ids = prods.index.tolist()
    prods = prods.merge(right=papply([df_timeline[df_timeline.product_id.isin(pids)] for pids in grouper(group_size, product_ids)],combine_list_by_prod),\
                how='left', left_index=True, right_index=True)
    # calculate average 
    prods['prod_avg_order_nums_intervals'] = prods.user_prod_reorder_nums_intervals.apply(lambda x: np.mean(x)).astype(np.float16)
    prods['prod_avg_order_days_intervals'] = prods.user_prod_reorder_days_intervals.apply(lambda x: np.mean(x)).astype(np.float16)
    prods['product_id'] = prods.index
    return prods

# wrapper function
def get_prod_stats(df_priors,df_td):
    stats_priors = get_prod_stats_from_prior(df_priors)
    stats_td = get_prod_stats_from_td(df_td)
    return pd.merge(stats_priors,stats_td,how='left',on='product_id')

def get_user_stats_from_priors(priors):
    ### User orders, reorders, reorder_rate, total items, total orders, average order size,
    users = pd.DataFrame()
    users['user_orders'] = priors.groupby(priors.user_id).size().astype(np.int32)
    users['user_reorders'] = priors[priors.reordered>0].groupby(priors.user_id).size().astype(np.uint16)
    users['user_total_items'] = priors.groupby(['user_id','order_number']).size().groupby('user_id').sum().astype(np.uint16)
    users['user_total_orders'] = priors.groupby(priors.user_id).order_number.max()
    users['user_avg_order_size'] = users['user_total_items'] / users['user_total_orders'].astype(np.float16)
    users['user_id']=users.index
    return users

# wrapper function
def get_user_stats(df_priors,df_td):
    stats_priors = get_user_stats_from_priors(df_priors)
    return stats_priors

### userXprod statistics
def get_userXprod_stats(df_priors,df_td):
    userXprod = df_td[['user_id', 'product_id','user_prod_days_since_last_ordered', 
                       'user_prod_no_of_orders','user_prod_order_rate',
                       'user_prod_orders_since_first_ordered','user_prod_orders_since_last_ordered',
                       'user_prod_reorder_rate']]
    userXprod['user_id'] = userXprod['user_id'].astype(np.int32)
    userXprod['product_id'] = userXprod['product_id'].astype(np.uint16)
    userXprod['user_prod_days_since_last_ordered'] = userXprod['user_prod_days_since_last_ordered'].astype(np.uint16)
    userXprod['user_prod_no_of_orders'] = userXprod['user_prod_no_of_orders'].astype(np.uint16)
    userXprod['user_prod_orders_since_first_ordered'] = userXprod['user_prod_orders_since_first_ordered'].astype(np.uint16)
    userXprod['user_prod_orders_since_last_ordered'] = userXprod['user_prod_orders_since_last_ordered'].astype(np.uint16)
    userXprod['user_prod_order_rate'] = userXprod['user_prod_order_rate'].astype(np.float32)
    userXprod['user_prod_reorder_rate'] = userXprod['user_prod_reorder_rate'].astype(np.float32)

    return userXprod

# in case only 1 reorder e.g. [1.0] which should be different with [1.0,1.0,1.0,1.0] although both scale == 1
# so, finally we need to multiply with belief, i.e. the total reorders for a given product
# another thing : for scale parameter, we just use mean value of number intervals, the sp.stats.expon.fit result will be mean value - 1
def get_expect_value(df_data):
    if df_data.prod_avg_order_days_intervals >= 1.0 and df_data.prod_avg_order_nums_intervals >= 1.0:
        prod_days_prob = sp.stats.expon.pdf(df_data.user_prod_days_since_last_ordered, loc=0, scale=df_data.prod_avg_order_days_intervals)
        prod_days_expect = prod_days_prob * df_data.prod_reorders
        prod_nums_prob = sp.stats.expon.pdf(df_data.user_prod_orders_since_last_ordered, loc=1, scale=df_data.prod_avg_order_nums_intervals) 
        prod_nums_expect = prod_nums_prob * df_data.prod_reorders
    else:
        prod_days_prob = 0.0
        prod_days_expect = 0.0
        prod_nums_prob = 0.0
        prod_nums_expect = 0.0
    return pd.Series({'user_prod_order_days_prob': prod_days_prob,\
                      'user_prod_order_days_expect': prod_days_expect,\
                      'user_prod_order_nums_prob': prod_nums_prob,\
                      'user_prod_order_nums_expect': prod_nums_expect})

def get_expect_values(df_data):
    return df_data.apply(get_expect_value, axis=1)

# user and product category data processing
def load_user_prod_cat_data(num_topics=20, rename=True):
    ### load cat data
    user_cat_name = "prior_user_cat_"+str(num_topics)
    df_user_cat = common.load_df("../data/", user_cat_name)
    df_user_cat = df_user_cat.set_index('user_id', drop=False)

    prod_cat_name = "prior_prod_cat_"+str(num_topics)
    df_prod_cat = common.load_df("../data/", prod_cat_name)
    df_prod_cat = df_prod_cat.set_index('prod_id', drop=False)

    user_prod_match_name = "prior_user_prod_match_"+str(num_topics)
    df_user_prod_match = common.load_df("../data/", user_prod_match_name)
    
    # reset dtype to category
    df_user_cat['user_cat'] = df_user_cat['user_cat'].astype('category')
    df_prod_cat['prod_cat'] = df_prod_cat['prod_cat'].astype('category')
    
    if rename:
        df_user_cat.rename(columns={"user_cat": "user_cat_"+str(num_topics)},inplace=True)
        df_prod_cat.rename(columns={"prod_cat": "prod_cat_"+str(num_topics)}, inplace=True)
        df_prod_cat.rename(columns={"prod_id": "product_id"}, inplace=True)
        df_user_prod_match.rename(columns={"user_prod_match": "user_prod_match_"+str(num_topics)}, inplace=True)
        df_user_prod_match.rename(columns={'prod_id':'product_id'},inplace=True)
    
    return df_user_cat, df_prod_cat, df_user_prod_match

    
def get_user_prod_cat_data(num_cats_list=[20,50,100]):
    user_cat = pd.DataFrame()
    prod_cat = pd.DataFrame()
    user_prod_cat_match = pd.DataFrame()
    for num_cats in num_cats_list:
        user_cat_data, prod_cat_data, user_prod_cat_match_data = load_user_prod_cat_data(num_cats)
        if user_cat.shape[0]>0:
            user_cat = pd.merge(user_cat, user_cat_data[['user_id','user_cat_'+str(num_cats)]], \
                                                         how='left', on ='user_id')
            prod_cat = pd.merge(prod_cat, prod_cat_data[['product_id','prod_cat_'+str(num_cats)]], \
                                                         how='left', on ='product_id')
            user_prod_cat_match = pd.merge(user_prod_cat_match, user_prod_cat_match_data, \
                                                         how='left', on=['product_id','user_id'])
        else:
            user_cat = (user_cat_data[['user_id','user_cat_'+str(num_cats)]]).copy()
            prod_cat = (prod_cat_data[['product_id','prod_cat_'+str(num_cats)]]).copy()
            user_prod_cat_match = user_prod_cat_match_data.copy()
                                                         
    return user_cat, prod_cat, user_prod_cat_match

def get_user_cat_stats(priors):
    user_cat_stats = pd.DataFrame()
    user_cat_stats['user_cat_20_orders'] = priors.groupby('user_cat_20').size().astype(np.int32)
    user_cat_stats['user_cat_20_reorders'] = priors[priors.reordered==1].groupby(priors.user_cat_20).size().astype(np.int32)
    user_cat_stats['user_cat'] = user_cat_stats.index
    return user_cat_stats

def get_prod_cat_stats(priors):
    prod_cat_stats = pd.DataFrame()
    prod_cat_stats['prod_cat_20_orders'] = priors.groupby('prod_cat_20').size().astype(np.int32)
    prod_cat_stats['prod_cat_20_reorders'] = priors[priors.reordered==1].groupby(priors.prod_cat_20).size().astype(np.int32)
    prod_cat_stats['prod_cat'] = prod_cat_stats.index
    return prod_cat_stats

def preprocessing_data():
    IDIR = '../input/'

    # load the basic data
    priors, train, orders, products, aisles, departments  = common.load_raw_data(IDIR)

    ### Preprocessing, combine order, product information into priors
    print('add order info to priors')
    orders.set_index('order_id', inplace=True, drop=False)
    priors = priors.join(orders, on='order_id', rsuffix='_')
    priors.drop('order_id_', inplace=True, axis=1)
    
    ### 
    print('add product info to priors')
    priors=pd.merge(priors,products,how='left',on='product_id')
    
    return priors, train, orders, products, aisles, departments

def processing_data():
    ### loading the data
    IDIR = '../input/'

    # load the basic data
    priors, train, orders, products, aisles, departments  = common.load_raw_data(IDIR)

    print('priors {}: {}'.format(priors.shape, ', '.join(priors.columns)))
    print('orders {}: {}'.format(orders.shape, ', '.join(orders.columns)))
    print('train {}: {}'.format(train.shape, ', '.join(train.columns)))

    # load timeline related data 
    timeline_data = get_timeline_data()
    
    # load user and product category data
    user_cat_data, prod_cat_data, user_prod_cat_match_data = get_user_prod_cat_data()
    
    ### Preprocessing, combine order, product information into priors
    print('add order info to priors')
    orders.set_index('order_id', inplace=True, drop=False)
    priors = priors.join(orders, on='order_id', rsuffix='_')
    priors.drop('order_id_', inplace=True, axis=1)
    
    ### 
    print('add product info to priors')
    priors=pd.merge(priors,products,how='left',on='product_id')
    
    ### get product statistic data
    prod_stats = get_prod_stats(priors,timeline_data)
    print('prod_stats {}: {}'.format(prod_stats.shape, ', '.join(prod_stats.columns)))

    ### get user statistic data
    user_stats = get_user_stats(priors,timeline_data)
    print('user_stats {}: {}'.format(user_stats.shape, ', '.join(user_stats.columns)))
    
    ### get userXprod statistic data
    userXprod_stats = get_userXprod_stats(priors,timeline_data)
    print('userXprod_stats {}: {}'.format(userXprod_stats.shape, ', '.join(userXprod_stats.columns)))
    
    ### get user category statistic data
    print('add user_cat info to priors')
    priors=pd.merge(priors, user_cat_data, how='left', on='user_id')
    
    user_cat_stats = get_user_cat_stats(priors)
    print('user_cat_stats {}: {}'.format(user_cat_stats.shape, ', '.join(user_cat_stats.columns)))

    print('add prod_cat info into priors')
    priors=pd.merge(priors, prod_cat_data, how='left', on='product_id')

    prod_cat_stats = get_prod_cat_stats(priors)
    print('prod_cat_stats {}: {}'.format(prod_cat_stats.shape, ', '.join(prod_cat_stats.columns)))

    ## question here
    ## why not merge user_cat_stats and prod_cat_stats?
    ## Note the function only processed stats for cat_20
    
    ### postprocessing for training data
    # user userXprod_stat as base to construct train and test data.
    print('post processing stats data')
    df_x = userXprod_stats
    df_x = df_x.merge(prod_stats.drop(['user_prod_reorder_nums_intervals', 'user_prod_reorder_days_intervals'],axis=1),\
                      how='left',on='product_id')
    df_x = df_x.merge(user_stats,how='left', on='user_id')
    df_x = df_x.merge(orders[orders.eval_set!='prior'],how='left',on='user_id')
    
    # merge category data
    df_x = df_x.merge(user_cat_data,how='left',on='user_id')
    df_x = df_x.merge(prod_cat_data,how='left', on='product_id')
    df_x = df_x.merge(user_prod_cat_match_data,how='left',on=['user_id','product_id'])
    
    #df_train = df_train.merge(user_cat_data,how='left',on='user_id')
    #df_train = df_train.merge(prod_cat_data,how='left', on='product_id')
    #df_train=pd.merge(df_train,user_prod_cat_match_data, how='left',on=['user_id', 'product_id'])
    #df_test = df_test.merge(user_cat_data,how='left',on='user_id')
    #df_test = df_test.merge(prod_cat_data,how='left', on='product_id')
    #df_test=pd.merge(df_test,user_prod_cat_match_data, how='left',on=['user_id', 'product_id'])
    
    print df_x.shape
    print df_x.memory_usage()
    
    ### release memory after merge
    del priors
    del timeline_data
    del prod_stats, user_stats, userXprod_stats
    
    # calculate extra features
    print('processing extra feature')
    df_x['user_prod_days_since_last_ordered'] = df_x['user_prod_days_since_last_ordered'] + df_x['days_since_prior_order']
    df_x['user_prod_orders_since_last_ordered'] = df_x['user_prod_orders_since_last_ordered'] + 1
    
    print('processing expect values')
    # combine reorder_nums_interval list which grouped by product_id
    group_size = 20000
    user_ids = df_x.user_id.unique().tolist()
    
    df_x = df_x.merge(right=papply([df_x[df_x.user_id.isin(uids)] for uids in grouper(group_size, user_ids)],get_expect_values),\
                how='left', left_index=True, right_index=True)
    # df_x['user_prod_days_prob'] = df_x.apply(lambda x: sp.stats.expon.pdf(x.user_prod_days_since_last_ordered, loc=0,scale=x.prod_avg_order_days_intervals),axis=1)
    # df_x['user_prod_orders_prob'] = df_x.apply(lambda x: sp.stats.expon.pdf(x.user_prod_orders_since_last_ordered, loc=1,scale=x.prod_avg_order_nums_intervals),axis=1)
    print('finished processing expect values')
    
    # split the train and test for df_x
    df_train=df_x[df_x.eval_set=='train']
    df_test=df_x[df_x.eval_set=='test']
    
    # merge the label into df_train
    df_train=df_train.merge(train,how='left',on=['order_id','product_id'])
    df_train['reordered'] = df_train.reordered.fillna(0)
    
    print("processing data finished!")
    return df_train, df_test

### other function
def get_processed_data(path="../data/", train_data_name = "train_data", test_data_name = "test_data"):
    if os.path.isfile(path+train_data_name+".csv.gz") and os.path.isfile(path+test_data_name+".csv.gz"):
        # load the data
        df_train = common.load_df(path,train_data_name)
        df_test = common.load_df(path,test_data_name)
    else:
        print "no data, start processing"
        df_train, df_test = processing_data()
        # save the df_train, df_test
        common.save_df(df_train,"../data/", train_data_name, index=False)
        common.save_df(df_test,"../data/", test_data_name, index=False)
    return df_train, df_test

# dow processing
def get_p_dow(order):
    p_dow=[1,1,1,1,1,1,1]
    for i in order.order_dow.tolist():
        p_dow[i]=p_dow[i]+1
    tot = float(sum(p_dow))
    return [p/tot for p in p_dow]

# some other features
def extra_feature_processing(path="../data/", train_data_name = "train_data", test_data_name = "test_data"):
    # load the data
    df_train = common.load_df(path,train_data_name)
    df_test = common.load_df(path,test_data_name)
    
    print df_train.shape
    print df_test.shape
    
    # user prod add_to_cart_order mean, std
    priors, train, orders, products, aisles, departments = preprocessing_data()
    
    # get mean of add to cart order
    user_prod_avg_add_to_cart_order=\
        pd.DataFrame(priors.groupby(['user_id','product_id']).add_to_cart_order.agg(np.mean))
    user_prod_avg_add_to_cart_order.rename( columns={'add_to_cart_order':'user_prod_avg_add_to_cart_order'},\
                                           inplace=True)
    df_train = df_train.merge(user_prod_avg_add_to_cart_order,\
                          how='left', left_on=['user_id','product_id'], right_index=True)
    df_test = df_test.merge(user_prod_avg_add_to_cart_order,\
                          how='left', left_on=['user_id','product_id'], right_index=True)
    
    
    # order_dow processing
    # priors.groupby(['user_id','product_id']).apply(get_p_dow)
    
    # load timeline related data 
    # timeline_data = get_timeline_data()
    
    # save the df_train, df_test
    print "save the processed data"
    common.save_df(df_train,"../data/", train_data_name, index=False)
    common.save_df(df_test,"../data/", test_data_name, index=False)
    
# reorder_rate_processing
def extra_reorder_rate_processing(path="../data/", train_data_name = "train_data", test_data_name = "test_data"):
    # load the data
    df_train = common.load_df(path,train_data_name)
    df_test = common.load_df(path,test_data_name)
    
    df_all = pd.concat([df_train,df_test])
    assert(df_all.shape[0] == df_train.shape[0]+df_test.shape[0])
    
    print df_all.shape
    print df_all.keys()
    
    # here the cat 50, 100 is totally wrong.
    # re-processing here
    # drop old cat columns
    drop_cols = ['aisle_reorder_rate', 'department_reorder_rate',
                 'prod_cat_20','prod_cat_50','prod_cat_100',
                 'user_cat_20','user_cat_50','user_cat_100',
                 'user_prod_match_20', 'user_prod_match_50','user_prod_match_100',
                 'prod_cat_20_reorder_rate', 'user_cat_20_prod_reorder_rate',
                 'user_cat_20_prod_cat_20_reorder_rate', 'prod_cat_50_reorder_rate',
                 'user_cat_50_prod_reorder_rate',
                 'user_cat_50_prod_cat_50_reorder_rate', 'prod_cat_100_reorder_rate',
                 'user_cat_100_prod_reorder_rate',
                 'user_cat_100_prod_cat_100_reorder_rate']
    df_all.drop(drop_cols,axis=1, inplace=True, errors='ignore')
    print df_all.shape
    
    # load user and product category data
    user_cat_data, prod_cat_data, user_prod_cat_match_data = get_user_prod_cat_data()
    
    # merge category data
    df_all = df_all.merge(user_cat_data,how='left',on='user_id')
    df_all = df_all.merge(prod_cat_data,how='left', on='product_id')
    df_all = df_all.merge(user_prod_cat_match_data,how='left',on=['user_id','product_id'])
    
    # reorder_rate processing
    # aisle reorder rate
    df_all=df_all.merge(pd.DataFrame(df_all.groupby('aisle_id').apply(lambda orders: \
                                    sum(orders.user_prod_no_of_orders-1) / float(sum(orders.user_prod_orders_since_first_ordered)) \
                                    if (sum(orders.user_prod_orders_since_first_ordered)) > 0 else 0.0),columns=['aisle_reorder_rate']),\
                                    how='left', left_on='aisle_id', right_index=True)
    assert(df_all.groupby('aisle_id').aisle_reorder_rate.apply(lambda x: x.unique().shape[0]==1).sum() == df_all.aisle_id.unique().shape[0])
    
    # department reorder rate
    df_all=df_all.merge(pd.DataFrame(df_all.groupby('department_id').apply(lambda orders: \
                                    sum(orders.user_prod_no_of_orders-1) / float(sum(orders.user_prod_orders_since_first_ordered)) \
                                    if (sum(orders.user_prod_orders_since_first_ordered)) > 0 else 0.0),columns=['department_reorder_rate']),\
                                    how='left', left_on='department_id', right_index=True)
    assert(df_all.groupby('department_id').department_reorder_rate.apply(lambda x: x.unique().shape[0]==1).sum() ==\
                                                                         df_all.department_id.unique().shape[0])
    
    nt_list=[20, 50, 100]
    for num_topics in nt_list:
        # prod_cat_reorder_rate
        prod_cat_str = 'prod_cat_'+str(num_topics)    
        df_all = df_all.merge(pd.DataFrame(df_all.groupby(prod_cat_str).apply(lambda orders: \
                                           sum(orders.user_prod_no_of_orders-1) / float(sum(orders.user_prod_orders_since_first_ordered)) \
                                           if (sum(orders.user_prod_orders_since_first_ordered)) > 0 else 0.0),columns=[prod_cat_str+'_reorder_rate']),\
                                           how='left', left_on=prod_cat_str, right_index=True)
        assert(df_all.groupby(prod_cat_str)[prod_cat_str+'_reorder_rate'].apply(lambda x: x.unique().shape[0]==1).sum() \
               == df_all[prod_cat_str].unique().shape[0])
        
        # user_cat_prod_reorder_rate
        # for a given product, the reorder rate of all users who belongs to a particle category
        user_cat_str = 'user_cat_'+str(num_topics)
        df_all=df_all.merge(pd.DataFrame(df_all.groupby([user_cat_str, 'product_id']).apply(lambda orders: \
                                         sum(orders.user_prod_no_of_orders-1) / float(sum(orders.user_prod_orders_since_first_ordered)) \
                                         if (sum(orders.user_prod_orders_since_first_ordered)) > 0 else 0.0), columns=[user_cat_str+'_prod_reorder_rate']),\
                                         how='left', left_on=[user_cat_str,'product_id'],right_index=True)
        assert(df_all.groupby([user_cat_str,'product_id'])[user_cat_str+'_prod_reorder_rate'].apply(lambda x: x.unique().shape[0]==1).sum() \
               == len(df_all.groupby([user_cat_str,'product_id']).groups.keys()))
        
        # user_cat_prod_cat_reorder_rate
        df_all=df_all.merge(pd.DataFrame(df_all.groupby([user_cat_str, prod_cat_str]).apply(lambda orders: \
                        sum(orders.user_prod_no_of_orders-1) / float(sum(orders.user_prod_orders_since_first_ordered)) \
                        if (sum(orders.user_prod_orders_since_first_ordered)) > 0 else 0.0), columns=[user_cat_str+'_'+prod_cat_str+'_reorder_rate']), \
                        how='left', left_on=[user_cat_str, prod_cat_str], right_index=True)
        assert(df_all.groupby([user_cat_str,prod_cat_str])[user_cat_str+'_'+prod_cat_str+'_reorder_rate'].apply(lambda x: x.unique().shape[0]==1).sum() \
               == len(df_all.groupby([user_cat_str,prod_cat_str]).groups.keys()))
    
    # set dtypes
    category_cols = ['eval_set', 'prod_cat_20','prod_cat_50','prod_cat_100',
                     'user_cat_20','user_cat_50','user_cat_100']
    for col in category_cols:
        df_all[col]=df_all[col].astype('category')
    
    rate_cols = ['aisle_reorder_rate','department_reorder_rate',
                 'prod_cat_20_reorder_rate', 'user_cat_20_prod_reorder_rate',
                 'user_cat_20_prod_cat_20_reorder_rate', 'prod_cat_50_reorder_rate',
                 'user_cat_50_prod_reorder_rate',
                 'user_cat_50_prod_cat_50_reorder_rate', 'prod_cat_100_reorder_rate',
                 'user_cat_100_prod_reorder_rate',
                 'user_cat_100_prod_cat_100_reorder_rate']
    for col in rate_cols:
        df_all[col]=df_all[col].astype('float32')
    
    print df_all.shape
    # use float32 to save
    # split the train and test for df_all
    # df_train=df_all[df_all.eval_set=='train'].drop(['add_to_cart_order'],axis=1)
    #  df_test=(df_all[df_all.eval_set=='test']).drop(['add_to_cart_order','reordered'],axis=1)
    df_train=df_all[df_all.eval_set=='train']
    df_train['reordered']=df_train['reordered'].astype(np.uint8)
    df_test=(df_all[df_all.eval_set=='test']).drop(['reordered'],axis=1)
    
    print df_train.shape
    print df_test.shape
    
    # save the df_train, df_test
    print "save the processed data"
    common.save_df(df_train,"../data/", train_data_name, index=False)
    common.save_df(df_test,"../data/", test_data_name, index=False)

# given some dataframe, fetch the cat data from file, then merge and return it
# not save it 
def get_extra_cat_data(df_data):

    print df_data.shape
    print df_data.keys()

    # load user and product category data
    user_cat_data, prod_cat_data, user_prod_cat_match_data = get_user_prod_cat_data([150,300])
    
    # merge category data
    df_data = df_data.merge(user_cat_data,how='left',on='user_id')
    df_data = df_data.merge(prod_cat_data,how='left', on='product_id')
    df_data = df_data.merge(user_prod_cat_match_data,how='left',on=['user_id','product_id'])
    
    return df_data


def pnone_processing(path="../data/", train_data_name = "train_data", test_data_name = "test_data"):
    # load the data
    df_train = common.load_df(path,train_data_name)
    df_test = common.load_df(path,test_data_name)
    
    print df_train.shape
    print df_test.shape
    
    # user prod add_to_cart_order mean, std
    priors, train, orders, products, aisles, departments = preprocessing_data()
    
    df_user_pnone=pd.DataFrame(priors.groupby('user_id').\
        apply(lambda user_orders: sum(user_orders.groupby('order_id').reordered.sum() == 0)\
        / float(user_orders.order_id.unique().shape[0])),columns=['pnone'])
    
    df_train = df_train.merge(df_user_pnone,\
                          how='left', left_on=['user_id'], right_index=True)
    df_test = df_test.merge(df_user_pnone,\
                          how='left', left_on=['user_id'], right_index=True)
    
    
    # save the df_train, df_test
    print "save the processed data"
    common.save_df(df_train,"../data/", train_data_name, index=False)
    common.save_df(df_test,"../data/", test_data_name, index=False)


if __name__ == '__main__':
    df_train, df_test = get_processed_data()
    extra_reorder_rate_processing()
    pnone_processing()
    
    # debug for extra cat data
    # load the data
    df_train = common.load_df('../data/','train_data')
    df_test = common.load_df('../data/','test_data')

    df_train = get_extra_cat_data(df_train)
    df_test = get_extra_cat_data(df_test)
