#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:23:43 2017

@author: eyulush
"""
import numpy as np
import pandas as pd
import time
import common

from multiprocessing import Process, Pool
from ast import literal_eval

def get_timeline_data(filename="../data/user_product_timeline_result_1_206209.csv.gz"):
    # loading the data
    timeline_data=pd.read_csv(filename, compression='gzip', converters={"user_prod_order_number_list": literal_eval,\
                                                                        "user_prod_reorder_days_intervals": literal_eval,\
                                                                        "user_prod_reorder_nums_intervals": literal_eval})
    return timeline_data

# timeline processing for each product of given user
# return: user_prod_order_number_list: list of order_number in which given user has ordered given product
#         user_prod_orders_since_first_ordered: how many orders which given user has made after a given product first ordered.
#         user_prod_order_rate: the NO. of orders which user ordered given product divided by user_total_no_of_orders
#         user_prod_reorder_rate: the NO. of reorders for given product divided by user_prod_orders_since_first_ordered
#         user_prod_days_since_last_ordered: how many days passed since the last of user order for given product
#         user_prod_orders_since_last_ordered: how many orders user has made since the last of user order for given product
#         user_prod_reorder_days_intervals: 
#         user_prod_reorder_nums_intervals: for a given product and given user
def user_prod_timeline_processing(one_user_prod_orders, timeline):
    # get product which ordered in what order-numbers
    prod_order_numbers = one_user_prod_orders.sort_values('order_number').order_number
    
    user_prod_order_number_list=prod_order_numbers.values.tolist()
    user_total_no_of_orders = timeline.index[0]
    user_prod_first_order_number = prod_order_numbers.values[0]
    user_prod_orders_since_first_ordered = user_total_no_of_orders-user_prod_first_order_number
    
    return pd.Series({'user_prod_order_number_list': user_prod_order_number_list,\
                      'user_prod_no_of_orders': prod_order_numbers.shape[0],\
                      'user_prod_orders_since_first_ordered': user_prod_orders_since_first_ordered,\
                      'user_prod_order_rate': float(prod_order_numbers.shape[0])/user_total_no_of_orders,\
                      'user_prod_reorder_rate': float(prod_order_numbers.shape[0]-1)/user_prod_orders_since_first_ordered\
                          if user_prod_orders_since_first_ordered > 0 else 0.0,\
                      'user_prod_days_since_last_ordered': timeline.loc[prod_order_numbers].iloc[-1],\
                      'user_prod_orders_since_last_ordered': user_total_no_of_orders-prod_order_numbers.values[-1],\
                      'user_prod_reorder_days_intervals': timeline.loc[prod_order_numbers[::-1]].diff().dropna().values.tolist(),\
                      'user_prod_reorder_nums_intervals': prod_order_numbers.diff().dropna().values.tolist()
                      })
    
# timeline processing for each user
def user_timeline_processing(one_user_orders):
    # build the timeline for this user
    # 1. drop duplicates for order_number,
    # 2. descendingly sort by order_number
    # 3. set index as order_number
    # 4. cumsum on days since prior order. 
    # 5. shift and fillna as 0.
    # final result will be [order_number, days_since_last_order]
    timeline = one_user_orders.drop_duplicates('order_number')\
        .sort_values('order_number',ascending=False)\
        .set_index('order_number')\
        .days_since_prior_order.cumsum()\
        .shift().fillna(0.0)

    # group by products
    return one_user_orders.groupby('product_id').apply(user_prod_timeline_processing, timeline=timeline)


def timeline_processing(user_orders):
    result = user_orders.groupby('user_id').apply(user_timeline_processing)
    result.reset_index(inplace=True)
    return result 

from itertools import izip_longest
def grouper(n, iterable, fillvalue=None):
    # "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

if __name__ == '__main__':
    # loading the data
    IDIR = '../input/'

    priors, train, orders, products, aisles, departments  = common.load_raw_data(IDIR)

    print('priors {}: {}'.format(priors.shape, ', '.join(priors.columns)))
    print('orders {}: {}'.format(orders.shape, ', '.join(orders.columns)))
    print('train {}: {}'.format(train.shape, ', '.join(train.columns)))


    ### Preprocessing, combine order, product information into priors
    print('add order info to priors')
    orders.set_index('order_id', inplace=True, drop=False)
    priors = priors.join(orders, on='order_id', rsuffix='_')
    priors.drop('order_id_', inplace=True, axis=1)

    ### 
    print('add product info to priors')
    priors=pd.merge(priors,products,how='left',on='product_id')

    ### total user number 206209, 5 cpu worker, each group has size 42000
    user_ids = priors.user_id.sort_values().unique().tolist()
    print "total number of users %d"%len(user_ids)
    
    group_size = 42000
    print "each group has "+str(group_size)+" users"    
    
    debug_flag=False
    if debug_flag:
        # debug code
        user_ids = user_ids[0:10]
        priors=priors[priors.user_id.isin(user_ids)]
        group_size = 2
        # debug code end
    
    # time record
    start_time = time.time()
    
    pool = Pool(processes=5) 
    result = pool.map(timeline_processing,[priors[priors.user_id.isin(uids)] for uids in grouper(group_size, user_ids)])

    end_time = time.time()
    print "Cost : " + str(end_time-start_time) + " seconds"
    print "Cost : " + str((end_time-start_time)/3600) + " hours"

    result=pd.concat(result)
    filename = "user_product_timeline_result_" + str(user_ids[0]) + "_" + str(user_ids[-1]) + ".csv.gz"
    
    print "save result to file %s"%filename
    result.to_csv("../data/" + filename,compression='gzip', index=False)
