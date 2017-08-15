# -*- coding: utf-8 -*-
"""
This file is to build user products corpus

@author: eyulush
"""
import pandas as pd
import common
import sys

def get_user_corpus(users_orders):
    # get grouped count, groupby by user_id, product_id
    grouped_cnt = users_orders.groupby(['user_id','product_id']).apply(len)
    # generate dictionary, grouped_cnt has 2 levels index.    
    user_corpus = {uid : zip(grouped_cnt.loc[uid].index, grouped_cnt.loc[uid]) for uid in grouped_cnt.index.levels[0]}
    # convert to dataframe [user_id, user_corpus]    
    df_user_corpus = pd.DataFrame(data={'user_id' : user_corpus.keys(),'user_corpus' : user_corpus.values()})
    return df_user_corpus

if __name__ == "__main__":
    print 'Number of arguments:', len(sys.argv), 'arguments.'
    print 'Argument List:', str(sys.argv)
    
    if len(sys.argv) >= 2:
        objects = sys.argv[1:len(sys.argv)+1]
    else:
        objects = ['priors','train']
    
    if __debug__:
        print "In debug mode, None-debug mode command : python -O " + __file__ + "\n\n"

        # the following is self test code for 10 users
        # load the data
        ten_user_orders = pd.read_csv("/tmp/ten_user_orders.csv.gz", compression='gzip')
        
        # test for get_user_corpus
        ten_user_corpus = get_user_corpus(ten_user_orders)  
        
        # check get_user_corpus result
        print "compare user_corpus"
        print ten_user_corpus[ten_user_corpus.user_id==202277].user_corpus.apply(lambda row: list(zip(*row)[0]))
        print ten_user_orders[ten_user_orders.user_id==202277].product_id.sort_values().tolist()
    
        # save corpus
        common.save_df(ten_user_corpus,"/tmp/", "ten_user_corpus", index=False)
        # load corpus back    
        load_ten_user_corpus = common.load_df("/tmp/", "ten_user_corpus")
        print load_ten_user_corpus
    else:
        ### formal code
        IDIR = "../input/"
        priors, train, orders, products, aisles, departments = common.load_raw_data(IDIR)
    
    	# only build corpus for priors which is used for cross-validation
        print('add order info to priors')
        orders = orders.set_index('order_id', inplace=True, drop=False)
        priors = priors.join(orders, on='order_id', rsuffix='_')
        priors.drop('order_id_', inplace=True, axis=1)
        
        if "priors" in objects:
    	    priors_user_corpus = get_user_corpus(priors)
            print "compare user_corpus"
            print priors_user_corpus[priors_user_corpus.user_id==202277].user_corpus.apply(lambda row: list(zip(*row)[0]))
            print priors[priors.user_id==202277].product_id.sort_values().tolist()
    
    	    common.save_df(priors_user_corpus, "../data/", "priors_user_corpus", index=False)
            
        # build corpus for priors and train which is used for final test
        print('add order info to trains')
        orders = orders.set_index('order_id', inplace=True, drop=False)
        train = train.join(orders, on='order_id', rsuffix='_')
        train.drop('order_id_', inplace=True, axis=1)
        
    	# concate priors and train order data
        train = pd.concat([priors, train])
        
        if "train" in objects:
            train_user_corpus = get_user_corpus(train)
            print "compare user_corpus"
            print train_user_corpus[train_user_corpus.user_id==202277].user_corpus.apply(lambda row: list(zip(*row)[0]))
            print train[train.user_id==202277].product_id.sort_values().tolist()
            common.save_df(train_user_corpus, "../data/", "train_user_corpus", index=False)
