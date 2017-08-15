#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

This file is to identify the most likely category for each user and product.
input file is the output from lda_processing

This file also calculate user product category match value for all ordered product of any user

@author: eyulush
"""

from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from ast import literal_eval
import common
import pandas as pd
import numpy as np

def transform_to_vector(topic_list,num_topics=10):
    vec = np.zeros(num_topics)
    if len(topic_list) > 0:
        # use zip(*list(tuple)) to do unzip action
        vec[list(zip(*topic_list)[0])] = list(zip(*topic_list)[1])
    return vec.tolist()

# note, df_user_prod.index should be user_id
def get_user_cat(lda,df_user_prods):
    # get user cats
    user_cat = dict()
    for user_id in df_user_prods.index:
        user_cat[user_id] = transform_to_vector(lda.get_document_topics(df_user_prods.loc[user_id].user_corpus), lda.num_topics)
    # transform to dataframe
    df_user_cat = pd.DataFrame.from_dict(user_cat, orient='index')
    df_user_cat.columns=['user_cat_'+str(c) for c in df_user_cat.columns]
    df_user_cat['user_id'] = df_user_cat.index
    return df_user_cat

# note, df_prod.index should be product_id
def get_prod_cat(lda,df_prods):
    prod_cat = dict()
    for prod_id in df_prods.product_id:        
        prod_cat[prod_id] = transform_to_vector(lda.get_term_topics(prod_id), lda.num_topics)
        
    df_prod_cat = pd.DataFrame.from_dict(prod_cat, orient='index')
    df_prod_cat.columns=['prod_cat_'+str(c) for c in df_prod_cat.columns]
    df_prod_cat['prod_id'] = df_prod_cat.index
    return df_prod_cat

def get_user_prod_match(df_user_cat, df_prod_cat, df_user_prods):
    user_prods_match = list()
    user_prods = df_user_prods.user_corpus.apply(lambda row: list(zip(*row)[0]))
    
    for user_id in user_prods.index:
        user_cat_list = df_user_cat.loc[user_id]['user_cat_list'][0]
        for prod_id in user_prods.loc[user_id]:
            prod_cat_list = df_prod_cat.loc[prod_id]['prod_cat_list'][0]
            user_prods_match.append({'user_id' : user_id,
                                     'prod_id' : prod_id,
                                     'user_prod_match' : np.dot(user_cat_list, prod_cat_list)})
    return pd.DataFrame(user_prods_match).set_index('user_id',drop=False)

if __name__ == "__main__":

    # load the raw data
    IDIR = "../input/"
    products = pd.read_csv(IDIR+'products.csv.gz', compression='gzip')
    products = products.set_index('product_id',drop=False)
        
    # build id to product_name dictionary
    id2prod = products[['product_id','product_name']].to_dict(orient='dict')['product_name']

	# set name for different data set
    object = 'priors'
    if object == 'priors':
        user_corpus_name = "priors_user_corpus"
        model_fn_prefix = "../models/prior_up_"
        user_cat_name_prefix = "prior_user_cat_"
        prod_cat_name_prefix = "prior_prod_cat_"
        user_prod_match_name_prefix = "prior_user_prod_match_"
    else: 
        user_corpus_name = "train_user_corpus"
        model_fn_prefix = "../models/train_up_"
        user_cat_name_prefix = "train_user_cat_"
        prod_cat_name_prefix = "train_prod_cat_"
        user_prod_match_name_prefix = "train_user_prod_match_"


    print user_corpus_name 
    print model_fn_prefix
    print user_cat_name_prefix
    print prod_cat_name_prefix

    # load and build user products
    # products correspond to corpus
    df_user_prods = common.load_df("../data/", user_corpus_name, converters={"user_corpus": literal_eval})
    df_user_prods = df_user_prods.set_index('user_id', drop=False)
    
    
    # build lda model
    debug_flag=False
    if debug_flag:
        num_topics=20
        print "In debug mode, None-debug mode command : python -O " + __file__ + "\n\n"
        print "Processing 200 users, 10 topics"
        user_prods_corpus = list(df_user_prods.user_corpus)
        up_lda = LdaMulticore(corpus=user_prods_corpus[0:1000], id2word = id2prod, workers=3, num_topics=num_topics)
        up_lda.save('/tmp/up.lda')
        loaded_up_lda = LdaModel.load('/tmp/up.lda')
        loaded_up_lda.show_topics()
        loaded_up_lda.get_document_topics(user_prods_corpus[0])
        loaded_up_lda.get_term_topics(13176)
        
        # get user cats
        df_user_cat = get_user_cat(loaded_up_lda, df_user_prods.iloc[100:120])
        user_cat_cols = ['user_cat_' + str(c) for c in range(num_topics)]
        df_user_cat['user_cat'] = df_user_cat[user_cat_cols].apply(lambda row: np.argmax(list(row)),axis=1)
        df_user_cat['user_cat_list'] = df_user_cat[user_cat_cols].apply(lambda row: [row.tolist()],axis=1)
        # get prod cats
        i = 3
        cat_prods = [x[0] for x in loaded_up_lda.get_topic_terms(i,topn=20)] 
        df_prod_cat = get_prod_cat(loaded_up_lda, products.loc[cat_prods])
        prod_cat_cols = ['prod_cat_' + str(c) for c in range(num_topics)]
        df_prod_cat['prod_cat'] = df_prod_cat[prod_cat_cols].apply(lambda row: np.argmax(list(row)),axis=1)        
        df_prod_cat['prod_cat_list'] = df_prod_cat[prod_cat_cols].apply(lambda row: [row.tolist()],axis=1)

        print df_prod_cat
        
        # find the user prod match
    else:
        print "In None-debug mode \n\n"
    
        import time
        start_time = time.time()
        print time.ctime()
        
        num_topics_list = [150,300]
        
        # processing
        for num_topics in num_topics_list:
            # load user cat
            user_cat_name = user_cat_name_prefix + str(num_topics)
            df_user_cat = common.load_df("../data/",user_cat_name)
            df_user_cat = df_user_cat.set_index('user_id',drop=False)
            # load prod cat
            prod_cat_name = prod_cat_name_prefix+str(num_topics)
            df_prod_cat = common.load_df("../data/",prod_cat_name)
            df_prod_cat = df_prod_cat.set_index('prod_id',drop=False)
            
            # find user cat
            user_cat_cols = ['user_cat_' + str(c) for c in range(num_topics)]
            df_user_cat['user_cat'] = df_user_cat[user_cat_cols].apply(lambda row: np.argmax(list(row)),axis=1)
            df_user_cat['user_cat_list'] = df_user_cat[user_cat_cols].apply(lambda row: [row.tolist()],axis=1)
            
            print "the first 20 users max cat value"
            print df_user_cat.loc[0:20][user_cat_cols].apply(lambda row: np.max(list(row)),axis=1)
            print "the first 20 users cats"
            print df_user_cat.loc[0:20]['user_cat']
            print "user cats distribution"
            print df_user_cat[['user_id','user_cat']].groupby('user_cat').size()
            
            # find product cat
            prod_cat_cols = ['prod_cat_' + str(c) for c in range(num_topics)]
            df_prod_cat['prod_cat'] = df_prod_cat[prod_cat_cols].apply(lambda row: np.argmax(list(row)),axis=1)
            df_prod_cat['prod_cat_list'] = df_prod_cat[prod_cat_cols].apply(lambda row: [row.tolist()],axis=1)
            
            print "the first 20 products max cat value"
            print df_prod_cat.loc[0:20][prod_cat_cols].apply(lambda row: np.max(list(row)),axis=1)
            print "the first 20 products cats"
            print df_prod_cat.loc[0:20]['prod_cat']
            print "prod cats distribution"
            print df_prod_cat[['prod_id','prod_cat']].groupby('prod_cat').size()
            print df_prod_cat[['prod_id','prod_cat']].groupby('prod_cat').size().sum()
            
            # calculate user product match
            # debug show some users' products ordered times vs user-product match value
            debug_match = get_user_prod_match(df_user_cat, df_prod_cat, df_user_prods.loc[20:25])
            for i in range(20,25):
                print df_user_prods.loc[i].user_corpus
                print debug_match[debug_match.user_id==i]

            
            df_user_prod_match = get_user_prod_match(df_user_cat, df_prod_cat, df_user_prods)
            
            print "save "+user_cat_name
            common.save_df(df_user_cat, "../data/", user_cat_name, index=False)

            print "save "+prod_cat_name
            common.save_df(df_prod_cat, "../data/", prod_cat_name, index=False)
            
            user_prod_match_name = user_prod_match_name_prefix+str(num_topics)
            print "save "+user_prod_match_name
            common.save_df(df_user_prod_match, "../data/", user_prod_match_name,index=False)
            
            print df_user_cat.user_cat.unique().tolist()
            print df_prod_cat.prod_cat.unique().tolist()
            print time.ctime()
            
        print "finished"    
        print time.ctime()
        print time.time() - start_time
