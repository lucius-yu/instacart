#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 22:31:29 2017

@author: eyulush
"""

from gensim import corpora
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
    else: 
        user_corpus_name = "train_user_corpus"
        model_fn_prefix = "../models/train_up_"
        user_cat_name_prefix = "train_user_cat_"
        prod_cat_name_prefix = "train_prod_cat_"

    print user_corpus_name 
    print model_fn_prefix
    print user_cat_name_prefix
    print prod_cat_name_prefix

    # load and build user products
    # products correspond to corpus
    df_user_prods = common.load_df("../data/", user_corpus_name, converters={"user_corpus": literal_eval})
    df_user_prods = df_user_prods.set_index('user_id', drop=False)
    user_prods = list(df_user_prods.user_corpus)
    
    # build lda model
    if __debug__:
        print "In debug mode, None-debug mode command : python -O " + __file__ + "\n\n"
        print "Processing 200 users, 10 topics"
        up_lda = LdaMulticore(corpus=user_prods[0:1000], id2word = id2prod, workers=3, num_topics=20)
        up_lda.save('/tmp/up.lda')
        loaded_up_lda = LdaModel.load('/tmp/up.lda')
        loaded_up_lda.show_topics()
        loaded_up_lda.get_document_topics(user_prods[0])
        loaded_up_lda.get_term_topics(13176)
        
        # get user cats
        df_user_cat = get_user_cat(loaded_up_lda, df_user_prods.iloc[100:120])
    
        # get prod cats
        i = 3
        cat_prods = [x[0] for x in loaded_up_lda.get_topic_terms(i,topn=20)] 
        df_prod_cat = get_prod_cat(loaded_up_lda, products.loc[cat_prods])
        print df_prod_cat
    else:
        print "In None-debug mode \n\n"
    
        import time
        start_time = time.time()
        print time.ctime()
        
        num_topics_list = [150,300]
        for num_topics in num_topics_list:
            print "processing with num_topics = " + str(num_topics)
            # reduce the minimum_probability, otherwise, many product has not catetory assignment. the 1e-8 is model allowed min value
            # default batch is false, i.e. using online-train with batch size=2000
            # passes=1 ??
            
            up_lda = LdaMulticore(corpus=user_prods, id2word = id2prod, workers=2, num_topics=num_topics, 
                                  minimum_probability=1e-8, chunksize=2000, passes=1)
            
            model_fn = model_fn_prefix+str(num_topics)+".lda"
            print "save model into " + model_fn
            up_lda.save(model_fn)
            # debug reload model
            # up_lda = LdaModel.load(model_fn)
            
            df_user_cat = get_user_cat(up_lda, df_user_prods)
            user_cat_name = user_cat_name_prefix + str(num_topics)
            print "save "+user_cat_name
            common.save_df(df_user_cat, "../data/", user_cat_name, index=False)
    
            df_prod_cat = get_prod_cat(up_lda, products)
            prod_cat_name = prod_cat_name_prefix+str(num_topics)
            print "save "+prod_cat_name
            common.save_df(df_prod_cat, "../data/", prod_cat_name, index=False)
            
            print time.ctime()
            
        print "finished"
        print time.ctime()
        print time.time() - start_time
