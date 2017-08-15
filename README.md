## Directory structure
src:
    python source directories
    
    common.py : common support functions 
    timeline_processing.py : build the timeline data 
   
    # topic model processing
    lda_corpus.py : use the history data to build corpus, concatenate history to 
    lda_cat.py : identify categories by using topic model. gensim needed.
    lda_processing.py : posting processing


    data_processing.py : the general data processing file. which do statistics on different dataset
    F1Optimizer.py : Faron's script for optimatize the expectation for F1 measure. This is very important and useful. Generally it is a discrete optimization
    f1_optimal.py : another f1 optimazer, which is in sh1ng's solution
    
    pNoneEstimator.py : use an independent lightgbm to estimate the pNone. the estimation result will be feeded into F1Optimizer as probability of None. 
                        this is another very important part

    lgbm_cv.py : use lightgbm model to do cross validation for parameter tunning
    lgbm_submission.py : use lightgbm model with tunned parameter to generate result

    utils.py : utils functions, which sh1ng's solution
    training_data_processing.py : combine the intermediate data which generated from sh1ng's solution
    change_pickle.py : transfer the sh1ng's intermediate data which is based on word2vec model

input:
    the raw data directory

data:
    the directory for saving intermedia data

logs:
    the directory for saving logs

output:
    the directory for saving intermedia outputs. e.g. cross validation result

submit:
    final result directory


imba:
    sh1ng's solution. which is also public on the github just before 1 week of deadline


## Final quick summary

In general, I gain a lot from this instacart competition. 

1. The starting script which use 'order, product' and 'reordered' as binary classification is a feasible approach. 
2. Since evaluation is based on order f1 score. one order will have multiple 'order_id, product_id' pair. and data is imbalanced,
   so, based on binary classification result (which are probabilites) we need to make decision to choose which product will be reordered
   in the future order. 
   There is an simple approach, setting a threshold to make decision. and starting script use 0.2 as example
3. the first gain is the approach from starting script. the second gain is my personal thinking later. when doing binary classification
   the gdbt will output probability for each input. the total sum of probability will be the number of positive samples which gdbt believe
   for example, 75000 'order, product' pair as input. then output probability sum e.g. 8900. i.e. gdbt believe there will be 8900 reorder
   will happen. then basically it mean we should choose 8900 results as positive output. then simple to do that is sort the output probability descend
   then choose 8900th value of probability as optimal threshold.
   I did not use this. Since I already use Faron's F1 Optimizer.
4. At the first glance, F1 Optimizer sounds just to optimize the score, not focus on improving accurancy. but in fact, in real application, it is quite 
   common to use this discrete optimizer technic. the model first estimate the probabilites and then more or less we need to use discrete optimizer technic
   to optimize for some purpose. Not just take the result from model learning model output. in this area, there could be linear, dynamic, and constraint 
   programming will be used.
5. I used LDA model for user and product category. I think it is kind of useful. Other people also tried LDA and NMF, or word2vec model.
6. I identified an important feature. the product reorder in order interval probability. use exponent distribution to fit for a product. input is how many 
   orders passed since a user not reorder that product. it is useful
7. Use an independent model for 'None' prediction then feed into F1 optimizer. it is big improvement. It give me the silver medal. without it I am on edge of 
   bronze medal


Lesson
1. the understanding of reorder rate for a user and product. we should build the timeline of user orders and correct calculate for reorder-rate 
   which is very important feature 
2. After F1 optimization, the output could be mixed 'None' and products. which is a big mistake I made until I see Sh1ng's code
3. Some people mention Faron's script not consider the correlation between products. I think it is reasonable. it is worth to try consider correlation as constraint
   to do discrete optimization. No time to try it. 
  

## Good Luck for next time
