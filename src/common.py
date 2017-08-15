# Author lucius yu
# provide support function

import numpy as np
import pandas as pd
import os
import json
import functools

from multiprocessing import Pool
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from itertools import izip_longest
from ast import literal_eval

def grouper(n, iterable, fillvalue=None):
    # "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

def get_logger(name,log_filename):
    import logging

    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)-8s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=log_filename,
                    filemode='a')

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    logger = logging.getLogger(name)
    logger.addHandler(ch)

    return logger

def logging_dict(logger,xdict, name=""):
    logger.debug(name + '\n' + json.dumps(xdict,indent=4, separators=(',', ': ')))

def save_dict(xdict,filename):
    with open(filename, 'w+') as outfile:
        json.dump(xdict, outfile, indent=4, separators=(',', ': '))

def load_dict(filename):
    if os.path.isfile(filename):
        with open(filename) as json_data:
            return json.load(json_data)      
    else:
        return list()
        
def save_df(df, path, name, **kwargs ):
    df.dtypes.to_pickle(path + name + '.dtype')
    df.to_csv(path + name + '.csv.gz', compression='gzip', **kwargs)

def load_df(path, name, no_converter=False, **kwargs):
    obj_dtype = pd.read_pickle(path + name + ".dtype")
    
    # build up converters
    converters = dict()
    if no_converter==False:
        for k in obj_dtype.keys():
            # use getatter, when no hasobject method, we get false
            if getattr(obj_dtype[k], "hasobject",False):
                converters[k] = literal_eval
    # update int key to str key
    obj_dtype_dict = obj_dtype.to_dict()
    for k in obj_dtype_dict.keys():
        # change number key to str
        if type(k)==int:
            obj_dtype_dict[str(k)]=obj_dtype_dict.pop(k)
            
    # if converter in args, then update converter with parameters
    if 'converters' in kwargs.keys():
        converters.update(kwargs['converters'])
        
    # update parameters
    if len(converters.keys()) > 0:
        kwargs['converters'] = converters
    return pd.read_csv(path + name + '.csv.gz', compression='gzip', dtype=obj_dtype_dict, **kwargs)

def cache_df(path, name, *args,**kwargs):
    pass

# this file will load the raw data to dataframe
def load_raw_data(IDIR="../input/"):
	print('loading prior')
	priors = pd.read_csv(IDIR + 'order_products__prior.csv.gz', compression='gzip', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

	print('loading train')
	train = pd.read_csv(IDIR + 'order_products__train.csv.gz', compression='gzip', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

	print('loading orders')
	orders = pd.read_csv(IDIR + 'orders.csv.gz', compression='gzip', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

	print('loading products')
	products = pd.read_csv(IDIR + 'products.csv.gz', compression='gzip', dtype={
        'product_id': np.uint16,
        'order_id': np.int32,
        'aisle_id': np.uint8,
        'department_id': np.uint8},
        usecols=['product_id', 'aisle_id', 'department_id'])


	print('loading aisles')
	aisles = pd.read_csv(IDIR + 'aisles.csv.gz', compression='gzip', dtype={
        'aisle_id': np.uint8,
        'aisle': str},
        index_col=['aisle_id'],
        usecols=['aisle_id', 'aisle'])

	print('loading department')
	departments = pd.read_csv(IDIR + 'departments.csv.gz', compression='gzip', dtype={
        'department_id': np.uint8,
        'department': str},
        index_col=['department_id'],
        usecols=['department_id', 'department'])

	return priors, train, orders, products, aisles, departments

def file_exists(path,filename):
	return os.path.isfile(path+filename) 

'''
papply function is to use multiprocessing for paralle processing
input
    df_list is a list of dataframes
    fxn is the function working on each dataframe in df_list
    fxn should always reture pd.Series or pd.DataFrame
return
    dataframe
    
example of fxn

def foo(record):
    return pd.Series({'mean' :np.mean(record.Z)})
'''
def papply(df_list, fxn, pmax=8):
    
    papply_pool = Pool(min(len(df_list), pmax))
    result = papply_pool.map(fxn, df_list)
    df_result = pd.concat(result)
    
    # clean up
    papply_pool.close()
    del papply_pool
    
    return df_result

'''
function for mean group f1 score
'''
def get_f1_score(data,threshold=0.21):
    
    # if true is None and predict is None, return 1.
    # else true is None and predict is not None, return 0.
    # else true is not None and predict is None, return 0.
    # else f1_score
    return data.groupby('group').apply(lambda x: \
                       1.0 if x.label.sum()==0 and sum(x.pred>=threshold)==0 else\
                       0.0 if x.label.sum()==0 and sum(x.pred>=threshold)>0 else\
                       0.0 if x.label.sum()>0 and sum(x.pred>=threshold)==0 else\
                       f1_score(x.label.tolist(), x.pred>=threshold))
    
def lgb_mean_group_f1_score(preds, dtrain):
    data = pd.DataFrame()
    data['label']=dtrain.get_label()
    data['pred']=preds
    data['group']=dtrain.get_group()
    
    group_size = 10000
    groups = data.group.unique().tolist()
    
    scores = papply([data[data.group.isin(gids)] for gids in grouper(group_size, groups)],get_f1_score, pmax=5)
    return 'mean f1-score', scores.mean(), True

# not tested yet.

class MeanGroupF1Eval:
    # groups, [n_samples], e.g. [0,0,1,1] means 4 samples 2 groups
    def __init__(self,groups,pgroup_size=10000, threshold=0.21):
        self.groups = groups
        self.threshold = threshold
        self.pgroup_size = pgroup_size
        
    # calculate f1 score for samples in same group    
    # then calculate the mean value of groups
    def evaluate(self, preds, valid_data):
        data = pd.DataFrame()
        data['label']=valid_data.get_label()
        data['pred']=preds
        data['group']=self.groups
        
        groups = data['group'].unique().tolist()
        scores = papply([data[data.group.isin(gids)] for gids in grouper(self.pgroup_size, groups)],\
                              functools.partial(get_f1_score,threshold=self.threshold))
        return 'mean f1-score', scores.mean(), True

def split_groups(data, labels, groups, n_splits=3):
    group_kfold = GroupKFold(n_splits)
    for train_index, test_index in group_kfold.split(data, labels, groups):
        yield train_index, test_index, groups[train_index], groups[test_index]

def save_model(model,params,best_score=None, best_iteration=0,model_path='../mdl/',notes=''):
    if file_exists(model_path,'models_info.csv'):
        models_info = pd.read_csv(model_path+'models_info.csv')
        model_id = models_info.id.max()+1
    else:
        models_info = pd.DataFrame(columns=['id','training_score','best_iteration','params','model_file'])
        model_id = 1
    
    models_info['id']=models_info['id'].astype(np.int16)
    models_info['best_iteration']=models_info['best_iteration'].astype(np.int16)
    
    model_info=pd.Series()
    model_info['id'] = model_id
    if best_score:
        model_info['training_score'] = best_score
        model_info['best_iteration'] = best_iteration
    else:
        if 'training' in model.best_score.keys():
            model_info['training_score'] = model.best_score['training'].values()[0]
        else :
            model_info['training_score'] = 0.0
        model_info['best_iteration'] = model.best_iteration
    
    model_info['params'] = json.dumps(params)
    model_info['model_file'] = "model_"+str(model_id)+'.mdl'
    model_info['notes'] = notes
    
    model.save_model(model_path+model_info['model_file'])
    
    models_info=models_info.append(model_info,ignore_index=True)
    models_info.to_csv(model_path+'models_info.csv', index=False)

from sys import getsizeof
def get_var_mems():
    global_var_mems=[(k,getsizeof(v)/1048000.0) for k,v in zip(globals().keys(),globals().values())]
    local_var_mems=[(k,getsizeof(v)/1048000.0) for k,v in zip(locals().keys(),locals().values())]
    return global_var_mems, local_var_mems, sum([v for k,v in global_var_mems]), sum([v for k,v in local_var_mems])

def load_lgb_model(model_path='../mdl/', model_id=None):
    if not file_exists(model_path,'models_info.csv'):
        print "No models_info.csv found"
        return None, None
    
    # find the best
    models_info = pd.read_csv(model_path+'models_info.csv')
    
    if model_id==None:
        selected_model_info = models_info.loc[models_info['training_score'].idxmax()]
    else:
        selected_model_info = models_info[models_info.id==model_id]
        
    print(selected_model_info)
    
    import lightgbm as lgb
    model = lgb.Booster(model_file=model_path+selected_model_info['model_file'].iloc[0])
    
    return model, json.loads(selected_model_info['params'].iloc[0])

def process_test(test_order, threshold=0.5):
    products=test_order[test_order.preds>=threshold].product_id.tolist()
    if len(products) == 0:
        return 'None'
    else:
        return ' '.join(map(str,products))

# output format, 
# columns 'order_id', 'product'
#         12334, 112 11 345
def mk_submission(filename, df_test, preds=None, threshold=0.21):
    if preds:
        df_test['preds'] = preds
    print threshold
    result=df_test.groupby('order_id').apply(functools.partial(process_test, threshold=threshold))
    # from pd Series to DataFrame
    df_result = pd.DataFrame(result,columns=['products'])
    df_result.to_csv(filename)

def process_none_test(test_order, threshold=0.5):
    products=test_order[test_order.preds>=threshold].product_id.tolist()
    if len(products) == 0:
        return 'None'
    else:
        # replace none product 
        return ' '.join(map(lambda prod: 'None' if prod==none_product else str(prod) ,products))

def mk_none_submission(filename, df_test, preds=None):
    if preds:
        df_test['preds'] = preds
    
    result=df_test.groupby('order_id').apply(functools.partial(process_none_test))
    # from pd Series to DataFrame
    df_result = pd.DataFrame(result,columns=['products'])
    df_result.to_csv(filename)
    
# def f1 optimize function
# input: grouped data frame
# ouput: data frame with columns product_id, pred (hard decision, 0 or 1)
from F1Optimizer import F1Optimizer

# input  columns ['product_id', 'reordered', 'preds']
# output columns ['product_id', 'reordered', 'bst_preds]
# 2017-08-13 : update with support pnone est
none_product = 50000
def get_best_prediction(df_preds):
    df_preds = df_preds.sort_values(by='preds', axis=0, ascending=False)
    P = df_preds.preds.values
    
    pNone = df_preds.pnone_pred.values[0] if 'pnone_pred' in df_preds.keys() else None
    if pNone is None:
        pNone = (1.0 - P).prod()

    opt = F1Optimizer.maximize_expectation(P, pNone)
    
    df_best_preds = df_preds.copy()
    df_best_preds['bst_preds'] = np.concatenate([np.ones(opt[0]),np.zeros(df_preds.shape[0]-opt[0])])
    
    # for none-product
    if 'reordered' in df_best_preds.keys():
        none_reordered = 1 if df_best_preds.reordered.sum() == 0 else 0
    else:
        none_reordered = 0

    df_best_preds=df_best_preds.append({'order_id' : df_preds.order_id.values[0], \
                                        'product_id' : none_product, 'preds': pNone,\
                                        'reordered': none_reordered,\
                                        'bst_preds' : 1 if opt[1] else 0}, ignore_index=True)        
    return df_best_preds

def group_best_prediction(df_data, pNone=None):
    df_data['group'] = df_data['order_id']
    
    result = df_data.groupby('group').apply(get_best_prediction)
    result.reset_index(drop=True, inplace=True)
    result.drop('group',axis=1,inplace=True)
    return result

def pget_best_prediction(data):
    groups = data['order_id'].unique().tolist()
    pgroup_size = len(groups)/8+1
    result=papply([data[data.order_id.isin(gids)] for gids in grouper(pgroup_size, groups)],\
                        functools.partial(group_best_prediction,pNone=None))
    return result

def f1_evaluate(data, threshold):
    groups = data['group'].unique().tolist()
    pgroup_size = len(groups)/8+1
    scores = papply([data[data.group.isin(gids)] for gids in grouper(pgroup_size, groups)],\
                          functools.partial(get_f1_score,threshold=threshold))
    print scores.mean()
    return scores

def get_f1_none_score(data):
    # 2 version looks same, the 1st one fast
    return data.groupby('group').apply(lambda x: f1_score(x.label.tolist(), x.pred.tolist()))
    # return data.groupby('group').apply(lambda x: \
    #                   f1_score(x[x.product_id!=50000].label.tolist(), x[x.product_id!=50000].pred.tolist()) \
    #                       if (x[x.product_id==50000].label.values[0]==0 and x[x.product_id==50000].pred.values[0]==0) \
    #                       else  f1_score(x.label.tolist(), x.pred.tolist()) )

def f1_none_evaluate(data):
    groups = data['group'].unique().tolist()
    pgroup_size = len(groups)/8+1
    scores = papply([data[data.group.isin(gids)] for gids in grouper(pgroup_size, groups)],get_f1_none_score)
    print scores.mean()
    return scores

class TrainingCtrl:
    def __init__(self, init_learning_rate=0.1,decay=0.99, min_learning_rate=0.05, half_min_round=500):
        self.iter = 0
        self.learning_rate=init_learning_rate
        self.min_learning_rate=min_learning_rate
        self.decay = decay
        self.half_min_round = half_min_round
      
    def get_learning_rate(self,iter):
        self.iter+=1
        if self.iter % self.half_min_round == 0 and self.min_learning_rate > 0.005 :
            if self.learning_rate == self.min_learning_rate:
                self.learning_rate = self.min_learning_rate/2.0
            self.min_learning_rate = self.min_learning_rate/2.0
        self.learning_rate = max(self.min_learning_rate, self.learning_rate * self.decay )
        print self.learning_rate
        return self.learning_rate
    
'''
main function use to self test
'''	

if __name__ == '__main__':
	IDIR="../input/"
	priors, train, orders, products, aisles, departments  = load_raw_data(IDIR)

	print priors.shape
	print priors.head()
	print train.shape
	print train.head()
	print orders.shape
	print orders.head()
	print products.shape
	print products.head()
	print aisles.shape
	print aisles.head()
	print departments.shape
	print departments.head()

