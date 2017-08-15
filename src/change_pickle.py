# other functions
import pickle
def change_pickle_protocol(filepath,protocol=2):
    with open(filepath,'rb') as f:
        obj = pickle.load(f)
    with open(filepath,'wb') as f:
        pickle.dump(obj,f,protocol=protocol)

change_pickle_protocol('../data/lgb_order_train' + ".dtype")
change_pickle_protocol('../data/lgb_order_test' + ".dtype")
