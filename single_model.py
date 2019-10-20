import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error
import datetime
from single import *
import pickle
from sklearn.metrics import accuracy_score
import math
from scipy.stats import zscore
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

from single_feature import *
from basic import *
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout

import itertools

from sklearn.linear_model import Lasso,Ridge,ElasticNet

class nn_preprocess:
    def __init__(self):
        self.steps = [
            ("parse_area",parse_area_size()),
            ("parse_room",parse_rooms()),
            ("parse_old",parse_how_old()),
            ("height_enc",height_encoder()),
            ("ex_dist",extract_district()),
            ("label_dist",district_encoder()),
            ("m_d_p",add_mean_dist_price()),
            ("dist_price_per_area",dist_and_price_per_area()),
            ("dist_oh",district_onehot()),

            ("acc_ext",access_extractor()),
            ("tr_enc",train_encoder()),

            ("mean_walk",add_mean_walk_price()),
            ("mean_moyori",add_moyori_walk_price()),

            ("parking_encoder",parking_encoder()),
            ("dir_enc",direction_encoder()),
            ("angle_stat",add_mean_angle_price()),
            ("dir_oh",direction_onehot()),

            ("info_enc",info_encoder()),
            ("p_con_time",parse_contract_time()),
            ("fac",fac_encoder()),
            ("bath",bath_encoder()),
            ("kit",kitchin_encoder()),
            ("env",env_encoder()),

            ("structure_enc",structure_label_encoder()),
            ("mean_struct",add_mean_structure_price()),
            ("st_onehot",structure_onehot()),

            ("moyori_drop",moyori_drop()),
            ("drop_unnecessary",drop_unnecessary()),
            ("cross",cross_features()),
            ("dist2main_st",dist_to_main_station()),
            ("short_main_st",shortest2main_st()),

            # ("actual_height",actual_height()),
            # ("middle_class",middle_class_centers()),
            # ("high_class",height_class_center()),

        ]

class ridge_model:
    def __init__(self):
        rp = nn_preprocess()
        rpstep = rp.steps
        pp_steps = [
            ("pre",Pipeline(steps=rpstep)),
            ("dummy",dummy())
        ]
        self.preprocess = Pipeline(steps=pp_steps)
        self.model = Ridge()
    def fit(self,x,y):
        self.preprocess.fit(x,y)
        data = self.preprocess.predict(x)
        data = scale(data)
        self.model.fit(data,y)
        return self
    def predict(self,x):
        data = self.preprocess.predict(x)
        data = scale(data)
        pred = self.model.predict(data)
        return pred

    def transform(self,x):
        data = self.preprocess.predict(x)
        data = scale(data)
        pred = self.model.predict(data)
        return x.assign(non_tree=pred)
    
    def get_params(self,deep=True):
        return {}

class lasso_model:
    def __init__(self):
        rp = nn_preprocess()
        rpstep = rp.steps
        pp_steps = [
            ("pre",Pipeline(steps=rpstep)),
            ("dummy",dummy())
        ]
        self.preprocess = Pipeline(steps=pp_steps)
        self.model = Lasso()
    def fit(self,x,y):
        self.preprocess.fit(x,y)
        data = self.preprocess.predict(x)
        data = scale(data)
        self.model.fit(data,y)
        return self
    def predict(self,x):
        data = self.preprocess.predict(x)
        data = scale(data)
        pred = self.model.predict(data)
        return pred

    def transform(self,x):
        data = self.preprocess.predict(x)
        data = scale(data)
        pred = self.model.predict(data)
        return x.assign(non_tree=pred)
    
    def get_params(self,deep=True):
        return {}

class nn_model:
    def __init__(self):
        rp = nn_preprocess()
        rpstep = rp.steps
        pp_steps = [
            ("pre",Pipeline(steps=rpstep)),
            ("dummy",dummy())
        ]
        self.preprocess = Pipeline(steps=pp_steps)
        ########
        self.model = Sequential()
        self.model.add(Dense(100))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(100))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(100))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))
        self.model.add(Activation("relu"))
        self.model.compile(optimizer="Adam",loss="mean_squared_error")
        #########
    def fit(self,x,y):
        self.preprocess.fit(x,y,verbose=0)
        data = self.preprocess.predict(x)
        data = scale(data)
        ############
        self.model.fit(data,y.values,epochs=200, batch_size=32)
        ###########
        return self
    def predict(self,x):
        data = self.preprocess.predict(x)
        data = scale(data)
        pred = self.model.predict(data)
        return pred

    def transform(self,x):
        data = self.preprocess.predict(x)
        data = scale(data)
        pred = self.model.predict(data)
        return x.assign(nn_pred=pred)
    
    def get_params(self,deep=True):
        return {}

class my_preprocess:
    def __init__(self,seed):
        self.steps = [
            ("parse_area",parse_area_size()),
            ("parse_room",parse_rooms()),
            ("parse_old",parse_how_old()),
            ("height_enc",height_encoder()),
            ("ex_dist",extract_district()),
            ("dist_price_per_area",dist_and_price_per_area()),
            ("label_dist",district_encoder()),
            ("acc_ext",access_extractor()),
            ("tr_enc",train_encoder()),
            ("parking_encoder",parking_encoder()),
            ("dir_enc",direction_encoder()),
            ("info_enc",info_encoder()),
            ("m_d_p",add_mean_dist_price()),
            ("p_con_time",parse_contract_time()),
            ("fac",fac_encoder()),
            ("bath",bath_encoder()),
            ("kit",kitchin_encoder()),
            ("env",env_encoder()),
            ("angle_stat",add_mean_angle_price()),
            ("structure_enc",structure_label_encoder()),

            ("mean_struct",add_mean_structure_price()),
            ("mean_walk",add_mean_walk_price()),
            ("mean_moyori",add_moyori_walk_price()),

            ("drop_unnecessary",drop_unnecessary()),

            ("cross",cross_features()),
            ("dist2main_st",dist_to_main_station()),
            ("short_main_st",shortest2main_st()),

            ("area_predictor",area_pre_predictor(seed)),
            ("area_pre_price_predictor",area_per_price_predictor(seed)),
            ("knn_pred",Knn_regression()),
            ("knn_tika1",knn_tika1()),
            ("knn_tika2",knn_tika2()),
            ("NMF_train_walk",NMF_train_walk(seed)),
            ("NMF_fac",NMF_fac(seed)),
            ("NMF_kit",NMF_kit(seed)),
            ("NMF_env_dist",NMF_env_dist(seed)),
            ("NMF_env",NMF_env(seed)),

            ("actual_height",actual_height()),
            ("middle_class",middle_class_centers()),
            ("high_class",height_class_center()),

            ("only_rich",only_rich_model(seed)),

            ("pre_cat",pre_predictor(seed))
]


# def pre_checker(x,y):
#     rp = my_preprocess()
#     rpstep = rp.steps
#     m3 = [
#             ("pre", Pipeline(steps = rpstep)),
#             ("summy",dummy())
#         ]
#     m3 = Pipeline(steps=m3)
#     m3.fit(x,y)
#     return m3
      
        
class my_model:
    def __init__(self,seed=7777):
        rp = my_preprocess(seed)
        rpstep = rp.steps
        rich_step_xgb = [
            ("non_tree",ridge_model()),
            # ("nn_model",nn_model()),
            ("pre",Pipeline(steps=rpstep)),
            ("xgb",xgb.XGBRegressor(random_state=seed,max_depth=8))
            # ("cbr",CatBoostRegressor(random_state=seed,logging_level='Silent'))
        ]
        self.models = [
            Pipeline(steps=rich_step_xgb),
        ]
        
    def fit(self,x,y):
        tar = y
        for model in self.models:
            model.fit(x,tar)
        return self
    def predict(self,x):
        temp = np.zeros(len(x.values))
        for model in self.models:
            pred = model.predict(x)
            temp += np.array(pred)
        temp = temp/len(self.models)
        predict = temp
        return predict
    
    def get_params(self,deep=True):
        return {}

class double_predict_model:
    def __init__(self):
        self.model = my_model()
        self.diff_predictor = my_model()
    def fit(self,x,y):
        kf = KFold(n_splits=4)
        diffs = []
        for train_ind,test_ind in kf.split(x):
            model = my_model()
            train_x = x.iloc[train_ind]
            train_y = y.iloc[train_ind]
            test_x = x.iloc[test_ind]
            test_y = y.iloc[test_ind]
            model.fit(train_x,train_y)
            kf_pred = model.predict(test_x)
            diff = np.array(test_y)-np.array(kf_pred)
            diffs.append(diff)
        diffs = np.array(list(itertools.chain.from_iterable(diffs)))
        self.diff_predictor.fit(x,diffs)
        hoge = x.assign(pre_diff=diffs)
        self.model.fit(hoge,y)
        return self
    def predict(self,x):
        diffs = self.diff_predictor.predict(x)
        hoge = x.assign(pre_diff=diffs)
        pred = self.model.predict(hoge)
        return pred
    def get_params(self,deep=True):
        return {}

class double_predict_sum_model:
    def __init__(self):
        self.model = my_model()
        self.diff_predictor = my_model()
    def fit(self,x,y):
        kf = KFold(n_splits=4)
        diffs = []
        for train_ind,test_ind in kf.split(x):
            model = my_model()
            train_x = x.iloc[train_ind]
            train_y = y.iloc[train_ind]
            test_x = x.iloc[test_ind]
            test_y = y.iloc[test_ind]
            model.fit(train_x,train_y)
            kf_pred = model.predict(test_x)
            diff = np.array(test_y)-np.array(kf_pred)
            diffs.append(diff)
        diffs = np.array(list(itertools.chain.from_iterable(diffs)))
        self.diff_predictor.fit(x,diffs)
        self.model.fit(x,y)
        return self
    def predict(self,x):
        diffs = self.diff_predictor.predict(x)
        pred = self.model.predict(x)
        return np.array(pred)+np.array(diffs)
    def get_params(self,deep=True):
        return {}



def check_model(comment,train_x,train_y):
    now = datetime.datetime.now()
    name = str(now.month)+"_"+str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)
    f = open("logs.txt","a")
    f.write("\n")
    f.write(name + "\n")
    scores = cross_val_score(my_model(),train_x,train_y,scoring="neg_mean_squared_error",cv=4)
    scores = np.sqrt(-np.array(scores))
    print(scores)
    f.write("---cross val scoers---\n")
    for i in range(4):
        f.write(str(scores[i])+" ")
    f.write("\n")
    f.write(str(np.mean(scores)))
    f.write("\n")
    f.write("---comment---\n")
    f.write(comment)
    f.write("\n")
    
    easy = my_model()
    easy.fit(train_x,train_y)
    f = open("./feature_importances/"+name+"_ftxt","w")
    rm = easy.models[0]
    d1 = rm[-1].get_booster().get_score(importance_type='gain')
    for key in d1:
        f.write(key+" "+str(d1[key])+"\n")
    f.write("\n")
    f.close()

def check_nn(comment,train_x,train_y):
    scores = cross_val_score(non_tree_model(),train_x,train_y,scoring="neg_mean_squared_error",cv=4)
    scores = np.sqrt(-np.array(scores))
    print(scores)

    
def commit(train_x,train_y,test,name,seeds):
    pred_all = []
    for i in range(len(seeds)):
        print(i)
        model = my_model(seeds[i])
        model.fit(train_x,train_y)
        pred = model.predict(test)
        pred_all.append(pred)
        pickle.dump(model, open(name+"seed_"+str(seeds[i])+".pkl", "wb"))
    pred_all = np.array(pred_all)
    pred = np.mean(pred_all,axis=0)
    pred = pd.DataFrame(pred)
    pred.columns=["pred"]
    pred.index = test.index
    pred = pd.concat([test["id"],pred],axis=1)
    pred.to_csv(name+".csv",header=False,index=False)




def classify_ensemble(models,x_valid):
    p = []
    predict = [0 for i in range(len(x_valid))]
    for m in models:
        p.append(m.predict(x_valid))
    for i in range(len(x_valid)):
        temp = 0
        for j in range(len(models)):
            temp += p[j][i]/3
        predict[i] = temp
        if temp > 0.9:
            predict[i] = 1
        else:
            predict[i] = 0
    predict = np.array(predict,dtype=np.int64)
    return predict


class split_double_predict_model:
    def __init__(self,threshold=300000,seed=7777):
        temp = my_preprocess(seed)
        forest_pre = temp.steps
        temp = nn_preprocess()
        linear_pre = temp.steps
        m1 = [
            ("pre", Pipeline(steps = forest_pre)),
            ("xgb",xgb.XGBRegressor(max_depth=8,min_child_weight=0,random_state=7777,objective="reg:logistic"))
        ]
        m2 = [
            ("pre", Pipeline(steps = forest_pre)),
            ("rfr",RandomForestRegressor(random_state=7777))
        ]
        m3 = [
            ("pre", Pipeline(steps = linear_pre)),
            ("lgi",LogisticRegression(random_state=7777))
        ]

        self.c_models = [
            Pipeline(steps=m1),
            Pipeline(steps=m2),
            Pipeline(steps=m3),
        ]
        self.threshold = threshold
        self.model = my_model()
        self.diff_predictor = my_model()
    def fit(self,x,y):
        temp = y.values
        is_rich_label = [0 for i in range(len(temp))]
        for i in range(len(temp)):
            if temp[i] > self.threshold:
                is_rich_label[i] = 1
        for i in range(len(self.c_models)):
            self.c_models[i].fit(x,is_rich_label)

        kf = KFold(n_splits=4)
        diffs = []
        for train_ind,test_ind in kf.split(x):
            model = my_model()
            train_x = x.iloc[train_ind]
            train_y = y.iloc[train_ind]
            test_x = x.iloc[test_ind]
            test_y = y.iloc[test_ind]
            model.fit(train_x,train_y)
            kf_pred = model.predict(test_x)
            diff = np.array(test_y)-np.array(kf_pred)
            diffs.append(diff)
        diffs = np.array(list(itertools.chain.from_iterable(diffs)))
        self.diff_predictor.fit(x,diffs)
        # pre_diff_cross_isrich = np.array(is_rich_label)*diffs
        # hoge = x.assign(pre_diff_cross_isrich=pre_diff_cross_isrich)
        # hoge = hoge.assign(pre_diff=diffs)
        self.model.fit(x,y)
        return self
    def predict(self,x):
        sep = classify_ensemble(self.c_models,x)
        diffs = self.diff_predictor.predict(x)
        pre_diff_cross_isrich = sep*np.array(diffs)
        # hoge = x.assign(pre_diff_cross_isrich=pre_diff_cross_isrich)
        # hoge = hoge.assign(pre_diff=diffs)
        pred = self.model.predict(x)

        ##################
        mask = np.abs(np.array(pred)) < 100000
        pre_diff_cross_isrich[mask] = 0
        ##################
        
        return np.array(pred)+np.array(pre_diff_cross_isrich)

    def get_params(self,deep=True):
        return {}
    