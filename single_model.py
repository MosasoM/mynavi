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

from single_feature import *
from basic import *


class my_preprocess:
    def __init__(self):
        self.steps = [
            ("parse_area",parse_area_size()),
            ("parse_room",parse_rooms()),
            ("parse_old",parse_how_old()),
            ("height_enc",height_encoder()),
            ("ex_dist",extract_district()),
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
            ("cross",cross_features()),
            ("dist2main_st",dist_to_main_station()),
            ("drop_unnecessary",drop_unnecessary())
        ]


def pre_checker(x,y):
    rp = my_preprocess()
    rpstep = rp.steps
    m3 = [
            ("pre", Pipeline(steps = rpstep)),
            ("summy",dummy())
        ]
    m3 = Pipeline(steps=m3)
    m3.fit(x,y)
    return m3
      
        
class my_model:
    def __init__(self,is_log=False):
        rp = my_preprocess()
        rpstep = rp.steps
        rich_step_xgb = [
            ("pre",Pipeline(steps=rpstep)),
            ("xgb",xgb.XGBRegressor(random_state=8888,max_depth=8))
        ]
        self.models = [
            Pipeline(steps=rich_step_xgb),
        ]
        self.is_log=is_log
    def fit(self,x,y):
        tar = y
        if self.is_log:
            tar = np.log(y)
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
        if self.is_log:
            predict = np.exp(predict)
        return predict
    
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

    
def commit(model,train_x,train_y,test,name):
    model.fit(train_x,train_y)
    pred = model.predict(test)
    pred = pd.DataFrame(pred)
    pred.columns=["pred"]
    pred.index = test.index
    pred = pd.concat([test["id"],pred],axis=1)
    pred.to_csv(name+".csv",header=False,index=False)
    pickle.dump(model, open(name+".pkl", "wb"))





    