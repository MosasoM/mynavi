import numpy as np
import xgboost as xgb

from sklearn.pipeline import Pipeline
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error
import datetime
from single import *
import pickle
from sklearn.metrics import accuracy_score

from rich_features import *

class rich_pre:
    def __init__(self):
        self.steps = [
            ("parse_area",r_parse_area_size()),
            ("parse_room",r_parse_rooms()),
            ("parse_old",r_parse_how_old()),
            ("height_enc",r_height_encoder()),
            ("ex_dist",r_extract_district()),
            ("label_dist",r_district_encoder()),
            ("acc_ext",r_access_extractor()),
            ("tr_enc",r_train_encoder()),
            ("parking_encoder",r_parking_encoder()),
            ("dir_enc",r_direction_encoder()),
            ("info_enc",r_info_encoder()),
            ("m_d_p",r_add_mean_dist_price()),
            ("p_con_time",r_parse_contract_time()),
            ("fac",r_fac_encoder()),
            ("bath",r_bath_encoder()),
            ("kit",r_kitchin_encoder()),
            # ("env",r_env_encoder()),
            ("structure_enc",r_structure_label_encoder()),
            ("mean_walk",r_add_mean_walk_price()),
            ("mean_moyori",r_add_moyori_walk_price()),
            ("mean_rldk",r_add_rldk_price()),
            ("cross",r_cross_features()),
            ("area_walk",r_area_and_walk_label()),
            ("area_walk_mean",r_aw_label_mean()),
            ("drop_unnecessary",r_drop_unnecessary())
        ]


def pre_checker_rich(x,y):
    rp = rich_pre()
    rpstep = rp.steps
    m3 = [
            ("pre", Pipeline(steps = rpstep)),
            ("summy",dummy())
        ]
    m3 = Pipeline(steps=m3)
    m3.fit(x,y)
    return m3


class easy_model_rich:
    def __init__(self,is_log=False):
        rp = rich_pre()
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

def check_rich(comment,train_x,train_y):
    now = datetime.datetime.now()
    name = str(now.month)+"_"+str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)
    f = open("logs_rich.txt","a")
    f.write("\n")
    f.write(name + "\n")
    scores = cross_val_score(easy_model_rich(),train_x,train_y,scoring="neg_mean_squared_error",cv=4)
    scores = np.sqrt(-np.array(scores))
    print(scores)
    print(np.mean(scores))
    f.write("---cross val scoers---\n")
    for i in range(4):
        f.write(str(scores[i])+" ")
    f.write("\n")
    f.write(str(np.mean(scores)))
    f.write("\n")
    f.write("---comment---\n")
    f.write(comment)
    f.write("\n")
    
    easy = easy_model_rich()
    easy.fit(train_x,train_y)
    f = open("./feature_importances/"+name+"_f_rich.txt","w")
    rm = easy.models[0]
    d1 = rm[-1].get_booster().get_score(importance_type='gain')
    for key in d1:
        f.write(key+" "+str(d1[key])+"\n")
    f.write("\n")
    f.close()



