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
from sklearn.ensemble import RandomForestRegressor

from single_feature import *
from basic import *
from keras.models import Sequential
from keras.layers import Dense, Activation

import itertools

from sklearn.linear_model import Lasso,Ridge,ElasticNet

from model1 import *
from model2 import *
from model3 import *
from model4 import *
from model5 import *
from model6 import *
from model7 import *

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
            ("NMF_train_walk",NMF_train_walk(seed)),
            ("NMF_fac",NMF_fac(seed)),
            ("NMF_kit",NMF_kit(seed)),
            ("NMF_env_dist",NMF_env_dist(seed)),
            ("NMF_env",NMF_env(seed)),
]


class stacking_model:
    def __init__(self):
        self.first_models=[
            model1(7777,5),
            model1(7777,8),
            model1(7777,9),
            model2(7777,None),
            model2(7777,10),
            model2(7777,12),

            model1(7778,5),
            model1(7778,8),
            model1(7778,9),
            model2(7778,None),
            model2(7778,10),
            model2(7778,12),


            model5(7777,0.1),
            model5(7777,1),
            model5(7777,10),

            model5(7778,0.1),
            model5(7778,1),
            model5(7778,10),

            model6(30,"distance"),
            model6(30,"uniform"),
            model6(60,"distance"),
            model6(60,"uniform"),
            model6(100,"distance"),
            model6(100,"uniform"),

            model7(30,"distance"),
            model7(30,"uniform"),
            model6(60,"distance"),
            model6(60,"uniform"),
            model7(100,"distance"),
            model7(100,"uniform"),
        ]
        rp = my_preprocess(8888)
        rpstep = rp.steps
        rich_step_xgb = [
            ("pre",Pipeline(steps=rpstep)),
            ("xgb",xgb.XGBRegressor(random_state=8888,max_depth=8))
        ]
        self.main_models = Pipeline(steps=rich_step_xgb)
        # self.main_model=RandomForestRegressor(random_state=7777)
    def fit(self,x,y):
        kf = KFold(n_splits=5)
        first_preds = pd.DataFrame()
        ind = 0
        for model in self.first_models:
            buf = []
            for train_ind,test_ind in kf.split(x):
                train_x = x.iloc[train_ind]
                train_y = y.iloc[train_ind]
                test_x = x.iloc[test_ind]
                test_y = y.iloc[test_ind]
                model.fit(train_x,train_y)
                kf_pred = model.predict(test_x)
                buf.append(kf_pred)
            buf = np.array(list(itertools.chain.from_iterable(buf)))
            buf = pd.DataFrame(buf)
            buf.index=x.index
            buf.columns=["first_predict"+str(ind)]
            first_preds = pd.concat([first_preds,buf],axis=1)
            ind += 1
        hoge = pd.concat([x,first_preds],axis=1)
        self.main_model.fit(hoge,y)
        return self
    def predict(self,x):
        kf = KFold(n_splits=3)
        first_preds = pd.DataFrame()
        ind = 0
        for model in self.first_models:
            pred = model.predict(x)
            buf = pd.DataFrame(pred)
            buf.index=x.index
            buf.columns=["first_predict"+str(ind)]
            first_preds = pd.concat([first_preds,buf],axis=1)
            ind += 1
        hoge = pd.concat([x,first_preds],axis=1)
        pred = self.main_model.predict(hoge,y)
        return pred

    def get_params(self,deep=True):
        return {}

