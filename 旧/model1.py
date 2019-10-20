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

from single_feature import *
from basic import *
from keras.models import Sequential
from keras.layers import Dense, Activation

import itertools

from sklearn.linear_model import Lasso,Ridge,ElasticNet

class pre_model1:
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
                    # ("NMF_train_walk",NMF_train_walk(seed)),
                    # ("NMF_fac",NMF_fac(seed)),
                    # ("NMF_kit",NMF_kit(seed)),
                    # ("NMF_env_dist",NMF_env_dist(seed)),
                    # ("NMF_env",NMF_env(seed)),
        ]

class model1:
    def __init__(self,seed,depth):
        rp = pre_model1(seed)
        rpstep = rp.steps
        rich_step_xgb = [
            ("pre",Pipeline(steps=rpstep)),
            ("xgb",xgb.XGBRegressor(random_state=seed,max_depth=depth))
        ]
        self.model=Pipeline(steps=rich_step_xgb)
    def fit(self,x,y):
        self.model.fit(x,y)
        return self
    def predict(self,x):
        pred = self.model.predict(x)
        return pred
    def get_params(self,deep=True):
        return {}