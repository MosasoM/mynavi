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

class pre_model5:
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
        ]

class model5:
    def __init__(self,seed,alpha):
        rp = pre_model5()
        rpstep = rp.steps
        pp_steps = [
            ("pre",Pipeline(steps=rpstep)),
            ("dummy",dummy())
        ]
        self.preprocess = Pipeline(steps=pp_steps)
        self.model = Ridge(random_state=seed,alpha=alpha)
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
    def get_params(self,deep=True):
        return {}