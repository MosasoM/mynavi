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



class model7:
    def __init__(self,neigh,weight):
        self.model = KNeighborsRegressor(n_neighbors=neigh,weights=weight)
    def fit(self,x,y):
        ex_var = fuga[["ido","keido"]].values
        ex_var = zscore(ex_var)
        ty = np.array(y)
        self.model.fit(ex_var,ty)
        return self
    def predict(self,x):
        ex_var = fuga[["ido","keido"]].values
        ex_var = zscore(ex_var)
        pred = self.model.predict(ex_var)
        return pred

