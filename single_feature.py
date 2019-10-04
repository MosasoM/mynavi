import re
import pandas as pd
import numpy as np
import xgboost as xgb
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,scale,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import zscore
from sklearn.decomposition import NMF

from single import *
from cross import *
    
class cross_features:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        hoge = x.copy()
        
        apr,apr_all = apr_(x)
        hoge = hoge.assign(apr=apr)
        hoge = hoge.assign(apr_all=apr_all)
        acr,acr_all = acr_(x)
        hoge = hoge.assign(acr=acr)
        hoge = hoge.assign(acr_all=acr_all)
        
        apr_sq,apr_all_sq = apr_sq_(x)
        hoge = hoge.assign(apr_sq=apr_sq)
        hoge = hoge.assign(apr_all_sq=apr_all_sq)
        acr_sq,acr_all_sq = acr_sq_(x)
        hoge = hoge.assign(acr_sq=acr_sq)
        hoge = hoge.assign(acr_all_sq=acr_all_sq)
        
        
        per,diff = relative_height_(x)
        hoge = hoge.assign(height_percent = per)
        hoge = hoge.assign(height_diff=diff)
        
        temp = x["mf_year"].values/x["mf_what_floor"].values
        hoge = hoge.assign(year_floor=temp)
        
        temp = x["mf_areasize"].values*x["mf_what_floor"].values
        hoge = hoge.assign(area_fot_floor = temp)
        
        return hoge

class shortest2main_st:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        arr = [x["dist_main_st1"].values,x["dist_main_st2"].values,
        x["dist_main_st3"].values,
        x["dist_main_st4"].values,
        x["dist_main_st5"].values,
        x["dist_main_st6"].values]

        ans = x["dist_main_st0"].values

        for i in range(len(ans)):
            for j in range(len(arr)):
                if arr[j][i] < ans[i]:
                    ans[i] = arr[j][i]
        return x.assign(shortest_mainst=ans)

class area_pre_predictor:
    def __init__(self,rand_s):
        self.model = xgb.XGBRegressor(random_state=rand_s)
        pass
    def fit(self,x,y):
        tar = x["mf_areasize"].values
        ex_var = x.drop(["mf_area_sq","mf_areasize"],axis = 1)
        self.model.fit(ex_var,tar)
        return self
    def transform(self,x):
        ex_var = x.drop(["mf_area_sq","mf_areasize"],axis = 1)
        pred = self.model.predict(ex_var)
        hoge = x.assign(pred_area=pred)
        hoge = hoge.assign(area_diff=hoge["mf_areasize"].values-hoge["pred_area"].values)
        return hoge

class area_per_price_predictor:
    def __init__(self,rand_s):
        self.model = xgb.XGBRegressor(random_state=rand_s)
        pass
    def fit(self,x,y):
        tar = np.array(y)/x["mf_areasize"].values
        ex_var = x.drop(["mf_area_sq","mf_areasize"],axis = 1)
        self.model.fit(ex_var,tar)
        return self
    def transform(self,x):
        ex_var = x.drop(["mf_area_sq","mf_areasize"],axis = 1)
        pred = self.model.predict(ex_var)
        hoge = x.assign(pred_area=pred)
        hoge = hoge.assign(area_diff=hoge["mf_areasize"].values-hoge["pred_area"].values)
        hoge = hoge.assign(avg_cross_pred = hoge["mf_areasize"].values*np.array(pred))
        return hoge

class Knn_regression:
    def __init__(self):
        self.model = KNeighborsRegressor(n_neighbors=30,weights="distance")
    def fit(self,x,y):
        ex_var = x[["ido","keido"]].values
        ex_var = zscore(ex_var)
        ty = np.array(y)/x["mf_areasize"].values
        self.model.fit(ex_var,ty)
        return self
    def transform(self,x):
        ex_var = x[["ido","keido"]].values
        ex_var = zscore(ex_var)
        pred = self.model.predict(ex_var)
        hoge = x.assign(knn_pred=pred)
        temp = pred*x["mf_areasize"].values
        hoge = hoge.assign(knn_area_price=temp)
        return hoge

class NMF_train_walk:
    def __init__(self,rand_s):
        self.cols = []
        self.model = NMF(n_components=20, init='nndsvd', random_state=rand_s)
    def fit(self,x,y):
        for col in x.columns:
            if "train_walk_" in col: #ここと
                self.cols.append(col)
        ex_var = x[self.cols]
        self.model.fit(ex_var)
        return self
    def transform(self,x):
        ex_var = x[self.cols]
        p = self.model.transform(ex_var)
        hoge = x.drop(self.cols,axis = 1)
        p = pd.DataFrame(p)
        p.index = x.index
        n_col = len(p.columns)
        col = []
        for i in range(n_col):
            col.append("tr_wa_nmf"+str(i)) #ここを変更する必要
        p.columns = col
        return pd.concat([hoge,p],axis = 1)

class NMF_fac:
    def __init__(self,rand_s):
        self.cols = []
        self.model = NMF(n_components=15, init='nndsvd', random_state=rand_s)
    def fit(self,x,y):
        for col in x.columns:
            if "fac" in col: #ここと
                self.cols.append(col)
        ex_var = x[self.cols]
        self.model.fit(ex_var)
        return self
    def transform(self,x):
        ex_var = x[self.cols]
        p = self.model.transform(ex_var)
        hoge = x.drop(self.cols,axis = 1)
        p = pd.DataFrame(p)
        p.index = x.index
        n_col = len(p.columns)
        col = []
        for i in range(n_col):
            col.append("fac_nmf"+str(i)) #ここを変更する必要
        p.columns = col
        return pd.concat([hoge,p],axis = 1)

class NMF_kit:
    def __init__(self,rand_s):
        self.cols = []
        self.model = NMF(n_components=10, init='nndsvd', random_state=rand_s)
    def fit(self,x,y):
        for col in x.columns:
            if "kit" in col: #ここと
                self.cols.append(col)
        ex_var = x[self.cols]
        self.model.fit(ex_var)
        return self
    def transform(self,x):
        ex_var = x[self.cols]
        p = self.model.transform(ex_var)
        hoge = x.drop(self.cols,axis = 1)
        p = pd.DataFrame(p)
        p.index = x.index
        n_col = len(p.columns)
        col = []
        for i in range(n_col):
            col.append("kit_nmf"+str(i)) #ここを変更する必要
        p.columns = col
        return pd.concat([hoge,p],axis = 1)

class NMF_env_dist:
    def __init__(self,rand_s):
        self.cols = []
        self.model = NMF(n_components=10, init='nndsvd', random_state=rand_s)
    def fit(self,x,y):
        for col in x.columns:
            if "env_dist" in col: #ここと
                self.cols.append(col)
        ex_var = x[self.cols]
        self.model.fit(ex_var)
        return self
    def transform(self,x):
        ex_var = x[self.cols]
        p = self.model.transform(ex_var)
        hoge = x.drop(self.cols,axis = 1)
        p = pd.DataFrame(p)
        p.index = x.index
        n_col = len(p.columns)
        col = []
        for i in range(n_col):
            col.append("env_dist_nmf"+str(i)) #ここを変更する必要
        p.columns = col
        return pd.concat([hoge,p],axis = 1)

class NMF_env:
    def __init__(self,rand_s):
        self.cols = []
        self.model = NMF(n_components=10, init='nndsvd', random_state=rand_s)
    def fit(self,x,y):
        for col in x.columns:
            if "envv" in col: #ここと
                self.cols.append(col)
        ex_var = x[self.cols]
        self.model.fit(ex_var)
        return self
    def transform(self,x):
        ex_var = x[self.cols]
        p = self.model.transform(ex_var)
        hoge = x.drop(self.cols,axis = 1)
        p = pd.DataFrame(p)
        p.index = x.index
        n_col = len(p.columns)
        col = []
        for i in range(n_col):
            col.append("envv_nmf"+str(i)) #ここを変更する必要
        p.columns = col
        return pd.concat([hoge,p],axis = 1)

class NMF_info:
    def __init__(self,rand_s):
        self.cols = []
        self.model = NMF(n_components=5, init='nndsvd', random_state=rand_s)
    def fit(self,x,y):
        for col in x.columns:
            if "info" in col: #ここと
                self.cols.append(col)
        ex_var = x[self.cols]
        self.model.fit(ex_var)
        return self
    def transform(self,x):
        ex_var = x[self.cols]
        p = self.model.transform(ex_var)
        hoge = x.drop(self.cols,axis = 1)
        p = pd.DataFrame(p)
        p.index = x.index
        n_col = len(p.columns)
        col = []
        for i in range(n_col):
            col.append("info_nmf"+str(i)) #ここを変更する必要
        p.columns = col
        return pd.concat([hoge,p],axis = 1)



class dist_and_price_per_area:
    def __init__(self):
        self.means = {}
        self.mean_pad = 0
    def fit(self,x,y):

        tx = x["mf_areasize"].values
        ty = np.array(y)/tx
        ty = pd.DataFrame(y)
        ty.columns=["p_per_a"]
        ty.index = x.index
        temptemp = pd.concat([x,ty],axis = 1)
        label = temptemp.groupby("district").mean().index.values

        temp = temptemp.groupby("district").mean()["p_per_a"].values
        for i in range(len(label)):
            self.means[label[i]] = temp[i]
        self.mean_pad = round(np.mean(temp))
        return self
    def transform(self,x):
        buf1 = [0 for i in range(len(x.values))]
        temp = x["district"].values
        for i in range(len(x.values)):
            if temp[i] in self.means:
                buf1[i] = self.means[temp[i]]
            else:
                buf1[i] = self.mean_pad
        hoge = x.assign(p_per_a_d=buf1)
        temp = np.array(buf1)*x["mf_areasize"].values
        hoge = hoge.assign(p_per_a_d_c_a=temp)
        return hoge






class dummy:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        return x
    def predict(self,x):
        return x

