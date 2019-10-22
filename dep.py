import numpy as np
import re
import pandas as pd
import xgboost as xgb

def apr_(x):
    num_space = x["mf_r"].values+x["mf_l"].values+np.maximum(x["mf_d"].values,x["mf_k"].values)+x["mf_s"].values
    num_space = np.array(num_space)
    apr = x["mf_areasize"].values/x["mf_r"].values
    apr_all = x["mf_areasize"].values/num_space
    return apr,apr_all

def apr_sq_(x):
    num_space = x["mf_r"].values+x["mf_l"].values+np.maximum(x["mf_d"].values,x["mf_k"].values)+x["mf_s"].values
    num_space = np.array(num_space)
    apr = x["mf_area_sq"].values/x["mf_r"].values
    apr_all = x["mf_area_sq"].values/num_space
    return apr,apr_all

def acr_(x):
    num_space =x["mf_r"].values+x["mf_l"].values+np.maximum(x["mf_d"].values,x["mf_k"].values)+x["mf_s"].values
    num_space = np.array(num_space)
    acr = x["mf_areasize"].values*x["mf_r"].values
    acr_all = x["mf_areasize"].values*num_space
    return acr,acr_all


def acr_sq_(x):
    num_space = x["mf_r"].values+x["mf_l"].values+np.maximum(x["mf_d"].values,x["mf_k"].values)+x["mf_s"].values
    num_space = np.array(num_space)
    acr = x["mf_area_sq"].values*x["mf_r"].values
    acr_all = x["mf_area_sq"].values*num_space
    return acr,acr_all

def relative_height_(x):
    a = x["mf_what_floor"].values
    b = x["mf_height_bld"].values
    per = a/b
    diff = a-b
    return per,diff

class actual_height:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        data = x["field_height"].values
        hoge = data + x["mf_what_floor"].values*3
        return x.assign(actual_height=hoge)

class shortest2main_st:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        arr = [
            x["dist_main_st1"].values,
            x["dist_main_st2"].values,
            x["dist_main_st3"].values,
            x["dist_main_st4"].values,
            x["dist_main_st5"].values,
            x["dist_main_st6"].values
        ]

        ans = x["dist_main_st0"].values

        for i in range(len(ans)):
            for j in range(len(arr)):
                if arr[j][i] < ans[i]:
                    ans[i] = arr[j][i]
        return x.assign(shortest_mainst=ans)

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