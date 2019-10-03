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
#  * @param float $lat1 緯度１
#  * @param float $lon1 経度１
#  * @param float $lat2 緯度２
#  * @param float $lon2 経度２

def google_distance(lat1, lon1, lat2, lon2):
    radLat1 = math.radians(lat1) 
    radLon1 = math.radians(lon1) 
    radLat2 = math.radians(lat2) 
    radLon2 = math.radians(lon2)

    r = 6378137.0

    averageLat = (radLat1 - radLat2) / 2
    averageLon = (radLon1 - radLon2) / 2
    return r * 2 * math.asin(math.sqrt(pow(math.sin(averageLat), 2) + math.cos(radLat1) * math.cos(radLat2) * pow(math.sin(averageLon), 2)))
    

class dist_to_main_station:
    def __init__(self):
        """
        池袋駅 座標(WGS84)　緯度: 35.729503 経度: 139.7109
        新宿駅 座標(WGS84)　緯度: 35.689738 経度: 139.700391
        渋谷駅 座標(WGS84)　緯度: 35.658034 経度: 139.701636
        東京駅 座標(WGS84)　緯度: 35.681236 経度: 139.767125
        上野駅 座標(WGS84)　緯度: 35.714167 経度: 139.777409
        品川駅 座標(WGS84)　緯度: 35.628471 経度: 139.73876
        新橋駅 座標(WGS84)　緯度: 35.666379 経度: 139.75834
        """
        self.main_st = [[35.729503,139.7109],
                        [35.689738,139.700391],
                        [35.658034,139.701636],
                        [35.681236,139.767125],
                        [35.714167,139.777409],
                        [35.628471,139.73876],
                        [35.666379,139.75834]]

    def fit(self,x,y):
        return self
    def transform(self,x):
        ido = x["ido"].values
        keido = x["keido"].values
        dist = [[10000 for i in range(len(self.main_st))] for j in range(len(ido))]
        for i in range(len(ido)):
            for j in range(len(self.main_st)):
                to_ido = self.main_st[j][0]
                to_keido = self.main_st[j][1]
                d = google_distance(ido[i],keido[i],to_ido,to_keido)
                dist[i][j] = d
        dist = pd.DataFrame(dist)
        dist.index = x.index
        col = []
        for i in range(len(self.main_st)):
            col.append("dist_main_st"+str(i))
        dist.columns = col
        return pd.concat([x,dist],axis=1)






class dummy:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        return x
    def predict(self,x):
        return x

