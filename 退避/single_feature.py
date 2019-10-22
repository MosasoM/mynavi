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
from annoy import AnnoyIndex
from catboost import CatBoostRegressor

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

class knn_tika1:
    def __init__(self):
        df = pd.read_csv("./Tika1.csv")
        df = df[["緯度","経度","住居表示","地積","Ｈ３１価格"]]
        df = df[df["住居表示"].str.contains("区")]
        df["緯度"] = df["緯度"].values/3600
        df["経度"] = df["経度"].values/3600
        df = df[["緯度","経度","Ｈ３１価格"]]
        df.columns = ["ido","keido","price"]
        self.df = df
        self.model = KNeighborsRegressor(n_neighbors=3,weights="distance")

    def fit(self,x,y):
        ex_var = self.df[["ido","keido"]].values
        ex_var = zscore(ex_var)
        ty = self.df["price"].values
        self.model.fit(ex_var,ty)
        return self

    def transform(self,x):
        ex_var = x[["ido","keido"]].values
        ex_var = zscore(ex_var)
        pred = self.model.predict(ex_var)
        hoge = x.assign(tika1=pred)
        temp = pred*x["mf_areasize"].values
        hoge = hoge.assign(knn_area_tika1=temp)
        return hoge

class knn_tika2:
    def __init__(self):
        df = pd.read_csv("./Tika2.csv")
        df = df[["緯度","経度","住居表示","地積","Ｈ３０価格"]]
        df = df[df["住居表示"].str.contains("区")]
        df["緯度"] = df["緯度"].values/3600
        df["経度"] = df["経度"].values/3600
        df = df[["緯度","経度","Ｈ３０価格"]]
        df.columns = ["ido","keido","price"]
        self.df = df
        self.model = KNeighborsRegressor(n_neighbors=3,weights="distance")

    def fit(self,x,y):
        ex_var = self.df[["ido","keido"]].values
        ex_var = zscore(ex_var)
        ty = self.df["price"].values
        self.model.fit(ex_var,ty)
        return self

    def transform(self,x):
        ex_var = x[["ido","keido"]].values
        ex_var = zscore(ex_var)
        pred = self.model.predict(ex_var)
        hoge = x.assign(tika2=pred)
        temp = pred*x["mf_areasize"].values
        hoge = hoge.assign(knn_area_tika2=temp)
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

class NMF_trainOH:
    def __init__(self,rand_s):
        self.cols = []
        self.model = NMF(n_components=15, init='nndsvd', random_state=rand_s)
    def fit(self,x,y):
        for col in x.columns:
            if "train_OH" in col: #ここと
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
            col.append("train_OH"+str(i)) #ここを変更する必要
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
        self.out = None
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        return x
    def predict(self,x):
        self.out = x
        return x

class dist_salary:
    def __init__(self):
        self.keys= {"港区":1115.0756,"千代田区":944.5295,"渋谷区":801.1137,
                    "中央区":634.5647,"文京区":610.0272,"目黒区":602.0157,
                    "世田谷区":544.9548,"新宿区":518.5807,"杉並区":465.1943,
                    "品川区":462.9208,"豊島区":429.0091,"江東区":423.4898,
                    "大田区":422.6802,"台東区":416.9283,"練馬区":414.0596,
                    "中野区":412.9672,"墨田区":371.3652,"北区":368.4201,
                    "荒川区":362.0571,"板橋区":360.8186,"江戸川区":357.8159,
                    "葛飾区":343.4164,"足立区":338.9533,
                    }
    def fit(self,x,y):
        return self
    def transform(self,x):
        data = x["district"].values
        buf = [0 for i in range(len(data))]
        for i in range(len(data)):
            buf[i] = self.keys[data[i]]
        return x.assign(dist_salary=buf)

class pre_predictor:
    def __init__(self,seed):
        self.model = CatBoostRegressor(random_state=seed,logging_level='Silent')
        self.model2 = RandomForestRegressor(random_state=seed,n_estimators=100)

    def fit(self,x,y):
        self.model.fit(x,y)
        self.model2.fit(x,y)
        return self
    def transform(self,x):
        pred = self.model.predict(x)
        pred2 = self.model2.predict(x)
        hoge = x.assign(pre_rfr=pred2)
        # hoge = x.copy()
        return hoge.assign(pre_xgb=pred)

class annoy_area:
    def __init__(self):
        self.model = AnnoyIndex(2,metric="euclidean")
        self.y = None
        self.others = None
        self.col = None
        col = []
        for i in range(10):
            col.append("kit_nmf"+str(i))
        for i in range(15):
            col.append("fac_nmf"+str(i)) 
        col.append("mf_areasize")
        col.append("mf_year")
        col.append("mf_r")
        col.append("mf_l")
        col.append("mf_d")
        col.append("mf_k")
        col.append("mf_s")
        for i in range(8):
            col.append("bath"+str(i))
        self.col = col
    def fit(self,x,y):
        self.y = np.array(y)
        self.others = zscore(x[self.col].values)
        data = x[["ido","keido"]].values
        for i in range(len(data)):
            a,b = ido_calc_xy(data[i][0],data[i][1],35.681236,139.767125)
            self.model.add_item(i,[a,b])
        self.model.build(10)
        return self
    def transform(self,x):
        data = x[["ido","keido"]].values
        others = zscore(x[self.col].values)
        buf = [0 for i in range(len(data))]
        for i in range(len(data)):
            a,b = ido_calc_xy(data[i][0],data[i][1],35.681236,139.767125)
            temp = self.model.get_nns_by_vector([a,b],100)
            sim = 100000000000000
            for ind in temp:
                hoge = np.dot(self.others[ind],others[i])
                if sim > hoge:
                    sim = hoge
                    buf[i] = self.y[ind]
        return x.assign(annoy_area = buf)

def ido_calc_xy(phi_deg, lambda_deg, phi0_deg, lambda0_deg):
    """ 緯度経度を平面直角座標に変換する
    - input:
        (phi_deg, lambda_deg): 変換したい緯度・経度[度]（分・秒でなく小数であることに注意）
        (phi0_deg, lambda0_deg): 平面直角座標系原点の緯度・経度[度]（分・秒でなく小数であることに注意）
    - output:
        x: 変換後の平面直角座標[m]
        y: 変換後の平面直角座標[m]
    """
    # 緯度経度・平面直角座標系原点をラジアンに直す
    phi_rad = np.deg2rad(phi_deg)
    lambda_rad = np.deg2rad(lambda_deg)
    phi0_rad = np.deg2rad(phi0_deg)
    lambda0_rad = np.deg2rad(lambda0_deg)

    # 補助関数
    def A_array(n):
        A0 = 1 + (n**2)/4. + (n**4)/64.
        A1 = -     (3./2)*( n - (n**3)/8. - (n**5)/64. ) 
        A2 =     (15./16)*( n**2 - (n**4)/4. )
        A3 = -   (35./48)*( n**3 - (5./16)*(n**5) )
        A4 =   (315./512)*( n**4 )
        A5 = -(693./1280)*( n**5 )
        return np.array([A0, A1, A2, A3, A4, A5])

    def alpha_array(n):
        a0 = np.nan # dummy
        a1 = (1./2)*n - (2./3)*(n**2) + (5./16)*(n**3) + (41./180)*(n**4) - (127./288)*(n**5)
        a2 = (13./48)*(n**2) - (3./5)*(n**3) + (557./1440)*(n**4) + (281./630)*(n**5)
        a3 = (61./240)*(n**3) - (103./140)*(n**4) + (15061./26880)*(n**5)
        a4 = (49561./161280)*(n**4) - (179./168)*(n**5)
        a5 = (34729./80640)*(n**5)
        return np.array([a0, a1, a2, a3, a4, a5])

    # 定数 (a, F: 世界測地系-測地基準系1980（GRS80）楕円体)
    m0 = 0.9999 
    a = 6378137.
    F = 298.257222101

    # (1) n, A_i, alpha_iの計算
    n = 1. / (2*F - 1)
    A_array = A_array(n)
    alpha_array = alpha_array(n)

    # (2), S, Aの計算
    A_ = ( (m0*a)/(1.+n) )*A_array[0] # [m]
    S_ = ( (m0*a)/(1.+n) )*( A_array[0]*phi0_rad + np.dot(A_array[1:], np.sin(2*phi0_rad*np.arange(1,6))) ) # [m]

    # (3) lambda_c, lambda_sの計算
    lambda_c = np.cos(lambda_rad - lambda0_rad)
    lambda_s = np.sin(lambda_rad - lambda0_rad)

    # (4) t, t_の計算
    t = np.sinh( np.arctanh(np.sin(phi_rad)) - ((2*np.sqrt(n)) / (1+n))*np.arctanh(((2*np.sqrt(n)) / (1+n)) * np.sin(phi_rad)) )
    t_ = np.sqrt(1 + t*t)

    # (5) xi', eta'の計算
    xi2  = np.arctan(t / lambda_c) # [rad]
    eta2 = np.arctanh(lambda_s / t_)

    # (6) x, yの計算
    x = A_ * (xi2 + np.sum(np.multiply(alpha_array[1:],
                                       np.multiply(np.sin(2*xi2*np.arange(1,6)),
                                                   np.cosh(2*eta2*np.arange(1,6)))))) - S_ # [m]
    y = A_ * (eta2 + np.sum(np.multiply(alpha_array[1:],
                                        np.multiply(np.cos(2*xi2*np.arange(1,6)),
                                                    np.sinh(2*eta2*np.arange(1,6)))))) # [m]
    # return
    return x, y # [m]

class isnan_label:
    def __init__(self):
        self.keys = ['方角', 'バス・トイレ', 'キッチン', '放送・通信', '室内設備', '駐車場', '周辺環境', '契約期間']
    def fit(self,x,y):
        return self
    def transform(self,x):
        bufs = [[0 for i in range(len(self.keys))] for j in range(len(x.values))]
        for i in range(len(self.keys)):
            data = x[self.keys[i]].values
            for j in range(len(data)):
                if data[j] != data[j]:
                    bufs[j][i] = 1
        p = pd.DataFrame(bufs)
        p.index = x.index
        n_col = len(p.columns)
        col = []
        for i in range(n_col):
            col.append("isnan"+str(i))
        p.columns = col
        return pd.concat([x,p],axis = 1)