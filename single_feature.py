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
from sklearn.decomposition import PCA

from single import *
from cross import *

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
    
class ido_keido2xy:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        ido = x["ido"].values
        keido = x["keido"].values
        idox = [0 for i in range(len(ido))]
        idoy = [0 for i in range(len(ido))]
        for i in range(len(ido)):
            a,b = ido_calc_xy(ido[i],keido[i], 35.681236,139.767125)
            idox[i] = a
            idoy[i] = b
        hoge = x.assign(ido_x=idox)
        hoge = hoge.assign(ido_y=idoy)
        return hoge


class Seppen_pred:
    def __init__(self,rand_s):
        self.model1 = xgb.XGBRegressor(random_state=rand_s)
        self.model2 = xgb.XGBRegressor(random_state=rand_s)
    def fit(self,x,y):
        tar1 = x["avg_cross_pred"].values-np.array(y)
        tar2 = x["knn_area_price"].values-np.array(y)
        ex1 = x.drop(["avg_cross_pred","knn_area_price","avg_pred","knn_pred"],axis=1)
        ex2 = x.drop(["avg_cross_pred","knn_area_price","avg_pred","knn_pred"],axis=1)
        self.model1.fit(ex1,tar1)
        self.model2.fit(ex2,tar2)
        return self
    def transform(self,x):
        ex1 = x.drop(["avg_cross_pred","knn_area_price","avg_pred","knn_pred"],axis=1)
        ex2 = x.drop(["avg_cross_pred","knn_area_price","avg_pred","knn_pred"],axis=1)
        pred1 = self.model1.predict(ex1)
        pred2 = self.model2.predict(ex2)
        hoge = x.assign(diff_pred1=pred1)
        hoge = hoge.assign(diff_pred2=pred2)
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

