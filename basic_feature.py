import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestRegressor 
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from single import *
from cross import *
from stats_fea import *

class drop_id:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        return x.drop("id",axis = 1)
class parse_area_size:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        hoge = x.copy()
        temp = area_size_(x)
        hoge = hoge.assign(mf_areasize=temp)
        temp = area_size_sq_(x)
        hoge = hoge.assign(mf_area_sq=temp)
        return hoge

class parse_rooms:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        r,l,d,k,s = rldks_(x)
        hoge = x.copy()
        hoge = hoge.assign(mf_r = r)
        hoge = hoge.assign(mf_l = l)
        hoge = hoge.assign(mf_d = d)
        hoge = hoge.assign(mf_k = k)
        hoge = hoge.assign(mf_s = s)
        return hoge
    
class parse_how_old:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        year,month = how_old_(x)
        hoge = x.copy()
        hoge  = hoge.assign(mf_year=year)
        return hoge
    
    
class height_encoder:
    def __init__ (self,add_cat=True):
        self.add_cat = add_cat
    def fit(self,x,y):
        return self
    def transform(self,x):
        where,what=height_of_it_(x)
        fuga = x.copy()
        fuga = fuga.assign(mf_what_floor=where)
        fuga = fuga.assign(mf_height_bld=what)
        return fuga


class extract_district:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        dist,area = address_of_it_(x)
        hoge = x.copy()
        hoge = hoge.drop("所在地",axis = 1)
        hoge = hoge.assign(district=dist)
        hoge = hoge.assign(city = area)
        return hoge
    

class district_encoder:
    def __init__(self):
        pass
    def fit(self,x,y):
        self.dist_label = {}
        self.city_label = {}
        ind = 1
        for key in x["district"].unique():
            self.dist_label[key] = ind
            ind += 1
        ind = 1
        for key in x["city"].unique():
            self.city_label[key] = ind
            ind += 1
        return self
      
    def transform(self,x):
        temp = x["district"].values
        buf = [0 for i in range(len(temp))]
        hoge = x.drop("district",axis=1)
        for i in range(len(temp)):
            if temp[i] in self.dist_label:
                buf[i] = self.dist_label[temp[i]]
        hoge = hoge.assign(mf_dist=buf)
        
        buf = [0 for i in range(len(temp))]
        temp = x["city"].values
        hoge = hoge.drop("city",axis=1)
        for i in range(len(temp)):
            if temp[i] in self.city_label:
                buf[i] = self.city_label[temp[i]]
        hoge = hoge.assign(mf_city=buf)
        return hoge
    
    
class direction_encoder:
    def __init__(self):
        self.classi = {'北西': 0, '北東': 1, '北': 2, '南西': 3,
                       '南東': 4, '南': 5, '西': 6, '東': 7}
    def fit(self,x,y):
        return self
    def transform(self,x):
        temp = x["方角"]
        temp = temp.fillna("北")
        temp = temp.values
        ans = [0 for i in range(len(temp))]
        for i in range(len(temp)):
            if temp[i] in self.classi:
                ans[i] = self.classi[temp[i]]
        hoge = x.drop("方角",axis=1)
        hoge = hoge.assign(mf_angle = ans)
                    
        return hoge
    
class access_extractor:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        train,oth_train,oth_walk,avgwalk = train_and_walk_(x)
        hoge = hoge = x.drop("アクセス",axis = 1)
        hoge = hoge.assign(train=train)
        hoge = hoge.assign(train2=oth_train[0])
        hoge = hoge.assign(train3=oth_train[1])
        hoge = hoge.assign(walk= oth_walk[0])
        hoge = hoge.assign(walk2 = oth_walk[1])
        hoge = hoge.assign(walk3 = oth_walk[2])
        hoge = hoge.assign(avgwalk = avgwalk)
        return hoge
    
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
        hoge = hoge.assign(acr_sq=acr_sq)
        per,diff = relative_height_(x)
        hoge = hoge.assign(height_percent = per)
        hoge = hoge.assign(height_diff=diff)
        return hoge

            

        
class add_mean_dist_price: #くごとの家賃平均を追加。分散、中央値もたす。
    def __init__(self):
        self.means = {}
        self.mean_pad = 120000
        self.stds = {}
        self.std_pad = 50000
        self.medians = {}
        self.medi_pad = 90000
    def fit(self,x,y):
#         ty = pd.DataFrame(y)
#         ty.columns=["賃料"]
#         ty.index = x.index
#         temptemp = pd.concat([x,ty],axis = 1)
#         label = temptemp.groupby("mf_dist").mean().index.values
        
#         temp = np.round(temptemp.groupby("mf_dist").mean()["賃料"].values)
#         for i in range(len(label)):
#             self.means[label[i]] = temp[i]
#         self.mean_pad = round(np.mean(temp))
        
#         temp = np.round(temptemp.groupby("mf_dist").std()["賃料"].values)
#         for i in range(len(label)):
#             self.stds[label[i]] = temp[i]
#         self.std_pad = round(np.std(temp))
        
#         temp = np.round(temptemp.groupby("mf_dist").median()["賃料"].values)
#         for i in range(len(label)):
#             self.medians[label[i]] = temp[i]
#         self.medi_pad = round(np.median(temp))
        means,mean_pad,stds,std_pad,medians,medi_pad = fit_price_stats_(x,y,"mf_dist")
        self.means = means
        self.mean_pad = mean_pad
        self.stds = stds
        self.std_pad = std_pad
        self.medians = medians
        self.medi_pad = medi_pad
        
        return self
    def transform(self,x):
#         buf = [0 for i in range(len(x.values))]
#         temp = x["mf_dist"].values
#         for i in range(len(x.values)):
#             if temp[i] in self.means:
#                 buf[i] = self.means[temp[i]]
#             else:
#                 buf[i] = self.mean_pad
#         hoge = x.assign(d_p_mean =buf)
        
#         buf = [0 for i in range(len(x.values))]
#         for i in range(len(x.values)):
#             if temp[i] in self.stds:
#                 buf[i] = self.stds[temp[i]]
#             else:
#                 buf[i] = self.std_pad
#         hoge = x.assign(d_p_std =buf)
        
        
#         buf = [0 for i in range(len(x.values))]
#         for i in range(len(x.values)):
#             if temp[i] in self.medians:
#                 buf[i] = self.medians[temp[i]]
#             else:
#                 buf[i] = self.medi_pad
#         hoge = x.assign(d_p_medi =buf)
        b_mean,b_std,b_medi = transform_price_stats_(x,"mf_dist",self.means,self.mean_pad,self.stds,self.std_pad,self.medians,self.medi_pad)
        hoge = x.copy()
        hoge = hoge.assign(dist_p_mean=b_mean)
        hoge = hoge.assign(dist_p_std=b_std)
        hoge = hoge.assign(dist_p_medi=b_medi)
        return hoge
    
class add_mean_angle_price: #方角の家賃平均を追加。分散、中央値もたす。
    def __init__(self):
        self.means = {}
        self.mean_pad = 120000
        self.stds = {}
        self.std_pad = 50000
        self.medians = {}
        self.medi_pad = 90000
    def fit(self,x,y):
        ty = pd.DataFrame(y)
        ty.columns=["賃料"]
        ty.index = x.index
        temptemp = pd.concat([x,ty],axis = 1)
        label = temptemp.groupby("mf_angle").mean().index.values
        
        temp = np.round(temptemp.groupby("mf_angle").mean()["賃料"].values)
        for i in range(len(label)):
            self.means[label[i]] = temp[i]
        self.mean_pad = round(np.mean(temp))
        
        temp = np.round(temptemp.groupby("mf_angle").std()["賃料"].values)
        for i in range(len(label)):
            self.stds[label[i]] = temp[i]
        self.std_pad = round(np.std(temp))
        
        temp = np.round(temptemp.groupby("mf_angle").median()["賃料"].values)
        for i in range(len(label)):
            self.medians[label[i]] = temp[i]
        self.medi_pad = round(np.median(temp))
        
        return self
    def transform(self,x):
        buf = [0 for i in range(len(x.values))]
        temp = x["mf_angle"].values
        for i in range(len(x.values)):
            if temp[i] in self.means:
                buf[i] = self.means[temp[i]]
            else:
                buf[i] = self.mean_pad
        hoge = x.assign(angle_p_mean =buf)
        
        buf = [0 for i in range(len(x.values))]
        for i in range(len(x.values)):
            if temp[i] in self.stds:
                buf[i] = self.stds[temp[i]]
            else:
                buf[i] = self.std_pad
        hoge = x.assign(angle_p_std =buf)
        
        
        buf = [0 for i in range(len(x.values))]
        for i in range(len(x.values)):
            if temp[i] in self.medians:
                buf[i] = self.medians[temp[i]]
            else:
                buf[i] = self.medi_pad
        hoge = x.assign(angle_p_medi =buf)
        return hoge
    
class add_mean_structure_price: #方角の家賃平均を追加。分散、中央値もたす。
    def __init__(self):
        self.means = {}
        self.mean_pad = 120000
        self.stds = {}
        self.std_pad = 50000
        self.medians = {}
        self.medi_pad = 90000
    def fit(self,x,y):
        ty = pd.DataFrame(y)
        ty.columns=["賃料"]
        ty.index = x.index
        temptemp = pd.concat([x,ty],axis = 1)
        label = temptemp.groupby("mf_structure").mean().index.values
        
        temp = np.round(temptemp.groupby("mf_structure").mean()["賃料"].values)
        for i in range(len(label)):
            self.means[label[i]] = temp[i]
        self.mean_pad = round(np.mean(temp))
        
        temp = np.round(temptemp.groupby("mf_structure").std()["賃料"].values)
        for i in range(len(label)):
            self.stds[label[i]] = temp[i]
        self.std_pad = round(np.std(temp))
        
        temp = np.round(temptemp.groupby("mf_structure").median()["賃料"].values)
        for i in range(len(label)):
            self.medians[label[i]] = temp[i]
        self.medi_pad = round(np.median(temp))
        
        return self
    def transform(self,x):
        buf = [0 for i in range(len(x.values))]
        temp = x["mf_structure"].values
        for i in range(len(x.values)):
            if temp[i] in self.means:
                buf[i] = self.means[temp[i]]
            else:
                buf[i] = self.mean_pad
        hoge = x.assign(angle_p_mean =buf)
        
        buf = [0 for i in range(len(x.values))]
        for i in range(len(x.values)):
            if temp[i] in self.stds:
                buf[i] = self.stds[temp[i]]
            else:
                buf[i] = self.std_pad
        hoge = x.assign(angle_p_std =buf)
        
        
        buf = [0 for i in range(len(x.values))]
        for i in range(len(x.values)):
            if temp[i] in self.medians:
                buf[i] = self.medians[temp[i]]
            else:
                buf[i] = self.medi_pad
        hoge = x.assign(angle_p_medi =buf)
        return hoge


    
            


class train_encoder:
    def __init__(self):
        self.train_dic = None
    def fit(self,x,y):
        train_kind = set()
        for key in x["train"].unique():
            train_kind.add(key)
        for key in x["train2"].unique():
            train_kind.add(key)
        for key in x["train3"].unique():
            train_kind.add(key)
        train_dic = {}
        ind = 0
        for key in train_kind:
            train_dic[key] = ind
            ind += 1
        self.train_dic = train_dic
        return self
    def transform(self,x):
        temp = [[100 for i in range(len(self.train_dic))] for j in range(len(x.values))]
        moyori = [0 for i in range(len(x.values))]
        fuga = x["train"].values
        piyo = x["walk"].values
        train_dic = self.train_dic
        for i in range(len(x["train"].values)):
            key = fuga[i]
            if key in self.train_dic:
                temp[i][train_dic[key]] = piyo[i]
           
                
        fuga = x["train2"].values
        piyo = x["walk2"].values
        for i in range(len(x["train2"].values)):
            key = fuga[i]
            if key in self.train_dic:
                temp[i][train_dic[key]] = min(piyo[i],temp[i][train_dic[key]])
                
        fuga = x["train3"].values
        piyo = x["walk3"].values
        for i in range(len(x["train3"].values)):
            key = fuga[i]
            if key in self.train_dic:
                temp[i][train_dic[key]] = min(piyo[i],temp[i][train_dic[key]])
        temp = pd.DataFrame(temp)
        col = []
        c_num = len(temp.columns)
        for i in range(c_num):
            col.append("train_walk_"+str(i))
        temp.columns = col
        hoge = x.copy()
        temp.index = hoge.index
        hoge = pd.concat([hoge,temp],axis = 1)
        
        train_freq = make_freq_elem(x["train"],self.train_dic)
        temp = x["train"].values
        t_f = [0 for i in range(len(temp))]
        for i in range(len(temp)):
            if temp[i] in train_freq:
                t_f[i] = train_freq[temp[i]]
        hoge = hoge.assign(train_freq=t_f)
            

        hoge = hoge.drop("train",axis = 1)
        hoge = hoge.drop("train2",axis = 1)
        hoge = hoge.drop("train3",axis = 1)
        
        
        return hoge





class parking_encoder:
    def __init__(self):
        self.exist = [re.compile(r"駐輪場.+?有"),re.compile(r"駐車場.+?有"),re.compile(r"バイク置き場.+?有")]
    def fit(self,x,y):
        return self
    def transform(self,x):
        temp = x["駐車場"].values
        d = [[0 for i in range(3)] for j in range(len(temp))]
        parking_cost = [0 for i in range(len(temp))]
        parking_dist = [0 for i in range(len(temp))]
        for i in range(len(temp)):
            if temp[i] != temp[i]:
                continue
            for j in range(len(self.exist)):
                if self.exist[j].search(temp[i]):
                    d[i][j] += 1
        setubi = pd.DataFrame(d)
        c_num = len(setubi.columns)
        col = []
        for i in range(c_num):
            col.append("park"+str(i))
        setubi.columns = col
        hoge = x.drop("駐車場",axis = 1)
        setubi.index = hoge.index
        hoge =  pd.concat([hoge,setubi],axis = 1)
        
        pat = re.compile(r"駐車場.+近隣.+円.+?m")
        pat2 = re.compile(r"駐車場\t+?近隣")
        p2 = re.compile(r"[0-9\,]+?円")
        p3 = re.compile(r"\,")
        p4 = re.compile(r"距離[0-9]+?m")
        kinrin = [0 for i in range(len(temp))]
        for i in range(len(temp)):
            if temp[i]!=temp[i] or d[i][1] == 1:
                continue
#             cost = 0
            dist = 0
            m = pat.search(temp[i])
            if m:
                kinrin[i] = 1
                txt = m[0]
#                 cost = p2.search(txt)[0]
#                 cost = p3.sub("",cost)
#                 cost = int(cost[:-1])
                dist = p4.search(txt)[0]
                dist = int(dist[2:-1])
            else:
                m = pat2.search(temp[i])
                if m:
                    kinrin[i] = 1
#                     cost = 20000
                    dist = 200
#             parking_cost[i] = cost
            parking_dist[i] = dist
#         hoge = hoge.assign(p_cost=parking_cost)
        hoge = hoge.assign(p_dist=parking_dist)
        hoge = hoge.assign(kinrin=kinrin)
        
        return hoge



    
class info_encoder:
    def __init__(self):
        self.keys = {'インターネット対応': 0, 'CATV': 1, 'CSアンテナ': 2, 'BSアンテナ': 3,
                     '光ファイバー': 4, '高速インターネット': 5, 'インターネット使用料無料': 6,"有線放送":7}
    def fit(self,x,y):
        return self
    def transform(self,x):
        temp = x["放送・通信"].values
        setubi = [[0 for i in range(len(self.keys))] for j in range(len(temp))]
        pat = re.compile(r"／")
        for i in range(len(temp)):
            if temp[i] != temp[i]:
                continue
            else:
                block = temp[i].split()
                for b in block:
                    if pat.sub("",b) in self.keys:
                        setubi[i][self.keys[pat.sub("",b)]] = 1
        setubi = pd.DataFrame(setubi)
        c_num = len(setubi.columns)
        col = []
        for i in range(c_num):
            col.append("info"+str(i))
        setubi.columns = col
        hoge = x.drop("放送・通信",axis = 1)
        setubi.index = hoge.index
        hoge =  pd.concat([hoge,setubi],axis = 1)
        
#         info_freq = make_freq_elem(x["放送・通信"],self.keys)
#         temp = x["放送・通信"].values
#         t_f = [0 for i in range(len(temp))]
#         for i in range(len(temp)):
#             if temp[i] in info_freq:
#                 t_f[i] = info_freq[temp[i]]
#         hoge = hoge.assign(info_freq=t_f)
        return hoge



        
class drop_unnecessary:
    def __init__(self):
        self.to_drop = []
        self.valid = ['id', '賃料', '所在地', 'アクセス', '間取り', '築年数', '方角', '面積', '所在階', 'バス・トイレ',
       'キッチン', '放送・通信', '室内設備', '駐車場', '周辺環境', '建物構造', '契約期間',"train"]
        self.pat = []
    def fit(self,x,y):
        return self
    def transform(self,x):
        tmp = x.drop(self.to_drop,axis = 1)
        for name in self.valid:
            if name in tmp.columns:
                tmp = tmp.drop(name,axis = 1)
        return tmp

    
class parse_contract_time:
    def __init__(self):
        self.teiki_pat = re.compile(r".*\t.*")
        self.year_pat = re.compile(r"[0-9]+年間")
        self.month_pat = re.compile(r"[0-9]+ヶ月間")
        self.due_year_pat = re.compile(r"[0-9]+年")
        self.due_month_pat = re.compile(r"[0-9]+月まで")
        self.double_pat = re.compile(r"[0-9]+年[0-9]+ヶ月間")
    def fit(self,x,y):
        return self
    def transform(self,x):
        hoge = x.copy()
        temp = x["契約期間"].values
        isteiki = [0 for i in range(len(temp))]
        add_year = [0 for i in range(len(temp))]
        add_month = [0 for i in range(len(temp))]
        for i in range(len(temp)):
            if not temp[i] == temp[i]:
                add_year[i] = 2
                add_month[i] = 0   
                continue
            if self.teiki_pat.match(temp[i]):
                isteiki[i] = 1
            if self.double_pat.match(temp[i]):
                year = self.due_year_pat.search(temp[i])[0][:-1]
                month = self.month_pat.search(temp[i])[0][:-3]
                add_year[i] = int(year)
                add_month[i] = int(month)
            else:
                if self.due_month_pat.search(temp[i]):
                    year = self.due_year_pat.search(temp[i])[0][:-1]
                    month = self.due_month_pat.search(temp[i])[0][:-3]
                    year = int(year)-2019
                    month = int(month)-9
                    if month < 0:
                        year -= 1
                        month += 12
                    add_year[i] = int(year)
                    add_month[i] = int(month)
                else:
                    if self.year_pat.match(temp[i]):
                        year = self.year_pat.match(temp[i])[0][:-2]
                        month = 0
                    else:
                        year = 0
                        month = self.month_pat.match(temp[i])[0][:-3]
                    add_year[i] = int(year)
                    add_month[i] = int(month)
        hoge = hoge.drop(["契約期間"],axis = 1)
        hoge = hoge.assign(is_teiki=isteiki)
        hoge = hoge.assign(cont_year= add_year)
        hoge = hoge.assign(cont_month= add_month)
        return hoge
class fac_encoder:
    def __init__(self):
        self.keys = {'エアコン付': 0, 'シューズボックス': 1, 'バルコニー': 2, 'フローリング': 3,
                     '室内洗濯機置場': 4, '敷地内ごみ置き場': 5, 'エレベーター': 6, '公営水道': 7,
                     '下水': 8, '都市ガス': 9, 'タイル張り': 10, 'ウォークインクローゼット': 11, '2面採光': 12,
                     '24時間換気システム': 13, '3面採光': 14, 'ペアガラス': 15, '専用庭': 16, '水道その他': 17,
                     '冷房': 18, 'クッションフロア': 19, '床暖房': 20, 'プロパンガス': 21, 'ロフト付き': 22,
                     '出窓': 23, 'トランクルーム': 24, 'オール電化': 25, 'ルーフバルコニー': 26, '室外洗濯機置場': 27,
                     '床下収納': 28, 'バリアフリー': 29, '防音室': 30, '二重サッシ': 31, '洗濯機置場なし': 32}
    def fit(self,x,y):
        return self
    def transform(self,x):
        temp = x["室内設備"].values
        setubi = [[0 for i in range(len(self.keys))] for j in range(len(temp))]
        pat = re.compile(r"／")
        for i in range(len(temp)):
            if temp[i] != temp[i]:
                continue
            else:
                block = temp[i].split()
                for b in block:
                    if pat.sub("",b) in self.keys:
                        setubi[i][self.keys[pat.sub("",b)]] = 1
        setubi = pd.DataFrame(setubi)
        c_num = len(setubi.columns)
        col = []
        for i in range(c_num):
            col.append("fac"+str(i))
        setubi.columns = col
        hoge = x.drop("室内設備",axis = 1)
        setubi.index = hoge.index
        hoge = pd.concat([hoge,setubi],axis = 1)
    
#         info_freq = make_freq_elem(x["室内設備"],self.keys)
#         temp = x["室内設備"].values
#         t_f = [0 for i in range(len(temp))]
#         for i in range(len(temp)):
#             if temp[i] in info_freq:
#                 t_f[i] = info_freq[temp[i]]
#         hoge = hoge.assign(fac_freq=t_f)
        return hoge
    
class structure_label_encoder:
    def __init__(self):
        self.classi = {'ブロック':0, '木造':1,'軽量鉄骨':2,'鉄骨造':3,'ALC（軽量気泡コンクリート）':4,
                      'RC（鉄筋コンクリート）':5,'SRC（鉄骨鉄筋コンクリート）':6, 'その他':5,
                      'PC（プレキャスト・コンクリート（鉄筋コンクリート））':7,'HPC（プレキャスト・コンクリート（重量鉄骨））':8}

    def fit(self,x,y):
        return self
    def transform(self,x):
        temp = x["建物構造"].values
        ans = [0 for i in range(len(temp))]
        for i in range(len(temp)):
            if temp[i] in self.classi:
                ans[i] = self.classi[temp[i]]
            else:
                ans[i] = 5
        hoge = x.drop("建物構造",axis=1)
        hoge = hoge.assign(mf_structure=ans)
        return hoge
    
class bath_encoder:
    def __init__(self):
        self.keys = {'専用バス':0,'専用トイレ':1,'バス・トイレ別':2,'シャワー':3,
                     '浴室乾燥機':4,'温水洗浄便座':5,'洗面台独立':6,'脱衣所':7,
                     '追焚機能':8}
    def fit(self,x,y):
        return self
    def transform(self,x):
        temp = x["バス・トイレ"].values
        setubi = [[0 for i in range(len(self.keys))] for j in range(len(temp))]
        pat = re.compile(r"／")
        for i in range(len(temp)):
            if temp[i] != temp[i]:
                continue
            else:
                block = temp[i].split()
                for b in block:
                    if pat.sub("",b) in self.keys:
                        setubi[i][self.keys[pat.sub("",b)]] = 1
        setubi = pd.DataFrame(setubi)
        c_num = len(setubi.columns)
        col = []
        for i in range(c_num):
            col.append("bath"+str(i))
        setubi.columns = col
        hoge = x.drop("バス・トイレ",axis = 1)
        setubi.index = hoge.index
        hoge =  pd.concat([hoge,setubi],axis = 1)
    
#         info_freq = make_freq_elem(x["バス・トイレ"],self.keys)
#         temp = x["バス・トイレ"].values
#         t_f = [0 for i in range(len(temp))]
#         for i in range(len(temp)):
#             if temp[i] in info_freq:
#                 t_f[i] = info_freq[temp[i]]
#         hoge = hoge.assign(bath_freq=t_f)
        return hoge
    
class kitchin_encoder:
    def __init__(self):
        self.keys = {'ガスコンロ': 0, 'コンロ2口': 1, 'システムキッチン': 2, '給湯': 3, '独立キッチン': 4,
                     'コンロ3口': 5, 'IHコンロ': 6, 'コンロ1口': 7, '冷蔵庫あり': 8, 'コンロ設置可': 9,
                     'カウンターキッチン': 10, 'L字キッチン': 11, '電気コンロ': 12, 'コンロ4口以上': 13}
    def fit(self,x,y):
        return self
    def transform(self,x):
        temp = x["キッチン"].values
        setubi = [[0 for i in range(len(self.keys))] for j in range(len(temp))]
        pat = re.compile(r"／")
        p2 = re.compile(r"コンロ設置可.*")
        for i in range(len(temp)):
            if temp[i] != temp[i]:
                        continue
            else:
                block = temp[i].split()
                for b in block:
                    f = pat.sub("",b)
                    if p2.match(f):
                        f = "コンロ設置可"
                    if f in self.keys:
                        setubi[i][self.keys[f]] = 1
        setubi = pd.DataFrame(setubi)
        c_num = len(setubi.columns)
        col = []
        for i in range(c_num):
            col.append("kit"+str(i))
        setubi.columns = col
        hoge = x.drop("キッチン",axis = 1)
        setubi.index = hoge.index
        hoge =  pd.concat([hoge,setubi],axis = 1)
        
#         info_freq = make_freq_elem(x["キッチン"],self.keys)
#         temp = x["キッチン"].values
#         t_f = [0 for i in range(len(temp))]
#         for i in range(len(temp)):
#             if temp[i] in info_freq:
#                 t_f[i] = info_freq[temp[i]]
#         hoge = hoge.assign(kit_freq=t_f)
        return hoge
    
class env_encoder:
    def __init__(self):
        self.keys = {'【小学校】': 0, '【大学】': 1, '【公園】': 2, '【飲食店】': 3,
                     '【スーパー】': 4, '【コンビニ】': 5, '【ドラッグストア】': 6, '【郵便局】': 7,
                     '【病院】': 8, '【図書館】': 9, '【銀行】': 10, '【学校】': 11, '【幼稚園・保育園】': 12,
                     '【総合病院】': 13, '【デパート】': 14}
    def fit(self,x,y):
        return self
    
    def transform(self,x):
        temp = x["周辺環境"].values
        setubi = [[0 for i in range(len(self.keys))] for j in range(len(temp))]
        pat = re.compile(r"／")
        p2 = re.compile("【.*】")
        for i in range(len(temp)):
            if temp[i] != temp[i]:
                continue
            else:
                block = temp[i].split()
                for b in block:
                    key = pat.sub("",b)
                    if p2.match(key):
                        if key in self.keys:
                            setubi[i][self.keys[key]] = 1                 
        setubi = pd.DataFrame(setubi)
        c_num = len(setubi.columns)
        col = []
        for i in range(c_num):
            col.append("env"+str(i))
        setubi.columns = col
        hoge = x.drop("周辺環境",axis = 1)
        setubi.index = hoge.index
        hoge = pd.concat([hoge,setubi],axis = 1)
    
#         info_freq = make_freq_elem(x["室内設備"],self.keys)
#         temp = x["室内設備"].values
#         t_f = [0 for i in range(len(temp))]
#         for i in range(len(temp)):
#             if temp[i] in info_freq:
#                 t_f[i] = info_freq[temp[i]]
#         hoge = hoge.assign(fac_freq=t_f)
        
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

