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

from single import *
from cross import *

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
        ind = 0
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
        hoge = x.copy()
        for i in range(len(temp)):
            if temp[i] in self.dist_label:
                buf[i] = self.dist_label[temp[i]]
        hoge = hoge.assign(mf_dist=buf)
        
        buf = [0 for i in range(len(temp))]
        temp = x["city"].values
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
        temp = [[30 for i in range(len(self.train_dic))] for j in range(len(x.values))]
        onehot = [[0 for i in range(len(self.train_dic))] for j in range(len(x.values))]
        moyori = [0 for i in range(len(x.values))]
        fuga = x["train"].values
        piyo = x["walk"].values
        train_dic = self.train_dic
        for i in range(len(x["train"].values)):
            key = fuga[i]
            if key in self.train_dic:
                moyori[i] = train_dic[key]+1
        
        for i in range(len(x["train"].values)):
            key = fuga[i]
            if key in self.train_dic:
                temp[i][train_dic[key]] = piyo[i]
                onehot[i][train_dic[key]] = 1
           
                
        fuga = x["train2"].values
        piyo = x["walk2"].values
        for i in range(len(x["train2"].values)):
            key = fuga[i]
            if key in self.train_dic:
                temp[i][train_dic[key]] = min(piyo[i],temp[i][train_dic[key]])
                onehot[i][train_dic[key]] = 1
                
        fuga = x["train3"].values
        piyo = x["walk3"].values
        for i in range(len(x["train3"].values)):
            key = fuga[i]
            if key in self.train_dic:
                temp[i][train_dic[key]] = min(piyo[i],temp[i][train_dic[key]])
                onehot[i][train_dic[key]] = 1
        
        temp = pd.DataFrame(temp)
        col = []
        c_num = len(temp.columns)
        for i in range(c_num):
            col.append("train_walk_"+str(i))
        temp.columns = col
        hoge = x.copy()
        temp.index = hoge.index
        hoge = pd.concat([hoge,temp],axis = 1)

        temp = pd.DataFrame(onehot)
        col = []
        c_num = len(temp.columns)
        for i in range(c_num):
            col.append("train_OH_"+str(i))
        temp.columns = col
        temp.index = hoge.index
        hoge = pd.concat([hoge,temp],axis = 1)
        
        train_freq = make_freq_elem(x["train"],self.train_dic)
        temp = x["train"].values
        t_f = [0 for i in range(len(temp))]
        for i in range(len(temp)):
            if temp[i] in train_freq:
                t_f[i] = train_freq[temp[i]]
        hoge = hoge.assign(train_freq=t_f)
            
        hoge = hoge.assign(moyori=moyori)

        hoge = hoge.drop("train",axis = 1)
        hoge = hoge.drop("train2",axis = 1)
        hoge = hoge.drop("train3",axis = 1)
        
        
        return hoge





class parking_encoder:
    def __init__(self):
        self.exist = [re.compile(r"駐輪場\t.{2,}有"),re.compile(r"駐車場\t.{2,}有"),re.compile(r"バイク置き場\t.{2,}有")]
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
            dist = 0
            m = pat.search(temp[i])
            if m:
                kinrin[i] = 1
                txt = m[0]
                dist = p4.search(txt)[0]
                dist = int(dist[2:-1])
            else:
                m = pat2.search(temp[i])
                if m:
                    kinrin[i] = 1
                    dist = 200
            parking_dist[i] = dist
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
        
        return hoge



        
class drop_unnecessary:
    def __init__(self):
        self.to_drop = []
        self.valid = ['id', '賃料', '所在地', 'アクセス', '間取り', '築年数', '方角', '面積', '所在階', 'バス・トイレ',
       'キッチン', '放送・通信', '室内設備', '駐車場', '周辺環境', '建物構造', '契約期間',"train","city","district"]
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
                '室内洗濯機置場': 4, '敷地内ごみ置き場': 5, 'エレベーター': 6, '都市ガス': 7, 'タイル張り': 8,
                 'ウォークインクローゼット': 9, '2面採光': 10,
                '24時間換気システム': 11, '3面採光': 12, 'ペアガラス': 13, '専用庭': 14,
                '冷房': 0, 'クッションフロア': 15, '床暖房': 16, 'プロパンガス': 17, 'ロフト付き': 18,
                '出窓': 19, 'トランクルーム': 20, 'オール電化': 21, 'ルーフバルコニー': 22, '室外洗濯機置場': 23,
                '床下収納': 24, 'バリアフリー': 25, '防音室': 26, '二重サッシ': 27, '洗濯機置場なし': 28}
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
    
        return hoge
    
class kitchin_encoder:
    def __init__(self):
        # self.keys = {
        #     'ガスコンロ':0, 'コンロ設置可（コンロ1口）':1, 'コンロ設置可（コンロ3口）':2, '給湯':3,
        #      'コンロ設置可（コンロ2口）':4, 'コンロ4口以上':5, 'L字キッチン':6, '電気コンロ':7,
        #       '冷蔵庫あり':8, 'コンロ設置可（コンロ4口以上）':9, 'IHコンロ':10, 'コンロ3口':11,
        #        '独立キッチン':12, 'カウンターキッチン':13, 'コンロ1口':14, 'コンロ設置可（口数不明）':15,
        #         'コンロ2口':16, 'システムキッチン':17
        # }
        self.keys = {
            'コンロ設置可（コンロ4口以上）':0, 'システムキッチン':1, 'ガスコンロ':2,
             'コンロ設置可（コンロ3口）':3, 'コンロ設置可（コンロ2口）':4, 'L字キッチン':5,
              '冷蔵庫あり':6, 'IHコンロ':7, 'コンロ3口':8, 'コンロ1口':9
              , 'コンロ設置可（コンロ1口）':10, '給湯':11, 'コンロ4口以上':12, '電気コンロ':13
              , '独立キッチン':14, 'コンロ2口':15, 'カウンターキッチン':16, 'コンロ設置可（口数不明）':17
        }
    def fit(self,x,y):
        return self
    def transform(self,x):
        temp = x["キッチン"].values
        setubi = [[0 for i in range(len(self.keys))] for j in range(len(temp))]
        pat = re.compile(r"／")
        # p2 = re.compile(r"コンロ設置可.*")
        for i in range(len(temp)):
            if temp[i] != temp[i]:
                        continue
            else:
                block = temp[i].split()
                for b in block:
                    f = pat.sub("",b)
                    # if p2.match(f):
                    #     f = "コンロ設置可"
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
                        txt = p2.match(key)[0]
                        if txt in self.keys:
                            setubi[i][self.keys[key]] += 1                 
        setubi = pd.DataFrame(setubi)
        c_num = len(setubi.columns)
        col = []
        for i in range(c_num):
            col.append("envv"+str(i))
        setubi.columns = col
        hoge = x.drop("周辺環境",axis = 1)
        setubi.index = hoge.index
        hoge = pd.concat([hoge,setubi],axis = 1)
        
        temp = x["周辺環境"].values
        setubi = [[10000 for i in range(len(self.keys))] for j in range(len(temp))]
        pat = re.compile(r"／")
        p2 = re.compile("【.*】")
        p3 = re.compile("[0-9]+?m")
        for i in range(len(temp)):
            if temp[i] != temp[i]:
                continue
            else:
                block = temp[i].split()
                for b in block:
                    key = pat.sub("",b)
                    if p2.match(key):
                        txt = p2.match(key)[0]
                        if p3.search(key):
                            dist = p3.search(key)[0][:-1]
                            if txt in self.keys:
                                setubi[i][self.keys[key]] = min(int(dist),setubi[i][self.keys[key]])       
        setubi = pd.DataFrame(setubi)
        c_num = len(setubi.columns)
        col = []
        for i in range(c_num):
            col.append("env_dist"+str(i))
        setubi.columns = col
        setubi.index = hoge.index
        hoge = pd.concat([hoge,setubi],axis = 1)
        

        
        return hoge


class add_mean_dist_price: #くごとの家賃平均を追加。分散、中央値もたす。
    def __init__(self):
        self.means = {}
        self.mean_pad = 120000
        self.stds = {}
        self.std_pad = 50000
        self.medians = {}
        self.medi_pad = 90000
        self.maxs = {}
        self.max_pad = 1000000
        self.mins = {}
        self.min_pad = 0
    def fit(self,x,y):
        means,mean_pad,stds,std_pad,medians,medi_pad,maxs,max_pad,mins,min_pad = fit_price_stats_(x,y,"mf_dist")
        self.means = means
        self.mean_pad = mean_pad
        self.stds = stds
        self.std_pad = std_pad
        self.medians = medians
        self.medi_pad = medi_pad
        self.maxs = maxs
        self.max_pad =max_pad
        self.mins=mins
        self.min_pad = min_pad
        
        return self
    def transform(self,x):
        b_mean,b_std,b_medi,b_max,b_min = transform_price_stats_(x,"mf_dist",self.means,self.mean_pad,self.stds,self.std_pad,self.medians,self.medi_pad,self.maxs,self.max_pad,self.mins,self.min_pad)
        hoge = x.copy()
        hoge = hoge.assign(dist_p_mean=b_mean)
        # hoge = hoge.assign(dist_p_std=b_std)
        hoge = hoge.assign(dist_p_medi=b_medi)
        # hoge = hoge.assign(dist_p_max=b_max)
        # hoge = hoge.assign(dist_p_min=b_min)
        # temp = np.array(b_max)-np.array(b_min)
        # hoge = hoge.assign(dist_mm= temp)
        return hoge
    
class add_mean_angle_price: #方角の家賃平均を追加。分散、中央値もたす。
    def __init__(self):
        self.means = {}
        self.mean_pad = 120000
        self.stds = {}
        self.std_pad = 50000
        self.medians = {}
        self.medi_pad = 90000
        self.maxs = {}
        self.max_pad = 1000000
        self.mins = {}
        self.min_pad = 0
    def fit(self,x,y):
        means,mean_pad,stds,std_pad,medians,medi_pad,maxs,max_pad,mins,min_pad = fit_price_stats_(x,y,"mf_angle")
        self.means = means
        self.mean_pad = mean_pad
        self.stds = stds
        self.std_pad = std_pad
        self.medians = medians
        self.medi_pad = medi_pad
        self.maxs = maxs
        self.max_pad =max_pad
        self.mins=mins
        self.min_pad = min_pad
        
        return self
    def transform(self,x):
        b_mean,b_std,b_medi,b_max,b_min = transform_price_stats_(x,"mf_angle",self.means,self.mean_pad,self.stds,self.std_pad,self.medians,self.medi_pad,self.maxs,self.max_pad,self.mins,self.min_pad)
        hoge = x.copy()
        hoge = hoge.assign(ang_p_mean=b_mean)
        # hoge = hoge.assign(dist_p_std=b_std)
        hoge = hoge.assign(ang_p_medi=b_medi)
        # hoge = hoge.assign(ang_p_max=b_max)
        # hoge = hoge.assign(ang_p_min=b_min)
        # temp = np.array(b_max)-np.array(b_min)
        # hoge = hoge.assign(ang_mm= temp)
        return hoge
    
class add_mean_structure_price: #方角の家賃平均を追加。分散、中央値もたす。
    def __init__(self):
        self.means = {}
        self.mean_pad = 120000
        self.stds = {}
        self.std_pad = 50000
        self.medians = {}
        self.medi_pad = 90000
        self.maxs = {}
        self.max_pad = 1000000
        self.mins = {}
        self.min_pad = 0
    def fit(self,x,y):
        means,mean_pad,stds,std_pad,medians,medi_pad,maxs,max_pad,mins,min_pad = fit_price_stats_(x,y,"mf_structure")
        self.means = means
        self.mean_pad = mean_pad
        self.stds = stds
        self.std_pad = std_pad
        self.medians = medians
        self.medi_pad = medi_pad
        self.maxs = maxs
        self.max_pad =max_pad
        self.mins=mins
        self.min_pad = min_pad
        
        return self
    def transform(self,x):
        b_mean,b_std,b_medi,b_max,b_min = transform_price_stats_(x,"mf_structure",self.means,self.mean_pad,self.stds,self.std_pad,self.medians,self.medi_pad,self.maxs,self.max_pad,self.mins,self.min_pad)
        hoge = x.copy()
        hoge = hoge.assign(str_p_mean=b_mean)
        # hoge = hoge.assign(dist_p_std=b_std)
        hoge = hoge.assign(str_p_medi=b_medi)
        # hoge = hoge.assign(str_p_max=b_max)
        # hoge = hoge.assign(str_p_min=b_min)
        # temp = np.array(b_max)-np.array(b_min)
        # hoge = hoge.assign(str_mm= temp)
        return hoge


class add_mean_walk_price: #方角の家賃平均を追加。分散、中央値もたす。
    def __init__(self):
        self.means = {}
        self.mean_pad = 120000
        self.stds = {}
        self.std_pad = 50000
        self.medians = {}
        self.medi_pad = 90000
        self.maxs = {}
        self.max_pad = 1000000
        self.mins = {}
        self.min_pad = 0
    def fit(self,x,y):
        means,mean_pad,stds,std_pad,medians,medi_pad,maxs,max_pad,mins,min_pad = fit_price_stats_(x,y,"walk")
        self.means = means
        self.mean_pad = mean_pad
        self.stds = stds
        self.std_pad = std_pad
        self.medians = medians
        self.medi_pad = medi_pad
        self.maxs = maxs
        self.max_pad =max_pad
        self.mins=mins
        self.min_pad = min_pad
        
        return self
    def transform(self,x):
        b_mean,b_std,b_medi,b_max,b_min = transform_price_stats_(x,"walk",self.means,self.mean_pad,self.stds,self.std_pad,self.medians,self.medi_pad,self.maxs,self.max_pad,self.mins,self.min_pad)
        hoge = x.copy()
        hoge = hoge.assign(walk_p_mean=b_mean)
        # hoge = hoge.assign(dist_p_std=b_std)
        hoge = hoge.assign(walk_p_medi=b_medi)
        # hoge = hoge.assign(walk_p_max=b_max)
        # hoge = hoge.assign(walk_p_min=b_min)
        # temp = np.array(b_max)-np.array(b_min)
        # hoge = hoge.assign(walk_mm= temp)
        return hoge
    
class add_moyori_walk_price: #方角の家賃平均を追加。分散、中央値もたす。
    def __init__(self):
        self.means = {}
        self.mean_pad = 120000
        self.stds = {}
        self.std_pad = 50000
        self.medians = {}
        self.medi_pad = 90000
        self.maxs = {}
        self.max_pad = 1000000
        self.mins = {}
        self.min_pad = 0
    def fit(self,x,y):
        means,mean_pad,stds,std_pad,medians,medi_pad,maxs,max_pad,mins,min_pad = fit_price_stats_(x,y,"moyori")
        self.means = means
        self.mean_pad = mean_pad
        self.stds = stds
        self.std_pad = std_pad
        self.medians = medians
        self.medi_pad = medi_pad
        self.maxs = maxs
        self.max_pad =max_pad
        self.mins=mins
        self.min_pad = min_pad
        
        return self
    def transform(self,x):
        b_mean,b_std,b_medi,b_max,b_min = transform_price_stats_(x,"moyori",self.means,self.mean_pad,self.stds,self.std_pad,self.medians,self.medi_pad,self.maxs,self.max_pad,self.mins,self.min_pad)
        hoge = x.copy()
        hoge = hoge.assign(moyo_p_mean=b_mean)
        # hoge = hoge.assign(dist_p_std=b_std)
        hoge = hoge.assign(moyo_p_medi=b_medi)
        # hoge = hoge.assign(moyo_p_max=b_max)
        # hoge = hoge.assign(moyo_p_min=b_min)
        # temp = np.array(b_max)-np.array(b_min)
        # hoge = hoge.assign(moyo_mm= temp)
        return hoge



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

        # s,c = rldk_fea_(x)
        # hoge = hoge.assign(rldk_sum=s)
        # hoge = hoge.assign(rldk_cross=c)
        
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
        # hoge = x.assign(pred_area_pre_price=pred)
        hoge = x.copy()
        hoge = hoge.assign(avg_cross_pred = hoge["mf_areasize"].values*np.array(pred))
        return hoge

class Knn_regression:
    def __init__(self,k):
        self.model = KNeighborsRegressor(n_neighbors=k,weights="distance")
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
        # hoge = x.assign(knn_pred=pred)
        hoge = x.copy()
        temp = pred*x["mf_areasize"].values
        hoge = hoge.assign(knn_area_price=temp)
        return hoge

class Knn_area:
    def __init__(self,k):
        self.model = KNeighborsRegressor(n_neighbors=k,weights="distance")
    def fit(self,x,y):
        ex_var = x[["ido","keido"]].values
        ex_var = zscore(ex_var)
        ty = x["mf_areasize"].values
        self.model.fit(ex_var,ty)
        return self
    def transform(self,x):
        ex_var = x[["ido","keido"]].values
        ex_var = zscore(ex_var)
        pred = self.model.predict(ex_var)
        hoge = x.assign(knn_pred_area=pred)
        hoge = hoge.assign(knn_area_diff = pred-x["mf_areasize"].values)
        return hoge



class NMF_train_walk:
    def __init__(self,rand_s):
        self.cols = []
        self.model = NMF(n_components=20, init='nndsvd', random_state=rand_s)
        # self.model = NMF(init='nndsvd', random_state=rand_s)
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
        # self.model = NMF(init='nndsvd', random_state=rand_s)
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
        # self.model = NMF(init='nndsvd', random_state=rand_s)
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
        # self.model = NMF(init='nndsvd', random_state=rand_s)
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
        # self.model = NMF(init='nndsvd', random_state=rand_s)
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
        # self.model = NMF(n_components=5, init='nndsvd', random_state=rand_s)
        self.model = NMF(init='nndsvd', random_state=rand_s)
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
        # self.model = NMF(init='nndsvd', random_state=rand_s)
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

class pre_predict:
    def __init__(self,rand_s):
        self.model1 = xgb.XGBRegressor(random_state=rand_s)
        self.model2 = xgb.XGBRegressor(random_state=rand_s)
    def fit(self,x,y):
        self.model1.fit(x,y)
        pred = self.model1.predict(x)
        diff = np.array(pred)-np.array(y)
        self.model2.fit(x,diff)
        return self
    def transform(self,x):
        pred = self.model2.predict(x)
        return x.assign(pre_diff = pred)


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

class homes_in_nkm:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        r = 500
        resolution = 50
        
        ido = x["ido"].values
        keido = x["keido"].values
        pos_x = [0 for i in range(len(ido))]
        pos_y = [0 for i in range(len(ido))]
        buf =  [0 for i in range(len(ido))]
        for i in range(len(ido)):
            a,b = ido_calc_xy(ido[i],keido[i],35.681236,139.767125)
            pos_x[i] = a
            pos_y[i] = b
        grid_num = 40000//resolution
        bucket = [[0 for i in range(grid_num)] for j in range(grid_num)]
        around = (2*r)//resolution
        for i in range(len(ido)):
            a = int(pos_x[i]//resolution)+grid_num//2
            b = int(pos_y[i]//resolution)+grid_num//2
            bucket[a][b] += 1
        for i in range(len(ido)):
            a = int(pos_x[i]//resolution)+grid_num//2
            b = int(pos_y[i]//resolution)+grid_num//2
            temp = 0
            for j in range(around+1):
                for k in range(around+1):
                    temp += bucket[a+j-around//2][b+k-around//2]
            buf[i] = temp


        return x.assign(house_in_1km=buf)

class annoy_all:
    def __init__(self):
        self.model = AnnoyIndex(62+9+8,metric='angular')
        self.col = []
        col = []
        for i in range(7):
            col.append("info"+str(i))
        for i in range(10):
            col.append("env_dist_nmf"+str(i))
        for i in range(10):
            col.append("kit_nmf"+str(i))
        for i in range(15):
            col.append("fac_nmf"+str(i))  
        for i in range(20):
            col.append("tr_wa_nmf"+str(i))      
        for i in range(8):
            col.append("bath"+str(i))
        col.append("mf_areasize")
        col.append("mf_year")
        col.append("mf_r")
        col.append("mf_l")
        col.append("mf_d")
        col.append("mf_k")
        col.append("mf_s")
        col.append("ido")
        col.append("keido")
        self.col = col
        self.price = None
    def fit(self,x,y):
        self.price = np.array(y)
        data = x[self.col].values
        data = zscore(data)
        for i in range(len(data)):
            self.model.add_item(i,data[i])
        self.model.build(10)
        return self
    def transform(self,x):
        data = x[self.col].values
        data = zscore(data)
        buf = [0 for i in range(len(data))]
        for i in range(len(data)):
            ind = self.model.get_nns_by_vector(data[i],1)
            buf[i] = self.price[ind[0]]
        return x.assign(annoy_pred=buf)

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

class rldk_label:
    def __init__(self):
        self.tags={
            '1K': 0, '1R': 1, '2LDK': 2, '2DK': 3, '1DK': 4,
            '1LDK': 5, '3LDK': 6, '3DK': 7, '1LDK+S(納戸)': 8,
            '4K': 9, '2K': 10, '1K+S(納戸)': 11, '4LDK': 12,
            '3LDK+S(納戸)': 13, '5LDK+S(納戸)': 14, '5LDK': 15,
            '3K': 16, '4DK': 17, '2LDK+S(納戸)': 18, '2DK+S(納戸)': 19,
            '4LDK+S(納戸)': 20, '5DK': 21, '3DK+S(納戸)': 22,
            '1DK+S(納戸)': 23, '5K': 24, '6LDK': 25, '2K+S(納戸)': 26,
            '1LK+S(納戸)': 27, '5DK+S(納戸)': 28, '3K+S(納戸)': 29
        }
    def fit(self,x,y):
        return self
    def transform(self,x):
        data = x["間取り"].values
        buf = [0 for i in range(len(data))]
        for i in range(len(data)):
            buf[i] = self.tags[data[i]]
        return x.assign(madori_label=buf)

class rldk_dist_label:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        buf = [0 for i in range(len(x.values))]
        madori = x["madori_label"].values
        dist = x["mf_dist"].values
        for i in range(len(x.values)):
            buf[i] = madori[i]*23+dist[i]
        return x.assign(rldk_dist_label=buf)


class add_rldk_dist_price: #くごとの家賃平均を追加。分散、中央値もたす。
    def __init__(self):
        self.means = {}
        self.mean_pad = 120000
        self.stds = {}
        self.std_pad = 50000
        self.medians = {}
        self.medi_pad = 90000
        self.maxs = {}
        self.max_pad = 1000000
        self.mins = {}
        self.min_pad = 0
    def fit(self,x,y):
        means,mean_pad,stds,std_pad,medians,medi_pad,maxs,max_pad,mins,min_pad = fit_price_stats_(x,y,"rldk_dist_label")
        self.means = means
        self.mean_pad = mean_pad
        self.stds = stds
        self.std_pad = std_pad
        self.medians = medians
        self.medi_pad = medi_pad
        self.maxs = maxs
        self.max_pad =max_pad
        self.mins=mins
        self.min_pad = min_pad
        
        return self
    def transform(self,x):
        b_mean,b_std,b_medi,b_max,b_min = transform_price_stats_(x,"rldk_dist_label",self.means,self.mean_pad,self.stds,self.std_pad,self.medians,self.medi_pad,self.maxs,self.max_pad,self.mins,self.min_pad)
        hoge = x.copy()
        hoge = hoge.assign(rldk_dist_p_mean=b_mean)
        hoge = hoge.assign(rldk_dist_p_medi=b_medi)
        return hoge

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
