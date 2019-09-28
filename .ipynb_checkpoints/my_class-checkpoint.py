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

class drop_id:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        return x.drop("id",axis = 1)
class parse_area_size:
    def __init__(self,has_sq = True):
        self.add_sq = has_sq
    def fit(self,x,y):
        return self
    def transform(self,x):
        hoge = x.copy()
        temp = x["面積"].values
        ans = [0 for i in range(len(temp))]
        for i in range(len(temp)):
            ans[i] = float(re.search(r"[0-9\.]+",temp[i])[0])
        hoge = hoge.drop("面積",axis = 1)
        hoge = hoge.assign(areasize=ans)
        hoge = hoge.assign(zyou = np.array(ans)/1.82)
        if self.add_sq:
            hoge = hoge.assign(area_sq=np.sqrt(ans))
        return hoge
class parse_rooms:
    def __init__(self,area_par_room=True):
        self.add_par = area_par_room
    def fit(self,x,y):
        return self
    def transform(self,x):
        hoge = x.copy()
        temp = x["間取り"].values
        room = [0 for i in range(len(temp))]
        head = ["L","D","K","S"]
        setubi = [[0 for i in range(len(temp))] for j in range(4)]
        for i in range(len(temp)):
            room[i] = int(temp[i][0])
            for j in range(4):
                if head[j] in temp[i]:
                    setubi[j][i] = 1
        hoge = hoge.drop("間取り",axis = 1)
        hoge = hoge.assign(room = room)
        hoge = hoge.assign(L = setubi[0])
        hoge = hoge.assign(D = setubi[1])
        hoge = hoge.assign(K = setubi[2])
        hoge = hoge.assign(S = setubi[3])
        if self.add_par:
            temp = x["areasize"].values/hoge["room"].values
            hoge = hoge.assign(apr=temp)
        return hoge
class parse_how_old:
    def __init__(self,has_category=True):
        self.year_pat = re.compile(r"[0-9]+年")
        self.month_pat = re.compile(r"[0-9]+ヶ月")
        self.add_cat = has_category
    def fit(self,x,y):
        return self
    def transform(self,x):
        hoge = x.copy()
        temp = x["築年数"].values
        add_year = [0 for i in range(len(temp))]
        add_month = [0 for i in range(len(temp))]
        for i in range(len(temp)):
            year = self.year_pat.search(temp[i])
            month = self.month_pat.search(temp[i])
            if year:
                year = year[0][:-1]
            else:
                year = 0
            if month:
                month = month[0][:-2]
            else:
                month = 0
            add_year[i] = int(year)
            add_month[i] = int(month)
        hoge = hoge.drop(["築年数"],axis = 1)
        hoge  = hoge.assign(year=add_year)
        if self.add_cat:
            thre = [5,10,20,100000000]
            cat = [0 for i in range(len(temp))]
            for i in range(len(temp)):
                y = add_year[i]
                for j in range(len(thre)):
                    if y <= thre[j]:
                        cat[i] = j
                        continue
            hoge = hoge.assign(year_cat=cat)        
#         hoge = hoge.assign(month= add_month)
        return hoge
    
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
        


class structure_label_encoder:
    def __init__(self):
        self.classi = {'ブロック':0, '木造':0,'軽量鉄骨':1,'鉄骨造':1,'ALC（軽量気泡コンクリート）':1,
                      'RC（鉄筋コンクリート）':2,'SRC（鉄骨鉄筋コンクリート）':2, 'その他':2,
                      'PC（プレキャスト・コンクリート（鉄筋コンクリート））':2,'HPC（プレキャスト・コンクリート（重量鉄骨））':2}
#         self.encoder = OneHotEncoder(sparse=False,handle_unknown="ignore")

    def fit(self,x,y):
#         self.encoder.fit(x["建物構造"].values.reshape(-1,1))
        return self
    def transform(self,x):
        temp = x["建物構造"].values
        ans = [0 for i in range(len(temp))]
        for i in range(len(temp)):
            if temp[i] in self.classi:
                ans[i] = self.classi[temp[i]]
            else:
                ans[i] = 2
        hoge = x.drop("建物構造",axis = 1)
        hoge = hoge.assign(structure=ans)
        return hoge
#         tmp = pd.DataFrame(self.encoder.transform(x["建物構造"].values.reshape(-1,1)))
#         c_num = len(tmp.columns)
#         col = []
#         for i in range(c_num):
#             col.append("str"+str(i))
#         tmp.columns = col
#         hoge = x.drop("建物構造",axis = 1)
#         tmp.index = hoge.index
#         return pd.concat([hoge,tmp],axis = 1)
    
class height_encoder:
    def __init__ (self,add_cat=True):
        self.add_cat = add_cat
    def fit(self,x,y):
        return self
    def transform(self,x):
        fuga = x.copy()
        tmp = x["所在階"].values
        where = [0 for i in range(len(tmp))]
        what = [0 for i in range(len(tmp))]
        for i in range(len(tmp)):
            try:
                hoge =  tmp[i].split("／")
            except:
                hoge = ["2階","5階建て"]
            if len(hoge) == 2:
                if hoge[0] == "":
                    hoge[0] = "2階"
                if hoge[1] == "":
                    hoge[1] = "3階建て"
                x = int(re.search(r"[0-9]+",hoge[0])[0])
                y = int(re.search(r"[0-9]+",hoge[1])[0])
            else:
                x = 2
                y = 3
            where[i] = x
            what[i] = y
        fuga = fuga.drop("所在階",axis = 1)
        fuga = fuga.assign(what_floor=where)
        fuga = fuga.assign(height_bld=what)
        fuga = fuga.assign(height_percent = np.array(where)/np.array(what))
        if self.add_cat:
            thre = [20,6,0]
            ans = [0 for i in range(len(tmp))]
            for i in range(len(tmp)):
                for j in range(len(thre)):
                    if what[i] >= thre[j]:
                        ans[i] = 2-j
            fuga = fuga.assign(istower = ans)
        return fuga

class direction_encoder:
    def __init__(self):
#         self.encoder = OneHotEncoder(sparse=False,handle_unknown="ignore")
        self.classi = {'北西': -3, '北東': -1, '北': -4, '南西': 1, '南東': 3, '南': 4, '西': -2, '東': 2}
    def fit(self,x,y):
#         self.encoder.fit(x["方角"].fillna("南").values.reshape(-1,1))
        return self
    def transform(self,x):
        temp = x["方角"].values
#         temp = temp.fillna("南東")
        ans = [0 for i in range(len(temp))]
        for i in range(len(temp)):
            if temp[i] in self.classi:
                ans[i] = self.classi[temp[i]]
            else:
                ans[i] = 1.5
        hoge = x.drop("方角",axis = 1)
        hoge = hoge.assign(direction=ans)
        return hoge
#         tmp = pd.DataFrame(self.encoder.transform(tmp.values.reshape(-1,1)))
#         c_num = len(tmp.columns)
#         col = []
#         for i in range(c_num):
#             col.append("dir"+str(i))
#         tmp.columns = col
#         hoge = x.drop("方角",axis = 1)
#         tmp.index = hoge.index
#         return pd.concat([hoge,tmp],axis = 1)
class extract_district:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        pat = re.compile(r"東京都.+区")
        dist = []
        tmp = x["所在地"].values
        for i in range(len(tmp)):
            m = pat.search(tmp[i])
            dist.append(m[0][3:-1])
        hoge = x.copy()
        hoge = hoge.drop("所在地",axis = 1)
        hoge = hoge.assign(district=dist)
        return hoge
            
class district_encoder:
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown="ignore",sparse=False)
    def fit(self,x,y):
        self.encoder.fit(x["district"].values.reshape(-1,1))
        return self
    def transform(self,x):
        tmp = pd.DataFrame(self.encoder.transform(x["district"].values.reshape(-1,1)))
        c_num = len(tmp.columns)
        col = []
        for i in range(c_num):
            col.append("dist"+str(i))
        tmp.columns = col
        hoge = x.drop("district",axis = 1)
        tmp.index = hoge.index
        return pd.concat([hoge,tmp],axis = 1)
class access_extractor:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        tmp = x["アクセス"].values
        train = ["" for i in range(len(tmp))]
        walk = [0 for i in range(len(tmp))]
        avgwalk = [0 for i in range(len(tmp))]
        for i in range(len(tmp)):
            train[i] = re.search(r".+?(線|ライン|ライナー|プレス|かもめ)",tmp[i])[0]
            walk[i] = int(re.search(r"徒歩[0-9]+?分",tmp[i])[0][2:-1])
            avg = 0
            ind = 0
            for m in re.finditer(r"徒歩[0-9]+分",tmp[i]):
                avg += int(m[0][2:-1])
                ind += 1
            if ind == 0:
                ind = 1
            avg = avg/ind
            avgwalk[i] = avg
        hoge = hoge = x.drop("アクセス",axis = 1)
        hoge = hoge.assign(train=train)
        hoge = hoge.assign(walk= walk)
        hoge = hoge.assign(avgwalk = avgwalk)
        return hoge

class train_encoder:
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown="ignore",sparse=False)
    def fit(self,x,y):
        self.encoder.fit(x["train"].values.reshape(-1,1))
        return self
    def transform(self,x):
        tmp = pd.DataFrame(self.encoder.transform(x["train"].values.reshape(-1,1)))
        c_num = len(tmp.columns)
        col = []
        for i in range(c_num):
            col.append("train"+str(i))
        tmp.columns = col
        hoge = x.drop("train",axis = 1)
        tmp.index = hoge.index
        return pd.concat([hoge,tmp],axis = 1)

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
        return pd.concat([hoge,setubi],axis = 1)

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
        return pd.concat([hoge,setubi],axis = 1)

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
        return pd.concat([hoge,setubi],axis = 1)

class info_encoder:
    def __init__(self):
        self.keys = {'インターネット対応': 0, 'CATV': 1, 'CSアンテナ': 2, 'BSアンテナ': 3,
                     '光ファイバー': 4, '高速インターネット': 5, 'インターネット使用料無料': 6}
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
        return pd.concat([hoge,setubi],axis = 1)

class parking_encoder:
    def __init__(self):
        self.exist = [re.compile(r"駐輪場\t空有"),re.compile(r"駐車場\t空有"),re.compile(r"バイク置き場\t空有")]
    def fit(self,x,y):
        return self
    def transform(self,x):
        temp = x["駐車場"].values
        d = [[0 for i in range(3)] for j in range(len(temp))]
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
        return pd.concat([hoge,setubi],axis = 1)

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
        return pd.concat([hoge,setubi],axis = 1)

class dummy:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def predict(self,x):
        return x
        
class drop_unnecessary:
    def __init__(self):
        self.to_drop = []
        self.valid = ['id', '賃料', '所在地', 'アクセス', '間取り', '築年数', '方角', '面積', '所在階', 'バス・トイレ',
       'キッチン', '放送・通信', '室内設備', '駐車場', '周辺環境', '建物構造', '契約期間']
        self.pat = []
    def fit(self,x,y):
        return self
    def transform(self,x):
        tmp = x.drop(self.to_drop,axis = 1)
        for name in self.valid:
            if name in tmp.columns:
                tmp = tmp.drop(name,axis = 1)
        return tmp

class normalize:
    def __init__(self):
        self.numeric = [""]
    def fit(self,x,y):
        return self
    def transform(self,x):
        return x