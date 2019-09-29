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

class drop_id:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        return x.drop("id",axis = 1)
class parse_area_size:
    def __init__(self,has_sq = True,is_log = False):
        self.add_sq = has_sq
        self.is_log = is_log
    def fit(self,x,y):
        return self
    def transform(self,x):
        hoge = x.copy()
        temp = x["面積"].values
        ans = [0 for i in range(len(temp))]
        is_large = [0 for i in range(len(temp))]
        for i in range(len(temp)):
            ans[i] = float(re.search(r"[0-9\.]+",temp[i])[0])
        hoge = hoge.drop("面積",axis = 1)
        if self.is_log:
            hoge = hoge.assign(areasize=np.log(ans))
        else:
            hoge = hoge.assign(areasize=np.round(ans))
        for i in range(len(temp)):
            if ans[i] > 70:
                is_large[i] = 1
        hoge = hoge.assign(is_large=is_large)
        if self.add_sq:
            hoge = hoge.assign(area_sq=np.round(np.sqrt(ans)))
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
            if int(year) > 63:
                year = "15"
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
                        break
            hoge = hoge.assign(year_cat=cat)        
#         hoge = hoge.assign(month= add_month)
        return hoge
    
    
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
                        break
            fuga = fuga.assign(istower = ans)
            is_high = [0 for i in range(len(tmp))]
            for i in range(len(tmp)):
                if where[i] > 9:
                    is_high[i] = 1
            fuga = fuga.assign(ishigh = is_high)
        return fuga


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


class dummy:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        return x
    def predict(self,x):
        return x

