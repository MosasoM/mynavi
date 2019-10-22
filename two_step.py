import numpy as np
import re
import pandas as pd

def make_freq_elem(x_column_not_value,elems):
    ans = {}
    for key in elems.keys():
        temp = x_column_not_value.str.contains(key).sum()
        ans[key] = temp
    return ans

def address_of_it_(x):
    pat = re.compile(r"東京都.+区")
    p2 = re.compile(r"区.+?[0-9|０-９]丁目")
    p3 = re.compile(r"区.+?町")
    p4 = re.compile(r"区.+?[0-9|０-９]")
    dist = ["" for i in range(len(x.values))]
    area = ["" for i in range(len(x.values))]
    tmp = x["所在地"].values
    for i in range(len(tmp)):
        m = pat.search(tmp[i])
        dist[i] = m[0][3:]
        m = p2.search(tmp[i])
        if m:
            area[i] = m[0][1:-3]
        else:
            m = p3.search(tmp[i])
            if m:
                area[i] = m[0][1:]
            else:
                m = p4.search(tmp[i])
                if m:
                    area[i] = m[0][1:-1]
    return dist,area

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
        # hoge = hoge.drop("city",axis=1)
        for i in range(len(temp)):
            if temp[i] in self.city_label:
                buf[i] = self.city_label[temp[i]]
        hoge = hoge.assign(mf_city=buf)
        return hoge
        
##########################################################


def train_and_walk_(x):
    tmp = x["アクセス"].values
    train = ["" for i in range(len(tmp))]
    oth_train = [["" for i in range(len(tmp))] for j in range(2)]
    avgwalk = [0 for i in range(len(tmp))]
    oth_walk = [[30 for i in range(len(tmp))] for j in range(3)]
    for i in range(len(tmp)):
        train[i] = re.match(r".+?(線|ライン|ライナー|プレス|かもめ)",tmp[i])[0]
        avg = 0
        ind = 0
        for m in re.finditer(r"(徒歩|バス.*?)[0-9]+分",tmp[i]):
            if "バス" in m[0]:
                ind += 1
                if ind > 2:
                    break
                continue
            avg += int(m[0][2:-1])
            oth_walk[ind][i] = int(m[0][2:-1])
            ind += 1
            if ind > 2:
                break
        if ind == 0:
            ind = 1
        avg = avg/ind
        avgwalk[i] = avg
        ind = 0
        for m in re.finditer(r"\t\t.+?(線|ライン|ライナー|プレス|かもめ)",tmp[i]):
            oth_train[ind][i] = m[0][2:]
            ind += 1
            if ind > 1:
                break
    return train,oth_train,oth_walk,avgwalk

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
        # onehot = [[0 for i in range(len(self.train_dic))] for j in range(len(x.values))]
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
                # onehot[i][train_dic[key]] = 1
           
                
        fuga = x["train2"].values
        piyo = x["walk2"].values
        for i in range(len(x["train2"].values)):
            key = fuga[i]
            if key in self.train_dic:
                temp[i][train_dic[key]] = min(piyo[i],temp[i][train_dic[key]])
                # onehot[i][train_dic[key]] = 1
                
        fuga = x["train3"].values
        piyo = x["walk3"].values
        for i in range(len(x["train3"].values)):
            key = fuga[i]
            if key in self.train_dic:
                temp[i][train_dic[key]] = min(piyo[i],temp[i][train_dic[key]])
                # onehot[i][train_dic[key]] = 1
        
        temp = pd.DataFrame(temp)
        col = []
        c_num = len(temp.columns)
        for i in range(c_num):
            col.append("train_walk_"+str(i))
        temp.columns = col
        hoge = x.copy()
        temp.index = hoge.index
        hoge = pd.concat([hoge,temp],axis = 1)

        # temp = pd.DataFrame(onehot)
        # col = []
        # c_num = len(temp.columns)
        # for i in range(c_num):
        #     col.append("train_OH_"+str(i))
        # temp.columns = col
        # temp.index = hoge.index
        # hoge = pd.concat([hoge,temp],axis = 1)
        
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