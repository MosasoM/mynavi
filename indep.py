import numpy as np
import re
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
import math
from scipy.stats import zscore

def area_size_(x):
    temp = x["面積"].values
    ans = [0 for i in range(len(temp))]
    for i in range(len(temp)):
        ans[i] = float(re.search(r"[0-9\.]+",temp[i])[0])
    return ans

def area_size_sq_(x):
    temp = x["面積"].values
    ans = [0 for i in range(len(temp))]
    for i in range(len(temp)):
        ans[i] = float(re.search(r"[0-9\.]+",temp[i])[0])
    return np.power(ans,2)

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

def rldks_(x):
    temp = x["間取り"].values
    room = [0 for i in range(len(temp))]
    head = ["L","D","K","S"]
    setubi = [[0 for i in range(len(temp))] for j in range(4)]
    for i in range(len(temp)):
        room[i] = int(temp[i][0])
        for j in range(4):
            if head[j] in temp[i]:
                setubi[j][i] = 1
    r = np.array(room)
    l = np.array(setubi[0])
    d = np.array(setubi[1])
    k = np.array(setubi[2])
    s = np.array(setubi[3])
    return r,l,d,k,s

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

def how_old_(x):
    temp = x["築年数"].values
    add_year = [0 for i in range(len(temp))]
    add_month = [0 for i in range(len(temp))]
    year_pat = re.compile(r"[0-9]+年")
    month_pat = re.compile(r"[0-9]+ヶ月")
    for i in range(len(temp)):
        year = year_pat.search(temp[i])
        month = month_pat.search(temp[i])
        if re.search(r"新築",temp[i]):
            year = -1
            month = 0
        else:
            if year:
                year = year[0][:-1]
            else:
                year = 10
            if month:
                month = month[0][:-2]
            else:
                month = 0
            if int(year) > 100:
                year = "15"
            add_year[i] = int(year)
            add_month[i] = int(month)
    return add_year,add_month

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

def height_of_it_(x):
    tmp = x["所在階"].values
    where = [0 for i in range(len(tmp))]
    what = [0 for i in range(len(tmp))]
    has_under = [0 for i in range(len(tmp))]
    all_of_bld = [0 for i in range(len(tmp))]
    u_pat = re.compile(r"地下")
    not_all_pat = re.compile(r"／")

    for i in range(len(tmp)):
        if tmp[i] != tmp[i]:
            where[i] = -1
            what[i] = -1
            continue
        if not_all_pat.search(tmp[i]):
            hoge =  tmp[i].split("／")
            if hoge[0] == "":
                hoge[0] = "2"
            if hoge[1] == "":
                hoge[1] = "3"
            x = int(re.search(r"[0-9]+",hoge[0])[0])
            y = int(re.search(r"[0-9]+",hoge[1])[0])
        else:
            all_of_bld[i] = 1
            x = int(re.search(r"[0-9]+",tmp[i])[0])
            y = x
        if u_pat.search(tmp[i]):
            has_under[i] = 1
        where[i] = x
        what[i] = y

    return where,what,has_under,all_of_bld

class height_encoder:
    def __init__ (self,add_cat=True):
        self.add_cat = add_cat
    def fit(self,x,y):
        return self
    def transform(self,x):
        where,what,has_under,all_of_bld=height_of_it_(x)
        fuga = x.copy()
        fuga = fuga.assign(mf_what_floor=where)
        fuga = fuga.assign(mf_height_bld=what)
        fuga = fuga.assign(has_under=has_under)
        return fuga

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
        hoge = x.copy()
        hoge = hoge.assign(mf_angle = ans)
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
        hoge = x.copy()
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
        # hoge = hoge.assign(p_dist=parking_dist)
        # hoge = hoge.assign(kinrin=kinrin)
        
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
        hoge = x.copy()
        setubi.index = hoge.index
        hoge =  pd.concat([hoge,setubi],axis = 1)

        temp = np.sum(setubi,axis=1)
        hoge = hoge.assign(info_sum=temp)
        
        return hoge

class parse_contract_time:
    def __init__(self):
        self.teiki_pat = re.compile(r".*定期借家.*")
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
        hoge = hoge.copy()
        hoge = hoge.assign(is_teiki=isteiki)
        hoge = hoge.assign(cont_year= add_year)
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
        hoge = x.copy()
        setubi.index = hoge.index
        hoge = pd.concat([hoge,setubi],axis = 1)

        temp = np.sum(setubi,axis=1)
        hoge = hoge.assign(fac_sum=temp)
    
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
        hoge = x.copy()
        setubi.index = hoge.index
        hoge =  pd.concat([hoge,setubi],axis = 1)
        return hoge
    
class kitchin_encoder:
    def __init__(self):
        self.keys = {
            'ガスコンロ':0, 'コンロ設置可（コンロ1口）':1, 'コンロ設置可（コンロ3口）':2, '給湯':3,
             'コンロ設置可（コンロ2口）':4, 'コンロ4口以上':5, 'L字キッチン':6, '電気コンロ':7,
              '冷蔵庫あり':8, 'コンロ設置可（コンロ4口以上）':9, 'IHコンロ':10, 'コンロ3口':11,
               '独立キッチン':12, 'カウンターキッチン':13, 'コンロ1口':14, 'コンロ設置可（口数不明）':15,
                'コンロ2口':16, 'システムキッチン':17
        }
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
        hoge = x.copy()
        setubi.index = hoge.index
        hoge =  pd.concat([hoge,setubi],axis = 1)

        temp = np.sum(setubi,axis=1)
        hoge = hoge.assign(kit_sum=temp)
        
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
        hoge = x.copy()
        setubi.index = hoge.index
        hoge = pd.concat([hoge,setubi],axis = 1)

        # temp = np.sum(setubi,axis=1)
        # hoge = hoge.assign(env_sum=temp)
        
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

class middle_class_centers:
    def __init__(self):
        self.ks = KMeans(n_clusters=8,random_state=7777)
        self.centers = None
    def fit(self,x,y):
        fuga = pd.DataFrame(y.values)
        fuga.columns=["price_ff"]
        fuga.index=x.index
        hoge = pd.concat([x,fuga],axis=1)
        hoge = hoge.query("200000<price_ff<300000")
        
        coord = hoge[["ido","keido"]].values
        tar = hoge["price_ff"].values

        self.ks.fit(coord,tar)
        self.centers = self.ks.cluster_centers_
        return self
    def transform(self,x):
        ido = x["ido"].values
        keido = x["keido"].values
        buf = [[0 for i in range(len(self.centers))] for j in range(len(x.values))]
        for i in range(len(x.values)):
            for j in range(len(self.centers)):
                to_ido = self.centers[j][0]
                to_keido = self.centers[j][1]
                buf[i][j] = google_distance(ido[i],keido[i],to_ido,to_keido)

        dist = pd.DataFrame(buf)
        dist.index = x.index
        col = []
        for i in range(len(self.centers)):
            col.append("dist_middle_center"+str(i))
        dist.columns = col
        hoge = x.copy()
        shortest = np.amin(buf,axis=1)
        hoge = x.assign(shortest_to_middle=shortest)
        middle_label = np.argmin(buf,axis=1)
        hoge = hoge.assign(middle_label=middle_label)
        return pd.concat([hoge,dist],axis=1)


class heigh_class_center:
    def __init__(self):
        self.center =  None
    def fit(self,x,y):
        fuga = pd.DataFrame(y.values)
        fuga.columns=["price_ff"]
        fuga.index=x.index
        hoge = pd.concat([x,fuga],axis=1)
        hoge = hoge.query("price_ff>300000")

        coord = hoge[["ido","keido"]].values
        tar = hoge["price_ff"]

        self.center = np.average(coord,axis=0,weights=tar)
        return self
    def transform(self,x):
        ido = x["ido"].values
        keido = x["keido"].values
        buf = [0 for i in range(len(x.values))]
        for i in range(len(x.values)):
            to_ido = self.center[0]
            to_keido = self.center[1]
            buf[i] = google_distance(ido[i],keido[i],to_ido,to_keido)
        return x.assign(dist_high_center=buf)

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

class drop_unnecessary:
    def __init__(self):
        self.to_drop = []
        self.valid = ['id', '賃料', '所在地', 'アクセス', '間取り', '築年数', '方角', '面積', '所在階', 'バス・トイレ',
       'キッチン', '放送・通信', '室内設備', '駐車場', '周辺環境', '建物構造', '契約期間',"train","district","city"]
        self.pat = []
    def fit(self,x,y):
        return self
    def transform(self,x):
        tmp = x.drop(self.to_drop,axis = 1)
        for name in self.valid:
            if name in tmp.columns:
                tmp = tmp.drop(name,axis = 1)
        return tmp

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

class drop_for_linear:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        to_drop = [
            "cont_year",
            "moyori",
        ]
        for col in x.columns:
            if "env_dist" in col:
                to_drop.append(col)
        return x.drop(to_drop,axis=1)