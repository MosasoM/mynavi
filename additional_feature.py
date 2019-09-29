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
        self.encoder = OneHotEncoder(sparse=False,handle_unknown="ignore")

    def fit(self,x,y):
        self.encoder.fit(x["建物構造"].values.reshape(-1,1))
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
        tmp = pd.DataFrame(self.encoder.transform(x["建物構造"].values.reshape(-1,1)))
        c_num = len(tmp.columns)
        col = []
        for i in range(c_num):
            col.append("str"+str(i))
        tmp.columns = col
#         hoge = x.drop("建物構造",axis = 1)
        tmp.index = hoge.index
        return pd.concat([hoge,tmp],axis = 1)
        
#         return hoge


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