from sklearn.decomposition import NMF
import pandas as pd

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