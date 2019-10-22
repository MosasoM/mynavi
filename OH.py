import pandas as pd
class district_onehot:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        buf = [[0 for i in range(23)] for j in range(len(x.values))]
        data = x["mf_dist"].values
        for i in range(len(data)):
            buf[i][data[i]] = 1
        
        hoge = x.drop("mf_dist",axis=1)
        hgoe = hoge.drop("mf_city",axis=1)

        setubi = pd.DataFrame(buf)
        c_num = len(setubi.columns)
        col = []
        for i in range(c_num):
            col.append("dist_oh"+str(i))
        setubi.columns = col
        setubi.index = hoge.index
        hoge =  pd.concat([hoge,setubi],axis = 1)
        return hoge


class direction_onehot:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        buf = [[0 for i in range(8)] for j in range(len(x.values))]
        data = x["mf_angle"].values
        for i in range(len(data)):
            buf[i][data[i]] = 1
        
        hoge = x.drop("mf_angle",axis=1)

        setubi = pd.DataFrame(buf)
        c_num = len(setubi.columns)
        col = []
        for i in range(c_num):
            col.append("angle_oh"+str(i))
        setubi.columns = col
        setubi.index = hoge.index
        hoge =  pd.concat([hoge,setubi],axis = 1)
        return hoge

class structure_onehot:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        buf = [[0 for i in range(9)] for j in range(len(x.values))]
        data = x["mf_structure"].values
        for i in range(len(data)):
            buf[i][data[i]] = 1
        
        hoge = x.drop("mf_structure",axis=1)

        setubi = pd.DataFrame(buf)
        c_num = len(setubi.columns)
        col = []
        for i in range(c_num):
            col.append("structure_oh"+str(i))
        setubi.columns = col
        setubi.index = hoge.index
        hoge =  pd.concat([hoge,setubi],axis = 1)
        return hoge

