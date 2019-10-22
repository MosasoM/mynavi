import numpy as np
import re
import pandas as pd

def fit_price_stats_(x,y,category_col):
    ty = pd.DataFrame(y)
    ty.columns=["賃料"]
    ty.index = x.index
    temptemp = pd.concat([x,ty],axis = 1)
    label = temptemp.groupby(category_col).mean().index.values
    
    means = {}
    mean_pad = 120000
    stds = {}
    std_pad = 50000
    medians = {}
    medi_pad = 90000
    maxs = {}
    max_pad = 1000000
    mins = {}
    min_pad = 1000000
        
    temp = np.round(temptemp.groupby(category_col).mean()["賃料"].values)
    for i in range(len(label)):
        means[label[i]] = temp[i]
    mean_pad = round(np.mean(temp))
        
    temp = np.round(temptemp.groupby(category_col).std()["賃料"].values)
    for i in range(len(label)):
        stds[label[i]] = temp[i]
    std_pad = round(np.std(temp))
        
    temp = np.round(temptemp.groupby(category_col).median()["賃料"].values)
    for i in range(len(label)):
        medians[label[i]] = temp[i]
    medi_pad = round(np.median(temp))

    temp = np.round(temptemp.groupby(category_col).max()["賃料"].values)
    for i in range(len(label)):
        maxs[label[i]] = temp[i]
    max_pad = round(np.max(temp))

    temp = np.round(temptemp.groupby(category_col).min()["賃料"].values)
    for i in range(len(label)):
        mins[label[i]] = temp[i]
    min_pad = round(np.min(temp))
    
    return means,mean_pad,stds,std_pad,medians,medi_pad,maxs,max_pad,mins,min_pad



def transform_price_stats_(x,category_col,means,mean_pad,stds,std_pad,medians,medi_pad,maxs,max_pad,mins,min_pad):
    buf1 = [0 for i in range(len(x.values))]
    temp = x[category_col].values
    for i in range(len(x.values)):
        if temp[i] in means:
            buf1[i] = means[temp[i]]
        else:
            buf1[i] = mean_pad
        
    buf2 = [0 for i in range(len(x.values))]
    for i in range(len(x.values)):
        if temp[i] in stds:
            buf2[i] = stds[temp[i]]
        else:
            buf2[i] = std_pad
        
        
    buf3 = [0 for i in range(len(x.values))]
    for i in range(len(x.values)):
        if temp[i] in medians:
            buf3[i] = medians[temp[i]]
        else:
            buf3[i] = medi_pad

    buf4 = [0 for i in range(len(x.values))]
    for i in range(len(x.values)):
        if temp[i] in maxs:
            buf4[i] = maxs[temp[i]]
        else:
            buf4[i] = max_pad

    buf5 = [0 for i in range(len(x.values))]
    for i in range(len(x.values)):
        if temp[i] in mins:
            buf5[i] = mins[temp[i]]
        else:
            buf5[i] = min_pad
    
    return buf1,buf2,buf3,buf4,buf5

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
        hoge = hoge.assign(dist_p_max=b_max)
        hoge = hoge.assign(dist_p_min=b_min)
        temp = np.array(b_max)-np.array(b_min)
        hoge = hoge.assign(dist_mm= temp)
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
        hoge = hoge.assign(ang_p_max=b_max)
        hoge = hoge.assign(ang_p_min=b_min)
        temp = np.array(b_max)-np.array(b_min)
        hoge = hoge.assign(ang_mm= temp)
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
        hoge = hoge.assign(str_p_max=b_max)
        hoge = hoge.assign(str_p_min=b_min)
        temp = np.array(b_max)-np.array(b_min)
        hoge = hoge.assign(str_mm= temp)
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
        hoge = hoge.assign(walk_p_max=b_max)
        hoge = hoge.assign(walk_p_min=b_min)
        temp = np.array(b_max)-np.array(b_min)
        hoge = hoge.assign(walk_mm= temp)
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
        hoge = hoge.assign(moyo_p_max=b_max)
        hoge = hoge.assign(moyo_p_min=b_min)
        temp = np.array(b_max)-np.array(b_min)
        hoge = hoge.assign(moyo_mm= temp)
        return hoge

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