import re
import numpy as np
import pandas as pd

def apr_(x):
    num_space = x["mf_r"].values+x["mf_l"].values+np.maximum(x["mf_d"].values,x["mf_k"].values)+x["mf_s"].values
    num_space = np.array(num_space)
    apr = x["mf_areasize"].values/x["mf_r"].values
    apr_all = x["mf_areasize"].values/num_space
    return apr,apr_all

def apr_sq_(x):
    num_space = x["mf_r"].values+x["mf_l"].values+np.maximum(x["mf_d"].values,x["mf_k"].values)+x["mf_s"].values
    num_space = np.array(num_space)
    apr = x["mf_area_sq"].values/x["mf_r"].values
    apr_all = x["mf_area_sq"].values/num_space
    return apr,apr_all

def acr_(x):
    num_space =x["mf_r"].values+x["mf_l"].values+np.maximum(x["mf_d"].values,x["mf_k"].values)+x["mf_s"].values
    num_space = np.array(num_space)
    acr = x["mf_areasize"].values*x["mf_r"].values
    acr_all = x["mf_areasize"].values*num_space
    return acr,acr_all


def acr_sq_(x):
    num_space = x["mf_r"].values+x["mf_l"].values+np.maximum(x["mf_d"].values,x["mf_k"].values)+x["mf_s"].values
    num_space = np.array(num_space)
    acr = x["mf_area_sq"].values*x["mf_r"].values
    acr_all = x["mf_area_sq"].values*num_space
    return acr,acr_all

def relative_height_(x):
    a = x["mf_what_floor"].values
    b = x["mf_height_bld"].values
    per = a/b
    diff = a-b
    return per,diff


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
    