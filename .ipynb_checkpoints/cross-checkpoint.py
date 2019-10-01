import re
import numpy as np
import pandas as pd

def apr_(x):
    num_space = x["mf_r"].values+x["mf_l"].values+x["mf_d"].values+x["mf_k"].values+x["mf_s"].values
    num_space = np.array(num_space)
    apr = x["mf_areasize"].values/x["mf_r"].values
    apr_all = x["mf_areasize"].values/num_space
    return apr,apr_all

def apr_sq_(x):
    num_space = x["mf_r"].values+x["mf_l"].values+x["mf_d"].values+x["mf_k"].values+x["mf_s"].values
    num_space = np.array(num_space)
    apr = x["mf_area_sq"].values/x["mf_r"].values
    apr_all = x["mf_area_sq"].values/num_space
    return apr,apr_all

def acr_(x):
    num_space = x["mf_r"].values+x["mf_l"].values+x["mf_d"].values+x["mf_k"].values+x["mf_s"].values
    num_space = np.array(num_space)
    acr = x["mf_areasize"].values*x["mf_r"].values
    acr_all = x["mf_areasize"].values*num_space
    return acr,acr_all


def acr_sq_(x):
    num_space = x["mf_r"].values+x["mf_l"].values+x["mf_d"].values+x["mf_k"].values+x["mf_s"].values
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