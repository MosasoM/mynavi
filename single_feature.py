import re
import pandas as pd
import numpy as np
import xgboost as xgb
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,scale,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import zscore
from sklearn.decomposition import NMF
from annoy import AnnoyIndex

from single import *
from cross import *
    





class dummy:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def transform(self,x):
        return x
    def predict(self,x):
        return x

